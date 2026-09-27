"""Reading the log back: spend, spend by tier, and savings against each baseline.

"Savings" here always means: what the log says was spent, against what the same
requests would have cost had every one of them gone to tier X. X ranges over
every configured tier, not just the expensive one, because a comparison only
against the top tier is the flattering one and proves nothing -- switching the
default to a cheaper model would have produced that saving with no router at
all.

Two honesty guards travel with each baseline:

  * `unpriced_rows` -- rows where tier X carried no configured price. Their
    counterfactual is 0, which means "unknown", not "free". A baseline with
    unpriced rows understates its own cost and therefore overstates savings.
  * `ineligible_rows` -- rows tier X could not have served at all (context
    window, missing tools). Those are not real alternatives, so the baseline
    including them is counterfactual fiction. `savings_usd_eligible_only`
    restricts the comparison to rows the baseline could actually have taken.

Two spend figures are reported and they are not the same number. `total_cost_usd`
is the whole bill, verification and escalation included, and it is the one every
savings figure is measured against -- a router that pays for a second opinion has
spent that money whether or not the report mentions it. `TierSpend.cost_usd` is
the routed call alone, because "what is tier X costing us" must not be inflated
by a verifier tier X never chose. The verification block prints the difference
explicitly, as a share of the bill, so the loop's overhead cannot hide inside a
by-tier table.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from .calls import MIN_EVIDENCE, WINDOW, accepts_task_row, pooled_shape
from .config import TaskShape
from .db import RequestLog


@dataclass
class TierSpend:
    tier: str
    requests: int
    cost_usd: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int


@dataclass
class VerificationStats:
    """What the verification loop did, and what it added to the bill."""

    verified: int
    passed: int
    failed: int
    verifier_errors: int
    unparseable: int
    skipped: int
    escalated: int
    # Failures no judge ever read: an empty answer, or one cut off at the tier's
    # own output budget. Reported apart because they point at a knob rather than
    # at the model -- and because they are free, so a loop full of them is
    # paying for reviews it does not need.
    unreviewed_failures: int = 0
    # Failures repaired by asking the SAME tier again, with no escalation.
    retried: int = 0
    verifier_cost_usd: float = 0.0
    escalation_cost_usd: float = 0.0
    skips_by_reason: dict[str, int] = field(default_factory=dict)

    @property
    def total_cost_usd(self) -> float:
        return self.verifier_cost_usd + self.escalation_cost_usd

    @property
    def fail_rate(self) -> float | None:
        """Share of reviewed answers the verifier rejected.

        None rather than 0.0 when nothing was reviewed: no data and a perfect
        record read identically otherwise, and only one of them is good news.
        """
        return self.failed / self.verified if self.verified else None


@dataclass
class ClassifierStats:
    """What the deployed model scored, and whether those scores meant anything.

    The training report is a claim about held-out history. This is the check on
    live traffic, and it is the one that can go stale underneath you: the corpus
    a model was fitted on ages, and nothing in the request path notices.

    Read `unreviewed` first. A classifier that is always obeyed sends every
    prompt it scores high to the strong tier, where no verifier looks at it, so
    those rows can never contradict it. The table below is therefore drawn from
    the requests the model believed were easy, plus whatever `router.explore_rate`
    deliberately kept cheap. With exploration off, the two columns converge on
    saying what the model already believed and nothing else.
    """

    scored: int
    reviewed: int
    explored: int
    # fingerprint -> rows. More than one means the column spans a retraining and
    # the bands below are two models' numbers in one table.
    models: dict[str, int] = field(default_factory=dict)
    # (label, rows, failures) per score band.
    bands: list[tuple[str, int, int]] = field(default_factory=list)
    mean_score_failed: float | None = None
    mean_score_passed: float | None = None

    @property
    def unreviewed(self) -> int:
        return self.scored - self.reviewed

    @property
    def separation(self) -> float | None:
        """Mean score on failures minus mean score on passes.

        The whole model in one number. At or below zero it is scoring noise,
        however well it read at training time.
        """
        if self.mean_score_failed is None or self.mean_score_passed is None:
            return None
        return self.mean_score_failed - self.mean_score_passed


@dataclass
class Baseline:
    """What everything would have cost at this one tier."""

    tier: str
    total_usd: float
    savings_usd: float
    unpriced_rows: int
    ineligible_rows: int
    # Same comparison over only the rows this tier could have served.
    baseline_usd_eligible_only: float
    actual_usd_eligible_only: float
    savings_usd_eligible_only: float


@dataclass
class BillingStats:
    """The provider's own figures beside the price-table estimate.

    Compared only on rows that have both. The estimate over every row next to
    the bill over some of them would be a gap made of missing rows.
    """

    billed_rows: int
    reconciled_rows: int
    unbilled_rows: int
    # Unbilled rows with zero tokens and a provider id: almost always a stream
    # the client left before the usage frame. `reconcile` can cost them.
    recoverable_rows: int
    estimate_usd: float
    billed_usd: float

    @property
    def drift(self) -> float | None:
        """(billed - estimate) / estimate, or None when there is nothing to divide."""
        if self.estimate_usd <= 0:
            return None
        return (self.billed_usd - self.estimate_usd) / self.estimate_usd


# Beyond this the price table is wrong, not rounding: a savings line built on it
# is built on a number the invoice contradicts.
DRIFT_TOLERANCE = 0.05


@dataclass
class TierShape:
    tier: str
    evidence: int  # outcomes accepted as shape evidence
    without_cache: int  # outcomes with usage but no cache tokens: a caller that never reports them
    rejected: int  # outcomes accepts_task_row refuses for another reason: cache over input, or negative fields
    source: str  # "observed:<n>" | "config" | "none"
    input_per_output: float | None = None
    cache_read: float | None = None
    cache_write: float | None = None


@dataclass
class TaskCostStats:
    """Route-only decisions: the shape each tier is priced by, and which decisions it changed.

    An observed shape is the traffic a tier was sent as much as the model: a tier
    routed harder tasks runs longer loops and looks dearer per output token.
    """

    config_shape: dict[str, Any] | None
    tiers: list[TierShape] = field(default_factory=list)
    flips: dict[str, int] = field(default_factory=dict)  # "<chosen> over <shapeless pick>" -> decisions


@dataclass
class UnsureStats:
    """Decisions where Jev read a requirement inside the unsure band, and what the floor did.

    There is no outcome for the tier the floor passed over, so whether it pays is
    read by comparing the raised decisions' pass rate with the unsure ones it left alone.
    """

    band: str  # "[0.30, 0.70]", or "mixed" when the log spans a band change
    decisions: int = 0
    raised: int = 0
    unmet: int = 0
    estimated_extra_usd: float = 0.0  # what the router believed the floor cost, summed `extra` (signed)
    measured_extra_usd: float = 0.0  # proxied rows: route cost minus the passed-over tier's counterfactual
    measured_rows: int = 0
    raised_pass: int = 0
    raised_fail: int = 0
    kept_pass: int = 0
    kept_fail: int = 0
    by_requirement: dict[str, int] = field(default_factory=dict)


@dataclass
class Stats:
    requests: int
    errors: int
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    log_failures: int
    by_tier: list[TierSpend] = field(default_factory=list)
    baselines: list[Baseline] = field(default_factory=list)
    verification: VerificationStats | None = None
    classifier: ClassifierStats | None = None
    billing: BillingStats | None = None
    task_cost: TaskCostStats | None = None
    # Refused because Jev did not answer under `on_jev_failure: reject`.
    jev_rejected_chat: int = 0
    jev_rejected_route: int = 0
    unsure: UnsureStats | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect(log: RequestLog, task_shape: TaskShape | None = None) -> Stats:
    totals = log.query(
        """
        SELECT COUNT(*)                                       AS requests,
               COALESCE(SUM(cost_usd), 0)                     AS cost,
               COALESCE(SUM(input_tokens), 0)                 AS input_tokens,
               COALESCE(SUM(output_tokens), 0)                AS output_tokens,
               COALESCE(SUM(http_status >= 400), 0)           AS errors
        FROM requests
        """
    )[0]

    by_tier = [
        TierSpend(
            tier=row["tier"] or "(unrouted)",
            requests=row["requests"],
            cost_usd=row["cost"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cached_tokens=row["cached_tokens"],
        )
        for row in log.query(
            """
            SELECT tier,
                   COUNT(*)                          AS requests,
                   COALESCE(SUM(route_cost_usd), 0)  AS cost,
                   COALESCE(SUM(input_tokens), 0)  AS input_tokens,
                   COALESCE(SUM(output_tokens), 0) AS output_tokens,
                   COALESCE(SUM(cached_tokens), 0) AS cached_tokens
            FROM requests
            GROUP BY tier
            ORDER BY cost DESC
            """
        )
    ]

    baselines = [
        Baseline(
            tier=row["tier"],
            total_usd=row["baseline"],
            savings_usd=row["baseline"] - totals["cost"],
            unpriced_rows=row["unpriced"],
            ineligible_rows=row["ineligible"],
            baseline_usd_eligible_only=row["baseline_eligible"],
            actual_usd_eligible_only=row["actual_eligible"],
            savings_usd_eligible_only=row["baseline_eligible"] - row["actual_eligible"],
        )
        for row in log.query(
            """
            SELECT c.tier                                                    AS tier,
                   COALESCE(SUM(c.cost_usd), 0)                              AS baseline,
                   COALESCE(SUM(CASE WHEN c.priced = 0 THEN 1 ELSE 0 END), 0) AS unpriced,
                   COALESCE(SUM(CASE WHEN c.eligible = 0 THEN 1 ELSE 0 END), 0) AS ineligible,
                   COALESCE(SUM(CASE WHEN c.eligible = 1 THEN c.cost_usd ELSE 0 END), 0) AS baseline_eligible,
                   COALESCE(SUM(CASE WHEN c.eligible = 1 THEN r.cost_usd ELSE 0 END), 0) AS actual_eligible
            FROM counterfactuals c
            JOIN requests r ON r.id = c.request_row_id
            GROUP BY c.tier
            ORDER BY baseline DESC
            """
        )
    ]

    return Stats(
        verification=_verification(log),
        classifier=_classifier(log),
        billing=_billing(log),
        task_cost=_task_cost(log, task_shape),
        unsure=_unsure(log),
        requests=totals["requests"],
        errors=totals["errors"],
        total_cost_usd=totals["cost"],
        total_input_tokens=totals["input_tokens"],
        total_output_tokens=totals["output_tokens"],
        log_failures=len(log.query("SELECT id FROM log_failures")),
        by_tier=by_tier,
        baselines=baselines,
        # Not filtered on route_score: a refused row has none.
        jev_rejected_chat=log.query(
            "SELECT COUNT(*) AS n FROM requests WHERE route_reason LIKE '{\"rule\":\"jev_failure\"%'")[0]["n"],
        jev_rejected_route=log.query("SELECT COUNT(*) AS n FROM route_rejections")[0]["n"],
    )


def _billing(log: RequestLog) -> BillingStats | None:
    row = log.query(
        """
        SELECT
          SUM(billed_cost_usd IS NOT NULL)                              AS billed,
          SUM(billed_source = 'reconciled')                             AS reconciled,
          SUM(billed_cost_usd IS NULL)                                  AS unbilled,
          SUM(billed_cost_usd IS NULL AND upstream_id IS NOT NULL
              AND input_tokens = 0 AND output_tokens = 0)               AS recoverable,
          COALESCE(SUM(CASE WHEN billed_cost_usd IS NOT NULL
                            THEN cost_usd END), 0)                      AS estimate,
          COALESCE(SUM(billed_cost_usd), 0)                             AS billed_usd,
          SUM(upstream_id IS NOT NULL)                                  AS with_id
        FROM requests
        """
    )[0]
    # A log with no provider figures at all -- fake backends, or a database
    # from before v4 -- has nothing to compare, and an all-zero block would read
    # as "estimate and invoice agree".
    if not (row["billed"] or row["with_id"]):
        return None
    return BillingStats(
        billed_rows=row["billed"] or 0,
        reconciled_rows=row["reconciled"] or 0,
        unbilled_rows=row["unbilled"] or 0,
        recoverable_rows=row["recoverable"] or 0,
        estimate_usd=row["estimate"],
        billed_usd=row["billed_usd"],
    )


def _format_billing(b: BillingStats) -> list[str]:
    lines = ["", "billing: the provider's own figure beside the price-table estimate"]
    lines.append(
        f"  rows the provider billed  {b.billed_rows}"
        + (f"  ({b.reconciled_rows} reconciled after the fact)" if b.reconciled_rows else "")
    )
    lines.append(f"  estimate on those rows    ${b.estimate_usd:.6f}")
    drift = b.drift
    lines.append(
        f"  billed on those rows      ${b.billed_usd:.6f}"
        + (f"  drift {drift:+.1%}" if drift is not None else "")
    )
    if b.billed_rows and b.billed_usd == 0 and b.estimate_usd > 0:
        lines.append(
            "  note: billed $0 against a priced estimate -- a free route priced at list "
            "price, or the prices belong to another model"
        )
    elif drift is not None and abs(drift) > DRIFT_TOLERANCE:
        lines.append(
            f"  DRIFT {drift:+.1%}: the price table does not match the invoice. Fix `prices` "
            "before believing any savings line below -- they are all built on it."
        )
    if b.unbilled_rows:
        lines.append(
            f"  NOT BILLED {b.unbilled_rows} row(s): estimate only (a provider that reports "
            "no cost, an error, or a review whose bill was missing)"
        )
    if b.recoverable_rows:
        lines.append(
            f"  RECOVERABLE {b.recoverable_rows} row(s) with zero tokens and a provider id -- "
            "abandoned streams, costed at nothing: run `jev-model-router reconcile`"
        )
    return lines


def _task_cost(log: RequestLog, task_shape: TaskShape | None) -> TaskCostStats | None:
    """None when the log has no route-only decisions at all, so the report stays silent about it."""
    decided_tiers = log.query("SELECT DISTINCT tier FROM route_decisions")
    if not decided_tiers:
        return None
    evidence: dict[str, list[tuple[int, int, int, int]]] = {row["tier"]: [] for row in decided_tiers}
    without: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    # Replays the same rows startup seeding does (`log.task_usage()`), applying the same row rule
    # `CallRouter.record_task` applies live -- tracked here rather than through a `CallRouter`,
    # since stats has no config, prices or ledger to build one, only the log.
    reports_cache: set[str] = set()
    for row in log.task_usage():
        tier, cached, written = row["tier"], row["cached"], row["written"]
        if accepts_task_row(row["input"], row["output"], cached, written, tier in reports_cache):
            if cached + written > 0:
                reports_cache.add(tier)
            evidence.setdefault(tier, []).append((row["input"], cached, written, row["output"]))
        elif accepts_task_row(row["input"], row["output"], cached, written, True):
            without[tier] += 1  # would be fine, except this tier has not shown it reports the cache yet
        else:
            rejected[tier] += 1  # doesn't add up on its own: cache over input, or a negative field

    flips: Counter[str] = Counter()
    for row in log.query("SELECT tier, route_reason FROM route_decisions WHERE route_reason IS NOT NULL"):
        try:
            reason = json.loads(row["route_reason"])
        except (ValueError, TypeError):
            continue
        if isinstance(reason, dict) and reason.get("shapeless_pick"):
            flips[f"{row['tier']} over {reason['shapeless_pick']}"] += 1

    tiers = []
    for tier, seen in sorted(evidence.items()):
        shape = pooled_shape(seen[-WINDOW:]) if len(seen) >= MIN_EVIDENCE else task_shape
        tiers.append(TierShape(
            tier=tier, evidence=len(seen), without_cache=without[tier], rejected=rejected[tier],
            source=shape.source if shape else "none",
            input_per_output=round(shape.input_per_output, 1) if shape else None,
            cache_read=round(shape.cache_read, 3) if shape else None,
            cache_write=round(shape.cache_write, 3) if shape else None,
        ))
    return TaskCostStats(config_shape=asdict(task_shape) if task_shape else None, tiers=tiers, flips=dict(flips))


def _format_task_cost(t: TaskCostStats) -> list[str]:
    lines = ["", "task cost (route-only decisions): the shape each tier's task is priced by"]
    if t.config_shape:
        c = t.config_shape
        shape = f"{c['input_per_output']:g} in/out, read {c['cache_read']:.3f}, write {c['cache_write']:.3f}"
        lines.append(f"  {'config':<28} {'':<12} {shape}")
    for row in t.tiers:
        shape = (f"{row.input_per_output:g} in/out, read {row.cache_read:.3f}, write {row.cache_write:.3f}"
                 if row.input_per_output is not None else "one call")
        lines.append(f"  {row.tier:<28} {row.source:<12} {shape}")
        if row.without_cache:
            lines.append(f"      note: {row.without_cache} outcome(s) with usage but no cache tokens")
        if row.rejected:
            lines.append(
                f"      note: {row.rejected} outcome(s) whose tokens don't add up "
                "(cache over input, or zero/negative)"
            )
    for pair, n in sorted(t.flips.items(), key=lambda kv: -kv[1]):
        lines.append(f"  shape flipped   {pair}: {n}")
    lines.append("  note: an observed shape is the traffic a tier was sent as much as the model")
    return lines


def _unsure(log: RequestLog) -> UnsureStats | None:
    """None when no decision in the log read a requirement as unsure, so the report stays silent about it."""
    rows = [
        (row["route_reason"], row["outcome"], None, None)
        for row in log.query(
            "SELECT route_reason, outcome FROM route_decisions WHERE route_reason LIKE '%\"unsure\"%'"
        )
    ] + [
        (row["route_reason"], row["outcome"], row["id"], row["cost"])
        for row in log.query(
            "SELECT r.id AS id, r.route_reason AS route_reason, r.route_cost_usd AS cost, v.verdict AS outcome "
            "FROM requests r LEFT JOIN verifications v ON v.request_row_id = r.id "
            "WHERE r.route_reason LIKE '%\"unsure\"%'"
        )
    ]
    stats = UnsureStats(band="")
    bands: set[tuple[float, float]] = set()
    mixed = False
    reqs: Counter[str] = Counter()
    for text, outcome, row_id, cost in rows:
        try:
            unsure = json.loads(text).get("unsure")
        except (ValueError, TypeError, AttributeError):
            continue
        if not isinstance(unsure, dict):
            continue
        stats.decisions += 1

        band = unsure.get("band")
        if (
            isinstance(band, list)
            and len(band) == 2
            and all(isinstance(b, (int, float)) for b in band)
        ):
            bands.add((float(band[0]), float(band[1])))
        else:
            mixed = True

        req_list = unsure.get("reqs")
        if isinstance(req_list, list):
            reqs.update(r for r in req_list if isinstance(r, str))

        raised_from = unsure.get("raised_from")
        raised = isinstance(raised_from, list) and bool(raised_from) and isinstance(raised_from[0], str)

        if unsure.get("unmet"):
            stats.unmet += 1
        if raised:
            stats.raised += 1
            if isinstance(unsure.get("extra"), (int, float)):
                stats.estimated_extra_usd += unsure["extra"]
            if row_id is not None:
                passed_over = log.query(
                    "SELECT cost_usd FROM counterfactuals WHERE request_row_id = ? AND tier = ? AND priced = 1",
                    (row_id, raised_from[0]),
                )
                if passed_over:
                    stats.measured_extra_usd += cost - passed_over[0]["cost_usd"]
                    stats.measured_rows += 1
        if outcome == "pass":
            if raised:
                stats.raised_pass += 1
            else:
                stats.kept_pass += 1
        elif outcome == "fail":
            if raised:
                stats.raised_fail += 1
            else:
                stats.kept_fail += 1
    if not stats.decisions:
        return None
    if not mixed and len(bands) == 1:
        low, high = next(iter(bands))
        stats.band = f"[{low:.2f}, {high:.2f}]"
    else:
        stats.band = "mixed"
    stats.by_requirement = dict(reqs.most_common())
    return stats


def _format_unsure(u: UnsureStats) -> list[str]:
    lines = ["", f"unsure floor: a requirement Jev read inside {u.band}"]
    lines.append(f"  unsure decisions {u.decisions:>6}   raised {u.raised}   unmet {u.unmet}")
    lines.append(f"  raised: estimated extra  ${u.estimated_extra_usd:.6f}")
    if u.measured_rows:
        lines.append(
            f"  raised: measured extra   ${u.measured_extra_usd:.6f}  over {u.measured_rows} proxied "
            "request(s) with a priced counterfactual"
        )
    lines.append(f"  raised outcomes      pass {u.raised_pass}  fail {u.raised_fail}")
    lines.append(f"  unsure, not raised   pass {u.kept_pass}  fail {u.kept_fail}")
    if u.by_requirement:
        lines.append("  by requirement       " + "  ".join(f"{k} {n}" for k, n in u.by_requirement.items()))
    lines.append("  note: the tier the floor passed over has no outcome; compare the two pass rates")
    return lines


def _verification(log: RequestLog) -> VerificationStats | None:
    """None when the loop has never run, so the report stays silent about it."""
    row = log.query(
        """
        SELECT COUNT(*)                                                       AS rows_,
               COALESCE(SUM(verdict = 'pass'), 0)                             AS passed,
               COALESCE(SUM(verdict = 'fail'), 0)                             AS failed,
               COALESCE(SUM(verdict = 'error'), 0)                            AS verifier_errors,
               COALESCE(SUM(verdict = 'skipped'), 0)                          AS skipped,
               COALESCE(SUM(unparseable), 0)                                  AS unparseable,
               COALESCE(SUM(escalated), 0)                                    AS escalated,
               COALESCE(SUM(verdict = 'fail' AND verifier_tier IS NULL), 0)   AS unreviewed,
               COALESCE(SUM(verifier_cost_usd), 0)                            AS verifier_cost,
               COALESCE(SUM(escalation_cost_usd), 0)                          AS escalation_cost
        FROM verifications
        """
    )[0]
    if not row["rows_"]:
        return None

    # The reason column carries the detail after a colon ("verifier_ineligible:
    # estimated 9000 tokens exceeds..."), and the detail varies per request. The
    # grouping therefore happens HERE rather than in SQL: a GROUP BY on the raw
    # column returns one row per distinct token count, which is a list of unique
    # strings pretending to be a summary.
    tally: Counter[str] = Counter(
        str(entry["reason"] or "unknown").split(":", 1)[0]
        for entry in log.query("SELECT reason FROM verifications WHERE verdict = 'skipped'")
    )
    skips = dict(tally.most_common())
    # A retry names the tier that served the request, so it is the join that
    # tells it apart from an escalation -- `verifications` alone cannot.
    retried = log.query(
        "SELECT COUNT(*) AS n FROM verifications v JOIN requests r ON v.request_row_id = r.id "
        "WHERE v.escalated = 1 AND v.escalated_to = r.tier"
    )[0]["n"]
    return VerificationStats(
        verified=row["rows_"] - row["skipped"],
        passed=row["passed"],
        failed=row["failed"],
        verifier_errors=row["verifier_errors"],
        unparseable=row["unparseable"],
        skipped=row["skipped"],
        escalated=row["escalated"],
        unreviewed_failures=row["unreviewed"],
        retried=retried,
        verifier_cost_usd=row["verifier_cost"],
        escalation_cost_usd=row["escalation_cost"],
        skips_by_reason=skips,
    )


_BANDS: list[tuple[float, float, str]] = [
    (0.0, 0.2, "0.0-0.2"),
    (0.2, 0.4, "0.2-0.4"),
    (0.4, 0.6, "0.4-0.6"),
    (0.6, 0.8, "0.6-0.8"),
    (0.8, 1.01, "0.8-1.0"),
]


def _classifier(log: RequestLog) -> ClassifierStats | None:
    """None when no request was ever scored, so the report stays silent about it."""
    scored = log.query(
        """
        SELECT COALESCE(route_model, '(unknown)') AS model,
               COUNT(*)                           AS rows_,
               COALESCE(SUM(route_reason LIKE 'explore%'), 0) AS explored
        FROM requests
        WHERE route_score IS NOT NULL
        GROUP BY route_model
        """
    )
    if not scored:
        return None

    # Only rows a verifier actually judged can say whether a score was right,
    # and the join is what restricts them. `unparseable` rows are excluded for
    # the same reason training excludes them: the verdict came from a fallback
    # policy rather than from a reviewer, so it is evidence about the config.
    judged = log.query(
        """
        SELECT r.route_score AS score, v.verdict AS verdict
        FROM requests r
        JOIN verifications v ON v.request_row_id = r.id
        WHERE r.route_score IS NOT NULL
          AND v.verdict IN ('pass', 'fail')
          AND v.unparseable = 0
        """
    )

    bands: list[tuple[str, int, int]] = []
    for low, high, label in _BANDS:
        rows = [j for j in judged if low <= j["score"] < high]
        if rows:
            bands.append((label, len(rows), sum(1 for j in rows if j["verdict"] == "fail")))

    failed = [j["score"] for j in judged if j["verdict"] == "fail"]
    passed = [j["score"] for j in judged if j["verdict"] == "pass"]
    return ClassifierStats(
        scored=sum(row["rows_"] for row in scored),
        reviewed=len(judged),
        explored=sum(row["explored"] for row in scored),
        models={row["model"]: row["rows_"] for row in scored},
        bands=bands,
        mean_score_failed=(sum(failed) / len(failed)) if failed else None,
        mean_score_passed=(sum(passed) / len(passed)) if passed else None,
    )


def _format_classifier(c: ClassifierStats) -> list[str]:
    lines = ["", "classifier"]
    fingerprints = ", ".join(f"{name} ({count})" for name, count in c.models.items())
    lines.append(f"  scored         {c.scored:>6}   by {fingerprints}")
    if len(c.models) > 1:
        lines.append(
            "  MORE THAN ONE MODEL scored these rows. The bands below mix them, "
            "and the mixture is not a model of anything -- filter by route_model."
        )
    lines.append(
        f"  reviewed       {c.reviewed:>6}   scored requests a verifier also judged"
    )
    if c.unreviewed:
        lines.append(
            f"  unreviewed     {c.unreviewed:>6}   scored and never checked -- these "
            "rows cannot contradict the model"
        )
    if c.explored:
        lines.append(
            f"  explored       {c.explored:>6}   would-be escalations kept cheap on "
            "purpose, so they could be judged"
        )
    elif c.unreviewed:
        lines.append(
            "  router.explore_rate is 0, so every request the model escalated left "
            "no evidence behind. The table below describes the easy half of the "
            "traffic and will keep agreeing with the model whatever it does."
        )

    if c.separation is not None:
        lines.append(
            f"  mean score     {c.mean_score_failed:.3f} on answers that failed, "
            f"{c.mean_score_passed:.3f} on answers that passed"
        )
        if c.separation <= 0:
            lines.append(
                "  THE SCORES ARE BACKWARDS OR NOISE: failures do not score higher "
                "than passes on this traffic. Retrain, or go back to kind: static."
            )
    if c.bands:
        lines.append("  score band      rows   failed   actual fail rate")
        for label, rows, failures in c.bands:
            lines.append(
                f"    {label:<10} {rows:>7} {failures:>8}   {failures / rows * 100:>6.1f}%"
            )
    return lines


def format_text(stats: Stats) -> str:
    """Plain-text report. No dashboard in this step, by design."""
    lines: list[str] = []
    lines.append(f"requests        {stats.requests}")
    lines.append(f"errors          {stats.errors}")
    lines.append(f"input tokens    {stats.total_input_tokens}")
    lines.append(f"output tokens   {stats.total_output_tokens}")
    lines.append(f"actual spend    ${stats.total_cost_usd:.6f}")
    if stats.log_failures:
        lines.append(f"LOG FAILURES    {stats.log_failures}  (rows that could not be written)")
    rejected = stats.jev_rejected_chat + stats.jev_rejected_route
    if rejected:
        lines.append(f"JEV REJECTED    {rejected}  (chat {stats.jev_rejected_chat}, route {stats.jev_rejected_route})")

    lines.append("")
    lines.append("spend by tier")
    if not stats.by_tier:
        lines.append("  (no requests logged)")
    for row in stats.by_tier:
        lines.append(
            f"  {row.tier:<14} {row.requests:>6} req  ${row.cost_usd:>12.6f}  "
            f"in {row.input_tokens:>9}  out {row.output_tokens:>9}  cached {row.cached_tokens:>9}"
        )

    if stats.billing is not None:
        lines.extend(_format_billing(stats.billing))

    if stats.classifier is not None:
        lines.extend(_format_classifier(stats.classifier))

    if stats.verification is not None:
        lines.extend(_format_verification(stats.verification, stats.total_cost_usd))

    if stats.task_cost is not None:
        lines.extend(_format_task_cost(stats.task_cost))

    if stats.unsure is not None:
        lines.extend(_format_unsure(stats.unsure))

    lines.append("")
    lines.append("counterfactual baselines: what everything would have cost at one tier")
    if not stats.baselines:
        lines.append("  (no counterfactuals logged)")
    for base in stats.baselines:
        lines.append(
            f"  always {base.tier:<10} ${base.total_usd:>12.6f}  "
            f"savings ${base.savings_usd:>12.6f}"
        )
        notes = []
        if base.unpriced_rows:
            notes.append(f"{base.unpriced_rows} row(s) had no price configured for this tier")
        if base.ineligible_rows:
            notes.append(
                f"{base.ineligible_rows} row(s) were ineligible for it; "
                f"eligible-only savings ${base.savings_usd_eligible_only:.6f}"
            )
        for note in notes:
            lines.append(f"      note: {note}")

    return "\n".join(lines)


def _format_verification(v: VerificationStats, total_spend: float) -> list[str]:
    lines = ["", "verification"]
    lines.append(
        f"  reviewed       {v.verified:>6}   pass {v.passed}  fail {v.failed}  "
        f"verifier errors {v.verifier_errors}"
    )
    if v.fail_rate is not None:
        lines.append(f"  fail rate      {v.fail_rate * 100:>6.1f}%  of reviewed answers")
    lines.append(
        f"  escalated      {v.escalated:>6}   a second, dearer answer the client actually received"
    )
    if v.retried:
        lines.append(
            f"  of which retry {v.retried:>6}   answered by asking the SAME tier again, "
            "at no dearer tier's price"
        )
    if v.unreviewed_failures:
        lines.append(
            f"  free failures  {v.unreviewed_failures:>6}   empty or cut off at the output "
            "budget: failed by a rule, with no verifier paid"
        )
        if v.failed and v.unreviewed_failures / v.failed >= 0.5:
            lines.append(
                "     those are most of your failures, and none of them is a difficulty "
                "signal. Look at the tier's output budget and its temperature before "
                "looking at a bigger model or a classifier"
            )
    lines.append(f"  verifier spend        ${v.verifier_cost_usd:>12.6f}")
    lines.append(f"  escalation spend      ${v.escalation_cost_usd:>12.6f}")
    if total_spend > 0:
        share = v.total_cost_usd / total_spend * 100
        # The number the whole design turns on. A loop costing a third of the
        # bill has to be buying a third of the bill in quality.
        lines.append(
            f"  loop overhead         ${v.total_cost_usd:>12.6f}  "
            f"({share:.1f}% of actual spend)"
        )
    if v.unparseable:
        lines.append(
            f"  UNPARSEABLE    {v.unparseable:>6}   verdicts the fallback decided, "
            f"not the verifier -- the pass rate above is that much less real"
        )
    if v.skipped:
        detail = ", ".join(f"{reason} {count}" for reason, count in v.skips_by_reason.items())
        lines.append(f"  skipped        {v.skipped:>6}   {detail}")
    return lines
