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

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

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
    verifier_cost_usd: float
    escalation_cost_usd: float
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect(log: RequestLog) -> Stats:
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
        requests=totals["requests"],
        errors=totals["errors"],
        total_cost_usd=totals["cost"],
        total_input_tokens=totals["input_tokens"],
        total_output_tokens=totals["output_tokens"],
        log_failures=len(log.query("SELECT id FROM log_failures")),
        by_tier=by_tier,
        baselines=baselines,
    )


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
    return VerificationStats(
        verified=row["rows_"] - row["skipped"],
        passed=row["passed"],
        failed=row["failed"],
        verifier_errors=row["verifier_errors"],
        unparseable=row["unparseable"],
        skipped=row["skipped"],
        escalated=row["escalated"],
        verifier_cost_usd=row["verifier_cost"],
        escalation_cost_usd=row["escalation_cost"],
        skips_by_reason=skips,
    )


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

    lines.append("")
    lines.append("spend by tier")
    if not stats.by_tier:
        lines.append("  (no requests logged)")
    for row in stats.by_tier:
        lines.append(
            f"  {row.tier:<14} {row.requests:>6} req  ${row.cost_usd:>12.6f}  "
            f"in {row.input_tokens:>9}  out {row.output_tokens:>9}  cached {row.cached_tokens:>9}"
        )

    if stats.verification is not None:
        lines.extend(_format_verification(stats.verification, stats.total_cost_usd))

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
