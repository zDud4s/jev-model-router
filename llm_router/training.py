"""Turning the request log into a difficulty model, and judging what came out.

The labels are already being written. Every verified request carries a verdict:
`pass` means the cheap tier answered and the answer held, `fail` means it did
not. Those are the two classes, produced by ordinary traffic rather than by
hand, which is why the verification loop was built first.

Four filters run before anything is fitted, and each one is the difference
between a number and a fiction:

* **Only rows whose verdict came from the verifier.** `unparseable = 1` means
  the reply carried no verdict and the configured fallback decided. With
  `on_unparseable: accept` that fallback manufactures `pass` labels; training on
  them teaches the model that a broken verifier is a well-answered prompt.
* **Only rows served by the tier being modelled.** A verdict is about one
  model's answer. Mixing tiers produces a model of nothing in particular.
* **Only rows whose prompt was stored.** `log.store_prompts` is false by
  default -- prompts are user data -- so by default the log holds SHA-256 and
  nothing to learn from. That is a deliberate default and an absolute barrier,
  and training says so plainly instead of fitting on an empty corpus.
* **Only successful requests.** A 500 has no answer to have judged.

Then the split, which is where this kind of work usually goes wrong. Grouping is
by conversation, never by row; `split_by_row` is kept and reported alongside so
the gap between the honest number and the flattering one is printed rather than
described.

Finally the part that decides whether any of it was worth doing: the held-out
set is priced under each policy, with money and bad answers in SEPARATE columns.
They are not summed. This module does not know what a wrong answer costs the
operator, will not pretend to, and a single "score" that blends the two would be
exactly that pretence.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from .classifier import (
    DifficultyModel,
    Example,
    build_model,
    conversation_key,
    features_from_prompt_text,
    group_fraction,
    split_by_row,
)
from .db import RequestLog


class TrainingError(Exception):
    """Raised when the log cannot support a model, with the reason."""


@dataclass(frozen=True)
class Row:
    """One usable log row: the label, and what the alternatives cost on it."""

    example: Example
    route_cost_usd: float
    loop_cost_usd: float
    strong_cost_usd: float
    strong_priced: bool
    strong_eligible: bool
    # False when the verdict came from a rule rather than a judge: an answer
    # that was empty or cut off at the tier's budget, failed for nothing by the
    # verification loop. A corpus made mostly of those is not a difficulty
    # corpus -- the router already catches them without a model.
    reviewed: bool = True


@dataclass
class Metrics:
    """Quality of the scores, always next to the baseline that needs no model."""

    examples: int
    positives: int
    accuracy: float
    # Accuracy of answering the majority class every time. A classifier that
    # does not beat this has learned the class balance and nothing else, and on
    # imbalanced data it can look excellent while doing so.
    majority_accuracy: float
    precision: float | None
    recall: float | None
    escalation_rate: float
    # Ranking quality, independent of where the threshold sits: the chance that
    # a failed request scored above a passing one. 0.5 is a coin toss. Reported
    # because accuracy against a 17% base rate rewards "never fails", and
    # because a model can rank well and still be cut in the wrong place.
    auc: float | None = None
    # A 95% interval around that AUC. Printed because the held-out set is small
    # -- a few dozen failures -- and an AUC read off it without an interval is
    # a number with no error bar being treated as a measurement.
    auc_ci: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Policy:
    """What one routing policy would have cost on the held-out set."""

    name: str
    cost_usd: float
    # Answers the client would have received that the verifier had rejected.
    # Priced at nothing here because this module does not know their price.
    bad_answers: int
    escalations: int
    note: str = ""


@dataclass
class TrainingReport:
    model: DifficultyModel
    rows: int
    groups: int
    train_size: int
    test_size: int
    grouped: Metrics
    # The same fit and evaluation under a random by-row split. Reported to be
    # compared with `grouped`, never to be believed on its own.
    ungrouped: Metrics
    # Set when --tune ran: the settings it chose and what it compared.
    tuning: "Tuning | None" = None
    policies: list[Policy] = field(default_factory=list)
    sweep: list[tuple[float, float, int]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "rows": self.rows,
            "groups": self.groups,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "grouped": self.grouped.to_dict(),
            "ungrouped": self.ungrouped.to_dict(),
            "tuning": self.tuning.to_dict() if self.tuning else None,
            "policies": [asdict(p) for p in self.policies],
            "sweep": [
                {"threshold": t, "cost_usd": c, "bad_answers": b} for t, c, b in self.sweep
            ],
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------
# reading the log
# --------------------------------------------------------------------------


def load_rows(log: RequestLog, *, predicts_tier: str, strong_tier: str) -> list[Row]:
    """Every row of the log that can honestly be trained on."""
    raw = log.query(
        """
        SELECT r.prompt_text                                   AS prompt_text,
               r.route_cost_usd                                AS route_cost,
               v.verdict                                       AS verdict,
               v.verifier_tier                                 AS verifier_tier,
               v.verifier_cost_usd + v.escalation_cost_usd     AS loop_cost,
               c.cost_usd                                      AS strong_cost,
               c.priced                                        AS strong_priced,
               c.eligible                                      AS strong_eligible
        FROM requests r
        JOIN verifications v ON v.request_row_id = r.id
        LEFT JOIN counterfactuals c
               ON c.request_row_id = r.id AND c.tier = ?
        WHERE v.verdict IN ('pass', 'fail')
          AND v.unparseable = 0
          AND r.http_status = 200
          AND r.tier = ?
        ORDER BY r.id
        """,
        (strong_tier, predicts_tier),
    )
    if not raw:
        raise TrainingError(
            f"no verified requests served by tier {predicts_tier!r} in the log. "
            "Turn the verification loop on (verification.enabled: true, "
            "sample_rate: 1.0) and let it run -- the labels are a by-product of "
            "traffic, so there is no shortcut that does not involve traffic."
        )

    with_prompts = [row for row in raw if row["prompt_text"]]
    if not with_prompts:
        raise TrainingError(
            f"{len(raw)} verified request(s) found, and not one stored its prompt. "
            "log.store_prompts is false by default because prompts are user data, "
            "and a SHA-256 cannot be read for difficulty. Set log.store_prompts: "
            "true, accept that the database then holds the conversations, and "
            "collect again -- rows already written cannot be recovered."
        )

    rows: list[Row] = []
    for entry in with_prompts:
        vector = features_from_prompt_text(entry["prompt_text"])
        if vector is None:
            continue
        rows.append(
            Row(
                example=Example(
                    vector=vector,
                    label=1 if entry["verdict"] == "fail" else 0,
                    group=conversation_key(entry["prompt_text"]),
                ),
                route_cost_usd=float(entry["route_cost"] or 0.0),
                loop_cost_usd=float(entry["loop_cost"] or 0.0),
                strong_cost_usd=float(entry["strong_cost"] or 0.0),
                strong_priced=bool(entry["strong_priced"]),
                strong_eligible=bool(entry["strong_eligible"]),
                reviewed=entry["verifier_tier"] is not None,
            )
        )
    if not rows:
        raise TrainingError("no row's stored prompt could be read back as a message list")
    return rows


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def roc_auc(scored: Sequence[tuple[float, int]]) -> float | None:
    """Mann-Whitney U over (score, label) pairs. None when a class is absent."""
    positives = [s for s, label in scored if label]
    negatives = [s for s, label in scored if not label]
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def auc_interval(scored: Sequence[tuple[float, int]]) -> tuple[float, float] | None:
    """A 95% confidence interval for the AUC of `scored`, or None.

    Hanley and McNeil's standard error, clamped to [0, 1]. Analytic rather than
    bootstrapped: it is deterministic, needs no seed to reproduce, and costs
    nothing once the AUC is known. Checked against a 2000-resample bootstrap on
    the GSM8K corpus, which gave [0.424, 0.743] where this gives [0.436, 0.745].

    The interval is what makes a held-out AUC readable. On 96 rows with 17
    failures, anything from "actively misleading" to "genuinely useful" fits
    inside it, and the point estimate alone does not say so.
    """
    area = roc_auc(scored)
    if area is None:
        return None
    positives = sum(1 for _, label in scored if label)
    negatives = len(scored) - positives
    # Hanley & McNeil 1982: q1 and q2 are the probabilities that two positives
    # (resp. two negatives) both outrank a single drawn member of the other class.
    q1 = area / (2 - area)
    q2 = 2 * area * area / (1 + area)
    variance = (
        area * (1 - area)
        + (positives - 1) * (q1 - area * area)
        + (negatives - 1) * (q2 - area * area)
    ) / (positives * negatives)
    half = 1.959964 * math.sqrt(max(variance, 0.0))
    return (max(0.0, area - half), min(1.0, area + half))


def evaluate(model: DifficultyModel, examples: Sequence[Example], threshold: float) -> Metrics:
    positives = sum(e.label for e in examples)
    if not examples:
        return Metrics(0, 0, 0.0, 0.0, None, None, 0.0)

    scored = [(model.score(e.vector), e.label) for e in examples]
    tp = fp = tn = fn = 0
    for example in examples:
        predicted = 1 if model.score(example.vector) >= threshold else 0
        if predicted and example.label:
            tp += 1
        elif predicted and not example.label:
            fp += 1
        elif not predicted and example.label:
            fn += 1
        else:
            tn += 1

    total = len(examples)
    majority = max(positives, total - positives) / total
    return Metrics(
        examples=total,
        positives=positives,
        accuracy=(tp + tn) / total,
        majority_accuracy=majority,
        # None rather than 0.0 when the denominator is empty: "never predicted a
        # failure" and "predicted failures and got them all wrong" are different
        # facts and must not print the same way.
        precision=(tp / (tp + fp)) if (tp + fp) else None,
        recall=(tp / (tp + fn)) if (tp + fn) else None,
        escalation_rate=(tp + fp) / total,
        auc=roc_auc(scored),
        auc_ci=auc_interval(scored),
    )


def price_policies(
    model: DifficultyModel, rows: Sequence[Row], threshold: float
) -> list[Policy]:
    """What the held-out traffic would have cost under each way of running.

    Every figure uses the token counts actually recorded, so the comparison is
    arithmetic over one set of requests rather than a simulation. What it cannot
    do is know how the strong tier would have answered a prompt it never saw --
    so `always strong` is credited with zero bad answers by assumption, and the
    assumption is printed rather than buried.
    """
    if not rows:
        return []

    cheap_only = sum(r.route_cost_usd for r in rows)
    failures = sum(r.example.label for r in rows)
    as_logged = sum(r.route_cost_usd + r.loop_cost_usd for r in rows)
    always_strong = sum(r.strong_cost_usd for r in rows)

    routed_cost = 0.0
    escalations = 0
    missed = 0
    for row in rows:
        escalate = model.score(row.example.vector) >= threshold
        if escalate and row.strong_eligible:
            routed_cost += row.strong_cost_usd
            escalations += 1
        else:
            routed_cost += row.route_cost_usd
            missed += row.example.label

    unpriced = sum(1 for r in rows if not r.strong_priced)
    strong_note = (
        f"{unpriced} row(s) had no price for the strong tier, so this understates it"
        if unpriced
        else "assumes the strong tier's own answers would all have held; untested"
    )

    return [
        Policy(
            name=f"classifier @ {threshold:.2f}, no verifier",
            cost_usd=routed_cost,
            bad_answers=missed,
            escalations=escalations,
            note="the failures it did not catch are shipped to the client",
        ),
        Policy(
            name="always cheap, no verifier",
            cost_usd=cheap_only,
            bad_answers=failures,
            escalations=0,
            note="the floor: cheapest possible, and every failure reaches the client",
        ),
        Policy(
            name="always cheap + verify all",
            cost_usd=as_logged,
            bad_answers=0,
            escalations=sum(1 for r in rows if r.example.label),
            note="what the log actually did; catches failures by paying on every request",
        ),
        Policy(
            name="always strong",
            cost_usd=always_strong,
            bad_answers=0,
            escalations=0,
            note=strong_note,
        ),
    ]


def calibrate_threshold(
    model: DifficultyModel, examples: Sequence[Example], escalation_rate: float
) -> float:
    """The threshold that escalates this fraction of the TRAINING split.

    A fixed 0.5 assumes the scores spread across the unit interval, and a
    regularised fit on a weak signal does not: measured on a 400-question
    corpus, every score fell between 0.36 and 0.45, so 0.5 escalated nothing and
    0.4 escalated everything. The threshold is then not an economic knob at all
    -- it is a cliff nobody can see. Reading it off the training scores makes
    "escalate the hardest 15%" mean what it says, whatever the fit did.
    """
    scores = sorted((model.score(e.vector) for e in examples), reverse=True)
    if not scores:
        return 0.5
    if escalation_rate <= 0:
        return scores[0] + 1e-9
    index = min(int(round(escalation_rate * len(scores))), len(scores)) - 1
    return scores[max(index, 0)]


def calibrate_for_recall(
    model: DifficultyModel, examples: Sequence[Example], recall: float
) -> float:
    """The highest threshold that still catches this share of the TRAINING
    split's failures -- i.e. the cheapest policy meeting a quality target.

    The other knob, `calibrate_threshold`, is priced in traffic: "escalate the
    hardest 15%". This one is priced in quality: "catch 80% of the failures".
    Operators have a number for the second and rarely for the first, and on a
    weak model the translation between them is brutal and worth seeing -- catch
    80% of failures here and you escalate most of the traffic to do it.

    Read off the training split like every other threshold in this module. What
    the held-out set then reports is a measurement; picking the point on the
    held-out set would make it a fit.
    """
    scores = [model.score(e.vector) for e in examples]
    failures = sorted((score for score, e in zip(scores, examples) if e.label), reverse=True)
    if not scores:
        return 0.5
    escalate_none = max(scores) + 1e-9
    if recall <= 0 or not failures:
        return escalate_none
    wanted = min(math.ceil(recall * len(failures)), len(failures))
    return failures[wanted - 1]


def sweep_thresholds(
    model: DifficultyModel, rows: Sequence[Row], thresholds: Sequence[float]
) -> list[tuple[float, float, int]]:
    """Cost and bad answers at each threshold.

    The threshold is an economic knob, not a statistical one. 0.5 is the default
    only because something has to be, and this table is how an operator picks the
    one that matches what a wrong answer actually costs them.
    """
    out: list[tuple[float, float, int]] = []
    for threshold in thresholds:
        cost = 0.0
        bad = 0
        for row in rows:
            if model.score(row.example.vector) >= threshold and row.strong_eligible:
                cost += row.strong_cost_usd
            else:
                cost += row.route_cost_usd
                bad += row.example.label
        out.append((threshold, cost, bad))
    return out


# --------------------------------------------------------------------------
# choosing the knobs, on the training split only
# --------------------------------------------------------------------------

# Small on purpose. A grid large enough to find something in noise is a way of
# overfitting the cross-validation itself, and each point costs a full fit.
TUNING_GRID: tuple[dict[str, Any], ...] = tuple(
    {"use_words": use_words, "min_df": min_df, "l2": l2}
    for use_words, min_df in ((True, 3), (True, 10), (False, 3))
    for l2 in (1e-4, 1e-2, 1e-1)
)


@dataclass
class Tuning:
    """What the search chose, and what it was choosing between."""

    settings: dict[str, Any]
    cv_auc: float | None
    folds: int
    considered: int
    # Every point, best first, so a flat grid is visible as a flat grid.
    ranking: list[tuple[float, dict[str, Any]]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings": dict(self.settings),
            "cv_auc": self.cv_auc,
            "folds": self.folds,
            "considered": self.considered,
            "ranking": [{"cv_auc": a, "settings": s} for a, s in self.ranking],
        }


def cross_validate(
    examples: Sequence[Example],
    *,
    predicts_tier: str,
    folds: int = 5,
    seed: int = 7,
    **fit_kwargs: Any,
) -> float | None:
    """Mean AUC over grouped folds. A conversation is wholly inside one fold."""
    buckets: list[list[Example]] = [[] for _ in range(folds)]
    for example in examples:
        buckets[min(int(group_fraction(example.group, seed) * folds), folds - 1)].append(example)

    scores: list[float] = []
    for index, held in enumerate(buckets):
        rest = [e for other, bucket in enumerate(buckets) if other != index for e in bucket]
        if not held or not rest or not any(e.label for e in rest):
            continue
        model = build_model(rest, predicts_tier=predicts_tier, **fit_kwargs)
        fold_auc = roc_auc([(model.score(e.vector), e.label) for e in held])
        if fold_auc is not None:
            scores.append(fold_auc)
    return sum(scores) / len(scores) if scores else None


def tune(
    examples: Sequence[Example],
    *,
    predicts_tier: str,
    grid: Sequence[dict[str, Any]] = TUNING_GRID,
    folds: int = 5,
    seed: int = 7,
    **fixed: Any,
) -> Tuning:
    """Pick fit settings by cross-validated AUC, seeing only the training split.

    The held-out set is not consulted here and must not be: choosing the knobs
    on the same rows that report the result is how a model comes to describe its
    own evaluation.
    """
    ranked: list[tuple[float, dict[str, Any]]] = []
    for point in grid:
        score = cross_validate(
            examples, predicts_tier=predicts_tier, folds=folds, seed=seed, **point, **fixed
        )
        if score is not None:
            ranked.append((score, dict(point)))
    # Ties broken by the simpler model: fewer features beats more.
    ranked.sort(key=lambda item: (-item[0], item[1].get("use_words", True), -item[1]["min_df"]))
    if not ranked:
        return Tuning(settings={}, cv_auc=None, folds=folds, considered=len(grid))
    best_auc, best = ranked[0]
    return Tuning(
        settings=best, cv_auc=best_auc, folds=folds, considered=len(grid), ranking=ranked
    )


# --------------------------------------------------------------------------
# the whole job
# --------------------------------------------------------------------------


def train_from_log(
    log: RequestLog,
    *,
    predicts_tier: str,
    strong_tier: str,
    judged_by: str | None = None,
    threshold: float = 0.5,
    holdout: float = 0.25,
    min_examples: int = 40,
    min_positives: int = 5,
    seed: int = 0,
    tune_settings: bool = False,
    target_escalation: float | None = None,
    target_recall: float | None = None,
    **fit_kwargs: Any,
) -> TrainingReport:
    rows = load_rows(log, predicts_tier=predicts_tier, strong_tier=strong_tier)
    examples = [row.example for row in rows]
    positives = sum(e.label for e in examples)
    groups = {e.group for e in examples}
    warnings: list[str] = []

    if len(examples) < min_examples:
        raise TrainingError(
            f"only {len(examples)} usable label(s); at least {min_examples} are needed. "
            "A model fitted on fewer memorises them. Keep the loop running, or lower "
            "--min-examples if you know you are producing a toy."
        )
    if positives < min_positives or positives == len(examples):
        raise TrainingError(
            f"{positives} failure(s) in {len(examples)} label(s). A classifier needs "
            f"both classes and at least {min_positives} of the rare one; with fewer, "
            "'always cheap' is the correct router and the log already says so."
        )
    if len(groups) < 8:
        warnings.append(
            f"only {len(groups)} distinct conversation(s) in the corpus -- the grouped "
            "split has little to hold out, so the honest number below is itself noisy"
        )

    # The rows are split, and the examples follow them. Splitting the examples
    # and looking the rows up afterwards would collapse every request in a
    # conversation onto one row's costs, which is a pricing error that reads as
    # a suspiciously tidy report.
    train_rows = [r for r in rows if group_fraction(r.example.group, seed) >= holdout]
    test_rows = [r for r in rows if group_fraction(r.example.group, seed) < holdout]
    train = [r.example for r in train_rows]
    test = [r.example for r in test_rows]
    if not train or not test:
        raise TrainingError(
            "the grouped split put every conversation on one side. Collect traffic "
            "from more conversations, or lower --holdout."
        )
    if not any(e.label for e in test):
        warnings.append(
            "the held-out set contains no failures, so recall and the bad-answer "
            "columns below are vacuous"
        )

    # The search sees the training split and nothing else, so the held-out
    # numbers below still describe rows that took no part in any decision.
    tuning: Tuning | None = None
    if tune_settings:
        tuning = tune(train, predicts_tier=predicts_tier, **fit_kwargs)
        fit_kwargs = {**fit_kwargs, **tuning.settings}

    # The shipped model is the one fitted on the training split, so the numbers
    # reported describe the exact weights in the file. Refitting on everything
    # afterwards would produce a slightly better model whose evaluation belongs
    # to a different one, and that substitution is how a report stops being
    # about the thing it is attached to.
    model = build_model(
        train,
        predicts_tier=predicts_tier,
        judged_by=judged_by,
        threshold=threshold,
        seed=seed,
        **fit_kwargs,
    )
    if target_escalation is not None and target_recall is not None:
        raise TrainingError(
            "give a target in traffic or a target in quality, not both: "
            "--target-escalation and --target-recall set the same threshold "
            "from different ends and would silently disagree"
        )
    if target_escalation is not None:
        threshold = calibrate_threshold(model, train, target_escalation)
        model.threshold = threshold
    elif target_recall is not None:
        threshold = calibrate_for_recall(model, train, target_recall)
        model.threshold = threshold
    # Every point is a share of the TRAINING split's scores, so the table spans
    # the model's actual range instead of a grid it may never touch.
    sweep_points = sorted(
        {calibrate_threshold(model, train, rate) for rate in (0.05, 0.1, 0.2, 0.3, 0.5)}
        | {threshold}
    )
    grouped = evaluate(model, test, threshold)

    # The same procedure with the wrong split, for contrast only.
    row_train, row_test = split_by_row(examples, holdout=holdout, seed=seed)
    leaky = build_model(
        row_train,
        predicts_tier=predicts_tier,
        judged_by=judged_by,
        threshold=threshold,
        seed=seed,
        **fit_kwargs,
    )
    # Its own threshold, calibrated on its own training rows: a threshold read
    # off another fit's scores would make the contrast a comparison of
    # thresholds rather than of splits.
    leaky_threshold = (
        calibrate_threshold(leaky, row_train, target_escalation)
        if target_escalation is not None
        else threshold
    )
    ungrouped = evaluate(leaky, row_test, leaky_threshold)

    if grouped.auc is not None and grouped.auc <= 0.55:
        warnings.append(
            f"held-out AUC is {grouped.auc:.3f}: this model ranks a failed request above a "
            "passing one barely more often than a coin would"
            + (", and below chance means the features are actively misleading on unseen "
               "conversations" if grouped.auc < 0.5 else "")
        )
    unreviewed = sum(1 for row in rows if row.example.label and not row.reviewed)
    if positives and unreviewed / positives >= 0.5:
        warnings.append(
            f"{unreviewed} of the {positives} failure(s) here were failed without a review "
            "-- an empty or cut-off answer, which the verification loop already catches for "
            "nothing. A classifier fitted on this corpus is mostly learning to predict how "
            "long an answer will be, and nothing needs that predicted. Raise the tier's "
            "output budget and collect again"
        )
    if grouped.auc_ci is not None and grouped.auc_ci[0] <= 0.5 <= grouped.auc_ci[1]:
        warnings.append(
            f"the 95% interval around the held-out AUC is "
            f"[{grouped.auc_ci[0]:.3f}, {grouped.auc_ci[1]:.3f}], which contains 0.500: "
            f"{grouped.positives} failure(s) in {grouped.examples} held-out request(s) is "
            "not enough evidence to say this model ranks better than chance, whatever the "
            "point estimate above reads"
        )
    if grouped.examples and grouped.escalation_rate in (0.0, 1.0):
        sent = "every request" if grouped.escalation_rate else "no request"
        warnings.append(
            f"at threshold {threshold} the model sends {sent} to the strong tier on the "
            "held-out set: the threshold sits outside the scores this model produces, so "
            "the policy table below is that one policy under another name"
        )

    report = TrainingReport(
        model=model,
        rows=len(rows),
        groups=len(groups),
        train_size=len(train),
        test_size=len(test),
        grouped=grouped,
        ungrouped=ungrouped,
        tuning=tuning,
        policies=price_policies(model, test_rows, threshold),
        sweep=sweep_thresholds(model, test_rows, sweep_points),
        warnings=warnings,
    )
    model.metrics = {
        "grouped": grouped.to_dict(),
        "ungrouped": ungrouped.to_dict(),
        "train_size": len(train),
        "test_size": len(test),
        "groups": len(groups),
    }
    return report


def format_report(report: TrainingReport) -> str:
    model = report.model
    lines: list[str] = []
    lines.append(f"corpus          {report.rows} labelled request(s) in {report.groups} conversation(s)")
    lines.append(f"                {model.positives} failure(s) in the training split of {report.train_size}")
    lines.append(f"held out        {report.test_size} request(s), whole conversations only")
    lines.append(f"model           {model.fingerprint}  predicts {model.predicts_tier}"
                 + (f", judged by {model.judged_by}" if model.judged_by else ""))
    lines.append(f"                {len(model.weights)} weight(s); an unseen prompt scores "
                 f"{model.base_rate:.3f}, the base rate")

    if report.tuning is not None:
        lines.append("")
        if report.tuning.cv_auc is None:
            lines.append("tuning          no fold could be scored; the shipped defaults were used")
        else:
            chosen = ", ".join(f"{k}={v}" for k, v in sorted(report.tuning.settings.items()))
            lines.append(
                f"tuning          {report.tuning.considered} setting(s) by "
                f"{report.tuning.folds}-fold cross-validation, on the training split only"
            )
            lines.append(f"                chose {chosen}  (cv auc {report.tuning.cv_auc:.3f})")
            spread = report.tuning.ranking[0][0] - report.tuning.ranking[-1][0]
            if spread < 0.02:
                lines.append(
                    f"                every setting scored within {spread:.3f} AUC of every "
                    "other: the grid found nothing to choose between, and the winner is noise"
                )

    lines.append("")
    lines.append("held out by conversation -- the number to believe")
    lines.extend(_format_metrics(report.grouped))
    lines.append("")
    lines.append("held out by row -- the same fit with the WRONG split, for contrast")
    lines.extend(_format_metrics(report.ungrouped))
    gap = report.ungrouped.accuracy - report.grouped.accuracy
    if gap > 0.01:
        lines.append(
            f"  the row split reads {gap * 100:.1f} points better on the same data. "
            "That gap is the model recognising conversations, not difficulty."
        )

    if report.policies:
        lines.append("")
        lines.append("what the held-out traffic would have cost, per policy")
        lines.append(f"  {'policy':<34} {'cost':>12}  {'bad answers':>11}  escalations")
        for policy in report.policies:
            lines.append(
                f"  {policy.name:<34} ${policy.cost_usd:>11.6f}  "
                f"{policy.bad_answers:>11}  {policy.escalations:>11}"
            )
            if policy.note:
                lines.append(f"      {policy.note}")
        lines.append(
            "  Money and bad answers are separate columns because they are separate "
            "things. What a wrong answer costs you is yours to supply; this report "
            "will not invent a price for it."
        )

    if report.sweep:
        lines.append("")
        lines.append(
            "threshold sweep on the held-out set -- each point is a share of the "
            "training split's own scores"
        )
        for threshold, cost, bad in report.sweep:
            marker = "  <- configured" if abs(threshold - model.threshold) < 1e-9 else ""
            lines.append(
                f"  {threshold:>6.4f}   ${cost:>11.6f}   {bad:>4} bad answer(s){marker}"
            )

    top = model.top_features()
    if top:
        lines.append("")
        lines.append("strongest weights (+ pushes toward the strong tier)")
        for name, weight in top:
            lines.append(f"  {weight:>+8.3f}  {name}")

    for warning in report.warnings:
        lines.append("")
        lines.append(f"WARNING: {warning}")

    return "\n".join(lines)


def _format_metrics(metrics: Metrics) -> list[str]:
    if not metrics.examples:
        return ["  (nothing held out)"]
    precision = f"{metrics.precision:.3f}" if metrics.precision is not None else "n/a"
    recall = f"{metrics.recall:.3f}" if metrics.recall is not None else "n/a"
    auc = f"{metrics.auc:.3f}" if metrics.auc is not None else "n/a"
    lines = [
        f"  accuracy        {metrics.accuracy:.3f}   "
        f"majority-class baseline {metrics.majority_accuracy:.3f}",
        f"  auc             {auc}   0.500 is a coin toss; this one needs no threshold",
        f"  precision       {precision}   of the requests it sent to the strong tier",
        f"  recall          {recall}   of the failures it caught",
        f"  escalates       {metrics.escalation_rate * 100:.1f}% of requests",
    ]
    if metrics.auc_ci is not None:
        low, high = metrics.auc_ci
        verdict = (
            "includes 0.500, so this held-out set cannot tell the model from chance"
            if low <= 0.5 <= high
            else "clear of 0.500"
        )
        lines.insert(2, f"  95% CI          [{low:.3f}, {high:.3f}]   {verdict}")
    base_rate = metrics.positives / metrics.examples
    if metrics.precision is not None and base_rate:
        # Accuracy punishes any escalation at all when failures are rare, so it
        # cannot answer "was escalating these ones better than escalating at
        # random?". This can: 1.0 means the model picked no better than a dice.
        lines.append(
            f"  lift            {metrics.precision / base_rate:.2f}x   its escalations "
            f"were failures {metrics.precision * 100:.1f}% of the time, against a "
            f"{base_rate * 100:.1f}% base rate"
        )

    ranks = (metrics.auc or 0.0) > 0.55 and (metrics.precision or 0.0) > base_rate * 1.1
    if metrics.accuracy <= metrics.majority_accuracy and ranks:
        lines.append(
            "  accuracy is below the majority baseline, which is what any escalating model "
            "scores when failures are rare -- read the lift and the AUC instead, and the "
            "cost table below"
        )
    elif metrics.accuracy <= metrics.majority_accuracy:
        lines.append(
            "  this model does not beat answering the majority class every time; "
            "it has learned the class balance and nothing else"
        )
    return lines


__all__ = [
    "Metrics",
    "Policy",
    "Row",
    "TrainingError",
    "TrainingReport",
    "evaluate",
    "format_report",
    "load_rows",
    "price_policies",
    "sweep_thresholds",
    "train_from_log",
]
