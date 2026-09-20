"""Choosing the fit's settings and the threshold without touching the held-out set."""

from __future__ import annotations

import random

from llm_router.classifier import Example, build_model
from llm_router.training import (
    TUNING_GRID,
    auc_interval,
    calibrate_threshold,
    cross_validate,
    roc_auc,
    tune,
)


def test_auc_is_the_chance_a_failure_outranks_a_pass() -> None:
    assert roc_auc([(0.9, 1), (0.1, 0)]) == 1.0
    assert roc_auc([(0.1, 1), (0.9, 0)]) == 0.0
    # Ties are half a win each, so a model that scores everything the same is
    # a coin toss rather than a winner.
    assert roc_auc([(0.5, 1), (0.5, 0)]) == 0.5
    assert roc_auc([(0.5, 1), (0.5, 1)]) is None


def _separable(n: int = 120) -> list[Example]:
    rng = random.Random(3)
    out = []
    for i in range(n):
        hard = i % 3 == 0
        out.append(
            Example(
                vector={"n:log_chars": 0.9 if hard else 0.2, "w:x": rng.random()},
                label=1 if hard else 0,
                group=f"g{i}",
            )
        )
    return out


def test_cross_validation_scores_a_signal_above_chance_and_noise_at_it() -> None:
    signal = cross_validate(_separable(), predicts_tier="cheap")
    noise = cross_validate(
        [Example(vector={"n:log_chars": 0.5}, label=i % 3 == 0, group=f"g{i}") for i in range(120)],
        predicts_tier="cheap",
    )

    assert signal > 0.9
    assert noise is not None and 0.4 <= noise <= 0.6


def test_a_fold_never_splits_a_conversation() -> None:
    # Every row shares one group: no fold can hold both sides of it, so nothing
    # is scorable and the honest answer is None rather than a number.
    same = [Example(vector={"n:log_chars": 0.5}, label=i % 2, group="one") for i in range(40)]

    assert cross_validate(same, predicts_tier="cheap") is None


def test_tuning_reports_what_it_compared_not_just_what_it_picked() -> None:
    result = tune(_separable(), predicts_tier="cheap")

    assert result.considered == len(TUNING_GRID)
    assert len(result.ranking) == len(TUNING_GRID)
    assert result.cv_auc == result.ranking[0][0]
    assert set(result.settings) == {"use_words", "min_df", "l2"}


def test_a_tie_is_broken_towards_the_smaller_model() -> None:
    # Two settings that cannot differ: the corpus has no word features at all,
    # so dropping them must be the choice rather than a coin flip.
    flat = [
        Example(vector={"n:log_chars": 0.9 if i % 3 else 0.2}, label=i % 3 == 0, group=f"g{i}")
        for i in range(90)
    ]

    assert tune(flat, predicts_tier="cheap").settings["use_words"] is False


def test_words_can_be_left_out_of_the_fit_entirely() -> None:
    examples = _separable()

    with_words = build_model(examples, predicts_tier="cheap")
    without = build_model(examples, predicts_tier="cheap", use_words=False)

    assert any(name.startswith("w:") for name in with_words.weights)
    assert not any(name.startswith("w:") for name in without.weights)


def test_the_threshold_is_read_off_the_scores_the_model_actually_produces() -> None:
    # A graded feature, so the scores are distinct: on a weak signal they sit
    # in a narrow band where a fixed 0.5 escalates all of them or none.
    examples = [
        Example(vector={"n:log_chars": i / 120}, label=int(i > 90), group=f"g{i}")
        for i in range(120)
    ]
    model = build_model(examples, predicts_tier="cheap", use_words=False)

    threshold = calibrate_threshold(model, examples, 0.25)
    escalated = sum(model.score(e.vector) >= threshold for e in examples)

    assert abs(escalated / len(examples) - 0.25) <= 0.02


def test_escalating_nobody_is_a_threshold_above_every_score() -> None:
    examples = _separable()
    model = build_model(examples, predicts_tier="cheap")

    threshold = calibrate_threshold(model, examples, 0.0)

    assert all(model.score(e.vector) < threshold for e in examples)


def test_the_interval_is_wide_when_there_is_little_to_go_on() -> None:
    # The same ranking at two sample sizes. The point estimate says the same
    # thing both times and only the interval distinguishes "measured" from
    # "happened to come out that way", which is why it is printed at all.
    # Repeating one pattern leaves the AUC untouched -- every pairwise
    # comparison is duplicated in step -- so only the sample size changes.
    pattern = [(0.9, 1), (0.95, 0), (0.6, 0), (0.4, 0)]

    small = auc_interval(pattern * 3)
    large = auc_interval(pattern * 300)

    assert small is not None and large is not None
    assert roc_auc(pattern * 3) == roc_auc(pattern * 300)
    assert (small[1] - small[0]) > 3 * (large[1] - large[0])
    assert large[0] > 0.5


def test_an_interval_stays_inside_the_range_an_auc_can_take() -> None:
    perfect = [(0.9, 1), (0.9, 1), (0.1, 0), (0.1, 0)]

    low, high = auc_interval(perfect)

    assert high == 1.0
    assert 0.0 <= low <= 1.0
    # A class that never appears has no interval, for the same reason it has no
    # AUC: there is no pair to rank.
    assert auc_interval([(0.9, 1), (0.8, 1)]) is None
