"""One ability scale across benchmarks, fitted from the models they share."""

from __future__ import annotations

import pytest

from llm_router.scores_scale import Obs, fit_scale


def _synthetic():
    theta = {(f"m{i}", None): -1.5 + 3.5 * i / 23 for i in range(24)}
    bench = {"b1": (0.8, -0.5), "b2": (1.0, 0.4), "b3": (1.3, 0.0), "b4": (0.9, 0.9)}
    obs = [Obs(b, n, a * t - beta, 1.0) for b, (a, beta) in bench.items() for n, t in theta.items()]
    return theta, bench, obs


def test_the_scale_recovers_known_difficulties_spreads_and_abilities():
    theta, bench, obs = _synthetic()
    fitted = fit_scale(obs)
    # The truth, put on the same footing: mean alpha 1, mean beta/alpha 0.
    ma = sum(a for a, _ in bench.values()) / len(bench)
    shift = sum(b / a for a, b in bench.values()) / len(bench)
    for n, t in theta.items():
        assert fitted.theta[n] == pytest.approx(ma * (t - shift), abs=0.08)  # the priors shrink a little
    for b, (a, _) in bench.items():
        assert fitted.alpha[b] == pytest.approx(a / ma, abs=0.08)
    assert fit_scale(obs) == fitted  # deterministic


def test_models_seen_on_one_benchmark_only_do_not_move_its_scale():
    _, _, obs = _synthetic()
    crowd = [Obs("b1", (f"weak-{i}", None), -3.0, 1.0) for i in range(20)]
    assert fit_scale(obs + crowd) == fit_scale(obs)
    assert fit_scale(obs + crowd).ability("b1", -3.0) is not None  # still read through b1's scale


def test_only_the_largest_linked_group_of_benchmarks_is_the_scale():
    # {b1, b2} and {b3, b4} share no model: nothing places one pair against the other.
    main = [Obs(b, (f"m{i}", None), 0.3 * i - s, 1.0) for b, s in (("b1", 0.0), ("b2", 0.5)) for i in range(6)]
    island = [Obs(b, (f"n{i}", None), 0.2 * i, 1.0) for b in ("b3", "b4") for i in range(3)]
    fitted = fit_scale(main + island)
    assert set(fitted.alpha) == {"b1", "b2"} and fitted.detached == ("b3", "b4")
    assert fitted.ability("b3", 0.0) is None
    assert fitted == fit_scale(main + island[::-1])  # order-independent
    # A tie in benchmarks goes to the group with more linking models, then to the first name.
    assert fit_scale(main[:6] + main[6:] + [Obs(b, (f"n{i}", None), 0.1, 1.0) for b in ("b3", "b4")
                                            for i in range(7)]).detached == ("b1", "b2")
    even = [Obs(b, (f"n{i}", None), 0.2 * i, 1.0) for b in ("b3", "b4") for i in range(6)]
    assert fit_scale(main + even).detached == ("b3", "b4")
