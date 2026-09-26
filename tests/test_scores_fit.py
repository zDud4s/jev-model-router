"""`benchmarks fit`: what each benchmark is worth, and the level mapping, fitted from judged outcomes.

The outcomes are generated from known parameters with exact pass fractions
(round(n x p)), so a fit that works recovers them and the test is deterministic.
"""

from __future__ import annotations

import json
import math

import pytest

from jev_model_router.calibration import Outcome
from jev_model_router.capabilities import CapabilityRouter
from jev_model_router.catalog import CatalogReport, Offered, Source
from jev_model_router.config import parse_config
from jev_model_router.discovery import expand
from jev_model_router.scores import parse_scores, write_sidecar
from jev_model_router.scores_derive import blend, effort_prior, evidence_for, line_for
from jev_model_router.scores_fit import fit, write_fit

from test_scores import raw_config

# code score, base score, profile level
MODELS = {"m-hi": (70, 60, 2.5), "m-lo": (70, 60, 0.5), "m-a": (40, 45, 1.5), "m-b": (50, 50, 1.5)}
NEEDS = {"reasoning": 1.0, "niche": 0.0}
PRIOR = 0.7


# A second benchmark on the same requirement, which orders the models differently.
CODE2 = {"m-hi": 50, "m-lo": 80, "m-a": 45, "m-b": 65}


def curated(two: bool = False) -> dict:
    points = [{"benchmark": "code", "model": m, "effort": "high", "score": c, "origin": "independent"}
              for m, (c, _, _) in MODELS.items()]
    points += [{"benchmark": "base", "model": m, "effort": "high", "score": b, "origin": "independent"}
               for m, (_, b, _) in MODELS.items()]
    benches = {"code": {"description": "d", "requirements": {"reasoning": PRIOR}},
               "base": {"description": "b", "requirements": {}}}
    if two:
        benches["code2"] = {"description": "d2", "requirements": {"reasoning": 0.5}}
        points += [{"benchmark": "code2", "model": m, "effort": "high", "score": c, "origin": "independent"}
                   for m, c in CODE2.items()]
    return {"benchmarks": benches, "points": points}


def world(two: bool = False, sidecar: dict | None = None, full: bool = False):
    raw = raw_config(benchmarks={"path": "b.yaml"}, profiles=[
        {"match": m, "level": level, "list_prices": {"input": 1.0, "output": 5.0}}
        for m, (_, _, level) in MODELS.items()
    ])
    config = parse_config(raw)
    source = Source(name="cli", models={m: Offered(m, efforts=("high",)) for m in MODELS}, cli_version="9")
    report = CatalogReport(checked_at="now", tiers={}, sources={}, unconfigured={}, discovered={"cli": source})
    scores = parse_scores(curated(two), config.router.capabilities, sidecar=sidecar)
    expanded, _, _ = expand(config, report, scores)
    return (expanded, scores, config, report) if full else (expanded, scores)


def outcomes(expanded, scores, n: int, *, scale: float = 0.2 / PRIOR) -> list[Outcome]:
    caps = expanded.router.capabilities
    router = CapabilityRouter(expanded, ask=None)
    profiles = {m: next(p for p in caps.profiles if p.match == m) for m in MODELS}
    line = line_for(caps, scores, [((m,), "high", profiles[m]) for m in MODELS])
    out = []
    for m in MODELS:
        tier = f"cli:{m}@high"
        prior, _ = effort_prior(caps, profiles[m], "high")
        ev = evidence_for(caps, scores, (m,), "high", {"code": {"reasoning": PRIOR * scale}, "base": {}})
        levels = blend(ev, prior, caps.requirements, a=line[0], k=line[1], c0=scores.c0)
        passes = round(n * router.success(NEEDS, tier, levels=levels))
        out += [Outcome(tier, NEEDS, i < passes, "outcome") for i in range(n)]
    return out


def test_no_outcomes_fit_nothing():
    expanded, scores = world()
    result = fit(expanded, scores, [])
    assert result.outcomes == 0 and result.scales == {} and (result.delta_a, result.k_ratio) == (0.0, 1.0)


def test_a_handful_of_outcomes_leaves_the_parameters_near_the_prior():
    expanded, scores = world()
    result = fit(expanded, scores, outcomes(expanded, scores, 2))
    assert result.scales["code"] == pytest.approx(1.0, abs=0.15)
    assert result.delta_a == pytest.approx(0.0, abs=0.3) and result.k_ratio == pytest.approx(1.0, abs=0.3)


def test_many_outcomes_recover_the_benchmark_scale_that_generated_them():
    expanded, scores = world()
    result = fit(expanded, scores, outcomes(expanded, scores, 1000))
    assert result.scales["code"] == pytest.approx(0.2 / PRIOR, abs=0.07)
    assert result.delta_a == pytest.approx(0.0, abs=0.1) and result.k_ratio == pytest.approx(1.0, abs=0.1)
    assert result.loglik > result.loglik_prior
    assert [b for b, _ in result.moved] == ["code"]


def test_the_fit_is_deterministic():
    expanded, scores = world()
    data = outcomes(expanded, scores, 50)
    assert fit(expanded, scores, data, fitted_at="t") == fit(expanded, scores, data, fitted_at="t")


def test_an_outcome_on_a_tier_no_longer_carded_is_dropped():
    expanded, scores = world()
    stale = [Outcome("cli:gone@high", NEEDS, False, "outcome")] * 50
    assert fit(expanded, scores, stale).outcomes == 0


def test_write_fit_touches_only_the_sidecars_fit_block_and_keeps_earlier_scales(tmp_path):
    expanded, scores = world()
    side = tmp_path / "b.derived.json"
    write_sidecar(side, jev={"code": {"hash": "h", "needs": {}}})
    earlier = parse_scores({"benchmarks": {"code": {"description": "d", "requirements": {"reasoning": PRIOR}},
                                           "base": {"description": "b", "requirements": {}}}, "points": []},
                           expanded.router.capabilities,
                           sidecar={"fit": {"scales": {"old": {"scale": 0.4, "hash": "x"}}}})
    write_fit(side, fit(expanded, scores, outcomes(expanded, scores, 20)), earlier, expanded)
    data = json.loads(side.read_text(encoding="utf-8"))
    assert data["jev"] == {"code": {"hash": "h", "needs": {}}}
    assert data["fit"]["outcomes"] == 80
    assert set(data["fit"]["scales"]) == {"old", "code"} and data["fit"]["scales"]["old"]["scale"] == 0.4
    assert set(data["fit"]) == {"scales", "delta_a", "k_ratio", "outcomes", "fitted_at", "loglik"}


def served_loglik(expanded, data) -> float:
    """log L of the outcomes on the cards serving builds, clamped as the fit clamps."""
    router = CapabilityRouter(expanded, ask=None)
    total = 0.0
    for o in data:
        p = min(1 - 1e-4, max(1e-4, router.success(o.needs, o.tier)))
        total += math.log(p if o.passed else 1 - p)
    return total


def test_the_fitted_log_l_is_the_log_l_of_the_cards_serving_builds_and_a_refit_stays_put(tmp_path):
    expanded, scores = world(two=True)
    data = outcomes(expanded, scores, 300)
    first = fit(expanded, scores, data)
    assert set(first.scales) == {"code", "code2"}
    assert first.loglik_prior == pytest.approx(served_loglik(expanded, data), abs=1e-6)
    side = tmp_path / "b.derived.json"
    write_fit(side, first, scores, expanded)
    served, rescored = world(two=True, sidecar=json.loads(side.read_text(encoding="utf-8")))
    assert first.loglik == pytest.approx(served_loglik(served, data), abs=1e-6)
    again = fit(served, rescored, data)
    assert again.loglik_prior == pytest.approx(first.loglik, abs=1e-6)
    assert again.loglik >= first.loglik - 1e-6
    for b in first.scales:
        assert again.scales[b] == pytest.approx(first.scales[b], abs=0.02)
    assert again.delta_a == pytest.approx(first.delta_a, abs=0.02)
    assert again.k_ratio == pytest.approx(first.k_ratio, abs=0.02)
