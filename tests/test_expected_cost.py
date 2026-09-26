"""The expected-cost rule, the cheap-first cascade, and calibration learnt from the log.

Jev is never called: every router answers from a scripted `Ask`.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from jev_model_router.calibration import fit_family_scales, log_outcomes, write_family_scales
from jev_model_router.capabilities import CapabilityRouter
from jev_model_router.config import ConfigError, parse_config
from jev_model_router.db import LogEntry, RequestLog
from jev_model_router.verification import Verdict, VerificationOutcome, Verifier

from conftest import FakeBackend, make_request
from test_capabilities import ALL, Ask, decide, raw_config, router

MID = {"reasoning": 0.6, "code": 0.6}


def cost_of(r: CapabilityRouter, tier: str) -> float:
    return r.cost(tier, 100)


# ---------------------------------------------------------------- the rule
def test_what_a_failure_costs_moves_the_bar():
    low = router(Ask(MID), rule="expected_cost")
    picks = {k: decide(low, packet=p).tier for k, p in
             {"low": {"stakes": "low"}, "high": {"stakes": "high"}, "caught": {"verifiable": True}}.items()}
    # High stakes buy a stronger model; a failure the client's tests will catch
    # costs only the redo, so the cheap try is worth it.
    assert cost_of(low, picks["high"]) > cost_of(low, picks["low"])
    assert cost_of(low, picks["caught"]) <= cost_of(low, picks["low"])
    assert picks["high"] == "sub"


def test_an_easy_task_goes_cheap_under_either_rule():
    for rule in ("target", "expected_cost"):
        assert decide(router(Ask({"reasoning": 0.05, "code": 0.05}), rule=rule)).tier == "cheap"


def test_without_a_stakes_field_jev_precision_reading_sets_them():
    config = raw_config(rule="expected_cost")
    config["router"]["capabilities"]["requirements"]["precision"] = "The state is a task packet. Costly if wrong?"
    r = CapabilityRouter(parse_config(config), ask=Ask({**MID, "precision": 1.0}))
    detail = asyncio.run(r.decide(make_request(), ALL)).detail
    assert detail["stakes_from"] == "jev precision" and detail["stakes"] == pytest.approx(3.0)
    r = CapabilityRouter(parse_config(config), ask=Ask({**MID, "precision": 0.0}))
    assert asyncio.run(r.decide(make_request(), ALL)).detail["stakes"] == pytest.approx(0.5)


def test_every_option_shows_its_expected_cost_and_the_pick_is_the_least():
    d = decide(router(Ask(MID), rule="expected_cost"))
    options = [o for o in d.detail["options"] if o["expected"] is not None]
    assert d.tier == min(options, key=lambda o: o["expected"])["tier"]
    assert "expected" in json.loads(d.reason)
    assert d.detail["rule"] == "least expected cost" and d.detail["redo_tier"] in ALL


def test_the_router_own_verifier_counts_as_catching_a_failure():
    raw = raw_config(rule="expected_cost")
    raw["verification"] = {"enabled": True, "verifier_tier": "top", "escalate_to": "top"}
    r = CapabilityRouter(parse_config(raw), ask=Ask(MID))
    assert r._verified_share("cheap") == 1.0 and r._verified_share("top") == 0.0
    plain = router(Ask(MID), rule="expected_cost")
    assert cost_of(r, decide(r).tier) <= cost_of(plain, decide(plain).tier)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"rule": "vibes"}, "rule must be one of"),
        ({"failure": {"detected": -1}}, "cannot be negative"),
        ({"failure": {"surprise": 1}}, "takes only"),
        ({"family_scales": {"x": 9}}, "family_scales"),
    ],
)
def test_settings_that_would_mislead_are_refused(change, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(raw_config(**change))


# ---------------------------------------------------------------- the cascade
def test_a_retry_after_a_failure_never_goes_to_a_weaker_tier():
    # `mid` failed; cheap is rated below it and is no longer offered.
    d = decide(router(Ask(MID)), packet={"failed_tiers": ["mid"]})
    assert d.tier != "cheap" and d.detail["failed_bar"] is not None


def test_escalation_names_a_tier_as_strong_or_none():
    r = router(Ask(MID))
    assert asyncio.run(r.escalation(make_request(), "cheap", ALL)) not in (None, "cheap")
    # Both level-3 tiers already failed: nothing left that is rated as strong.
    request = make_request(packet={"failed_tiers": ["sub"]})
    assert asyncio.run(r.escalation(request, "top", ALL)) is None


def test_the_verifier_escalates_where_the_router_says(fake_backends):
    raw = raw_config()
    raw["verification"] = {"enabled": True, "verifier_tier": "top", "escalate_to": "auto"}
    config = parse_config(raw)
    backends = {name: FakeBackend(tier) for name, tier in config.tiers.items()}

    class Chooser:
        calls = []

        async def escalation(self, request, served, eligible):
            self.calls.append(served)
            return "sub"

    chooser = Chooser()
    verifier = Verifier(config, backends, router=chooser)
    outcome = VerificationOutcome(verdict=Verdict.FAIL)
    asyncio.run(verifier._escalate(make_request(), "cheap", ALL, outcome))
    assert chooser.calls == ["cheap"] and outcome.escalated_to == "sub" and backends["sub"].calls

    class Nothing:
        async def escalation(self, request, served, eligible):
            return None

    outcome = VerificationOutcome(verdict=Verdict.FAIL)
    asyncio.run(Verifier(config, backends, router=Nothing())._escalate(make_request(), "cheap", ALL, outcome))
    assert not outcome.escalated and "as strong" in outcome.reason


# ---------------------------------------------------------------- learning from the log
def _row(log: RequestLog, i: int, tier: str, needs: dict, verdict: Verdict | None, failed=()):
    reason = {"rule": "x", "need": needs, **({"skipped_failed": list(failed)} if failed else {})}
    log.record(LogEntry(
        request_id=f"r{i}", prompt_text="t", tier=tier, route_model="capabilities:abc",
        route_reason=json.dumps(reason),
        verification=VerificationOutcome(verdict=verdict) if verdict else None,
    ))


def test_a_family_that_passes_more_than_its_card_predicts_gets_a_smaller_scale():
    config = parse_config(raw_config())
    r = CapabilityRouter(config, ask=Ask())
    log = RequestLog(":memory:")
    # `mid` is predicted to pass MID tasks ~70%; it passed 29 of 30.
    for i in range(30):
        _row(log, i, "mid", MID, Verdict.PASS if i else Verdict.FAIL)
    # A client resend: `cheap` failed this task.
    _row(log, 99, "mid", MID, None, failed=["cheap"])
    outcomes = log_outcomes(log, config)
    assert len(outcomes) == 31 and sum(o.source == "client" for o in outcomes) == 1
    fits = {f.family: f for f in fit_family_scales(r, outcomes, min_samples=20)}
    mid = fits["mid"]
    assert mid.n == 30 and mid.scale is not None and mid.scale < 1.0
    assert r.success(MID, "mid", scale=mid.scale) == pytest.approx(29 / 30, abs=0.03)
    assert fits["cheap"].scale is None  # one trial is not evidence


def test_a_fitted_family_scale_overrides_the_global_one():
    from test_discovery import expanded

    config, _, _ = expanded(miss_scale=1.0, family_scales={"claude-opus-*": 0.2})
    r = CapabilityRouter(config, ask=Ask())
    needs = {"reasoning": 0.9, "niche": 0.9}
    opus, luna = "claude:claude-opus-5-5@high", "codex:gpt-5.6-luna@high"
    assert config.router.capabilities.cards[opus].family == "claude-opus-*"
    assert r.success(needs, opus) == pytest.approx(r.success(needs, opus, scale=0.2))
    assert r.success(needs, luna) == pytest.approx(r.success(needs, luna, scale=1.0))


def test_family_scales_are_written_beside_miss_scale(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("router:\n  capabilities:\n    miss: [0.9, 0.5, 0.2, 0.05]\n    miss_scale: 0.5\n", encoding="utf-8")
    write_family_scales(path, {"claude-opus-5-5": 0.41234})
    text = path.read_text(encoding="utf-8")
    assert '    family_scales: {"claude-opus-5-5": 0.412}' in text
    write_family_scales(path, {"claude-opus-5-5": 0.3, "gpt-5.5": 0.9})
    assert path.read_text(encoding="utf-8").count("family_scales:") == 1
    cfg = parse_config(raw_config(family_scales={"claude-opus-5-5": 0.3}))
    assert cfg.router.capabilities.family_scales == {"claude-opus-5-5": 0.3}
