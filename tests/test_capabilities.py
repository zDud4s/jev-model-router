"""The capabilities router: Jev's reading of the task, the cards, and the two currencies.

Jev is never called: every router here is given an `ask` that answers from a
script, and records the packet and questions it was shown.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from jev_model_router.app import create_app
from jev_model_router.calls import CallRouter
from jev_model_router.capabilities import CapabilityRouter, Option, build_packet, failed_tiers
from jev_model_router.config import ConfigError, Prices, parse_config
from jev_model_router.db import RequestLog
from jev_model_router.routing import JevUnavailable, build_router
from jev_model_router.schemas import Usage

from conftest import BASE_CONFIG, make_request

REQUIREMENTS = {
    "reasoning": "The state is a task packet. Does the task need deep, multi-step reasoning?",
    "code": "The state is a task packet. Does the task need writing or changing code?",
}

CAPS: dict[str, Any] = {
    "jev_tier": "judge",
    "target": 0.8,
    "requirements": REQUIREMENTS,
    "subscriptions": {"claude": {"window_hours": 5}},
    "cards": {
        "cheap": {"levels": {"reasoning": 1, "code": 1}},
        "mid": {"levels": {"reasoning": 2, "code": 2}},
        "top": {"levels": {"reasoning": 3, "code": 3}},
        "sub": {"levels": {"reasoning": 3, "code": 3}, "list_prices": {"input": 5.0, "output": 25.0}},
        "cx": {"levels": {"reasoning": 2, "code": 2}, "list_prices": {"input": 2.0, "output": 10.0}},
    },
}


def raw_config(**caps_overrides: Any) -> dict[str, Any]:
    raw = copy.deepcopy(BASE_CONFIG)
    raw["tiers"].update(
        {
            "sub": {"backend": "claude_cli", "model": "claude-opus-5", "context_window": 1_000_000, "effort": "high"},
            "cx": {"backend": "codex_cli", "model": "gpt-6-sol", "context_window": 272_000},
            "judge": {"backend": "jev", "model": "jev-latest", "context_window": 32000},
        }
    )
    raw["router"] = {"kind": "capabilities", "default_tier": "mid", "capabilities": {**copy.deepcopy(CAPS), **caps_overrides}}
    return raw


class Ask:
    """Answers every requirement with a scripted probability, or raises."""

    def __init__(self, needs: dict[str, float] | None = None, error: Exception | None = None) -> None:
        self.needs = needs or {}
        self.error = error
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def __call__(self, packet: str, questions: dict[str, str]) -> dict[str, float]:
        self.calls.append((packet, questions))
        if self.error:
            raise self.error
        return {k: self.needs.get(k, 0.0) for k in questions}


def router(ask: Ask, clock=None, **caps_overrides: Any) -> CapabilityRouter:
    config = parse_config(raw_config(**caps_overrides))
    kwargs = {"clock": clock} if clock else {}
    return CapabilityRouter(config, ask=ask, **kwargs)


ALL = ["cheap", "mid", "top", "sub", "cx"]
HARD = {"reasoning": 0.9, "code": 0.9}
EASY = {"reasoning": 0.05, "code": 0.05}


def decide(r: CapabilityRouter, candidates=ALL, **request):
    return asyncio.run(r.decide(make_request(**request), list(candidates)))


# ---------------------------------------------------------------- config
def test_the_kind_is_built_from_config():
    config = parse_config(raw_config())
    assert isinstance(build_router(config), CapabilityRouter)
    assert config.router.capabilities.cards["sub"].list_prices.output == 25.0


@pytest.mark.parametrize(
    "change, message",
    [
        ({"jev_tier": "mid"}, "backend 'jev'"),
        ({"cards": {"cheap": {"levels": {"speed": 2}}}}, "unknown requirements"),
        ({"cards": {"cheap": {"levels": {"code": 4}}}}, "between 0 and 3"),
        ({"cards": {"sub": {"levels": {"code": 3}}}}, "list_prices"),
        ({"cards": {"judge": {"levels": {"code": 3}}}}, "cannot answer"),
        ({"target": 0}, "target"),
        ({"miss": [0.9, 0.5]}, "four probabilities"),
        ({"miss": [0.9, 0.2, 0.5, 0.05]}, "never rise"),
        ({"floor": 1.0}, "floor"),
    ],
    ids=["jev-tier", "stray-level", "level-range", "no-list-prices", "judge-card", "target", "miss", "miss-order",
         "floor"],
)
def test_a_card_or_setting_that_would_mislead_is_refused(change, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(raw_config(**change))


def test_a_capabilities_block_on_another_router_kind_is_refused():
    raw = raw_config()
    raw["router"]["kind"] = "static"
    with pytest.raises(ConfigError, match="does nothing"):
        parse_config(raw)


def test_the_jev_failure_mode_defaults_to_fallback_and_accepts_reject():
    assert parse_config(raw_config()).router.capabilities.on_jev_failure == "fallback"
    assert parse_config(raw_config(on_jev_failure="reject")).router.capabilities.on_jev_failure == "reject"


def test_an_unknown_jev_failure_mode_is_refused():
    with pytest.raises(ConfigError, match=r"on_jev_failure must be one of \['fallback', 'reject'\], got 'retry'"):
        parse_config(raw_config(on_jev_failure="retry"))


# ---------------------------------------------------------------- choosing
def test_an_easy_task_goes_to_the_cheapest_tier_that_covers_it():
    ask = Ask(EASY)
    d = decide(router(ask))
    assert d.tier == "cheap"
    # Below Jev's "no" band: the task needs neither, and the free tier is certain.
    assert d.score == pytest.approx(1.0)
    assert d.model.startswith("capabilities:")
    # One call, every requirement asked.
    assert len(ask.calls) == 1 and set(ask.calls[0][1]) == set(REQUIREMENTS)


def test_a_hard_task_goes_to_the_subscription_while_its_quota_is_cheap():
    d = decide(router(Ask(HARD)))
    # top and sub both cover it at 0.914; the subscription weighs 5/25 against 15/75.
    assert d.tier == "sub"
    reason = json.loads(d.reason)
    assert reason["need"] == HARD
    assert reason["pick"][0] == "sub"
    # The cheaper tiers that did not cover it, with their estimated success.
    assert [row[0] for row in reason["passed_over"]] == ["mid", "cx", "cheap"]
    assert all(row[1] < 0.8 for row in reason["passed_over"])


def test_when_nothing_reaches_the_target_the_most_likely_tier_wins():
    d = decide(router(Ask(HARD), target=0.99))
    assert d.tier == "sub"  # ties with top on success; cheaper
    assert json.loads(d.reason)["rule"].startswith("none >=")


def test_an_explicit_tier_is_served_without_asking_jev():
    ask = Ask(HARD)
    d = decide(router(ask), model="cheap")
    assert (d.tier, d.reason) == ("cheap", "requested")
    assert ask.calls == []


def test_only_eligible_carded_tiers_are_offered():
    d = decide(router(Ask(HARD)), candidates=["cheap", "mid"])
    assert d.tier == "mid"


def test_a_jev_failure_routes_to_the_default_and_says_why():
    d = decide(router(Ask(error=RuntimeError("timeout"))))
    assert d.tier == "mid"
    reason = json.loads(d.reason)
    assert reason["rule"] == "fallback" and "timeout" in reason["why"]
    assert d.score is None


def test_a_jev_failure_under_fallback_says_which_mode_served_it():
    d = decide(router(Ask(error=RuntimeError("timeout"))))
    assert d.tier == "mid"
    assert json.loads(d.reason)["on_jev_failure"] == "fallback"
    assert d.detail["on_jev_failure"] == "fallback"


@pytest.mark.parametrize("error", [RuntimeError("timeout"), ValueError("jev gave no probability for 'code'")])
def test_a_jev_failure_under_reject_raises_instead_of_routing_blind(error):
    r = router(Ask(error=error), on_jev_failure="reject")
    with pytest.raises(JevUnavailable) as caught:
        decide(r)
    exc = caught.value
    assert str(error) in exc.why and type(error).__name__ in exc.why
    assert exc.router == r.fingerprint and exc.jev_ms is not None
    assert json.loads(exc.reason) == {"rule": "jev_failure", "on_jev_failure": "reject", "why": exc.why}


def test_reject_leaves_the_fallbacks_that_are_not_a_jev_failure_alone():
    ask = Ask(error=RuntimeError("down"))
    d = decide(router(ask, on_jev_failure="reject"), packet={"failed_tiers": ALL})
    assert json.loads(d.reason)["rule"] == "fallback" and not ask.calls


def test_the_failure_mode_changes_the_fingerprint_only_when_it_is_reject():
    assert router(Ask(), on_jev_failure="fallback").fingerprint == router(Ask()).fingerprint
    assert router(Ask(), on_jev_failure="reject").fingerprint != router(Ask()).fingerprint


def test_with_jev_down_under_reject_there_is_no_escalation():
    r = router(Ask(error=RuntimeError("down")), on_jev_failure="reject")
    assert asyncio.run(r.escalation(make_request(), "mid", ALL)) is None


def test_tiers_the_client_says_already_failed_are_not_offered_again():
    d = decide(router(Ask(HARD)), packet={"failed_tiers": ["sub"]})
    assert d.tier == "top"
    assert json.loads(d.reason)["skipped_failed"] == ["sub"]


# ---------------------------------------------------------------- quota
def test_a_subscription_costs_its_tokens_however_much_of_the_window_is_used():
    r = router(Ask(HARD))
    before = r.cost("sub", 1000)
    r.observe("sub", Usage(prompt_tokens=0, completion_tokens=1_000_000), 200)
    assert r.ledger.used("claude") == pytest.approx(25.0)  # shown, never weighed
    assert r.cost("sub", 1000) == before
    assert decide(r).tier == "sub"


def test_a_429_locks_the_subscription_for_its_window():
    now = [0.0]
    r = router(Ask(HARD), clock=lambda: now[0])
    r.observe("sub", Usage(), 429)
    assert r.state()["subscriptions"]["claude"]["locked"] is True
    assert decide(r).tier == "top"
    now[0] += 5 * 3600 + 1
    assert decide(r).tier == "sub"


def test_api_and_free_tiers_are_never_charged_to_a_subscription():
    r = router(Ask(HARD))
    r.observe("top", Usage(prompt_tokens=1000, completion_tokens=1000), 200)
    r.observe("cheap", Usage(prompt_tokens=1000, completion_tokens=1000), 200)
    assert r.ledger.used("claude") == 0.0


def test_every_subscription_spent_falls_back_to_the_default():
    r = router(Ask(HARD))
    r.observe("sub", Usage(), 429)
    r.observe("cx", Usage(), 429)
    d = decide(r, candidates=["sub", "cx", "mid"])
    assert d.tier == "mid"


def test_an_injected_call_router_is_the_one_the_router_prices_with():
    config = parse_config(raw_config())
    caps = config.router.capabilities
    calls = CallRouter(config, caps, list_prices={n: c.list_prices for n, c in caps.cards.items()})
    r = CapabilityRouter(config, ask=Ask(HARD), calls=calls)
    calls.record_spend("sub", Usage(), 429, Prices())
    assert math.isinf(r.cost("sub", 1000))
    assert r.ledger is calls.ledger


# ---------------------------------------------------------------- the packet
def test_the_packet_leads_with_what_the_client_supplied_and_measures_the_rest():
    request = make_request(
        messages=[
            {"role": "system", "content": "You are a careful engineer."},
            {"role": "user", "content": "Fix the bug in core/src/http.rs and jev_model_router/app.py\n```rust\nfn x() {}\n```"},
        ],
        packet={"kind": "debug", "size": "small", "failed_tiers": ["cx"]},
        tools=[{"type": "function", "function": {"name": "bash", "parameters": {}}}],
    )
    packet = build_packet(request, max_chars=6000)
    assert packet.startswith("TASK PACKET\nkind: debug\nsize: small\n")
    assert "failed_tiers" not in packet
    assert "files_mentioned: 2 (core/src/http.rs, jev_model_router/app.py)" in packet
    assert "code_blocks: 1 (rust)" in packet
    assert "tools_offered: bash" in packet
    assert "instructions:\nYou are a careful engineer." in packet
    assert failed_tiers(request) == {"cx"}
    # The router's input is not forwarded to the model that answers.
    assert "packet" not in request.forwardable()


def test_a_long_task_keeps_its_head_and_tail_within_the_budget():
    task = "GOAL " + "x" * 20_000 + " CONSTRAINT"
    packet = build_packet(make_request(messages=[{"role": "user", "content": task}]), max_chars=2000)
    assert len(packet) <= 2000
    assert "GOAL" in packet and "CONSTRAINT" in packet and "[...]" in packet


# ---------------------------------------------------------------- through the app
def test_the_app_awaits_the_decision_logs_it_and_charges_the_subscription(backend_factory, fake_backends):
    config = parse_config(raw_config())
    log = RequestLog(":memory:")
    r = CapabilityRouter(config, ask=Ask(HARD))
    app = create_app(config, backend_factory=backend_factory, log=log, router=r)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "refactor it"}], "packet": {"size": "large"}},
        )
        assert response.status_code == 200
        assert response.headers["X-Router-Tier"] == "sub"
        assert float(response.headers["X-Router-Score"]) == pytest.approx(0.914, abs=1e-3)
        # The packet stayed with the router.
        assert "packet" not in fake_backends["sub"].calls[0].forwardable()
        health = client.get("/healthz").json()
        row = log.query("SELECT route_reason, route_model FROM requests")[0]
    assert health["router"] == "capabilities"
    # The fake answered 100 in / 50 out: charged at sub's list prices.
    assert health["routing"]["subscriptions"]["claude"]["used_usd"] == pytest.approx((100 * 5 + 50 * 25) / 1e6, abs=1e-4)
    assert json.loads(row["route_reason"])["pick"][0] == "sub"
    assert row["route_model"] == r.fingerprint


@pytest.mark.parametrize("stream", [False, True])
def test_under_reject_a_jev_failure_is_a_logged_503_and_nothing_runs(backend_factory, fake_backends, stream):
    config = parse_config(raw_config(on_jev_failure="reject"))
    log = RequestLog(":memory:")
    r = CapabilityRouter(config, ask=Ask(error=RuntimeError("jev timed out")))
    app = create_app(config, backend_factory=backend_factory, log=log, router=r)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "auto", "stream": stream, "messages": [{"role": "user", "content": "refactor it"}]})
        row = log.query("SELECT tier, http_status, error, route_model, route_reason FROM requests")[0]
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["type"] == error["code"] == "jev_unavailable"
    assert "on_jev_failure is reject" in error["message"] and "jev timed out" in error["message"]
    assert all(not b.calls for b in fake_backends.values())
    assert row["tier"] is None and row["http_status"] == 503 and "jev timed out" in row["error"]
    assert row["route_model"] == r.fingerprint
    assert json.loads(row["route_reason"]) == {"rule": "jev_failure", "on_jev_failure": "reject",
                                               "why": "jev error: RuntimeError: jev timed out"}


def test_a_free_tier_costs_its_shadow_price_and_a_billed_tier_its_real_one():
    r = router(Ask(EASY), cards={**CAPS["cards"], "cheap": {"levels": {"reasoning": 1, "code": 1},
                                                             "list_prices": {"input": 1.0, "output": 1.0}}})
    assert r.cost("cheap", 1000) == pytest.approx((1000 + 1500) / 1e6)
    assert r.cost("top", 0) == pytest.approx(1500 * 75 / 1e6)


def test_a_backends_own_prompt_is_part_of_what_a_call_costs():
    cards = {**CAPS["cards"], "cx": {**CAPS["cards"]["cx"], "input_overhead": 4000}}
    r = router(Ask(EASY), cards=cards)
    assert r.cost("cx", 1000) == pytest.approx(((1000 + 4000) * 2.0 + 1500 * 10.0) / 1e6)


def test_success_can_be_asked_about_levels_other_than_the_cards():
    r = router(Ask())
    needs = {"reasoning": 0.9, "code": 0.9}
    assert r.success(needs, "mid", levels={"reasoning": 3, "code": 3}) == pytest.approx(r.success(needs, "top"))
    assert r._caps.cards["mid"].levels == {"reasoning": 2, "code": 2}  # the card is untouched


def test_a_family_scale_fitted_for_a_hand_written_card_is_applied():
    # A hand-written card has no family; calibrate --from-log keys its scale by the tier name.
    needs = {"reasoning": 0.9, "code": 0.9}
    fitted = router(Ask(), family_scales={"mid": 0.1})
    assert fitted.success(needs, "mid") == pytest.approx(fitted.success(needs, "mid", scale=0.1))
    assert fitted.success(needs, "top") == pytest.approx(fitted.success(needs, "top", scale=1.0))


def test_a_tier_is_priced_at_what_it_bills_or_else_at_its_cards_list_price():
    r = router(Ask())
    assert (r.prices("mid").input, r.prices("mid").output) == (3.0, 15.0)
    assert (r.prices("sub").input, r.prices("sub").output) == (5.0, 25.0)


def test_a_cap_lowers_a_hand_written_card_and_changes_the_fingerprint():
    capped_router = router(Ask(), level_caps={"mid": {"reasoning": 1.0}})
    assert capped_router._caps.cards["mid"].levels == {"reasoning": 1.0, "code": 2}
    assert capped_router.fingerprint != router(Ask()).fingerprint


@pytest.mark.parametrize(
    "caps, message",
    [
        ("mid", "must map a family"),
        ({"mid": {"speed": 1.0}}, "unknown requirements"),
        ({"mid": {"reasoning": 4}}, "between 0 and 3"),
        ({"*": {"reasoning": 1.0}}, "fallback profile"),
        ({"mid": {"reasoning": True}}, "numbers"),
    ],
    ids=["shape", "requirement", "range", "fallback", "boolean"],
)
def test_level_caps_that_would_mislead_are_refused(caps, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(raw_config(level_caps=caps))


def with_tools(**caps_overrides: Any) -> dict[str, Any]:
    """raw_config with cx and sub taking tools, like mid and top, so dominance is decided by the cards alone."""
    raw = raw_config(**caps_overrides)
    for name in ("cx", "sub"):
        raw["tiers"][name]["supports_tools"] = True
    return raw


def test_state_reports_the_dominated_tiers():
    assert CapabilityRouter(parse_config(with_tools()), ask=Ask()).state()["dominated"] == {"mid": ["cx"], "top": ["sub"]}
    # As configured, cx and sub take no tools: a request with tools can go to mid and top only.
    assert router(Ask()).state()["dominated"] == {}


def test_healthz_reports_dominance_without_the_catalog_check(backend_factory, capsys):
    config = parse_config(with_tools())
    app = create_app(config, backend_factory=backend_factory, log=RequestLog(":memory:"),
                     router=CapabilityRouter(config, ask=Ask()))
    with TestClient(app) as client:
        assert client.get("/healthz").json()["routing"]["dominated"] == {"mid": ["cx"], "top": ["sub"]}
    assert "dominance: 2 of 5 carded tier(s) are never the cheapest adequate choice" in capsys.readouterr().err


def test_a_tier_the_catalog_finds_unavailable_dominates_nothing_in_healthz(backend_factory):
    from types import SimpleNamespace

    raw = with_tools()
    raw["catalog"] = {"check_on_start": True, "path": None}
    config = parse_config(raw)
    report = SimpleNamespace(unavailable={"cx": "model not served"}, discovered=None, checked_at="now", unconfigured=[])
    app = create_app(config, backend_factory=backend_factory, log=RequestLog(":memory:"),
                     router=CapabilityRouter(config, ask=Ask()), catalog_check=lambda config: report)
    with TestClient(app) as client:
        assert client.get("/healthz").json()["routing"]["dominated"] == {"top": ["sub"]}


# ---------------------------------------------------------------- a whole task
# Two tiers that both cover a hard task: one dearer per token but reading its
# cache at 2.5% of input, one cheaper per token reading it at 10%. Invented names;
# the prices are the shape of two real listings measured on 2026-09-26.
CACHEY = {"input": 10.0, "output": 50.0, "cache_read": 0.25, "cache_write": 12.5}
PLAIN = {"input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_write": 6.25}
CACHE_HEAVY = {"input_per_output": 295, "cache_read": 0.99, "cache_write": 0.005}
POOLED = {"input_per_output": 295, "cache_read": 0.971, "cache_write": 0.026}
PAIR = ["sub", "cx", "top"]


def cache_router(**caps):
    cards = copy.deepcopy(CAPS["cards"])
    cards["sub"]["list_prices"] = CACHEY
    cards["cx"] = {"levels": {"reasoning": 3, "code": 3}, "list_prices": PLAIN}
    return router(Ask(HARD), cards=cards, **caps)


def decide_task(r, candidates=PAIR, task=True):
    return asyncio.run(r.decide(make_request(), list(candidates), task=task))


def test_a_task_goes_to_the_tier_whose_cache_makes_it_cheaper():
    r = cache_router(task_shape=CACHE_HEAVY)
    one_call = decide_task(r, task=False)
    assert one_call.tier == "cx"  # one call: cheaper per token wins
    assert "cost_basis" not in json.loads(one_call.reason)  # one-call basis: not named in the log
    d = decide_task(r)
    assert d.tier == "sub" and d.detail["cost_basis"] == "task_shape"
    reason = json.loads(d.reason)
    assert reason["cost_basis"] == "task_shape"  # so the logged pick/passed_over costs are traceable to their basis
    assert reason["shape_flipped"] is True and reason["shapeless_pick"] == "cx"
    assert d.detail["shapeless_pick"] == "cx"


def test_a_shape_too_weak_to_reorder_leaves_the_pick():
    d = decide_task(cache_router(task_shape=POOLED))
    assert d.tier == "cx" and d.detail["cost_basis"] == "task_shape"
    assert "shape_flipped" not in json.loads(d.reason)


def test_without_task_the_price_is_todays():
    shaped, plain = cache_router(task_shape=CACHE_HEAVY), cache_router()
    a, b = decide_task(shaped, task=False), decide_task(plain, task=False)
    assert a.tier == b.tier and a.detail["options"] == b.detail["options"]
    assert a.detail["cost_basis"] == "one_call"


def test_expected_cost_picks_the_same_tier_when_every_cost_is_scaled_alike():
    r = router(Ask(HARD), rule="expected_cost")
    live = [Option("cheap", 0.4, 0.001), Option("mid", 0.75, 0.01), Option("top", 0.95, 0.05)]
    request = make_request()
    first = r._pick(live, HARD, request)[0].tier
    assert r._pick([replace(o, cost=o.cost * 37) for o in live], HARD, request)[0].tier == first


def test_one_observed_tier_without_a_config_shape_prices_the_whole_decision_as_one_call():
    r = cache_router()
    for _ in range(5):
        r.calls.record_task("sub", Usage(prompt_tokens=300_000, completion_tokens=1000, cached_tokens=299_000))
    d = decide_task(r)
    assert d.detail["cost_basis"] == "one_call" and d.tier == "cx"


# ---------------------------------------------------------------- unsure floor
# Jev read `reasoning` at 0.5: it cannot tell. The cheap card (level 1) still
# reaches the target on it -- 1 - 0.375 * 0.5 = 0.8125 -- which is the gap the floor closes.
UNSURE = {"reasoning": 0.5, "code": 0.05}


def test_the_unsure_floor_is_off_unless_configured():
    assert parse_config(raw_config()).router.capabilities.unsure is None


def test_an_empty_unsure_block_turns_it_on_with_the_defaults():
    floor = parse_config(raw_config(unsure={})).router.capabilities.unsure
    assert floor.band == (0.30, 0.70) and floor.min_level == 2.0


def test_the_unsure_floor_takes_its_band_and_level():
    floor = parse_config(raw_config(unsure={"band": [0.4, 0.6], "min_level": 3})).router.capabilities.unsure
    assert floor.band == (0.4, 0.6) and floor.min_level == 3.0


@pytest.mark.parametrize(
    "unsure, message",
    [
        ({"band": [0.3, 0.7], "level": 2}, "takes only"),
        ("yes", "takes only"),
        ({"band": [0.7, 0.3]}, "low < high"),
        ({"band": [0.3, 1.2]}, "low < high"),
        ({"band": [0.3]}, "two numbers"),
        ({"band": [True, 0.7]}, "two numbers"),
        ({"min_level": 4}, "0 to 3"),
        ({"min_level": True}, "0 to 3"),
        ({"min_level": "high"}, "0 to 3"),
        ({"min_level": -1}, "0 to 3"),
        ({"band": 0.5}, "two numbers"),
    ],
    ids=["unknown-key", "not-a-mapping", "band-order", "band-range", "band-length", "band-bool", "level-range",
         "level-bool", "level-text", "level-negative", "band-scalar"],
)
def test_an_unsure_floor_that_would_mislead_is_refused(unsure, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(raw_config(unsure=unsure))


def test_the_unsure_floor_changes_the_fingerprint_only_when_set():
    base = router(Ask()).fingerprint
    assert router(Ask()).fingerprint == base
    assert router(Ask(), unsure={}).fingerprint != base
    assert router(Ask(), unsure={}).fingerprint != router(Ask(), unsure={"min_level": 3}).fingerprint


def test_without_the_floor_an_unsure_reading_goes_to_the_cheapest_tier():
    d = decide(router(Ask(UNSURE)))
    assert d.tier == "cheap"
    assert "unsure" not in json.loads(d.reason) and d.detail["unsure"] is None


def test_an_unsure_requirement_raises_the_pick_to_a_card_at_the_level_and_logs_the_cost():
    r = router(Ask(UNSURE), unsure={})
    d = decide(r)
    assert d.tier == "cx"  # the cheapest card at level 2 on reasoning
    tokens = d.detail["prompt_tokens"]
    u = json.loads(d.reason)["unsure"]
    assert u["reqs"] == ["reasoning"] and u["band"] == [0.3, 0.7] and u["min_level"] == 2.0
    assert u["raised_from"][0] == "cheap"
    assert u["extra"] == pytest.approx(round(r.cost("cx", tokens) - r.cost("cheap", tokens), 6))
    assert d.detail["unsure"] == u
    # The tier the floor passed over is still named as passed over.
    assert "cheap" in [row[0] for row in json.loads(d.reason)["passed_over"]]


def test_the_band_is_inclusive_and_a_confident_reading_is_not_unsure():
    d = decide(router(Ask({"reasoning": 0.7, "code": 0.71}), unsure={}))
    assert json.loads(d.reason)["unsure"]["reqs"] == ["reasoning"]
    sure = decide(router(Ask(EASY), unsure={}))
    assert sure.tier == "cheap" and "unsure" not in json.loads(sure.reason)


def test_a_card_already_at_the_level_is_not_raised():
    d = decide(router(Ask(UNSURE), unsure={"min_level": 1}))
    assert d.tier == "cheap"
    u = json.loads(d.reason)["unsure"]
    assert u == {"reqs": ["reasoning"], "band": [0.3, 0.7], "min_level": 1.0}


def test_when_no_tier_reaches_the_level_the_pick_stands_and_says_so():
    d = decide(router(Ask(UNSURE), unsure={"min_level": 3}), candidates=["cheap", "mid"])
    assert d.tier == "cheap"
    u = json.loads(d.reason)["unsure"]
    assert u["unmet"] is True and "raised_from" not in u


def test_the_floor_composes_with_a_failed_tier():
    # cx failed: the bar is its 0.925, which mid meets and cheap does not. A floor at 3
    # then leaves top and sub, and sub is cheaper.
    d = decide(router(Ask(UNSURE), unsure={"min_level": 3}), packet={"failed_tiers": ["cx"]})
    assert d.tier == "sub"
    reason = json.loads(d.reason)
    assert reason["skipped_failed"] == ["cx"] and reason["unsure"]["raised_from"][0] == "mid"


def test_under_expected_cost_the_redo_tier_also_clears_the_floor():
    r = router(Ask(UNSURE), rule="expected_cost", unsure={})
    d = decide(r)
    cards = r._caps.cards
    assert cards[d.tier].levels["reasoning"] >= 2
    assert cards[d.detail["redo_tier"]].levels["reasoning"] >= 2


def _with_flo(**caps_overrides: Any) -> Any:
    """raw_config with an extra tier 'flo', a level-1 card, for a fitted-scale scenario."""
    raw = raw_config(cards={**CAPS["cards"], "flo": {"levels": {"reasoning": 1, "code": 1},
                                                       "list_prices": {"input": 1.0, "output": 5.0}}},
                      **caps_overrides)
    raw["tiers"]["flo"] = {"backend": "openai_compatible", "model": "flo-model",
                            "base_url": "https://example.invalid/v1", "context_window": 200000,
                            "supports_tools": True}
    return raw


def test_the_floor_reads_a_fitted_scales_calibrated_miss_not_the_raw_level():
    # flo's card is level 1 on reasoning, but its fitted family scale (0.3) makes
    # its effective miss (0.3 * 0.5 = 0.15) as good as the level-2 bar (1.0 * 0.2).
    r = CapabilityRouter(parse_config(_with_flo(family_scales={"flo": 0.3}, unsure={})), ask=Ask(UNSURE))
    options = [Option("cheap", 0.0, 0.0), Option("mid", 0.0, 0.0), Option("flo", 0.0, 0.0)]
    floored = {o.tier for o in r._floored(options, ["reasoning"])}
    assert floored == {"mid", "flo"}  # cheap (scale 1.0) still misses the bar; flo, calibrated, clears it


def test_the_floor_never_drops_a_tier_that_dominates_one_it_keeps():
    config = parse_config(_with_flo(family_scales={"flo": 0.3}, unsure={}))
    r = CapabilityRouter(config, ask=Ask(UNSURE))
    # flo's calibrated miss beats mid's on every requirement at no more cost: it dominates mid.
    assert "flo" in r.dominated.get("mid", [])
    d = decide(r, candidates=["cheap", "mid", "flo"])
    # The floor must not exclude flo (a dominator) while mid (what it dominates) stays eligible.
    assert d.tier == "flo"


def test_a_shape_flip_is_measured_on_the_floored_tiers():
    cards = copy.deepcopy(CAPS["cards"])
    cards["sub"]["list_prices"] = CACHEY
    cards["cx"] = {"levels": {"reasoning": 3, "code": 3}, "list_prices": PLAIN}
    r = router(Ask(UNSURE), cards=cards, task_shape=CACHE_HEAVY, unsure={})
    d = asyncio.run(r.decide(make_request(), ["cheap", "sub", "cx"], task=True))
    assert d.detail["cost_basis"] == "task_shape"
    reason = json.loads(d.reason)
    # Unfloored, cheap wins both ways; floored, sub on the task and cx on one call.
    assert d.tier == "sub" and reason["shapeless_pick"] == "cx"
    tokens = d.detail["prompt_tokens"]
    assert reason["unsure"]["raised_from"][0] == "cheap"
    assert reason["unsure"]["extra"] == pytest.approx(
        round(r.task_cost("sub", tokens) - r.task_cost("cheap", tokens), 6))
