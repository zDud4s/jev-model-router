"""The capabilities router: Jev's reading of the task, the cards, and the two currencies.

Jev is never called: every router here is given an `ask` that answers from a
script, and records the packet and questions it was shown.
"""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from llm_router.app import create_app
from llm_router.capabilities import CapabilityRouter, build_packet, failed_tiers
from llm_router.config import ConfigError, parse_config
from llm_router.db import RequestLog
from llm_router.routing import build_router
from llm_router.schemas import Usage

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
        ({"floor": 1.0}, "floor"),
    ],
    ids=["jev-tier", "stray-level", "level-range", "no-list-prices", "judge-card", "target", "miss", "floor"],
)
def test_a_card_or_setting_that_would_mislead_is_refused(change, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(raw_config(**change))


def test_a_capabilities_block_on_another_router_kind_is_refused():
    raw = raw_config()
    raw["router"]["kind"] = "static"
    with pytest.raises(ConfigError, match="does nothing"):
        parse_config(raw)


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


# ---------------------------------------------------------------- the packet
def test_the_packet_leads_with_what_the_client_supplied_and_measures_the_rest():
    request = make_request(
        messages=[
            {"role": "system", "content": "You are a careful engineer."},
            {"role": "user", "content": "Fix the bug in core/src/http.rs and llm_router/app.py\n```rust\nfn x() {}\n```"},
        ],
        packet={"kind": "debug", "size": "small", "failed_tiers": ["cx"]},
        tools=[{"type": "function", "function": {"name": "bash", "parameters": {}}}],
    )
    packet = build_packet(request, max_chars=6000)
    assert packet.startswith("TASK PACKET\nkind: debug\nsize: small\n")
    assert "failed_tiers" not in packet
    assert "files_mentioned: 2 (core/src/http.rs, llm_router/app.py)" in packet
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
