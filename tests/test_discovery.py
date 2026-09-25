"""Discovery: every served model, at every effort, becomes a tier with a card.

The catalog report is built by hand here; `test_catalog.py` covers reading it.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any

import pytest
from fastapi.testclient import TestClient

from llm_router.app import create_app
from llm_router.capabilities import CapabilityRouter
from llm_router.catalog import CatalogReport, Offered, Source
from llm_router.config import ConfigError, parse_config
from llm_router.db import RequestLog
from llm_router.discovery import expand

from conftest import BASE_CONFIG, make_request

REQUIREMENTS = {
    "reasoning": "The state is a task packet. Does it need deep reasoning?",
    "niche": "The state is a task packet. Does it need niche knowledge?",
}
EFFORTS = ("low", "medium", "high", "xhigh", "max")


def raw_config(**caps: Any) -> dict[str, Any]:
    raw = copy.deepcopy(BASE_CONFIG)
    raw["tiers"]["judge"] = {"backend": "jev", "model": "jev-latest"}
    raw["router"] = {
        "kind": "capabilities",
        "default_tier": "mid",
        "capabilities": {
            "jev_tier": "judge",
            "requirements": REQUIREMENTS,
            "discover": {
                "claude": {"backend": "claude_cli", "base_url": "C:/bin/claude.exe", "context_window": 1_000_000},
                "codex": {"backend": "codex_cli", "base_url": "C:/bin/codex.exe", "input_overhead": 4500},
                "local": {"backend": "ollama", "base_url": "http://localhost:11434", "context_window": 16384},
            },
            "profiles": [
                {"match": "claude-opus-*", "level": 2.5, "list_prices": {"input": 5.0, "output": 25.0}},
                {"match": "claude-haiku-4-5*", "level": 1.3, "list_prices": {"input": 1.0, "output": 5.0}},
                {"match": "gpt-*-luna", "level": 1.5, "levels": {"niche": 1.0}, "list_prices": {"input": 0.2, "output": 1.2}},
                {"match": "qwen*", "level": 0.6, "list_prices": {"input": 0.1, "output": 0.4}, "output_tokens": 600},
            ],
            "fallback_profile": {"level": 1.0, "list_prices": {"input": 10.0, "output": 50.0}},
            **caps,
        },
    }
    return raw


def served(**models: Offered) -> Source:
    return Source(name="x", models=dict(models), cli_version="2.1.280")


def report(**discovered: Source) -> CatalogReport:
    return CatalogReport(checked_at="now", tiers={}, sources={}, unconfigured={}, discovered=discovered)


REPORT = report(
    claude=served(**{
        "claude-opus-5-5": Offered("claude-opus-5-5", efforts=EFFORTS, min_cli="2.1.280"),
        "claude-opus-9": Offered("claude-opus-9", efforts=("high",), min_cli="3.0.0"),
        "claude-haiku-4-5-20251001": Offered("claude-haiku-4-5-20251001", efforts=(), aliases=("claude-haiku-4-5",)),
    }),
    codex=served(**{
        "gpt-5.6-luna": Offered("gpt-5.6-luna", efforts=("low", "high")),
        "gpt-7-nova": Offered("gpt-7-nova", efforts=("medium",)),
        "codex-auto-review": Offered("codex-auto-review", efforts=("medium",), hidden=True),
    }),
    local=served(**{
        "qwen3.5:4b": Offered("qwen3.5:4b"),
        "nomic-embed-text:latest": Offered("nomic-embed-text:latest"),
    }),
)


def expanded(**caps: Any):
    return expand(parse_config(raw_config(**caps)), REPORT)


# ---------------------------------------------------------------- what becomes a tier
def test_every_served_model_at_every_effort_is_a_tier():
    config, found, unprofiled = expanded()
    names = {f.tier for f in found}
    assert {f"claude:claude-opus-5-5@{e}" for e in EFFORTS} <= names
    # A model whose catalog lists no effort is one tier at its own default.
    assert "claude:claude-haiku-4-5-20251001" in names
    assert {"codex:gpt-5.6-luna@low", "codex:gpt-5.6-luna@high", "codex:gpt-7-nova@medium", "local:qwen3.5:4b"} <= names
    tier = config.tiers["codex:gpt-5.6-luna@high"]
    assert (tier.backend, tier.model, tier.effort, tier.subscription) == ("codex_cli", "gpt-5.6-luna", "high", "codex")
    assert tier.base_url == "C:/bin/codex.exe"
    assert config.tiers["local:qwen3.5:4b"].subscription is None


def test_only_what_cannot_answer_a_chat_is_left_out():
    _, found, _ = expanded()
    names = {f.tier for f in found}
    assert not any("nomic-embed" in n for n in names)  # an embedding model
    assert not any("codex-auto-review" in n for n in names)  # a hidden utility
    assert not any("claude-opus-9" in n for n in names)  # needs a newer CLI than the one installed


def test_a_model_no_profile_matches_is_offered_on_the_fallback_and_reported():
    config, found, unprofiled = expanded()
    assert unprofiled == ["codex:gpt-7-nova"]
    card = config.router.capabilities.cards["codex:gpt-7-nova@medium"]
    assert card.list_prices.output == 50.0


def test_effort_moves_the_thinking_levels_and_the_output_not_the_knowledge():
    config, _, _ = expanded()
    cards = config.router.capabilities.cards
    low, high, top = (cards[f"claude:claude-opus-5-5@{e}"] for e in ("low", "high", "max"))
    assert high.levels == {"reasoning": 2.5, "niche": 2.5}
    assert low.levels["reasoning"] == pytest.approx(1.75) and low.levels["niche"] == 2.5
    assert top.levels["reasoning"] == pytest.approx(2.9)
    assert (low.output_tokens, high.output_tokens, top.output_tokens) == (400, 2000, 5000)
    # The CLI's own prompt rides on every Codex call.
    assert cards["codex:gpt-5.6-luna@low"].input_overhead == 4500


def test_an_explicit_tier_is_not_added_twice_and_gets_its_card_from_the_profile():
    raw = raw_config()
    raw["tiers"]["opus-high"] = {"backend": "claude_cli", "model": "claude-opus-5-5", "effort": "high",
                                 "base_url": "C:/bin/claude.exe"}
    config, found, _ = expand(parse_config(raw), REPORT)
    assert "claude:claude-opus-5-5@high" not in {f.tier for f in found}
    assert config.router.capabilities.cards["opus-high"].levels == {"reasoning": 2.5, "niche": 2.5}


def test_the_router_picks_among_discovered_tiers():
    config, _, _ = expanded()

    async def ask(packet, questions):
        return {"reasoning": 0.05, "niche": 0.05}

    router = CapabilityRouter(config, ask=ask)
    candidates = [n for n, t in config.tiers.items() if t.can_serve]
    decision = asyncio.run(router.decide(make_request(), candidates))
    assert decision.tier == "local:qwen3.5:4b"


def test_levels_between_the_table_points_are_interpolated():
    config, _, _ = expanded(miss=[0.9, 0.5, 0.2, 0.0])
    router = CapabilityRouter(config, ask=None)
    assert router._miss(2.5) == pytest.approx(0.1)
    assert router._miss(0.25) == pytest.approx(0.8)
    assert router._miss(3.0) == 0.0


# ---------------------------------------------------------------- config
@pytest.mark.parametrize(
    "change, message",
    [
        ({"fallback_profile": None}, "fallback_profile"),
        ({"discover": {"x": {"backend": "openai_compatible"}}}, "claude_cli, codex_cli or ollama"),
        ({"discover": {"a:b": {"backend": "ollama"}}}, "cannot contain ':'"),
        ({"profiles": [{"match": "gpt-*", "level": 2}]}, "list_prices is required"),
        ({"profiles": [{"match": "gpt-*", "level": 4, "list_prices": {"input": 1}}]}, "between 0 and 3"),
        ({"effort_rules": {"turbo": {"shift": 1}}}, "unknown effort"),
        ({"thinking": ["speed"]}, "unknown requirements"),
    ],
    ids=["no-fallback", "backend", "name", "no-prices", "level", "effort", "thinking"],
)
def test_discovery_settings_that_would_mislead_are_refused(change, message):
    raw = raw_config(**change)
    if change.get("fallback_profile", 1) is None:
        del raw["router"]["capabilities"]["fallback_profile"]
    with pytest.raises(ConfigError, match=message):
        parse_config(raw)


# ---------------------------------------------------------------- through the app
def test_at_startup_the_discovered_tiers_are_served_and_listed(backend_factory, fake_backends):
    raw = raw_config()
    raw["catalog"] = {"check_on_start": True, "path": None}
    app = create_app(parse_config(raw), backend_factory=backend_factory, log=RequestLog(":memory:"),
                     catalog_check=lambda config: REPORT)
    with TestClient(app) as client:
        listed = {m["id"] for m in client.get("/v1/models").json()["data"]}
        health = client.get("/healthz").json()
        # A client may still ask for one by name.
        response = client.post("/v1/chat/completions", json={
            "model": "codex:gpt-5.6-luna@high", "messages": [{"role": "user", "content": "hi"}]})
    assert "claude:claude-opus-5-5@xhigh" in listed and "local:qwen3.5:4b" in listed
    assert health["catalog"]["discovered_tiers"] == len({t for t in listed if ":" in t})
    assert health["catalog"]["unprofiled"] == ["codex:gpt-7-nova"]
    assert response.headers["X-Router-Tier"] == "codex:gpt-5.6-luna@high"
    assert len(fake_backends["codex:gpt-5.6-luna@high"].calls) == 1


def test_an_effort_ceiling_keeps_the_levels_above_it_out_of_the_catalog():
    config, found, _ = expanded(max_effort="high")
    names = {f.tier for f in found}
    assert {f"claude:claude-opus-5-5@{e}" for e in ("low", "medium", "high")} <= names
    assert not any(n.endswith(("@xhigh", "@max")) for n in names)
    # A model with no effort list is untouched by the ceiling.
    assert "claude:claude-haiku-4-5-20251001" in names
    with pytest.raises(ConfigError, match="max_effort"):
        parse_config(raw_config(max_effort="turbo"))


from llm_router.discovery import unused_caps


def test_a_cap_lowers_the_final_level_at_every_effort_and_never_raises_one():
    config, _, _ = expanded(level_caps={"claude-opus-*": {"reasoning": 2.0, "niche": 3.0}})
    cards = config.router.capabilities.cards
    assert cards["claude:claude-opus-5-5@high"].levels["reasoning"] == 2.0   # 2.5 capped
    assert cards["claude:claude-opus-5-5@max"].levels["reasoning"] == 2.0    # 2.9 capped
    assert cards["claude:claude-opus-5-5@low"].levels["reasoning"] == pytest.approx(1.75)  # already below
    assert cards["claude:claude-opus-5-5@high"].levels["niche"] == 2.5       # a cap of 3 raises nothing


def test_a_cap_whose_family_has_no_card_is_reported_not_refused():
    config, _, _ = expanded(level_caps={"retired-model": {"reasoning": 1.0}})
    assert unused_caps(config) == ["retired-model"]


def test_with_the_catalog_check_healthz_reports_dominance_and_startup_names_unused_caps(backend_factory, capsys):
    raw = raw_config(level_caps={"retired-model": {"reasoning": 1.0}})
    raw["catalog"] = {"check_on_start": True, "path": None}
    app = create_app(parse_config(raw), backend_factory=backend_factory, log=RequestLog(":memory:"),
                     catalog_check=lambda config: REPORT)
    with TestClient(app) as client:
        dominated = client.get("/healthz").json()["routing"]["dominated"]
    assert isinstance(dominated, dict) and dominated  # 30-odd discovered tiers: some are always beaten
    err = capsys.readouterr().err
    assert "level_caps for no card" in err and "retired-model" in err
