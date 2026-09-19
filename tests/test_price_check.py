"""The configured prices, checked against the provider's published list."""

from __future__ import annotations

from typing import Any

from llm_router.config import parse_config
from llm_router.price_check import check_prices, format_report, listed_prices

from conftest import BASE_CONFIG

OPENROUTER = "https://openrouter.ai/api/v1"

# Shapes copied from OpenRouter's /models on 2026-09-19: per token, as strings.
SONNET = {
    "prompt": "0.000003",
    "completion": "0.000015",
    "input_cache_read": "0.0000003",
    "input_cache_write": "0.00000375",
    "overrides": [{"min_prompt_tokens": 200000, "prompt": "0.000006"}],
}
MINI = {"prompt": "0.00000015", "completion": "0.0000006"}
FREE = {"prompt": "0", "completion": "0"}


def _config(**remote: dict[str, Any]):
    tiers = dict(BASE_CONFIG["tiers"])
    for name, fields in remote.items():
        tiers[name] = {
            "backend": "openai_compatible",
            "base_url": OPENROUTER,
            "context_window": 65536,
            **fields,
        }
    return parse_config({**BASE_CONFIG, "tiers": tiers})


def _catalog(models: dict[str, dict[str, Any]]):
    asked: list[str] = []

    def catalog(base_url: str) -> dict[str, dict[str, Any]]:
        asked.append(base_url)
        return models

    return catalog, asked


def _by_tier(report):
    return {t.tier: t for t in report.tiers}


def test_the_catalog_is_read_per_million_tokens() -> None:
    assert listed_prices(SONNET) == {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.3,
        "cache_write": 3.75,
    }


def test_a_model_with_no_cache_price_bills_cached_tokens_at_the_input_rate() -> None:
    # The same default `Prices.parse` applies, so a config that says nothing
    # about caching agrees with a provider that lists nothing.
    assert listed_prices(MINI)["cache_read"] == 0.15


def test_prices_that_match_the_list_pass() -> None:
    config = _config(remote={"model": "openai/gpt-4o-mini", "prices": {"input": 0.15, "output": 0.6}})
    catalog, _ = _catalog({"openai/gpt-4o-mini": MINI})

    report = check_prices(config, catalog=catalog)

    assert _by_tier(report)["remote"].status == "ok"
    assert report.ok


def test_a_stale_price_is_named_with_both_numbers() -> None:
    config = _config(
        remote={
            "model": "anthropic/claude-sonnet-4.5",
            "prices": {"input": 3.0, "output": 12.0, "cache_read": 0.3, "cache_write": 3.75},
        }
    )
    catalog, _ = _catalog({"anthropic/claude-sonnet-4.5": SONNET})

    report = check_prices(config, catalog=catalog)
    remote = _by_tier(report)["remote"]

    assert remote.status == "mismatch"
    assert remote.differences == {"output": (12.0, 15.0)}
    assert not report.ok
    assert "output: config 12, provider 15" in format_report(report)


def test_an_unstated_cache_write_price_is_caught() -> None:
    # Omitted cache_write defaults to 0, and the provider charges for it.
    config = _config(
        remote={
            "model": "anthropic/claude-sonnet-4.5",
            "prices": {"input": 3.0, "output": 15.0, "cache_read": 0.3},
        }
    )
    catalog, _ = _catalog({"anthropic/claude-sonnet-4.5": SONNET})

    remote = _by_tier(check_prices(config, catalog=catalog))["remote"]

    assert remote.differences == {"cache_write": (0.0, 3.75)}


def test_long_prompt_pricing_is_noted_because_the_config_cannot_express_it() -> None:
    config = _config(
        remote={
            "model": "anthropic/claude-sonnet-4.5",
            "prices": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
        }
    )
    catalog, _ = _catalog({"anthropic/claude-sonnet-4.5": SONNET})

    remote = _by_tier(check_prices(config, catalog=catalog))["remote"]

    assert remote.status == "ok"
    assert "200000" in remote.notes[0]


def test_a_charged_model_with_no_prices_is_flagged_not_free() -> None:
    config = _config(remote={"model": "openai/gpt-4o-mini"})
    catalog, _ = _catalog({"openai/gpt-4o-mini": MINI})

    report = check_prices(config, catalog=catalog)

    assert _by_tier(report)["remote"].status == "unpriced"
    assert not report.ok


def test_a_free_model_with_no_prices_is_right() -> None:
    config = _config(remote={"model": "openai/gpt-oss-120b:free"})
    catalog, _ = _catalog({"openai/gpt-oss-120b:free": FREE})

    assert _by_tier(check_prices(config, catalog=catalog))["remote"].status == "ok"


def test_a_misspelt_model_id_is_not_listed() -> None:
    config = _config(remote={"model": "openai/gpt-4o-mni", "prices": {"input": 0.15}})
    catalog, _ = _catalog({"openai/gpt-4o-mini": MINI})

    report = check_prices(config, catalog=catalog)

    assert _by_tier(report)["remote"].status == "not_listed"
    assert not report.ok


def test_other_providers_are_unchecked_and_the_list_is_fetched_once() -> None:
    config = _config(
        a={"model": "openai/gpt-4o-mini", "prices": {"input": 0.15, "output": 0.6}},
        b={"model": "openai/gpt-4o-mini", "prices": {"input": 0.15, "output": 0.6}},
    )
    catalog, asked = _catalog({"openai/gpt-4o-mini": MINI})

    report = check_prices(config, catalog=catalog)

    assert asked == [OPENROUTER]
    # cheap, mid and top are Ollama and a generic proxy: nothing to compare.
    assert {t.tier for t in report.tiers if t.status == "unchecked"} == {"cheap", "mid", "top"}
    assert report.ok


def test_an_unreachable_list_is_an_error_not_a_pass() -> None:
    config = _config(remote={"model": "openai/gpt-4o-mini", "prices": {"input": 0.15}})

    def down(base_url: str):
        raise ConnectionError("no route")

    report = check_prices(config, catalog=down)

    assert not report.ok
    assert "ConnectionError" in report.errors[0]
