"""Config parsing: the model catalog lives here and nowhere else."""

from __future__ import annotations

import pytest
import yaml

from llm_router.config import ConfigError, load_config, parse_config

from conftest import BASE_CONFIG


def test_a_tier_without_prices_parses_and_is_marked_unpriced(config) -> None:
    # The headline case: a local model needs no price configuration at all.
    assert config.tiers["cheap"].prices.configured is False
    assert config.tiers["cheap"].prices.input == 0.0


def test_ollama_gets_a_default_base_url() -> None:
    raw = {"tiers": {"local": {"backend": "ollama", "model": "llama3.1:8b"}}}

    assert parse_config(raw).tiers["local"].base_url == "http://localhost:11434"


def test_an_unknown_backend_kind_is_refused() -> None:
    raw = {"tiers": {"x": {"backend": "anthropic", "model": "claude-opus-5"}}}

    with pytest.raises(ConfigError, match="openai_compatible"):
        parse_config(raw)


def test_a_config_with_no_tiers_is_refused() -> None:
    with pytest.raises(ConfigError, match="at least one tier"):
        parse_config({"tiers": {}})


def test_a_default_tier_that_does_not_exist_is_refused() -> None:
    raw = {**BASE_CONFIG, "router": {"default_tier": "nonexistent"}}

    with pytest.raises(ConfigError, match="default_tier"):
        parse_config(raw)


def test_a_model_map_pointing_at_an_unknown_tier_is_refused() -> None:
    raw = {**BASE_CONFIG, "router": {"default_tier": "cheap", "model_map": {"gpt-4o": "nope"}}}

    with pytest.raises(ConfigError, match="unknown tier"):
        parse_config(raw)


def test_an_api_key_is_read_from_the_environment_not_the_file(monkeypatch) -> None:
    raw = {
        "tiers": {
            "remote": {
                "backend": "openai_compatible",
                "model": "m",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "TEST_ROUTER_KEY",
            }
        }
    }
    tier = parse_config(raw).tiers["remote"]

    assert tier.api_key is None
    monkeypatch.setenv("TEST_ROUTER_KEY", "secret-value")
    # Read at call time, so a config file can be committed without a credential.
    assert tier.api_key == "secret-value"


def test_the_shipped_example_config_is_valid(tmp_path) -> None:
    import pathlib

    example = pathlib.Path(__file__).resolve().parents[1] / "config.example.yaml"
    config = load_config(example)

    assert set(config.tiers) == {"cheap", "mid", "top"}
    assert config.router.default_tier == "cheap"
    assert config.log.store_prompts is False


def test_prices_reject_an_unknown_field() -> None:
    raw = {
        "tiers": {
            "x": {
                "backend": "ollama",
                "model": "m",
                "prices": {"input": 1.0, "per_request": 0.5},
            }
        }
    }

    with pytest.raises(ConfigError, match="unknown price fields"):
        parse_config(raw)


def test_a_missing_config_file_is_refused(tmp_path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_yaml_round_trip(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(BASE_CONFIG), encoding="utf-8")

    assert set(load_config(path).tiers) == {"cheap", "mid", "top"}
