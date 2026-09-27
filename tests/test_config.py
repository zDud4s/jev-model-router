"""Config parsing: the model catalog lives here and nowhere else."""

from __future__ import annotations

import pytest
import yaml

from jev_model_router.config import ConfigError, Prices, TaskShape, load_config, parse_config

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

    assert set(config.tiers) == {"cheap", "mid", "top", "judge"}
    assert config.router.default_tier == "cheap"
    assert config.log.store_prompts is False
    # The example ships a judge that cannot serve, so it is also the test that a
    # catalog may carry one without the router ever trying to answer from it.
    assert config.tier("judge").can_serve is False
    assert all(config.tier(name).can_serve for name in ("cheap", "mid", "top"))


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


def test_cache_write_is_marked_configured_only_when_written():
    assert Prices.parse({"input": 1.0, "cache_write": 1.25}).cache_write_configured is True
    assert Prices.parse({"input": 1.0, "cache_write": 0}).cache_write_configured is True
    assert Prices.parse({"input": 1.0}).cache_write_configured is False
    assert Prices().cache_write_configured is False


def test_a_missing_config_file_is_refused(tmp_path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_yaml_round_trip(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(BASE_CONFIG), encoding="utf-8")

    assert set(load_config(path).tiers) == {"cheap", "mid", "top"}


def test_a_classifier_router_must_say_where_escalations_go() -> None:
    raw = {**BASE_CONFIG, "router": {"kind": "classifier", "default_tier": "cheap"}}
    with pytest.raises(ConfigError, match="strong_tier"):
        parse_config(raw)


def test_a_classifier_router_must_name_a_model_file() -> None:
    raw = {
        **BASE_CONFIG,
        "router": {"kind": "classifier", "default_tier": "cheap", "strong_tier": "top"},
    }
    with pytest.raises(ConfigError, match="model_path"):
        parse_config(raw)


def test_escalating_to_the_tier_you_came_from_is_refused() -> None:
    raw = {
        **BASE_CONFIG,
        "router": {
            "kind": "classifier",
            "default_tier": "cheap",
            "strong_tier": "cheap",
            "model_path": "m.json",
        },
    }
    with pytest.raises(ConfigError, match="nothing to escalate to"):
        parse_config(raw)


def test_an_unknown_router_kind_is_caught_at_load_time() -> None:
    with pytest.raises(ConfigError, match="router.kind"):
        parse_config({**BASE_CONFIG, "router": {"kind": "magic"}})


def test_the_sampling_knobs_must_be_fractions() -> None:
    for field, value in (("threshold", 1.5), ("explore_rate", -0.1)):
        with pytest.raises(ConfigError, match=field):
            parse_config({**BASE_CONFIG, "router": {"kind": "static", field: value}})


from test_capabilities import raw_config as caps_raw

SHAPE = {"input_per_output": 295, "cache_read": 0.971, "cache_write": 0.026}


def test_a_task_shape_is_read():
    caps = parse_config(caps_raw(task_shape=SHAPE)).router.capabilities
    assert caps.task_shape == TaskShape(295.0, 0.971, 0.026, "config")
    assert parse_config(caps_raw()).router.capabilities.task_shape is None


@pytest.mark.parametrize(
    "shape, message",
    [
        ({**SHAPE, "input_per_output": 0}, "input_per_output must be positive"),
        ({**SHAPE, "cache_read": 1.2}, r"fractions in \[0, 1\]"),
        ({**SHAPE, "cache_read": 0.9, "cache_write": 0.2}, "cannot exceed 1"),
        ({"input_per_output": 10, "cache_read": 0.9}, "needs numbers"),
        ({**SHAPE, "fresh": 0.1}, "takes only"),
    ],
    ids=["ratio", "range", "sum", "missing", "unknown"],
)
def test_a_task_shape_that_is_not_a_share_of_input_is_refused(shape, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(caps_raw(task_shape=shape))
