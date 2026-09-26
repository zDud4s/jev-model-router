"""API keys: the environment, then the user's secrets file; and a tier with several providers.

`conftest` points JEV_MODEL_ROUTER_HOME at a fresh directory for every test, so the
file read here is never the user's.
"""

from __future__ import annotations

import copy
import io
import json

import pytest
from fastapi.testclient import TestClient

from jev_model_router import cli, keystore
from jev_model_router.capabilities import CapabilityRouter
from jev_model_router.config import ConfigError, parse_config
from jev_model_router.db import RequestLog

from test_capabilities import Ask, raw_config

KEYS = ("TS_KEY", "OR_KEY", "TOP_KEY")


@pytest.fixture(autouse=True)
def no_keys_in_env(monkeypatch):
    for name in KEYS:
        monkeypatch.delenv(name, raising=False)


def judge_raw(**judge):
    """A config whose judge tier can be reached through two providers."""
    raw = {
        "log": {"path": ":memory:"},
        "catalog": {"check_on_start": False},
        "router": {"kind": "static", "default_tier": "cheap"},
        "tiers": {
            "cheap": {"backend": "ollama", "model": "small", "context_window": 8000},
            "judge": {
                "backend": "jev", "model": "judge-latest", "context_window": 32000,
                "endpoints": [
                    {"label": "First", "base_url": "https://first.example/v1", "api_key_env": "TS_KEY"},
                    {"label": "Second", "base_url": "https://second.example/api/v1/", "model": "vendor/judge",
                     "api_key_env": "OR_KEY", "key_url": "https://second.example/keys"},
                ],
                **judge,
            },
        },
    }
    return raw


# ---------------------------------------------------------------- keystore
def test_a_stored_key_is_found_and_the_environment_wins(monkeypatch):
    assert keystore.lookup("TS_KEY") is None and keystore.source("TS_KEY") is None
    keystore.store("TS_KEY", "  from-file \n")
    assert keystore.lookup("TS_KEY") == "from-file" and keystore.source("TS_KEY") == "file"
    monkeypatch.setenv("TS_KEY", "from-env")
    assert keystore.lookup("TS_KEY") == "from-env" and keystore.source("TS_KEY") == "env"


def test_storing_replaces_one_key_and_keeps_the_others():
    keystore.store("TS_KEY", "a")
    keystore.store("OR_KEY", "b")
    keystore.store("TS_KEY", "c")
    assert keystore.read() == {"OR_KEY": "b", "TS_KEY": "c"}
    assert keystore.remove("TS_KEY") and not keystore.remove("TS_KEY")
    assert keystore.read() == {"OR_KEY": "b"}


def test_the_file_lives_under_the_home_it_is_given(tmp_path, monkeypatch):
    monkeypatch.setenv(keystore.HOME_ENV, str(tmp_path / "home"))
    written = keystore.store("TS_KEY", "k")
    assert written == tmp_path / "home" / "secrets.env"
    assert "TS_KEY=k" in written.read_text(encoding="utf-8")


@pytest.mark.parametrize("name,value", [("1BAD", "k"), ("A-B", "k"), ("OK", ""), ("OK", "two\nlines")])
def test_a_bad_name_or_value_is_refused(name, value):
    with pytest.raises(ValueError):
        keystore.store(name, value)
    assert keystore.read() == {}


def test_a_tier_key_is_read_from_the_file_at_call_time():
    tier = parse_config(judge_raw()).tiers["judge"]
    assert tier.api_key is None
    keystore.store("TS_KEY", "late")
    assert tier.api_key == "late"


# ---------------------------------------------------------------- endpoints
def test_with_no_key_set_the_first_provider_is_picked_so_errors_name_its_key():
    tier = parse_config(judge_raw()).tiers["judge"]
    assert (tier.base_url, tier.model, tier.api_key_env) == ("https://first.example/v1", "judge-latest", "TS_KEY")
    assert [e.label for e in tier.endpoints] == ["First", "Second"]


def test_the_first_provider_with_a_key_serves_with_its_own_model_id(monkeypatch):
    monkeypatch.setenv("OR_KEY", "k")
    tier = parse_config(judge_raw()).tiers["judge"]
    assert (tier.base_url, tier.model, tier.api_key_env) == ("https://second.example/api/v1", "vendor/judge", "OR_KEY")
    assert tier.key_url == "https://second.example/keys"
    keystore.store("TS_KEY", "k")
    assert parse_config(judge_raw()).tiers["judge"].api_key_env == "TS_KEY"  # order decides a tie


def test_an_endpoint_merges_extra_body_over_the_tiers():
    raw = judge_raw(extra_body={"a": 1, "b": 1})
    raw["tiers"]["judge"]["endpoints"][0]["extra_body"] = {"b": 2}
    assert parse_config(raw).tiers["judge"].extra_body == {"a": 1, "b": 2}


@pytest.mark.parametrize("endpoints,message", [
    ([], "non-empty list"),
    (["x"], "must be a mapping"),
    ([{"base_url": "https://x", "api_key_env": "K", "prices": {}}], "unknown fields"),
])
def test_a_malformed_endpoint_list_is_a_config_error(endpoints, message):
    raw = judge_raw()
    raw["tiers"]["judge"]["endpoints"] = endpoints
    with pytest.raises(ConfigError, match=message):
        parse_config(raw)


def test_an_endpoint_with_no_url_anywhere_is_a_config_error():
    raw = judge_raw()
    raw["tiers"]["judge"]["endpoints"] = [{"api_key_env": "K"}]
    with pytest.raises(ConfigError, match="needs a base_url"):
        parse_config(raw)


# ---------------------------------------------------------------- jev-model-router keys
@pytest.fixture
def config_file(tmp_path):
    import yaml

    raw = judge_raw()
    raw["tiers"]["top"] = {"backend": "openai_compatible", "model": "big", "base_url": "https://top.example/v1",
                           "api_key_env": "TOP_KEY"}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def keys(config_file, *argv, stdin=""):
    import sys

    old = sys.stdin
    sys.stdin = io.StringIO(stdin)
    try:
        return cli.main(["-c", config_file, "keys", *argv])
    finally:
        sys.stdin = old


def test_keys_lists_every_provider_and_fails_while_a_tier_has_none(config_file, capsys, monkeypatch):
    monkeypatch.setenv("OR_KEY", "k")
    assert keys(config_file) == 1
    out, err = capsys.readouterr()
    assert "judge  (the first provider with a key is used)" in out
    rows = [line.split() for line in out.splitlines()]
    assert ["*", "OR_KEY", "env", "Second"] in rows and ["TS_KEY", "missing", "First"] in rows
    assert ["TOP_KEY", "missing"] in rows and "1 tier(s) have no key" in err


def test_keys_set_asks_which_provider_and_stores_that_key(config_file, capsys):
    # judge: provider 2, then its key; top: its key.
    assert keys(config_file, "set", stdin="2\nor-secret\ntop-secret\n") == 0
    assert keystore.read() == {"OR_KEY": "or-secret", "TOP_KEY": "top-secret"}
    err = capsys.readouterr().err
    assert "1) First  (TS_KEY)" in err and "get a key at https://second.example/keys" in err
    assert keys(config_file) == 0
    assert parse_config(judge_raw()).tiers["judge"].api_key_env == "OR_KEY"


def test_keys_set_defaults_to_the_first_provider(config_file):
    assert keys(config_file, "set", stdin="\nts-secret\ntop-secret\n") == 0
    assert keystore.read() == {"TS_KEY": "ts-secret", "TOP_KEY": "top-secret"}


def test_keys_set_refuses_a_provider_it_did_not_offer(config_file, capsys):
    assert keys(config_file, "set", stdin="9\ntop-secret\n") == 1
    assert keystore.read() == {"TOP_KEY": "top-secret"}
    assert "no provider '9'" in capsys.readouterr().err


def test_keys_set_name_needs_no_config(tmp_path):
    assert keys(str(tmp_path / "absent.yaml"), "set", "OR_KEY", stdin="k\n") == 0
    assert keystore.read() == {"OR_KEY": "k"}
    assert keys(str(tmp_path / "absent.yaml"), "unset", "OR_KEY") == 0
    assert keystore.read() == {}


def test_keys_set_with_nothing_missing_changes_nothing(config_file, capsys, monkeypatch):
    monkeypatch.setenv("TS_KEY", "k")
    monkeypatch.setenv("TOP_KEY", "k")
    assert keys(config_file, "set") == 0
    assert "every key the config names is set" in capsys.readouterr().out
    assert keystore.read() == {}


def test_check_names_the_tiers_without_a_key(config_file, capsys):
    assert cli.main(["-c", config_file, "check"]) == 0
    err = capsys.readouterr().err
    assert "tier 'judge' has no key: set TS_KEY or OR_KEY" in err
    assert "tier 'top' has no key: set TOP_KEY" in err


# ---------------------------------------------------------------- /v1/route
def test_a_route_that_fell_back_says_why(backend_factory):
    from jev_model_router.app import create_app

    config = parse_config(raw_config())
    router = CapabilityRouter(config, ask=Ask(error=RuntimeError("401 no key")))
    app = create_app(config, backend_factory=backend_factory, log=RequestLog(":memory:"), router=router)
    with TestClient(app) as client:
        body = client.post("/v1/route", json={"task": "Fix it"}).json()
    assert body["rule"] == "fallback" and "401 no key" in body["why"]
