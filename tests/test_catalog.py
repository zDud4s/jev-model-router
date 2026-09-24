"""The startup catalog check: tiers against what each provider says it serves.

No CLI runs and no network is touched: `run` and `get` are scripted, and the
Claude and Codex caches are written into a temporary directory.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from llm_router.app import create_app
from llm_router.catalog import OK, UNAVAILABLE, UNVERIFIED, check_catalog, write_if_changed
from llm_router.config import ConfigError, parse_config
from llm_router.db import RequestLog

from conftest import BASE_CONFIG

CODEX_MODELS = {
    "models": [
        {"slug": "gpt-5.6-sol", "visibility": "list", "supported_reasoning_levels": [{"effort": e} for e in ("low", "medium", "high")]},
        {"slug": "gpt-5.5", "visibility": "list", "supported_reasoning_levels": [{"effort": "low"}, {"effort": "xhigh"}]},
        {"slug": "codex-auto-review", "visibility": "hide", "supported_reasoning_levels": []},
    ]
}


def claude_catalog(tmp: Path, stale: bool = False) -> Path:
    folder = tmp / "claude" / "cache" / "model-catalog"
    folder.mkdir(parents=True)
    now_ms = time.time() * 1000
    effort = {"type": "effort", "effort_options": [{"id": e} for e in ("low", "medium", "high", "max")]}
    payload = {
        "version": 2,
        "fetchedAt": now_ms - 1000,
        "staleAt": now_ms - 10 if stale else now_ms + 3_600_000,
        "catalog": {"config": {"models": [
            {"id": "claude-opus-5-5", "short_name": "Opus", "section": "main", "thinking": effort,
             "min_claude_code_version": "2.1.280"},
            {"id": "claude-sonnet-5", "short_name": "Sonnet", "section": "main", "thinking": effort},
            {"id": "claude-haiku-4-5-20251001", "short_name": "Haiku", "section": "main", "thinking": {"type": "none"}},
            {"id": "claude-opus-4-8", "short_name": "Opus", "section": "overflow", "thinking": effort},
        ]}},
    }
    (folder / "cat.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp / "claude"


class Run:
    """Scripted CLI: `codex debug models` and `claude --version`."""

    def __init__(self, codex: Any = CODEX_MODELS, claude_version: str = "2.1.281 (Claude Code)") -> None:
        self.codex = codex
        self.claude_version = claude_version
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], timeout: float) -> str:
        self.calls.append(argv)
        if argv[1:] == ["debug", "models"]:
            if isinstance(self.codex, Exception):
                raise self.codex
            return json.dumps(self.codex)
        if argv[1:] == ["--version"]:
            return self.claude_version
        raise AssertionError(argv)


def config(tmp: Path, tiers: dict[str, Any], **catalog: Any):
    raw = copy.deepcopy(BASE_CONFIG)
    raw["tiers"] = {"judge": {"backend": "jev", "model": "jev-latest"}, **tiers}
    raw["router"] = {"kind": "static", "default_tier": next(iter(tiers))}
    raw["catalog"] = {"claude_dir": str(tmp / "claude"), "codex_home": str(tmp / "codex"), **catalog}
    return parse_config(raw)


def ollama(models: list[str]):
    def get(url: str) -> Any:
        assert url.endswith("/api/tags")
        return {"models": [{"name": m} for m in models]}
    return get


def down(url: str) -> Any:
    raise ConnectionError("refused")


# ---------------------------------------------------------------- codex
def test_a_codex_model_the_account_no_longer_serves_is_unavailable(tmp_path):
    cfg = config(tmp_path, {
        "sol": {"backend": "codex_cli", "model": "gpt-5.6-sol", "effort": "high"},
        "gone": {"backend": "codex_cli", "model": "gpt-6-sol", "effort": "high"},
    })
    report = check_catalog(cfg, run=Run())
    assert report.tiers["sol"].status == OK
    assert report.tiers["gone"].status == UNAVAILABLE
    assert "gpt-6-sol" in report.unavailable["gone"] and "codex" in report.unavailable["gone"]


def test_an_effort_the_model_does_not_take_is_unavailable(tmp_path):
    cfg = config(tmp_path, {"old": {"backend": "codex_cli", "model": "gpt-5.5", "effort": "max"}})
    reason = check_catalog(cfg, run=Run()).unavailable["old"]
    assert "effort 'max'" in reason and "low, xhigh" in reason


def test_hidden_utility_models_are_not_reported_as_new(tmp_path):
    cfg = config(tmp_path, {"sol": {"backend": "codex_cli", "model": "gpt-5.6-sol"}})
    assert check_catalog(cfg, run=Run()).unconfigured == {"codex": ["gpt-5.5"]}


def test_a_failed_refresh_falls_back_to_the_cache_and_says_it_is_stale(tmp_path):
    (tmp_path / "codex").mkdir()
    (tmp_path / "codex" / "models_cache.json").write_text(
        json.dumps({"fetched_at": "2026-09-24T17:43:56Z", **CODEX_MODELS}), encoding="utf-8"
    )
    cfg = config(tmp_path, {"sol": {"backend": "codex_cli", "model": "gpt-5.6-sol"}})
    report = check_catalog(cfg, run=Run(codex=RuntimeError("offline")))
    assert report.tiers["sol"].status == OK
    source = report.sources["codex"]
    assert source.stale and "offline" in source.error


def test_no_catalog_at_all_leaves_the_tier_in_as_unverified(tmp_path):
    cfg = config(tmp_path, {"sol": {"backend": "codex_cli", "model": "gpt-5.6-sol"}})
    report = check_catalog(cfg, run=Run(codex=RuntimeError("offline")))
    # A missing cache is not evidence that the model is gone.
    assert report.tiers["sol"].status == UNVERIFIED
    assert report.unavailable == {}


# ---------------------------------------------------------------- claude
def test_claude_ids_match_dated_and_short_names(tmp_path):
    claude_catalog(tmp_path)
    cfg = config(tmp_path, {
        "haiku": {"backend": "claude_cli", "model": "claude-haiku-4-5"},
        "alias": {"backend": "claude_cli", "model": "sonnet", "effort": "low"},
        "opus": {"backend": "claude_cli", "model": "claude-opus-5-5", "effort": "high"},
    })
    report = check_catalog(cfg, run=Run())
    assert {name: s.status for name, s in report.tiers.items() if name != "judge"} == {
        "haiku": OK, "alias": OK, "opus": OK,
    }
    assert report.unconfigured == {"claude": ["claude-opus-4-8"]}


def test_a_model_that_takes_no_effort_refuses_one(tmp_path):
    claude_catalog(tmp_path)
    cfg = config(tmp_path, {"haiku": {"backend": "claude_cli", "model": "claude-haiku-4-5", "effort": "low"}})
    assert "takes: none" in check_catalog(cfg, run=Run()).unavailable["haiku"]


def test_a_model_newer_than_the_cli_is_unavailable_until_the_cli_is_updated(tmp_path):
    claude_catalog(tmp_path)
    cfg = config(tmp_path, {"opus": {"backend": "claude_cli", "model": "claude-opus-5-5"}})
    reason = check_catalog(cfg, run=Run(claude_version="2.1.278 (Claude Code)")).unavailable["opus"]
    assert "needs Claude Code >= 2.1.280" in reason and "2.1.278" in reason


def test_a_stale_claude_catalog_is_still_used_and_flagged(tmp_path):
    claude_catalog(tmp_path, stale=True)
    cfg = config(tmp_path, {"s": {"backend": "claude_cli", "model": "claude-sonnet-5"}})
    report = check_catalog(cfg, run=Run())
    assert report.tiers["s"].status == OK
    assert report.sources["claude"].stale
    assert "stale" in report.summary()


# ---------------------------------------------------------------- ollama and the rest
def test_ollama_is_asked_for_its_tags_and_an_unreachable_server_serves_nothing(tmp_path):
    tiers = {"local": {"backend": "ollama", "model": "qwen3.5", "base_url": "http://localhost:11434"}}
    assert check_catalog(config(tmp_path, tiers), get=ollama(["qwen3.5:latest"])).tiers["local"].status == OK
    missing = check_catalog(config(tmp_path, tiers), get=ollama(["llama3:8b"]))
    assert missing.tiers["local"].status == UNAVAILABLE
    unreachable = check_catalog(config(tmp_path, tiers), get=down)
    assert "unreachable" in unreachable.unavailable["local"]


def test_a_backend_with_no_catalog_source_is_not_checked(tmp_path):
    report = check_catalog(config(tmp_path, {"api": {
        "backend": "openai_compatible", "model": "m", "base_url": "https://example.invalid/v1"}}))
    assert report.tiers["api"].status == "not_checked"
    assert report.unavailable == {}


# ---------------------------------------------------------------- the file
def test_the_file_is_rewritten_only_when_the_catalog_changes(tmp_path):
    cfg = config(tmp_path, {"sol": {"backend": "codex_cli", "model": "gpt-5.6-sol"}})
    path = tmp_path / "catalog.json"
    assert write_if_changed(check_catalog(cfg, run=Run()), path) is True
    # Same models, a later check: nothing to write.
    assert write_if_changed(check_catalog(cfg, run=Run()), path) is False
    changed = {"models": CODEX_MODELS["models"][1:]}
    assert write_if_changed(check_catalog(cfg, run=Run(codex=changed)), path) is True
    assert json.loads(path.read_text(encoding="utf-8"))["tiers"]["sol"]["status"] == UNAVAILABLE


def test_catalog_settings_are_validated():
    raw = copy.deepcopy(BASE_CONFIG)
    raw["catalog"] = {"refresh": True}
    with pytest.raises(ConfigError, match="unknown catalog fields"):
        parse_config(raw)
    raw["catalog"] = {"path": ""}
    assert parse_config(raw).catalog.path is None
    assert parse_config(BASE_CONFIG).catalog.check_on_start is False  # the suite's own setting
    del raw["catalog"]
    assert parse_config(raw).catalog.check_on_start is True
    assert parse_config(raw).catalog.path == "llm-router.catalog.json"


# ---------------------------------------------------------------- through the app
def test_at_startup_an_unavailable_tier_leaves_eligibility_with_its_reason(tmp_path, backend_factory, fake_backends):
    raw = copy.deepcopy(BASE_CONFIG)
    raw["catalog"] = {"check_on_start": True, "path": None}
    cfg = parse_config(raw)

    class Report:
        checked_at = "2026-09-24T18:00:00+00:00"
        unavailable = {"cheap": "'llama3.1:8b' is not in the catalog of ollama http://localhost:11434"}
        unconfigured = {"ollama http://localhost:11434": ["qwen3.5:4b"]}

    log = RequestLog(":memory:")
    app = create_app(cfg, backend_factory=backend_factory, log=log, catalog_check=lambda c: Report())
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
        # `cheap` is the default tier; the static router falls back to the next eligible one.
        assert response.status_code == 200
        assert response.headers["X-Router-Tier"] == "mid"
        health = client.get("/healthz").json()
        row = log.query("SELECT eligibility_rejections FROM requests")[0]
    assert health["catalog"]["unavailable"] == Report.unavailable
    assert health["catalog"]["unconfigured"] == Report.unconfigured
    assert fake_backends["cheap"].calls == []
    rejected = {r["tier"]: r for r in json.loads(row["eligibility_rejections"])}
    assert rejected["cheap"]["reason"] == "unavailable"
    assert "not in the catalog" in rejected["cheap"]["detail"]


def test_a_check_that_crashes_does_not_stop_the_proxy(backend_factory):
    raw = copy.deepcopy(BASE_CONFIG)
    raw["catalog"] = {"check_on_start": True, "path": None}

    def boom(config):
        raise RuntimeError("catalog source exploded")

    app = create_app(parse_config(raw), backend_factory=backend_factory, log=RequestLog(":memory:"), catalog_check=boom)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 200
        assert response.headers["X-Router-Tier"] == "cheap"
        assert "catalog" not in client.get("/healthz").json()
