"""Editing the config file from the routing page: in place, comments kept, validated first."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from jev_model_router import config_edit
from jev_model_router.app import create_app
from jev_model_router.config import load_config
from jev_model_router.db import RequestLog

EXAMPLE = Path(__file__).resolve().parent.parent / "config.example.yaml"

COMMENTED = """\
# The top comment.
log:
  path: ":memory:"
catalog:
  check_on_start: false   # tests pass catalog_check instead

router:
  kind: static
  default_tier: cheap

tiers:
  # The local one.
  cheap:
    backend: ollama
    model: llama3.1:8b
    context_window: 4096

  # The paid one.
  top:
    backend: openai_compatible
    model: top-model
    base_url: https://example.invalid/v1
    prices: {input: 15.0, output: 75.0}
    context_window: 200000
"""


def test_a_value_is_changed_in_its_line_and_every_comment_survives():
    text = config_edit.apply_changes(COMMENTED, [{"path": ["catalog", "check_on_start"], "value": True}])
    assert "  check_on_start: true   # tests pass catalog_check instead\n" in text
    assert text.replace("true   #", "false   #") == COMMENTED


def test_missing_keys_and_their_parents_are_added_under_the_right_block():
    text = config_edit.apply_changes(COMMENTED, [
        {"path": ["verification", "sample_rate"], "value": 0.5},
        {"path": ["tiers", "cheap", "supports_tools"], "value": True},
    ])
    data = yaml.safe_load(text)
    assert data["verification"] == {"sample_rate": 0.5}
    assert data["tiers"]["cheap"]["supports_tools"] is True
    # Inserted after cheap's own lines, not after the comment that introduces top.
    assert text.index("supports_tools: true") < text.index("# The paid one.")


def test_a_tier_is_added_as_a_block_and_removed_without_the_next_ones_comment():
    added = config_edit.apply_changes(COMMENTED, [
        {"path": ["tiers", "mid"], "value": {"backend": "ollama", "model": "qwen3:4b"}}])
    assert yaml.safe_load(added)["tiers"]["mid"] == {"backend": "ollama", "model": "qwen3:4b"}
    removed = config_edit.apply_changes(COMMENTED, [{"path": ["tiers", "cheap"], "delete": True}])
    assert "cheap" not in yaml.safe_load(removed)["tiers"]
    assert "# The paid one." in removed


def test_a_key_written_inline_is_refused_rather_than_rewritten():
    with pytest.raises(config_edit.EditError, match="inline"):
        config_edit.apply_changes(COMMENTED, [{"path": ["tiers", "top", "prices", "input"], "value": 3.0}])


def test_every_edit_on_the_example_config_lands_exactly():
    text = EXAMPLE.read_text(encoding="utf-8")
    changes = [
        {"path": ["verification", "enabled"], "value": True},
        {"path": ["verification", "verify_tiers"], "value": ["cheap", "mid"]},
        {"path": ["tiers", "mid", "prices", "input"], "value": 2.5},
        {"path": ["tiers", "judge", "jev", "threshold"], "value": 0.8},
        {"path": ["router", "model_map", "gpt-5"], "value": "top"},
        {"path": ["tiers", "mid", "api_key_env"], "delete": True},
    ]
    out = config_edit.apply_changes(text, changes)
    assert config_edit.check(out)["ok"]
    assert len(out.splitlines()) == len(text.splitlines())  # one added, one removed, the rest in place


def test_check_names_what_the_parser_refuses():
    bad = COMMENTED.replace("default_tier: cheap", "default_tier: nowhere")
    result = config_edit.check(bad)
    assert not result["ok"] and "nowhere" in result["error"]
    assert not config_edit.check("tiers: [unclosed")["ok"]


# ------------------------------------------------------------------ endpoints

def served(tmp_path: Path, backend_factory, text: str = COMMENTED) -> tuple[TestClient, Path]:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    app = create_app(load_config(path), backend_factory=backend_factory, log=RequestLog(":memory:"), config_path=path)
    return TestClient(app), path


def test_the_page_reads_the_file_it_was_started_from(tmp_path, backend_factory):
    client, path = served(tmp_path, backend_factory)
    with client:
        body = client.get("/routing/config").json()
    assert body["text"] == COMMENTED and body["served"] and body["check"]["ok"]
    assert body["data"]["router"]["default_tier"] == "cheap"
    assert "claude_cli" in body["options"]["backends"]


def test_the_form_offers_the_jev_failure_modes_the_parser_accepts(tmp_path, backend_factory):
    client, _ = served(tmp_path, backend_factory)
    with client:
        body = client.get("/routing/config").json()
        page = client.get("/routing").text
    assert body["options"]["jev_failure_modes"] == ["fallback", "reject"]
    assert '"on_jev_failure"' in page and "o.jev_failure_modes" in page


def test_a_review_writes_nothing_and_a_save_writes_in_place(tmp_path, backend_factory):
    client, path = served(tmp_path, backend_factory)
    change = {"changes": [{"path": ["router", "default_tier"], "value": "top"}]}
    with client:
        digest = client.get("/routing/config").json()["digest"]
        review = client.post("/routing/config/check", json=change).json()
        assert review["ok"] and "+  default_tier: top" in review["diff"]
        assert path.read_text(encoding="utf-8") == COMMENTED
        saved = client.put("/routing/config", json={**change, "base": digest}).json()
        assert saved["saved"] and not saved["served"]
        assert client.get("/routing/config").json()["served"] is False
    assert load_config(path).router.default_tier == "top"
    assert "# The paid one." in path.read_text(encoding="utf-8")


def test_a_file_the_router_would_refuse_is_not_written(tmp_path, backend_factory):
    client, path = served(tmp_path, backend_factory)
    with client:
        response = client.put("/routing/config", json={"text": "tiers: {}\n"})
    assert response.status_code == 400 and "tier" in response.json()["error"]["message"]
    assert path.read_text(encoding="utf-8") == COMMENTED


def test_a_save_over_a_file_changed_since_it_was_read_is_refused(tmp_path, backend_factory):
    client, path = served(tmp_path, backend_factory)
    with client:
        digest = client.get("/routing/config").json()["digest"]
        path.write_text(COMMENTED + "\n# edited by hand\n", encoding="utf-8")
        response = client.put("/routing/config", json={"text": COMMENTED, "base": digest})
    assert response.status_code == 409
    assert path.read_text(encoding="utf-8").endswith("# edited by hand\n")


def test_another_origin_cannot_write_the_config(tmp_path, backend_factory):
    client, path = served(tmp_path, backend_factory)
    with client:
        foreign = client.put("/routing/config", json={"text": COMMENTED}, headers={"Origin": "https://evil.example"})
        form = client.put("/routing/config", content="text=x", headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert foreign.status_code == 403 and form.status_code == 415


def test_without_a_config_file_there_is_nothing_to_edit(backend_factory, config):
    with TestClient(create_app(config, backend_factory=backend_factory, log=RequestLog(":memory:"))) as client:
        assert client.get("/routing/config").json()["path"] is None
        assert client.put("/routing/config", json={"text": "x"}).status_code == 404


def test_sources_on_offer_are_inits_list_minus_what_the_file_discovers(monkeypatch):
    from jev_model_router import setup_wizard

    setup = {"sources": {
        "cli": {"label": "A CLI", "executable": "cli", "backend": "claude_cli", "context_window": 1000},
        "api": {"label": "An API", "backend": "openai_compatible", "base_url": "https://example.invalid/v1",
                "api_key_env": "SOME_KEY"},
    }}
    monkeypatch.setattr(setup_wizard, "load_setup", lambda: setup)
    monkeypatch.setattr(setup_wizard, "detect", lambda s: [
        setup_wizard.Detected(n, s["sources"][n]["label"], True, "/bin/cli" if n == "cli" else "SOME_KEY")
        for n in s["sources"]])
    data = {"router": {"capabilities": {"discover": {"api": {}}}}}
    [offer] = config_edit.sources_on_offer(data)
    assert offer["name"] == "cli" and offer["ready"]
    # What is written is a discover entry: the executable resolved, init's own keys gone.
    assert offer["entry"] == {"backend": "claude_cli", "context_window": 1000, "base_url": "/bin/cli"}
