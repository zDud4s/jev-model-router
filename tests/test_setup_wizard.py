"""`jev-model-router init`: a working config from what the machine has.

Detection is faked (no CLI is run, no port is probed, no listing is fetched):
`which`, `get` and the catalog check are all injected.
"""

from __future__ import annotations

import filecmp
from pathlib import Path

import pytest
import yaml

from jev_model_router import cli, keystore
from jev_model_router import setup_wizard as wizard
from jev_model_router.catalog import Offered
from jev_model_router.config import load_config, parse_config

from test_discovery import report, served

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- the shipped data
def test_the_shipped_benchmarks_are_the_repositorys():
    assert filecmp.cmp(ROOT / "benchmarks.yaml", wizard.DATA / "benchmarks.yaml", shallow=False), (
        "jev_model_router/data/benchmarks.yaml drifted from benchmarks.yaml: copy it over"
    )


def test_the_shipped_setup_names_sources_a_judge_and_a_parseable_base():
    setup = wizard.load_setup()
    assert setup["base"]["router"]["capabilities"]["requirements"]
    assert {s["backend"] for s in setup["sources"].values()} >= {"claude_cli", "codex_cli", "ollama"}
    assert len(setup["judge"]["endpoints"]) >= 2


# ---------------------------------------------------------------- finding the executable
def test_a_plain_path_entry_is_used_as_is(tmp_path):
    exe = tmp_path / "tool"
    exe.write_text("")
    assert wizard.find_executable("tool", which=lambda n: str(exe)) == str(exe)
    assert wizard.find_executable("tool", which=lambda n: None) is None


def test_an_npm_shim_that_calls_an_exe_resolves_to_it(tmp_path):
    exe = tmp_path / "node_modules" / "@acme" / "tool" / "bin" / "tool.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    shim = tmp_path / "tool.CMD"
    shim.write_text('CALL :find_dp0\n"%dp0%\\node_modules\\@acme\\tool\\bin\\tool.exe"   %*\n')
    assert Path(wizard.find_executable("tool", which=lambda n: str(shim))) == exe


def test_an_npm_shim_that_calls_a_script_resolves_to_the_exe_in_its_package(tmp_path):
    package = tmp_path / "node_modules" / "@acme" / "tool"
    (package / "bin").mkdir(parents=True)
    (package / "package.json").write_text("{}")
    (package / "bin" / "tool.js").write_text("")
    exe = package / "node_modules" / "@acme" / "tool-win" / "vendor" / "bin" / "tool.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    shim = tmp_path / "tool.cmd"
    shim.write_text('"%_prog%"  "%dp0%\\node_modules\\@acme\\tool\\bin\\tool.js" %*\n')
    assert Path(wizard.find_executable("tool", which=lambda n: str(shim))) == exe


def test_a_shim_with_nothing_runnable_behind_it_is_kept(tmp_path):
    shim = tmp_path / "tool.cmd"
    shim.write_text("@echo off\n")
    assert wizard.find_executable("tool", which=lambda n: str(shim)) == str(shim)


# ---------------------------------------------------------------- detection
def test_detection_by_backend_kind(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    def get(url):
        raise OSError("refused")

    found = {d.name: d for d in wizard.detect(wizard.load_setup(), which=lambda n: f"/usr/bin/{n}", get=get)}
    assert found["claude"].found and Path(found["claude"].where) == Path("/usr/bin/claude")
    assert not found["local"].found and "not answering" in found["local"].note
    assert found["or"].found and found["or"].where == "OPENROUTER_API_KEY"


def test_an_api_source_without_its_key_is_offered_but_not_chosen_by_default(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    found = {d.name: d for d in wizard.detect(wizard.load_setup(), which=lambda n: None, get=lambda u: {})}
    assert not found["or"].found and found["or"].note == "needs OPENROUTER_API_KEY"
    assert found["local"].found  # Ollama answered


# ---------------------------------------------------------------- building the config
def claude(where="C:/bin/claude.exe"):
    return wizard.Detected("claude", "Claude Code", True, where)


CATALOG = report(claude=served(**{
    "claude-opus-5-5": Offered("claude-opus-5-5", efforts=("high",)),
    "claude-sonnet-5": Offered("claude-sonnet-5", efforts=("high",)),
    "claude-haiku-4-5": Offered("claude-haiku-4-5"),
}))


def test_the_config_names_the_found_executable_and_puts_the_chosen_judge_first(tmp_path):
    raw = wizard.build_raw(wizard.load_setup(), [claude()], "OpenRouter", tmp_path)
    caps = raw["router"]["capabilities"]
    assert caps["discover"]["claude"]["base_url"] == "C:/bin/claude.exe"
    assert "label" not in caps["discover"]["claude"] and "executable" not in caps["discover"]["claude"]
    assert raw["tiers"]["judge"]["endpoints"][0]["label"] == "OpenRouter"
    assert Path(raw["log"]["path"]).parent == tmp_path
    assert caps["benchmarks"]["path"] == str(tmp_path / "benchmarks.yaml")


def test_nothing_chosen_is_refused(tmp_path):
    with pytest.raises(ValueError):
        wizard.build_raw(wizard.load_setup(), [], "TypeSafe", tmp_path)


def test_the_default_is_the_cheapest_tier_able_enough(tmp_path):
    raw = wizard.build_raw(wizard.load_setup(), [claude()], "TypeSafe", tmp_path)
    name, tier = wizard.pick_default(raw, check=lambda config: CATALOG)
    assert name == "claude:claude-sonnet-5@high"  # opus is abler but dearer; haiku is cheaper but not able enough
    assert tier == {"backend": "claude_cli", "model": "claude-sonnet-5", "base_url": "C:/bin/claude.exe",
                    "context_window": 1000000, "timeout_s": 900.0, "effort": "high"}


def test_with_nothing_able_enough_the_default_is_the_strongest(tmp_path):
    weak = report(claude=served(**{"claude-haiku-4-5": Offered("claude-haiku-4-5"),
                                   "unknown-1": Offered("unknown-1")}))
    raw = wizard.build_raw(wizard.load_setup(), [claude()], "TypeSafe", tmp_path)
    assert wizard.pick_default(raw, check=lambda config: weak)[0] == "claude:claude-haiku-4-5"


def test_no_model_found_is_an_error_that_says_where_to_look(tmp_path):
    raw = wizard.build_raw(wizard.load_setup(), [claude()], "TypeSafe", tmp_path)
    with pytest.raises(ValueError, match="no model was found"):
        wizard.pick_default(raw, check=lambda config: report(claude=served()))


def test_the_written_config_loads_and_the_data_is_copied_once(tmp_path):
    raw = wizard.build_raw(wizard.load_setup(), [claude()], "TypeSafe", tmp_path)
    final = wizard.finish(raw, wizard.pick_default(raw, check=lambda config: CATALOG))
    wizard.write(final, tmp_path / "config.yaml")
    config = load_config(tmp_path / "config.yaml")
    assert config.router.default_tier == "claude:claude-sonnet-5@high"
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8").startswith("# Written by")
    assert {p.name for p in wizard.copy_data(tmp_path)} == set(wizard.DATA_FILES)
    (tmp_path / "benchmarks.yaml").write_text("edited", encoding="utf-8")
    assert wizard.copy_data(tmp_path) == [] and (tmp_path / "benchmarks.yaml").read_text() == "edited"


def test_an_api_default_carries_its_effort_in_the_body_not_as_a_cli_effort():
    from jev_model_router.config import Prices, TierConfig

    tier = TierConfig(name="or:v/m@high", backend="openai_compatible", model="v/m", base_url="https://api/v1",
                      api_key_env="K", effort="high", extra_body={"reasoning": {"effort": "high"}},
                      prices=Prices(input=1.0, output=2.0, configured=True))
    out = wizard.explicit_tier(tier)
    assert "effort" not in out and out["extra_body"] == {"reasoning": {"effort": "high"}}
    assert out["prices"]["output"] == 2.0 and out["api_key_env"] == "K"
    raw = {"router": {"default_tier": "x"}, "tiers": {"x": out, "c": {"backend": "ollama", "model": "m"}}}
    parse_config(raw)  # an explicit API tier with `effort` would be refused


# ---------------------------------------------------------------- the command
def test_init_yes_writes_a_config_every_command_then_reads(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # no ./config.yaml here
    monkeypatch.delenv(cli.CONFIG_ENV, raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(wizard, "detect", lambda setup: [
        claude(), wizard.Detected("or", "API", False, "OPENROUTER_API_KEY", "needs OPENROUTER_API_KEY")])
    monkeypatch.setattr(wizard, "pick_default", lambda raw: ("claude:claude-sonnet-5@high", {
        "backend": "claude_cli", "model": "claude-sonnet-5", "base_url": "C:/bin/claude.exe", "effort": "high"}))
    assert cli.main(["init", "--yes"]) == 0
    written = keystore.home() / "config.yaml"
    assert written.is_file() and cli.default_config() == str(written)
    raw = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert list(raw["router"]["capabilities"]["discover"]) == ["claude"]  # the API had no key: not chosen
    out = capsys.readouterr().out
    assert f"wrote {written}" in out and "Next:" in out
    assert cli.main(["init", "--yes"]) == 2  # an existing config is not replaced without --force


def test_the_config_is_found_in_order_env_then_here_then_the_users(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(cli.CONFIG_ENV, raising=False)
    assert cli.default_config() == cli.DEFAULT_CONFIG  # nothing anywhere
    users = keystore.home() / "config.yaml"
    users.parent.mkdir(parents=True, exist_ok=True)
    users.write_text("x", encoding="utf-8")
    assert cli.default_config() == str(users)
    (tmp_path / "config.yaml").write_text("x", encoding="utf-8")
    assert cli.default_config() == cli.DEFAULT_CONFIG
    monkeypatch.setenv(cli.CONFIG_ENV, "/elsewhere.yaml")
    assert cli.default_config() == "/elsewhere.yaml"
