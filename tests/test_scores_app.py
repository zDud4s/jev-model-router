"""Benchmark evidence in discovery, at startup and on the command line."""

from __future__ import annotations

import pytest

from llm_router.catalog import CatalogReport, Offered, Source
from llm_router.config import parse_config
from llm_router.discovery import expand, model_ids, served_ids
from llm_router.scores_derive import served_keys, startup_lines

from test_scores import BENCHES, linked, pt, raw_config, scored


def catalog(**models: Offered) -> CatalogReport:
    source = Source(name="cli", models=dict(models), cli_version="9.9.9")
    return CatalogReport(checked_at="now", tiers={}, sources={}, unconfigured={}, discovered={"cli": source})


REPORT = catalog(**{
    "acme-large": Offered("acme-large", efforts=("medium", "high"), aliases=("acme-large-latest",)),
    "acme-small": Offered("acme-small"),
    "unknown-1": Offered("unknown-1", efforts=("high",)),
})
POINTS = linked(pt("code", "acme-large-latest", "high", 80), pt("code", "acme-small", None, 40),
                pt("code", "zeta-1", "high", 60))


def test_expand_without_scores_builds_todays_cards():
    expanded, _, _ = expand(parse_config(raw_config()), REPORT)
    assert expanded.router.capabilities.cards["cli:acme-large@medium"].levels == {
        "reasoning": pytest.approx(1.7), "niche": 2.0}


def test_expand_with_scores_derives_the_discovered_cards_through_their_aliases():
    _, scores = scored(POINTS)
    expanded, _, _ = expand(parse_config(raw_config()), REPORT, scores)
    cards = expanded.router.capabilities.cards
    assert cards["cli:acme-large@high"].levels["reasoning"] > cards["cli:acme-small"].levels["reasoning"]
    assert cards["cli:acme-large@high"].family == "acme-large"
    assert cards["cli:unknown-1@high"].levels == {"reasoning": 0.5, "niche": 0.5}  # no evidence: the profile


def test_an_explicit_tier_without_a_card_is_derived_too():
    raw = raw_config()
    raw["tiers"]["large"] = {"backend": "claude_cli", "model": "acme-large", "effort": "high",
                             "base_url": "C:/bin/cli.exe"}
    _, scores = scored(POINTS)
    expanded, _, _ = expand(parse_config(raw), REPORT, scores)
    assert expanded.router.capabilities.cards["large"].levels["reasoning"] > 2.0


def test_one_model_at_several_efforts_is_not_ambiguous_but_two_models_on_one_key_are():
    _, scores = scored(POINTS)
    keys, ambiguous = served_keys(scores, {"cli:a@low": ("acme-large",), "cli:a@high": ("acme-large",),
                                           "cli:b": ("acme_large",), "cli:c": ("zeta-1",)})
    assert ambiguous == {"acme-large"} and keys["cli:a@high"] == () and keys["cli:c"] == ("zeta-1",)


def test_model_ids_and_served_ids_name_every_id_a_point_may_use():
    _, found, _ = expand(parse_config(raw_config()), REPORT)
    assert model_ids(REPORT)["acme-large"] == ("acme-large", "acme-large-latest")
    assert served_ids(found, REPORT)["cli:acme-large"] == ("acme-large", "acme-large-latest")


def test_startup_lines_name_bare_and_vendor_only_models_and_unread_benchmarks():
    benches = {**BENCHES, "fresh": {"description": "never read"}}
    caps, scores = scored(linked(pt("code", "acme-large", "high", 80), pt("code", "zeta-1", "high", 60,
                                                                           origin="independent")), benches)
    served = {"cli:acme-large": ("acme-large",), "cli:acme-small": ("acme-small",), "cli:zeta-1": ("zeta-1",)}
    lines = startup_lines(scores, caps, served)
    assert "benchmarks: 1 served model(s) with no evidence: cli:acme-small" in lines
    assert not any("vendor evidence only" in line for line in lines)  # linked() adds independent base points
    assert any(line.startswith("benchmarks: unread") and "fresh" in line for line in lines)
    ambiguous = {**served, "cli:acme_large": ("acme_large",)}
    assert "benchmarks: ambiguous model key(s), used for no tier: acme-large" in         startup_lines(scores, caps, ambiguous)
    caps, vendor = scored([pt("code", "acme-large", "high", 80), pt("code", "zeta-1", "high", 60)])
    assert "benchmarks: 2 served model(s) with vendor evidence only: cli:acme-large, cli:zeta-1" in \
        startup_lines(vendor, caps, served)


# ---------------------------------------------------------------- startup
import json  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

import yaml  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from llm_router.app import create_app  # noqa: E402
from llm_router.db import RequestLog  # noqa: E402

from test_scores_import import META, SOURCES, Fetch, zipped  # noqa: E402


def _app_config(tmp_path, body: dict, **settings):
    path = tmp_path / "b.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    raw = raw_config(benchmarks={"path": str(path), **settings})
    raw["catalog"] = {"check_on_start": True, "path": None}
    return parse_config(raw)


CURATED = {"benchmarks": BENCHES, "points": POINTS}


def test_at_startup_evidence_is_loaded_and_reported_without_asking_jev(tmp_path, capsys, backend_factory):
    config = _app_config(tmp_path, CURATED)
    app = create_app(config, backend_factory=backend_factory, log=RequestLog(":memory:"),
                     catalog_check=lambda c: REPORT)
    err = capsys.readouterr().err
    assert "benchmarks: 1 served model(s) with no evidence: cli:unknown-1" in err
    cards = app.state.config.router.capabilities.cards
    assert cards["cli:acme-large@high"].levels["reasoning"] > cards["cli:acme-small"].levels["reasoning"]


def test_a_stale_import_is_refreshed_at_startup_and_a_failing_one_does_not_stop_it(tmp_path, capsys, backend_factory):
    table = [{"Model version": "acme-large_high", "Score": "0.8"}, {"Model version": "zeta-1_high", "Score": "0.5"}]
    body = {"sources": {"hub": SOURCES["hub"]}, "benchmarks": {
        **BENCHES, "hubbed": {"description": "d", "requirements": {"reasoning": 1.0},
                              "data": {"source": "hub", "table": "t.csv", "score": "Score"}}}, "points": POINTS}
    config = _app_config(tmp_path, body)
    fetch = Fetch(**{"https://hub_test": zipped(t__csv=table, meta__csv=META)})
    create_app(config, backend_factory=backend_factory, log=RequestLog(":memory:"),
               catalog_check=lambda c: REPORT, benchmark_fetch=fetch)
    assert len(fetch.calls) == 1 and "benchmarks: import refreshed (hub: 2 point(s))" in capsys.readouterr().err
    create_app(config, backend_factory=backend_factory, log=RequestLog(":memory:"),
               catalog_check=lambda c: REPORT, benchmark_fetch=fetch)
    assert len(fetch.calls) == 1  # fresh: no second import
    old = json.loads((tmp_path / "b.imported.json").read_text(encoding="utf-8"))
    old["imported_at"] = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    (tmp_path / "b.imported.json").write_text(json.dumps(old), encoding="utf-8")
    app = create_app(config, backend_factory=backend_factory, log=RequestLog(":memory:"),
                     catalog_check=lambda c: REPORT, benchmark_fetch=Fetch())
    assert "import failed for hub, keeping its previous points" in capsys.readouterr().err
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200


def test_a_broken_benchmarks_file_does_not_stop_startup(tmp_path, capsys, backend_factory):
    config = _app_config(tmp_path, {"points": [pt("nope", "m", "high", 1)]})
    app = create_app(config, backend_factory=backend_factory, log=RequestLog(":memory:"),
                     catalog_check=lambda c: REPORT)
    assert "benchmarks failed to load" in capsys.readouterr().err
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
    assert app.state.config.router.capabilities.cards["cli:acme-large@high"].levels["reasoning"] == 2.0


# ---------------------------------------------------------------- command line
from llm_router import cli  # noqa: E402


def _cli_config(tmp_path, monkeypatch, body: dict) -> str:
    bench = tmp_path / "b.yaml"
    bench.write_text(yaml.safe_dump(body), encoding="utf-8")
    raw = raw_config(benchmarks={"path": str(bench)})
    raw["catalog"] = {"check_on_start": False, "path": None}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setattr("llm_router.catalog.check_catalog", lambda config: REPORT)
    return str(path)


def test_benchmarks_check_prints_benchmarks_evidence_per_model_and_levels(tmp_path, monkeypatch, capsys):
    path = _cli_config(tmp_path, monkeypatch, CURATED)
    assert cli.main(["-c", path, "benchmarks", "check"]) == 0
    out = capsys.readouterr().out
    flat = " ".join(out.split())
    assert "code: difficulty" in flat and "3 model(s), 3 linking" in flat
    assert "weights (manual): reasoning=1.00" in flat
    assert "cli:acme-small: base (None); code (None)" in flat
    assert "cli:unknown-1: no evidence" in flat
    assert "cli:acme-large@high: output_tokens" in flat
    assert "niche 2.00 profile C=0.00" in flat  # uncovered, still listed


def test_benchmarks_needs_the_block(tmp_path):
    raw = raw_config()
    del raw["router"]["capabilities"]["benchmarks"]
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert cli.main(["-c", str(path), "benchmarks", "check"]) == 2


def test_benchmarks_import_fetches_and_reports(tmp_path, monkeypatch, capsys):
    table = [{"Model version": "acme-large_high", "Score": "0.8"}]
    body = {"sources": {"hub": SOURCES["hub"]}, "benchmarks": {
        "hubbed": {"description": "d", "requirements": {"reasoning": 1.0},
                   "data": {"source": "hub", "table": "t.csv", "score": "Score"}}}, "points": []}
    path = _cli_config(tmp_path, monkeypatch, body)
    fetch = Fetch(**{"https://hub_test": zipped(t__csv=table, meta__csv=META)})
    monkeypatch.setattr("llm_router.scores_import.http_fetch", lambda timeout: fetch)
    assert cli.main(["-c", path, "benchmarks", "import"]) == 0
    assert "hub: 1 row(s) read, 0 skipped, 1 point(s) in 1 benchmark(s)" in capsys.readouterr().out
    assert cli.main(["-c", path, "benchmarks", "import", "--source", "nope"]) == 2
    monkeypatch.setattr("llm_router.scores_import.http_fetch", lambda timeout: Fetch())
    assert cli.main(["-c", path, "benchmarks", "import"]) == 1
    assert "failed: hub" in capsys.readouterr().err


def test_benchmarks_fit_with_no_outcomes_says_so_and_exits_zero(tmp_path, monkeypatch, capsys):
    path = _cli_config(tmp_path, monkeypatch, CURATED)
    assert cli.main(["-c", path, "benchmarks", "fit", "--db", str(tmp_path / "empty.db")]) == 0
    assert "nothing to fit" in capsys.readouterr().out
    assert not (tmp_path / "b.derived.json").exists()


def test_a_snapshot_and_its_undated_id_served_apart_are_one_model_not_an_ambiguity():
    _, scores = scored(POINTS, extra={"model_keys": {"date_suffixes": [r"[-_]\d{8}$"]}})
    keys, ambiguous = served_keys(scores, {"cli:acme-large": ("acme-large",),
                                           "api:acme-large-20260101": ("acme-large-20260101",),
                                           "cli:b": ("acme_large",)})
    assert ambiguous == {"acme-large"}  # the underscore is a different model string, not a date
    keys, ambiguous = served_keys(scores, {"cli:acme-large": ("acme-large",),
                                           "api:acme-large-20260101": ("acme-large-20260101",)})
    assert ambiguous == set() and keys["cli:acme-large"] == keys["api:acme-large-20260101"] == ("acme-large",)


def test_expand_reads_the_benchmark_weights_once_and_derive_takes_them(monkeypatch):
    import llm_router.discovery as discovery
    import llm_router.scores_derive as scores_derive
    from llm_router.scores import benchmark_weights
    from test_scores import profile

    caps, scores = scored(POINTS)
    card = scores_derive.derive(caps, scores, ("acme-large",), "high", profile(caps), (2.0, 1.0), weights={})
    assert card.coverage["reasoning"] == 0.0 and card.levels["reasoning"] == 2.0
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return benchmark_weights(*args, **kwargs)

    monkeypatch.setattr(scores_derive, "benchmark_weights", counted)
    monkeypatch.setattr(discovery, "benchmark_weights", counted, raising=False)
    expand(parse_config(raw_config()), REPORT, scores)
    assert len(calls) == 1


def test_served_ids_cover_the_explicit_tiers_expand_derives_too(tmp_path, capsys, backend_factory):
    raw = raw_config()
    raw["tiers"]["small"] = {"backend": "claude_cli", "model": "acme-small", "base_url": "C:/bin/cli.exe"}
    config = parse_config(raw)
    _, found, _ = expand(config, REPORT)
    assert "cli:acme-small" not in {f"{f.source}:{f.model}" for f in found}  # the explicit tier covers it
    assert served_ids(found, REPORT, config)["cli:acme-small"] == ("acme-small",)
    # At startup, a model served only through an explicit tier is named when it has no evidence.
    body = {"benchmarks": BENCHES, "points": linked(pt("code", "acme-large", "high", 80), pt("code", "zeta-1", "high", 60))}
    path = tmp_path / "b.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    raw["router"]["capabilities"]["benchmarks"] = {"path": str(path)}
    raw["catalog"] = {"check_on_start": True, "path": None}
    create_app(parse_config(raw), backend_factory=backend_factory, log=RequestLog(":memory:"),
               catalog_check=lambda c: REPORT)
    assert "no evidence: cli:acme-small, cli:unknown-1" in capsys.readouterr().err


def test_benchmarks_fit_write_with_no_outcome_on_a_derived_tier_writes_nothing(tmp_path, monkeypatch, capsys):
    from llm_router.calibration import Outcome

    path = _cli_config(tmp_path, monkeypatch, CURATED)
    monkeypatch.setattr("llm_router.calibration.log_outcomes",
                        lambda log, config: [Outcome("explicit-only", {"reasoning": 1.0}, False, "verdict")])
    db = tmp_path / "log.db"
    RequestLog(str(db)).close()
    assert cli.main(["-c", path, "benchmarks", "fit", "--db", str(db), "--write"]) == 0
    assert "nothing to fit" in capsys.readouterr().out
    assert not (tmp_path / "b.derived.json").exists()
