"""`benchmarks import`: rows from a data source become points. No test touches the network."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import datetime, timezone

import pytest
import yaml

from llm_router.config import parse_config
from llm_router.scores import load_scores
from llm_router.scores_import import import_sources

from test_scores import raw_config

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


def zipped(**tables: list[dict[str, object]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, rows in tables.items():
            text = io.StringIO()
            writer = csv.DictWriter(text, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            archive.writestr(name.replace("__", "."), text.getvalue())
    return buffer.getvalue()


class Fetch:
    def __init__(self, **by_url: bytes | list[bytes]) -> None:
        self.by_url = by_url
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.deadlines: list[float] = []

    def __call__(self, url: str, headers: dict[str, str], deadline: float) -> bytes:
        self.calls.append((url, headers))
        self.deadlines.append(deadline)
        for prefix, body in self.by_url.items():
            if url.startswith(prefix.replace("_", ".")):
                if isinstance(body, list):
                    return body[len([c for c in self.calls if c[0].startswith(prefix.replace("_", "."))]) - 1]
                return body
        raise OSError(f"no route to {url}")


SOURCES = {
    "hub": {"format": "csv_zip", "url": "https://hub.test/data.zip", "model": "Model version", "origin": "independent",
            "metadata": {"file": "meta.csv", "join": "source_file", "score": "score_column", "scale": "scale",
                         "baseline": "random_baseline", "ceiling": "score_ceiling"}},
    "api": {"format": "json", "url": "https://api.test/models", "model": "name", "origin": "independent",
            "items": "data", "api_key_env": "TEST_BENCH_KEY", "paging": {"param": "page", "more": "more"}},
}


def setup(tmp_path, benchmarks: dict, **extra) -> tuple:
    path = tmp_path / "b.yaml"
    body = {"sources": SOURCES, "efforts": {"labels": {"extra high": "xhigh"}, "patterns": [r"^\d+k$"]},
            "model_keys": {"date_suffixes": [r"[-_]\d{8}$"]}, "benchmarks": benchmarks, "points": [], **extra}
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    config = parse_config(raw_config(benchmarks={"path": str(path)}))
    return config, path


def imported(tmp_path) -> dict:
    return json.loads((tmp_path / "b.imported.json").read_text(encoding="utf-8"))


HUB_TABLE = [
    {"Model version": "acme-large_max", "Score": "0.5", "Reasoning level": "Medium", "Cost": "2.0", "Harness": "h1"},
    {"Model version": "acme-large_max", "Score": "0.7", "Reasoning level": "Medium", "Cost": "4.0", "Harness": "h2"},
    {"Model version": "acme-large_xhigh", "Score": "0.6", "Reasoning level": "0.99", "Cost": "", "Harness": "h1"},
    {"Model version": "acme-small", "Score": "0.3", "Reasoning level": "", "Cost": "", "Harness": "h1"},
    {"Model version": "zeta-1_promax", "Score": "0.4", "Reasoning level": "Extra High", "Cost": "", "Harness": "h1"},
    {"Model version": "", "Score": "0.9", "Reasoning level": "", "Cost": "", "Harness": "h1"},
    {"Model version": "zeta-2", "Score": "", "Reasoning level": "", "Cost": "", "Harness": "h1"},
]
META = [{"source_file": "code.csv", "score_column": "Score", "scale": "1", "random_baseline": "0.25",
         "score_ceiling": "1"}]
CODE = {"description": "fix code", "requirements": {"reasoning": 1.0},
        "data": {"source": "hub", "table": "code.csv", "effort": "Reasoning level", "cost": "Cost"}}


def test_a_csv_zip_source_becomes_points_with_efforts_scales_and_collapsed_harnesses(tmp_path):
    config, _ = setup(tmp_path, {"code": CODE})
    fetch = Fetch(**{"https://hub_test": zipped(code__csv=HUB_TABLE, meta__csv=META)})
    report = import_sources(config, load_scores(config), fetch, now=NOW)
    assert report.errors == {}
    assert (report.sources["hub"].rows, report.sources["hub"].skipped, report.sources["hub"].points) == (7, 2, 4)
    block = imported(tmp_path)["benchmarks"]["code"]
    assert block["baseline"] == 0.25 and block["fetched_at"] == "2026-09-25T12:00:00+00:00"
    points = {(p["model"], p["effort"]): p for p in block["points"]}
    # The column says medium; the id's `_max` is still removed from the key. Two harnesses: the median.
    assert points[("acme-large_max", "medium")]["score"] == pytest.approx(60.0)
    assert points[("acme-large_max", "medium")]["cost_usd"] == pytest.approx(3.0)
    # A column value that does not resolve falls through to the suffix.
    assert ("acme-large_xhigh", "xhigh") in points
    # Not an effort, so part of the name; the column resolves through `efforts.labels`.
    assert ("zeta-1_promax", "xhigh") in points
    # A bare row where the same source has efforts for that key means "not stated".
    assert points[("acme-small", None)]["score"] == pytest.approx(30.0)
    scores = load_scores(config)
    assert set(scores.readings.get("code", {})) <= {"acme-large", "acme-small", "zeta-1-promax"}


def test_a_null_effort_becomes_unknown_only_beside_tier_matchable_efforts(tmp_path):
    table = [
        {"Model version": "acme-large", "Score": "0.5"}, {"Model version": "acme-large_high", "Score": "0.6"},
        {"Model version": "acme-small", "Score": "0.3"}, {"Model version": "acme-small_32K", "Score": "0.35"},
        {"Model version": "acme-small_unknown", "Score": "0.32"},
    ]
    bench = {"description": "d", "requirements": {"reasoning": 1.0},
             "data": {"source": "hub", "table": "t.csv", "score": "Score"}}
    config, _ = setup(tmp_path, {"code": bench})
    import_sources(config, load_scores(config), Fetch(**{"https://hub_test": zipped(t__csv=table, meta__csv=META)}),
                   now=NOW)
    efforts = {(p["model"], p["effort"]) for p in imported(tmp_path)["benchmarks"]["code"]["points"]}
    assert ("acme-large", "unknown") in efforts  # beside `high`
    assert ("acme-small", None) in efforts  # budgets and unknown do not count


def test_a_json_source_is_read_over_its_pages_with_the_key_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_BENCH_KEY", "secret")
    page1 = json.dumps({"data": [{"name": "Acme Large (Adaptive Reasoning, Extra High Effort)",
                                  "evaluations": {"index": 55.0}, "cost": {"per_task": {"total": 2.5}}}],
                        "more": True}).encode()
    page2 = json.dumps({"data": [{"name": "Zeta 1 (low)", "evaluations": {"index": None}, "cost": None},
                                 {"name": "Acme Small (high)", "evaluations": {"index": 30.0}, "cost": None}],
                        "more": False}).encode()
    bench = {"description": "an index", "requirements": {"reasoning": 1.0}, "scale": 0.01,
             "data": {"source": "api", "score": "evaluations.index", "cost": "cost.per_task.total"}}
    config, _ = setup(tmp_path, {"index": bench})
    fetch = Fetch(**{"https://api_test": [page1, page2]})
    report = import_sources(config, load_scores(config), fetch, now=NOW)
    assert [c[0] for c in fetch.calls] == ["https://api.test/models?page=1", "https://api.test/models?page=2"]
    assert fetch.calls[0][1]["x-api-key"] == "secret"
    assert report.sources["api"].skipped == 1  # no score
    points = {(p["model"], p["effort"]): p for p in imported(tmp_path)["benchmarks"]["index"]["points"]}
    assert points[("Acme Large (Adaptive Reasoning, Extra High Effort)", "xhigh")]["score"] == pytest.approx(55.0)
    assert points[("Acme Large (Adaptive Reasoning, Extra High Effort)", "xhigh")]["cost_usd"] == 2.5
    assert load_scores(config).key("Acme Large (Adaptive Reasoning, Extra High Effort)") == "acme-large"


def test_a_missing_key_skips_only_its_source_and_a_missing_column_fails_only_its_benchmark(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_BENCH_KEY", raising=False)
    broken = {**CODE, "data": {**CODE["data"], "cost": "Price"}}
    index = {"description": "i", "requirements": {"reasoning": 1.0}, "data": {"source": "api", "score": "x"}}
    lore = {"description": "l", "requirements": {"niche": 1.0},
            "data": {"source": "hub", "table": "code.csv", "score": "Score"}}
    config, _ = setup(tmp_path, {"code": broken, "index": index, "lore": lore})
    report = import_sources(config, load_scores(config),
                            Fetch(**{"https://hub_test": zipped(code__csv=HUB_TABLE, meta__csv=META)}), now=NOW)
    assert "TEST_BENCH_KEY" in report.errors["api"] and "Price" in report.errors["code"]
    assert set(imported(tmp_path)["benchmarks"]) == {"lore"}


def test_a_failed_import_keeps_the_previous_points(tmp_path):
    config, _ = setup(tmp_path, {"code": CODE})
    import_sources(config, load_scores(config),
                   Fetch(**{"https://hub_test": zipped(code__csv=HUB_TABLE, meta__csv=META)}), now=NOW)
    before = imported(tmp_path)["benchmarks"]["code"]
    report = import_sources(config, load_scores(config), Fetch(), now=NOW)
    assert "hub" in report.errors
    assert imported(tmp_path)["benchmarks"]["code"] == before
    assert imported(tmp_path)["sources"]["hub"]["status"] == "failed"


def test_where_keeps_only_matching_rows(tmp_path):
    bench = {**CODE, "data": {**CODE["data"], "where": {"Harness": "h2"}}}
    config, _ = setup(tmp_path, {"code": bench})
    import_sources(config, load_scores(config),
                   Fetch(**{"https://hub_test": zipped(code__csv=HUB_TABLE, meta__csv=META)}), now=NOW)
    assert [p["score"] for p in imported(tmp_path)["benchmarks"]["code"]["points"]] == [pytest.approx(70.0)]


def test_an_alias_maps_a_display_name_and_two_rows_on_one_key_collapse(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_BENCH_KEY", "k")
    doc = json.dumps({"data": [{"name": "Zeta 4B (Reasoning)", "i": 20.0},
                               {"name": "Zeta 4B (Non-reasoning)", "i": 10.0}], "more": False}).encode()
    bench = {"description": "i", "requirements": {"reasoning": 1.0}, "scale": 0.01,
             "data": {"source": "api", "score": "i"}}
    config, _ = setup(tmp_path, {"index": bench},
                      aliases={"Zeta 4B (Reasoning)": "zeta:4b", "Zeta 4B (Non-reasoning)": "zeta:4b"})
    import_sources(config, load_scores(config), Fetch(**{"https://api_test": doc}), now=NOW)
    points = imported(tmp_path)["benchmarks"]["index"]["points"]
    assert len(points) == 1 and points[0]["score"] == pytest.approx(15.0) and points[0]["effort"] is None
    assert load_scores(config).key("zeta:4b") == "zeta-4b"


def test_paging_stops_at_the_cap(tmp_path, monkeypatch):
    from llm_router.scores_import import MAX_PAGES

    monkeypatch.setenv("TEST_BENCH_KEY", "k")
    endless = json.dumps({"data": [{"name": "acme-large (high)", "i": 50.0}], "more": True}).encode()
    bench = {"description": "i", "requirements": {"reasoning": 1.0}, "scale": 0.01,
             "data": {"source": "api", "score": "i"}}
    config, _ = setup(tmp_path, {"index": bench})
    fetch = Fetch(**{"https://api_test": [endless] * (MAX_PAGES + 5)})
    import_sources(config, load_scores(config), fetch, now=NOW)
    assert len(fetch.calls) == MAX_PAGES


def test_a_csv_zip_and_a_json_document_yield_the_same_points_for_the_same_data(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_BENCH_KEY", "k")
    rows = [("acme-large_high", 0.6, 2.0), ("acme-small_low", 0.3, 0.5)]
    table = [{"Model version": m, "Score": str(s), "Cost": str(c)} for m, s, c in rows]
    doc = json.dumps({"data": [{"name": m, "score": s, "cost": c} for m, s, c in rows], "more": False}).encode()
    benches = {
        "a": {"description": "d", "requirements": {"reasoning": 1.0},
              "data": {"source": "hub", "table": "t.csv", "score": "Score", "cost": "Cost"}},
        "b": {"description": "d", "requirements": {"reasoning": 1.0},
              "data": {"source": "api", "score": "score", "cost": "cost"}},
    }
    config, _ = setup(tmp_path, benches)
    fetch = Fetch(**{"https://hub_test": zipped(t__csv=table, meta__csv=META), "https://api_test": doc})
    import_sources(config, load_scores(config), fetch, now=NOW)
    blocks = imported(tmp_path)["benchmarks"]
    assert blocks["a"]["points"] == blocks["b"]["points"]


def test_a_json_path_that_is_in_no_row_fails_its_benchmark(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_BENCH_KEY", "k")
    doc = json.dumps({"data": [{"name": "acme-large (high)", "i": 50.0}], "more": False}).encode()
    bench = {"description": "i", "requirements": {"reasoning": 1.0},
             "data": {"source": "api", "score": "i", "cost": "price.per_task"}}
    config, _ = setup(tmp_path, {"index": bench})
    report = import_sources(config, load_scores(config), Fetch(**{"https://api_test": doc}), now=NOW)
    assert "price.per_task" in report.errors["index"]


def test_an_alias_can_carry_the_effort_its_display_name_means_so_two_variants_stay_apart(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_BENCH_KEY", "k")
    doc = json.dumps({"data": [{"name": "Zeta 4B (Reasoning)", "i": 20.0},
                               {"name": "Zeta 4B (Non-reasoning)", "i": 10.0}], "more": False}).encode()
    bench = {"description": "i", "requirements": {"reasoning": 1.0}, "scale": 0.01,
             "data": {"source": "api", "score": "i"}}
    config, _ = setup(tmp_path, {"index": bench}, efforts={"labels": {"none": "none"}},
                      aliases={"Zeta 4B (Reasoning)": "zeta:4b", "Zeta 4B (Non-reasoning)": "zeta:4b (none)"})
    import_sources(config, load_scores(config), Fetch(**{"https://api_test": doc}), now=NOW)
    points = {p["effort"]: p["score"] for p in imported(tmp_path)["benchmarks"]["index"]["points"]}
    assert points == {None: pytest.approx(20.0), "none": pytest.approx(10.0)}
    scores = load_scores(config)
    assert {(scores.key(p.model), p.effort) for p in scores.points} == {("zeta-4b", None), ("zeta-4b", "none")}


def test_the_api_key_never_follows_a_redirect_to_another_host_or_to_http():
    import urllib.request

    from llm_router.scores_import import _PrivateHeaders

    handler = _PrivateHeaders({"x-api-key"})
    request = urllib.request.Request("https://api.test/models", headers={"x-api-key": "k", "User-Agent": "r"})
    same = handler.redirect_request(request, None, 302, "Found", {}, "https://API.test/v2/models")
    assert same.get_header("X-api-key") == "k"
    for url in ("https://other.test/models", "http://api.test/models", "https://api.test:8443/models"):
        moved = handler.redirect_request(request, None, 302, "Found", {}, url)
        assert moved.get_header("X-api-key") is None and moved.get_header("User-agent") == "r"


def test_a_keyed_source_over_plain_http_is_refused_before_any_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_BENCH_KEY", "k")
    bench = {"description": "i", "requirements": {"reasoning": 1.0}, "data": {"source": "api", "score": "i"}}
    config, _ = setup(tmp_path, {"index": bench}, sources={**SOURCES, "api": {**SOURCES["api"], "url": "http://api.test/m"}})
    fetch = Fetch()
    report = import_sources(config, load_scores(config), fetch, now=NOW)
    assert "https" in report.errors["api"] and fetch.calls == []


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_a_source_has_refresh_timeout_s_for_all_its_pages_and_fails_with_timed_out(tmp_path, monkeypatch):
    import llm_router.scores_import as scores_import

    monkeypatch.setenv("TEST_BENCH_KEY", "k")
    clock = Clock()
    monkeypatch.setattr(scores_import, "_clock", clock)
    page = json.dumps({"data": [{"name": "acme-large (high)", "i": 50.0}], "more": True}).encode()

    class Slow(Fetch):
        def __call__(self, url, headers, deadline):
            clock.now += 20.0  # each page takes 20 s; the source has 30
            return super().__call__(url, headers, deadline)

    bench = {"description": "i", "requirements": {"reasoning": 1.0}, "scale": 0.01, "data": {"source": "api", "score": "i"}}
    config, _ = setup(tmp_path, {"index": bench})
    fetch = Slow(**{"https://api_test": [page] * 10})
    report = import_sources(config, load_scores(config), fetch, now=NOW)
    assert fetch.deadlines[0] == 1000.0 + 30.0 and len(fetch.calls) == 2
    assert "timed out" in report.errors["api"] and "index" not in imported(tmp_path)["benchmarks"]


def test_a_trickling_response_is_cut_at_the_deadline(monkeypatch):
    import llm_router.scores_import as scores_import

    clock = Clock()
    monkeypatch.setattr(scores_import, "_clock", clock)

    class Trickle:
        def read(self, n=-1):
            clock.now += 10.0
            return b"x"

    with pytest.raises(TimeoutError):
        scores_import._read(Trickle(), deadline=1025.0)
    assert clock.now == 1030.0


def test_a_benchmark_that_yields_no_points_is_an_error_and_keeps_its_previous_points(tmp_path):
    config, _ = setup(tmp_path, {"code": CODE})
    import_sources(config, load_scores(config),
                   Fetch(**{"https://hub_test": zipped(code__csv=HUB_TABLE, meta__csv=META)}), now=NOW)
    before = imported(tmp_path)["benchmarks"]["code"]
    empty = [{**HUB_TABLE[0], "Score": ""}]
    report = import_sources(config, load_scores(config),
                            Fetch(**{"https://hub_test": zipped(code__csv=empty, meta__csv=META)}), now=NOW)
    assert "no point" in report.errors["code"]
    assert imported(tmp_path)["benchmarks"]["code"] == before


def test_an_unreadable_table_fails_only_its_benchmark(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("bad.csv", b"Model version,Score\n\xff\xfe,0.5\n")
        text = io.StringIO()
        csv.DictWriter(text, fieldnames=list(HUB_TABLE[0])).writeheader()
        csv.DictWriter(text, fieldnames=list(HUB_TABLE[0])).writerows(HUB_TABLE)
        archive.writestr("code.csv", text.getvalue())
        meta = io.StringIO()
        csv.DictWriter(meta, fieldnames=list(META[0])).writeheader()
        csv.DictWriter(meta, fieldnames=list(META[0])).writerows(META)
        archive.writestr("meta.csv", meta.getvalue())
    bad = {"description": "b", "requirements": {"reasoning": 1.0},
           "data": {"source": "hub", "table": "bad.csv", "score": "Score"}}
    config, _ = setup(tmp_path, {"code": CODE, "bad": bad})
    report = import_sources(config, load_scores(config), Fetch(**{"https://hub_test": buffer.getvalue()}), now=NOW)
    assert "UnicodeDecodeError" in report.errors["bad"]
    assert set(imported(tmp_path)["benchmarks"]) == {"code"}


def test_a_refresh_where_every_source_failed_is_retried_at_the_next_start(tmp_path):
    from datetime import timedelta

    from llm_router.scores_import import refresh

    config, _ = setup(tmp_path, {"code": CODE})
    import_sources(config, load_scores(config),
                   Fetch(**{"https://hub_test": zipped(code__csv=HUB_TABLE, meta__csv=META)}), now=NOW)
    later = NOW + timedelta(days=2)
    failing = Fetch()
    assert refresh(config, load_scores(config), failing, now=later)[0]
    assert imported(tmp_path)["imported_at"] == NOW.isoformat(timespec="seconds")  # not advanced
    refresh(config, load_scores(config), failing, now=later + timedelta(minutes=1))
    assert len(failing.calls) == 2  # still stale: tried again


def test_a_response_or_a_zip_member_over_the_size_cap_fails_with_an_error(tmp_path, monkeypatch):
    import llm_router.scores_import as scores_import

    monkeypatch.setattr(scores_import, "MAX_BYTES", 1000)

    class Endless:
        def read(self, n=-1):
            return b"x" * 300

    with pytest.raises(scores_import.ImportFailure, match="1000"):
        scores_import._read(Endless(), deadline=float("inf"))
    config, _ = setup(tmp_path, {"code": CODE})
    big = [{**HUB_TABLE[0], "Harness": "h" * 2000}]
    report = import_sources(config, load_scores(config),
                            Fetch(**{"https://hub_test": zipped(code__csv=big, meta__csv=META)}), now=NOW)
    assert "1000" in report.errors["code"]
