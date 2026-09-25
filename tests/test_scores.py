"""Benchmark evidence: measured points per (model, effort) become card levels.

Every model id, benchmark and source here is invented: the code must work for
models that do not exist yet, so the tests never lean on one that does.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest

from llm_router.config import ConfigError, parse_config

from conftest import BASE_CONFIG

REQUIREMENTS = {
    "reasoning": "The state is a task packet. Does it need deep reasoning?",
    "niche": "The state is a task packet. Does it need niche knowledge?",
}


def raw_config(**caps: Any) -> dict[str, Any]:
    raw = copy.deepcopy(BASE_CONFIG)
    raw["tiers"]["judge"] = {"backend": "jev", "model": "jev-latest"}
    raw["router"] = {
        "kind": "capabilities",
        "default_tier": "mid",
        "capabilities": {
            "jev_tier": "judge",
            "requirements": REQUIREMENTS,
            "discover": {"cli": {"backend": "claude_cli", "base_url": "C:/bin/cli.exe"}},
            "profiles": [
                {"match": "acme-large", "level": 2.0, "list_prices": {"input": 5.0, "output": 25.0}},
                {"match": "acme-small", "level": 1.0, "list_prices": {"input": 1.0, "output": 5.0}},
                {"match": "zeta-*", "level": 1.5, "list_prices": {"input": 2.0, "output": 10.0}},
            ],
            "fallback_profile": {"level": 0.5, "list_prices": {"input": 10.0, "output": 50.0}},
            "benchmarks": {"path": "benchmarks.yaml"},
            **caps,
        },
    }
    return raw


def logit(s: float) -> float:
    return math.log(s / (1 - s))


# ---------------------------------------------------------------- config
def test_the_benchmarks_block_is_optional_and_has_defaults():
    b = parse_config(raw_config()).router.capabilities.benchmarks
    assert (b.path, b.a, b.k, b.profile_weight, b.vendor_weight, b.refresh_hours, b.refresh_timeout_s) == (
        "benchmarks.yaml", None, None, 0.25, 0.5, 24.0, 30.0)
    raw = raw_config()
    del raw["router"]["capabilities"]["benchmarks"]
    assert parse_config(raw).router.capabilities.benchmarks is None


@pytest.mark.parametrize(
    "block, message",
    [
        ({"a": 1.0}, "path is required"),
        ({"path": "b.yaml", "a": 3.5}, "a must be in"),
        ({"path": "b.yaml", "k": 0.0}, "k must be in"),
        ({"path": "b.yaml", "profile_weight": 0}, "profile_weight must be positive"),
        ({"path": "b.yaml", "vendor_weight": 1.5}, "vendor_weight must be in"),
        ({"path": "b.yaml", "refresh_hours": -1}, "refresh_hours must be positive"),
        ({"path": "b.yaml", "a": "high"}, "could not convert"),
        ({"path": "b.yaml", "lambda": 1}, "unknown fields"),
    ],
    ids=["no-path", "a", "k", "profile-weight", "vendor-weight", "refresh", "not-a-number", "unknown"],
)
def test_a_benchmarks_block_that_would_mislead_is_refused(block, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(raw_config(benchmarks=block))


# ---------------------------------------------------------------- the file
from llm_router.scores import EffortRules, KeyRules, content_hash, date_collisions, load_scores, parse_scores, split_model  # noqa: E402


def caps_with(**settings: Any):
    return parse_config(raw_config(benchmarks={"path": "b.yaml", **settings})).router.capabilities


# `base` measures nothing the router asks about. It is there so that models are
# linked across two benchmarks, which is what places a benchmark on the shared scale.
BENCHES = {
    "code": {"description": "fix code", "requirements": {"reasoning": 1.0}},
    "lore": {"description": "know things", "requirements": {"niche": 1.0}},
    "base": {"description": "general", "requirements": {}},
}


def pt(bench: str, model: str, effort: str | None, score: float, **extra: Any) -> dict[str, Any]:
    return {"benchmark": bench, "model": model, "effort": effort, "score": score, **extra}


def linked(*points: dict[str, Any], base: float = 50.0) -> list[dict[str, Any]]:
    """The points, plus one `base` point per (model, effort) among them, so every node links two benchmarks."""
    nodes = sorted({(p["model"], p["effort"]) for p in points}, key=str)
    return list(points) + [pt("base", m, e, base, origin="independent") for m, e in nodes]


def scored(points, benchmarks=None, *, sidecar=None, imported=None, extra=None, **settings):
    caps = caps_with(**settings)
    raw = {"benchmarks": copy.deepcopy(benchmarks or BENCHES), "points": points, **(extra or {})}
    return caps, parse_scores(raw, caps, sidecar=sidecar, imported=imported)


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"points": [pt("nope", "m", "high", 50)]}, "unknown benchmark"),
        ({"points": [pt("code", "m", 3, 50)]}, "effort must be null or a string"),
        ({"points": [pt("code", "m", "high", 120)]}, r"score must be a percentage in \[0, 100\]"),
        ({"points": [pt("code", "m", "high", 50, cost_usd=-1)]}, "cost_usd cannot be negative"),
        ({"points": [pt("code", "m", "high", 50, origin="rumour")]}, "origin must be one of"),
        ({"benchmarks": {"code": {"description": "d", "requirements": {"speed": 1.0}}}}, "unknown requirement"),
        ({"benchmarks": {"code": {"description": "d", "requirements": {"reasoning": 1.5}}}}, r"must be in \[0, 1\]"),
        ({"benchmarks": {"code": {}}}, "needs a description"),
        ({"benchmarks": {"code": {"description": "d", "baseline": 0.5, "ceiling": 0.5}}}, "baseline must be below"),
        ({"benchmarks": {"code": {"description": "d", "data": {"source": "nope"}}}}, "unknown source"),
        ({"sources": {"s": {"format": "csv_zip", "url": "u", "model": "m", "origin": "independent"}},
          "benchmarks": {"code": {"description": "d", "data": {"source": "s"}}}}, "needs a 'table'"),
        ({"sources": {"s": {"format": "xml", "url": "u", "model": "m", "origin": "independent"}}}, "format must be"),
        ({"sources": {"s": {"format": "json", "url": "u", "model": "m", "origin": "independent",
                            "paging": {"param": "page"}}}}, "paging needs both"),
        ({"efforts": {"patterns": ["("]}}, "not a valid regular expression"),
    ],
    ids=["benchmark", "effort", "score", "cost", "origin", "requirement", "weight", "empty", "baseline",
         "data-source", "table", "format", "paging", "regex"],
)
def test_a_malformed_benchmarks_file_is_refused_naming_the_entry(raw, message):
    caps = caps_with()
    body = {"benchmarks": copy.deepcopy(BENCHES), "points": [], **raw}
    with pytest.raises(ConfigError, match=message):
        parse_scores(body, caps)


def test_an_effort_outside_the_known_ones_is_not_an_error():
    _, scores = scored([pt("code", "m", "turbo", 50)])
    assert scores.points[0].effort == "turbo"


def test_load_scores_reads_the_three_files_and_treats_broken_generated_ones_as_empty(tmp_path):
    path = tmp_path / "b.yaml"
    path.write_text("benchmarks:\n  code: {description: fix code, requirements: {reasoning: 1.0}}\n"
                    "points:\n  - {benchmark: code, model: m, effort: high, score: 50, date: 2026-09-22}\n",
                    encoding="utf-8")
    config = parse_config(raw_config(benchmarks={"path": str(path)}))
    scores = load_scores(config)
    assert scores.points[0].date == "2026-09-22"  # YAML's date, kept as text
    assert scores.sidecar_path == str(tmp_path / "b.derived.json")
    assert scores.imported_path == str(tmp_path / "b.imported.json") and scores.errors == ()
    (tmp_path / "b.derived.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "b.imported.json").write_text("[]", encoding="utf-8")
    broken = load_scores(config)
    assert len(broken.errors) == 2 and broken.jev == {} and len(broken.points) == 1


def test_load_scores_is_none_without_a_block_and_an_error_for_a_missing_file(tmp_path):
    raw = raw_config()
    del raw["router"]["capabilities"]["benchmarks"]
    assert load_scores(parse_config(raw)) is None
    with pytest.raises(ConfigError, match="not found"):
        load_scores(parse_config(raw_config(benchmarks={"path": str(tmp_path / "none.yaml")})))


# ---------------------------------------------------------------- matching
EFFORTS_RULES = EffortRules(labels={"extra high": "xhigh", "none": "none"}, patterns=(r"^\d+k( thinking)?$",))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("acme-9.1-sol_xhigh", ("acme-9.1-sol", "xhigh")),
        ("acme-9.1-sol_promax", ("acme-9.1-sol_promax", None)),  # not an effort: part of the name
        ("Zeta-1_8B", ("Zeta-1_8B", None)),
        ("Acme Large 5.1 (Adaptive Reasoning, Extra High Effort)", ("Acme Large 5.1", "xhigh")),
        ("Acme Large 5.1 (Extra High)", ("Acme Large 5.1", "xhigh")),  # whole text before single words
        ("acme-small (16K thinking)", ("acme-small", "16k thinking")),
        ("acme-small_32K", ("acme-small", "32k")),
        ("Zeta 4B (Non-reasoning)", ("Zeta 4B (Non-reasoning)", None)),
        ("acme-large_unknown", ("acme-large", "unknown")),
        ("acme-large_none", ("acme-large", "none")),
    ],
)
def test_only_text_that_resolves_to_an_effort_is_split_off(text, expected):
    assert split_model(text, EFFORTS_RULES) == expected


def test_one_key_rule_matches_ids_display_names_prefixes_and_dates():
    rules = KeyRules(date_suffixes=(r"[-_]\d{8}$",))
    key = lambda text: rules.key(text, EFFORTS_RULES)
    assert key("acme-large-20260101_low") == key("Acme Large (low)") == key("acme-large-20260101") == "acme-large"
    assert key("host/acme-9.1-sol_xhigh") == key("ACME 9.1 Sol (xhigh)") == "acme-9-1-sol"
    assert key("opt-1.3b") != key("opt-13b")
    assert key("acme-9.1-sol_promax") == "acme-9-1-sol-promax"


def test_two_dated_snapshots_that_would_share_a_key_keep_their_dates():
    rules = KeyRules(date_suffixes=(r"[-_]\d{8}$",))
    kept = date_collisions(["acme-3-20250101", "acme-3-20250601_high", "zeta-1-20250101"], rules, EFFORTS_RULES)
    assert kept == {"acme-3"}
    rules = KeyRules(rules.date_suffixes, keep_dates=kept)
    assert rules.key("acme-3-20250101", EFFORTS_RULES) != rules.key("acme-3-20250601", EFFORTS_RULES)
    assert rules.key("zeta-1-20250101", EFFORTS_RULES) == "zeta-1"


def test_an_alias_maps_a_display_name_that_orders_words_differently():
    rules = KeyRules(aliases={"large acme (reasoning)": "acme-large"})
    assert rules.key("Large Acme (Reasoning)", EFFORTS_RULES) == "acme-large"


# ---------------------------------------------------------------- pooling
def test_duplicates_resolve_by_origin_then_exactness_then_date_then_order():
    _, scores = scored([
        pt("code", "m", "high", 80, origin="vendor", date="2026-09-01"),
        pt("code", "m", "high", 60, origin="independent", date="2026-01-01"),
        pt("code", "m", "low", 10, approx=True, date="2026-02-02", origin="independent"),
        pt("code", "m", "low", 20, date="2026-02-02", origin="independent"),
        pt("code", "m", "medium", 30, date="2026-03-03", origin="independent"),
        pt("code", "m", "medium", 40, date="2026-03-03", origin="independent"),
    ])
    assert {(p.effort, p.score) for p in scores.points} == {("high", 60), ("low", 20), ("medium", 40)}


def test_an_independent_point_supersedes_the_models_vendor_curve_unless_no_tier_could_use_it():
    _, scores = scored([
        pt("code", "m", "low", 40), pt("code", "m", "high", 60),
        pt("code", "m", "max", 70, origin="independent"),
        pt("lore", "m", "low", 40), pt("lore", "m", "unknown", 55, origin="independent"),
    ])
    held = {(p.benchmark, p.effort) for p in scores.points}
    assert held == {("code", "max"), ("lore", "low"), ("lore", "unknown")}
    assert {(p.benchmark, p.effort) for p in scores.superseded} == {("code", "low"), ("code", "high")}


def test_a_benchmark_counts_by_the_models_that_link_it():
    points = [pt("code", f"m{i}", "high", 40 + 10 * i) for i in range(3)]
    for n, expected in ((0, 0.0), (1, 1 / 3), (2, 2 / 3), (3, 1.0)):
        _, scores = scored(points + [pt("base", f"m{i}", "high", 50, origin="independent") for i in range(n)])
        assert scores.rho["code"] == pytest.approx(expected)


def test_unknown_and_budget_points_never_link_benchmarks():
    _, scores = scored([pt("code", "m", "unknown", 50), pt("base", "m", "unknown", 50),
                        pt("code", "m", "32k", 50), pt("base", "m", "32k", 50)])
    assert scores.links["code"] == 0
