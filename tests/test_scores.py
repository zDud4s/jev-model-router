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


# ---------------------------------------------------------------- derivation
from llm_router.scores_derive import derive, effort_prior, evidence_for, line_for, profile_line, reading_at  # noqa: E402


def profile(caps, match: str = "acme-large"):
    return next(p for p in caps.profiles if p.match == match)


def card(caps, scores, model: str, effort: str | None, line=(2.0, 1.0), prof: str | None = None):
    return derive(caps, scores, (model,), effort, profile(caps, prof or model), line)


def test_effort_prior_is_todays_card_rule():
    caps = caps_with()
    levels, output = effort_prior(caps, profile(caps), "medium")
    assert levels == {"reasoning": pytest.approx(1.7), "niche": 2.0}  # only thinking requirements move
    assert output == 900  # 2000 x 0.45
    assert effort_prior(caps, profile(caps), None) == ({"reasoning": 2.0, "niche": 2.0}, 2000)


THREE = linked(pt("code", "acme-large", "high", 80), pt("code", "acme-small", "high", 40),
               pt("code", "zeta-1", "high", 60))


def test_a_model_above_average_ability_is_rated_above_a_and_one_below_it_below():
    caps, scores = scored(THREE)
    large, small = card(caps, scores, "acme-large", "high"), card(caps, scores, "acme-small", "high")
    assert large.levels["reasoning"] > 2.0 > small.levels["reasoning"]
    # A requirement no benchmark covers stays on the profile.
    assert large.levels["niche"] == 2.0 and large.source["niche"] == "profile" and large.coverage["niche"] == 0.0
    assert large.coverage["reasoning"] > 0


def test_the_level_is_the_blend_of_the_line_and_the_profile_by_coverage():
    caps, scores = scored(THREE)
    ev = evidence_for(caps, scores, ("acme-large",), "high")
    c, t = ev.coverage["reasoning"], ev.ability["reasoning"]
    lam = c / (c + 0.25)
    expected = lam * min(3.0, max(0.0, 2.3 + 0.6 * t)) + (1 - lam) * 2.0
    assert card(caps, scores, "acme-large", "high", line=(2.3, 0.6)).levels["reasoning"] == pytest.approx(expected)


def test_effort_value_depends_on_the_requirement():
    # Medium beats high where the work is `reasoning`; high beats medium on `niche`.
    caps, scores = scored(linked(
        pt("code", "acme-large", "medium", 70), pt("code", "acme-large", "high", 60), pt("code", "zeta-1", "high", 50),
        pt("code", "acme-small", "high", 40),
        pt("lore", "acme-large", "medium", 40), pt("lore", "acme-large", "high", 60), pt("lore", "zeta-1", "high", 50),
        pt("lore", "acme-small", "high", 40),
    ))
    medium, high = card(caps, scores, "acme-large", "medium"), card(caps, scores, "acme-large", "high")
    assert medium.levels["reasoning"] > high.levels["reasoning"]
    assert high.levels["niche"] > medium.levels["niche"]


def test_an_unmeasured_effort_is_interpolated_between_measured_ones_and_never_extrapolated():
    caps, scores = scored(linked(pt("code", "acme-large", "low", 50), pt("code", "acme-large", "high", 70),
                                 pt("code", "zeta-1", "high", 60), pt("code", "acme-small", "high", 40)))
    low = reading_at(scores, "code", ("acme-large",), "low")
    high = reading_at(scores, "code", ("acme-large",), "high")
    medium = reading_at(scores, "code", ("acme-large",), "medium")
    assert medium == pytest.approx(((low[0] + high[0]) / 2, (low[1] + high[1]) / 2))
    assert reading_at(scores, "code", ("acme-large",), "xhigh") is None
    assert card(caps, scores, "acme-large", "xhigh").coverage["reasoning"] == 0.0


def test_a_tier_without_efforts_matches_only_null_points_and_unknown_matches_no_tier():
    caps, scores = scored(linked(pt("code", "acme-small", None, 80), pt("code", "acme-small", "high", 20),
                                 pt("code", "zeta-1", "high", 50), pt("code", "acme-large", "high", 60))
                          + [pt("code", "zeta-1", "unknown", 70)])
    assert reading_at(scores, "code", ("acme-small",), None) is not None
    assert reading_at(scores, "code", ("zeta-1",), None) is None
    assert reading_at(scores, "code", ("zeta-1",), "unknown") is not None  # it is read, but no tier asks for it


def test_a_vendor_only_model_counts_vendor_weight_times_an_independent_one():
    rows = lambda origin: linked(
        pt("code", "acme-large", "high", 70, origin=origin), pt("code", "acme-small", "high", 40, origin="independent"),
        pt("code", "zeta-1", "high", 55, origin="independent"))
    caps, vendor = scored(rows("vendor"))
    _, independent = scored(rows("independent"))
    ratio = (evidence_for(caps, vendor, ("acme-large",), "high").coverage["reasoning"]
             / evidence_for(caps, independent, ("acme-large",), "high").coverage["reasoning"])
    assert ratio == pytest.approx(0.5)


def test_a_score_near_the_ceiling_carries_less_evidence_than_one_near_the_middle():
    caps, scores = scored(linked(pt("code", "acme-large", "high", 99), pt("code", "acme-small", "high", 50),
                                 pt("code", "zeta-1", "high", 60)))
    top = evidence_for(caps, scores, ("acme-large",), "high").coverage["reasoning"]
    mid = evidence_for(caps, scores, ("acme-small",), "high").coverage["reasoning"]
    assert top < 0.1 * mid


def test_the_baseline_moves_chance_to_zero():
    benches = {**BENCHES, "code": {**BENCHES["code"], "baseline": 0.25}}
    _, scores = scored(linked(pt("code", "acme-large", "high", 25), pt("code", "zeta-1", "high", 62.5, origin="independent"),
                              pt("code", "acme-small", "high", 40)), benches)
    # 62.5% with a 25% baseline is 50% of the way to full marks: u = 4 x 0.5 x 0.5.
    assert scores.readings["code"]["zeta-1"]["high"].u == pytest.approx(1.0)


def test_a_point_is_matched_by_any_of_the_models_keys():
    caps, scores = scored(THREE + linked(pt("code", "acme-large-alias", "high", 80)))
    ev = evidence_for(caps, scores, ("acme-large-2026", "acme-large-alias"), "high")
    assert ev.coverage["reasoning"] > 0


def test_an_unmatched_point_informs_the_scale_when_its_model_links_benchmarks():
    caps, before = scored(THREE)
    _, crowd = scored(THREE + [pt("code", "ghost-9", "high", 10)])  # on one benchmark: no effect
    _, linking = scored(THREE + linked(pt("code", "ghost-9", "high", 10)))
    at = lambda s: evidence_for(caps, s, ("acme-large",), "high").ability["reasoning"]
    assert at(crowd) == pytest.approx(at(before))
    assert at(linking) != pytest.approx(at(before))


def test_the_blend_is_the_profile_at_no_coverage_and_moves_continuously_toward_the_evidence():
    levels = []
    for weight in (0.0, 1e-6, 0.25, 1.0):
        benches = {**BENCHES, "code": {"description": "d", "requirements": {"reasoning": weight}}}
        caps, scores = scored(THREE, benches)
        levels.append(card(caps, scores, "acme-large", "high").levels["reasoning"])
    assert levels[0] == 2.0  # exactly today's card
    assert levels[1] == pytest.approx(2.0, abs=1e-4)  # no jump as a weight leaves zero
    assert levels[0] < levels[2] < levels[3]


def test_a_measured_effort_gap_beats_the_rule_only_when_it_is_larger_than_c0_over_c_times_the_shift():
    # c0 = 0.25, C ~ 1, rule shift high - medium = +0.3: the threshold is about 0.075
    # levels. 0.40 logits is clearly above it; 0.02 clearly below.
    def medium_minus_high(gap_logits: float) -> float:
        high = 0.6
        medium = 1 / (1 + math.exp(-(logit(high) + gap_logits)))
        caps, scores = scored(linked(
            pt("code", "acme-large", "medium", medium * 100), pt("code", "acme-large", "high", high * 100),
            pt("code", "zeta-1", "high", 50), pt("code", "acme-small", "high", 40),
        ))
        return card(caps, scores, "acme-large", "medium").levels["reasoning"] - \
            card(caps, scores, "acme-large", "high").levels["reasoning"]

    assert medium_minus_high(0.40) > 0  # the measurement wins
    assert medium_minus_high(0.02) < 0  # within noise: the rule decides


def test_the_profile_line_keeps_the_profiles_mean_and_spread():
    pairs = [(1.0, -1.0, 1.0), (2.0, 0.0, 1.0), (3.0, 1.0, 1.0), (2.5, 2.0, 1.0)]
    a, k = profile_line(pairs)
    levels = [a + k * x for _, x, _ in pairs]
    mean = lambda v: sum(v) / len(v)
    sd = lambda v: math.sqrt(mean([(x - mean(v)) ** 2 for x in v]))
    assert mean(levels) == pytest.approx(mean([p for p, _, _ in pairs]))
    assert sd(levels) == pytest.approx(sd([p for p, _, _ in pairs]))
    assert profile_line(pairs[:2]) == (2.0, 1.0)  # too few pairs


def test_the_line_takes_fitted_offsets_and_an_explicit_config_value_wins():
    tiers = lambda caps: [((m,), "high", profile(caps, m)) for m in ("acme-large", "acme-small")] + [
        (("zeta-1",), "high", profile(caps, "zeta-*"))]
    caps, scores = scored(THREE)
    a0, k0 = line_for(caps, scores, tiers(caps))
    caps, fitted = scored(THREE, sidecar={"fit": {"delta_a": 0.2, "k_ratio": 0.5}})
    assert line_for(caps, fitted, tiers(caps)) == pytest.approx((a0 + 0.2, k0 * 0.5))
    caps, fixed = scored(THREE, sidecar={"fit": {"delta_a": 0.2}}, a=1.5, k=2.0)
    assert line_for(caps, fixed, tiers(caps)) == (1.5, 2.0)


def test_a_fitted_scale_applies_only_while_the_benchmarks_words_are_unchanged():
    caps, scores = scored(THREE)
    digest = content_hash(scores.benchmarks["code"], caps.requirements)
    base = evidence_for(caps, scores, ("acme-large",), "high").coverage["reasoning"]
    _, fitted = scored(THREE, sidecar={"fit": {"scales": {"code": {"scale": 0.5, "hash": digest}}}})
    assert evidence_for(caps, fitted, ("acme-large",), "high").coverage["reasoning"] == pytest.approx(base / 2)
    changed = {**BENCHES, "code": {"description": "fix code", "requirements": {"reasoning": 0.9}}}
    _, stale = scored(THREE, changed, sidecar={"fit": {"scales": {"code": {"scale": 0.5, "hash": digest}}}})
    assert evidence_for(caps, stale, ("acme-large",), "high").coverage["reasoning"] == pytest.approx(base * 0.9)


def test_output_tokens_follow_the_published_cost_ratios_and_fall_back_to_the_rule():
    caps, scores = scored(linked(
        pt("code", "acme-large", "medium", 50, cost_usd=1.0), pt("code", "acme-large", "high", 55, cost_usd=2.0),
        pt("lore", "acme-large", "medium", 50, cost_usd=0.5), pt("lore", "acme-large", "high", 55, cost_usd=2.0),
        # a pair with a missing cost is left out of the median
        pt("code", "zeta-1", "high", 50), pt("lore", "zeta-1", "high", 50),
    ))
    assert card(caps, scores, "acme-large", "medium").output_tokens == 750  # 2000 x median(0.5, 0.25)
    assert card(caps, scores, "acme-large", "xhigh").output_tokens == 3200  # the rule's x1.6
    assert card(caps, scores, "acme-large", None).output_tokens == 2000


def test_costs_count_for_output_tokens_even_on_a_benchmark_off_the_shared_scale():
    caps, scores = scored([pt("lore", "acme-large", "medium", 50, cost_usd=1.0),
                           pt("lore", "acme-large", "high", 55, cost_usd=4.0)])
    assert scores.links["lore"] == 0  # unlinked: no reading, but the costs are still published ratios
    assert card(caps, scores, "acme-large", "medium").output_tokens == 500  # 2000 x 0.25


# ---------------------------------------------------------------- the imported file
HUB = {"hub": {"format": "json", "url": "https://hub_test", "model": "name", "origin": "independent"}}
HUBBED = {**BENCHES, "code": {**BENCHES["code"], "data": {"source": "hub"}}}
# Every node links through `base`, so `code` is on the scale whatever the import holds.
AROUND = linked(pt("code", "zeta-1", "high", 60, origin="independent"),
                pt("code", "acme-small", "high", 40, origin="independent")) + [
    pt("base", "acme-large", "high", 50, origin="independent")]


def imported(rows: list[dict[str, Any]], bench: str = "code", **meta: Any) -> dict[str, Any]:
    return {"imported_at": "2026-09-25T00:00:00+00:00",
            "benchmarks": {bench: {"source": "hub", "points": rows, **meta}}}


def from_hub(rows, benchmarks=None, **meta):
    return scored(AROUND, benchmarks or HUBBED, imported=imported(rows, **meta), extra={"sources": HUB})


def test_imported_bounds_that_divide_by_zero_or_invert_the_scale_are_dropped_with_an_error():
    rows = [{"model": "acme-large", "effort": "high", "score": 62.5}]
    _, scores = from_hub(rows, baseline=0.5, ceiling=0.5)
    assert len(scores.errors) == 1 and "code" in scores.errors[0] and "baseline" in scores.errors[0]
    assert scores.readings["code"]["acme-large"]["high"].u == pytest.approx(4 * 0.625 * 0.375)  # 0 and 1
    # A curated baseline and an imported ceiling below it: the imported bounds go, the curated one stays.
    benches = {**HUBBED, "code": {**HUBBED["code"], "baseline": 0.25}}
    _, scores = from_hub(rows, benches, ceiling=0.2)
    assert len(scores.errors) == 1
    assert scores.readings["code"]["acme-large"]["high"].u == pytest.approx(1.0)  # (0.625 - 0.25) / 0.75 = 0.5
    _, scores = from_hub(rows, baseline=float("nan"))
    assert len(scores.errors) == 1 and scores.readings["code"]["acme-large"]["high"].u == pytest.approx(0.9375)


def test_a_curated_effort_is_resolved_like_an_imported_one_and_an_unknown_label_is_kept():
    _, scores = scored([pt("code", "m", "High", 50), pt("code", "n", "extra  high", 50), pt("code", "o", "turbo", 50)],
                       extra={"efforts": {"labels": {"extra high": "xhigh"}}})
    assert {p.model: p.effort for p in scores.points} == {"m": "high", "n": "xhigh", "o": "turbo"}


def test_a_benchmark_group_that_shares_no_model_with_the_main_scale_counts_for_nothing():
    from llm_router.scores_derive import startup_lines

    benches = {**BENCHES, "far": {"description": "d", "requirements": {"niche": 1.0}},
               "away": {"description": "d", "requirements": {}}}
    island = [pt(b, f"q{i}", "high", 30 + 10 * i, origin="independent") for b in ("far", "away") for i in range(2)]
    caps, scores = scored(THREE + island, benches)
    assert scores.links["far"] == 2 and scores.rho["far"] == 0.0 and scores.rho["code"] == 1.0
    assert scores.scale.detached == ("away", "far")
    assert evidence_for(caps, scores, ("q1",), "high").coverage["niche"] == 0.0
    assert "benchmarks: not linked to the main scale (no model shared with it), counting for nothing: away, far" \
        in startup_lines(scores, caps, {})


def test_an_unreadable_source_status_in_the_imported_file_is_an_error_about_that_file():
    caps = caps_with()
    raw = {"sources": HUB, "benchmarks": copy.deepcopy(HUBBED), "points": AROUND}
    body = {**imported([{"model": "acme-large", "effort": "high", "score": 50}]), "sources": {"hub": "done"}}
    scores = parse_scores(raw, caps, imported=body, imported_path="b.imported.json")
    assert len(scores.errors) == 1 and scores.errors[0].startswith("b.imported.json: ")
    assert scores.imported_status == {} and len(scores.points) == len(AROUND) + 1


def test_a_bad_imported_row_drops_its_benchmarks_import_only_and_non_finite_numbers_are_skipped():
    caps = caps_with()
    benches = {**HUBBED, "lore": {**BENCHES["lore"], "data": {"source": "hub"}}}
    raw = {"sources": HUB, "benchmarks": benches, "points": AROUND}
    body = {"benchmarks": {
        "code": {"source": "hub", "points": [{"model": "acme-large", "effort": "high", "score": 50},
                                             {"model": "zeta-1", "effort": "high"}]},
        "lore": {"source": "hub", "points": [
            {"model": "acme-large", "effort": "high", "score": 50, "cost_usd": float("inf")},
            {"model": "zeta-1", "effort": "high", "score": float("nan")},
            {"model": "acme-small", "effort": "high", "score": 40, "cost_usd": 2.0}]}}}
    scores = parse_scores(raw, caps, imported=body, imported_path="b.imported.json")
    assert len(scores.errors) == 1 and "'code'" in scores.errors[0]
    held = {(p.benchmark, p.model): p.cost_usd for p in scores.points if p.imported}
    assert held == {("lore", "acme-large"): None, ("lore", "acme-small"): 2.0}
