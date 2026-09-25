"""Calibration: tiers the operator knows are (or are not) enough, turned into `miss_scale`.

Jev is never called: the router answers from a scripted `Ask`.
"""

from __future__ import annotations

import asyncio

import pytest

from llm_router.calibration import calibrate, load_anchors, with_scale, write_scale
from llm_router.config import ConfigError, parse_config
from llm_router.capabilities import CapabilityRouter

from test_capabilities import Ask, raw_config

HARD = {"reasoning": 0.9, "code": 0.9}


def run(anchors, needs=HARD, margin=0.0, **caps):
    config = parse_config(raw_config(**caps))
    router = CapabilityRouter(config, ask=Ask(needs))
    return config, router, asyncio.run(calibrate(config, anchors, router, margin=margin))


def test_a_scale_below_one_makes_every_tier_more_likely():
    config = parse_config(raw_config())
    router = CapabilityRouter(config, ask=Ask(HARD))
    assert router.success(HARD, "mid", scale=0.5) > router.success(HARD, "mid")
    assert router.success(HARD, "mid", scale=0.0) == pytest.approx(1.0)
    loose = CapabilityRouter(parse_config(raw_config(miss_scale=0.5)), ask=Ask(HARD))
    assert loose.success(HARD, "mid") == pytest.approx(router.success(HARD, "mid", scale=0.5))


def test_a_sufficient_anchor_lowers_the_scale_until_that_tier_reaches_the_target():
    config, router, (scale, results, conflicts) = run([{"task": "fix the bug", "sufficient": "mid"}])
    assert router.success(HARD, "mid") < 0.8  # the priors alone say no
    assert scale < 1.0 and not conflicts
    assert router.success(HARD, "mid", scale=scale) == pytest.approx(0.8, abs=1e-6)
    assert results[0].bounds["mid"][0] == "<="


def test_a_sufficient_tier_is_left_room_above_the_target():
    _, router, (scale, _, _) = run([{"task": "fix the bug", "sufficient": "mid"}], margin=0.05)
    assert router.success(HARD, "mid", scale=scale) == pytest.approx(0.85, abs=1e-6)


def test_anchors_never_make_the_router_trust_models_less_than_the_priors():
    _, _, (scale, _, _) = run([{"task": "trivial", "sufficient": "top"}], needs={"reasoning": 0.3, "code": 0.3})
    assert scale == 1.0


def test_an_insufficient_anchor_the_scale_breaks_is_reported_not_hidden():
    _, _, (scale, _, conflicts) = run([
        {"task": "fix the bug", "sufficient": "cheap"},
        {"task": "fix the bug", "insufficient": "mid"},
    ])
    assert conflicts and "mid" in conflicts[0]


def test_an_anchor_naming_an_unknown_tier_is_refused():
    with pytest.raises(ConfigError, match="not a carded tier"):
        run([{"task": "x", "sufficient": "nope"}])


def test_anchors_file_is_checked(tmp_path):
    path = tmp_path / "anchors.yaml"
    path.write_text("- task: x\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="no sufficient"):
        load_anchors(path)
    path.write_text("- task: x\n  sufficient: mid\n", encoding="utf-8")
    assert load_anchors(path)[0]["sufficient"] == "mid"


def test_the_scale_is_written_beside_miss_and_then_replaced(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("router:\n  capabilities:\n    # the table\n    miss: [0.9, 0.5, 0.2, 0.05]\n    floor: 0.2\n", encoding="utf-8")
    write_scale(path, 0.4321)
    text = path.read_text(encoding="utf-8")
    assert "    miss_scale: 0.432\n" in text and "# the table" in text
    write_scale(path, 0.5)
    text = path.read_text(encoding="utf-8")
    assert text.count("miss_scale:") == 1 and "miss_scale: 0.500" in text


def test_with_scale_changes_only_the_scale():
    config = parse_config(raw_config())
    scaled = with_scale(config, 0.3)
    assert scaled.router.capabilities.miss_scale == 0.3
    assert scaled.router.capabilities.cards == config.router.capabilities.cards


def test_because_must_name_a_requirement_and_sit_on_an_insufficient_anchor(tmp_path):
    path = tmp_path / "anchors.yaml"
    path.write_text("- task: x\n  sufficient: mid\n  because: reasoning\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="no insufficient tier"):
        load_anchors(path)
    path.write_text("- task: x\n  insufficient: mid\n  because: speed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown requirement 'speed'"):
        load_anchors(path, {"reasoning": "?", "code": "?"})
    path.write_text("- task: x\n  insufficient: mid\n  because: code\n", encoding="utf-8")
    assert load_anchors(path, {"reasoning": "?", "code": "?"})[0]["because"] == "code"
