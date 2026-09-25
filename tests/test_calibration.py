"""Calibration: tiers the operator knows are (or are not) enough, turned into `miss_scale`.

Jev is never called: the router answers from a scripted `Ask`.
"""

from __future__ import annotations

import asyncio

import pytest

from llm_router.calibration import Calibration, calibrate, load_anchors, with_caps, with_scale, write_scale
from llm_router.config import ConfigError, parse_config
from llm_router.capabilities import CapabilityRouter

from test_capabilities import Ask, raw_config

HARD = {"reasoning": 0.9, "code": 0.9}


def run(anchors, needs=HARD, margin=0.0, ask=None, **caps) -> tuple:
    config = parse_config(raw_config(**caps))
    router = CapabilityRouter(config, ask=ask or Ask(needs))
    return config, router, asyncio.run(calibrate(config, anchors, router, margin=margin))


def test_a_scale_below_one_makes_every_tier_more_likely():
    config = parse_config(raw_config())
    router = CapabilityRouter(config, ask=Ask(HARD))
    assert router.success(HARD, "mid", scale=0.5) > router.success(HARD, "mid")
    assert router.success(HARD, "mid", scale=0.0) == pytest.approx(1.0)
    loose = CapabilityRouter(parse_config(raw_config(miss_scale=0.5)), ask=Ask(HARD))
    assert loose.success(HARD, "mid") == pytest.approx(router.success(HARD, "mid", scale=0.5))


def test_a_sufficient_anchor_lowers_the_scale_until_that_tier_reaches_the_target():
    config, router, cal = run([{"task": "fix the bug", "sufficient": "mid"}])
    assert router.success(HARD, "mid") < 0.8  # the priors alone say no
    assert cal.scale < 1.0 and not cal.conflicts
    assert router.success(HARD, "mid", scale=cal.scale) == pytest.approx(0.8, abs=1e-6)
    assert cal.results[0].bounds["mid"][0] == "<="


def test_a_sufficient_tier_is_left_room_above_the_target():
    _, router, cal = run([{"task": "fix the bug", "sufficient": "mid"}], margin=0.05)
    assert router.success(HARD, "mid", scale=cal.scale) == pytest.approx(0.85, abs=1e-6)


def test_anchors_never_make_the_router_trust_models_less_than_the_priors():
    _, _, cal = run([{"task": "trivial", "sufficient": "top"}], needs={"reasoning": 0.3, "code": 0.3})
    assert cal.scale == 1.0


ANCHORS = [{"task": "fix the bug", "sufficient": "cheap"}, {"task": "fix it again", "insufficient": "mid"}]


def after(config, cal):
    return CapabilityRouter(with_caps(with_scale(config, cal.scale), cal.caps), ask=Ask(HARD))


def test_a_violated_insufficient_anchor_is_fixed_by_capping_the_weakest_link():
    config, router, cal = run(ANCHORS)
    assert not cal.conflicts
    (tier, family, req, old, new, how), = cal.results[1].capped
    assert (tier, family, req, old, how) == ("mid", "mid", "reasoning", 2.0, "weakest link")  # a tie: config order
    assert cal.caps == {"mid": {"reasoning": new}} and new < old
    fixed = after(config, cal)
    assert fixed.success(HARD, "mid") <= 0.8 and fixed.success(HARD, "cheap") >= 0.8 - 1e-9


def test_because_overrides_the_weakest_link():
    _, _, cal = run([ANCHORS[0], {**ANCHORS[1], "because": "code"}])
    assert cal.results[1].capped[0][2] == "code" and cal.results[1].capped[0][5] == "because"


def test_an_insufficient_tier_within_the_margin_of_the_target_is_a_violation():
    needs = {"reasoning": 0.69, "code": 0.69}  # mid lands near 0.77: short of 0.8, not of 0.75
    _, router, loose = run([{"task": "x", "insufficient": "mid"}], needs=needs, margin=0.0)
    assert 0.75 < router.success(needs, "mid") < 0.8
    assert not loose.results[0].capped
    _, _, strict = run([{"task": "x", "insufficient": "mid"}], needs=needs, margin=0.05)
    assert strict.results[0].capped and not strict.conflicts


class ByTask:
    """Answers with the needs of whichever task the packet carries."""

    def __init__(self, table: dict[str, dict[str, float]]) -> None:
        self.table = table

    async def __call__(self, packet: str, questions: dict[str, str]) -> dict[str, float]:
        needs = next(n for task, n in self.table.items() if task in packet)
        return {k: needs.get(k, 0.0) for k in questions}


def test_a_cap_that_would_break_a_sufficient_anchor_is_reported_and_not_kept():
    # mid is enough for the harder ALPHA and not for the easier BETA, on the same
    # requirement: only a cap on reasoning could satisfy BETA, and it breaks ALPHA.
    ask = ByTask({"ALPHA": {"reasoning": 0.9, "code": 0.2}, "BETA": {"reasoning": 0.6, "code": 0.2}})
    _, _, cal = run([{"task": "ALPHA job", "sufficient": "mid"}, {"task": "BETA job", "insufficient": "mid"}], ask=ask)
    assert cal.caps == {} and not cal.results[1].capped
    assert len(cal.conflicts) == 1 and "'ALPHA job'" in cal.conflicts[0] and "not capped" in cal.conflicts[0]


def test_a_task_jev_read_no_need_in_stays_a_conflict():
    _, _, cal = run([{"task": "x", "insufficient": "mid"}], needs={"reasoning": 0.1, "code": 0.1})
    assert cal.caps == {} and "no need" in cal.conflicts[0]


def test_a_second_run_over_the_same_anchors_changes_nothing():
    config, _, first = run(ANCHORS)
    capped_config = with_caps(with_scale(config, first.scale), first.caps)
    router = CapabilityRouter(capped_config, ask=Ask(HARD))
    second = asyncio.run(calibrate(capped_config, ANCHORS, router, margin=0.0))
    assert second.scale == pytest.approx(first.scale) and second.caps == first.caps
    assert not any(r.capped for r in second.results) and not second.conflicts


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


from llm_router.calibration import write_level_caps


def test_level_caps_are_written_beside_family_scales_merged_lower_wins_and_comments_kept(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("router:\n  capabilities:\n    # the table\n    miss: [0.9, 0.5, 0.2, 0.05]\n"
                    "    family_scales: {}\n", encoding="utf-8")
    write_level_caps(path, {"mid": {"reasoning": 0.3456}})
    write_level_caps(path, {"mid": {"reasoning": 0.5, "code": 1.0}})
    text = path.read_text(encoding="utf-8")
    assert text.count("level_caps:") == 1 and "# the table" in text
    assert '    level_caps: {"mid": {"code": 1.0, "reasoning": 0.345}}' in text  # floored, never rounded up
    assert parse_config(raw_config(level_caps={"mid": {"reasoning": 0.345}})).router.capabilities.level_caps


def test_a_run_with_family_scales_is_idempotent_through_the_written_config(tmp_path):
    # mid has a family scale, so the global scale never reaches it: a cap on mid must not
    # move the global scale on the next run (it did: the capped mid bounded it lower).
    import yaml

    from llm_router.config import load_config
    from llm_router.config import _DEFAULT_MISS

    ask = ByTask({"ALPHA": {"reasoning": 0.6, "code": 0.9}, "BETA": {"reasoning": 0.99, "code": 0.1},
                  "GAMMA": {"reasoning": 0.9, "code": 0.9}})
    anchors = [{"task": "ALPHA job", "sufficient": "mid"}, {"task": "BETA job", "insufficient": "mid"},
               {"task": "GAMMA job", "sufficient": "top"}]
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw_config(family_scales={"mid": 0.3}, miss=list(_DEFAULT_MISS)),
                                   default_flow_style=None), encoding="utf-8")

    def once():
        config = load_config(path)
        cal = asyncio.run(calibrate(config, anchors, CapabilityRouter(config, ask=ask), margin=0.0))
        write_scale(path, cal.scale)
        if cal.caps:
            write_level_caps(path, cal.caps)
        return cal

    first = once()
    assert first.caps and not first.conflicts
    second = once()
    assert f"{second.scale:.3f}" == f"{first.scale:.3f}" and not any(r.capped for r in second.results)
    assert not second.conflicts
    written = load_config(path).router.capabilities
    assert written.miss_scale == pytest.approx(round(first.scale, 3)) and set(written.level_caps) == {"mid"}


def test_a_tier_on_the_fallback_profile_is_never_capped_the_operator_is_told_to_profile_it():
    # A cap under "*" would cap every unprofiled model, future ones included.
    from test_discovery import expanded

    config, _, _ = expanded()
    tier = "codex:gpt-7-nova@medium"
    assert config.router.capabilities.cards[tier].family == "*"
    needs = {"reasoning": 0.44, "code": 0.0, "niche": 0.0}
    router = CapabilityRouter(config, ask=Ask(needs))
    assert router.success(needs, tier) > 0.8
    cal = asyncio.run(calibrate(config, [{"task": "x", "insufficient": tier}], router, margin=0.0))
    assert cal.caps == {} and not cal.results[0].capped
    assert len(cal.conflicts) == 1 and "fallback profile" in cal.conflicts[0] and "write a profile" in cal.conflicts[0]


def test_a_family_cap_names_every_other_tier_of_the_family_it_lowers():
    # The anchor is about opus at low effort; the cap is family-wide, so it also flattens
    # the higher efforts' reasoning. That is reported per tier, before and after.
    from test_discovery import expanded

    config, _, _ = expanded()
    tier = "claude:claude-opus-5-5@low"
    needs = {"reasoning": 0.6, "code": 0.0, "niche": 0.0}
    router = CapabilityRouter(config, ask=Ask(needs))
    cal = asyncio.run(calibrate(config, [{"task": "x", "insufficient": tier, "because": "reasoning"}], router))
    (_, family, req, old, new, _), = cal.results[0].capped
    assert family == "claude-opus-*" and req == "reasoning" and new < old
    lowered = {t: (r, before, after) for t, r, before, after in cal.results[0].lowered}
    assert tier not in lowered
    assert lowered["claude:claude-opus-5-5@high"] == ("reasoning", 2.5, new)
    cards = config.router.capabilities.cards
    assert all(cards[t].levels["reasoning"] > new for t in lowered)


BLOCK = ("router:\n  capabilities:\n    miss: [0.9, 0.5, 0.2, 0.05]\n    miss_scale: 1.0\n"
         "    level_caps:\n      top:\n        code: 3.0\n    floor: 0.2\n")


def test_a_block_form_level_caps_is_refused_not_corrupted(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(BLOCK, encoding="utf-8")
    with pytest.raises(ConfigError, match="level_caps"):
        write_level_caps(path, {"mid": {"reasoning": 1.0}})
    assert path.read_text(encoding="utf-8") == BLOCK


def test_calibrate_write_refuses_a_block_form_level_caps_and_writes_nothing(tmp_path, monkeypatch, capsys):
    import yaml

    from llm_router import cli
    from test_discovery import report

    raw = raw_config(miss_scale=1.0, level_caps={"top": {"code": 3.0}})
    raw["catalog"] = {"check_on_start": False, "path": None}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw, default_flow_style=False), encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    assert "level_caps:\n" in before  # block form
    anchors = tmp_path / "a.yaml"
    anchors.write_text(yaml.safe_dump(ANCHORS), encoding="utf-8")
    monkeypatch.setattr("llm_router.catalog.check_catalog", lambda config: report())
    monkeypatch.setattr("llm_router.capabilities.jev_asker", lambda tier: Ask(HARD))
    assert cli.main(["-c", str(path), "calibrate", "--anchors", str(anchors), "--write"]) == 2
    assert "level_caps" in capsys.readouterr().err
    assert path.read_text(encoding="utf-8") == before  # neither miss_scale nor level_caps written


def test_a_block_form_family_scales_is_refused_not_corrupted(tmp_path):
    from llm_router.calibration import write_family_scales

    path = tmp_path / "c.yaml"
    text = BLOCK.replace("level_caps:\n      top:\n        code: 3.0\n", "family_scales:\n      top: 0.5\n")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="family_scales"):
        write_family_scales(path, {"mid": 0.4})
    assert path.read_text(encoding="utf-8") == text


def test_a_cap_is_checked_and_reported_at_the_floored_value_that_is_written():
    # write_level_caps floors to 3 places; protection and "routed after" must see that value, not a finer one.
    from llm_router.calibration import floor_cap

    _, _, cal = run(ANCHORS)
    (_, _, _, _, new, _), = cal.results[1].capped
    assert new == floor_cap(new) and cal.caps == {"mid": {"reasoning": new}}
