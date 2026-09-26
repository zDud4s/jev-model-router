"""Dominance: a carded tier another one beats on every requirement, at no more cost at any prompt size."""

from __future__ import annotations

import copy

import pytest

from jev_model_router.capabilities import CapabilityRouter
from jev_model_router.config import parse_config
from jev_model_router.dominance import dominated, summary

from test_capabilities import CAPS, Ask, raw_config


def config(tiers=None, verification=None, **caps):
    raw = raw_config(**caps)
    # cx and sub take tools like the tiers they are compared with, so only the cards decide.
    for name in ("cx", "sub"):
        raw["tiers"][name]["supports_tools"] = True
    for name, change in (tiers or {}).items():
        raw["tiers"][name].update(change)
    if verification is not None:
        raw["verification"] = verification
    return parse_config(raw)


def found(tiers=None, verification=None, unavailable=(), **caps):
    return dominated(CapabilityRouter(config(tiers, verification, **caps), ask=Ask()), unavailable)


def cards(**changes):
    out = copy.deepcopy(CAPS["cards"])
    out.update(changes)
    return out


def test_a_tier_as_able_and_cheaper_dominates():
    # cx (2,2) at list 2/10 against mid (2,2) billing 3/15; sub (3,3) at 5/25 against top (3,3) at 15/75.
    assert found() == {"mid": ["cx"], "top": ["sub"]}


def test_equal_tiers_do_not_dominate_each_other():
    result = found(cards=cards(cx={"levels": {"reasoning": 2, "code": 2}, "list_prices": {"input": 3.0, "output": 15.0}}))
    assert "mid" not in result and "cx" not in result


def test_a_cost_order_that_flips_with_prompt_size_is_not_dominance():
    # cx: cheaper input, dearer output. Cheaper than mid at 100k tokens in, dearer at 1k.
    result = found(cards=cards(cx={"levels": {"reasoning": 2, "code": 2}, "list_prices": {"input": 1.0, "output": 30.0}}))
    assert "mid" not in result


def test_a_family_scale_can_take_dominance_away():
    assert "mid" not in found(family_scales={"cx": 3.0})


def test_a_level_cap_can_take_dominance_away():
    assert found(level_caps={"sub": {"reasoning": 1.0}}) == {"mid": ["cx"]}


def test_a_tiers_own_prices_beat_its_cards_list_prices():
    cheap_list = {"levels": {"reasoning": 2, "code": 2}, "list_prices": {"input": 0.1, "output": 0.1}}
    assert found(cards=cards(mid=cheap_list))["mid"] == ["cx"]


def test_a_smaller_context_window_takes_dominance_away():
    # A request between the two windows can go to mid and not to cx.
    assert "mid" not in found(tiers={"cx": {"context_window": 100_000}})


def test_a_tier_without_tools_does_not_dominate_one_with_them():
    assert "mid" not in found(tiers={"cx": {"supports_tools": False}})


def test_under_expected_cost_a_less_verified_tier_does_not_dominate():
    # mid's failures are caught by the verifier, cx's are not: a failure on cx costs more.
    verification = {"enabled": True, "verifier_tier": "top", "verify_tiers": ["mid"]}
    assert "mid" not in found(verification=verification, rule="expected_cost")
    # The target rule does not weigh failures: there it still dominates.
    assert found(verification=verification)["mid"] == ["cx"]


def test_an_unavailable_tier_dominates_nothing():
    assert "mid" not in found(unavailable={"cx": "model not served"})


def test_summary_names_at_most_three_dominators():
    lines = summary({"x": ["a", "b", "c", "d"]}, 9)
    assert lines == ["dominance: 1 of 9 carded tier(s) are never the cheapest adequate choice "
                     "while a tier that dominates them is eligible and available", "  x: beaten by a, b, c (+1 more)"]
    assert summary({}, 9) == []
