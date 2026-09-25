"""Dominance: a carded tier another one beats on every requirement, at no more cost at any prompt size."""

from __future__ import annotations

import copy

import pytest

from llm_router.capabilities import CapabilityRouter
from llm_router.config import parse_config
from llm_router.dominance import dominated, summary

from test_capabilities import CAPS, Ask, raw_config


def found(**caps):
    return dominated(CapabilityRouter(parse_config(raw_config(**caps)), ask=Ask()))


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


def test_summary_names_at_most_three_dominators():
    lines = summary({"x": ["a", "b", "c", "d"]}, 9)
    assert lines == ["dominance: 1 of 9 carded tier(s) can never be picked", "  x: beaten by a, b, c (+1 more)"]
    assert summary({}, 9) == []
