"""The call router: what a call on a tier costs right now, and whether it can be made at all."""

from __future__ import annotations

import copy
import math

import pytest

from jev_model_router.calls import MIN_EVIDENCE, WINDOW, CallRouter
from jev_model_router.config import Prices, parse_config
from jev_model_router.schemas import Usage

from test_capabilities import CAPS, raw_config


def calls(clock=None, **caps) -> CallRouter:
    config = parse_config(raw_config(**caps))
    c = config.router.capabilities
    kwargs = {"clock": clock} if clock else {}
    return CallRouter(config, c, list_prices={n: card.list_prices for n, card in c.cards.items()}, **kwargs)


def test_a_tier_that_bills_is_priced_by_its_bill_and_one_that_does_not_by_its_card():
    c = calls()
    assert c.prices("mid").input == 3.0  # the tier's own prices
    assert c.prices("sub").input == 5.0  # the card's list prices
    assert c.cost("sub", 1_000_000, 0) == pytest.approx(5.0)
    assert c.cost("sub", 0, 1_000_000) == pytest.approx(25.0)


def test_a_429_locks_the_subscription_until_its_window_turns():
    now = [0.0]
    c = calls(clock=lambda: now[0])
    c.record_spend("sub", Usage(), 429, Prices())
    assert c.cost("sub", 1000, 1000) == math.inf
    assert math.isfinite(c.cost("mid", 1000, 1000))
    now[0] += 5 * 3600 + 1
    assert math.isfinite(c.cost("sub", 1000, 1000))


def test_spend_is_charged_at_the_prices_given_and_only_to_subscriptions():
    c = calls()
    spend = Prices(input=2.0, configured=True)
    c.record_spend("sub", Usage(prompt_tokens=1_000_000), 200, spend)
    c.record_spend("mid", Usage(prompt_tokens=1_000_000), 200, spend)  # billed per token: no subscription
    c.record_spend("sub", Usage(prompt_tokens=1_000_000), 500, spend)  # a failed call spends nothing
    assert c.state() == {"subscriptions": {"claude": {"used_usd": 2.0, "locked": False}}}


SHAPE = {"input_per_output": 300, "cache_read": 0.9, "cache_write": 0.05}
TASK = Usage(prompt_tokens=30_000, completion_tokens=100, cached_tokens=29_000, cache_write_tokens=500)


def test_without_a_shape_a_task_costs_what_one_call_costs():
    c = calls()
    assert c.shape_for("sub") is None
    assert c.task_cost("sub", 5000, 1500) == c.cost("sub", 5000, 1500)


def test_a_shaped_task_is_priced_by_the_formula():
    cards = copy.deepcopy(CAPS["cards"])
    cards["sub"]["list_prices"] = {"input": 10.0, "output": 50.0, "cache_read": 0.25, "cache_write": 12.5}
    c = calls(cards=cards, task_shape=SHAPE)
    total_in = 300 * 1500
    expected = (total_in * (0.05 * 10.0 + 0.9 * 0.25 + 0.05 * 12.5) + 1500 * 50.0) / 1e6
    assert c.task_cost("sub", 5000, 1500) == pytest.approx(expected)


def test_an_unwritten_cache_price_gives_no_discount_and_an_unwritten_write_costs_the_input_rate():
    c = calls(task_shape=SHAPE)  # sub's list prices: input 5, output 25, no cache price
    assert c.task_cost("sub", 5000, 1500) == pytest.approx((300 * 1500 * 5.0 + 1500 * 25.0) / 1e6)


def test_a_locked_tier_costs_inf_as_a_task_too():
    c = calls(task_shape=SHAPE)
    c.record_spend("sub", Usage(), 429, Prices())
    assert c.task_cost("sub", 5000, 1500) == math.inf


def test_an_observed_shape_takes_over_at_min_evidence_and_not_before():
    c = calls(task_shape=SHAPE)
    for _ in range(MIN_EVIDENCE - 1):
        assert c.record_task("sub", TASK)
    assert c.shape_for("sub").source == "config"
    c.record_task("sub", TASK)
    shape = c.shape_for("sub")
    assert shape.source == f"observed:{MIN_EVIDENCE}"
    assert (shape.input_per_output, shape.cache_read, shape.cache_write) == pytest.approx((300.0, 29 / 30, 0.5 / 30))
    assert c.shape_for("cx").source == "config"  # never pooled across tiers: the runner shapes it


def test_the_window_keeps_the_last_outcomes():
    c = calls()
    for _ in range(WINDOW):
        c.record_task("sub", Usage(prompt_tokens=1000, completion_tokens=100, cached_tokens=900))
    for _ in range(WINDOW):
        c.record_task("sub", TASK)
    shape = c.shape_for("sub")
    assert shape.source == f"observed:{WINDOW}" and shape.input_per_output == pytest.approx(300.0)


def test_an_outcome_without_cache_tokens_is_not_evidence():
    c = calls()
    for _ in range(MIN_EVIDENCE):
        assert not c.record_task("sub", Usage(prompt_tokens=30_000, completion_tokens=100))
    assert c.shape_for("sub") is None


def test_tiers_with_no_cache_price_are_named_only_while_a_shape_is_in_use():
    assert calls().uncached() == []
    named = calls(task_shape=SHAPE).uncached()
    assert "sub" in named and "cx" in named
    assert "mid" not in named and "cheap" not in named  # a cache price, and free


def test_a_task_row_that_does_not_add_up_is_rejected():
    c = calls()
    over = Usage(prompt_tokens=1000, completion_tokens=100, cached_tokens=900, cache_write_tokens=200)
    negative = Usage(prompt_tokens=1000, completion_tokens=100, cached_tokens=-1)
    assert not c.record_task("sub", over)
    assert not c.record_task("sub", negative)
    assert c.shape_for("sub") is None


def test_a_tier_that_has_reported_cache_keeps_its_cold_tasks_too():
    c = calls()
    for _ in range(MIN_EVIDENCE):
        assert c.record_task("sub", TASK)
    before = c.shape_for("sub")
    cold = Usage(prompt_tokens=1000, completion_tokens=100)  # no cache tokens at all
    assert c.record_task("sub", cold)
    after = c.shape_for("sub")
    assert after.cache_read < before.cache_read
