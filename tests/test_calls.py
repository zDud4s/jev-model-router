"""The call router: what a call on a tier costs right now, and whether it can be made at all."""

from __future__ import annotations

import math

import pytest

from jev_model_router.calls import CallRouter
from jev_model_router.config import Prices, parse_config
from jev_model_router.schemas import Usage

from test_capabilities import raw_config


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
