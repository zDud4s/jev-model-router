"""The call router: what a call on a tier costs right now, and whether it can be made.

The capabilities router asks two things of every tier: will it succeed (Jev,
the cards, calibration) and what will it cost. This module answers the second,
and only that: prices -- what the tier bills, else its card's list prices --
and the subscription ledger, whose 429 lock makes a tier cost `inf` until its
window turns. It never imports the model router; the model router is given one.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Callable, Mapping

from .config import CapabilitiesConfig, Config, Prices
from .schemas import Usage


# ---------------------------------------------------------------- quota
class QuotaLedger:
    """Per subscription: 429 lockouts, and list-price spend over the window for display.

    The spend never moves a decision -- cost is the price per token, the same
    for a subscription as for the API. A 429 does: that subscription is not
    available until its window turns.
    """

    def __init__(self, config: CapabilitiesConfig, clock: Callable[[], float] = time.time) -> None:
        self._budgets = config.subscriptions
        self._clock = clock
        self._spent: dict[str, deque[tuple[float, float]]] = {}
        self._locked_until: dict[str, float] = {}

    def record(self, subscription: str, usd: float) -> None:
        self._spent.setdefault(subscription, deque()).append((self._clock(), usd))

    def lock(self, subscription: str) -> None:
        budget = self._budgets.get(subscription)
        hours = budget.window_hours if budget else 1.0
        self._locked_until[subscription] = self._clock() + hours * 3600

    def used(self, subscription: str) -> float:
        budget = self._budgets.get(subscription)
        entries = self._spent.get(subscription)
        if not entries:
            return 0.0
        horizon = self._clock() - (budget.window_hours if budget else 5.0) * 3600
        while entries and entries[0][0] < horizon:
            entries.popleft()
        return sum(usd for _, usd in entries)

    def locked(self, subscription: str) -> bool:
        return self._clock() < self._locked_until.get(subscription, 0.0)

    def state(self) -> dict[str, Any]:
        names = set(self._budgets) | set(self._spent) | set(self._locked_until)
        return {
            name: {
                "used_usd": round(self.used(name), 4),
                "locked": self.locked(name),
            }
            for name in sorted(names)
        }


def _usd(prices: Prices, prompt_tokens: int, output_tokens: int) -> float:
    return (prompt_tokens * prices.input + output_tokens * prices.output) / 1_000_000


class CallRouter:
    """Prices and availability per tier. `CapabilityRouter` weighs them; it does not keep them."""

    def __init__(
        self,
        config: Config,
        caps: CapabilitiesConfig,
        *,
        list_prices: Mapping[str, Prices],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._list_prices = dict(list_prices)
        self.ledger = QuotaLedger(caps, clock)

    def prices(self, tier_name: str) -> Prices:
        """What a tier really bills wins; its card's list prices stand in where it bills nothing."""
        tier = self._config.tier(tier_name)
        return tier.prices if tier.prices.configured else self._list_prices[tier_name]

    def locked(self, tier_name: str) -> bool:
        tier = self._config.tier(tier_name)
        return bool(tier.subscription) and self.ledger.locked(tier.subscription)

    def cost(self, tier_name: str, input_tokens: int, output_tokens: int) -> float:
        if self.locked(tier_name):
            return math.inf  # answered 429: not available until the window turns
        return _usd(self.prices(tier_name), input_tokens, output_tokens)

    def record_spend(self, tier_name: str, usage: Usage, status: int, spend_prices: Prices) -> None:
        """Charge a finished call to its subscription at `spend_prices`, or lock it on a 429."""
        tier = self._config.tiers.get(tier_name)
        if tier is None or not tier.subscription:
            return
        if status == 429:
            self.ledger.lock(tier.subscription)
            return
        if status < 400:
            self.ledger.record(tier.subscription, _usd(spend_prices, usage.prompt_tokens, usage.completion_tokens))

    def state(self) -> dict[str, Any]:
        return {"subscriptions": self.ledger.state()}


__all__ = ["CallRouter", "QuotaLedger"]
