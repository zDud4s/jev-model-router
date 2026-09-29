"""The call router: what a call on a tier costs right now, and whether it can be made.

The capabilities router asks two things of every tier: will it succeed (Jev,
the cards, calibration) and what will it cost. This module answers the second,
and only that: prices -- what the tier bills, else its card's list prices --
and the subscription ledger, whose 429 lock makes a tier cost `inf` until its
window turns. It never imports the model router; the model router is given one.

A route-only decision is a whole task, not one call: an agent loop that
re-sends a growing context every turn, mostly from the provider's prompt cache.
`task_cost` prices it by a task shape (`config.TaskShape`): observed on the
tier from reported outcomes, else the config's. Scaled to one answer's output,
it orders tiers; it does not budget a task.
"""

from __future__ import annotations

import math
import time
from bisect import insort
from collections import deque
from typing import Any, Callable, Mapping, Sequence

from .config import CapabilitiesConfig, Config, Prices, TaskShape
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

    def record(self, subscription: str, usd: float, at: float | None = None) -> None:
        """Record spend when it happened, keeping late cross-process events in time order."""
        insort(self._spent.setdefault(subscription, deque()), (self._clock() if at is None else at, usd))

    def lock(self, subscription: str, at: float | None = None) -> None:
        """Off the table for one window from `at` (a 429 seen then, maybe by another process), else from now."""
        budget = self._budgets.get(subscription)
        hours = budget.window_hours if budget else 1.0
        until = (self._clock() if at is None else at) + hours * 3600
        # An older 429 read late must not shorten a lock a newer one set.
        self._locked_until[subscription] = max(self._locked_until.get(subscription, 0.0), until)

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


# Outcomes with cache tokens a tier needs before its own shape replaces the
# config's, and how many of the latest it is pooled over.
MIN_EVIDENCE = 5
WINDOW = 200


def accepts_task_row(
    prompt_tokens: int, completion_tokens: int, cached_tokens: int, cache_write_tokens: int, reports_cache: bool,
) -> bool:
    """Whether one outcome's tokens count as task-shape evidence.

    The rule `CallRouter.record_task` applies to a live call; `stats._task_cost` pools the same
    log rows without a `CallRouter` in hand, and shares this function so the two never disagree
    about what counts.
    """
    if prompt_tokens <= 0 or completion_tokens <= 0:
        return False
    if cached_tokens < 0 or cache_write_tokens < 0:
        return False
    cache = cached_tokens + cache_write_tokens
    if cache > prompt_tokens:
        return False  # doesn't add up: e.g. a caller reporting fresh input only, not on top
    if cache <= 0 and not reports_cache:
        return False  # a caller that does not report the cache, not a task that never cached
    return True


def pooled_shape(rows: Sequence[tuple[int, int, int, int]]) -> TaskShape | None:
    """(input, cached, written, output) per task, pooled: sums, not a mean of ratios, so long tasks weigh more."""
    total_in = sum(r[0] for r in rows)
    total_out = sum(r[3] for r in rows)
    if total_in <= 0 or total_out <= 0:
        return None
    read = min(sum(r[1] for r in rows) / total_in, 1.0)
    # A provider that counts writes outside the prompt could push the sum past 1.
    write = min(sum(r[2] for r in rows) / total_in, 1.0 - read)
    return TaskShape(total_in / total_out, read, write, f"observed:{len(rows)}")


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
        self._shape = caps.task_shape
        self._tasks: dict[str, deque[tuple[int, int, int, int]]] = {}
        # Tiers that have ever reported a nonzero cache field. Once one has, a
        # later zero-cache task from it is a real cold task, not silence.
        self._reports_cache: set[str] = set()

    def prices(self, tier_name: str) -> Prices:
        """What a tier really bills wins; its card's list prices stand in where it bills nothing."""
        tier = self._config.tier(tier_name)
        return tier.prices if tier.prices.configured else self._list_prices[tier_name]

    def locked(self, tier_name: str) -> bool:
        tier = self._config.tier(tier_name)
        return bool(tier.subscription) and self.ledger.locked(tier.subscription)

    def lock(self, tier_name: str, at: float | None = None) -> None:
        """A 429 on this tier's subscription that another process sharing the log saw, at `at`."""
        tier = self._config.tiers.get(tier_name)
        if tier is not None and tier.subscription:
            self.ledger.lock(tier.subscription, at=at)

    def cost(self, tier_name: str, input_tokens: int, output_tokens: int) -> float:
        if self.locked(tier_name):
            return math.inf  # answered 429: not available until the window turns
        return _usd(self.prices(tier_name), input_tokens, output_tokens)

    def record_spend(
        self, tier_name: str, usage: Usage, status: int, spend_prices: Prices, at: float | None = None,
    ) -> None:
        """Charge a finished call to its subscription at `spend_prices`, or lock it on a 429."""
        tier = self._config.tiers.get(tier_name)
        if tier is None or not tier.subscription:
            return
        if status == 429:
            self.ledger.lock(tier.subscription, at=at)
            return
        if status < 400:
            self.ledger.record(
                tier.subscription, _usd(spend_prices, usage.prompt_tokens, usage.completion_tokens), at=at,
            )

    def record_task(self, tier_name: str, usage: Usage) -> bool:
        """One routed task's whole usage, as evidence of the tier's shape.

        Without cache tokens it is none -- unless this tier has already shown it reports the
        cache, in which case a cold task is real evidence too, not a caller staying silent.
        """
        if not accepts_task_row(
            usage.prompt_tokens, usage.completion_tokens, usage.cached_tokens, usage.cache_write_tokens,
            tier_name in self._reports_cache,
        ):
            return False
        if usage.cached_tokens + usage.cache_write_tokens > 0:
            self._reports_cache.add(tier_name)
        self._tasks.setdefault(tier_name, deque(maxlen=WINDOW)).append(
            (usage.prompt_tokens, usage.cached_tokens, usage.cache_write_tokens, usage.completion_tokens)
        )
        return True

    def shape_for(self, tier_name: str) -> TaskShape | None:
        """This tier's observed shape once it has `MIN_EVIDENCE` outcomes, else the config's, else None."""
        seen = self._tasks.get(tier_name)
        if seen is not None and len(seen) >= MIN_EVIDENCE:
            return pooled_shape(list(seen))
        return self._shape

    def task_cost(self, tier_name: str, input_tokens: int, output_tokens: int) -> float:
        """A whole task by its shape, scaled to `output_tokens` of output; `cost(...)` when the tier has no shape.

        `input_tokens` is used only in that unshaped fallback: a shape states its own input.
        """
        shape = self.shape_for(tier_name)
        if shape is None:
            return self.cost(tier_name, input_tokens, output_tokens)
        if self.locked(tier_name):
            return math.inf
        p = self.prices(tier_name)
        total_in = shape.input_per_output * output_tokens
        fresh = max(0.0, 1.0 - shape.cache_read - shape.cache_write)
        # Unwritten, a cache write is billed as the input it is; its 0.0 default would make it free.
        write = p.cache_write if p.cache_write_configured else p.input
        per_input = fresh * p.input + shape.cache_read * p.cache_read + shape.cache_write * write
        return (total_in * per_input + output_tokens * p.output) / 1_000_000

    def uncached(self) -> list[str]:
        """Carded tiers with no cache discount, while a shape is in use: priced as if they cached nothing."""
        if self._shape is None and not any(len(seen) >= MIN_EVIDENCE for seen in self._tasks.values()):
            return []
        out = []
        for tier_name in sorted(self._list_prices):
            p = self.prices(tier_name)
            if p.input > 0 and p.cache_read >= p.input:
                out.append(tier_name)
        return out

    def state(self) -> dict[str, Any]:
        return {"subscriptions": self.ledger.state()}


__all__ = ["CallRouter", "MIN_EVIDENCE", "QuotaLedger", "TaskShape", "WINDOW", "accepts_task_row", "pooled_shape"]
