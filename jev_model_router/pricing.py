"""Cost and counterfactual cost.

Counterfactual cost is the reason this project exists. A savings claim measured
only against "we would always have used the most expensive tier" is not
defensible: configuring a cheaper model is free and saves a lot on its own, so
that baseline flatters any router. Recording what the request would have cost at
EVERY configured tier lets a reader compute savings against the baseline they
actually believe in, including the cheap one that makes the router look bad.

The counterfactual reuses the SERVED request's token counts. That is an
approximation and worth naming: another model would have emitted a different
number of output tokens for the same prompt. It is the only honest arithmetic
available without running the request twice, and it is consistent across tiers,
which is what a comparison needs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config, Prices
from .schemas import Usage

_PER_MILLION = 1_000_000.0


def cost_usd(prices: Prices, usage: Usage) -> float:
    """Cost of one request at these prices. Unpriced tiers cost zero."""
    # Cached input tokens are billed at the cache-read rate INSTEAD of the input
    # rate, so they are subtracted from the billable input rather than added on
    # top. Backends that report no caching send cached_tokens=0 and this is a
    # no-op for them.
    billable_input = max(usage.prompt_tokens - usage.cached_tokens, 0)
    total = (
        billable_input * prices.input
        + usage.cached_tokens * prices.cache_read
        + usage.cache_write_tokens * prices.cache_write
        + usage.completion_tokens * prices.output
    )
    return total / _PER_MILLION


@dataclass(frozen=True)
class Counterfactual:
    tier: str
    cost_usd: float
    # False when the tier carries no prices at all. Zero cost then means
    # "unknown baseline", not "free", and stats must not silently treat the two
    # the same way.
    priced: bool
    # Whether this tier could have served the request at all. A counterfactual
    # against a tier that would have rejected the request is not a real
    # alternative, and a savings claim that leans on one is fiction.
    eligible: bool


def counterfactuals(
    config: Config, usage: Usage, *, eligible_tiers: set[str] | None = None
) -> list[Counterfactual]:
    """What this request would have cost at each configured tier."""
    eligible = eligible_tiers if eligible_tiers is not None else set(config.tiers)
    return [
        Counterfactual(
            tier=name,
            cost_usd=cost_usd(tier.prices, usage),
            priced=tier.prices.configured,
            eligible=name in eligible,
        )
        for name, tier in config.tiers.items()
    ]
