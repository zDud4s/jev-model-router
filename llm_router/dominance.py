"""Dominance: carded tiers that are never the cheapest adequate choice while a tier that dominates them is there.

Y dominates X when, for every requirement, Y's effective miss (its scale times
`miss` at its level) is no higher than X's. That alone means success(Y) >=
success(X) for every task Jev could read. On top of that, three cost terms must
hold, none of which depends on the prompt:

- input price;
- the backend's own overhead times input price;
- typical output times output price.

Together these mean Y is no dearer at any prompt size. At least one of all
these comparisons must be strict.

Y must also be able to take every request X can, or a request that X can serve
and Y cannot would still go to X:

- Y's context window is at least X's;
- Y takes tools whenever X does;
- Y is not unavailable (the catalog check's verdict);
- under the `expected_cost` rule, a failure on Y costs no more than one on X:
  Y's failure factor (from how much of it the router's verifier checks) is no
  higher.

Sampling prompt sizes instead would be wrong. On the real config of
2026-09-25, 64 pairs of tiers swap cost order between a 1k and a 100k prompt.

What it does NOT promise: that X is never picked. A tie on score and cost (a
need vector no requirement of theirs tells apart) goes by config order, and
X can still be picked by a retry after Y failed (the retry never goes below
the tier that failed, and skips it), while Y's subscription is locked after a
429, or by the fallback when Jev cannot be read. So the report says X is never
the cheapest adequate choice while Y is eligible and available.

It is a report. What it signals is a card that may be wrong, such as a new
model rated below an older, dearer one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Collection

if TYPE_CHECKING:
    from .capabilities import CapabilityRouter


def dominated(router: "CapabilityRouter", unavailable: Collection[str] = ()) -> dict[str, list[str]]:
    """Dominated carded tier -> the tiers that dominate it, cheapest first. Pure; O(n^2 x R).

    `unavailable` names tiers the catalog check found not served: they dominate nothing.
    """
    caps = router._caps
    config = router._config
    names = sorted(caps.cards)
    miss = {
        t: [min(1.0, router.scale_for(t) * router._miss(caps.cards[t].levels.get(r, 0.0))) for r in caps.requirements]
        for t in names
    }
    cost: dict[str, tuple[float, float, float]] = {}
    for t in names:
        prices, card = router.prices(t), caps.cards[t]
        cost[t] = (prices.input, card.input_overhead * prices.input, card.output_tokens * prices.output)
    # Must hold, never strictly: what each tier can take, and what a failure on it costs.
    reach: dict[str, tuple[float, ...]] = {}
    for t in names:
        tier = config.tier(t)
        # Larger is better in each, so they are negated to read "no higher" like the misses.
        terms = [-float(tier.context_window), -float(tier.supports_tools)]
        if caps.rule == "expected_cost":
            caught = router._verified_share(t)
            terms.append(caught * caps.failure.detected + (1 - caught) * caps.failure.undetected)
        reach[t] = tuple(terms)
    out: dict[str, list[str]] = {}
    for x in names:
        by = []
        for y in names:
            if y == x or y in unavailable:
                continue
            if not all(a <= b for a, b in zip(reach[y], reach[x])):
                continue
            pairs = list(zip(miss[y], miss[x])) + list(zip(cost[y], cost[x]))
            if all(a <= b for a, b in pairs) and any(a < b for a, b in pairs):
                by.append(y)
        if by:
            out[x] = sorted(by, key=lambda t: (cost[t][2], cost[t][0], t))
    return out


def summary(found: dict[str, list[str]], total: int) -> list[str]:
    if not found:
        return []
    lines = [f"dominance: {len(found)} of {total} carded tier(s) are never the cheapest adequate choice "
             f"while a tier that dominates them is eligible and available"]
    for tier, by in sorted(found.items()):
        more = f" (+{len(by) - 3} more)" if len(by) > 3 else ""
        lines.append(f"  {tier}: beaten by {', '.join(by[:3])}{more}")
    return lines


__all__ = ["dominated", "summary"]
