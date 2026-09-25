"""Dominance: carded tiers that can never be picked, because another is as able and no dearer.

Y dominates X when, for every requirement, Y's effective miss (its scale times
`miss` at its level) is no higher than X's. That alone means success(Y) >=
success(X) for every task Jev could read. On top of that, three cost terms must
hold, none of which depends on the prompt:

- input price;
- the backend's own overhead times input price;
- typical output times output price.

Together these mean Y is no dearer at any prompt size. At least one of all
these comparisons must be strict.

Sampling prompt sizes instead would be wrong. On the real config of
2026-09-25, 64 pairs of tiers swap cost order between a 1k and a 100k prompt.

It is a report. A dominated tier already never wins under either pick rule, so
removing it would change no decision. What it signals is a card that may be
wrong, such as a new model rated below an older, dearer one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .capabilities import CapabilityRouter


def dominated(router: "CapabilityRouter") -> dict[str, list[str]]:
    """Dominated carded tier -> the tiers that dominate it, cheapest first. Pure; O(n^2 x R)."""
    caps = router._caps
    names = sorted(caps.cards)
    miss = {
        t: [min(1.0, router.scale_for(t) * router._miss(caps.cards[t].levels.get(r, 0.0))) for r in caps.requirements]
        for t in names
    }
    cost: dict[str, tuple[float, float, float]] = {}
    for t in names:
        prices, card = router.prices(t), caps.cards[t]
        cost[t] = (prices.input, card.input_overhead * prices.input, card.output_tokens * prices.output)
    out: dict[str, list[str]] = {}
    for x in names:
        by = []
        for y in names:
            if y == x:
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
    lines = [f"dominance: {len(found)} of {total} carded tier(s) can never be picked"]
    for tier, by in sorted(found.items()):
        more = f" (+{len(by) - 3} more)" if len(by) > 3 else ""
        lines.append(f"  {tier}: beaten by {', '.join(by[:3])}{more}")
    return lines


__all__ = ["dominated", "summary"]
