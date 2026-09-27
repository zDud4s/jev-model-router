"""The Tiers view served at /routing/tiers: every carded tier, where each of its levels came from, what beats it.

Built once at startup from the same derivation `benchmarks check` prints
(`discovery.derived_cards`), so the page and the command line cannot disagree.
Per requirement a cell carries the level served, the level the profile alone
would give (the card `expand` builds without evidence), the source of the level
("benchmark", "blended" or "profile"), the evidence C behind it, and the level
cap when one applies. A card written in the config has no profile: its profile
level is its own level.

A report only: nothing here changes what is served, and the caller keeps a
failure from stopping startup.
"""

from __future__ import annotations

from typing import Any

from .config import EFFORTS, Config

_EPS = 1e-9


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def _evidence(scores: Any) -> dict[str, Any]:
    from .scores_derive import link_state

    marked: dict[str, list[str]] = {"thin": [], "detached": [], "unlinked": []}
    for bench in scores.benchmarks:
        state = link_state(scores, bench)
        if state:
            marked[state].append(bench)
    return {
        "imported_at": scores.imported_at,
        "benchmarks_count": len(scores.benchmarks),
        "points": len(scores.points),
        "sources": [
            {"name": name, "origin": source.origin, "status": scores.imported_status.get(name, "not imported")}
            for name, source in scores.sources.items()
        ],
        "benchmarks": marked,
    }


def _prices(router: Any, config: Config, name: str) -> dict[str, float]:
    prices_of = getattr(router, "prices", None)
    prices = prices_of(name) if callable(prices_of) else config.tier(name).prices
    return {k: getattr(prices, k) for k in ("input", "output", "cache_read", "cache_write")}


def build(
    config: Config,
    router: Any,
    *,
    report: Any = None,
    configured: Config | None = None,
    scores: Any = None,
) -> dict[str, Any]:
    """The view's JSON.

    `config` is the config served (after discovery); `configured` the one given
    to `expand`, which rebuilds the profile cards; `scores` the evidence the
    served cards were derived from, or None when they are the profiles.
    """
    caps = config.router.capabilities
    if caps is None:
        return {"error": None, "global": None, "tiers": []}
    profile_cards = caps.cards
    derivations: dict[str, Any] = {}
    line = None
    if report is not None and configured is not None:
        from .discovery import derived_cards, expand, model_ids

        profile_cards = expand(configured, report, None)[0].router.capabilities.cards
        if scores is not None:
            (a, k), derivations = derived_cards(config, scores, model_ids(report))
            line = {"a": _round(a), "k": _round(k), "profile_weight": _round(scores.c0)}
    dominated = getattr(router, "dominated", None) or {}
    rows = []
    for name in sorted(caps.cards):
        card = caps.cards[name]
        tier = config.tiers.get(name)
        derived = derivations.get(name)
        profile = profile_cards.get(name, card)
        ceilings = caps.level_caps.get(card.family or name) or {}
        levels = {}
        for r in caps.requirements:
            level = card.levels.get(r, 0.0)
            cap = ceilings.get(r)
            levels[r] = {
                "level": _round(level),
                "profile_level": _round(profile.levels.get(r, 0.0)),
                "source": derived.source[r] if derived is not None else "profile",
                "evidence": _round(derived.coverage.get(r, 0.0)) if derived is not None else 0.0,
                "cap": _round(cap),
                # The ceiling binds: the level sits on it.
                "capped": cap is not None and level >= cap - _EPS,
            }
        rows.append({
            "name": name,
            "backend": tier.backend if tier else None,
            "model": tier.model if tier else None,
            "effort": tier.effort if tier else None,
            "family": card.family,
            "card": "derived" if derived is not None else ("profile" if card.family is not None else "configured"),
            "prices": _prices(router, config, name),
            # The listing, the profile or the config; "tier" when the tier bills its own prices.
            "prices_from": "tier" if tier is not None and tier.prices.configured else card.list_prices_from,
            "output_tokens": card.output_tokens,
            "input_overhead": card.input_overhead,
            "levels": levels,
            "dominated_by": list(dominated.get(name, [])),
        })
    return {
        "error": None,
        "global": {
            "requirements": list(caps.requirements),
            "efforts": list(EFFORTS),  # their order, for sorting
            "miss_scale": caps.miss_scale,
            "target": caps.target,
            "rule": caps.rule,
            "line": line,
            "evidence": _evidence(scores) if scores is not None else None,
            "dominance_error": getattr(router, "dominance_error", None),
        },
        "tiers": rows,
    }


__all__ = ["build"]
