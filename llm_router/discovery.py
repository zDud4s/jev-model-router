"""Discovery: every model a provider serves, at every effort it takes, becomes a tier.

The catalog is not a list someone keeps. `discover` names providers -- the Claude
Code subscription, the Codex subscription, a local Ollama -- and at startup each
one is asked what it serves (see `catalog.py`, which costs no quota). Each
(model, effort) pair it answers with becomes a tier named
`<source>:<model>@<effort>`, with a card built from the first profile whose glob
matches the model and the rule for that effort.

Nothing is excluded by hand. A model that is dearer than a newer one and no
better is still in the catalog; it simply never wins "the cheapest tier that
covers the task", which is the whole choice the router makes. A model no profile
matches is offered too, on `fallback_profile` -- low levels, high prices, so it
is the last resort rather than a surprise -- and is named in the report so a
profile can be written for it.

Performance is the one thing no catalog publishes, so profiles are priors. The
router logs every decision with Jev's reading and the card fingerprint, and that
is what recalibrates them.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, replace

from .catalog import CatalogReport, Offered, _older
from .config import EFFORTS, CapabilitiesConfig, Config, DiscoverSource, ModelCard, ModelProfile, TierConfig
from .scores import Scores, benchmark_weights
from .scores_derive import DerivedCard, derive, effort_prior, line_for, served_keys


@dataclass(frozen=True)
class Discovered:
    tier: str
    source: str
    model: str
    effort: str | None
    profile: str  # the glob that matched, or "*fallback*"


def profile_for(caps: CapabilitiesConfig, model: Offered) -> tuple[ModelProfile, bool]:
    names = (model.id, *model.aliases)
    for profile in caps.profiles:
        if any(fnmatch.fnmatchcase(name, profile.match) for name in names):
            return profile, True
    assert caps.fallback_profile is not None, "parse_config requires one alongside `discover`"
    return caps.fallback_profile, False


def card_for(
    caps: CapabilitiesConfig,
    profile: ModelProfile,
    effort: str | None,
    source: DiscoverSource,
    derived: DerivedCard | None = None,
) -> ModelCard:
    if derived is None:
        levels, output = effort_prior(caps, profile, effort)
    else:
        levels, output = dict(derived.levels), derived.output_tokens
    return ModelCard(
        levels=levels,
        output_tokens=output,
        input_overhead=source.input_overhead,
        list_prices=profile.list_prices,
        family=profile.match,
    )


def _usable(source: DiscoverSource, model: Offered, matched: bool, cli_version: str | None = None) -> bool:
    """Whether a served model can answer a chat request at all.

    Not a preference: an embedding model cannot write an answer, and a hidden
    Codex utility (its auto-approval reviewer) is not offered to people. A
    hidden model a profile names on purpose is kept.
    """
    if source.backend == "ollama" and "embed" in model.id.lower():
        return False
    if model.hidden and not matched:
        return False
    if model.min_cli and cli_version and _older(cli_version, model.min_cli):
        return False  # the installed CLI would refuse it
    return True


def expand(
    config: Config, report: CatalogReport, scores: Scores | None = None
) -> tuple[Config, list[Discovered], list[str]]:
    """The config with every discovered tier and card added.

    Returns (config, discovered tiers, models served on the fallback profile).
    A (backend, model, effort) an explicit tier already covers is not added
    twice: the explicit tier and its card win. With `scores`, every card it
    builds is derived from benchmark evidence (see `scores_derive.py`), on one
    (a, k) line fitted across all of them.
    """
    caps = config.router.capabilities
    if caps is None or not caps.discover:
        return config, [], []
    tiers = dict(config.tiers)
    cards = dict(caps.cards)
    explicit = {(t.backend, t.model, t.effort) for t in config.tiers.values()}
    found: list[Discovered] = []
    unprofiled: list[str] = []
    # (tier, model ids, effort, profile, source): cards are built once all are known,
    # because the line from ability to level is fitted across every one of them.
    pending: list[tuple[str, tuple[str, ...], str | None, ModelProfile, DiscoverSource]] = []
    for name, source in caps.discover.items():
        served = report.discovered.get(name)
        if served is None:
            continue
        for model in served.models.values():
            profile, matched = profile_for(caps, model)
            if not _usable(source, model, matched, served.cli_version):
                continue
            if not matched:
                unprofiled.append(f"{name}:{model.id}")
            # A model whose catalog lists no efforts (Haiku, a local model) is
            # one tier at its own default.
            efforts: tuple[str | None, ...] = model.efforts or (None,)
            if caps.max_effort is not None:
                ceiling = EFFORTS.index(caps.max_effort)
                efforts = tuple(e for e in efforts if e is None or e not in EFFORTS or EFFORTS.index(e) <= ceiling)
            for effort in efforts:
                if (source.backend, model.id, effort) in explicit or any(
                    (source.backend, alias, effort) in explicit for alias in model.aliases
                ):
                    continue
                tier_name = f"{name}:{model.id}" + (f"@{effort}" if effort else "")
                tiers[tier_name] = TierConfig(
                    name=tier_name,
                    backend=source.backend,  # type: ignore[arg-type]
                    model=model.id,
                    base_url=source.base_url,
                    context_window=source.context_window,
                    supports_tools=source.supports_tools,
                    timeout_s=source.timeout_s,
                    effort=effort,
                    subscription=source.subscription,
                )
                pending.append((tier_name, (model.id, *model.aliases), effort, profile, source))
                found.append(Discovered(tier_name, name, model.id, effort, profile.match if matched else "*fallback*"))
    # Explicit subscription/local tiers with no card of their own get one from
    # their profile too, so every served tier is weighed the same way.
    for tier_name, tier in config.tiers.items():
        if tier_name in cards or not tier.can_serve:
            continue
        source = next((s for s in caps.discover.values() if s.backend == tier.backend), None)
        served = report.discovered.get(source.name) if source else None
        model = served.models.get(tier.model) if served else None
        if source and model:
            profile, _ = profile_for(caps, model)
            pending.append((tier_name, (model.id, *model.aliases), tier.effort, profile, source))
    if scores is None:
        for tier_name, _, effort, profile, source in pending:
            cards[tier_name] = card_for(caps, profile, effort, source)
    else:
        keys, _ = served_keys(scores, {ids[0]: ids for _, ids, _, _, _ in pending})
        weights = benchmark_weights(scores, caps)
        line = line_for(caps, scores, [(keys[ids[0]], e, p) for _, ids, e, p, _ in pending], weights)
        for tier_name, ids, effort, profile, source in pending:
            derived = derive(caps, scores, keys[ids[0]], effort, profile, line, weights)
            cards[tier_name] = card_for(caps, profile, effort, source, derived)
    new_caps = replace(caps, cards=cards)
    new_router = replace(config.router, capabilities=new_caps)
    return replace(config, tiers=tiers, router=new_router), found, unprofiled


def model_ids(report: CatalogReport) -> dict[str, tuple[str, ...]]:
    """Served model id -> every id a benchmark point may use for it."""
    return {
        model.id: (model.id, *model.aliases)
        for served in report.discovered.values()
        for model in served.models.values()
    }


def served_ids(found: list[Discovered], report: CatalogReport) -> dict[str, tuple[str, ...]]:
    """`<source>:<model>` -> its ids, for every model that became a tier."""
    out: dict[str, tuple[str, ...]] = {}
    for f in found:
        model = report.discovered[f.source].models[f.model]
        out[f"{f.source}:{f.model}"] = (model.id, *model.aliases)
    return out


__all__ = ["Discovered", "card_for", "expand", "model_ids", "profile_for", "served_ids"]
