"""Discovery: every model a provider serves, at every effort it takes, becomes a tier.

The catalog is not a list someone keeps. `discover` names providers -- the Claude
Code subscription, the Codex subscription, a local Ollama, any API with an
OpenAI-compatible `/models` listing -- and at startup each
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

A listing of hundreds of models is the exception to "nothing is excluded": most
of it is models no one has measured, and on the fallback profile they would win
on price alone. So a source may `require_evidence` (a profile or benchmark
points), and `include`/`exclude` globs narrow it; both are config, not code.
What the listing states per model -- context, prices, tools -- is used as stated.

Performance is the one thing no catalog publishes, so profiles are priors. The
router logs every decision with Jev's reading and the card fingerprint, and that
is what recalibrates them.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, replace

from .catalog import CatalogReport, Offered, _older
from .config import (
    EFFORTS,
    CapabilitiesConfig,
    Config,
    DiscoverSource,
    ModelCard,
    ModelProfile,
    Prices,
    TierConfig,
    capped,
)
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
    names = ids_of(model)  # a variant takes the profile of the model it is
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
    levels = capped(levels, caps.level_caps.get(profile.match))  # last: after the effort rule or the benchmark blend
    return ModelCard(
        levels=levels,
        output_tokens=output,
        input_overhead=source.input_overhead,
        list_prices=profile.list_prices,
        family=profile.match,
    )


def ids_of(model: Offered) -> tuple[str, ...]:
    """Every id benchmark evidence may use for this model, its identity first.

    A variant (`x:free`) or alias (`~x-latest`) is the model it names, so that
    one leads: two sources serving the same model agree on who owns its key.
    """
    return tuple(dict.fromkeys(i for i in (model.same_as, model.id, *model.aliases) if i))


def _selected(source: DiscoverSource, model: Offered) -> bool:
    """The source's include/exclude globs, over every id the model goes by."""
    names = (model.id, *model.aliases)
    if source.include and not any(fnmatch.fnmatchcase(n, g) for n in names for g in source.include):
        return False
    return not any(fnmatch.fnmatchcase(n, g) for n in names for g in source.exclude)


def _body_for(source: DiscoverSource, effort: str | None) -> dict:
    """The source's extra_body, with the effort written in where its effort_body says."""
    body = dict(source.extra_body)
    if effort is not None and source.effort_body is not None:

        def fill(value):
            if isinstance(value, dict):
                return {k: fill(v) for k, v in value.items()}
            return effort if value == "{effort}" else value

        body.update(fill(source.effort_body))
    return body


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
    if model.unpriceable:
        return False  # priced per call by something unstated: never costable before it
    return True


def _explicit(
    config: Config, report: CatalogReport
) -> list[tuple[str, Offered, str | None, ModelProfile, DiscoverSource]]:
    """(tier, served model, effort, profile, source) for each explicit tier whose card `expand` builds.

    Explicit subscription/local tiers with no card of their own get one from
    their profile too, so every served tier is weighed the same way.
    """
    caps = config.router.capabilities
    assert caps is not None
    out = []
    for tier_name, tier in config.tiers.items():
        if tier_name in caps.cards or not tier.can_serve:
            continue
        source = next((s for s in caps.discover.values() if s.backend == tier.backend), None)
        served = report.discovered.get(source.name) if source else None
        model = served.models.get(tier.model) if served else None
        if source and model:
            out.append((tier_name, model, tier.effort, profile_for(caps, model)[0], source))
    return out


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
            if not _selected(source, model):
                continue
            profile, matched = profile_for(caps, model)
            if not _usable(source, model, matched, served.cli_version):
                continue
            # A model whose catalog lists no efforts (Haiku, a local model) is
            # one tier at its own default. So is an API model when the source
            # does not say how to send an effort.
            efforts: tuple[str | None, ...] = model.efforts or (None,)
            if source.subscription is None and source.backend != "ollama" and source.effort_body is None:
                efforts = (None,)
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
                    api_key_env=source.api_key_env,
                    context_window=model.context_window or source.context_window,
                    supports_tools=source.supports_tools if model.supports_tools is None else model.supports_tools,
                    # The listing's own price, so the log costs what was billed;
                    # the card's list_prices is the profile's guess. Through
                    # `Prices.parse`, so a cache price the listing leaves out is
                    # the input rate, not free.
                    prices=Prices.parse(model.prices) if model.prices else Prices(),
                    timeout_s=source.timeout_s,
                    extra_body=_body_for(source, effort),
                    effort=effort,
                    subscription=source.subscription,
                )
                pending.append((tier_name, ids_of(model), effort, profile, source))
                found.append(Discovered(tier_name, name, model.id, effort, profile.match if matched else "*fallback*"))
    pending += [(t, ids_of(m), e, p, s) for t, m, e, p, s in _explicit(config, report)]
    pending = _with_evidence(caps, scores, pending, tiers, found)
    unprofiled += [f"{d.source}:{d.model}" for d in found if d.profile == "*fallback*"]
    unprofiled = list(dict.fromkeys(unprofiled))
    if scores is None:
        for tier_name, _, effort, profile, source in pending:
            cards[tier_name] = card_for(caps, profile, effort, source)
    else:
        keys, _ = served_keys(scores, _served(pending))
        weights = benchmark_weights(scores, caps)
        line = line_for(caps, scores, [(keys[_named(s, ids)], e, p) for _, ids, e, p, s in pending], weights)
        for tier_name, ids, effort, profile, source in pending:
            derived = derive(caps, scores, keys[_named(source, ids)], effort, profile, line, weights)
            cards[tier_name] = card_for(caps, profile, effort, source, derived)
    new_caps = replace(caps, cards=cards)
    new_router = replace(config.router, capabilities=new_caps)
    return replace(config, tiers=tiers, router=new_router), found, unprofiled


def _named(source: DiscoverSource, ids: tuple[str, ...]) -> str:
    """`<source>:<model>`: the served name `served_keys` scopes ambiguity by."""
    return f"{source.name}:{ids[0]}"


def _served(pending) -> dict[str, tuple[str, ...]]:
    return {_named(source, ids): ids for _, ids, _, _, source in pending}


def _with_evidence(caps, scores, pending, tiers, found):
    """`pending` without the tiers of a `require_evidence` source that nothing speaks for.

    Evidence is a profile that names the model, or a benchmark point under one
    of its keys. The dropped tiers leave `tiers` and `found` too.
    """
    strict = {name for name, source in caps.discover.items() if source.require_evidence}
    if not strict:
        return pending
    evidenced: set[str] = set()
    if scores is not None:
        keys, _ = served_keys(scores, _served(pending))
        with_points = {scores.key(p.model) for p in scores.points}
        evidenced = {name for name, own in keys.items() if set(own) & with_points}
    source_of = {d.tier: d.source for d in found}
    kept, dropped = [], set()
    for entry in pending:
        tier_name, ids, _, profile, source = entry
        if (source_of.get(tier_name) in strict and profile is caps.fallback_profile
                and _named(source, ids) not in evidenced):
            dropped.add(tier_name)
            continue
        kept.append(entry)
    for tier_name in dropped:
        tiers.pop(tier_name, None)
    found[:] = [d for d in found if d.tier not in dropped]
    return kept


def model_ids(report: CatalogReport) -> dict[str, tuple[str, ...]]:
    """Served model id -> every id a benchmark point may use for it, its identity first."""
    return {
        model.id: ids_of(model)
        for served in report.discovered.values()
        for model in served.models.values()
    }


def derived_tiers(
    config: Config, scores: Scores, ids_by_model: dict[str, tuple[str, ...]]
) -> dict[str, tuple[tuple[str, ...], str | None, ModelProfile]]:
    """tier -> (model keys, effort, profile), for every card `expand` derived in this expanded config.

    Those are the cards with a family: a card written in the config has none.
    The same set `expand` fits its line over, so a caller's line is serving's.
    """
    caps = config.router.capabilities
    assert caps is not None
    tiers = {n: t for n, t in config.tiers.items() if n in caps.cards and caps.cards[n].family is not None}
    ids = {n: ids_by_model.get(t.model, (t.model,)) for n, t in tiers.items()}
    keys, _ = served_keys(scores, ids)
    return {
        n: (keys[n], tiers[n].effort, profile_for(caps, Offered(i[0], aliases=tuple(i[1:])))[0])
        for n, i in ids.items()
    }


def derived_cards(
    config: Config, scores: Scores, ids_by_model: dict[str, tuple[str, ...]]
) -> tuple[tuple[float, float], dict[str, DerivedCard]]:
    """(line, tier -> its derivation) for every card `expand` derived in this expanded config.

    The derivation carries each requirement's source and evidence C; the levels
    served are the expanded config's cards, which have the level caps applied.
    """
    caps = config.router.capabilities
    assert caps is not None
    tiers = derived_tiers(config, scores, ids_by_model)
    weights = benchmark_weights(scores, caps)
    line = line_for(caps, scores, list(tiers.values()), weights)
    return line, {t: derive(caps, scores, k, e, p, line, weights) for t, (k, e, p) in tiers.items()}


def served_ids(
    found: list[Discovered], report: CatalogReport, config: Config | None = None
) -> dict[str, tuple[str, ...]]:
    """`<source>:<model>` -> its ids, for every model that became a tier.

    With `config` (the one given to `expand`, before it), the explicit tiers
    whose cards `expand` derives count too: the same set of models it derives.
    """
    out: dict[str, tuple[str, ...]] = {}
    for f in found:
        model = report.discovered[f.source].models[f.model]
        out[f"{f.source}:{f.model}"] = ids_of(model)
    caps = config.router.capabilities if config is not None else None
    if caps is not None and caps.discover:
        for _, model, _, _, source in _explicit(config, report):
            out.setdefault(f"{source.name}:{model.id}", ids_of(model))
    return out


def unused_caps(config: Config) -> list[str]:
    """`level_caps` keys no card has: not an error, since a model can leave the catalog."""
    caps = config.router.capabilities
    if caps is None:
        return []
    keys = {card.family or name for name, card in caps.cards.items()}
    return sorted(set(caps.level_caps) - keys)


__all__ = [
    "Discovered", "card_for", "derived_cards", "derived_tiers", "expand", "model_ids", "profile_for", "served_ids", "unused_caps",
]
