"""From benchmark readings to card levels, per (model, effort).

Per requirement r, a tier's ability is the evidence-weighted mean of its
abilities on the benchmarks that measure r, where a benchmark's weight on r is
w_{b,r} (Jev's reading or a manual one, times a fitted scale) times how well
it is linked to the others (rho_b) times what the point says (u). Their sum C_r
is the evidence behind the requirement. The level is

    level_r = lambda x clamp(a + k x ability_r, 0, 3) + (1 - lambda) x profile_r,
    lambda  = C_r / (C_r + profile_weight),

so no evidence is exactly today's card, and a weight moving off zero moves a
level a little, never by a jump. `a` and `k` default to the line through the
profiles' own levels: the evidence reorders the models the operator already
rated, keeping the average and spread of their levels.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .config import EFFORTS, CapabilitiesConfig, ModelProfile
from .scores import Reading, Scores, base_weights, benchmark_weights

# A line needs this many (profile, ability) pairs before it is fitted.
MIN_PAIRS = 3


@dataclass(frozen=True)
class Evidence:
    ability: dict[str, float]  # requirement -> evidence-weighted ability, where C_r > 0
    coverage: dict[str, float]  # requirement -> C_r


@dataclass(frozen=True)
class DerivedCard:
    levels: dict[str, float]
    output_tokens: int
    source: dict[str, str]  # requirement -> "benchmark" | "blended" | "profile"
    coverage: dict[str, float]  # requirement -> C_r


def effort_prior(caps: CapabilitiesConfig, profile: ModelProfile, effort: str | None) -> tuple[dict[str, float], int]:
    """The profile at `effort` by the global rule: what a card is when no evidence covers it."""
    rule = caps.effort_rules.get(effort) if effort else None
    levels = dict(profile.levels)
    output = profile.output_tokens
    if rule is not None:
        for key in caps.thinking:
            if key in levels:
                levels[key] = min(3.0, max(0.0, levels[key] + rule.shift))
        output = max(1, round(output * rule.output))
    return levels, output


def _curve(scores: Scores, bench: str, keys: tuple[str, ...]) -> dict[str | None, Reading]:
    """effort -> reading, over all of the model's keys on `bench`; the earlier key wins at the same effort."""
    curve: dict[str | None, Reading] = {}
    for key in reversed(keys):
        curve.update(scores.readings.get(bench, {}).get(key) or {})
    return curve


def reading_at(scores: Scores, bench: str, keys: tuple[str, ...], effort: str | None) -> tuple[float, float] | None:
    """(ability, evidence) at `effort`: measured, or linear between the nearest measured efforts. Never extrapolated."""
    curve = _curve(scores, bench, keys)
    if effort in curve:
        return curve[effort].t, curve[effort].u
    if effort is None or effort not in EFFORTS:
        return None  # a model without efforts, or an effort with no order: exact matches only
    i = EFFORTS.index(effort)
    known = sorted((EFFORTS.index(e), r) for e, r in curve.items() if e in EFFORTS)
    below = [(j, r) for j, r in known if j < i]
    above = [(j, r) for j, r in known if j > i]
    if not below or not above:
        return None
    (j0, r0), (j1, r1) = below[-1], above[0]
    f = (i - j0) / (j1 - j0)
    return r0.t + (r1.t - r0.t) * f, r0.u + (r1.u - r0.u) * f


def readings_for(
    scores: Scores, keys: tuple[str, ...], effort: str | None, benches
) -> dict[str, tuple[float, float]]:
    """benchmark -> (ability, evidence x rho) of the tier, for each of `benches` that is linked and has a reading."""
    readings: dict[str, tuple[float, float]] = {}
    for bench in benches:
        rho = scores.rho.get(bench, 0.0)
        if rho <= 0:
            continue
        at = reading_at(scores, bench, keys, effort)
        if at is not None:
            readings[bench] = (at[0], at[1] * rho)
    return readings


def combine(requirements, readings: dict[str, tuple[float, float]], weights: dict[str, dict[str, float]]) -> Evidence:
    """The evidence behind each requirement, from a tier's readings and the benchmarks' weights."""
    ability: dict[str, float] = {}
    coverage: dict[str, float] = {}
    for r in requirements:
        c = sum(weights[b].get(r, 0.0) * v for b, (_, v) in readings.items())
        coverage[r] = c
        if c > 0:
            ability[r] = sum(weights[b].get(r, 0.0) * v * t for b, (t, v) in readings.items()) / c
    return Evidence(ability, coverage)


def evidence_for(
    caps: CapabilitiesConfig,
    scores: Scores,
    keys: tuple[str, ...],
    effort: str | None,
    weights: dict[str, dict[str, float]] | None = None,
) -> Evidence:
    """Per requirement, the tier's ability and the evidence behind it. Independent of `a` and `k`."""
    weights = benchmark_weights(scores, caps) if weights is None else weights
    return combine(caps.requirements, readings_for(scores, keys, effort, weights), weights)


def profile_line(pairs: list[tuple[float, float, float]]) -> tuple[float, float]:
    """(a, k) from (profile level, ability, weight) pairs: the line that keeps the profiles' weighted mean and spread."""
    pairs = [p for p in pairs if p[2] > 0]
    if len(pairs) < MIN_PAIRS:
        return 2.0, 1.0
    total = sum(w for _, _, w in pairs)
    mean_p = sum(p * w for p, _, w in pairs) / total
    mean_x = sum(x * w for _, x, w in pairs) / total
    sd_p = math.sqrt(sum(w * (p - mean_p) ** 2 for p, _, w in pairs) / total)
    sd_x = math.sqrt(sum(w * (x - mean_x) ** 2 for _, x, w in pairs) / total)
    if sd_p <= 1e-12 or sd_x <= 1e-12:
        return 2.0, 1.0
    k = max(0.1, sd_p / sd_x)
    return mean_p - k * mean_x, k


def line_pairs(prior: dict[str, float], ev: Evidence) -> list[tuple[float, float, float]]:
    """A tier's (profile level, ability, evidence) pairs, one per covered requirement, for `profile_line`."""
    return [(prior.get(r, 0.0), ev.ability[r], ev.coverage[r]) for r in ev.ability]


def line_for(
    caps: CapabilitiesConfig,
    scores: Scores,
    tiers: list[tuple[tuple[str, ...], str | None, ModelProfile]],
    weights: dict[str, dict[str, float]] | None = None,
) -> tuple[float, float]:
    """The (a, k) every card uses: config values if set, else the profile line plus the fitted offsets."""
    pairs = []
    for keys, effort, profile in tiers:
        prior, _ = effort_prior(caps, profile, effort)
        pairs += line_pairs(prior, evidence_for(caps, scores, keys, effort, weights))
    a_line, k_line = profile_line(pairs)
    a = scores.settings.a if scores.settings.a is not None else a_line + scores.fit.delta_a
    k = scores.settings.k if scores.settings.k is not None else k_line * scores.fit.k_ratio
    return a, k


def blend(ev: Evidence, prior: dict[str, float], requirements, *, a: float, k: float, c0: float) -> dict[str, float]:
    levels: dict[str, float] = {}
    for r in requirements:
        base = prior.get(r)
        c = ev.coverage.get(r, 0.0)
        if c <= 0:
            if base is not None:
                levels[r] = base
            continue
        derived = min(3.0, max(0.0, a + k * ev.ability[r]))
        lam = c / (c + c0)
        levels[r] = lam * derived + (1 - lam) * (base or 0.0)
    return levels


def _source(c: float, c0: float) -> str:
    if c <= 0:
        return "profile"
    # At lambda >= 0.8 the measurement carries four fifths of the level or more.
    return "benchmark" if c / (c + c0) >= 0.8 else "blended"


def _output_tokens(scores: Scores, keys: tuple[str, ...], effort: str | None, profile: ModelProfile, fallback: int) -> int:
    """The profile's output at `high`, times the median cost ratio effort/high the evidence publishes."""
    if effort is None:
        return fallback
    ratios = []
    for bench in scores.benchmarks:
        for key in keys:
            at, ref = scores.point_at(bench, key, effort), scores.point_at(bench, key, "high")
            if at and ref and (at.cost_usd or 0) > 0 and (ref.cost_usd or 0) > 0:  # 0: not a published cost
                ratios.append(at.cost_usd / ref.cost_usd)
                break
    if not ratios:
        return fallback
    return max(1, round(profile.output_tokens * statistics.median(ratios)))


def derive(
    caps: CapabilitiesConfig,
    scores: Scores,
    keys: tuple[str, ...],
    effort: str | None,
    profile: ModelProfile,
    line: tuple[float, float],
    weights: dict[str, dict[str, float]] | None = None,
) -> DerivedCard:
    """A card's levels and output for (model, effort), from its evidence and its profile. Pure.

    `weights`: `benchmark_weights`, when the caller already has them for many cards.
    """
    prior, fallback_output = effort_prior(caps, profile, effort)
    ev = evidence_for(caps, scores, keys, effort, weights)
    levels = blend(ev, prior, caps.requirements, a=line[0], k=line[1], c0=scores.c0)
    return DerivedCard(
        levels=levels,
        output_tokens=_output_tokens(scores, keys, effort, profile, fallback_output),
        source={r: _source(ev.coverage[r], scores.c0) for r in caps.requirements},
        coverage=ev.coverage,
    )


def link_state(scores: Scores, bench: str) -> str | None:
    """"detached" (off the main scale: counts for nothing), "unlinked", "thin", or None when well linked."""
    if bench in scores.scale.detached:
        return "detached"
    links = scores.links.get(bench, 0)
    return None if links >= 3 else ("unlinked" if links == 0 else "thin")


def served_keys(scores: Scores, served: dict[str, tuple[str, ...]]) -> tuple[dict[str, tuple[str, ...]], set[str]]:
    """(served name -> its model keys, ambiguous keys). A key two different models share is used for neither.

    A model is its first id without a date suffix: one model served at several
    efforts, or by two sources as a dated snapshot and its undated id, is one.
    Different models are told apart within one catalog -- the part of a served
    name before ":" -- because a catalog does not list one model under two
    spellings, while two catalogs routinely do (`vendor/x-5.5` beside `x-5-5`).
    So two ids sharing a key are ambiguous only when one catalog serves both.
    """
    owners: dict[str, dict[str, set[str]]] = {}
    for name, ids in served.items():
        scope = name.split(":", 1)[0] if ":" in name else ""
        for i in ids:
            owners.setdefault(scores.key(i), {}).setdefault(scope, set()).add(scores.keys.undated(ids[0].strip()))
    ambiguous = {k for k, by_scope in owners.items() if any(len(names) > 1 for names in by_scope.values())}
    out = {
        name: tuple(dict.fromkeys(k for k in (scores.key(i) for i in ids) if k not in ambiguous))
        for name, ids in served.items()
    }
    return out, ambiguous


def evidence_summary(scores: Scores, keys: tuple[str, ...]) -> dict[str, tuple[set[str | None], set[str]]]:
    """benchmark -> (efforts, origins) of the points behind these keys, usable or not."""
    out: dict[str, tuple[set[str | None], set[str]]] = {}
    wanted = set(keys)
    for p in scores.points:
        if scores.key(p.model) in wanted:
            efforts, origins = out.setdefault(p.benchmark, (set(), set()))
            efforts.add(p.effort)
            origins.add(p.origin)
    return out


def some(names, limit: int = 12) -> str:
    """A comma list of at most `limit` names, with how many more: a catalog of hundreds is not a log line."""
    names = list(names)
    more = f" (+{len(names) - limit} more)" if len(names) > limit else ""
    return ", ".join(names[:limit]) + more


def startup_lines(scores: Scores, caps: CapabilitiesConfig, served: dict[str, tuple[str, ...]]) -> list[str]:
    """What the operator should know: models with no or only vendor evidence, and entries doing nothing.

    `served`: served model name -> the ids a point may use for it (see `discovery.served_ids`).
    """
    lines: list[str] = []
    keys, ambiguous = served_keys(scores, served)
    bare, vendor_only = [], []
    for name, ks in sorted(keys.items()):
        summary = evidence_summary(scores, ks)
        origins = set().union(*(o for _, o in summary.values())) if summary else set()
        if not origins:
            bare.append(name)
        elif origins == {"vendor"}:
            vendor_only.append(name)
    if bare:
        lines.append(f"benchmarks: {len(bare)} served model(s) with no evidence: {some(bare)}")
    if vendor_only:
        lines.append(f"benchmarks: {len(vendor_only)} served model(s) with vendor evidence only: "
                     f"{some(vendor_only)}")
    unread = [b for b in scores.benchmarks if b not in base_weights(scores, caps)]
    if unread:
        lines.append(f"benchmarks: unread (run `jev-model-router benchmarks read`): {some(unread)}")
    if scores.scale.detached:
        lines.append("benchmarks: not linked to the main scale (no model shared with it), counting for nothing: "
                     + ", ".join(scores.scale.detached))
    if ambiguous:
        lines.append(f"benchmarks: ambiguous model key(s), used for no tier: {some(sorted(ambiguous))}")
    if scores.keys.keep_dates:
        lines.append("benchmarks: dated snapshot(s) kept apart, a served alias may no longer match: "
                     + ", ".join(sorted(scores.keys.keep_dates)))
    for error in scores.errors:
        lines.append(f"benchmarks: unreadable, ignoring it: {error}")
    return lines


__all__ = [
    "DerivedCard", "Evidence", "blend", "combine", "derive", "effort_prior", "evidence_for", "evidence_summary",
    "line_for", "line_pairs", "link_state", "profile_line", "reading_at", "readings_for", "served_keys", "startup_lines",
]
