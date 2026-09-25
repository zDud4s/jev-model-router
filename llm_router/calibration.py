"""Calibration from anchors: what the operator KNOWS is enough, turned into the one number that moves.

The cards are priors, and priors are wrong in a direction a person notices at
once: on 2026-09-24 the router sent a Windows debugging task to Claude Fable 5.1
at high effort, and the operator's answer was that Opus 5.5 at medium does such
tasks every day and is more than enough. That sentence is evidence. This module
takes sentences like it -- a task, and a tier known to be enough (or known NOT to
be) -- and finds the `miss_scale` that honours them.

`miss_scale` multiplies every entry of the `miss` table: success on a tier is
prod_r (1 - n_r * miss_scale * miss[level]). It is monotone -- a smaller scale
makes every tier more likely to succeed -- so each anchor bounds it:

* `sufficient: T` -- T must reach the target on that task, by `margin`:
  scale <= s_max. The margin is there because Jev's reading of the same task
  moves by a few hundredths between calls; a scale set exactly on the edge
  sends the anchor's own task to the dearer tier half the time.
* `insufficient: T` -- T must fall short: scale >= s_min.

The scale kept is the largest one every `sufficient` anchor allows, capped at 1
(anchors can make the router trust models more, never less than the priors on
their own say). A tier whose family has a fitted scale (`family_scales`) bounds
nothing: the global scale never reaches it.

An `insufficient` anchor that tier still reaches (above `target - margin`) is
then enforced on ONE requirement. No scale can do it: the `sufficient` anchors
often name the same family, and a scale moves every requirement at once. The
requirement is the anchor's `because:`, or else the weakest link: the need
that takes the most off the product. Its level is capped for the whole family
at the highest value that brings the tier under the line. A cap that would
break a `sufficient` anchor is not kept, and is reported instead. The scale
comes first and the caps second. The scale only raises trust and a cap only
lowers one level, so a second run over the same anchors changes nothing.

Each anchor costs one Jev call; no destination model runs.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .capabilities import CapabilityRouter
from .config import FALLBACK_KEY, Config, ConfigError, capped
from .schemas import ChatCompletionRequest


@dataclass
class AnchorResult:
    task: str
    sufficient: list[str]
    insufficient: list[str]
    needs: dict[str, float]
    bounds: dict[str, tuple[str, float]]  # tier -> ("<=" or ">=", scale); tiers the global scale reaches
    picked_after: str | None = None
    because: str | None = None
    # (tier, family key, requirement, old level, new level, "because" | "weakest link")
    capped: list[tuple[str, str, str, float, float, str]] = field(default_factory=list)
    # Other tiers of the capped family the cap also lowers: (tier, requirement, before, after).
    # A cap is family-wide, so an anchor about one effort flattens the higher ones too.
    lowered: list[tuple[str, str, float, float]] = field(default_factory=list)
    # Tiers with a fitted family scale: the global scale never reaches them, so they bound nothing.
    family_scaled: dict[str, float] = field(default_factory=dict)


def load_anchors(path: str | Path, requirements: dict[str, str] | None = None) -> list[dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{path}: anchors must be a non-empty list")
    for i, anchor in enumerate(raw):
        if not isinstance(anchor, dict) or not anchor.get("task"):
            raise ConfigError(f"{path}: anchor {i} needs a 'task'")
        if not anchor.get("sufficient") and not anchor.get("insufficient"):
            raise ConfigError(f"{path}: anchor {i} names no sufficient or insufficient tier")
        because = anchor.get("because")
        if because is not None:
            if not isinstance(because, str):
                raise ConfigError(f"{path}: anchor {i}: 'because' is one requirement key, got {because!r}")
            if not anchor.get("insufficient"):
                # It would be silently ignored: it says which requirement an insufficient tier lacks.
                raise ConfigError(f"{path}: anchor {i} has 'because' but no insufficient tier")
            if requirements is not None and because not in requirements:
                raise ConfigError(f"{path}: anchor {i}: 'because' names unknown requirement {because!r}")
    return raw


def _tiers(value: Any) -> list[str]:
    return [value] if isinstance(value, str) else list(value or [])


def _scale_bound(router: CapabilityRouter, needs: dict[str, float], tier: str, target: float) -> float:
    """The largest scale at which `tier` still reaches `target` on these needs."""
    if router.success(needs, tier, scale=0.0) < target:
        return 0.0  # cannot reach it at any scale
    lo, hi = 0.0, 4.0
    if router.success(needs, tier, scale=hi) >= target:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        if router.success(needs, tier, scale=mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo


@dataclass
class Calibration:
    scale: float
    caps: dict[str, dict[str, float]]  # every level cap: the config's, merged with the new ones (lower wins)
    results: list[AnchorResult]
    conflicts: list[str]


def merge_caps(*layers: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for layer in layers:
        for family, row in layer.items():
            for req, value in row.items():
                held = out.setdefault(family, {}).get(req)
                out[family][req] = value if held is None else min(held, value)
    return out


def with_caps(config: Config, level_caps: dict[str, dict[str, float]]) -> Config:
    """The config with `level_caps` set and applied to every card, as parsing and discovery apply them."""
    caps = config.router.capabilities
    assert caps is not None
    cards = {
        name: replace(card, levels=capped(card.levels, level_caps.get(card.family or name)))
        for name, card in caps.cards.items()
    }
    return replace(config, router=replace(config.router, capabilities=replace(caps, level_caps=level_caps, cards=cards)))


def _blame(router: CapabilityRouter, result: AnchorResult, tier: str, levels: dict[str, float], scale: float):
    """(requirement, how): the anchor's `because`, or the need that takes the most off the product."""
    if result.because:
        return result.because, "because"
    caps = router._caps
    best, most = None, 0.0
    for req in caps.requirements:  # config order breaks ties
        strength = max(0.0, result.needs.get(req, 0.0) - caps.floor) / (1.0 - caps.floor)
        loss = strength * min(1.0, scale * router._miss(levels.get(req, 0.0)))
        if loss > most:
            best, most = req, loss
    return best, "weakest link"


def _cap_level(success_at, current: float, line: float) -> float | None:
    """The highest level in [0, current] at which success is at or under `line`; None if even 0 is above it."""
    if success_at(0.0) > line:
        return None
    lo, hi = 0.0, current
    for _ in range(60):
        mid = (lo + hi) / 2
        if success_at(mid) <= line:
            lo = mid
        else:
            hi = mid
    return lo


async def calibrate(
    config: Config, anchors: list[dict[str, Any]], router: CapabilityRouter, margin: float = 0.03
) -> Calibration:
    """Fit `miss_scale` from the `sufficient` anchors, then cap one requirement per violated `insufficient` one.

    A tier whose family has a fitted scale bounds nothing: `success` uses its
    family scale, so the global scale never reaches it (`family_scaled`). Its
    anchors still count for the capping and its protection, under that scale.
    """
    from .capabilities import build_packet

    caps = config.router.capabilities
    assert caps is not None
    enough = min(0.999, caps.target + margin)
    short = caps.target - margin
    results: list[AnchorResult] = []
    for anchor in anchors:
        request = ChatCompletionRequest.model_validate({
            "model": "auto",
            "messages": [{"role": "user", "content": anchor["task"]}],
            **({"packet": anchor["packet"]} if anchor.get("packet") else {}),
        })
        packet = build_packet(request, max_chars=caps.max_packet_chars)
        needs = await router._ask(packet, caps.requirements)
        result = AnchorResult(anchor["task"], _tiers(anchor.get("sufficient")), _tiers(anchor.get("insufficient")),
                              needs, {}, because=anchor.get("because"))
        for tier in result.sufficient + result.insufficient:
            if tier not in caps.cards:
                raise ConfigError(f"anchor names {tier!r}, which is not a carded tier (is it discovered?)")
            if router.family_key(tier) in caps.family_scales:
                # Its own scale decides it, not miss_scale: a bound from it would move the global
                # scale for tiers it says nothing about, and move again once a cap lowers its levels.
                result.family_scaled[tier] = caps.family_scales[router.family_key(tier)]
            elif tier in result.sufficient:
                result.bounds[tier] = ("<=", _scale_bound(router, needs, tier, enough))
            else:
                result.bounds[tier] = (">=", _scale_bound(router, needs, tier, short))
        results.append(result)

    uppers = [s for r in results for op, s in r.bounds.values() if op == "<="]
    scale = min([1.0, *uppers])

    new: dict[str, dict[str, float]] = {}

    def levels(tier: str, trial: dict[str, float] | None = None) -> dict[str, float]:
        row = {**new.get(router.family_key(tier), {}), **(trial or {})}
        return capped(caps.cards[tier].levels, row)

    def p(needs: dict[str, float], tier: str, trial: dict[str, float] | None = None) -> float:
        return router.success(needs, tier, scale=router.scale_for(tier, scale), levels=levels(tier, trial))

    conflicts: list[str] = []
    for result in results:
        for tier in result.insufficient:
            if p(result.needs, tier) <= short:
                continue
            where = f"{tier} should fall short on {result.task[:60]!r}"
            key = router.family_key(tier)
            if key == FALLBACK_KEY:
                # A cap under the fallback's key would cap every unprofiled model, future ones too.
                conflicts.append(f"{where}, but its model is on the fallback profile: write a profile for it, "
                                 f"so a cap names that model alone")
                continue
            req, how = _blame(router, result, tier, levels(tier), router.scale_for(tier, scale))
            if req is None:
                conflicts.append(f"{where}, but Jev read no need in it to cap")
                continue
            now = levels(tier).get(req, 0.0)
            level = _cap_level(lambda v: p(result.needs, tier, {req: v}), now, short)
            if level is None:
                conflicts.append(f"{where}, and even {req} at 0 does not get it there")
                continue
            # The value written (floored, so still under the line), checked and reported as written.
            level = floor_cap(level)
            broken = [
                (other, t) for other in results for t in other.sufficient
                if router.family_key(t) == key
                and p(other.needs, t) >= enough - 1e-9 > p(other.needs, t, {req: level})
            ]
            if broken:
                other, t = broken[0]
                conflicts.append(f"{where}, but capping {key} {req} at {level:.2f} would break {t} on "
                                 f"{other.task[:60]!r}, which is enough; not capped")
                continue
            for other_tier in sorted(caps.cards):
                if other_tier != tier and router.family_key(other_tier) == key:
                    before = levels(other_tier).get(req, 0.0)
                    if level < before:
                        result.lowered.append((other_tier, req, before, level))
            new.setdefault(key, {})[req] = level
            result.capped.append((tier, key, req, now, level, how))
    return Calibration(scale, merge_caps(caps.level_caps, new), results, conflicts)


def with_scale(config: Config, scale: float) -> Config:
    caps = config.router.capabilities
    assert caps is not None
    return replace(config, router=replace(config.router, capabilities=replace(caps, miss_scale=scale)))


def _set_scale(text: str, scale: float, path: str | Path) -> str:
    line = f"miss_scale: {scale:.3f}"
    if re.search(r"^(\s*)miss_scale:.*$", text, flags=re.M):
        return re.sub(r"^(\s*)miss_scale:.*$", lambda m: m.group(1) + line, text, count=1, flags=re.M)
    match = re.search(r"^(\s*)miss:\s*\[.*\]\s*$", text, flags=re.M)
    if not match:
        raise ConfigError(f"{path}: no `miss:` line under router.capabilities to put miss_scale beside")
    indent = match.group(1)
    insert = f"\n{indent}# Set by `llm-router calibrate`: every miss entry is multiplied by it.\n{indent}{line}"
    return text[: match.end()] + insert + text[match.end():]


def write_scale(path: str | Path, scale: float) -> None:
    """Set `miss_scale` in the config file, keeping every comment around it."""
    target = Path(path)
    target.write_text(_set_scale(target.read_text(encoding="utf-8"), scale, path), encoding="utf-8")


def write_calibration(path: str | Path, scale: float, level_caps: dict[str, dict[str, float]]) -> None:
    """Write `miss_scale` and merge `level_caps` in one write: either both land or, on a ConfigError, neither."""
    target = Path(path)
    text = _set_scale(target.read_text(encoding="utf-8"), scale, path)
    if level_caps:
        text = _set_level_caps(text, level_caps, path)
    target.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- from outcomes
#
# Anchors are what the operator believes; the log is what happened. Every
# routed request whose answer was judged carries the needs Jev read (in
# `route_reason`), the tier it went to and a verdict -- one labelled trial of
# P(success | needs, tier). A request the client resent with `failed_tiers` is a
# second source: each tier it names failed THAT task, whose needs the resend's
# own row holds. A runner that routes through /v1/route reports its gate's
# verdict as the decision's outcome: a third, and the one that matters most. Enough trials per family and that family gets its own scale,
# the one under which the observed passes and failures are most likely.


@dataclass
class Outcome:
    tier: str
    needs: dict[str, float]
    passed: bool
    source: str  # "verdict" or "client"


@dataclass
class FamilyFit:
    family: str
    n: int
    passes: int
    scale: float | None  # None: too few trials to trust
    predicted: float  # mean estimated success under the fitted (or current) scale


def log_outcomes(log: Any, config: Config) -> list[Outcome]:
    caps = config.router.capabilities
    assert caps is not None
    rows = log.query(
        """
        SELECT r.tier, r.route_reason, v.verdict
        FROM requests r LEFT JOIN verifications v ON v.request_row_id = r.id
        WHERE r.route_model LIKE 'capabilities:%' AND r.route_reason IS NOT NULL
        """
    )
    out: list[Outcome] = []
    for row in rows:
        try:
            reason = json.loads(row["route_reason"])
        except (TypeError, ValueError):
            continue
        needs = reason.get("need") if isinstance(reason, dict) else None
        if not isinstance(needs, dict) or not needs:
            continue
        needs = {str(k): float(v) for k, v in needs.items()}
        if row["verdict"] in ("pass", "fail") and row["tier"] in caps.cards:
            out.append(Outcome(row["tier"], needs, row["verdict"] == "pass", "verdict"))
        for tier in reason.get("skipped_failed") or []:
            if tier in caps.cards:
                out.append(Outcome(tier, needs, False, "client"))
    decisions = log.query(
        """
        SELECT tier, route_reason, outcome FROM route_decisions
        WHERE route_model LIKE 'capabilities:%' AND route_reason IS NOT NULL
        """
    )
    for row in decisions:
        try:
            reason = json.loads(row["route_reason"])
        except (TypeError, ValueError):
            continue
        needs = reason.get("need") if isinstance(reason, dict) else None
        if not isinstance(needs, dict) or not needs:
            continue
        needs = {str(k): float(v) for k, v in needs.items()}
        if row["outcome"] in ("pass", "fail") and row["tier"] in caps.cards:
            out.append(Outcome(row["tier"], needs, row["outcome"] == "pass", "outcome"))
        for tier in reason.get("skipped_failed") or []:
            if tier in caps.cards:
                out.append(Outcome(tier, needs, False, "client"))
    return out


def _loglik(router: CapabilityRouter, trials: list[Outcome], scale: float) -> float:
    total = 0.0
    for t in trials:
        p = min(1 - 1e-4, max(1e-4, router.success(t.needs, t.tier, scale=scale)))
        total += math.log(p if t.passed else 1 - p)
    return total


def fit_family_scales(router: CapabilityRouter, outcomes: list[Outcome], min_samples: int = 30) -> list[FamilyFit]:
    """One scale per family, by maximum likelihood on its trials; pure Python, no numpy."""
    cards = router._caps.cards
    by_family: dict[str, list[Outcome]] = {}
    for o in outcomes:
        by_family.setdefault(cards[o.tier].family or o.tier, []).append(o)
    fits: list[FamilyFit] = []
    for family, trials in sorted(by_family.items()):
        passes = sum(t.passed for t in trials)
        scale: float | None = None
        if len(trials) >= min_samples:
            # A coarse log-spaced grid, then golden-section inside the best cell.
            grid = [0.05 * (80 ** (i / 60)) for i in range(61)]  # 0.05 .. 4.0
            best = max(range(len(grid)), key=lambda i: _loglik(router, trials, grid[i]))
            lo, hi = grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]
            g = (math.sqrt(5) - 1) / 2
            for _ in range(40):
                a, b = hi - g * (hi - lo), lo + g * (hi - lo)
                if _loglik(router, trials, a) >= _loglik(router, trials, b):
                    hi = b
                else:
                    lo = a
            scale = (lo + hi) / 2
        predicted = sum(router.success(t.needs, t.tier, scale=scale) for t in trials) / len(trials)
        fits.append(FamilyFit(family, len(trials), passes, scale, predicted))
    return fits


def write_family_scales(path: str | Path, scales: dict[str, float]) -> None:
    """Set `family_scales` in the config file, as one flow mapping beside `miss_scale`."""
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    line = "family_scales: " + json.dumps({k: round(v, 3) for k, v in sorted(scales.items())})
    held = _flow_line(text, "family_scales", path)
    if held:
        text = text[: held.start()] + held.group(1) + line + text[held.end():]
    else:
        match = re.search(r"^(\s*)miss_scale:.*$", text, flags=re.M) or re.search(
            r"^(\s*)miss:\s*\[.*\]\s*$", text, flags=re.M
        )
        if not match:
            raise ConfigError(f"{path}: no `miss:` line under router.capabilities to put family_scales beside")
        indent = match.group(1)
        insert = (f"\n{indent}# Fitted by `llm-router calibrate --from-log`: a family here ignores miss_scale."
                  f"\n{indent}{line}")
        text = text[: match.end()] + insert + text[match.end():]
    target.write_text(text, encoding="utf-8")


def _flow_line(text: str, key: str, path: str | Path) -> re.Match[str] | None:
    """The `key:` line, which must hold its whole value on that line (the flow form these writers write).

    A block-form value, on the lines below, would be left behind under the new
    line and corrupt the file, so it is refused instead.
    """
    match = re.search(rf"^([ \t]*){key}:(.*)$", text, flags=re.M)
    if match is None:
        return None
    value = match.group(2).split("#", 1)[0].strip()
    below = text[match.end():].splitlines()
    nxt = next((ln for ln in below if ln.strip() and not ln.strip().startswith("#")), "")
    deeper = len(nxt) - len(nxt.lstrip()) > len(match.group(1))
    if not value and deeper:
        raise ConfigError(f"{path}: `{key}:` is written as a block over several lines; write it as one flow "
                          f"mapping on one line (e.g. {key}: {{\"family\": {{\"requirement\": 1.0}}}}) and rerun")
    return match


def _set_level_caps(text: str, level_caps: dict[str, dict[str, float]], path: str | Path) -> str:
    match = _flow_line(text, "level_caps", path)
    try:
        held = (yaml.safe_load(match.group(2)) or {}) if match else {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: the `level_caps:` line is not one flow mapping: {exc}") from None
    if not isinstance(held, dict):
        raise ConfigError(f"{path}: the `level_caps:` line is not a mapping")
    merged = merge_caps(held, level_caps)
    row = {f: {r: floor_cap(v) for r, v in sorted(reqs.items())} for f, reqs in sorted(merged.items())}
    line = "level_caps: " + json.dumps(row)
    if match:
        return text[: match.start()] + match.group(1) + line + text[match.end():]
    anchor = (re.search(r"^(\s*)family_scales:.*$", text, flags=re.M)
              or re.search(r"^(\s*)miss_scale:.*$", text, flags=re.M)
              or re.search(r"^(\s*)miss:\s*\[.*\]\s*$", text, flags=re.M))
    if not anchor:
        raise ConfigError(f"{path}: no `miss:` line under router.capabilities to put level_caps beside")
    indent = anchor.group(1)
    insert = (f"\n{indent}# Set by `llm-router calibrate --anchors` from insufficient anchors: a family's"
              f"\n{indent}# ceiling on one requirement, applied last to every card.\n{indent}{line}")
    return text[: anchor.end()] + insert + text[anchor.end():]


def floor_cap(value: float) -> float:
    """A ceiling as written: floored to 3 places, never rounded up over the line it keeps the tier under."""
    # +1e-9: 1.001 * 1000 is 1000.9999999999999 in floating point.
    return math.floor(value * 1000 + 1e-9) / 1000


def write_level_caps(path: str | Path, level_caps: dict[str, dict[str, float]]) -> None:
    """Merge `level_caps` into the config file (the lower ceiling wins), as one line beside `family_scales`.

    Only the one-line flow form this writes is merged; a multi-line block
    `level_caps:` is refused with a ConfigError rather than corrupted.
    """
    target = Path(path)
    target.write_text(_set_level_caps(target.read_text(encoding="utf-8"), level_caps, path), encoding="utf-8")


__all__ = [
    "AnchorResult", "Calibration", "FamilyFit", "Outcome", "calibrate", "fit_family_scales", "load_anchors",
    "floor_cap", "log_outcomes", "merge_caps", "with_caps", "with_scale", "write_calibration",
    "write_family_scales", "write_level_caps", "write_scale",
]
