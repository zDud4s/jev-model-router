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
their own say). An `insufficient` anchor that this violates is reported as a
conflict rather than silently won.

Each anchor costs one Jev call; no destination model runs.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from .capabilities import CapabilityRouter
from .config import Config, ConfigError
from .schemas import ChatCompletionRequest


@dataclass
class AnchorResult:
    task: str
    sufficient: list[str]
    insufficient: list[str]
    needs: dict[str, float]
    bounds: dict[str, tuple[str, float]]  # tier -> ("<=" or ">=", scale)
    picked_after: str | None = None


def load_anchors(path: str | Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{path}: anchors must be a non-empty list")
    for i, anchor in enumerate(raw):
        if not isinstance(anchor, dict) or not anchor.get("task"):
            raise ConfigError(f"{path}: anchor {i} needs a 'task'")
        if not anchor.get("sufficient") and not anchor.get("insufficient"):
            raise ConfigError(f"{path}: anchor {i} names no sufficient or insufficient tier")
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


async def calibrate(
    config: Config, anchors: list[dict[str, Any]], router: CapabilityRouter, margin: float = 0.03
) -> tuple[float, list[AnchorResult], list[str]]:
    from .capabilities import build_packet

    caps = config.router.capabilities
    assert caps is not None
    results: list[AnchorResult] = []
    for anchor in anchors:
        request = ChatCompletionRequest.model_validate({
            "model": "auto",
            "messages": [{"role": "user", "content": anchor["task"]}],
            **({"packet": anchor["packet"]} if anchor.get("packet") else {}),
        })
        packet = build_packet(request, max_chars=caps.max_packet_chars)
        needs = await router._ask(packet, caps.requirements)
        result = AnchorResult(anchor["task"], _tiers(anchor.get("sufficient")), _tiers(anchor.get("insufficient")), needs, {})
        for tier in result.sufficient + result.insufficient:
            if tier not in caps.cards:
                raise ConfigError(f"anchor names {tier!r}, which is not a carded tier (is it discovered?)")
            op = "<=" if tier in result.sufficient else ">="
            target = min(0.999, caps.target + margin) if op == "<=" else caps.target
            result.bounds[tier] = (op, _scale_bound(router, needs, tier, target))
        results.append(result)

    uppers = [s for r in results for op, s in r.bounds.values() if op == "<="]
    scale = min([1.0, *uppers])
    conflicts = [
        f"{tier} should fall short on {r.task[:60]!r}, but reaches the target at scale {scale:.3f}"
        for r in results
        for tier, (op, s) in r.bounds.items()
        if op == ">=" and scale <= s
    ]
    return scale, results, conflicts


def with_scale(config: Config, scale: float) -> Config:
    caps = config.router.capabilities
    assert caps is not None
    return replace(config, router=replace(config.router, capabilities=replace(caps, miss_scale=scale)))


def write_scale(path: str | Path, scale: float) -> None:
    """Set `miss_scale` in the config file, keeping every comment around it."""
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    line = f"miss_scale: {scale:.3f}"
    if re.search(r"^(\s*)miss_scale:.*$", text, flags=re.M):
        text = re.sub(r"^(\s*)miss_scale:.*$", lambda m: m.group(1) + line, text, count=1, flags=re.M)
    else:
        match = re.search(r"^(\s*)miss:\s*\[.*\]\s*$", text, flags=re.M)
        if not match:
            raise ConfigError(f"{path}: no `miss:` line under router.capabilities to put miss_scale beside")
        indent = match.group(1)
        insert = f"\n{indent}# Set by `llm-router calibrate`: every miss entry is multiplied by it.\n{indent}{line}"
        text = text[: match.end()] + insert + text[match.end():]
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
    if re.search(r"^(\s*)family_scales:.*$", text, flags=re.M):
        text = re.sub(r"^(\s*)family_scales:.*$", lambda m: m.group(1) + line, text, count=1, flags=re.M)
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


__all__ = [
    "AnchorResult", "FamilyFit", "Outcome", "calibrate", "fit_family_scales", "load_anchors", "log_outcomes",
    "with_scale", "write_family_scales", "write_scale",
]
