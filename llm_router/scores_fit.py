"""`benchmarks fit`: what each benchmark is worth, and the level mapping, fitted from judged outcomes.

Every judged decision is a trial of P(success | needs, tier). A tier's levels
come from its benchmark readings through the weights w_{b,r} = scale_b x w0_{b,r}
and the line level = a + k x ability, blended with the profile. Jev's reading
(or a manual one) fixes each benchmark's *shape*, the requirements it measures
and in what proportion; the outcomes decide its *worth*, one scale per
benchmark. The line is fitted as offsets from the profile line, (delta_a,
k_ratio), so a change in the benchmark set, which moves the shared scale, moves
the fitted line with it.

MAP: the router's own `success` is the likelihood, with Gaussian priors at
scale 1, delta_a 0 and k_ratio 1, so a handful of outcomes leaves everything
where it was without a sample threshold. Every trial rebuilds the cards the way
serving will once the fit is written: the profile line is recomputed from the
trial weights over every derived tier, and the offsets apply to that line.

Pure Python: projected coordinate ascent, golden-section per parameter over its
box, deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .calibration import Outcome
from .capabilities import CapabilityRouter
from .catalog import Offered
from .config import Config
from .discovery import profile_for
from .scores import Scores, base_weights, content_hash, fitted_scales, write_sidecar
from .scores_derive import blend, combine, effort_prior, line_pairs, profile_line, readings_for, served_keys

SIGMA_SCALE, SIGMA_DA, SIGMA_KR = 0.5, 0.5, 0.5
BOX_SCALE, BOX_DA, BOX_KR = (0.0, 3.0), (-2.0, 2.0), (0.1, 4.0)
MAX_SWEEPS = 50
TOLERANCE = 1e-6
_CLAMP = 1e-4


@dataclass
class FitResult:
    scales: dict[str, float]  # every fitted benchmark
    delta_a: float
    k_ratio: float
    a_line: float
    k_line: float
    outcomes: int
    tiers: list[str]
    loglik_prior: float
    loglik: float
    moved: list[tuple[str, float]] = field(default_factory=list)  # (benchmark, fitted scale), moved by > 0.1
    unlinked: list[str] = field(default_factory=list)
    fitted_at: str = ""

    @property
    def a(self) -> float:
        return self.a_line + self.delta_a

    @property
    def k(self) -> float:
        return self.k_line * self.k_ratio


async def _never(packet: str, questions: dict[str, str]) -> dict[str, float]:
    raise RuntimeError("benchmarks fit asks nobody")


def _golden(f, lo: float, hi: float, iterations: int = 40) -> float:
    g = (math.sqrt(5) - 1) / 2
    for _ in range(iterations):
        x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
        if f(x1) >= f(x2):
            hi = x2
        else:
            lo = x1
    return (lo + hi) / 2


def fit(
    config: Config,
    scores: Scores,
    outcomes: list[Outcome],
    *,
    model_ids: dict[str, tuple[str, ...]] | None = None,
    fitted_at: str | None = None,
) -> FitResult:
    caps = config.router.capabilities
    assert caps is not None
    router = CapabilityRouter(config, ask=_never)
    w0 = {b: row for b, (_, row) in base_weights(scores, caps).items()}
    # Every derived tier (the cards `expand` built, the only ones with a family): its keys, effort and profile.
    # The line is the profile line through all of them, as serving computes it.
    derived = {n: t for n, t in config.tiers.items() if n in caps.cards and caps.cards[n].family is not None}
    ids = {n: (model_ids or {}).get(t.model, (t.model,)) for n, t in derived.items()}
    keys, _ = served_keys(scores, {n: i for n, i in ids.items()})
    profiles = {n: profile_for(caps, Offered(i[0], aliases=tuple(i[1:])))[0] for n, i in ids.items()}
    priors = {n: effort_prior(caps, profiles[n], derived[n].effort)[0] for n in derived}
    # A tier's readings do not depend on the weights: read once, reweighed per trial.
    reads = {n: readings_for(scores, keys[n], derived[n].effort, w0) for n in derived}
    saved = scores.settings
    served_scales = fitted_scales(scores, caps)
    subjects = sorted({o.tier for o in outcomes if o.tier in derived})
    trials: dict[str, list[tuple[dict[str, float], int, int]]] = {}
    for t in subjects:
        counts: dict[tuple[tuple[str, float], ...], list[int]] = {}
        for o in outcomes:
            if o.tier == t:
                row = counts.setdefault(tuple(sorted(o.needs.items())), [0, 0])
                row[0 if o.passed else 1] += 1
        trials[t] = [(dict(key), passes, fails) for key, (passes, fails) in counts.items()]

    # A benchmark that weighs nothing, or covers no tier with outcomes, has no worth to fit.
    touched = {b: [t for t in subjects if b in reads[t]] for b, row in w0.items() if any(w > 0 for w in row.values())}
    fitted = sorted(b for b, ts in touched.items() if ts)
    unlinked = sorted(b for b in w0 if scores.rho.get(b, 0.0) <= 0)
    tiers_with = {b: [n for n in derived if b in reads[n]] for b in w0}

    # The state serving uses now: the stored scales (1 where none) and offsets.
    scale = {b: served_scales.get(b, 1.0) for b in fitted}
    offsets = {"delta_a": scores.fit.delta_a, "k_ratio": scores.fit.k_ratio}
    weights = {b: {r: w * served_scales.get(b, 1.0) for r, w in row.items()} for b, row in w0.items()}
    evs = {n: combine(caps.requirements, reads[n], weights) for n in derived}
    line = [0.0, 0.0]

    def relink() -> None:
        line[:] = profile_line([p for n in derived for p in line_pairs(priors[n], evs[n])])

    def reweigh(b: str) -> None:
        weights[b] = {r: w * scale[b] for r, w in w0[b].items()}
        for n in tiers_with[b]:
            evs[n] = combine(caps.requirements, reads[n], weights)
        relink()

    def total() -> float:
        a = saved.a if saved.a is not None else line[0] + offsets["delta_a"]
        k = saved.k if saved.k is not None else line[1] * offsets["k_ratio"]
        out = 0.0
        for t in subjects:
            levels = blend(evs[t], priors[t], caps.requirements, a=a, k=k, c0=scores.c0)
            for needs, passes, fails in trials[t]:
                p = min(1 - _CLAMP, max(_CLAMP, router.success(needs, t, levels=levels)))
                out += passes * math.log(p) + fails * math.log(1 - p)
        return out

    def log_prior() -> float:
        lp = sum(-((s - 1.0) ** 2) / (2 * SIGMA_SCALE**2) for s in scale.values())
        lp -= offsets["delta_a"] ** 2 / (2 * SIGMA_DA**2)
        return lp - (offsets["k_ratio"] - 1.0) ** 2 / (2 * SIGMA_KR**2)

    relink()
    loglik_prior = total()
    params: list[tuple[str, str, tuple[float, float], float, float]] = [
        ("scale", b, BOX_SCALE, 1.0, SIGMA_SCALE) for b in fitted
    ]
    if saved.a is None:
        params.append(("offset", "delta_a", BOX_DA, 0.0, SIGMA_DA))
    if saved.k is None:
        params.append(("offset", "k_ratio", BOX_KR, 1.0, SIGMA_KR))

    def put(kind: str, name: str, value: float) -> None:
        if kind == "scale":
            scale[name] = value
            reweigh(name)
        else:
            offsets[name] = value

    if subjects:
        # Start inside the boxes; the search moves from where serving is.
        for kind, name, (lo, hi), _, _ in params:
            current = (scale if kind == "scale" else offsets)[name]
            put(kind, name, min(hi, max(lo, current)))
        best = total() + log_prior()
        for _ in range(MAX_SWEEPS):
            for kind, name, (lo, hi), mean, sigma in params:
                # A scale moves the line, and so every tier's levels: log L over all of them.
                def objective(value: float) -> float:
                    put(kind, name, value)
                    return total() - (value - mean) ** 2 / (2 * sigma**2)

                current = (scale if kind == "scale" else offsets)[name]
                candidate = _golden(objective, lo, hi)
                put(kind, name, candidate if objective(candidate) > objective(current) else current)
            score = total() + log_prior()
            if score - best < TOLERANCE:
                break
            best = score
    return FitResult(
        scales=scale,
        delta_a=offsets["delta_a"],
        k_ratio=offsets["k_ratio"],
        a_line=line[0],
        k_line=line[1],
        outcomes=sum(p + f for rows in trials.values() for _, p, f in rows),
        tiers=subjects,
        loglik_prior=loglik_prior,
        loglik=total(),
        moved=[(b, s) for b, s in scale.items() if abs(s - 1.0) > 0.1],
        unlinked=unlinked,
        fitted_at=fitted_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def write_fit(path: str | Path, result: FitResult, scores: Scores, config: Config) -> None:
    """Store the fit: each scale with its benchmark's hash, beside the earlier scales today's outcomes did not touch."""
    caps = config.router.capabilities
    assert caps is not None
    previous = {b: {"scale": s, "hash": h} for b, (s, h) in scores.fit.scales.items()}
    fresh = {b: {"scale": s, "hash": content_hash(scores.benchmarks[b], caps.requirements)}
             for b, s in result.scales.items()}
    write_sidecar(path, fit={
        "scales": {**previous, **fresh},
        "delta_a": result.delta_a,
        "k_ratio": result.k_ratio,
        "outcomes": result.outcomes,
        "fitted_at": result.fitted_at,
        "loglik": result.loglik,
    })


__all__ = ["FitResult", "fit", "write_fit"]
