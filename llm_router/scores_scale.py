"""One ability scale across benchmarks, fitted from the models they share.

Benchmarks differ in difficulty and in how widely they spread models, and each
is measured on a different crowd: one table has 15 frontier models, another 200,
most of them small. A median taken per benchmark would place the same model
differently depending on who else happened to be measured. So every benchmark
is read through one fitted scale:

    y_{i,b} = logit(normalised score) ~ alpha_b * theta_i - beta_b

for each node i (a model at an effort) and benchmark b. Only *linking* nodes,
those measured on at least two benchmarks, are fitted. A node seen once could
absorb its score in its own theta, so it carries nothing about b's difficulty;
fitted anyway, the prior on theta would let it pull alpha_b and beta_b toward
whoever else was measured there. On the 2026-09-25 Epoch data that moved one
benchmark's difficulty by 0.37.

Weighted ridge least squares, alternating closed forms, pure Python and
deterministic. The scale is then fixed so that the mean alpha_b is 1 and the
mean beta_b/alpha_b is 0: ability 0 means 50% above chance on a benchmark of
average difficulty, a reference the operator's choice of benchmarks sets rather
than the crowd each import brings.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

SIGMA_THETA, SIGMA_ALPHA, SIGMA_BETA = 2.0, 0.5, 3.0
MIN_ALPHA = 0.2
MAX_ROUNDS = 500
TOLERANCE = 1e-9

Node = tuple  # (model key, effort) for a matchable effort; (model key, label, benchmark) otherwise


@dataclass(frozen=True)
class Obs:
    bench: str
    node: Node
    y: float  # logit of the normalised score
    u: float  # its information: evidence weight x 4s(1 - s)


@dataclass(frozen=True)
class Scale:
    alpha: dict[str, float]  # only benchmarks with a linking point
    beta: dict[str, float]
    theta: dict[Node, float]  # only linking nodes
    rounds: int

    def ability(self, bench: str, y: float) -> float | None:
        """A score's ability on the shared scale, read through the benchmark's difficulty and spread."""
        if bench not in self.alpha:
            return None
        return (y + self.beta[bench]) / self.alpha[bench]


def linking(observations: list[Obs]) -> set[Node]:
    """Nodes measured on at least two benchmarks."""
    seen: dict[Node, set[str]] = {}
    for o in observations:
        seen.setdefault(o.node, set()).add(o.bench)
    return {node for node, benches in seen.items() if len(benches) >= 2}


def fit_scale(observations: list[Obs]) -> Scale:
    links = linking(observations)
    obs = sorted((o for o in observations if o.node in links), key=lambda o: (o.bench, repr(o.node)))
    by_node: dict[Node, list[Obs]] = {}
    by_bench: dict[str, list[Obs]] = {}
    for o in obs:
        by_node.setdefault(o.node, []).append(o)
        by_bench.setdefault(o.bench, []).append(o)
    alpha = {b: 1.0 for b in by_bench}
    beta = {b: 0.0 for b in by_bench}
    theta = {n: 0.0 for n in by_node}
    rounds = 0
    for rounds in range(1, MAX_ROUNDS + 1):
        moved = 0.0
        for node, rows in by_node.items():
            num = sum(o.u * alpha[o.bench] * (o.y + beta[o.bench]) for o in rows)
            den = sum(o.u * alpha[o.bench] ** 2 for o in rows) + 1 / SIGMA_THETA**2
            new = num / den
            moved, theta[node] = max(moved, abs(new - theta[node])), new
        for bench, rows in by_bench.items():
            # minimise sum u (a theta - b - y)^2 + (a - 1)^2 / sa^2 + b^2 / sb^2
            saa = sum(o.u * theta[o.node] ** 2 for o in rows) + 1 / SIGMA_ALPHA**2
            sab = -sum(o.u * theta[o.node] for o in rows)
            sbb = sum(o.u for o in rows) + 1 / SIGMA_BETA**2
            ra = sum(o.u * theta[o.node] * o.y for o in rows) + 1 / SIGMA_ALPHA**2
            rb = -sum(o.u * o.y for o in rows)
            det = saa * sbb - sab * sab
            a = (ra * sbb - sab * rb) / det
            b = (saa * rb - sab * ra) / det
            if a < MIN_ALPHA:
                a = MIN_ALPHA
                b = (rb - sab * a) / sbb
            moved = max(moved, abs(a - alpha[bench]), abs(b - beta[bench]))
            alpha[bench], beta[bench] = a, b
        if moved < TOLERANCE:
            break
    if alpha:
        mean_alpha = statistics.fmean(alpha.values())
        theta = {n: t * mean_alpha for n, t in theta.items()}
        alpha = {b: a / mean_alpha for b, a in alpha.items()}
        shift = statistics.fmean(beta[b] / alpha[b] for b in alpha)
        theta = {n: t - shift for n, t in theta.items()}
        beta = {b: beta[b] - shift * alpha[b] for b in alpha}
    return Scale(alpha, beta, theta, rounds)


__all__ = ["Obs", "Scale", "fit_scale", "linking"]
