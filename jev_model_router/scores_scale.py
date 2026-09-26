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

Benchmarks are placed against each other only through the models they share,
so a group of benchmarks that shares no linking model with the rest could only
be placed by the priors. The scale is the largest connected group (most
benchmarks, then most linking models, then the first name); the others are
`detached` and read nothing from it.
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
    detached: tuple[str, ...] = ()  # benchmarks with linking points outside the main group, sorted

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


def groups(observations: list[Obs]) -> list[tuple[frozenset[str], int]]:
    """(benchmarks, linking nodes) per connected group of the benchmark <-> linking node graph, main one first."""
    links = linking(observations)
    parent: dict[object, object] = {}

    def find(x: object) -> object:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for o in observations:
        if o.node in links:
            parent[find(("bench", o.bench))] = find(("node", o.node))
    members: dict[object, tuple[set[str], set[Node]]] = {}
    for o in observations:
        if o.node in links:
            benches, nodes = members.setdefault(find(("bench", o.bench)), (set(), set()))
            benches.add(o.bench)
            nodes.add(o.node)
    out = [(frozenset(b), len(n)) for b, n in members.values()]
    return sorted(out, key=lambda g: (-len(g[0]), -g[1], sorted(g[0])))


def fit_scale(observations: list[Obs]) -> Scale:
    links = linking(observations)
    found = groups(observations)
    main = found[0][0] if found else frozenset()
    detached = tuple(sorted(b for g, _ in found[1:] for b in g))
    obs = sorted((o for o in observations if o.node in links and o.bench in main),
                 key=lambda o: (o.bench, repr(o.node)))
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
    return Scale(alpha, beta, theta, rounds, detached)


__all__ = ["Obs", "Scale", "fit_scale", "groups", "linking"]
