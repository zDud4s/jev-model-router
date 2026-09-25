"""Benchmark evidence: what measured results say a model can do, per requirement and effort.

A profile's levels are a person's guess at effort `high`, moved for every other
effort by one global rule. Measured results say more, and say it per effort and
per kind of work: on CursorBench 4.0 (2026-09) one vendor's model gained 2.4
points from high to xhigh and 0.2 from xhigh to max, another's 2.0 and then 4.0.

Evidence comes from sources that measure every vendor's models the same way,
imported by `benchmarks import` (see `scores_import.py`), plus curated points in
the benchmarks file: a benchmark owner's leaderboard copied by hand, or a vendor
launch page, which is fast but self-reported and so counts for less. Every
point is read onto one ability scale fitted across benchmarks
(`scores_scale.py`), and a (model, effort) gets an ability per requirement,
weighted by what each benchmark measures (Jev's reading, or a manual one) and by
how much the point says. `scores_derive.py` turns that into card levels.

Nothing here names a model, a vendor, a benchmark or a source: all of them are
rows in the benchmarks file and the imported file.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import EFFORTS, BenchmarksConfig, CapabilitiesConfig, Config, ConfigError
from .scores_scale import Obs, Scale, fit_scale, linking

# Normalised scores are clamped to [0.5%, 99.5%] before the logit, so chance or
# full marks is a large ability rather than an infinite one.
_EPS = 0.005
SOURCE_FORMATS = ("csv_zip", "json")
ORIGINS = ("independent", "vendor")
# A benchmark is fully used once this many models link it to the others.
FULL_LINKS = 3


@dataclass(frozen=True)
class Source:
    name: str
    format: str
    url: str
    model: str  # the column or dotted path holding the model string
    origin: str
    cite: str = ""
    api_key_env: str | None = None
    api_key_header: str = "x-api-key"
    items: str | None = None  # json: dotted path to the list of rows
    paging: tuple[str, str] | None = None  # json: (query parameter, dotted path of "more pages")
    metadata: dict[str, str] | None = None  # csv_zip: file, join, score, scale, baseline, ceiling


@dataclass(frozen=True)
class DataMap:
    source: str
    table: str | None = None
    score: str | None = None
    effort: str | None = None
    cost: str | None = None
    date: str | None = None
    upstream: str | None = None
    where: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Benchmark:
    key: str
    description: str
    example_tasks: tuple[str, ...] = ()
    requirements: dict[str, float] | None = None  # a manual override of the weights
    data: DataMap | None = None
    scale: float | None = None
    baseline: float | None = None
    ceiling: float | None = None


@dataclass(frozen=True)
class EffortRules:
    """How sources write efforts. A label that resolves is an effort; anything else is part of a name."""

    labels: dict[str, str] = field(default_factory=dict)  # lowercased label -> effort
    patterns: tuple[str, ...] = ()

    def resolve(self, label: Any) -> str | None:
        if label is None:
            return None
        text = " ".join(str(label).split()).lower()
        if not text:
            return None
        if text in self.labels:
            return self.labels[text]
        if text in EFFORTS or text == "unknown":
            return text
        if any(re.fullmatch(p, text) for p in self.patterns):
            return text
        return None

    def matchable(self, effort: str | None) -> bool:
        """An effort a tier can have: none, one of EFFORTS, or a label mapped outside them."""
        if effort is None or effort in EFFORTS:
            return True
        return effort != "unknown" and effort in set(self.labels.values())


def split_model(text: str, efforts: EffortRules) -> tuple[str, str | None]:
    """(the model string without its effort text, the effort label) -- only a label that resolves is removed."""
    text = text.strip()
    if "_" in text:
        head, tail = text.rsplit("_", 1)
        label = efforts.resolve(tail)
        if label is not None:
            return head, label
    match = re.search(r"\s*\(([^()]*)\)\s*$", text)
    if match:
        rest, inner = text[: match.start()], match.group(1)
        label = efforts.resolve(inner)
        if label is not None:
            return rest, label
        words = re.findall(r"[^\s,;]+", inner)
        for n in range(len(words), 0, -1):
            for i in range(len(words) - n + 1):
                label = efforts.resolve(" ".join(words[i : i + n]))
                if label is not None:
                    return rest, label
    return text, None


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@dataclass(frozen=True)
class KeyRules:
    """The one rule that matches a source's model string to a served model id."""

    date_suffixes: tuple[str, ...] = ()
    aliases: dict[str, str] = field(default_factory=dict)  # lowercased string or key -> the key it means
    keep_dates: frozenset[str] = frozenset()  # undated keys that two dated snapshots would share

    def _parts(self, text: str, efforts: EffortRules) -> tuple[str, str]:
        rest, _ = split_model(text.strip().split("/")[-1], efforts)
        undated = rest
        for pattern in self.date_suffixes:
            stripped = re.sub(pattern, "", rest)
            if stripped != rest:
                undated = stripped
                break
        return _norm(rest), _norm(undated)

    def key(self, text: str, efforts: EffortRules) -> str:
        alias = self.aliases.get(text.strip().lower())
        if alias is not None:
            return _norm(alias)
        dated, undated = self._parts(text, efforts)
        key = dated if undated in self.keep_dates else undated
        alias = self.aliases.get(key)
        return _norm(alias) if alias is not None else key


def date_collisions(strings: list[str], rules: KeyRules, efforts: EffortRules) -> frozenset[str]:
    """Undated keys shared by two different dated strings: those snapshots stay apart."""
    dated: dict[str, set[str]] = {}
    for text in strings:
        if text.strip().lower() in rules.aliases:
            continue
        full, undated = rules._parts(text, efforts)
        if full != undated:
            dated.setdefault(undated, set()).add(full)
    return frozenset(k for k, fulls in dated.items() if len(fulls) >= 2)


@dataclass(frozen=True)
class Point:
    benchmark: str
    model: str  # as the source writes it
    effort: str | None
    score: float  # percent
    cost_usd: float | None = None
    date: str = ""
    source: str = ""
    origin: str = "vendor"
    approx: bool = False
    upstream: str = ""
    imported: bool = False
    order: int = 0


@dataclass(frozen=True)
class Reading:
    """One point on the shared scale: its ability `t` and the evidence `u` behind it."""

    t: float
    u: float
    point: Point


@dataclass(frozen=True)
class JevReading:
    hash: str
    needs: dict[str, float]
    read_at: str = ""
    jev_model: str = ""


@dataclass(frozen=True)
class Fit:
    scales: dict[str, tuple[float, str]] = field(default_factory=dict)  # benchmark -> (scale, its hash)
    delta_a: float = 0.0
    k_ratio: float = 1.0
    outcomes: int = 0
    fitted_at: str | None = None
    loglik: float | None = None


@dataclass(frozen=True)
class Scores:
    sources: dict[str, Source]
    efforts: EffortRules
    keys: KeyRules
    benchmarks: dict[str, Benchmark]
    points: tuple[Point, ...]  # pooled, deduplicated, superseded ones removed
    superseded: tuple[Point, ...]
    readings: dict[str, dict[str, dict[str | None, Reading]]]  # benchmark -> key -> effort -> reading
    slots: dict[tuple[str, str, str | None], Point]  # (benchmark, key, effort) -> its point, linked or not
    rho: dict[str, float]  # benchmark -> how far it is linked to the others, 0..1
    links: dict[str, int]  # benchmark -> linking models
    scale: Scale
    jev: dict[str, JevReading]
    fit: Fit
    settings: BenchmarksConfig
    imported_at: str | None = None
    imported_status: dict[str, str] = field(default_factory=dict)
    sidecar_path: str = ""
    imported_path: str = ""
    errors: tuple[str, ...] = ()  # an unreadable sidecar or imported file: reported, never fatal

    @property
    def c0(self) -> float:
        return self.settings.profile_weight

    def key(self, text: str) -> str:
        return self.keys.key(text, self.efforts)

    def point_at(self, bench: str, key: str, effort: str | None) -> Point | None:
        return self.slots.get((bench, key, effort))


def sidecar_for(path: str | Path) -> Path:
    """`benchmarks.yaml` -> `benchmarks.derived.json`, beside it."""
    return Path(path).with_suffix(".derived.json")


def imported_for(path: str | Path) -> Path:
    """`benchmarks.yaml` -> `benchmarks.imported.json`, beside it."""
    return Path(path).with_suffix(".imported.json")


def _read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
        return data
    except (OSError, ValueError) as exc:
        errors.append(f"{path}: {exc}")
        return {}


def read_curated(config: Config) -> tuple[Path, Any]:
    caps = config.router.capabilities
    assert caps is not None and caps.benchmarks is not None
    path = Path(caps.benchmarks.path)
    if not path.is_file():
        raise ConfigError(f"benchmarks file not found: {path}")
    try:
        return path, yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: not valid YAML: {exc}") from None


def load_scores(config: Config) -> Scores | None:
    caps = config.router.capabilities
    if caps is None or caps.benchmarks is None:
        return None
    path, raw = read_curated(config)
    errors: list[str] = []
    imported = _read_json(imported_for(path), errors)
    sidecar = _read_json(sidecar_for(path), errors)
    try:
        return parse_scores(
            raw, caps, imported=imported, sidecar=sidecar,
            sidecar_path=str(sidecar_for(path)), imported_path=str(imported_for(path)), errors=errors,
        )
    except (TypeError, ValueError, AttributeError) as exc:  # a non-numeric value, say: still the file's fault
        raise ConfigError(f"{path}: {exc}") from None


def _mapping(raw: Any, where: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping")
    return raw


def _only(raw: dict[str, Any], allowed: set[str], where: str) -> None:
    extra = set(raw) - allowed
    if extra:
        raise ConfigError(f"{where}: unknown fields {sorted(extra)}")


def _regexes(raw: Any, where: str) -> tuple[str, ...]:
    out = tuple(str(p) for p in (raw or []))
    for p in out:
        try:
            re.compile(p)
        except re.error as exc:
            raise ConfigError(f"{where}: {p!r} is not a valid regular expression: {exc}") from None
    return out


def _source(name: str, raw: Any) -> Source:
    where = f"sources[{name!r}]"
    raw = _mapping(raw, where)
    _only(raw, {"format", "url", "model", "origin", "cite", "api_key_env", "api_key_header", "items", "paging",
                "metadata"}, where)
    if raw.get("format") not in SOURCE_FORMATS:
        raise ConfigError(f"{where}.format must be one of {list(SOURCE_FORMATS)}, got {raw.get('format')!r}")
    if raw.get("origin") not in ORIGINS:
        raise ConfigError(f"{where}.origin must be one of {list(ORIGINS)}, got {raw.get('origin')!r}")
    for required in ("url", "model"):
        if not raw.get(required):
            raise ConfigError(f"{where}.{required} is required")
    paging = None
    if raw.get("paging") is not None:
        pg = _mapping(raw["paging"], f"{where}.paging")
        if not pg.get("param") or not pg.get("more"):
            raise ConfigError(f"{where}.paging needs both 'param' and 'more'")
        paging = (str(pg["param"]), str(pg["more"]))
    metadata = None
    if raw.get("metadata") is not None:
        md = _mapping(raw["metadata"], f"{where}.metadata")
        _only(md, {"file", "join", "score", "scale", "baseline", "ceiling"}, f"{where}.metadata")
        if not md.get("file") or not md.get("join"):
            raise ConfigError(f"{where}.metadata needs 'file' and 'join'")
        metadata = {str(k): str(v) for k, v in md.items()}
    return Source(
        name=name, format=raw["format"], url=str(raw["url"]), model=str(raw["model"]), origin=raw["origin"],
        cite=str(raw.get("cite") or ""), api_key_env=raw.get("api_key_env"),
        api_key_header=str(raw.get("api_key_header") or "x-api-key"), items=raw.get("items"), paging=paging,
        metadata=metadata,
    )


def _weights(raw: Any, requirements: dict[str, str], where: str) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must map requirements to weights")
    stray = set(raw) - set(requirements)
    if stray:
        raise ConfigError(f"{where}: unknown requirement(s) {sorted(stray)}")
    out = {str(k): float(v) for k, v in raw.items()}
    if any(not 0.0 <= v <= 1.0 for v in out.values()):
        raise ConfigError(f"{where}: every weight must be in [0, 1]")
    return out


def _optional(value: Any) -> float | None:
    return None if value is None else float(value)


def _benchmark(key: str, raw: Any, caps: CapabilitiesConfig, sources: dict[str, Source]) -> Benchmark:
    where = f"benchmarks[{key!r}]"
    raw = _mapping(raw, where)
    _only(raw, {"description", "example_tasks", "requirements", "data", "scale", "baseline", "ceiling"}, where)
    description = str(raw.get("description") or "").strip()
    tasks = raw.get("example_tasks") or []
    if not isinstance(tasks, list):
        raise ConfigError(f"{where}.example_tasks must be a list")
    manual = raw.get("requirements")
    if manual is not None:
        manual = _weights(manual, caps.requirements, f"{where}.requirements")
    if not description and not tasks and manual is None:
        raise ConfigError(f"{where}: needs a description, example_tasks or requirements")
    data = None
    if raw.get("data") is not None:
        d = _mapping(raw["data"], f"{where}.data")
        _only(d, {"source", "table", "score", "effort", "cost", "date", "upstream", "where"}, f"{where}.data")
        if d.get("source") not in sources:
            raise ConfigError(f"{where}.data: unknown source {d.get('source')!r}")
        if sources[d["source"]].format == "csv_zip" and not d.get("table"):
            raise ConfigError(f"{where}.data: a csv_zip source needs a 'table'")
        data = DataMap(
            source=d["source"], table=d.get("table"), score=d.get("score"), effort=d.get("effort"),
            cost=d.get("cost"), date=d.get("date"), upstream=d.get("upstream"),
            where={str(k): str(v) for k, v in _mapping(d.get("where"), f"{where}.data.where").items()},
        )
    baseline, ceiling = _optional(raw.get("baseline")), _optional(raw.get("ceiling"))
    if (baseline if baseline is not None else 0.0) >= (ceiling if ceiling is not None else 1.0):
        raise ConfigError(f"{where}: baseline must be below ceiling")
    return Benchmark(key, description, tuple(str(t) for t in tasks), manual, data,
                     _optional(raw.get("scale")), baseline, ceiling)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _point(entry: Any, benchmarks: dict[str, Benchmark], efforts: EffortRules, where: str, order: int) -> Point:
    entry = _mapping(entry, where)
    _only(entry, {"benchmark", "model", "effort", "score", "cost_usd", "date", "source", "origin", "approx"}, where)
    bench = entry.get("benchmark")
    if bench not in benchmarks:
        raise ConfigError(f"{where}: unknown benchmark {bench!r}")
    if not entry.get("model"):
        raise ConfigError(f"{where}: 'model' is required")
    effort = entry.get("effort")
    if effort is not None and not isinstance(effort, str):
        raise ConfigError(f"{where}: effort must be null or a string, got {effort!r}")
    score = entry.get("score")
    if not _number(score) or not 0 <= score <= 100:
        raise ConfigError(f"{where}: score must be a percentage in [0, 100], got {score!r}")
    cost = entry.get("cost_usd")
    if cost is not None and (not _number(cost) or cost < 0):
        raise ConfigError(f"{where}: cost_usd cannot be negative, got {cost!r}")
    origin = entry.get("origin", "vendor")
    if origin not in ORIGINS:
        raise ConfigError(f"{where}: origin must be one of {list(ORIGINS)}, got {origin!r}")
    date = entry.get("date")
    return Point(
        # Written as a source would write it (`High`, `extra high`): one reading for both files.
        benchmark=str(bench), model=str(entry["model"]),
        effort=None if effort is None else efforts.resolve(effort) or effort, score=float(score),
        cost_usd=None if cost is None else float(cost),
        date=date.isoformat() if hasattr(date, "isoformat") else str(date or ""),
        source=str(entry.get("source") or ""), origin=origin, approx=bool(entry.get("approx", False)), order=order,
    )


def _imported_points(
    imported: dict[str, Any], benchmarks: dict[str, Benchmark], sources: dict[str, Source]
) -> tuple[list[Point], dict[str, dict[str, float]]]:
    """Points from the imported file, for benchmarks the curated file still maps; and each one's metadata."""
    points: list[Point] = []
    meta: dict[str, dict[str, float]] = {}
    for bench, block in (imported.get("benchmarks") or {}).items():
        entry = benchmarks.get(bench)
        if entry is None or entry.data is None or entry.data.source != block.get("source"):
            continue  # the curated file no longer asks for it
        source = sources[entry.data.source]
        meta[bench] = {k: float(block[k]) for k in ("baseline", "ceiling") if block.get(k) is not None}
        for i, row in enumerate(block.get("points") or []):
            points.append(Point(
                benchmark=bench, model=str(row["model"]), effort=row.get("effort"), score=float(row["score"]),
                cost_usd=_optional(row.get("cost_usd")), date=str(row.get("date") or ""),
                source=source.cite or source.url, origin=source.origin, upstream=str(row.get("upstream") or ""),
                imported=True, order=i,
            ))
    return points, meta


def _rank(p: Point) -> tuple:
    # independent over vendor, exact over read off a chart, newer, imported, later in the file
    return (p.origin == "independent", not p.approx, p.date, p.imported, p.order)


def pool(
    points: list[Point], efforts: EffortRules, key_of: dict[str, str]
) -> tuple[list[Point], list[Point]]:
    """(kept, superseded): one point per (benchmark, model key, effort), and no vendor curve beside an independent one."""
    best: dict[tuple[str, str, str | None], Point] = {}
    for p in points:
        slot = (p.benchmark, key_of[p.model], p.effort)
        if slot not in best or _rank(p) > _rank(best[slot]):
            best[slot] = p
    measured = {
        (b, k) for (b, k, e), p in best.items() if p.origin == "independent" and efforts.matchable(e)
    }
    kept, superseded = [], []
    for (b, k, _), p in best.items():
        (superseded if p.origin == "vendor" and (b, k) in measured else kept).append(p)
    order = lambda p: (p.benchmark, key_of[p.model], str(p.effort))
    return sorted(kept, key=order), sorted(superseded, key=order)


def _logit(s: float) -> float:
    s = min(1 - _EPS, max(_EPS, s))
    return math.log(s / (1 - s))


def normalised(score: float, baseline: float, ceiling: float) -> float:
    """The score as a fraction of the way from chance to full marks, clamped for the logit."""
    return min(1 - _EPS, max(_EPS, (score / 100 - baseline) / (ceiling - baseline)))


def _bounds(
    bench: Benchmark, meta: dict[str, float], where: str, errors: list[str]
) -> tuple[float, float] | None:
    """(baseline, ceiling): the curated value, else the imported one, else 0 and 1.

    Imported bounds that are not finite or not in order (equal ones would divide
    by zero, reversed ones would turn every score upside down) are dropped with
    an error, and the curated values or the defaults are used instead. None only
    if even those are out of order, and then the benchmark's points are skipped.
    """
    def resolve(meta: dict[str, float]) -> tuple[float, float]:
        base = bench.baseline if bench.baseline is not None else meta.get("baseline", 0.0)
        ceil = bench.ceiling if bench.ceiling is not None else meta.get("ceiling", 1.0)
        return base, ceil

    def valid(base: float, ceil: float) -> bool:
        return math.isfinite(base) and math.isfinite(ceil) and base < ceil

    base, ceil = resolve(meta)
    if valid(base, ceil):
        return base, ceil
    fallback = resolve({})
    errors.append(f"{where}: baseline {base} must be below ceiling {ceil}; "
                  + ("using the curated values or 0 and 1" if valid(*fallback) else "its points are skipped"))
    return fallback if valid(*fallback) else None


def parse_scores(
    raw: Any,
    caps: CapabilitiesConfig,
    *,
    imported: dict[str, Any] | None = None,
    sidecar: dict[str, Any] | None = None,
    sidecar_path: str = "",
    imported_path: str = "",
    errors: list[str] | None = None,
) -> Scores:
    settings = caps.benchmarks or BenchmarksConfig(path="")
    errors = list(errors or [])
    raw = _mapping(raw, "benchmarks file")
    _only(raw, {"sources", "efforts", "model_keys", "aliases", "benchmarks", "points"}, "benchmarks file")
    sources = {str(n): _source(str(n), s) for n, s in _mapping(raw.get("sources"), "sources").items()}
    eff = _mapping(raw.get("efforts"), "efforts")
    _only(eff, {"labels", "patterns"}, "efforts")
    efforts = EffortRules(
        labels={str(k).lower(): str(v) for k, v in _mapping(eff.get("labels"), "efforts.labels").items()},
        patterns=_regexes(eff.get("patterns"), "efforts.patterns"),
    )
    mk = _mapping(raw.get("model_keys"), "model_keys")
    _only(mk, {"date_suffixes"}, "model_keys")
    aliases = {str(k).strip().lower(): str(v) for k, v in _mapping(raw.get("aliases"), "aliases").items()}
    rules = KeyRules(_regexes(mk.get("date_suffixes"), "model_keys.date_suffixes"), aliases)
    benchmarks = {
        str(k): _benchmark(str(k), v, caps, sources) for k, v in _mapping(raw.get("benchmarks"), "benchmarks").items()
    }
    raw_points = raw.get("points") or []
    if not isinstance(raw_points, list):
        raise ConfigError("benchmarks file: 'points' must be a list")
    curated = [_point(e, benchmarks, efforts, f"points[{i}]", i) for i, e in enumerate(raw_points)]
    try:
        from_import, meta = _imported_points(imported or {}, benchmarks, sources)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        from_import, meta = [], {}
        errors.append(f"{imported_path}: {type(exc).__name__}: {exc}")
    bounds = {b: _bounds(entry, meta.get(b, {}), f"{imported_path}: benchmark {b!r}", errors)
              for b, entry in benchmarks.items()}
    everything = [p for p in curated + from_import if bounds[p.benchmark] is not None]
    rules = KeyRules(rules.date_suffixes, rules.aliases,
                     date_collisions(sorted({p.model for p in everything}), rules, efforts))
    key_of = {p.model: rules.key(p.model, efforts) for p in everything}
    kept, superseded = pool(everything, efforts, key_of)

    def node(key: str, effort: str | None, bench: str) -> tuple:
        return (key, effort) if efforts.matchable(effort) else (key, effort, bench)

    obs: list[Obs] = []
    for p in kept:
        s = normalised(p.score, *bounds[p.benchmark])
        q = 1.0 if p.origin == "independent" else settings.vendor_weight
        obs.append(Obs(p.benchmark, node(key_of[p.model], p.effort, p.benchmark), _logit(s), q * 4 * s * (1 - s)))
    scale = fit_scale(obs)
    links = linking(obs)
    readings: dict[str, dict[str, dict[str | None, Reading]]] = {}
    for p, o in zip(kept, obs):
        t = scale.ability(p.benchmark, o.y)
        if t is not None:
            readings.setdefault(p.benchmark, {}).setdefault(key_of[p.model], {})[p.effort] = Reading(t, o.u, p)
    linked: dict[str, set[str]] = {}
    for o in obs:
        if o.node in links:
            linked.setdefault(o.bench, set()).add(o.node[0])
    link_counts = {b: len(linked.get(b, ())) for b in benchmarks}
    try:
        jev, fit = _parse_sidecar(sidecar or {})
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        jev, fit = {}, Fit()
        errors.append(f"{sidecar_path}: {type(exc).__name__}: {exc}")
    return Scores(
        sources=sources, efforts=efforts, keys=rules, benchmarks=benchmarks, points=tuple(kept),
        superseded=tuple(superseded), readings=readings,
        slots={(p.benchmark, key_of[p.model], p.effort): p for p in kept},
        # A benchmark off the main group is not placed against the others: unlinked, whatever its own links.
        rho={b: 0.0 if b in scale.detached else min(1.0, n / FULL_LINKS) for b, n in link_counts.items()},
        links=link_counts, scale=scale,
        jev=jev, fit=fit, settings=settings, imported_at=(imported or {}).get("imported_at"),
        imported_status={str(k): str(v.get("status", "")) for k, v in ((imported or {}).get("sources") or {}).items()},
        sidecar_path=sidecar_path, imported_path=imported_path, errors=tuple(errors),
    )


def _parse_sidecar(sidecar: dict[str, Any]) -> tuple[dict[str, JevReading], Fit]:
    jev = {
        str(key): JevReading(
            hash=str(entry["hash"]),
            needs={str(r): float(p) for r, p in entry["needs"].items()},
            read_at=str(entry.get("read_at") or ""),
            jev_model=str(entry.get("jev_model") or ""),
        )
        for key, entry in (sidecar.get("jev") or {}).items()
    }
    raw_fit = sidecar.get("fit") or {}
    fit = Fit(
        scales={str(b): (float(v["scale"]), str(v["hash"])) for b, v in (raw_fit.get("scales") or {}).items()},
        delta_a=float(raw_fit.get("delta_a") or 0.0),
        k_ratio=float(raw_fit.get("k_ratio") or 1.0),
        outcomes=int(raw_fit.get("outcomes") or 0),
        fitted_at=raw_fit.get("fitted_at"),
        loglik=_optional(raw_fit.get("loglik")),
    )
    return jev, fit


def content_hash(benchmark: Benchmark, requirements: dict[str, str]) -> str:
    """What a benchmark's weights depend on: its words, a manual override, and the questions asked."""
    body = json.dumps(
        {
            "description": benchmark.description,
            "example_tasks": list(benchmark.example_tasks),
            "requirements": benchmark.requirements,
            "questions": requirements,
        },
        sort_keys=True,
    )
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def base_weights(scores: Scores, caps: CapabilitiesConfig) -> dict[str, tuple[str, dict[str, float]]]:
    """benchmark -> (manual | jev, requirement -> w0). A benchmark with neither is absent (unread)."""
    out: dict[str, tuple[str, dict[str, float]]] = {}
    for key, bench in scores.benchmarks.items():
        if bench.requirements is not None:
            out[key] = ("manual", dict(bench.requirements))
            continue
        reading = scores.jev.get(key)
        if reading is not None and reading.hash == content_hash(bench, caps.requirements):
            out[key] = ("jev", {
                r: max(0.0, p - caps.floor) / (1.0 - caps.floor)
                for r, p in reading.needs.items() if r in caps.requirements
            })
    return out


def fitted_scales(scores: Scores, caps: CapabilitiesConfig) -> dict[str, float]:
    """benchmark -> its fitted scale, for benchmarks whose words have not changed since the fit."""
    return {
        b: scale for b, (scale, digest) in scores.fit.scales.items()
        if b in scores.benchmarks and digest == content_hash(scores.benchmarks[b], caps.requirements)
    }


def benchmark_weights(
    scores: Scores, caps: CapabilitiesConfig, scales: dict[str, float] | None = None
) -> dict[str, dict[str, float]]:
    """benchmark -> requirement -> w0 x its scale (the fitted one unless `scales` stands in)."""
    use = fitted_scales(scores, caps) if scales is None else scales
    return {b: {r: w * use.get(b, 1.0) for r, w in row.items()} for b, (_, row) in base_weights(scores, caps).items()}


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_sidecar(path: str | Path, **blocks: Any) -> None:
    """Replace the named blocks (`jev`, `fit`) of the sidecar, keeping the others."""
    target = Path(path)
    data = _read_json(target, [])  # regenerable: an unreadable sidecar is replaced, not merged
    data.update(blocks)
    write_json(target, data)


__all__ = [
    "Benchmark", "DataMap", "EffortRules", "Fit", "JevReading", "KeyRules", "Point", "Reading", "Scores", "Source",
    "base_weights", "benchmark_weights", "content_hash", "date_collisions", "fitted_scales", "imported_for",
    "load_scores", "normalised", "parse_scores", "pool", "read_curated", "sidecar_for", "split_model",
    "write_json", "write_sidecar",
]
