"""`benchmarks import`: evidence from sources that measure every vendor's models the same way.

A source is data in the benchmarks file: a format, a URL, and where the model,
score, effort and cost sit in its rows. There is a reader per *format* (a zip of
CSV tables, a JSON document), never per source, model or benchmark, so a new
source with a known format is an entry, and a new model is new rows.

Per row: the score is scaled to a percentage, the effort is read from the
effort column when it resolves and from the model string's suffix otherwise,
and rows for one (benchmark, model key, effort) -- one model run under several
harnesses, say -- collapse to their median. The result replaces each imported
benchmark's points in the imported file. A failure keeps the previous points of
what failed, and nothing else is touched.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import statistics
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .scores import Benchmark, Scores, Source, split_model, write_json

# fetch(url, headers, deadline): the body, or an exception once `_clock()` passes `deadline`.
Fetch = Callable[[str, dict[str, str], float], bytes]
MAX_PAGES = 50
_CHUNK = 1 << 16
_clock = time.monotonic
_MISSING = object()


class ImportFailure(Exception):
    """A mapped column or field the data does not have, or data that cannot be read."""


def _origin(url: str) -> tuple[str, str]:
    parts = urllib.parse.urlsplit(url)
    return parts.scheme.lower(), parts.netloc.lower()


class _PrivateHeaders(urllib.request.HTTPRedirectHandler):
    """Follows a redirect, but sends the caller's own headers (the API key) only to the same scheme and host."""

    def __init__(self, private: set[str]) -> None:
        self.private = {name.lower() for name in private}

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and _origin(req.full_url) != _origin(newurl):
            for name in [n for n in new.headers if n.lower() in self.private]:
                del new.headers[name]
        return new


def _read(response: Any, deadline: float) -> bytes:
    """The body, read in chunks so a server that trickles is cut at the deadline, not per socket read."""
    chunks: list[bytes] = []
    while True:
        if _clock() > deadline:
            raise TimeoutError("timed out")
        chunk = response.read(_CHUNK)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def http_fetch(timeout: float) -> Fetch:
    """`timeout` bounds each socket operation; `deadline` bounds the whole fetch."""
    def fetch(url: str, headers: dict[str, str], deadline: float) -> bytes:
        left = deadline - _clock()
        if left <= 0:
            raise TimeoutError("timed out")
        request = urllib.request.Request(url, headers={"User-Agent": "llm-router", **headers})
        opener = urllib.request.build_opener(_PrivateHeaders(set(headers)))
        with opener.open(request, timeout=min(timeout, left)) as response:  # noqa: S310 - URLs come from the operator's file
            return _read(response, deadline)

    return fetch


@dataclass
class SourceReport:
    rows: int = 0
    skipped: int = 0
    points: int = 0
    benchmarks: list[str] = field(default_factory=list)


@dataclass
class ImportReport:
    sources: dict[str, SourceReport] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)  # a source or a benchmark -> what went wrong


def _dotted(row: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(row, dict) or part not in row:
            return _MISSING
        row = row[part]
    return row


def _number(value: Any) -> float | None:
    if value is None or value is _MISSING:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # NaN is no score


def _pages(source: Source, fetch: Fetch, headers: dict[str, str], deadline: float) -> list[Any]:
    """The rows of a JSON source, over every page."""
    rows: list[Any] = []
    for page in range(1, MAX_PAGES + 1):
        url = source.url
        if source.paging:
            url += ("&" if "?" in url else "?") + f"{source.paging[0]}={page}"
        if _clock() > deadline:
            raise TimeoutError(f"timed out after {page - 1} page(s)")
        document = json.loads(fetch(url, headers, deadline))
        items = _dotted(document, source.items) if source.items else document
        if not isinstance(items, list):
            raise ImportFailure(f"{source.items or 'the document'} is not a list")
        rows += items
        more = _dotted(document, source.paging[1]) if source.paging else _MISSING
        if more is _MISSING or not more:
            break
    return rows


def _table(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    try:
        with archive.open(name) as handle:
            return list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8-sig")))
    except KeyError:
        raise ImportFailure(f"table {name!r} is not in the archive") from None


def _metadata(archive: zipfile.ZipFile, source: Source) -> dict[str, dict[str, str]]:
    """table name -> its metadata row, through the source's declared join column."""
    if not source.metadata:
        return {}
    join = source.metadata["join"]
    return {row.get(join, ""): row for row in _table(archive, source.metadata["file"]) if row.get(join)}


def _field(meta_row: dict[str, str] | None, source: Source, name: str) -> float | None:
    column = (source.metadata or {}).get(name)
    return _number(meta_row.get(column)) if meta_row and column else None


def _date(value: Any, today: str) -> str:
    text = str(value or "")
    return text[:10] if re.match(r"\d{4}-\d{2}-\d{2}", text) else today


def _rows_for(bench: Benchmark, source: Source, payload: Any, today: str, scores: Scores
              ) -> tuple[list[dict[str, Any]], int, int, dict[str, float]]:
    """(points before collapsing, rows read, rows skipped, metadata) for one benchmark."""
    data = bench.data
    assert data is not None
    meta: dict[str, float] = {}
    if source.format == "csv_zip":
        archive, meta_rows = payload
        rows: list[Any] = _table(archive, data.table or "")
        meta_row = meta_rows.get(data.table or "")
        get = lambda row, column: row.get(column, _MISSING) if column else None
        score_col = data.score or (meta_row or {}).get((source.metadata or {}).get("score", ""), "")
        for name in ("scale", "baseline", "ceiling"):
            value = _field(meta_row, source, name)
            if value is not None:
                meta[name] = value
        header = set(rows[0]) if rows else set()
        for column in (source.model, score_col, data.effort, data.cost, data.date, data.upstream, *data.where):
            if column and rows and column not in header:
                raise ImportFailure(f"column {column!r} is not in {data.table}")
        if not score_col:
            raise ImportFailure("no score column: set `score` or give the source metadata")
    else:
        rows = payload
        get = lambda row, path: _dotted(row, path) if path else None
        score_col = data.score or ""
        if not score_col:
            raise ImportFailure("a json source needs `score`")
        for path in (source.model, score_col, data.effort, data.cost, data.date, data.upstream, *data.where):
            if path and rows and all(_dotted(row, path) is _MISSING for row in rows):
                raise ImportFailure(f"field {path!r} is in no row")
    scale = bench.scale if bench.scale is not None else meta.get("scale", 1.0)
    points, skipped = [], 0
    for row in rows:
        if any(str(get(row, column)) != value for column, value in data.where.items()):
            skipped += 1
            continue
        model = get(row, source.model)
        raw = _number(get(row, score_col))
        if model is _MISSING or not str(model or "").strip() or raw is None:
            skipped += 1
            continue
        model = str(model).strip()
        column_effort = scores.efforts.resolve(get(row, data.effort)) if data.effort else None
        _, label = split_model(model, scores.efforts)
        upstream = get(row, data.upstream) if data.upstream else None
        points.append({
            "model": model,
            "effort": column_effort if column_effort is not None else label,
            "score": 100 * raw * scale,
            "cost_usd": _number(get(row, data.cost)) if data.cost else None,
            "date": _date(get(row, data.date) if data.date else None, today),
            "upstream": "" if upstream in (None, _MISSING) else str(upstream),
        })
    return points, len(rows), skipped, meta


def _collapse(points: list[dict[str, Any]], scores: Scores) -> list[dict[str, Any]]:
    """One point per (model key, effort): the median score and cost of its rows."""
    groups: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    for p in points:
        # After `_unstated`: an alias's effort is the operator's, not a label this source writes.
        p["effort"] = scores.effort_of(p["model"], p["effort"])
        groups.setdefault((scores.key(p["model"]), p["effort"]), []).append(p)
    out = []
    for (_, effort), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
        out.append({
            "model": sorted(r["model"] for r in rows)[0],
            "effort": effort,
            "score": statistics.median(r["score"] for r in rows),
            "cost_usd": statistics.median(costs) if costs else None,
            "date": max(r["date"] for r in rows),
            "upstream": next((r["upstream"] for r in rows if r["upstream"]), ""),
        })
    return out


def _unstated(points: dict[str, list[dict[str, Any]]], scores: Scores) -> None:
    """A null effort means "not stated" where the same source has tier-matchable efforts for that model key."""
    with_efforts = {
        scores.key(p["model"]) for rows in points.values() for p in rows
        if p["effort"] is not None and scores.efforts.matchable(p["effort"])
    }
    for rows in points.values():
        for p in rows:
            if p["effort"] is None and scores.key(p["model"]) in with_efforts:
                p["effort"] = "unknown"


def import_sources(
    config: Config, scores: Scores, fetch: Fetch, *, only: str | None = None, now: datetime | None = None
) -> ImportReport:
    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    today = stamp[:10]
    path = Path(scores.imported_path)
    previous: dict[str, Any] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            previous = previous if isinstance(previous, dict) else {}
        except (OSError, ValueError):
            previous = {}
    out = {"imported_at": stamp, "sources": dict(previous.get("sources") or {}),
           "benchmarks": dict(previous.get("benchmarks") or {})}
    report = ImportReport()
    wanted: dict[str, list[Benchmark]] = {}
    for bench in scores.benchmarks.values():
        if bench.data is not None and (only is None or bench.data.source == only):
            wanted.setdefault(bench.data.source, []).append(bench)
    for name in sorted(wanted):
        source = scores.sources[name]
        headers: dict[str, str] = {}
        if source.api_key_env:
            key = os.environ.get(source.api_key_env)
            if not key:
                report.errors[name] = f"environment variable {source.api_key_env} is not set; source skipped"
                out["sources"][name] = {**out["sources"].get(name, {}), "status": "no key"}
                continue
            if urllib.parse.urlsplit(source.url).scheme.lower() != "https":
                report.errors[name] = "a source with an API key must use https; source skipped"
                out["sources"][name] = {**out["sources"].get(name, {}), "status": "failed"}
                continue
            headers[source.api_key_header] = key
        # One budget for the whole source, every page included.
        deadline = _clock() + scores.settings.refresh_timeout_s
        try:
            if source.format == "csv_zip":
                archive = zipfile.ZipFile(io.BytesIO(fetch(source.url, headers, deadline)))
                payload: Any = (archive, _metadata(archive, source))
            else:
                payload = _pages(source, fetch, headers, deadline)
        except Exception as exc:  # noqa: BLE001 - one source's failure must not lose the others
            report.errors[name] = f"{type(exc).__name__}: {exc}"[:300]
            out["sources"][name] = {**out["sources"].get(name, {}), "status": "failed"}
            continue
        summary = report.sources.setdefault(name, SourceReport())
        found: dict[str, list[dict[str, Any]]] = {}
        metas: dict[str, dict[str, float]] = {}
        for bench in wanted[name]:
            try:
                rows, read, skipped, meta = _rows_for(bench, source, payload, today, scores)
            except ImportFailure as exc:
                report.errors[bench.key] = str(exc)
                continue
            found[bench.key], metas[bench.key] = rows, meta
            summary.rows += read
            summary.skipped += skipped
        _unstated(found, scores)
        for key, rows in found.items():
            points = _collapse(rows, scores)
            if not points:  # an empty table or a `where` that matches nothing: not a reason to lose the points
                report.errors[key] = "no point in the data (no row with a model and a score passed `where`)"
                continue
            summary.points += len(points)
            summary.benchmarks.append(key)
            out["benchmarks"][key] = {"source": name, "fetched_at": stamp, **metas[key], "points": points}
        out["sources"][name] = {"fetched_at": stamp, "rows": summary.rows, "status": "ok"}
    write_json(path, out)
    return report


def stale(imported_at: str | None, hours: float, now: datetime | None = None) -> bool:
    if not imported_at:
        return True
    try:
        then = datetime.fromisoformat(imported_at)
    except ValueError:
        return True
    return ((now or datetime.now(timezone.utc)) - then).total_seconds() > hours * 3600


def refresh(config: Config, scores: Scores, fetch: Fetch | None = None, now: datetime | None = None
            ) -> tuple[bool, list[str]]:
    """At startup: import when the last import is older than `refresh_hours`. (imported?, lines to print)."""
    settings = scores.settings
    if settings.refresh_hours is None or not any(b.data for b in scores.benchmarks.values()):
        return False, []
    if not stale(scores.imported_at, settings.refresh_hours, now):
        return False, []
    try:
        report = import_sources(config, scores, fetch or http_fetch(settings.refresh_timeout_s), now=now)
    except Exception as exc:  # noqa: BLE001 - old evidence still routes
        return False, [f"benchmarks: import failed, using the import from {scores.imported_at or 'never'}: "
                       f"{type(exc).__name__}: {exc}"]
    lines = []
    if report.sources:
        lines.append("benchmarks: import refreshed (" + ", ".join(
            f"{name}: {s.points} point(s)" for name, s in sorted(report.sources.items())) + ")")
    for name, error in sorted(report.errors.items()):
        lines.append(f"benchmarks: import failed for {name}, keeping its previous points: {error}")
    return True, lines


__all__ = ["Fetch", "ImportReport", "SourceReport", "http_fetch", "import_sources", "refresh", "stale"]
