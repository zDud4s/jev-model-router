"""Task shapes read from local agent transcripts, for `jev-model-router measure-shape`.

A task routed through `/v1/route` is an agent loop: it re-sends a growing
context every turn, and the provider serves most of it from its prompt cache.
The router prices a task by its shape (`config.TaskShape`); this measures one
from what this machine's agents actually ran. Read-only, offline, and it names
no model: it reports the model strings it finds. A line or file that does not
parse is counted and skipped.

- Claude Code, `~/.claude/projects/<project>/**/*.jsonl`: a message can be
  written across several lines as it streams, and an earlier line may carry
  only partial (streaming-start) usage, so lines are grouped by message id
  and only the largest value of each usage field counts, once, per message.
  One task is a session, or a subagent sidechain within one. `--since` drops
  individual messages timestamped before it.
- Codex, `~/.codex/sessions/**/rollout-*.jsonl`: usage is cumulative, so the
  last `total_token_usage` is the task's. Codex reports no cache writes.
  `--since` keeps or drops a whole rollout by its last timestamp, since an
  earlier cumulative total cannot be recovered on its own.

A `--since` date given with no timezone is read as UTC.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .calls import accepts_task_row
from .config import TaskShape


@dataclass
class Task:
    source: str  # "claude" | "codex"
    model: str
    fresh: int = 0
    read: int = 0
    write: int = 0
    output: int = 0

    @property
    def input(self) -> int:
        return self.fresh + self.read + self.write


@dataclass
class Scan:
    tasks: list[Task] = field(default_factory=list)
    bad_lines: int = 0
    bad_files: int = 0


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _records(path: Path, scan: Scan) -> Iterator[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    scan.bad_lines += 1
                    continue
                if isinstance(record, dict):
                    yield record
    except OSError:
        scan.bad_files += 1


def _before(stamp: Any, since: datetime | None) -> bool:
    if since is None or not isinstance(stamp, str):
        return False
    try:
        when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when < since


def claude_tasks(root: Path, *, match: str | None = None, since: datetime | None = None,
                 scan: Scan | None = None) -> Scan:
    scan = scan if scan is not None else Scan()
    # Model tally per (session, sidechain) group. Usage is merged globally by
    # message id, not per group: a resume/continue can copy a message into
    # another session's file, and that copy must not add its usage again. A
    # message belongs to the group it is first seen in; a later line for the
    # same id -- there or in another group -- only merges into it by max (a
    # message split across streamed lines may report partial usage on an
    # earlier line). A line with no id to merge on stands alone.
    groups: dict[tuple[Any, Any], Counter[str]] = {}
    merged: dict[Any, tuple[tuple[Any, Any], list[int]]] = {}
    for path in sorted(Path(root).glob("*/**/*.jsonl")):
        project = path.relative_to(root).parts[0]
        for record in _records(path, scan):
            message = record.get("message")
            usage = message.get("usage") if isinstance(message, dict) else None
            if not isinstance(usage, dict):
                continue
            if match and match.lower() not in f"{project} {record.get('cwd') or ''}".lower():
                continue
            if _before(record.get("timestamp"), since):
                continue
            agent = (record.get("agentId") or "sidechain") if record.get("isSidechain") else None
            key = (record.get("sessionId"), agent)
            groups.setdefault(key, Counter())[str(message.get("model") or "?")] += 1
            fields = [_int(usage.get("input_tokens")), _int(usage.get("cache_read_input_tokens")),
                      _int(usage.get("cache_creation_input_tokens")), _int(usage.get("output_tokens"))]
            mid = message.get("id") or record.get("requestId") or object()  # no id: stands alone
            existing = merged.get(mid)
            if existing is None:
                merged[mid] = (key, fields)
            else:
                first_key, current = existing
                merged[mid] = (first_key, [max(a, b) for a, b in zip(current, fields)])
    totals: dict[tuple[Any, Any], list[int]] = {}
    for group_key, fields in merged.values():
        total = totals.setdefault(group_key, [0, 0, 0, 0])
        for i, value in enumerate(fields):
            total[i] += value
    for key, (fresh, read, write, output) in totals.items():
        models = groups.get(key, Counter())
        task = Task("claude", models.most_common(1)[0][0] if models else "?",
                    fresh=fresh, read=read, write=write, output=output)
        if task.input > 0 and task.output > 0:
            scan.tasks.append(task)
    return scan


def codex_tasks(root: Path, *, match: str | None = None, since: datetime | None = None,
                scan: Scan | None = None) -> Scan:
    scan = scan if scan is not None else Scan()
    for path in sorted(Path(root).glob("**/rollout-*.jsonl")):
        model, cwd, last, stamp = None, "", None, None
        for record in _records(path, scan):
            payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
            if record.get("type") in ("session_meta", "turn_context") or payload.get("type") == "turn_context":
                model = payload.get("model") or model
                cwd = payload.get("cwd") or cwd
            if payload.get("type") == "token_count" and isinstance(payload.get("info"), dict):
                last = payload["info"].get("total_token_usage") or last
            stamp = record.get("timestamp") or stamp
        if not isinstance(last, dict) or (match and match.lower() not in str(cwd).lower()):
            continue
        if _before(stamp, since):
            continue
        raw_input = _int(last.get("input_tokens"))
        cached = _int(last.get("cached_input_tokens"))
        task = Task("codex", str(model or "?"), fresh=max(raw_input - cached, 0),
                    read=cached, output=_int(last.get("output_tokens")))
        # Codex always reports its cache (0 when unused), so this format's rows share
        # the same acceptance rule `calls.py` applies to a live call's outcome. The raw
        # (unclamped) input_tokens goes in, not task.input: clamping fresh to 0 when
        # cached_input_tokens overstates input_tokens would otherwise hide that the row
        # doesn't add up.
        if accepts_task_row(raw_input, task.output, task.read, task.write, reports_cache=True):
            scan.tasks.append(task)
    return scan


def pooled(tasks: list[Task]) -> TaskShape | None:
    total_in = sum(t.input for t in tasks)
    total_out = sum(t.output for t in tasks)
    if total_in <= 0 or total_out <= 0:
        return None
    return TaskShape(total_in / total_out, sum(t.read for t in tasks) / total_in,
                     sum(t.write for t in tasks) / total_in, f"measured:{len(tasks)}")


def config_block(shape: TaskShape, note: str) -> str:
    # Rounded down, so the two shares can never add up past 1 once printed.
    read = math.floor(shape.cache_read * 1000) / 1000
    write = math.floor(shape.cache_write * 1000) / 1000
    return (
        "router:\n  capabilities:\n"
        f"    task_shape:  # {note}\n"
        f"      input_per_output: {max(1, round(shape.input_per_output))}\n"
        f"      cache_read: {read}\n"
        f"      cache_write: {write}\n"
    )


def report(scan: Scan) -> str:
    lines = [f"tasks: {len(scan.tasks)}"
             + (f"  (skipped {scan.bad_lines} unreadable line(s), {scan.bad_files} file(s))"
                if scan.bad_lines or scan.bad_files else "")]
    if not scan.tasks:
        lines.append("no task with input and output found; nothing to measure")
        return "\n".join(lines)
    lines.append("tasks per model")
    for (source, model), n in Counter((t.source, t.model) for t in scan.tasks).most_common():
        lines.append(f"  {source:<7} {model:<40} {n:>6}")
    lines.append("shape: pooled in/out, median in/out, share read from cache, share written")
    for label in ("claude", "codex", "all"):
        group = scan.tasks if label == "all" else [t for t in scan.tasks if t.source == label]
        shape = pooled(group)
        if shape is None:
            continue
        median = statistics.median(t.input / t.output for t in group)
        lines.append(f"  {label:<7} {len(group):>6} tasks  {shape.input_per_output:>7.0f}  {median:>7.0f}  "
                     f"read {shape.cache_read:.3f}  write {shape.cache_write:.3f}")
    lines += ["", "paste into the config:",
              config_block(pooled(scan.tasks), f"measure-shape {date.today()}, {len(scan.tasks)} tasks")]
    return "\n".join(lines)


__all__ = ["Scan", "Task", "claude_tasks", "codex_tasks", "config_block", "pooled", "report"]
