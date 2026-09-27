"""Task shapes read from local agent transcripts, for `jev-model-router measure-shape`.

A task routed through `/v1/route` is an agent loop: it re-sends a growing
context every turn, and the provider serves most of it from its prompt cache.
The router prices a task by its shape (`config.TaskShape`); this measures one
from what this machine's agents actually ran. Read-only, offline, and it names
no model: it reports the model strings it finds. A line or file that does not
parse is counted and skipped.

- Claude Code, `~/.claude/projects/<project>/**/*.jsonl`: a message is written
  once per content block, so each `usage` counts once, by message id. One task
  is a session, or a subagent sidechain within one.
- Codex, `~/.codex/sessions/**/rollout-*.jsonl`: usage is cumulative, so the
  last `total_token_usage` is the task's. Codex reports no cache writes.
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
    groups: dict[tuple[Any, Any], tuple[Counter[str], Task]] = {}
    seen: set[str] = set()
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
            mid = message.get("id") or record.get("requestId") or record.get("uuid")
            if mid:
                if mid in seen:
                    continue
                seen.add(mid)
            key = (record.get("sessionId"), record.get("agentId") if record.get("isSidechain") else None)
            models, task = groups.setdefault(key, (Counter(), Task("claude", "")))
            models[str(message.get("model") or "?")] += 1
            task.fresh += _int(usage.get("input_tokens"))
            task.read += _int(usage.get("cache_read_input_tokens"))
            task.write += _int(usage.get("cache_creation_input_tokens"))
            task.output += _int(usage.get("output_tokens"))
    for models, task in groups.values():
        task.model = models.most_common(1)[0][0]
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
        cached = _int(last.get("cached_input_tokens"))
        task = Task("codex", str(model or "?"), fresh=max(_int(last.get("input_tokens")) - cached, 0),
                    read=cached, output=_int(last.get("output_tokens")))
        if task.input > 0 and task.output > 0:
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
