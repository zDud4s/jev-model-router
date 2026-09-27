"""measure-shape: task shapes read from local agent transcripts. Fixture files only; nothing real is read."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from jev_model_router import cli
from jev_model_router.config import parse_config
from jev_model_router.shape import Scan, claude_tasks, codex_tasks, config_block, pooled, report

from test_capabilities import raw_config

TS = "2026-09-20T10:00:00Z"


def write(path: Path, records: list[dict], junk: tuple[str, ...] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([json.dumps(r) for r in records] + list(junk)) + "\n", encoding="utf-8")


def turn(session, mid, *, fresh=10, read=900, write=50, out=20, agent=None, model="model-a", cwd="C:/work/proj",
        ts=TS):
    record = {"sessionId": session, "cwd": cwd, "timestamp": ts, "message": {
        "id": mid, "model": model, "usage": {"input_tokens": fresh, "cache_read_input_tokens": read,
                                             "cache_creation_input_tokens": write, "output_tokens": out}}}
    if agent:
        record.update(isSidechain=True, agentId=agent)
    return record


def rollout(cwd, *totals, ts=TS):
    records = [{"type": "session_meta", "timestamp": ts, "payload": {"cwd": cwd}},
               {"type": "turn_context", "timestamp": ts, "payload": {"model": "model-b", "cwd": cwd}}]
    for inp, cached, out in totals:
        records.append({"type": "event_msg", "timestamp": ts, "payload": {"type": "token_count", "info": {
            "total_token_usage": {"input_tokens": inp, "cached_input_tokens": cached, "output_tokens": out}}}})
    return records


def test_a_message_written_twice_counts_once(tmp_path):
    write(tmp_path / "proj" / "s1.jsonl", [turn("s1", "m1"), turn("s1", "m1"), turn("s1", "m2")])
    (task,) = claude_tasks(tmp_path).tasks
    assert (task.read, task.write, task.output) == (1800, 100, 40)


def test_a_sidechain_is_its_own_task(tmp_path):
    write(tmp_path / "proj" / "s1.jsonl", [turn("s1", "m1")])
    write(tmp_path / "proj" / "s1" / "subagents" / "agent-x.jsonl", [turn("s1", "m2", agent="x")])
    assert len(claude_tasks(tmp_path).tasks) == 2


def test_codex_uses_the_last_cumulative_usage(tmp_path):
    write(tmp_path / "2026" / "rollout-1.jsonl", rollout("C:/work/proj", (1000, 900, 10), (3000, 2800, 30)))
    (task,) = codex_tasks(tmp_path).tasks
    assert (task.fresh, task.read, task.write, task.output, task.model) == (200, 2800, 0, 30, "model-b")


def test_match_keeps_the_project_named(tmp_path):
    claude, codex = tmp_path / "claude", tmp_path / "codex"
    write(claude / "C--work-proj" / "s1.jsonl", [turn("s1", "m1")])
    write(claude / "C--work-other" / "s2.jsonl", [turn("s2", "m2", cwd="C:/work/other")])
    write(codex / "rollout-1.jsonl", rollout("C:/work/proj", (1000, 900, 10)))
    write(codex / "rollout-2.jsonl", rollout("C:/work/other", (1000, 900, 10)))
    assert len(claude_tasks(claude, match="proj").tasks) == 1
    assert len(codex_tasks(codex, match="proj").tasks) == 1


def test_a_line_that_does_not_parse_is_counted_and_skipped(tmp_path):
    write(tmp_path / "proj" / "s1.jsonl", [turn("s1", "m1")], junk=("{not json",))
    scan = claude_tasks(tmp_path)
    assert len(scan.tasks) == 1 and scan.bad_lines == 1


def test_the_printed_block_is_a_valid_task_shape(tmp_path):
    write(tmp_path / "proj" / "s1.jsonl", [turn("s1", "m1"), turn("s1", "m2", read=500, write=500)])
    block = config_block(pooled(claude_tasks(tmp_path).tasks), "test")
    shape = yaml.safe_load(block)["router"]["capabilities"]["task_shape"]
    assert parse_config(raw_config(task_shape=shape)).router.capabilities.task_shape.cache_write > 0


def test_the_command_prints_the_report(tmp_path, capsys):
    write(tmp_path / "claude" / "proj" / "s1.jsonl", [turn("s1", "m1")])
    code = cli.main(["measure-shape", "--claude-dir", str(tmp_path / "claude"),
                     "--codex-dir", str(tmp_path / "codex")])
    out = capsys.readouterr().out
    assert code == 0 and "model-a" in out and "task_shape:" in out
    assert report(Scan()).startswith("tasks: 0")


def test_a_split_message_counts_its_fullest_usage_once(tmp_path):
    write(tmp_path / "proj" / "s1.jsonl", [turn("s1", "m1", out=10), turn("s1", "m1", out=30)])
    (task,) = claude_tasks(tmp_path).tasks
    assert task.output == 30  # the larger of the two lines, not their sum


def test_since_drops_an_earlier_claude_message_but_keeps_a_later_one(tmp_path):
    since = datetime(2026, 9, 15, tzinfo=timezone.utc)
    write(tmp_path / "proj" / "s1.jsonl", [
        turn("s1", "m1", ts="2026-09-01T00:00:00Z", out=10),
        turn("s1", "m2", ts="2026-09-20T00:00:00Z", out=20),
    ])
    (task,) = claude_tasks(tmp_path, since=since).tasks
    assert task.output == 20


def test_since_drops_a_codex_rollout_whose_last_stamp_is_earlier(tmp_path):
    since = datetime(2026, 9, 15, tzinfo=timezone.utc)
    write(tmp_path / "rollout-1.jsonl", rollout("C:/work/proj", (1000, 900, 10), ts="2026-09-01T00:00:00Z"))
    assert codex_tasks(tmp_path, since=since).tasks == []


def test_an_unparseable_since_is_rejected(tmp_path):
    code = cli.main(["measure-shape", "--since", "nope", "--claude-dir", str(tmp_path / "claude"),
                     "--codex-dir", str(tmp_path / "codex")])
    assert code == 2
