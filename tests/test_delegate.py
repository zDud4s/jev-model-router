"""`jev-model-router delegate`: a routed task run on another agent's CLI, and its tokens logged once.

The "CLI" here is a small Python script that prints recorded events: real pipes, a real process
tree to kill, and no real agent CLI.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from jev_model_router import delegate
from jev_model_router.delegate import CannotStart, HttpSource, LogSource, run_delegate
from jev_model_router.delegates import DEPTH_ENV, Target
from jev_model_router.db import RequestLog

from test_capabilities import raw_config
from test_codex_cli import EVENTS

TARGET = Target(tier="cx", runner="codex", model="gpt-6-sol", effort=None, executable="codex")

FAKE = r'''
import json, os, pathlib, sys, time
spec = json.loads(pathlib.Path(sys.argv[1]).read_text())
brief = sys.stdin.read()
pathlib.Path(spec["seen"]).write_text(json.dumps({"brief": brief, "depth": os.environ.get("JEV_MODEL_ROUTER_DEPTH"),
                                                   "key": os.environ.get("OPENAI_API_KEY")}))
for line in spec["lines"]:
    print(line, flush=True)
time.sleep(spec["sleep"])
sys.exit(spec["code"])
'''


class FakeCli:
    """A `spawn` that records the argv it was given and starts the fake script instead."""

    def __init__(self, tmp_path: Path, events, code=0, sleep=0.0):
        self.script = tmp_path / "fake_cli.py"
        self.script.write_text(FAKE)
        self.seen_file = tmp_path / "seen.json"
        self.spec = tmp_path / "spec.json"
        self.spec.write_text(json.dumps({"lines": [json.dumps(e) for e in events], "code": code,
                                         "sleep": sleep, "seen": str(self.seen_file)}))
        self.argv = None

    def __call__(self, argv, env, cwd):
        self.argv, self.cwd = argv, cwd
        return delegate._spawn([sys.executable, str(self.script), str(self.spec)], env, cwd)

    @property
    def seen(self):
        return json.loads(self.seen_file.read_text())


class FakeSource:
    def __init__(self, target=TARGET):
        self._target = target
        self.usage, self.limited = [], []

    def target(self, decision_id):
        if isinstance(self._target, Exception):
            raise self._target
        return self._target

    def record_usage(self, decision_id, usage):
        self.usage.append(usage)

    def report_rate_limited(self, decision_id, detail):
        self.limited.append(detail)

    def close(self):
        pass


class Out:
    def __init__(self):
        self.text = ""

    def write(self, s):
        self.text += s

    def flush(self):
        pass


def go(source, cli, brief="Fix the scheduler", **kwargs):
    out, err = Out(), Out()
    environ = {k: v for k, v in os.environ.items() if k != DEPTH_ENV}
    environ["OPENAI_API_KEY"] = "sk-must-not-reach"
    code = run_delegate("rt_1", source, brief, spawn=cli, which=lambda exe: f"/resolved/{exe}",
                        environ=environ, cwd=os.getcwd(), out=out, err=err, **kwargs)
    return code, out.text, err.text


def test_a_run_prints_the_final_message_and_records_its_usage(tmp_path):
    source, cli = FakeSource(), FakeCli(tmp_path, EVENTS)
    code, out, err = go(source, cli)
    assert code == 0 and out.strip() == "final answer"
    assert [u.prompt_tokens for u in source.usage] == [1200]
    assert cli.argv[0] == "/resolved/codex" and "--sandbox" in cli.argv and "workspace-write" in cli.argv
    assert cli.seen == {"brief": "Fix the scheduler", "depth": "1", "key": None}
    assert "[codex] final answer" in err


def test_the_access_level_reaches_the_argv(tmp_path):
    cli = FakeCli(tmp_path, EVENTS)
    go(FakeSource(), cli, access="read-only")
    assert cli.argv[cli.argv.index("--sandbox") + 1] == "read-only"


def test_a_failed_run_exits_2_and_still_records_what_it_spent(tmp_path):
    failed = [*EVENTS[:-1], {"type": "turn.completed", "usage": {"input_tokens": 50, "output_tokens": 5}},
              {"type": "turn.failed", "error": {"message": "sandbox denied"}}]
    source = FakeSource()
    code, _, err = go(source, FakeCli(tmp_path, failed, code=1))
    assert code == 2 and "sandbox denied" in err and [u.prompt_tokens for u in source.usage] == [50]


def test_a_spent_window_exits_3_and_reports_it(tmp_path):
    source = FakeSource()
    code, _, _ = go(source, FakeCli(tmp_path, [{"type": "error", "message": "You've hit your usage limit"}], code=1))
    assert code == 3 and source.limited and source.usage == []


@pytest.mark.parametrize("problem", ["depth", "source", "path", "brief", "access"])
def test_what_cannot_start_exits_4_and_runs_nothing(tmp_path, problem, monkeypatch):
    cli = FakeCli(tmp_path, EVENTS)
    source = FakeSource(CannotStart("no decision 'rt_1'") if problem == "source" else TARGET)
    out, err = Out(), Out()
    environ = {"PATH": os.environ.get("PATH", "")}
    if problem == "depth":
        environ[DEPTH_ENV] = "1"
    code = run_delegate("rt_1", source, "" if problem == "brief" else "brief", spawn=cli,
                        which=(lambda exe: None) if problem == "path" else (lambda exe: exe),
                        access="everything" if problem == "access" else "workspace-write",
                        environ=environ, cwd=os.getcwd(), out=out, err=err)
    assert code == 4 and cli.argv is None and err.text.startswith("delegate: ")


def test_an_interrupted_run_kills_its_tree(tmp_path, monkeypatch):
    killed = []
    real_kill = delegate.kill_tree
    monkeypatch.setattr(delegate, "kill_tree", lambda proc: (killed.append(proc.pid), real_kill(proc)))

    class Interrupting:
        runner = "codex"

        def progress(self, line):
            raise KeyboardInterrupt

    cli = FakeCli(tmp_path, EVENTS, sleep=60)
    proc = cli(["x"], dict(os.environ), os.getcwd())
    with pytest.raises(KeyboardInterrupt):
        delegate._collect(proc, Interrupting(), "brief", None, Out())
    assert killed == [proc.pid] and proc.wait(timeout=30) is not None


def test_a_timeout_kills_the_run_and_records_what_it_saw(tmp_path):
    usage = {"type": "turn.completed", "usage": {"input_tokens": 70, "output_tokens": 7}}
    source = FakeSource()
    started = time.monotonic()
    code, _, err = go(source, FakeCli(tmp_path, [usage], sleep=60), timeout=1.0)
    assert code == 2 and "timed out" in err
    assert time.monotonic() - started < 30
    assert [u.prompt_tokens for u in source.usage] == [70]


# ------------------------------------------------------------- sources
def write_config(tmp_path: Path, log_path="logs/router.db") -> Path:
    raw = raw_config()
    raw["log"] = {"path": log_path}
    path = tmp_path / "cfg" / "config.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(raw))
    return path


def routed(log, tier="cx", runner="codex", model="gpt-6-sol"):
    log.record_decision("rt_1", task="t", tier=tier, model=model, effort=None, runner=runner, stage=None,
                        route_score=None, route_model=None, route_reason=None)


def test_the_log_source_reads_the_configs_log_beside_it_without_changing_directory(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    log = RequestLog(config.parent / "logs" / "router.db")
    routed(log)
    elsewhere = tmp_path / "project"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    source = LogSource(config)
    try:
        assert source.target("rt_1") == TARGET
        assert Path.cwd() == elsewhere
        with pytest.raises(CannotStart):
            source.target("rt_nope")
    finally:
        source.close()


def test_a_tier_the_config_no_longer_has_runs_its_runners_cli(tmp_path):
    config = write_config(tmp_path)
    routed(RequestLog(config.parent / "logs" / "router.db"), tier="codex:gone@high", model="m")
    source = LogSource(config)
    try:
        assert source.target("rt_1").executable == "codex"
    finally:
        source.close()


def test_the_log_source_records_usage_once_and_a_rate_limit_after_the_caller_is_fine(tmp_path):
    from jev_model_router.schemas import Usage

    config = write_config(tmp_path)
    log = RequestLog(config.parent / "logs" / "router.db")
    routed(log)
    source = LogSource(config)
    try:
        source.record_usage("rt_1", Usage(prompt_tokens=10, completion_tokens=1))
        source.record_usage("rt_1", Usage(prompt_tokens=99, completion_tokens=9))
        log.set_outcome("rt_1", "error", None, None, "app")
        source.report_rate_limited("rt_1", "limit")  # "exists": the caller reported first
    finally:
        source.close()
    row = log.decision("rt_1")
    assert row["outcome_input_tokens"] == 10 and row["outcome"] == "error"
    assert [r["writer"].startswith("delegate-") for r in log.query("SELECT writer FROM route_events WHERE kind='usage'")] == [True]


def test_the_http_source_refuses_a_router_that_is_not_this_machine():
    with pytest.raises(CannotStart):
        HttpSource("http://example.com:8080")
    HttpSource("http://localhost:8080").close()
    HttpSource("http://[::1]:8080").close()


def test_the_http_source_reads_the_decision_and_posts_usage_and_the_rate_limit(backend_factory):
    from test_route_api import TASK, client_for
    from jev_model_router.schemas import Usage

    client, log, _ = client_for(backend_factory)
    with client:
        decision = client.post("/v1/route", json={**TASK, "runners": ["codex"]}).json()["decision_id"]
        source = HttpSource("http://127.0.0.1:8080", client=client)
        assert source.target(decision) == Target("cx", "codex", "gpt-6-sol", None, "codex")
        source.record_usage(decision, Usage(prompt_tokens=10, completion_tokens=1))
        client.post(f"/v1/route/{decision}/outcome", json={"status": "error"})
        source.report_rate_limited(decision, "limit")  # a 409: counts as done
        with pytest.raises(CannotStart):
            source.target("rt_nope")
        row = log.decision(decision)  # inside the block: the lifespan closes the log
    assert row["outcome_input_tokens"] == 10 and row["outcome"] == "error"
