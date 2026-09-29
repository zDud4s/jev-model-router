"""`jev-model-router delegate`: run a routed task on the CLI of the tier chosen, in the caller's project.

The route tool may choose a tier the calling agent cannot run itself (a Claude Code session, a tier
`codex` runs). This runs it: the adapter for the tier's runner builds the command, the brief goes on
stdin, one progress line per event goes to stderr, the final message to stdout, and the run's tokens
to the log -- once, even when the run failed or timed out.

Exit codes: 0 ran; 2 the CLI failed or timed out; 3 rate limit, reported as the outcome here;
4 could not start (nothing was run).
"""

from __future__ import annotations

import ipaddress
import os
import shutil
import subprocess
import sys
import threading
import uuid
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import IO, Any, Callable, Mapping, Protocol
from urllib.parse import quote, urlsplit

import httpx

from .delegates import ACCESS, DEPTH_ENV, Adapter, RunResult, Target, adapter_for
from .schemas import Usage

RAN, FAILED, LIMITED, CANNOT = 0, 2, 3, 4


class CannotStart(Exception):
    """Exit 4: the reason nothing was run."""


def refused_at_depth(environ: Mapping[str, str]) -> str | None:
    try:
        depth = int(environ.get(DEPTH_ENV) or 0)
    except ValueError:
        depth = 1  # set but unreadable: something set it, so this is a delegated run
    return "this is already a delegated run; it cannot delegate again" if depth >= 1 else None


class Source(Protocol):
    def target(self, decision_id: str) -> Target: ...
    def record_usage(self, decision_id: str, usage: Usage) -> None: ...
    def report_rate_limited(self, decision_id: str, detail: str) -> None: ...
    def close(self) -> None: ...


class LogSource:
    """In process: the config's log, opened bare -- no router, no backends, no change of directory."""

    def __init__(self, config_path: str | os.PathLike[str]) -> None:
        from .config import ConfigError, load_config
        from .db import RequestLog

        path = Path(config_path).resolve()
        try:
            self._config = load_config(path)
        except ConfigError as exc:
            raise CannotStart(f"config error: {exc}") from exc
        log_path = self._config.log.path
        if log_path != ":memory:" and not Path(log_path).is_absolute():
            # What it means under `serve` run beside the config; this command runs in the project.
            log_path = str(path.parent / log_path)
        self._log = RequestLog(log_path)
        self.writer = f"delegate-{uuid.uuid4().hex}"

    def target(self, decision_id: str) -> Target:
        row = self._log.decision(decision_id)
        if row is None:
            raise CannotStart(f"no decision {decision_id!r} in {self._log.path}")
        runner = row["runner"] or ""
        tier = self._config.tiers.get(row["tier"])
        # A tier the catalog discovered is not in the config file: its runner's CLI on PATH runs it.
        executable = tier.base_url if tier is not None else runner
        return Target(tier=row["tier"], runner=runner, model=row["model"] or "", effort=row["effort"],
                      executable=executable)

    def record_usage(self, decision_id: str, usage: Usage) -> None:
        self._log.set_usage(decision_id, usage, self.writer)

    def report_rate_limited(self, decision_id: str, detail: str) -> None:
        # "exists" is fine: the caller reported first, and the first outcome is the one kept.
        self._log.set_outcome(decision_id, "rate_limited", detail, None, self.writer)

    def close(self) -> None:
        self._log.close()


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class HttpSource:
    """A running proxy, which keeps its own ledger and shapes. This machine only: its answer names an executable to run."""

    def __init__(self, url: str, client: httpx.Client | None = None) -> None:
        host = urlsplit(url).hostname or ""
        if not _loopback(host):
            raise CannotStart(f"--url must be this machine (localhost, 127.0.0.1, ::1), not {host!r}")
        self._client = client or httpx.Client(base_url=url.rstrip("/"), timeout=30.0)

    @staticmethod
    def _path(decision_id: str, tail: str = "") -> str:
        return f"/v1/route/{quote(decision_id, safe='')}{tail}"

    def target(self, decision_id: str) -> Target:
        try:
            response = self._client.get(self._path(decision_id))
        except httpx.HTTPError as exc:
            raise CannotStart(f"cannot reach the router: {type(exc).__name__}: {exc}") from exc
        if response.status_code == 404:
            raise CannotStart(f"no decision {decision_id!r}")
        if response.status_code >= 400:
            raise CannotStart(f"the router answered HTTP {response.status_code}: {response.text[:300]}")
        body = response.json()
        if not body.get("executable"):
            raise CannotStart(f"no delegate runs {body.get('runner')!r} tiers")
        return Target.from_dict(body)

    def record_usage(self, decision_id: str, usage: Usage) -> None:
        fields = {"prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens"}
        self._client.post(self._path(decision_id, "/usage"), json=usage.model_dump(include=fields)).raise_for_status()

    def report_rate_limited(self, decision_id: str, detail: str) -> None:
        response = self._client.post(self._path(decision_id, "/outcome"),
                                     json={"status": "rate_limited", "detail": detail})
        if response.status_code != 409:  # 409: the caller reported first
            response.raise_for_status()

    def close(self) -> None:
        self._client.close()


Spawn = Callable[[list[str], Mapping[str, str], str], "subprocess.Popen[str]"]


def _spawn(argv: list[str], env: Mapping[str, str], cwd: str) -> "subprocess.Popen[str]":
    # Its own process group, so a timeout can kill the shells and MCP servers the CLI started too.
    group: dict[str, Any] = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    )
    return subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=dict(env), cwd=cwd,
        text=True, encoding="utf-8", errors="replace", **group,
    )


def kill_tree(proc: "subprocess.Popen[str]") -> None:
    """The process and everything it started: the CLIs leave shells and MCP servers behind a plain kill."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        if proc.poll() is None:
            # A restricted host can deny taskkill; at least stop the CLI itself instead of hanging.
            proc.kill()
        return
    import signal

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def _collect(proc: "subprocess.Popen[str]", adapter: Adapter, brief: str, timeout: float | None,
             err: IO[str]) -> tuple[RunResult, bool]:
    lines: list[str] = []
    tail: deque[str] = deque(maxlen=50)

    def feed() -> None:
        try:
            proc.stdin.write(brief)
            proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass

    def drain() -> None:
        for line in proc.stderr:
            tail.append(line)

    helpers = [threading.Thread(target=feed, daemon=True), threading.Thread(target=drain, daemon=True)]
    for thread in helpers:
        thread.start()
    expired = threading.Event()
    timer = None
    if timeout:
        def expire() -> None:
            expired.set()
            kill_tree(proc)

        timer = threading.Timer(timeout, expire)
        timer.daemon = True
        timer.start()
    try:
        for line in proc.stdout:
            lines.append(line)
            note = adapter.progress(line)
            if note:
                print(f"[{adapter.runner}] {note}", file=err, flush=True)
    except BaseException:
        # Ctrl-C, or the host stopping a background task: the child is in its own process group,
        # so nothing reaches it unless this kills it.
        kill_tree(proc)
        raise
    finally:
        if timer is not None:
            timer.cancel()
    returncode = proc.wait()
    for thread in helpers:
        thread.join(timeout=5)
    return adapter.read(lines, returncode, "".join(tail)), expired.is_set()


def _safely(err: IO[str], what: str, call: Callable[..., Any], *args: Any) -> None:
    try:
        call(*args)
    except Exception as exc:  # noqa: BLE001 - the run happened; losing its record must not hide its result
        print(f"delegate: {what} failed: {type(exc).__name__}: {exc}", file=err)


def run_delegate(
    decision_id: str,
    source: Source,
    brief: str,
    *,
    access: str = "workspace-write",
    timeout: float | None = None,
    cwd: str | None = None,
    environ: Mapping[str, str] | None = None,
    spawn: Spawn | None = None,
    which: Callable[[str], str | None] | None = None,
    out: IO[str] | None = None,
    err: IO[str] | None = None,
) -> int:
    """Run the decision's tier on its CLI; the exit code (module docstring)."""
    out, err = out or sys.stdout, err or sys.stderr
    environ = os.environ if environ is None else environ
    cwd = cwd or os.getcwd()
    spawn, which = spawn or _spawn, which or shutil.which
    try:
        refusal = refused_at_depth(environ)
        if refusal:
            raise CannotStart(refusal)
        if access not in ACCESS:
            raise CannotStart(f"--access must be one of {list(ACCESS)}, not {access!r}")
        if not brief.strip():
            raise CannotStart("the brief is empty: pass --brief-file, or pipe it on stdin")
        target = source.target(decision_id)
        adapter = adapter_for(target.runner)
        if adapter is None:
            raise CannotStart(f"no delegate runs {target.runner!r} tiers")
        resolved = which(target.executable)
        if resolved is None:
            raise CannotStart(f"{target.executable!r} is not on PATH here")
    except CannotStart as exc:
        print(f"delegate: {exc}", file=err)
        return CANNOT
    argv = adapter.argv(replace(target, executable=resolved), access, cwd)
    env = adapter.env(environ)
    env[DEPTH_ENV] = "1"  # the run, and every shell it starts, cannot delegate again
    try:
        proc = spawn(argv, env, cwd)
    except OSError as exc:
        print(f"delegate: cannot run {resolved!r}: {exc}", file=err)
        return CANNOT
    try:
        result, timed_out = _collect(proc, adapter, brief, timeout, err)
    except KeyboardInterrupt:
        print("delegate: interrupted; the run was killed", file=err)
        return FAILED
    if result.usage.prompt_tokens or result.usage.completion_tokens:
        _safely(err, "recording the usage", source.record_usage, decision_id, result.usage)
    if timed_out:
        print(f"delegate: timed out after {timeout:.0f}s; the run was killed", file=err)
        return FAILED
    if result.rate_limited:
        _safely(err, "reporting the rate limit", source.report_rate_limited, decision_id,
                (result.failure or "rate limited")[:2000])
        print(f"delegate: {target.runner} is rate limited: {result.failure}", file=err)
        return LIMITED
    if result.failure is not None:
        print(f"delegate: {target.runner} failed: {result.failure}", file=err)
        return FAILED
    print(result.message or "", file=out)
    return RAN


__all__ = ["CANNOT", "CannotStart", "FAILED", "HttpSource", "LIMITED", "LogSource", "RAN", "kill_tree",
           "refused_at_depth", "run_delegate"]
