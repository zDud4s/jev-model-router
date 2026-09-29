"""`codex exec` given a routed task, in the caller's project, inside the sandbox its access level names."""

from __future__ import annotations

import json
from typing import Sequence

from ..backends.codex_cli import _read_events
from ..cli_runs import says_limited
from .base import DEPTH_ENV, Adapter, RunResult, Target, clip, parse_event

# An OS sandbox: workspace-write keeps writes in the workspace and the shell off the network.
_ACCESS = {
    "read-only": ["--sandbox", "read-only"],
    "workspace-write": ["--sandbox", "workspace-write"],
    "full": ["--dangerously-bypass-approvals-and-sandbox"],
}


class CodexAdapter(Adapter):
    runner = "codex"
    windows_shell = "powershell"  # Codex runs commands in PowerShell on Windows

    def argv(self, target: Target, access: str, cwd: str) -> list[str]:
        argv = [target.executable, "exec", "--json", "--skip-git-repo-check", "-C", cwd, "-m", target.model]
        if target.effort:
            argv += ["-c", f"model_reasoning_effort={json.dumps(target.effort)}"]
        # Codex filters its shell's environment by policy, so the depth set on this process may not
        # reach the commands the run starts. This sets it there whatever the user's policy says.
        argv += ["-c", f'shell_environment_policy.set.{DEPTH_ENV}="1"']
        # `-`: the brief comes on stdin, which a long, multi-line brief survives and argv does not.
        return argv + _ACCESS[access] + ["-"]

    def progress(self, line: str) -> str | None:
        event = parse_event(line)
        if event is None:
            return None
        kind = event.get("type")
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        if kind == "item.started" and item.get("type") == "command_execution":
            return f"$ {clip(item.get('command'))}"
        if kind == "item.completed" and item.get("type") == "agent_message":
            return clip(item.get("text"))
        if kind == "turn.completed":
            return "done"
        return None

    def read(self, lines: Sequence[str], returncode: int, stderr: str) -> RunResult:
        answer, usage, failure = _read_events("".join(lines))
        if failure is None and returncode != 0:
            failure = (stderr or "").strip()[-2000:] or f"exited {returncode}"
        if failure is not None:
            return RunResult(answer, usage, failure, says_limited(failure))
        if answer is None:
            return RunResult(None, usage, "no agent message")
        return RunResult(answer, usage, None)
