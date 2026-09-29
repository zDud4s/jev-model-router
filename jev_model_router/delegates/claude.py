"""`claude -p` given a routed task, in the caller's project, with the tools its access level allows."""

from __future__ import annotations

from typing import Sequence

from ..backends.claude_cli import _usage_of
from ..cli_runs import says_limited
from ..schemas import Usage
from .base import Adapter, RunResult, Target, clip, parse_event

# `claude -p` cannot ask for permission, so each level says up front what it may do. workspace-write
# pre-allows the shell and has no sandbox on any platform: a shell command can write anywhere and
# reach the network. It is not the guarantee codex's workspace-write is (DESIGN.md says so).
_ACCESS = {
    "read-only": ["--tools", "Read,Grep,Glob", "--permission-mode", "dontAsk"],
    "workspace-write": ["--permission-mode", "acceptEdits", "--allowedTools", "Bash", "PowerShell",
                        "--permission-prompts", "none"],
    "full": ["--dangerously-skip-permissions"],
}


class ClaudeAdapter(Adapter):
    runner = "claude"
    windows_shell = "posix"  # Claude Code runs commands in Git Bash on Windows

    def argv(self, target: Target, access: str, cwd: str) -> list[str]:
        # stream-json needs --verbose; its last `result` event carries usage, is_error and subtype.
        # The cwd is the process's own: `claude` has no directory flag.
        argv = [target.executable, "-p", "--output-format", "stream-json", "--verbose", "--model", target.model]
        if target.effort:
            argv += ["--effort", target.effort]
        return argv + _ACCESS[access]

    def progress(self, line: str) -> str | None:
        event = parse_event(line)
        if event is None:
            return None
        if event.get("type") == "result":
            return f"done ({event.get('subtype')})"
        if event.get("type") != "assistant":
            return None
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        for part in message.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "tool_use":
                return f"tool {part.get('name')}"
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                return clip(part["text"])
        return None

    def read(self, lines: Sequence[str], returncode: int, stderr: str) -> RunResult:
        result = None
        for line in lines:
            event = parse_event(line)
            if event is not None and event.get("type") == "result":
                result = event
        if result is None:
            failure = (stderr or "").strip()[-2000:] or f"exited {returncode} with no result"
            return RunResult(None, Usage(), failure, says_limited(failure))
        usage = _usage_of(result.get("usage"))
        text = result.get("result") if isinstance(result.get("result"), str) else None
        if result.get("is_error") or returncode != 0:
            failure = text or str(result.get("subtype") or f"exited {returncode}")
            # A spent window can arrive as an error result with no HTTP status: the text says so.
            limited = result.get("api_error_status") == 429 or says_limited(failure)
            return RunResult(text, usage, failure, limited)
        return RunResult(text, usage, None)
