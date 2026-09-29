"""What every delegate adapter is: one agent CLI, given a routed task, working in the caller's project.

Unlike the proxy's CLI backends (a clean room with no tools), a delegated run is an agent doing the
work where the caller is: it keeps the user's own CLI config, and its tools are what the access level
allows. Model and effort are set by flag, so that config cannot override the router's choice.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from ..cli_runs import scrubbed
from ..schemas import Usage

# read-only: look, never change. workspace-write (the default): edit and run commands in the
# project. full: no limits; only when the user asked for it (see the access table in DESIGN.md).
ACCESS = ("read-only", "workspace-write", "full")
# Set to 1 in every delegated run's environment; a run that sees it cannot delegate again.
DEPTH_ENV = "JEV_MODEL_ROUTER_DEPTH"


@dataclass(frozen=True)
class Target:
    """What a decision asks to run: its tier and runner, the model and effort, and the executable."""

    tier: str
    runner: str
    model: str
    effort: str | None
    executable: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Target":
        return cls(
            tier=str(raw["tier"]), runner=str(raw["runner"]), model=str(raw["model"]),
            effort=str(raw["effort"]) if raw.get("effort") else None, executable=str(raw["executable"]),
        )


@dataclass(frozen=True)
class RunResult:
    message: str | None  # the run's final message
    usage: Usage
    failure: str | None  # why the run failed; None when it did not
    rate_limited: bool = False


class Adapter:
    """One agent CLI. Subclasses set `runner` and implement `argv` and `read`."""

    runner = ""
    # The shell the host agent runs commands in on Windows, for quoting a command line it will
    # paste: "posix" (Git Bash) or "powershell".
    windows_shell = "posix"

    def available(self, target: Target, which: Callable[[str], str | None] = shutil.which) -> bool:
        return which(target.executable) is not None

    def argv(self, target: Target, access: str, cwd: str) -> list[str]:
        raise NotImplementedError

    def env(self, base: Mapping[str, str]) -> dict[str, str]:
        return scrubbed(base, self.runner)

    def progress(self, line: str) -> str | None:
        """One short line for stderr about an output line, or None to stay quiet."""
        return None

    def read(self, lines: Sequence[str], returncode: int, stderr: str) -> RunResult:
        raise NotImplementedError


def parse_event(line: str) -> dict[str, Any] | None:
    """One JSON event per line, as both CLIs print them; None for anything else."""
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def clip(text: Any, limit: int = 100) -> str:
    one = " ".join(str(text or "").split())
    return one if len(one) <= limit else one[: limit - 3] + "..."
