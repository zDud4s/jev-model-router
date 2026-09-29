"""Adapters that hand a routed task to another agent's CLI: one module per runner, registered here.

A third agent CLI is one module and one line below; nothing outside an adapter names a CLI's flags.
"""

from __future__ import annotations

from .base import ACCESS, DEPTH_ENV, Adapter, RunResult, Target, clip, parse_event
from .claude import ClaudeAdapter
from .codex import CodexAdapter

ADAPTERS: dict[str, Adapter] = {a.runner: a for a in (ClaudeAdapter(), CodexAdapter())}


def adapter_for(runner: str) -> Adapter | None:
    return ADAPTERS.get(runner)


__all__ = ["ACCESS", "ADAPTERS", "DEPTH_ENV", "Adapter", "RunResult", "Target", "adapter_for", "clip", "parse_event"]
