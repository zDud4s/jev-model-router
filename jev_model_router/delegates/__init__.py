"""Adapters that hand a routed task to another agent's CLI: one module per runner, registered here.

A third agent CLI is one module and one line below; nothing outside an adapter names a CLI's flags.
"""

from __future__ import annotations

from ..config import TierConfig
from ..route_api import runner_of
from .base import ACCESS, DEPTH_ENV, Adapter, RunResult, Target, clip, parse_event
from .claude import ClaudeAdapter
from .codex import CodexAdapter

ADAPTERS: dict[str, Adapter] = {a.runner: a for a in (ClaudeAdapter(), CodexAdapter())}


def adapter_for(runner: str) -> Adapter | None:
    return ADAPTERS.get(runner)


def target_of(name: str, tier: TierConfig) -> Target | None:
    """The tier as a delegate would run it, or None when no adapter runs its runner.

    For a CLI tier `base_url` names the executable, as it does for the proxy's backends.
    """
    runner = runner_of(tier)
    if runner not in ADAPTERS:
        return None
    return Target(tier=name, runner=runner, model=tier.model, effort=tier.effort, executable=tier.base_url)


__all__ = [
    "ACCESS", "ADAPTERS", "DEPTH_ENV", "Adapter", "RunResult", "Target", "adapter_for", "clip", "parse_event",
    "target_of",
]
