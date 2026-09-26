"""Route only: which model and effort should run a task, for a caller that runs it itself.

The proxy answers chat requests by calling a model. An agent runner cannot use
that: its work is many turns of tools in a worktree, and the CLI that does it
must be spawned by the runner. So the runner asks here instead -- one task in,
one `{model, effort, runner}` out, and a `decision_id` it later reports the
outcome against. The decision is the same one the proxy would make: the same
eligibility, the same router, one Jev call.

The outcome closes the loop. `pass`/`fail` is what the caller's own gate said,
and is the label `calibrate --from-log` learns from; `rate_limited` takes that
subscription off the table until its window turns, which the router could not
see for itself because it never ran the call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Config, TierConfig
from .schemas import ChatCompletionRequest

# What a caller spawns for each backend. The names are the CLIs' own.
RUNNERS = {"claude_cli": "claude", "codex_cli": "codex", "ollama": "ollama"}
OUTCOMES = ("pass", "fail", "rate_limited", "error")
# The gate's tail is what says why the last attempt failed; its head is noise.
GATE_OUTPUT_CHARS = 1500
_KNOWN = {"task", "stage", "files", "attempt", "gate_output", "failed", "runners", "packet", "models", "exclude"}


def runner_of(tier: TierConfig) -> str:
    return RUNNERS.get(tier.backend, tier.backend)


@dataclass(frozen=True)
class RouteAsk:
    request: ChatCompletionRequest
    runners: frozenset[str]  # empty: the caller can run anything
    stage: str | None
    failed: tuple[str, ...]  # tier names
    unknown_failed: tuple[str, ...]  # named as failed, matched no tier

    def can_run(self, tier: TierConfig) -> bool:
        return not self.runners or runner_of(tier) in self.runners


def parse_route_ask(payload: Any, config: Config) -> RouteAsk:
    """The request body, checked. Raises ValueError with a message fit for a 400."""
    if not isinstance(payload, dict):
        raise ValueError("the body must be a JSON object")
    extra = set(payload) - _KNOWN
    if extra:
        raise ValueError(f"unknown fields {sorted(extra)}; free-form context goes in 'packet'")
    task = payload.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("'task' is required: the description of the work to route")
    packet = payload.get("packet") or {}
    if not isinstance(packet, dict):
        raise ValueError("'packet' must be an object")
    runners = payload.get("runners") or []
    if not isinstance(runners, list) or not all(isinstance(r, str) for r in runners):
        raise ValueError("'runners' must be a list of runner names, e.g. [\"claude\", \"codex\"]")
    failed_raw = payload.get("failed") or []
    if not isinstance(failed_raw, list) or not all(isinstance(f, str) for f in failed_raw):
        raise ValueError("'failed' must be a list of tier names or model[@effort]")
    failed, unknown = _match_failed(failed_raw, config)
    shortlist: dict[str, list[str]] = {}
    for key in ("models", "exclude"):
        globs = payload.get(key)
        if globs is None:
            continue
        if not isinstance(globs, list) or not all(isinstance(g, str) and g for g in globs):
            raise ValueError(f"'{key}' must be a list of globs over tier names and model ids, e.g. [\"vendor/*\"]")
        shortlist[key] = globs

    context: dict[str, Any] = dict(packet)
    for key in ("stage", "files", "attempt"):
        if payload.get(key) is not None:
            context[key] = payload[key]
    gate = payload.get("gate_output")
    if isinstance(gate, str) and gate.strip():
        context["previous_attempt_failed_with"] = gate.strip()[-GATE_OUTPUT_CHARS:]
    if failed:
        context["failed_tiers"] = list(failed)
    request = ChatCompletionRequest.model_validate({
        "model": "auto",
        "messages": [{"role": "user", "content": task}],
        "packet": context,
        **shortlist,
    })
    stage = payload.get("stage")
    return RouteAsk(
        request=request,
        runners=frozenset(runners),
        stage=str(stage) if stage is not None else None,
        failed=failed,
        unknown_failed=unknown,
    )


def _match_failed(names: list[str], config: Config) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Tier names pass through; `model@effort` or a bare `model` is matched to the tiers serving it.

    A runner knows the model and effort it spawned, not this router's tier names.
    """
    matched: list[str] = []
    unknown: list[str] = []
    for name in names:
        if name in config.tiers:
            matched.append(name)
            continue
        model, _, effort = name.partition("@")
        hits = [
            t for t, tier in config.tiers.items()
            if tier.model == model and (not effort or tier.effort == effort)
        ]
        (matched.extend(hits) if hits else unknown.append(name))
    return tuple(dict.fromkeys(matched)), tuple(unknown)


__all__ = ["OUTCOMES", "RUNNERS", "RouteAsk", "parse_route_ask", "runner_of"]
