"""Backend that answers through the Claude Code CLI, on a Claude subscription.

Every other paid tier here reaches Claude through an API key, which bills per
token. An operator who already pays for a Claude subscription is paying twice
for the same model, and for a verifier -- a second call on every checked
request -- that is the larger half of the bill. `claude -p` is the supported
way to run that subscription headless, so this backend shells out to it.

Four decisions, each one a way this could quietly be something else:

**The subscription, not a key.** `ANTHROPIC_API_KEY` is removed from the
child's environment. With it present the CLI authenticates with the key and
bills per token -- exactly what this backend exists to avoid -- and nothing in
the response would say so. For the same reason `--bare` is never passed: it
restricts auth to an API key and ignores the subscription login.

**A clean room.** No tools (`--tools ""`), no settings files
(`--setting-sources ""`, which also keeps user hooks out), no MCP servers, no
session written to disk, and a working directory with no CLAUDE.md in it. The
CLI's own system prompt is always replaced: by the caller's system message, or
by a plain assistant prompt when there is none. A judge that
inherits a repository's instructions is judging by rules nobody configured.

**Tokens, not dollars.** The CLI reports `total_cost_usd`, but that is list
price, not a charge: a subscription bills nothing per call. `billed_usd` stays
None ("not reported"), and a tier with no prices estimates zero. What the
calls DO cost is subscription quota, which no field here can measure -- a rate
limit arrives as a 429 so `on_verifier_error` decides what it means.

**No output cap.** The CLI has no max-tokens flag, so `max_tokens` and
`verification.max_verdict_tokens` do not bind on this tier. A direct judge
stops by itself; one that thinks at length is bounded only by `timeout_s`.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from typing import Any, AsyncIterator, Callable

from ..config import TierConfig
from ..schemas import (
    ChatCompletionRequest,
    Usage,
    build_chunk,
    build_completion,
    new_completion_id,
)
from .base import BackendError, BackendResponse, StreamChunk, StreamEnd, StreamEvent

# argv, stdin text, env, cwd, timeout -> the finished process. Injected by tests.
Runner = Callable[[list[str], str, dict[str, str], str, float], subprocess.CompletedProcess]

# What a plain chat completion endpoint would be, for a request with no system message.
_DEFAULT_SYSTEM = "You are a helpful assistant."


def _run(argv: list[str], stdin: str, env: dict[str, str], cwd: str, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=cwd,
        timeout=timeout,
    )


class ClaudeCliBackend:
    """Runs one `claude -p` per request and translates its JSON result."""

    def __init__(self, tier: TierConfig, runner: Runner | None = None) -> None:
        self.tier = tier
        self._runner = runner or _run
        # A fresh empty directory: the CLI discovers CLAUDE.md files by walking
        # up from its cwd, and the proxy's own cwd is usually a repository.
        self._cwd = tempfile.mkdtemp(prefix="llm-router-claude-")

    def _argv(self, system: str | None) -> list[str]:
        argv = [
            # For this backend `base_url` names the executable, since there is
            # no URL; the default is `claude` on PATH.
            self.tier.base_url,
            "-p",
            "--output-format", "json",
            "--model", self.tier.model,
            "--tools", "",
            "--setting-sources", "",
            "--strict-mcp-config",
            "--no-session-persistence",
            # Always replaced, even when the request carries no system message.
            # The CLI's default prompt is an agent's: it describes Write, Bash
            # and the rest, which `--tools ""` removed, so the model answers
            # with pretend tool calls and no code. Measured: 16 of 40 failed
            # reviews in a 228-prompt run were exactly that.
            "--system-prompt", system or _DEFAULT_SYSTEM,
        ]
        if self.tier.effort:
            argv += ["--effort", self.tier.effort]
        return argv

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.pop("ANTHROPIC_API_KEY", None)
        return env

    async def complete(self, request: ChatCompletionRequest) -> BackendResponse:
        system, prompt = _split(request)
        # Made again on every call: it lives in the temp directory, and a temp
        # cleaner removed it under a proxy that had run all night -- every
        # call after that failed with "directory name is invalid".
        os.makedirs(self._cwd, exist_ok=True)
        try:
            proc = await asyncio.to_thread(
                self._runner, self._argv(system), prompt, self._env(), self._cwd, self.tier.timeout_s
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendError(f"{self.tier.name}: claude -p timed out after {self.tier.timeout_s:.0f}s", status=504) from exc
        except OSError as exc:
            raise BackendError(
                f"{self.tier.name}: cannot run {self.tier.base_url!r} ({exc}); set base_url to the claude executable",
                status=502,
            ) from exc

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = None
        if not isinstance(payload, dict):
            raise BackendError(
                f"{self.tier.name}: claude -p exited {proc.returncode} with no JSON result",
                status=502,
                body=(proc.stderr or proc.stdout)[-2000:],
            )
        if payload.get("is_error") or proc.returncode != 0:
            upstream = payload.get("api_error_status")
            raise BackendError(
                f"{self.tier.name}: claude -p failed ({payload.get('subtype')})",
                status=upstream if isinstance(upstream, int) and upstream >= 400 else 502,
                body=payload.get("result"),
            )

        content = payload.get("result")
        if not isinstance(content, str):
            raise BackendError(f"{self.tier.name}: result carried no text", status=502, body=payload)
        usage = _usage_of(payload.get("usage"))
        return BackendResponse(
            body=build_completion(model=self.tier.name, content=content, usage=usage),
            usage=usage,
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        # One chunk holding the whole answer. `--output-format json` returns the
        # result at the end, and a stream that pretends to be incremental would
        # only move the wait from before the first byte to after it.
        response = await self.complete(request)
        completion_id = new_completion_id()
        content = response.body["choices"][0]["message"]["content"]
        yield StreamChunk(
            data=build_chunk(
                model=self.tier.name,
                completion_id=completion_id,
                delta={"role": "assistant", "content": content},
            )
        )
        yield StreamChunk(
            data=build_chunk(model=self.tier.name, completion_id=completion_id, delta={}, finish_reason="stop")
        )
        yield StreamEnd(usage=response.usage)

    async def aclose(self) -> None:
        return None


def _split(request: ChatCompletionRequest) -> tuple[str | None, str]:
    """System messages become the system prompt; the rest becomes stdin.

    A single user message goes through as it is. A longer conversation is
    rendered as a labelled transcript, since `-p` takes one prompt and not a
    list of turns.
    """
    system: list[str] = []
    turns: list[tuple[str, str]] = []
    for message in request.messages:
        text = _text_of(message.content)
        if message.role == "system":
            if text:
                system.append(text)
        elif text:
            turns.append((message.role, text))
    if len(turns) == 1 and turns[0][0] == "user":
        prompt = turns[0][1]
    else:
        prompt = "\n\n".join(f"[{role}]\n{text}" for role, text in turns)
    return ("\n\n".join(system) or None), prompt


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _usage_of(raw: Any) -> Usage:
    """The CLI speaks the Anthropic spelling: cache reads and writes apart.

    Only the requested model's turn is counted. `modelUsage` also lists side
    calls the CLI makes on its own account (a small model, for housekeeping);
    those are the tool's overhead, not this request's answer.
    """
    raw = raw if isinstance(raw, dict) else {}
    fresh = int(raw.get("input_tokens", 0) or 0)
    read = int(raw.get("cache_read_input_tokens", 0) or 0)
    write = int(raw.get("cache_creation_input_tokens", 0) or 0)
    completion = int(raw.get("output_tokens", 0) or 0)
    prompt = fresh + read + write
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        cached_tokens=read,
        cache_write_tokens=write,
    )
