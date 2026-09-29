"""Backend that answers through the Codex CLI, on a ChatGPT subscription.

The twin of `claude_cli`, for the other subscription: `codex exec` is the
supported way to run it headless, so this backend shells out to it once per
request and reads the JSONL event stream it prints with `--json`.

The same four decisions, in Codex's own terms:

**The subscription, not a key.** `OPENAI_API_KEY` and `CODEX_API_KEY` are
removed from the child's environment. With either present the CLI can
authenticate with the key and bill per token, and nothing in the events would
say so.

**A clean room.** A fresh empty working directory (`-C`), a read-only sandbox,
no session written to disk (`--ephemeral`), and `--ignore-user-config` so the
operator's `~/.codex/config.toml` -- its default model, its MCP servers, its
effort -- cannot change what a tier asked for. Auth still comes from
`CODEX_HOME`, which is the point. Codex has no system-prompt flag, so a system
message leads the prompt as a labelled block.

**Tokens, not dollars.** `turn.completed` reports input, cached input and
output tokens and no price. `billed_usd` stays None and the tier estimates what
its `prices` say, which for a subscription tier is nothing. What the call costs
is quota, which the capabilities router weighs from list prices of its own.

**No output cap.** Like `claude -p`, `codex exec` has no max-tokens flag; the
effort level and `timeout_s` are the only bounds.

Event shapes, as the NucleOS runner reads the same CLI: the answer is the last
`item.completed` whose item is an `agent_message`; usage is on the last
`turn.completed`; a failure is an `error` or `turn.failed` event.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from typing import Any, AsyncIterator

from ..cli_runs import says_limited, scrubbed
from ..config import TierConfig
from ..schemas import ChatCompletionRequest, Usage, build_chunk, build_completion, new_completion_id
from .base import BackendError, BackendResponse, StreamChunk, StreamEnd, StreamEvent
from .claude_cli import Runner, _run, _split


class CodexCliBackend:
    """Runs one `codex exec --json` per request and translates its events."""

    def __init__(self, tier: TierConfig, runner: Runner | None = None) -> None:
        self.tier = tier
        self._runner = runner or _run
        self._cwd = tempfile.mkdtemp(prefix="jev-model-router-codex-")

    def _argv(self) -> list[str]:
        argv = [
            # `base_url` names the executable, as for claude_cli; default `codex`.
            self.tier.base_url,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "-s", "read-only",
            "-C", self._cwd,
            "-m", self.tier.model,
        ]
        if self.tier.effort:
            argv += ["-c", f"model_reasoning_effort={json.dumps(self.tier.effort)}"]
        # `-` reads the prompt from stdin: a prompt passed as an argument is
        # bounded by the Windows command line, and a long conversation is not.
        argv.append("-")
        return argv

    def _env(self) -> dict[str, str]:
        return scrubbed(os.environ, "codex")

    async def complete(self, request: ChatCompletionRequest) -> BackendResponse:
        system, prompt = _split(request)
        if system:
            prompt = f"[instructions]\n{system}\n\n[task]\n{prompt}"
        os.makedirs(self._cwd, exist_ok=True)
        try:
            proc = await asyncio.to_thread(
                self._runner, self._argv(), prompt, self._env(), self._cwd, self.tier.timeout_s
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendError(f"{self.tier.name}: codex exec timed out after {self.tier.timeout_s:.0f}s", status=504) from exc
        except OSError as exc:
            raise BackendError(
                f"{self.tier.name}: cannot run {self.tier.base_url!r} ({exc}); set base_url to the codex executable",
                status=502,
            ) from exc

        answer, usage, failure = _read_events(proc.stdout)
        if failure is not None or proc.returncode != 0:
            detail = failure or (proc.stderr or proc.stdout or "")[-2000:]
            limited = says_limited(detail)
            raise BackendError(
                f"{self.tier.name}: codex exec failed (exit {proc.returncode})",
                status=429 if limited else 502,
                body=detail,
            )
        if answer is None:
            raise BackendError(
                f"{self.tier.name}: codex exec produced no agent message",
                status=502,
                body=(proc.stderr or proc.stdout)[-2000:],
            )
        return BackendResponse(
            body=build_completion(model=self.tier.name, content=answer, usage=usage),
            usage=usage,
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        # One chunk holding the whole answer, for the same reason as claude_cli.
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


def _read_events(stdout: str) -> tuple[str | None, Usage, str | None]:
    """The last agent message, the last turn's usage, and the first failure, if any."""
    answer: str | None = None
    failure: str | None = None
    fresh = cached = output = 0
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event: Any = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                answer = item["text"]
        elif kind == "turn.completed":
            usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
            fresh = int(usage.get("input_tokens", 0) or 0)
            cached = int(usage.get("cached_input_tokens", 0) or 0)
            output = int(usage.get("output_tokens", 0) or 0)
        elif kind in ("error", "turn.failed") and failure is None:
            error = event.get("error")
            message = event.get("message") or (error.get("message") if isinstance(error, dict) else error)
            failure = str(message or kind)
    # OpenAI's convention: input_tokens already include the cached ones, so the
    # cached count is a part of the prompt, not added to it.
    usage = Usage(
        prompt_tokens=fresh,
        completion_tokens=output,
        total_tokens=fresh + output,
        cached_tokens=cached,
    )
    return answer, usage, failure
