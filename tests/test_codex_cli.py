"""The Codex CLI backend: the subscription, a clean room, and the event stream.

No process is ever started: every call goes through an injected runner that
prints the JSONL `codex exec --json` would.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Any

import pytest

from llm_router.backends import build_backend
from llm_router.backends.base import BackendError, StreamChunk, StreamEnd
from llm_router.backends.codex_cli import CodexCliBackend
from llm_router.config import ConfigError, parse_config
from llm_router.schemas import ChatCompletionRequest

from conftest import BASE_CONFIG

EVENTS = [
    {"type": "thread.started", "thread_id": "t-1"},
    {"type": "turn.started"},
    {"type": "item.completed", "item": {"type": "reasoning", "text": "thinking"}},
    {"type": "item.completed", "item": {"type": "agent_message", "text": "draft"}},
    {"type": "item.completed", "item": {"type": "agent_message", "text": "final answer"}},
    {"type": "turn.completed", "usage": {"input_tokens": 1200, "cached_input_tokens": 1000, "output_tokens": 80}},
]


def jsonl(events: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"


def tier(**overrides: Any):
    tiers = {**BASE_CONFIG["tiers"], "cx": {"backend": "codex_cli", "model": "gpt-6-sol", **overrides}}
    return parse_config({**BASE_CONFIG, "tiers": tiers}).tiers["cx"]


class Recorder:
    def __init__(self, stdout: str = jsonl(EVENTS), returncode: int = 0, raises: Exception | None = None) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def __call__(self, argv, stdin, env, cwd, timeout):
        self.calls.append({"argv": argv, "stdin": stdin, "env": env, "cwd": cwd})
        if self.raises:
            raise self.raises
        return subprocess.CompletedProcess(argv, self.returncode, stdout=self.stdout, stderr="")


def request(*messages: tuple[str, str]) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="cx", messages=[{"role": r, "content": c} for r, c in messages])


def run(backend: CodexCliBackend, req: ChatCompletionRequest):
    return asyncio.run(backend.complete(req))


def test_the_config_accepts_the_backend_on_the_codex_subscription():
    t = tier()
    assert t.base_url == "codex"
    assert t.subscription == "codex"
    assert t.can_serve
    assert isinstance(build_backend(t), CodexCliBackend)


def test_effort_is_refused_on_a_backend_that_has_no_such_flag():
    tiers = {
        **BASE_CONFIG["tiers"],
        "x": {"backend": "openai_compatible", "model": "m", "base_url": "https://example.invalid/v1", "effort": "high"},
    }
    with pytest.raises(ConfigError, match="extra_body"):
        parse_config({**BASE_CONFIG, "tiers": tiers})


def test_the_last_agent_message_is_the_answer_and_the_usage_is_read():
    result = run(CodexCliBackend(tier(), runner=Recorder()), request(("user", "hi")))
    assert result.body["choices"][0]["message"]["content"] == "final answer"
    assert result.usage.prompt_tokens == 1200
    assert result.usage.cached_tokens == 1000
    assert result.usage.completion_tokens == 80
    # A subscription call is not a charge.
    assert result.usage.billed_usd is None


def test_the_cli_runs_in_a_clean_room_with_the_model_and_effort_asked_for():
    runner = Recorder()
    backend = CodexCliBackend(tier(base_url="C:/bin/codex.cmd", effort="xhigh"), runner=runner)
    run(backend, request(("user", "hi")))
    argv = runner.calls[0]["argv"]
    assert argv[:2] == ["C:/bin/codex.cmd", "exec"]
    for flag in ("--json", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check"):
        assert flag in argv
    assert argv[argv.index("-s") + 1] == "read-only"
    assert argv[argv.index("-m") + 1] == "gpt-6-sol"
    assert argv[argv.index("-C") + 1] == runner.calls[0]["cwd"]
    assert 'model_reasoning_effort="xhigh"' in argv
    # The prompt goes on stdin, never on a command line with a length limit.
    assert argv[-1] == "-"


def test_no_effort_leaves_the_models_default():
    runner = Recorder()
    run(CodexCliBackend(tier(), runner=runner), request(("user", "hi")))
    assert not any("model_reasoning_effort" in a for a in runner.calls[0]["argv"])


def test_an_api_key_in_the_environment_never_reaches_the_cli(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("CODEX_API_KEY", "sk-nor-this")
    runner = Recorder()
    run(CodexCliBackend(tier(), runner=runner), request(("user", "hi")))
    env = runner.calls[0]["env"]
    assert "OPENAI_API_KEY" not in env and "CODEX_API_KEY" not in env


def test_a_system_message_leads_the_prompt_as_a_labelled_block():
    runner = Recorder()
    run(CodexCliBackend(tier(), runner=runner), request(("system", "Be terse."), ("user", "Why?")))
    assert runner.calls[0]["stdin"] == "[instructions]\nBe terse.\n\n[task]\nWhy?"


def test_a_spent_window_is_a_429_and_other_failures_a_502():
    limited = Recorder(jsonl([{"type": "error", "message": "You have hit your usage limit."}]), returncode=1)
    with pytest.raises(BackendError) as exc:
        run(CodexCliBackend(tier(), runner=limited), request(("user", "hi")))
    assert exc.value.status == 429

    broken = Recorder(jsonl([{"type": "turn.failed", "error": {"message": "model not found"}}]), returncode=1)
    with pytest.raises(BackendError) as exc:
        run(CodexCliBackend(tier(), runner=broken), request(("user", "hi")))
    assert exc.value.status == 502
    assert "model not found" in exc.value.body


def test_a_run_with_no_agent_message_is_an_error_not_an_empty_answer():
    silent = Recorder(jsonl([{"type": "turn.completed", "usage": {"input_tokens": 5}}]))
    with pytest.raises(BackendError, match="no agent message"):
        run(CodexCliBackend(tier(), runner=silent), request(("user", "hi")))


def test_a_missing_executable_and_a_timeout_say_so():
    with pytest.raises(BackendError, match="cannot run") as exc:
        run(CodexCliBackend(tier(), runner=Recorder(raises=FileNotFoundError("codex"))), request(("user", "hi")))
    assert exc.value.status == 502
    with pytest.raises(BackendError, match="timed out") as exc:
        run(CodexCliBackend(tier(), runner=Recorder(raises=subprocess.TimeoutExpired("codex", 1))), request(("user", "hi")))
    assert exc.value.status == 504


def test_a_stream_is_one_chunk_then_the_usage():
    async def collect():
        backend = CodexCliBackend(tier(), runner=Recorder())
        return [e async for e in backend.stream(request(("user", "hi")))]

    events = asyncio.run(collect())
    assert isinstance(events[0], StreamChunk)
    assert events[0].data["choices"][0]["delta"]["content"] == "final answer"
    assert isinstance(events[-1], StreamEnd) and events[-1].usage.completion_tokens == 80
