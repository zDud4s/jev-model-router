"""The Claude Code CLI backend: the subscription, a clean room, and honest errors.

The property this suite exists for is the first one. A tier on this backend is
chosen BECAUSE it bills nothing per call, and the one way to lose that without
any response saying so is an `ANTHROPIC_API_KEY` in the environment -- the CLI
then authenticates with the key and bills per token.
`test_an_api_key_in_the_environment_never_reaches_the_cli` keeps that shut.

No process is ever started: every call goes through an injected runner.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Any

import pytest

from llm_router.backends import build_backend
from llm_router.backends.base import BackendError, StreamChunk, StreamEnd
from llm_router.backends.claude_cli import ClaudeCliBackend
from llm_router.config import ConfigError, parse_config
from llm_router.schemas import ChatCompletionRequest

from conftest import BASE_CONFIG

CLI_TIER: dict[str, Any] = {"backend": "claude_cli", "model": "sonnet"}

OK_RESULT = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "VERDICT: PASS",
    # List price, not a charge -- and it must not become one.
    "total_cost_usd": 0.0044,
    "usage": {
        "input_tokens": 2,
        "cache_creation_input_tokens": 773,
        "cache_read_input_tokens": 100,
        "output_tokens": 35,
    },
}


def tier(**overrides: Any):
    tiers = {**BASE_CONFIG["tiers"], "sub": {**CLI_TIER, **overrides}}
    return parse_config({**BASE_CONFIG, "tiers": tiers}).tiers["sub"]


class Recorder:
    """A runner that records what it was asked and answers from a script."""

    def __init__(self, payload: Any = OK_RESULT, returncode: int = 0, raises: Exception | None = None) -> None:
        self.payload = payload
        self.returncode = returncode
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def __call__(self, argv, stdin, env, cwd, timeout):
        self.calls.append({"argv": argv, "stdin": stdin, "env": env, "cwd": cwd, "timeout": timeout})
        if self.raises:
            raise self.raises
        stdout = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return subprocess.CompletedProcess(argv, self.returncode, stdout=stdout, stderr="boom")


def request(*messages: tuple[str, str]) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="sub", messages=[{"role": r, "content": c} for r, c in messages]
    )


def run(backend: ClaudeCliBackend, req: ChatCompletionRequest):
    return asyncio.run(backend.complete(req))


def test_the_config_accepts_the_backend_and_defaults_the_executable_to_claude():
    t = tier()
    assert t.backend == "claude_cli"
    assert t.base_url == "claude"
    assert t.can_serve
    assert isinstance(build_backend(t), ClaudeCliBackend)


def test_an_unknown_backend_is_still_refused():
    with pytest.raises(ConfigError, match="claude_cli"):
        tier(backend="claude_api")


def test_the_system_message_replaces_the_cli_prompt_and_the_user_message_is_stdin():
    runner = Recorder()
    backend = ClaudeCliBackend(tier(base_url="C:/bin/claude.exe"), runner=runner)
    run(backend, request(("system", "You are a reviewer."), ("user", "Is 2+2=5?")))

    call = runner.calls[0]
    argv = call["argv"]
    assert argv[0] == "C:/bin/claude.exe"
    assert argv[argv.index("--system-prompt") + 1] == "You are a reviewer."
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert call["stdin"] == "Is 2+2=5?"


def test_a_request_with_no_system_message_still_replaces_the_agent_prompt():
    runner = Recorder()
    run(ClaudeCliBackend(tier(), runner=runner), request(("user", "Write a test.")))

    argv = runner.calls[0]["argv"]
    # Without this the CLI's own agent prompt describes tools that `--tools ""`
    # removed, and the model answers with pretend tool calls instead of code.
    assert argv[argv.index("--system-prompt") + 1] == "You are a helpful assistant."


def test_the_cli_runs_in_a_clean_room():
    runner = Recorder()
    backend = ClaudeCliBackend(tier(), runner=runner)
    run(backend, request(("user", "hi")))

    argv = runner.calls[0]["argv"]
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in argv
    assert "--no-session-persistence" in argv
    # `--bare` would drop the subscription login and demand an API key.
    assert "--bare" not in argv


def test_a_working_directory_deleted_under_a_running_proxy_is_made_again():
    import os
    import shutil

    seen: list[bool] = []

    def runner(argv, stdin, env, cwd, timeout):
        seen.append(os.path.isdir(cwd))
        return Recorder()(argv, stdin, env, cwd, timeout)

    backend = ClaudeCliBackend(tier(), runner=runner)
    # A temp cleaner took it overnight: every later call failed with WinError 267.
    shutil.rmtree(backend._cwd)
    run(backend, request(("user", "hi")))
    assert seen == [True]


def test_an_api_key_in_the_environment_never_reaches_the_cli(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-pass")
    monkeypatch.setenv("KEEP_ME", "1")
    runner = Recorder()
    run(ClaudeCliBackend(tier(), runner=runner), request(("user", "hi")))

    env = runner.calls[0]["env"]
    assert "ANTHROPIC_API_KEY" not in env
    assert env["KEEP_ME"] == "1"


def test_a_conversation_becomes_a_labelled_transcript():
    runner = Recorder()
    run(
        ClaudeCliBackend(tier(), runner=runner),
        request(("user", "first"), ("assistant", "reply"), ("user", "second")),
    )
    assert runner.calls[0]["stdin"] == "[user]\nfirst\n\n[assistant]\nreply\n\n[user]\nsecond"


def test_the_answer_and_tokens_come_back_and_list_price_is_not_a_bill():
    response = run(ClaudeCliBackend(tier(), runner=Recorder()), request(("user", "hi")))

    assert response.body["choices"][0]["message"]["content"] == "VERDICT: PASS"
    assert response.body["model"] == "sub"
    u = response.usage
    assert (u.prompt_tokens, u.completion_tokens) == (875, 35)
    assert (u.cached_tokens, u.cache_write_tokens) == (100, 773)
    assert u.billed_usd is None


def test_an_error_result_is_an_error_and_keeps_the_upstream_status():
    payload = {**OK_RESULT, "is_error": True, "subtype": "error", "api_error_status": 429, "result": "rate limited"}
    with pytest.raises(BackendError) as info:
        run(ClaudeCliBackend(tier(), runner=Recorder(payload=payload, returncode=1)), request(("user", "hi")))
    assert info.value.status == 429


@pytest.mark.parametrize(
    "runner, status",
    [
        (Recorder(payload="not json", returncode=1), 502),
        (Recorder(raises=subprocess.TimeoutExpired("claude", 5)), 504),
        (Recorder(raises=FileNotFoundError("claude")), 502),
        (Recorder(payload={**OK_RESULT, "result": None}), 502),
    ],
    ids=["no-json", "timeout", "missing-executable", "no-text"],
)
def test_every_other_failure_is_a_backend_error_never_an_empty_answer(runner, status):
    with pytest.raises(BackendError) as info:
        run(ClaudeCliBackend(tier(), runner=runner), request(("user", "hi")))
    assert info.value.status == status


def test_a_stream_is_one_chunk_then_the_end():
    backend = ClaudeCliBackend(tier(), runner=Recorder())

    async def collect():
        return [event async for event in backend.stream(request(("user", "hi")))]

    events = asyncio.run(collect())
    assert isinstance(events[-1], StreamEnd)
    chunks = [e for e in events if isinstance(e, StreamChunk)]
    assert chunks[0].data["choices"][0]["delta"]["content"] == "VERDICT: PASS"
    assert chunks[-1].data["choices"][0]["finish_reason"] == "stop"
    assert events[-1].usage.completion_tokens == 35


def test_the_tier_effort_reaches_the_cli_and_no_effort_leaves_its_default():
    runner = Recorder()
    run(ClaudeCliBackend(tier(effort="high"), runner=runner), request(("user", "hi")))
    argv = runner.calls[0]["argv"]
    assert argv[argv.index("--effort") + 1] == "high"

    runner = Recorder()
    run(ClaudeCliBackend(tier(), runner=runner), request(("user", "hi")))
    assert "--effort" not in runner.calls[0]["argv"]


def test_a_subscription_tier_names_its_pool_and_an_unknown_effort_is_refused():
    assert tier().subscription == "claude"
    assert tier(subscription="work").subscription == "work"
    with pytest.raises(ConfigError, match="effort must be one of"):
        tier(effort="extreme")
