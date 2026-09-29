"""The delegate adapters: argv per access level, the child's environment, and reading a run's events.

Every adapter is a pure function of its inputs; no CLI is ever started here.
"""

from __future__ import annotations

import json

import pytest

from jev_model_router.config import parse_config
from jev_model_router.delegates import ACCESS, Adapter, Target, adapter_for, parse_event, target_of

from test_capabilities import raw_config

TARGET = Target(tier="cx", runner="codex", model="gpt-6-sol", effort="high", executable="/bin/codex")


def jsonl(events):
    return [json.dumps(e) + "\n" for e in events]


def test_a_target_round_trips_through_a_dict():
    assert Target.from_dict(TARGET.as_dict()) == TARGET
    assert Target.from_dict({**TARGET.as_dict(), "effort": None}).effort is None


def test_an_adapter_is_available_when_its_executable_resolves():
    adapter = Adapter()
    assert adapter.available(TARGET, which=lambda exe: exe)
    assert not adapter.available(TARGET, which=lambda exe: None)


def test_parse_event_reads_one_json_object_per_line_and_nothing_else():
    assert parse_event('{"type": "x"}\n') == {"type": "x"}
    assert parse_event("progress text\n") is None
    assert parse_event("[1, 2]") is None
    assert parse_event('{"broken"') is None


def test_the_access_levels_are_the_three_the_spec_names():
    assert ACCESS == ("read-only", "workspace-write", "full")


def test_a_tier_an_adapter_runs_is_a_target_and_any_other_is_not():
    config = parse_config(raw_config())
    assert target_of("cx", config.tiers["cx"]) == Target("cx", "codex", "gpt-6-sol", None, "codex")
    assert target_of("sub", config.tiers["sub"]).executable == "claude"
    assert target_of("mid", config.tiers["mid"]) is None


CLAUDE = Target(tier="sub", runner="claude", model="claude-opus-5", effort="high", executable="/bin/claude")
RESULT = {"type": "result", "subtype": "success", "is_error": False, "result": "done: 3 files changed",
          "usage": {"input_tokens": 20, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 80,
                    "output_tokens": 40}}
STREAM = [
    {"type": "system", "subtype": "init"},
    {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Edit", "input": {}}]}},
    {"type": "assistant", "message": {"content": [{"type": "text", "text": "All tests pass."}]}},
    RESULT,
]


@pytest.mark.parametrize("access, flags", [
    ("read-only", ["--tools", "Read,Grep,Glob", "--permission-mode", "dontAsk"]),
    ("workspace-write", ["--permission-mode", "acceptEdits", "--allowedTools", "Bash", "PowerShell",
                         "--permission-prompts", "none"]),
    ("full", ["--dangerously-skip-permissions"]),
])
def test_claude_argv_per_access_level(access, flags):
    argv = adapter_for("claude").argv(CLAUDE, access, "/work")
    assert argv == ["/bin/claude", "-p", "--output-format", "stream-json", "--verbose",
                    "--model", "claude-opus-5", "--effort", "high", *flags]


def test_claude_argv_leaves_effort_out_when_the_tier_has_none():
    argv = adapter_for("claude").argv(Target(**{**CLAUDE.as_dict(), "effort": None}), "full", "/work")
    assert "--effort" not in argv


def test_claude_env_drops_what_would_bill_per_token_or_refuse_to_nest():
    env = adapter_for("claude").env({"ANTHROPIC_API_KEY": "k", "CLAUDECODE": "1", "PATH": "/bin"})
    assert env == {"PATH": "/bin"}


def test_claude_reads_the_final_result_and_its_usage():
    run = adapter_for("claude").read(jsonl(STREAM), 0, "")
    assert run.failure is None and run.message == "done: 3 files changed"
    assert (run.usage.prompt_tokens, run.usage.completion_tokens, run.usage.cached_tokens,
            run.usage.cache_write_tokens) == (1000, 40, 900, 80)


def test_claude_an_error_result_is_a_failure_and_a_spent_window_a_rate_limit():
    failed = adapter_for("claude").read(jsonl([{**RESULT, "is_error": True, "subtype": "error_during_execution",
                                                 "result": "tool crashed"}]), 1, "")
    limited = adapter_for("claude").read(jsonl([{**RESULT, "is_error": True, "result": "5-hour limit reached"}]), 1, "")
    status = adapter_for("claude").read(jsonl([{**RESULT, "is_error": True, "result": "x", "api_error_status": 429}]), 1, "")
    assert failed.failure == "tool crashed" and not failed.rate_limited
    assert limited.rate_limited and status.rate_limited


def test_claude_with_no_result_event_fails_with_its_stderr_and_no_usage():
    run = adapter_for("claude").read(["not json\n"], 1, "error: not logged in\n")
    assert run.failure == "error: not logged in" and run.usage.prompt_tokens == 0 and not run.rate_limited


def test_claude_progress_names_tools_and_text():
    adapter = adapter_for("claude")
    lines = [adapter.progress(line) for line in jsonl(STREAM)]
    assert lines == [None, "tool Edit", "All tests pass.", "done (success)"]


from test_codex_cli import EVENTS as CODEX_EVENTS


@pytest.mark.parametrize("access, flags", [
    ("read-only", ["--sandbox", "read-only"]),
    ("workspace-write", ["--sandbox", "workspace-write"]),
    ("full", ["--dangerously-bypass-approvals-and-sandbox"]),
])
def test_codex_argv_per_access_level(access, flags):
    argv = adapter_for("codex").argv(TARGET, access, "/work")
    assert argv == ["/bin/codex", "exec", "--json", "--skip-git-repo-check", "-C", "/work", "-m", "gpt-6-sol",
                    "-c", 'model_reasoning_effort="high"',
                    "-c", 'shell_environment_policy.set.JEV_MODEL_ROUTER_DEPTH="1"', *flags, "-"]


def test_codex_argv_leaves_effort_out_when_the_tier_has_none_and_keeps_the_depth():
    argv = adapter_for("codex").argv(Target(**{**TARGET.as_dict(), "effort": None}), "full", "/work")
    assert not any(a.startswith("model_reasoning_effort") for a in argv)
    assert 'shell_environment_policy.set.JEV_MODEL_ROUTER_DEPTH="1"' in argv


def test_codex_env_drops_the_api_keys():
    assert adapter_for("codex").env({"OPENAI_API_KEY": "o", "CODEX_API_KEY": "c", "PATH": "/bin"}) == {"PATH": "/bin"}


def test_codex_reads_the_last_message_and_the_turns_usage():
    run = adapter_for("codex").read(jsonl(CODEX_EVENTS), 0, "")
    assert run.failure is None and run.message == "final answer"
    assert (run.usage.prompt_tokens, run.usage.cached_tokens, run.usage.completion_tokens) == (1200, 1000, 80)


def test_codex_keeps_the_last_turns_usage_not_a_sum():
    two = [*CODEX_EVENTS, {"type": "turn.completed", "usage": {"input_tokens": 1500, "output_tokens": 90}}]
    assert adapter_for("codex").read(jsonl(two), 0, "").usage.prompt_tokens == 1500


def test_codex_a_failed_turn_fails_and_a_spent_window_is_a_rate_limit():
    failed = adapter_for("codex").read(jsonl([{"type": "turn.failed", "error": {"message": "sandbox denied"}}]), 1, "")
    limited = adapter_for("codex").read(jsonl([{"type": "error", "message": "You've hit your usage limit"}]), 1, "")
    silent = adapter_for("codex").read([], 2, "boom\n")
    no_answer = adapter_for("codex").read(jsonl([CODEX_EVENTS[-1]]), 0, "")
    assert failed.failure == "sandbox denied" and not failed.rate_limited
    assert limited.rate_limited
    assert silent.failure == "boom"
    assert no_answer.failure == "no agent message" and no_answer.usage.prompt_tokens == 1200


def test_codex_progress_names_commands_and_messages():
    adapter = adapter_for("codex")
    started = {"type": "item.started", "item": {"type": "command_execution", "command": "pytest -q"}}
    assert adapter.progress(json.dumps(started)) == "$ pytest -q"
    assert adapter.progress(json.dumps(CODEX_EVENTS[-2])) == "final answer"
    assert adapter.progress(json.dumps(CODEX_EVENTS[-1])) == "done"
    assert adapter.progress("warning: something") is None
