"""What the subscription CLIs share: what they must not see, and how they say a window is spent."""

from __future__ import annotations

from jev_model_router.cli_runs import DROP_ENV, says_limited, scrubbed


def test_each_runner_loses_the_variables_that_would_bill_per_token_or_stop_it():
    base = {"ANTHROPIC_API_KEY": "k", "ANTHROPIC_AUTH_TOKEN": "t", "CLAUDECODE": "1",
            "OPENAI_API_KEY": "o", "CODEX_API_KEY": "c", "KEEP": "1"}
    claude, codex = scrubbed(base, "claude"), scrubbed(base, "codex")
    assert set(claude) == {"OPENAI_API_KEY", "CODEX_API_KEY", "KEEP"}
    assert set(codex) == {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE", "KEEP"}
    assert scrubbed(base, "unknown") == base and base["CLAUDECODE"] == "1"  # a copy, untouched
    assert set(DROP_ENV) == {"claude", "codex"}


def test_a_spent_window_is_recognised_in_either_cli_s_words():
    assert says_limited("You've hit your usage limit. Try again later.")
    assert says_limited("5-hour limit reached - resets 3pm")
    assert says_limited("429 Too Many Requests")
    assert says_limited("Weekly limit reached")
    assert not says_limited("tests failed: 3 assertions")
    assert not says_limited("context length limit reached")
    assert not says_limited("max turn limit reached")
    assert not says_limited(None)
