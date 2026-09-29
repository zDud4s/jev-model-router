"""The delegate adapters: argv per access level, the child's environment, and reading a run's events.

Every adapter is a pure function of its inputs; no CLI is ever started here.
"""

from __future__ import annotations

import json

import pytest

from jev_model_router.delegates import ACCESS, Adapter, Target, parse_event

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
