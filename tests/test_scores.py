"""Benchmark evidence: measured points per (model, effort) become card levels.

Every model id, benchmark and source here is invented: the code must work for
models that do not exist yet, so the tests never lean on one that does.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest

from llm_router.config import ConfigError, parse_config

from conftest import BASE_CONFIG

REQUIREMENTS = {
    "reasoning": "The state is a task packet. Does it need deep reasoning?",
    "niche": "The state is a task packet. Does it need niche knowledge?",
}


def raw_config(**caps: Any) -> dict[str, Any]:
    raw = copy.deepcopy(BASE_CONFIG)
    raw["tiers"]["judge"] = {"backend": "jev", "model": "jev-latest"}
    raw["router"] = {
        "kind": "capabilities",
        "default_tier": "mid",
        "capabilities": {
            "jev_tier": "judge",
            "requirements": REQUIREMENTS,
            "discover": {"cli": {"backend": "claude_cli", "base_url": "C:/bin/cli.exe"}},
            "profiles": [
                {"match": "acme-large", "level": 2.0, "list_prices": {"input": 5.0, "output": 25.0}},
                {"match": "acme-small", "level": 1.0, "list_prices": {"input": 1.0, "output": 5.0}},
                {"match": "zeta-*", "level": 1.5, "list_prices": {"input": 2.0, "output": 10.0}},
            ],
            "fallback_profile": {"level": 0.5, "list_prices": {"input": 10.0, "output": 50.0}},
            "benchmarks": {"path": "benchmarks.yaml"},
            **caps,
        },
    }
    return raw


def logit(s: float) -> float:
    return math.log(s / (1 - s))


# ---------------------------------------------------------------- config
def test_the_benchmarks_block_is_optional_and_has_defaults():
    b = parse_config(raw_config()).router.capabilities.benchmarks
    assert (b.path, b.a, b.k, b.profile_weight, b.vendor_weight, b.refresh_hours, b.refresh_timeout_s) == (
        "benchmarks.yaml", None, None, 0.25, 0.5, 24.0, 30.0)
    raw = raw_config()
    del raw["router"]["capabilities"]["benchmarks"]
    assert parse_config(raw).router.capabilities.benchmarks is None


@pytest.mark.parametrize(
    "block, message",
    [
        ({"a": 1.0}, "path is required"),
        ({"path": "b.yaml", "a": 3.5}, "a must be in"),
        ({"path": "b.yaml", "k": 0.0}, "k must be in"),
        ({"path": "b.yaml", "profile_weight": 0}, "profile_weight must be positive"),
        ({"path": "b.yaml", "vendor_weight": 1.5}, "vendor_weight must be in"),
        ({"path": "b.yaml", "refresh_hours": -1}, "refresh_hours must be positive"),
        ({"path": "b.yaml", "a": "high"}, "could not convert"),
        ({"path": "b.yaml", "lambda": 1}, "unknown fields"),
    ],
    ids=["no-path", "a", "k", "profile-weight", "vendor-weight", "refresh", "not-a-number", "unknown"],
)
def test_a_benchmarks_block_that_would_mislead_is_refused(block, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(raw_config(benchmarks=block))
