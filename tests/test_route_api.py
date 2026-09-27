"""Route only: a runner asks which model and effort, runs it itself, and reports the outcome.

Jev is never called: the router answers from a scripted `Ask`.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from jev_model_router.calibration import log_outcomes
from jev_model_router.capabilities import CapabilityRouter
from jev_model_router.config import parse_config
from jev_model_router.db import RequestLog
from jev_model_router.schemas import Usage

from test_capabilities import HARD, Ask, raw_config

TASK = {"task": "Fix the race in the scheduler", "stage": "implement", "files": ["core/src/scheduler.rs"]}


def client_for(backend_factory, ask=None, log=None, **caps):
    from jev_model_router.app import create_app

    config = parse_config(raw_config(**caps))
    router = CapabilityRouter(config, ask=ask or Ask(HARD))
    log = log or RequestLog(":memory:")
    app = create_app(config, backend_factory=backend_factory, log=log, router=router)
    return TestClient(app), log, router


def test_a_task_gets_a_model_an_effort_and_a_runner_and_nothing_runs(backend_factory, fake_backends):
    ask = Ask(HARD)
    client, log, _ = client_for(backend_factory, ask)
    with client:
        body = client.post("/v1/route", json=TASK).json()
        row = log.decision(body["decision_id"])
        requests = log.query("SELECT COUNT(*) AS n FROM requests")[0]["n"]
    assert body["tier"] == "sub" and body["runner"] == "claude"
    assert (body["model"], body["effort"]) == ("claude-opus-5", "high")
    assert body["decision_id"].startswith("rt_") and body["success"] > 0.8
    assert all(not b.calls for b in fake_backends.values())
    # What Jev was shown carries the runner's context.
    packet = ask.calls[0][0]
    assert "Fix the race" in packet and "stage: implement" in packet and "scheduler.rs" in packet
    assert row["tier"] == "sub" and row["stage"] == "implement" and row["outcome"] is None
    # Decisions are not requests: no row of zeros among real calls.
    assert requests == 0


def test_a_caller_that_runs_only_codex_is_offered_only_codex(backend_factory):
    client, _, _ = client_for(backend_factory)
    with client:
        body = client.post("/v1/route", json={**TASK, "runners": ["codex"]}).json()
        assert body["tier"] == "cx" and body["runner"] == "codex"
        none = client.post("/v1/route", json={**TASK, "runners": ["gemini"]})
    assert none.status_code == 422 and "gemini" in none.json()["error"]["message"]


def test_a_failed_model_is_named_as_the_runner_knows_it(backend_factory):
    ask = Ask(HARD)
    client, _, _ = client_for(backend_factory, ask)
    with client:
        body = client.post("/v1/route", json={
            **TASK, "attempt": 2, "failed": ["claude-opus-5@high", "nope-model"],
            "gate_output": "x" * 5000 + "\nassertion failed: job ran twice",
        }).json()
    assert body["tier"] not in ("sub",) and body["unknown_failed"] == ["nope-model"]
    packet = ask.calls[0][0]
    # Jev reads why the last attempt failed -- the gate's tail, not all of it.
    # Which tiers failed is the router's business, not the task's.
    assert "job ran twice" in packet and "x" * 1500 not in packet and "failed_tiers" not in packet


def test_a_route_request_that_would_mislead_is_refused(backend_factory):
    client, _, _ = client_for(backend_factory)
    with client:
        assert client.post("/v1/route", json={"stage": "implement"}).status_code == 400
        assert client.post("/v1/route", json={**TASK, "prompt": "x"}).status_code == 400
        assert client.post("/v1/route", json={**TASK, "runners": "claude"}).status_code == 400


def test_the_outcome_is_kept_once_and_feeds_calibration(backend_factory):
    client, log, router = client_for(backend_factory)
    with client:
        decision = client.post("/v1/route", json=TASK).json()["decision_id"]
        url = f"/v1/route/{decision}/outcome"
        first = client.post(url, json={"status": "pass", "usage": {"prompt_tokens": 1000, "completion_tokens": 500}})
        assert first.status_code == 200 and first.json()["tier"] == "sub"
        assert client.post(url, json={"status": "fail"}).status_code == 409
        assert client.post("/v1/route/rt_nope/outcome", json={"status": "pass"}).status_code == 404
        assert client.post(url, json={"status": "great"}).status_code == 400
        assert log.decision(decision)["outcome"] == "pass"
        [outcome] = log_outcomes(log, parse_config(raw_config()))
    # Charged to the subscription for display, as any call would be.
    assert router.ledger.used("claude") > 0
    assert (outcome.tier, outcome.passed, outcome.source) == ("sub", True, "outcome")


def test_a_rate_limit_the_runner_saw_takes_that_subscription_off_the_table(backend_factory):
    client, _, router = client_for(backend_factory)
    with client:
        decision = client.post("/v1/route", json=TASK).json()["decision_id"]
        client.post(f"/v1/route/{decision}/outcome", json={"status": "rate_limited"})
        again = client.post("/v1/route", json=TASK).json()
    assert router.ledger.locked("claude") and again["runner"] != "claude"


SHAPE = {"input_per_output": 295, "cache_read": 0.971, "cache_write": 0.026}
CACHED = {"prompt_tokens": 300_000, "completion_tokens": 1000, "cached_tokens": 290_000, "cache_write_tokens": 8000}


def test_a_route_says_which_cost_basis_it_used(backend_factory):
    plain, _, _ = client_for(backend_factory)
    with plain:
        assert plain.post("/v1/route", json=TASK).json()["cost_basis"] == "one_call"
    shaped, _, _ = client_for(backend_factory, task_shape=SHAPE)
    with shaped:
        assert shaped.post("/v1/route", json=TASK).json()["cost_basis"] == "task_shape"


def test_outcomes_with_cache_tokens_become_the_tiers_observed_shape(backend_factory):
    client, log, router = client_for(backend_factory)
    with client:
        for _ in range(5):
            body = client.post("/v1/route", json=TASK).json()
            client.post(f"/v1/route/{body['decision_id']}/outcome", json={"status": "pass", "usage": CACHED})
        row = log.decision(body["decision_id"])
    assert (row["outcome_cached_tokens"], row["outcome_cache_write_tokens"]) == (290_000, 8000)
    shape = router.calls.shape_for(body["tier"])
    assert shape.source == "observed:5" and shape.input_per_output == pytest.approx(300.0)


def test_observed_shapes_are_read_back_from_the_log_at_startup(backend_factory):
    log = RequestLog(":memory:")
    for i in range(5):
        log.record_decision(f"rt_{i}", task="t", tier="sub", model=None, effort=None, runner=None, stage=None,
                            route_score=None, route_model=None, route_reason=None)
        log.set_outcome(f"rt_{i}", "pass", None, Usage(prompt_tokens=3000, completion_tokens=10, cached_tokens=2900))
    _, _, router = client_for(backend_factory, log=log)
    assert router.calls.shape_for("sub").source == "observed:5"


COLD = {"prompt_tokens": 5000, "completion_tokens": 200}  # no cache fields at all


def test_the_seeded_shape_matches_what_the_outcomes_built(backend_factory, tmp_path):
    """A cold task before any cache report is rejected; one after is real evidence. A restart must agree."""
    path = tmp_path / "log.db"
    log = RequestLog(path)
    client, _, router = client_for(backend_factory, log=log)
    tier = "sub"
    # Rejected: the tier has not yet shown it reports the cache. Then five cached
    # tasks establish that it does, a cold one afterwards counts, and three more
    # cached tasks keep building on it.
    usages = [COLD, *([CACHED] * 5), COLD, *([CACHED] * 3)]
    with client:
        for i, usage in enumerate(usages):
            log.record_decision(f"rt_seed_{i}", task="t", tier=tier, model=None, effort=None, runner=None,
                                stage=None, route_score=None, route_model=None, route_reason=None)
            client.post(f"/v1/route/rt_seed_{i}/outcome", json={"status": "pass", "usage": usage})
        live_shape = router.calls.shape_for(tier)
    # 9 of the 10 tasks count: the first cold one is rejected, proving record_task's
    # own rule rather than a coincidence of order.
    assert live_shape.source == "observed:9"

    reopened = RequestLog(path)
    _, _, restarted = client_for(backend_factory, log=reopened)
    assert restarted.calls.shape_for(tier) == live_shape
