"""Route only: a runner asks which model and effort, runs it itself, and reports the outcome.

Jev is never called: the router answers from a scripted `Ask`.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from llm_router.calibration import log_outcomes
from llm_router.capabilities import CapabilityRouter
from llm_router.config import parse_config
from llm_router.db import RequestLog

from test_capabilities import HARD, Ask, raw_config

TASK = {"task": "Fix the race in the scheduler", "stage": "implement", "files": ["core/src/scheduler.rs"]}


def client_for(backend_factory, ask=None, log=None):
    from llm_router.app import create_app

    config = parse_config(raw_config())
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
