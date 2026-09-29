"""The SQLite log: what gets written, what survives failure, and the stats read-back."""

from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from jev_model_router.app import create_app
from jev_model_router.backends.base import BackendError
from jev_model_router.calls import MIN_EVIDENCE
from jev_model_router.config import TaskShape
from jev_model_router.db import _MIGRATIONS, SCHEMA_VERSION, LogEntry, RequestLog, sha256_hex
from jev_model_router.pricing import Counterfactual
from jev_model_router.schemas import Usage
from jev_model_router.stats import collect, format_text
from jev_model_router.verification import Verdict, VerificationOutcome

from conftest import FakeBackend
from test_calls import calls as build_call_router


def _app_with_log(config, log, *, backend_error: BackendError | None = None, usage: Usage | None = None):
    def factory(tier):
        return FakeBackend(tier, error=backend_error, usage=usage)

    return create_app(config, backend_factory=factory, log=log)


def _post(client, **overrides):
    payload = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
    payload.update(overrides)
    return client.post("/v1/chat/completions", json=payload)


def test_the_migration_is_versioned_and_idempotent(tmp_path) -> None:
    path = tmp_path / "log.db"
    first = RequestLog(path)
    assert first.schema_version == SCHEMA_VERSION
    first.close()

    # Re-opening an existing database must not re-run the migration or fail.
    second = RequestLog(path)
    assert second.schema_version == SCHEMA_VERSION
    second.close()


def test_the_constant_and_the_migration_list_cannot_drift(tmp_path) -> None:
    # SCHEMA_VERSION is written by hand and the list is appended to by hand.
    # Nothing else notices when only one of the two is updated.
    assert SCHEMA_VERSION == len(_MIGRATIONS)


def test_an_old_database_is_upgraded_without_losing_what_it_already_said(tmp_path) -> None:
    path = tmp_path / "log.db"
    # Build a v1 database by hand: apply only the first migration and stop.
    conn = sqlite3.connect(path)
    conn.executescript(_MIGRATIONS[0])
    conn.execute("PRAGMA user_version = 1")
    conn.execute(
        """
        INSERT INTO requests (request_id, ts, tier, prompt_sha256, cost_usd)
        VALUES ('req_old', '2026-01-01T00:00:00+00:00', 'mid', 'abc', 0.25)
        """
    )
    conn.commit()
    conn.close()

    log = RequestLog(path)
    assert log.schema_version == SCHEMA_VERSION
    row = log.query("SELECT tier, cost_usd, route_cost_usd, final_tier FROM requests")[0]
    log.close()

    # A v1 row had exactly one call, so its route cost IS its total and the tier
    # it was routed to IS the tier that answered. Leaving the new columns at
    # their defaults would make every pre-existing request read as a free route
    # served by nobody.
    assert row["route_cost_usd"] == pytest.approx(0.25)
    assert row["final_tier"] == "mid"


def test_wal_is_enabled_on_a_file_database(tmp_path) -> None:
    log = RequestLog(tmp_path / "log.db")
    mode = log.query("PRAGMA journal_mode")[0][0]
    log.close()

    assert mode.lower() == "wal"


def test_a_prompt_is_hashed_and_not_stored_by_default(config) -> None:
    log = RequestLog(":memory:")
    app = _app_with_log(config, log)
    with TestClient(app) as client:
        _post(client)
        row = log.query("SELECT prompt_sha256, prompt_text FROM requests")[0]

        assert len(row["prompt_sha256"]) == 64
        # The default must not retain user text.
        assert row["prompt_text"] is None


def test_the_full_prompt_is_stored_only_when_the_operator_opts_in(config) -> None:
    log = RequestLog(":memory:", store_prompts=True)
    app = _app_with_log(config, log)
    with TestClient(app) as client:
        _post(client)
        row = log.query("SELECT prompt_sha256, prompt_text FROM requests")[0]

        assert "hi" in row["prompt_text"]
        assert row["prompt_sha256"] == sha256_hex(row["prompt_text"])


def test_a_served_request_records_tokens_cost_and_latency(config) -> None:
    log = RequestLog(":memory:")
    app = _app_with_log(config, log)
    with TestClient(app) as client:
        _post(client, model="mid")
        row = log.query("SELECT * FROM requests")[0]

        assert row["tier"] == "mid"
        assert row["backend"] == "openai_compatible"
        assert row["model"] == "mid-model"
        assert row["http_status"] == 200
        assert row["input_tokens"] == 100
        assert row["output_tokens"] == 50
        # mid: 100 in at $3/M plus 50 out at $15/M.
        assert row["cost_usd"] == pytest.approx((100 * 3.0 + 50 * 15.0) / 1_000_000)
        assert row["latency_ms"] >= 0


def test_every_configured_tier_gets_a_counterfactual_row(config) -> None:
    log = RequestLog(":memory:")
    app = _app_with_log(config, log)
    with TestClient(app) as client:
        _post(client)
        rows = {
            row["tier"]: row
            for row in log.query("SELECT tier, cost_usd, priced FROM counterfactuals")
        }

        assert set(rows) == {"cheap", "mid", "top"}
        assert rows["cheap"]["cost_usd"] == 0.0
        assert rows["cheap"]["priced"] == 0
        assert rows["mid"]["cost_usd"] == pytest.approx((100 * 3.0 + 50 * 15.0) / 1_000_000)
        assert rows["top"]["cost_usd"] == pytest.approx((100 * 15.0 + 50 * 75.0) / 1_000_000)


def test_the_log_survives_a_backend_error_and_records_it(config) -> None:
    log = RequestLog(":memory:")
    app = _app_with_log(config, log, backend_error=BackendError("upstream exploded", status=502))
    with TestClient(app) as client:
        response = _post(client)
        assert response.status_code == 502

        row = log.query("SELECT tier, http_status, error FROM requests")[0]
        # A failed request still costs latency and is still a fact about the
        # tier, so it must be in the log.
        assert row["tier"] == "cheap"
        assert row["http_status"] == 502
        assert "upstream exploded" in row["error"]


def test_eligibility_rejections_are_recorded_on_the_row(config) -> None:
    log = RequestLog(":memory:")
    app = _app_with_log(config, log)
    with TestClient(app) as client:
        _post(client, tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}])
        row = log.query("SELECT tier, eligibility_rejections FROM requests")[0]

        assert row["tier"] != "cheap"
        assert "tools_unsupported" in row["eligibility_rejections"]


def test_a_streamed_request_is_logged_after_the_stream_ends(config) -> None:
    log = RequestLog(":memory:")
    app = _app_with_log(config, log)
    with TestClient(app) as client:
        response = _post(client, model="mid", stream=True)
        assert response.status_code == 200

        row = log.query("SELECT stream, output_tokens, http_status FROM requests")[0]
        assert row["stream"] == 1
        assert row["output_tokens"] == 50
        assert row["http_status"] == 200


def test_a_broken_log_does_not_fail_the_request(config) -> None:
    log = RequestLog(":memory:")
    # Simulate the log breaking under the request path: the table disappears.
    log.query("DROP TABLE requests")
    app = _app_with_log(config, log)

    with TestClient(app) as client:
        response = _post(client)

    # The client is served regardless; the failure is counted, not raised.
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hello"
    assert log.failed_writes == 1


def test_a_log_write_failure_is_itself_recorded(config) -> None:
    log = RequestLog(":memory:")
    log.query("DROP TABLE requests")
    app = _app_with_log(config, log)
    with TestClient(app) as client:
        _post(client)
        # Read inside the client's context: the app's lifespan owns the log and closes
        # it on shutdown, so the same query one line below the `with` hits a closed
        # connection and fails for a reason that has nothing to do with log failures.
        failures = log.query("SELECT request_id, error FROM log_failures")

    assert len(failures) == 1
    assert "requests" in failures[0]["error"]


def test_record_returns_none_instead_of_raising_when_the_write_fails() -> None:
    log = RequestLog(":memory:")
    log.query("DROP TABLE requests")

    # Directly, without the app: `record` is the contract that must not raise.
    assert log.record(LogEntry(request_id="req_x", prompt_text="hi")) is None
    assert log.failed_writes == 1
    log.close()


def test_stats_reports_spend_by_tier_and_savings_against_every_baseline(config) -> None:
    log = RequestLog(":memory:")
    app = _app_with_log(config, log)
    with TestClient(app) as client:
        _post(client)            # -> cheap, free
        _post(client, model="mid")

        report = collect(log)

        assert report.requests == 2
        assert report.errors == 0
        by_tier = {row.tier: row for row in report.by_tier}
        assert set(by_tier) == {"cheap", "mid"}
        assert by_tier["cheap"].cost_usd == 0.0

        baselines = {b.tier: b for b in report.baselines}
        # A baseline exists for EVERY configured tier, not only the dearest one.
        assert set(baselines) == {"cheap", "mid", "top"}

        mid_per_request = (100 * 3.0 + 50 * 15.0) / 1_000_000
        top_per_request = (100 * 15.0 + 50 * 75.0) / 1_000_000
        assert baselines["top"].total_usd == pytest.approx(2 * top_per_request)
        assert baselines["mid"].total_usd == pytest.approx(2 * mid_per_request)
        # Actual spend was one free request plus one mid request.
        assert report.total_cost_usd == pytest.approx(mid_per_request)
        assert baselines["top"].savings_usd == pytest.approx(2 * top_per_request - mid_per_request)
        # The cheap baseline is unpriced, so its savings figure is not a claim.
        assert baselines["cheap"].unpriced_rows == 2

        text = format_text(report)
        assert "always top" in text
        assert "no price configured" in text


def test_stats_flags_a_baseline_that_could_not_have_served_the_request(config) -> None:
    log = RequestLog(":memory:")
    app = _app_with_log(config, log)
    with TestClient(app) as client:
        _post(client, tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}])

        baselines = {b.tier: b for b in collect(log).baselines}

        # `cheap` declares no tool support, so "we could have used cheap" is not
        # a real alternative for this request and the report says so.
        assert baselines["cheap"].ineligible_rows == 1
        assert baselines["mid"].ineligible_rows == 0


def test_counterfactual_rows_are_removed_with_their_request(tmp_path, config) -> None:
    log = RequestLog(tmp_path / "log.db")
    app = _app_with_log(config, log)
    with TestClient(app) as client:
        _post(client)
        row_id = log.query("SELECT id FROM requests")[0]["id"]
        log.query("DELETE FROM requests WHERE id = ?", (row_id,))

        assert log.query("SELECT * FROM counterfactuals") == []


def test_the_schema_rejects_a_counterfactual_without_a_request(tmp_path) -> None:
    log = RequestLog(tmp_path / "log.db")
    with pytest.raises(sqlite3.IntegrityError):
        log.query(
            "INSERT INTO counterfactuals (request_row_id, tier, cost_usd, priced, eligible) VALUES (?,?,?,?,?)",
            (999, "ghost", 1.0, 1, 1),
        )
    log.close()


def test_a_refused_route_is_kept_in_its_own_table() -> None:
    log = RequestLog(":memory:", store_prompts=True)
    assert log.record_rejection("rj_1", task="fix it", stage="implement", route_model="capabilities:abc",
                                route_reason='{"rule":"jev_failure"}') is True
    [row] = log.query("SELECT * FROM route_rejections")
    assert (row["request_id"], row["stage"], row["route_model"]) == ("rj_1", "implement", "capabilities:abc")
    assert row["task_sha256"] == sha256_hex("fix it") and row["task_text"] == "fix it"
    assert log.query("SELECT COUNT(*) AS n FROM route_decisions")[0]["n"] == 0


def test_record_rejection_returns_false_instead_of_raising() -> None:
    log = RequestLog(":memory:")
    log.query("DROP TABLE route_rejections")
    assert log.record_rejection("rj_x", task="t", stage=None, route_model=None, route_reason="{}") is False
    assert log.failed_writes == 1


def decided(log, i, tier, usage=None, reason=None):
    log.record_decision(f"rt_{i}", task="t", tier=tier, model=None, effort=None, runner=None, stage=None,
                        route_score=None, route_model=None, route_reason=reason)
    if usage is not None:
        log.set_outcome(f"rt_{i}", "pass", None, usage)


def test_a_route_outcome_keeps_its_cache_tokens() -> None:
    log = RequestLog(":memory:")
    decided(log, 1, "a", Usage(prompt_tokens=10, completion_tokens=2, cached_tokens=7, cache_write_tokens=1))
    decided(log, 2, "a", Usage(prompt_tokens=10, completion_tokens=2))
    row = log.decision("rt_1")
    assert (row["outcome_cached_tokens"], row["outcome_cache_write_tokens"]) == (7, 1)
    # Every outcome with usage comes back, cold or cached: record_task decides which count.
    assert [(r["tier"], r["cached"]) for r in log.task_usage()] == [("a", 7), ("a", 0)]


def test_the_task_cost_block_shows_each_tiers_shape_the_flips_and_the_gap() -> None:
    log = RequestLog(":memory:")
    for i in range(5):
        decided(log, i, "a", Usage(prompt_tokens=3000, completion_tokens=10, cached_tokens=2900, cache_write_tokens=60))
    flipped = json.dumps({"rule": "x", "shape_flipped": True, "shapeless_pick": "a"})
    decided(log, 5, "b", Usage(prompt_tokens=3000, completion_tokens=10), reason=flipped)
    stats = collect(log, task_shape=TaskShape(295, 0.971, 0.026))
    rows = {r.tier: r for r in stats.task_cost.tiers}
    assert rows["a"].source == "observed:5" and rows["a"].input_per_output == 300.0
    assert rows["b"].source == "config" and rows["b"].without_cache == 1
    assert stats.task_cost.flips == {"b over a": 1}
    text = format_text(stats)
    assert "task cost" in text and "b over a: 1" in text and "no cache tokens" in text
    assert collect(RequestLog(":memory:")).task_cost is None  # no route decisions, no block


def test_a_row_whose_cache_exceeds_its_input_is_rejected_not_counted_as_no_cache() -> None:
    log = RequestLog(":memory:")
    decided(log, 0, "c", Usage(prompt_tokens=1000, completion_tokens=100, cached_tokens=900, cache_write_tokens=200))
    stats = collect(log)
    row = {r.tier: r for r in stats.task_cost.tiers}["c"]
    assert row.rejected == 1 and row.without_cache == 0
    assert "don't add up" in format_text(stats)


def test_stats_pools_the_same_shape_the_router_does() -> None:
    """The same rows, replayed into a CallRouter and into the log: `collect` must agree with `shape_for`."""
    router = build_call_router()
    log = RequestLog(":memory:")
    cold_before = Usage(prompt_tokens=1000, completion_tokens=100)  # before "sub" has shown it reports cache
    cached = Usage(prompt_tokens=30_000, completion_tokens=100, cached_tokens=29_000, cache_write_tokens=500)
    cold_after = Usage(prompt_tokens=1000, completion_tokens=100)  # after: a real cold task, not silence

    for i, usage in enumerate([cold_before, *([cached] * MIN_EVIDENCE), cold_after]):
        decided(log, i, "sub", usage)
        router.record_task("sub", usage)

    stats = collect(log)
    row = {r.tier: r for r in stats.task_cost.tiers}["sub"]
    shape = router.shape_for("sub")
    assert row.source == shape.source
    assert (row.input_per_output, row.cache_read, row.cache_write) == (
        round(shape.input_per_output, 1), round(shape.cache_read, 3), round(shape.cache_write, 3),
    )


def test_stats_count_what_was_refused_because_jev_did_not_answer() -> None:
    log = RequestLog(":memory:")
    assert "JEV REJECTED" not in format_text(collect(log))
    log.record(LogEntry(request_id="req_1", prompt_text="hi", http_status=503,
                        route_reason='{"rule":"jev_failure","on_jev_failure":"reject","why":"x"}'))
    log.record(LogEntry(request_id="req_2", prompt_text="hi", http_status=200,
                        route_reason='{"rule":"fallback","why":"jev error: x","on_jev_failure":"fallback"}'))
    log.record_rejection("rj_1", task="t", stage=None, route_model=None, route_reason="{}")
    stats = collect(log)
    assert (stats.jev_rejected_chat, stats.jev_rejected_route) == (1, 1)
    assert "JEV REJECTED    2  (chat 1, route 1)" in format_text(stats)


def _unsure_reason(**extra) -> str:
    return json.dumps({"rule": "x", "unsure": {"band": [0.3, 0.7], "min_level": 2.0, **extra}})


def test_the_unsure_block_counts_what_the_floor_raised_and_what_it_cost() -> None:
    log = RequestLog(":memory:")
    raised = {"reqs": ["reasoning"], "raised_from": ["a", 0.81, 0.001], "extra": 0.004}
    decided(log, 1, "b", reason=_unsure_reason(**raised))
    log.set_outcome("rt_1", "pass", None, None)
    decided(log, 2, "a", reason=_unsure_reason(reqs=["reasoning", "code"]))
    log.set_outcome("rt_2", "fail", None, None)
    decided(log, 3, "a", reason=_unsure_reason(reqs=["code"], unmet=True))
    decided(log, 4, "a", reason=json.dumps({"rule": "x"}))  # Jev was sure: not counted
    # A proxied request the floor raised: $0.010 on b, where a would have cost $0.002.
    log.record(LogEntry(
        request_id="r1", prompt_text="p", tier="b", cost_usd=0.010, route_reason=_unsure_reason(**raised),
        counterfactuals=[Counterfactual("a", 0.002, True, True), Counterfactual("b", 0.010, True, True)],
        verification=VerificationOutcome(verdict=Verdict.FAIL),
    ))
    u = collect(log).unsure
    assert (u.decisions, u.raised, u.unmet) == (4, 2, 1)
    assert u.estimated_extra_usd == pytest.approx(0.008)
    assert u.measured_extra_usd == pytest.approx(0.008) and u.measured_rows == 1
    assert (u.raised_pass, u.raised_fail, u.kept_pass, u.kept_fail) == (1, 1, 0, 1)
    assert u.by_requirement == {"reasoning": 3, "code": 2}
    assert u.band == "[0.30, 0.70]"
    text = format_text(collect(log))
    assert "unsure floor" in text and "raised 2" in text and "measured extra" in text
    assert collect(RequestLog(":memory:")).unsure is None


def test_an_unsure_block_over_two_bands_says_mixed() -> None:
    log = RequestLog(":memory:")
    decided(log, 1, "a", reason=_unsure_reason(reqs=["code"]))
    decided(log, 2, "a", reason=json.dumps({"rule": "x", "unsure": {"reqs": ["code"], "band": [0.4, 0.6]}}))
    assert collect(log).unsure.band == "mixed"


def test_the_unsure_block_ignores_what_it_cannot_read() -> None:
    log = RequestLog(":memory:")
    raised = {"reqs": ["reasoning"], "raised_from": ["a", 0.81, 0.001], "extra": 0.004}
    # An unpriced counterfactual for the raised-from tier: no known cost, so not measured.
    log.record(LogEntry(
        request_id="r1", prompt_text="p", tier="b", cost_usd=0.010, route_reason=_unsure_reason(**raised),
        counterfactuals=[Counterfactual("a", 0.002, False, True)],
    ))
    # Route outcomes that are neither pass nor fail.
    decided(log, 1, "a", reason=_unsure_reason(reqs=["code"]))
    log.set_outcome("rt_1", "rate_limited", None, None)
    decided(log, 2, "a", reason=_unsure_reason(reqs=["code"]))
    log.set_outcome("rt_2", "error", None, None)
    # A verification verdict of ERROR: the verifier itself broke, not a judgement.
    log.record(LogEntry(
        request_id="r2", prompt_text="p", tier="a", cost_usd=0.001, route_reason=_unsure_reason(reqs=["code"]),
        verification=VerificationOutcome(verdict=Verdict.ERROR),
    ))
    # Malformed `unsure` fields: must not crash, and must not be read as raised.
    decided(log, 3, "a", reason=json.dumps(
        {"rule": "x", "unsure": {"raised_from": 5, "band": 0.5, "reqs": "code"}}
    ))

    u = collect(log).unsure
    assert u.decisions == 5
    assert u.measured_rows == 0 and u.measured_extra_usd == 0.0
    assert (u.raised_pass, u.raised_fail, u.kept_pass, u.kept_fail) == (0, 0, 0, 0)
    assert u.raised == 1  # only the first entry has a well-formed raised_from
    assert u.band == "mixed"  # the malformed band forces it
    assert isinstance(collect(log).to_dict()["unsure"], dict)


def routed(log, i=1, tier="sub"):
    log.record_decision(f"rt_{i}", task="t", tier=tier, model=None, effort=None, runner=None, stage=None,
                        route_score=None, route_model=None, route_reason=None)


RUN = Usage(prompt_tokens=100, completion_tokens=10, cached_tokens=80, cache_write_tokens=5)
GUESS = Usage(prompt_tokens=999, completion_tokens=99, cached_tokens=1, cache_write_tokens=1)


def tokens(row):
    return (row["outcome_input_tokens"], row["outcome_output_tokens"],
            row["outcome_cached_tokens"], row["outcome_cache_write_tokens"])


def events(log):
    return [(r["decision_id"], r["kind"], r["writer"]) for r in
            log.query("SELECT decision_id, kind, writer FROM route_events ORDER BY id")]


def test_a_delegates_usage_survives_an_outcome_that_brings_its_own() -> None:
    log = RequestLog(":memory:")
    routed(log)
    assert log.set_usage("rt_1", RUN, "delegate") == "ok"
    assert log.set_outcome("rt_1", "pass", None, GUESS, "app") == "kept_existing"
    row = log.decision("rt_1")
    assert row["outcome"] == "pass" and tokens(row) == (100, 10, 80, 5)


def test_an_outcome_first_leaves_room_for_the_usage_and_the_first_usage_is_kept() -> None:
    log = RequestLog(":memory:")
    routed(log)
    assert log.set_outcome("rt_1", "pass", None, None, "app") == "ok"
    assert tokens(log.decision("rt_1")) == (None, None, None, None)
    assert log.set_usage("rt_1", RUN, "delegate") == "ok"
    assert log.set_usage("rt_1", GUESS, "delegate") == "kept_existing"
    assert tokens(log.decision("rt_1")) == (100, 10, 80, 5)
    assert log.set_usage("rt_nope", RUN, "delegate") == "missing"


def test_two_sources_never_mix_in_one_row() -> None:
    log = RequestLog(":memory:")
    routed(log)
    log.set_usage("rt_1", Usage(prompt_tokens=100, completion_tokens=10), "delegate")  # no cache fields
    log.set_outcome("rt_1", "pass", None, RUN, "app")
    assert tokens(log.decision("rt_1")) == (100, 10, 0, 0)


def test_an_outcome_with_the_first_tokens_writes_them_and_says_ok() -> None:
    log = RequestLog(":memory:")
    routed(log)
    assert log.set_outcome("rt_1", "fail", "tests failed", RUN, "app") == "ok"
    assert tokens(log.decision("rt_1")) == (100, 10, 80, 5)
    assert log.set_outcome("rt_1", "pass", None, None, "app") == "exists"
    assert log.set_outcome("rt_nope", "pass", None, None, "app") == "missing"


def test_each_write_appends_one_event_and_only_when_it_changed_something() -> None:
    log = RequestLog(":memory:")
    routed(log, 1)
    routed(log, 2)
    log.set_usage("rt_1", RUN, "d")
    log.set_usage("rt_1", GUESS, "d")                  # kept: no event
    log.set_outcome("rt_1", "pass", None, GUESS, "a")  # outcome only: its usage was not kept
    log.set_outcome("rt_1", "fail", None, None, "a")   # exists: no event
    log.set_outcome("rt_2", "pass", None, RUN, "a")    # outcome, and the tokens it wrote
    assert events(log) == [("rt_1", "usage", "d"), ("rt_1", "outcome", "a"),
                           ("rt_2", "outcome", "a"), ("rt_2", "usage", "a")]


def test_events_after_leaves_out_the_readers_own_and_carries_the_row() -> None:
    log = RequestLog(":memory:")
    routed(log)
    log.set_usage("rt_1", RUN, "delegate")
    log.set_outcome("rt_1", "rate_limited", None, None, "app")
    [usage] = log.events_after(0, "app")
    assert (usage["kind"], usage["tier"], usage["input"], usage["output"], usage["cached"], usage["written"]) == (
        "usage", "sub", 100, 10, 80, 5)
    assert [r["kind"] for r in log.events_after(0, "delegate")] == ["outcome"]
    assert log.events_after(0, "delegate")[0]["outcome"] == "rate_limited"
    assert log.last_event_id() == 2 and log.events_after(2, "x") == []
    assert RequestLog(":memory:").last_event_id() == 0


def test_task_usage_includes_tokens_that_arrived_without_an_outcome() -> None:
    log = RequestLog(":memory:")
    routed(log)
    log.set_usage("rt_1", RUN, "delegate")
    assert [(r["tier"], r["input"]) for r in log.task_usage()] == [("sub", 100)]
