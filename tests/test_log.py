"""The SQLite log: what gets written, what survives failure, and the stats read-back."""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from jev_model_router.app import create_app
from jev_model_router.backends.base import BackendError
from jev_model_router.db import _MIGRATIONS, SCHEMA_VERSION, LogEntry, RequestLog, sha256_hex
from jev_model_router.schemas import Usage
from jev_model_router.stats import collect, format_text

from conftest import FakeBackend


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
