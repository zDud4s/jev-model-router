"""The provider's own bill, kept beside the estimate, and filled in after the fact."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from jev_model_router.app import billed_total, create_app
from jev_model_router.config import parse_config
from jev_model_router.db import LogEntry, RequestLog
from jev_model_router.pricing import counterfactuals
from jev_model_router.reconcile import reconcile
from jev_model_router.schemas import Usage
from jev_model_router.stats import collect, format_text
from jev_model_router.verification import Verdict, VerificationOutcome

from conftest import BASE_CONFIG, FakeBackend

# The usage block OpenRouter actually returned on 2026-09-19, trimmed of the
# fields nothing reads.
OPENROUTER_USAGE = {
    "prompt_tokens": 91,
    "completion_tokens": 465,
    "total_tokens": 556,
    "cost": 0.0000412,
    "prompt_tokens_details": {"cached_tokens": 91, "cache_write_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 458},
}


def test_the_providers_charge_is_read_from_the_usage_block() -> None:
    usage = Usage.from_openai(OPENROUTER_USAGE)

    assert usage.billed_usd == 0.0000412
    assert usage.cached_tokens == 91


def test_a_provider_that_reports_no_charge_leaves_it_unknown_not_free() -> None:
    assert Usage.from_openai({"prompt_tokens": 10, "completion_tokens": 5}).billed_usd is None


def test_openrouters_spelling_of_cache_writes_is_read() -> None:
    # Only `cache_creation_tokens` used to be read, so every cache write through
    # OpenRouter was billed at nothing.
    usage = Usage.from_openai(
        {"prompt_tokens": 100, "prompt_tokens_details": {"cache_write_tokens": 40}}
    )

    assert usage.cache_write_tokens == 40


# --- the request path ---------------------------------------------------------


def _app(usages: dict[str, Usage], log: RequestLog, verification: dict[str, Any] | None = None):
    raw = dict(BASE_CONFIG)
    if verification:
        raw["verification"] = verification
    config = parse_config(raw)

    def factory(tier):
        return FakeBackend(tier, content="VERDICT: PASS", usage=usages.get(tier.name))

    return create_app(config, backend_factory=factory, log=log)


def _row(log: RequestLog) -> Any:
    return log.query("SELECT * FROM requests")[0]


def test_the_bill_and_the_upstream_id_are_written_beside_the_estimate() -> None:
    log = RequestLog(":memory:")
    usage = Usage(prompt_tokens=100, completion_tokens=50, billed_usd=0.0011)
    with TestClient(_app({"mid": usage}, log)) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "mid", "messages": [{"role": "user", "content": "hi"}]},
        )
        row = _row(log)

    assert row["billed_cost_usd"] == 0.0011
    assert row["billed_source"] == "inline"
    assert row["upstream_id"]
    # The estimate is untouched: two numbers, never one overwriting the other.
    assert row["cost_usd"] == (100 * 3.0 + 50 * 15.0) / 1_000_000


def test_a_request_is_billed_only_when_every_call_it_made_was() -> None:
    reviewed = VerificationOutcome(
        verdict=Verdict.FAIL,
        verifier_tier="top",
        verifier_usage=Usage(billed_usd=0.02),
        escalated=True,
        escalated_to="top",
        escalation_usage=Usage(billed_usd=0.05),
    )
    assert billed_total(Usage(billed_usd=0.01), reviewed) == 0.01 + 0.02 + 0.05

    # Any one call unreported and there is no total: a two-of-three bill beside
    # a three-call estimate would read as a saving.
    reviewed.escalation_usage = Usage()
    assert billed_total(Usage(billed_usd=0.01), reviewed) is None
    assert billed_total(Usage(), None) is None


def test_a_skipped_review_made_no_call_and_needs_no_bill() -> None:
    skipped = VerificationOutcome(verdict=Verdict.SKIPPED, verifier_tier="top")

    assert billed_total(Usage(billed_usd=0.01), skipped) == 0.01


def test_a_verifier_that_failed_leaves_the_request_unbilled() -> None:
    # It may well have been charged; nobody said how much.
    errored = VerificationOutcome(verdict=Verdict.ERROR, verifier_tier="top")

    assert billed_total(Usage(billed_usd=0.01), errored) is None


# --- reconcile ------------------------------------------------------------------

ROUTED_CONFIG = {
    **BASE_CONFIG,
    "tiers": {
        **BASE_CONFIG["tiers"],
        "remote": {
            "backend": "openai_compatible",
            "model": "some/model",
            "base_url": "https://openrouter.ai/api/v1",
            "context_window": 65536,
            "prices": {"input": 1.0, "output": 2.0},
        },
    },
}


def _abandoned_stream(log: RequestLog, config) -> int:
    # What an abandoned stream leaves: the provider's id from the first frame,
    # and zero tokens, because the usage frame never came.
    entry = LogEntry(
        request_id="r1",
        prompt_text="hi",
        tier="remote",
        stream=True,
        upstream_id="gen-abc",
        http_status=200,
        counterfactuals=counterfactuals(config, Usage()),
    )
    return log.record(entry)


GENERATION = {
    "total_cost": 0.0007,
    "native_tokens_prompt": 200,
    "native_tokens_completion": 300,
    "native_tokens_cached": 0,
}


def test_an_abandoned_stream_is_costed_from_the_providers_record() -> None:
    config = parse_config(ROUTED_CONFIG)
    log = RequestLog(":memory:")
    row_id = _abandoned_stream(log, config)
    asked: list[str] = []

    def fetch(tier, upstream_id):
        asked.append(upstream_id)
        return GENERATION

    report = reconcile(config, log, fetch=fetch)

    assert asked == ["gen-abc"]
    assert report.filled == 1 and report.tokens_recovered == 1
    row = log.query("SELECT * FROM requests WHERE id = ?", (row_id,))[0]
    assert row["billed_cost_usd"] == 0.0007
    assert row["billed_source"] == "reconciled"
    assert (row["input_tokens"], row["output_tokens"]) == (200, 300)
    # The estimate and every counterfactual were zero; they are recomputed.
    assert row["cost_usd"] == (200 * 1.0 + 300 * 2.0) / 1_000_000
    top = log.query(
        "SELECT cost_usd FROM counterfactuals WHERE request_row_id = ? AND tier = 'top'",
        (row_id,),
    )[0]
    assert top["cost_usd"] == (200 * 15.0 + 300 * 75.0) / 1_000_000


def test_reconcile_waits_for_a_record_that_is_not_there_yet() -> None:
    config = parse_config(ROUTED_CONFIG)
    log = RequestLog(":memory:")
    _abandoned_stream(log, config)

    report = reconcile(config, log, fetch=lambda tier, upstream_id: None)

    assert report.not_yet == 1 and report.filled == 0
    assert log.query("SELECT billed_cost_usd FROM requests")[0][0] is None


def test_a_dry_run_writes_nothing() -> None:
    config = parse_config(ROUTED_CONFIG)
    log = RequestLog(":memory:")
    _abandoned_stream(log, config)

    report = reconcile(config, log, fetch=lambda tier, upstream_id: GENERATION, dry_run=True)

    assert report.filled == 1
    assert log.query("SELECT billed_cost_usd FROM requests")[0][0] is None


def test_a_provider_with_no_per_request_lookup_is_counted_not_skipped() -> None:
    config = parse_config(ROUTED_CONFIG)
    log = RequestLog(":memory:")
    log.record(LogEntry(request_id="r2", prompt_text="hi", tier="mid", upstream_id="chatcmpl-1"))

    def fetch(tier, upstream_id):
        raise AssertionError("mid has no lookup to ask")

    report = reconcile(config, log, fetch=fetch)

    assert report.unsupported == {"mid": 1}


# --- stats ------------------------------------------------------------------------


def test_stats_says_when_the_price_table_disagrees_with_the_invoice() -> None:
    log = RequestLog(":memory:")
    log.record(
        LogEntry(
            request_id="r3",
            prompt_text="hi",
            tier="mid",
            cost_usd=0.010,
            upstream_id="gen-1",
            billed_cost_usd=0.013,
            billed_source="inline",
        )
    )

    stats = collect(log)
    text = format_text(stats)

    assert stats.billing is not None
    assert round(stats.billing.drift, 3) == 0.3
    assert "DRIFT +30.0%" in text


def test_stats_names_rows_reconcile_could_recover() -> None:
    config = parse_config(ROUTED_CONFIG)
    log = RequestLog(":memory:")
    _abandoned_stream(log, config)

    assert "RECOVERABLE 1" in format_text(collect(log))


def test_a_log_with_no_provider_figures_prints_no_billing_block() -> None:
    log = RequestLog(":memory:")
    log.record(LogEntry(request_id="r4", prompt_text="hi", tier="mid", cost_usd=0.01))

    assert collect(log).billing is None


# --- a stream the client leaves ---------------------------------------------------


def test_a_stream_the_client_abandons_still_writes_its_row() -> None:
    # Found against a real uvicorn server: a client read one frame and hung up,
    # and the request left NO row at all -- not a zero-cost one, nothing, not
    # even after a clean shutdown. On disconnect Starlette cancels the
    # response's task group; anyio's cancellation is level-triggered, so the
    # await that writes the row in the stream's `finally` was cancelled before
    # the write was ever submitted. TestClient cannot show this -- it reads
    # every stream to the end -- so this runs a real server.
    import asyncio
    import socket
    import threading
    import time

    import httpx
    import uvicorn

    from jev_model_router.backends.base import StreamChunk
    from jev_model_router.schemas import build_chunk

    class SlowStream(FakeBackend):
        async def stream(self, request):
            yield StreamChunk(
                data={
                    **build_chunk(
                        model=self.tier.name,
                        completion_id="gen-slow",
                        delta={"role": "assistant", "content": "Once"},
                    ),
                    "id": "gen-slow",
                }
            )
            await asyncio.sleep(30)  # the rest never comes: the client has left

    log = RequestLog(":memory:")
    app = create_app(
        parse_config(BASE_CONFIG), backend_factory=lambda tier: SlowStream(tier), log=log
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="error", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)

        payload = {"model": "mid", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        with httpx.Client(timeout=10) as client:
            with client.stream(
                "POST", f"http://127.0.0.1:{port}/v1/chat/completions", json=payload
            ) as response:
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        break  # hang up after the first frame

        deadline = time.monotonic() + 5
        while not log.query("SELECT id FROM requests") and time.monotonic() < deadline:
            time.sleep(0.05)
        rows = log.query("SELECT * FROM requests")
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    assert len(rows) == 1
    row = rows[0]
    assert row["upstream_id"] == "gen-slow"
    # Nothing billed and no tokens: exactly what `reconcile` exists to fill.
    assert row["billed_cost_usd"] is None
    assert row["http_status"] == 499
    assert "disconnected" in row["error"]


def test_two_calls_tokens_add_and_an_unreported_bill_does_not_become_zero() -> None:
    # A request answered twice -- a retry, or a retry and then an escalation --
    # carries both calls in one row.
    reported = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, billed_usd=0.02)
    also = Usage(prompt_tokens=3, completion_tokens=7, total_tokens=10, billed_usd=0.01)

    both = reported + also

    assert (both.prompt_tokens, both.completion_tokens, both.total_tokens) == (13, 12, 25)
    assert both.billed_usd == 0.03

    # One side did not report what it charged, so the TOTAL is unknown. Adding
    # zero for it would print a bill that looks measured and is short.
    silent = Usage(prompt_tokens=1, completion_tokens=1)
    assert (reported + silent).billed_usd is None
    assert (reported + silent).prompt_tokens == 11
