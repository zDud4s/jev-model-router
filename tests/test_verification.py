"""The verification loop: verdicts, escalation, and what the loop costs.

The expensive question this suite exists to answer is not "does it escalate" but
"is the escalation paid for in the right column". A loop whose second and third
calls land on the counterfactual side of the ledger would report savings it did
not make, which is the exact failure mode the project was built to avoid.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from llm_router.app import create_app
from llm_router.backends.base import BackendError, BackendResponse
from llm_router.config import ConfigError, parse_config
from llm_router.db import RequestLog
from llm_router.schemas import ChatCompletionRequest, build_completion
from llm_router.stats import collect, format_text
from llm_router.verification import answer_text, build_review_request, parse_verdict

from conftest import BASE_CONFIG, FakeBackend, make_request

TOP_PER_CALL = (100 * 15.0 + 50 * 75.0) / 1_000_000
MID_PER_CALL = (100 * 3.0 + 50 * 15.0) / 1_000_000


class ScriptedBackend(FakeBackend):
    """A backend whose answers are a queue.

    The verifier tier and the escalation tier are the same tier by default, so a
    single-answer fake cannot express "review this, then answer it properly".
    """

    def __init__(self, tier, *, replies: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(tier, **kwargs)
        self.replies = list(replies or [])

    async def complete(self, request: ChatCompletionRequest) -> BackendResponse:
        self.calls.append(request)
        if self.error:
            raise self.error
        content = self.replies.pop(0) if self.replies else self.content
        return BackendResponse(
            body=build_completion(model=self.tier.name, content=content, usage=self.usage),
            usage=self.usage,
        )


def verifying(**verification: Any) -> Any:
    settings = {"enabled": True, "verifier_tier": "top"}
    settings.update(verification)
    return parse_config({**BASE_CONFIG, "verification": settings})


def build(config, scripts: dict[str, list[str]], log: RequestLog, **kw: Any):
    """An app whose every tier answers from a script."""
    backends: dict[str, ScriptedBackend] = {}

    def factory(tier):
        backend = ScriptedBackend(tier, replies=scripts.get(tier.name), **kw)
        backends[tier.name] = backend
        return backend

    return create_app(config, backend_factory=factory, log=log), backends


def post(client, **overrides: Any):
    payload = {"model": "auto", "messages": [{"role": "user", "content": "what is 2+2?"}]}
    payload.update(overrides)
    return client.post("/v1/chat/completions", json=payload)


def verification_row(log: RequestLog) -> Any:
    return log.query("SELECT * FROM verifications")[0]


# --- parsing ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("VERDICT: PASS", True),
        ("VERDICT: FAIL - the sum is wrong", False),
        ("**VERDICT:** FAIL", False),
        ("verdict pass", True),
        # A model that leads with reasoning and gets to the verdict eventually.
        ("Let me check the arithmetic.\n\nVERDICT: FAIL - 2+2 is 4", False),
        # No verdict token at all: the bare form still answers.
        ("PASS", True),
        ("FAIL, it is wrong", False),
    ],
)
def test_a_verdict_is_read_out_of_the_shapes_models_actually_write(text, expected) -> None:
    passed, _ = parse_verdict(text)
    assert passed is expected


def test_a_reply_with_no_verdict_in_it_is_unparseable_rather_than_guessed() -> None:
    # Guessing here would be the worst option available: it produces a pass rate
    # that looks like a measurement and is not one.
    assert parse_verdict("I think the answer looks fine to me.") == (None, None)
    assert parse_verdict("") == (None, None)


def test_the_reason_is_kept_and_stops_at_the_first_line() -> None:
    passed, reason = parse_verdict("VERDICT: FAIL - it answers a different question\nmore prose")

    assert passed is False
    assert reason == "it answers a different question"


def test_an_answer_that_is_only_tool_calls_has_no_text_to_review() -> None:
    body = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{}]}}]}

    assert answer_text(body) == ""


def test_the_review_prompt_carries_the_answer_and_drops_the_tools() -> None:
    config = verifying()
    request = make_request(
        messages=[{"role": "user", "content": "what is 2+2?"}],
        tools=[{"type": "function", "function": {"name": "calc", "parameters": {}}}],
    )

    review = build_review_request(request, "the answer is 5", config.verification)
    prompt = review.messages[-1].content

    assert "what is 2+2?" in prompt
    assert "the answer is 5" in prompt
    # Tool schemas are the largest part of an agentic request and would both
    # inflate the review's bill and invite the verifier to call a tool instead
    # of judging.
    assert review.tools is None
    assert review.max_tokens == config.verification.max_verdict_tokens


def test_a_long_answer_keeps_its_ending_when_truncated() -> None:
    config = verifying(max_answer_chars=200)
    answer = "start " + "x" * 5_000 + " END OF ANSWER"

    prompt = build_review_request(make_request(), answer, config.verification).messages[-1].content

    # "stops mid-thought" is a FAIL condition and it is visible only at the end,
    # so the tail must survive truncation.
    assert "END OF ANSWER" in prompt
    assert "characters omitted" in prompt


# --- the happy path ---------------------------------------------------------


def test_a_passing_verdict_leaves_the_cheap_answer_alone_and_still_bills_the_review() -> None:
    log = RequestLog(":memory:")
    app, backends = build(verifying(), {"top": ["VERDICT: PASS"]}, log)

    with TestClient(app) as client:
        response = post(client)
        row = log.query("SELECT * FROM requests")[0]
        check = verification_row(log)

        assert response.json()["choices"][0]["message"]["content"] == "hello"
        assert response.headers["X-Router-Verdict"] == "pass"
        assert row["final_tier"] == "cheap"
        assert check["verdict"] == "pass"
        assert check["escalated"] == 0
        # The review happened and was not free. A pass that costs nothing is the
        # bug this assertion exists to catch.
        assert check["verifier_cost_usd"] == pytest.approx(TOP_PER_CALL)
        assert row["cost_usd"] == pytest.approx(TOP_PER_CALL)
        # ...and the routed tier itself is still free.
        assert row["route_cost_usd"] == pytest.approx(0.0)
        assert len(backends["top"].calls) == 1


def test_a_failing_verdict_escalates_and_the_client_never_sees_the_bad_answer() -> None:
    log = RequestLog(":memory:")
    app, backends = build(
        verifying(), {"top": ["VERDICT: FAIL - it is wrong", "the proper answer"]}, log
    )

    with TestClient(app) as client:
        response = post(client)
        row = log.query("SELECT * FROM requests")[0]
        check = verification_row(log)

        assert response.json()["choices"][0]["message"]["content"] == "the proper answer"
        assert response.headers["X-Router-Tier"] == "top"
        assert response.headers["X-Router-Escalated-From"] == "cheap"
        assert check["verdict"] == "fail"
        assert check["reason"] == "it is wrong"
        assert check["escalated"] == 1 and check["escalated_to"] == "top"
        # Two calls at the top tier: the review and the answer.
        assert row["cost_usd"] == pytest.approx(2 * TOP_PER_CALL)
        # The tier written on the routing decision stays the tier that was
        # routed to; `final_tier` is who the client heard from. Collapsing the
        # two would make the counterfactual meaningless.
        assert row["tier"] == "cheap"
        assert row["final_tier"] == "top"
        assert len(backends["top"].calls) == 2


def test_the_escalated_request_is_the_original_one_not_the_review_prompt() -> None:
    log = RequestLog(":memory:")
    app, backends = build(verifying(), {"top": ["VERDICT: FAIL", "proper"]}, log)

    with TestClient(app) as client:
        post(client)

    review, escalation = backends["top"].calls
    assert "ANSWER UNDER REVIEW" in review.messages[-1].content
    # The second call must ask the question, not ask about the review.
    assert escalation.messages[-1].content == "what is 2+2?"


def test_the_counterfactual_is_still_one_call_at_one_tier() -> None:
    log = RequestLog(":memory:")
    app, _ = build(verifying(), {"top": ["VERDICT: FAIL", "proper"]}, log)

    with TestClient(app) as client:
        post(client)
        rows = {r["tier"]: r["cost_usd"] for r in log.query("SELECT tier, cost_usd FROM counterfactuals")}

        # "What would this request have cost had it gone straight to top" means
        # ONE call at top. Charging the baseline for a review it would never
        # have run is how a router flatters itself into a saving.
        assert rows["top"] == pytest.approx(TOP_PER_CALL)
        assert rows["mid"] == pytest.approx(MID_PER_CALL)


def test_a_loop_that_costs_more_than_the_baseline_reports_a_negative_saving() -> None:
    log = RequestLog(":memory:")
    app, _ = build(verifying(), {"top": ["VERDICT: FAIL", "proper"]}, log)

    with TestClient(app) as client:
        post(client)
        report = collect(log)

    baselines = {b.tier: b for b in report.baselines}
    # cheap answered free, then top reviewed AND re-answered: two top calls
    # against a baseline of one. The router lost money on this request and the
    # report has to say so out loud.
    assert report.total_cost_usd == pytest.approx(2 * TOP_PER_CALL)
    assert baselines["top"].savings_usd == pytest.approx(-TOP_PER_CALL)


def test_by_tier_spend_is_the_routed_call_and_not_the_loop() -> None:
    log = RequestLog(":memory:")
    app, _ = build(verifying(), {"top": ["VERDICT: PASS"]}, log)

    with TestClient(app) as client:
        post(client, model="mid")
        report = collect(log)

    by_tier = {row.tier: row for row in report.by_tier}
    # `mid` was paid for one call. The review was bought from `top` and must not
    # appear as mid's spend, or "what is this tier costing us" stops being answerable.
    assert by_tier["mid"].cost_usd == pytest.approx(MID_PER_CALL)
    assert report.total_cost_usd == pytest.approx(MID_PER_CALL + TOP_PER_CALL)


# --- the gate applies to the verifier too -----------------------------------


def test_a_verifier_that_cannot_fit_the_review_is_skipped_and_says_so() -> None:
    log = RequestLog(":memory:")
    # `mid` with a 256-token window cannot hold the reviewer instructions, the
    # transcript, the answer AND the reserved verdict.
    narrow = {**BASE_CONFIG}
    narrow["tiers"] = {**BASE_CONFIG["tiers"], "mid": {**BASE_CONFIG["tiers"]["mid"], "context_window": 256}}
    config = parse_config({**narrow, "verification": {"enabled": True, "verifier_tier": "mid"}})
    app, backends = build(config, {}, log)

    with TestClient(app) as client:
        response = post(client)
        check = verification_row(log)

    assert response.status_code == 200
    assert check["verdict"] == "skipped"
    assert check["reason"].startswith("verifier_ineligible")
    # The detail is the gate's own, so the operator can see which limit bit.
    assert "context_window" in check["reason"] or "exceeds context_window" in check["reason"]
    # And nothing was sent to a backend that would have rejected it.
    assert backends["mid"].calls == []


def test_the_gate_is_evaluated_against_the_review_prompt_not_the_original() -> None:
    log = RequestLog(":memory:")
    narrow = {**BASE_CONFIG}
    narrow["tiers"] = {**BASE_CONFIG["tiers"], "mid": {**BASE_CONFIG["tiers"]["mid"], "context_window": 256}}
    config = parse_config({**narrow, "verification": {"enabled": True, "verifier_tier": "mid"}})
    app, _ = build(config, {}, log)

    with TestClient(app) as client:
        # The ORIGINAL request fits `mid` easily -- it is four words. Only the
        # review prompt, which carries the instructions and the answer, does not.
        # A gate re-run against the original would have let this through.
        post(client)
        check = verification_row(log)

    assert check["verdict"] == "skipped"
    assert check["reason"].startswith("verifier_ineligible")


def test_an_escalation_target_that_could_not_serve_the_request_is_recorded_not_faked() -> None:
    log = RequestLog(":memory:")
    tiers = {
        **BASE_CONFIG["tiers"],
        "tiny": {"backend": "ollama", "model": "tiny", "base_url": "http://x", "context_window": 8},
    }
    config = parse_config(
        {
            **BASE_CONFIG,
            "tiers": tiers,
            "verification": {"enabled": True, "verifier_tier": "top", "escalate_to": "tiny"},
        }
    )
    app, backends = build(config, {"top": ["VERDICT: FAIL - wrong"]}, log)

    with TestClient(app) as client:
        response = post(client)
        check = verification_row(log)

    # Known bad and unfixable. The honest outcome is a recorded failure with the
    # original answer, not a pass and not a 500.
    assert response.status_code == 200
    assert check["verdict"] == "fail"
    assert check["escalated"] == 0
    assert "ineligible" in check["reason"]
    assert backends["tiny"].calls == []


# --- failure modes ----------------------------------------------------------


def test_a_broken_verifier_does_not_break_the_request() -> None:
    log = RequestLog(":memory:")
    config = verifying()

    def factory(tier):
        # Only the verifier is down; the tier that answers must still work, or
        # the test proves nothing about the verifier.
        error = BackendError("verifier down", status=503) if tier.name == "top" else None
        return ScriptedBackend(tier, error=error)

    app = create_app(config, backend_factory=factory, log=log)

    with TestClient(app) as client:
        response = post(client)
        check = verification_row(log)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hello"
    # ERROR, not FAIL: the verifier failing is not the cheap model being wrong,
    # and merging the two would make a broken verifier look like a bad tier.
    assert check["verdict"] == "error"
    assert "verifier down" in check["reason"]
    assert check["escalated"] == 0


def test_a_broken_verifier_escalates_when_the_operator_chose_that() -> None:
    log = RequestLog(":memory:")
    config = verifying(on_verifier_error="escalate")
    backends: dict[str, ScriptedBackend] = {}

    def factory(tier):
        # Only the review call fails; the escalation still has to work.
        error = BackendError("verifier down", status=503) if tier.name == "top" else None
        backend = ScriptedBackend(tier, replies=["proper"], error=error)
        backends[tier.name] = backend
        return backend

    app = create_app(config, backend_factory=factory, log=log)
    with TestClient(app) as client:
        post(client)
        check = verification_row(log)

    assert check["verdict"] == "error"
    # The escalation target is the same broken tier here, so it fails too -- and
    # the row says both things happened rather than only the first.
    assert "escalation failed" in check["reason"]


def test_an_unparseable_verdict_is_counted_and_the_fallback_is_named() -> None:
    log = RequestLog(":memory:")
    app, _ = build(verifying(), {"top": ["looks good to me!"]}, log)

    with TestClient(app) as client:
        response = post(client)
        check = verification_row(log)

    # The default refuses to spend money because a verifier is misconfigured...
    assert check["verdict"] == "pass"
    assert response.json()["choices"][0]["message"]["content"] == "hello"
    # ...but the flag is what stops that pass from being read as a measurement.
    assert check["unparseable"] == 1
    assert "on_unparseable=accept" in check["reason"]


def test_an_unparseable_verdict_escalates_when_the_operator_chose_that() -> None:
    log = RequestLog(":memory:")
    app, _ = build(verifying(on_unparseable="escalate"), {"top": ["no verdict here", "proper"]}, log)

    with TestClient(app) as client:
        response = post(client)
        check = verification_row(log)

    assert check["verdict"] == "fail"
    assert check["unparseable"] == 1
    assert check["escalated"] == 1
    assert response.json()["choices"][0]["message"]["content"] == "proper"


def test_an_upstream_failure_records_a_skip_because_there_was_no_answer_to_review() -> None:
    log = RequestLog(":memory:")
    config = verifying()
    backends: dict[str, ScriptedBackend] = {}

    def factory(tier):
        error = BackendError("cheap exploded", status=502) if tier.name == "cheap" else None
        backend = ScriptedBackend(tier, error=error)
        backends[tier.name] = backend
        return backend

    app = create_app(config, backend_factory=factory, log=log)
    with TestClient(app) as client:
        response = post(client)
        check = verification_row(log)

    assert response.status_code == 502
    assert check["verdict"] == "skipped"
    assert check["reason"] == "upstream_error"
    assert backends["top"].calls == []


# --- what the loop declines to do -------------------------------------------


def test_a_stream_is_never_verified_and_the_skip_is_recorded() -> None:
    log = RequestLog(":memory:")
    app, backends = build(verifying(), {}, log)

    with TestClient(app) as client:
        response = post(client, stream=True)
        assert response.status_code == 200
        check = verification_row(log)

    # Not a gap in the table: by the time a stream could be judged the client has
    # already read it, so the skip is a property of streaming and is named.
    assert check["verdict"] == "skipped"
    assert check["reason"] == "streaming"
    assert backends["top"].calls == []
    # And the client is told as much, rather than left to read a missing
    # header as "nothing to report".
    assert response.headers["X-Router-Verdict"] == "skipped"


def test_the_verifier_is_never_asked_to_review_its_own_answer() -> None:
    log = RequestLog(":memory:")
    app, backends = build(verifying(), {}, log)

    with TestClient(app) as client:
        post(client, model="top")
        check = verification_row(log)

    assert check["verdict"] == "skipped"
    assert check["reason"] == "tier_not_verified: top"
    # One call only: the answer. A self-review measures nothing and costs money.
    assert len(backends["top"].calls) == 1


def test_only_the_named_tiers_are_verified() -> None:
    log = RequestLog(":memory:")
    app, _ = build(verifying(verify_tiers=["cheap"]), {}, log)

    with TestClient(app) as client:
        post(client, model="mid")
        check = verification_row(log)

    assert check["verdict"] == "skipped"
    assert check["reason"] == "tier_not_verified: mid"


def test_an_answer_with_no_text_is_skipped_rather_than_failed() -> None:
    log = RequestLog(":memory:")
    config = verifying()
    backends: dict[str, Any] = {}

    class ToolCallBackend(ScriptedBackend):
        async def complete(self, request):
            self.calls.append(request)
            return BackendResponse(
                body={
                    "id": "x",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {},
                },
                usage=self.usage,
            )

    def factory(tier):
        backend = ToolCallBackend(tier) if tier.name == "cheap" else ScriptedBackend(tier)
        backends[tier.name] = backend
        return backend

    app = create_app(config, backend_factory=factory, log=log)
    with TestClient(app) as client:
        post(client)
        check = verification_row(log)

    assert check["verdict"] == "skipped"
    assert check["reason"] == "no_answer_text"
    assert backends["top"].calls == []


def test_sampling_decides_before_the_verifier_is_paid() -> None:
    from llm_router.backends import build_backends
    from llm_router.verification import Verifier

    log = RequestLog(":memory:")
    config = verifying(sample_rate=0.1)
    # Built here rather than inside the app so the verifier and the request path
    # share one set of fakes, and so the sampling decision is injected rather
    # than hoped for -- a test that leaves this to `random.random()` passes nine
    # times out of ten and is not a test.
    backends = build_backends(config, lambda tier: ScriptedBackend(tier, replies=["VERDICT: PASS"]))
    app = create_app(
        config,
        backend_factory=lambda tier: backends[tier.name],
        log=log,
        verifier=Verifier(config, backends, rng=lambda: 0.9),
    )

    with TestClient(app) as client:
        post(client)
        check = verification_row(log)

    assert check["verdict"] == "skipped"
    assert check["reason"] == "sampled_out"
    assert check["verifier_cost_usd"] == 0.0
    # The coin is tossed BEFORE the review prompt is built and sent, or sampling
    # would save nothing at all.
    assert backends["top"].calls == []


def test_verification_off_writes_no_verification_row_at_all() -> None:
    log = RequestLog(":memory:")
    app, _ = build(parse_config(BASE_CONFIG), {}, log)

    with TestClient(app) as client:
        post(client)

        assert log.query("SELECT * FROM verifications") == []
        assert collect(log).verification is None


# --- the report -------------------------------------------------------------


def test_stats_prints_what_the_loop_added_to_the_bill() -> None:
    log = RequestLog(":memory:")
    app, _ = build(
        verifying(), {"top": ["VERDICT: PASS", "VERDICT: FAIL - wrong", "proper"]}, log
    )

    with TestClient(app) as client:
        post(client)
        post(client)
        report = collect(log)

    loop = report.verification
    assert loop is not None
    assert (loop.verified, loop.passed, loop.failed, loop.escalated) == (2, 1, 1, 1)
    assert loop.fail_rate == pytest.approx(0.5)
    assert loop.verifier_cost_usd == pytest.approx(2 * TOP_PER_CALL)
    assert loop.escalation_cost_usd == pytest.approx(TOP_PER_CALL)

    text = format_text(report)
    assert "verification" in text
    # Both tiers answered free, so the loop IS the bill -- and the share is the
    # number the whole design turns on.
    assert "100.0% of actual spend" in text


def test_stats_says_out_loud_when_the_pass_rate_came_from_a_fallback() -> None:
    log = RequestLog(":memory:")
    app, _ = build(verifying(), {"top": ["no verdict at all"]}, log)

    with TestClient(app) as client:
        post(client)
        text = format_text(collect(log))

    assert "UNPARSEABLE" in text


def test_stats_groups_skips_by_reason_without_their_detail() -> None:
    log = RequestLog(":memory:")
    narrow = {**BASE_CONFIG}
    narrow["tiers"] = {**BASE_CONFIG["tiers"], "mid": {**BASE_CONFIG["tiers"]["mid"], "context_window": 256}}
    config = parse_config({**narrow, "verification": {"enabled": True, "verifier_tier": "mid"}})
    app, _ = build(config, {}, log)

    with TestClient(app) as client:
        post(client)
        post(client, messages=[{"role": "user", "content": "a different question"}])
        report = collect(log)

    # Two skips for the same reason with different token counts in the detail
    # must group as one reason, or the summary is a list of unique strings.
    assert report.verification is not None
    assert report.verification.skips_by_reason == {"verifier_ineligible": 2}


def test_the_fail_rate_is_absent_rather_than_perfect_when_nothing_was_reviewed() -> None:
    log = RequestLog(":memory:")
    app, _ = build(verifying(), {}, log)

    with TestClient(app) as client:
        post(client, stream=True)
        report = collect(log)

    assert report.verification is not None
    # 0/0 is not a clean record.
    assert report.verification.fail_rate is None


def test_healthz_says_whether_the_loop_is_running() -> None:
    log = RequestLog(":memory:")
    app, _ = build(verifying(), {}, log)

    with TestClient(app) as client:
        assert client.get("/healthz").json()["verification"] is True


# --- configuration ----------------------------------------------------------


def test_enabling_verification_without_a_verifier_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="verifier_tier"):
        parse_config({**BASE_CONFIG, "verification": {"enabled": True}})


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"enabled": True, "verifier_tier": "ghost"}, "not a configured tier"),
        ({"enabled": True, "verifier_tier": "top", "escalate_to": "ghost"}, "not a configured tier"),
        ({"enabled": True, "verifier_tier": "top", "verify_tiers": ["ghost"]}, "unknown tier"),
        ({"enabled": True, "verifier_tier": "top", "sample_rate": 1.5}, "sample_rate"),
        ({"enabled": True, "verifier_tier": "top", "on_unparseable": "panic"}, "on_unparseable"),
        ({"enabled": True, "verifier_tier": "top", "nonsense": 1}, "unknown verification fields"),
    ],
)
def test_a_misconfigured_loop_fails_at_load_rather_than_at_the_first_request(settings, message) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config({**BASE_CONFIG, "verification": settings})


def test_the_escalation_target_defaults_to_the_verifier() -> None:
    config = verifying()

    # It has already read the question; asking a third tier would pay for the
    # context twice for no stated reason.
    assert config.verification.escalate_to == "top"


# --- found on the first run against a real model -----------------------------
#
# qwen3.5:4b through Ollama, 2026-09-19. Three requests, three defects, none of
# them reachable with a backend that always answers politely.


class TruncatedBackend(ScriptedBackend):
    """Answers from a script, but reports every answer as cut off by length."""

    async def complete(self, request: ChatCompletionRequest) -> BackendResponse:
        self.calls.append(request)
        content = self.replies.pop(0) if self.replies else self.content
        return BackendResponse(
            body=build_completion(
                model=self.tier.name, content=content, usage=self.usage, finish_reason="length"
            ),
            usage=self.usage,
        )


def test_an_empty_answer_is_a_failure_and_needs_no_reviewer_to_say_so() -> None:
    # Measured: a thinking model spent its whole context reasoning, returned an
    # empty `content`, and the client received a 200 with nothing in it. The
    # loop recorded `no_answer_text` -- a skip designed for tool-call answers --
    # and let it through. An answer that is not there is not a thing to review.
    log = RequestLog(":memory:")
    backends: dict[str, Any] = {}

    def factory(tier):
        if tier.name == "cheap":
            backend = TruncatedBackend(tier, replies=[""])
        else:
            backend = ScriptedBackend(tier, replies=["the proper answer"])
        backends[tier.name] = backend
        return backend

    app = create_app(verifying(), backend_factory=factory, log=log)
    with TestClient(app) as client:
        response = post(client)
        check = verification_row(log)

    assert check["verdict"] == "fail"
    assert "empty answer" in check["reason"]
    assert "length" in check["reason"]
    # One call to `top`: the escalation. No review was bought to discover that
    # nothing had been said.
    assert len(backends["top"].calls) == 1
    assert check["verifier_cost_usd"] == 0.0
    assert check["escalated"] == 1
    assert response.json()["choices"][0]["message"]["content"] == "the proper answer"


def test_an_empty_answer_is_failed_even_when_it_was_sampled_out() -> None:
    # Detecting it is free, so sampling -- which exists to ration a paid review
    # -- has nothing to ration here.
    log = RequestLog(":memory:")
    config = verifying(sample_rate=0.0)
    backends: dict[str, Any] = {}

    def factory(tier):
        backend = (
            TruncatedBackend(tier, replies=[""])
            if tier.name == "cheap"
            else ScriptedBackend(tier, replies=["the proper answer"])
        )
        backends[tier.name] = backend
        return backend

    with TestClient(create_app(config, backend_factory=factory, log=log)) as client:
        post(client)
        check = verification_row(log)
    assert check["verdict"] == "fail"


def test_a_verifier_that_ran_out_of_tokens_is_named_as_such() -> None:
    # Measured: both reviews came back with exactly max_verdict_tokens (200)
    # generated and no verdict, because the model spent them thinking. Recorded
    # as a bare "unparseable verdict", which sends whoever reads it to the
    # regex. The finish reason says what actually happened.
    log = RequestLog(":memory:")

    def factory(tier):
        if tier.name == "top":
            return TruncatedBackend(tier, replies=["Let me think about whether"])
        return ScriptedBackend(tier, replies=["4"])

    with TestClient(create_app(verifying(), backend_factory=factory, log=log)) as client:
        post(client)
        check = verification_row(log)

    assert check["unparseable"] == 1
    assert "max_verdict_tokens" in check["reason"]
    assert str(verifying().verification.max_verdict_tokens) in check["reason"]


def test_the_reasoning_hint_names_the_verifier_backends_own_knob() -> None:
    # Measured 2026-09-19: the first remote judge (openai_compatible, on
    # OpenRouter) ran out of budget and the reason told the operator to set
    # `think: false` -- an Ollama option its provider never reads.
    def reason_with(verifier: str) -> str:
        log = RequestLog(":memory:")
        config = parse_config(
            {**BASE_CONFIG, "verification": {"enabled": True, "verifier_tier": verifier,
                                             "verify_tiers": ["mid"]}}
        )

        def factory(tier):
            if tier.name == verifier:
                return TruncatedBackend(tier, replies=["Let me think about whether"])
            return ScriptedBackend(tier, replies=["4"])

        with TestClient(create_app(config, backend_factory=factory, log=log)) as client:
            post(client, model="mid")
            return verification_row(log)["reason"]

    remote = reason_with("top")
    assert "think: false" not in remote
    assert "reasoning" in remote
    assert "'top'" in remote

    local = reason_with("cheap")
    assert "think: false" in local
    assert "'cheap'" in local
