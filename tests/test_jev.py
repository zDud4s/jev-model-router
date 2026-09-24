"""The Jev judge: a probability becomes a verdict, and nothing invents one.

Two properties carry this suite. The first is that the threshold is applied
where the config says and the probability survives into the reply, because a
recorded run whose verdicts kept their evidence can be re-thresholded afterwards
and one that kept only PASS/FAIL cannot.

The second is the one the project already paid to learn twice: a judge that
cannot answer must not look like a judge that passed. A text verifier reaches
that failure by spending its budget on reasoning and writing no verdict, which
`on_unparseable: accept` then turns into a pass nobody made. Jev cannot be
unparseable -- so the equivalent hole here is a response with no probability in
it, and `test_a_response_with_no_probability_is_an_error_and_never_a_pass` is
the assertion that keeps it shut.

No network: every call goes through `httpx.MockTransport`.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_router.app import create_app
from llm_router.backends.base import BackendError
from llm_router.backends.jev import DEFAULT_INSTRUCTIONS, JevBackend
from llm_router.config import Config, ConfigError, parse_config
from llm_router.db import RequestLog
from llm_router.verification import build_review_request

from conftest import BASE_CONFIG, FakeBackend, make_request

JEV_TIER: dict[str, Any] = {
    "backend": "jev",
    "model": "jev-latest",
    "context_window": 32000,
    # Output is free and the zero is written down rather than omitted, which is
    # the distinction the cache_read bug taught this project to care about.
    "prices": {"input": 0.042, "output": 0.0},
}


def config_with(**tier_overrides: Any) -> Config:
    tiers = {**BASE_CONFIG["tiers"], "judge": {**JEV_TIER, **tier_overrides}}
    return parse_config({**BASE_CONFIG, "tiers": tiers})


def judging(
    *,
    noul: float | None = 0.9,
    usage: dict[str, int] | None = None,
    status: int = 200,
    payload: Any = None,
    seen: list[dict[str, Any]] | None = None,
) -> httpx.AsyncClient:
    """A client that answers one noul question from a script."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        if payload is not None:
            return httpx.Response(status, json=payload)
        answers = {} if noul is None else {"verdict": {"noul": noul}}
        return httpx.Response(
            status,
            json={
                "model": "jev-1.13.0",
                "answers": answers,
                "usage": usage or {"input_tokens": 1000, "output_tokens": 0},
            },
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def a_review(config: Config, answer: str = "the answer under review"):
    return build_review_request(make_request(), answer, config.verification)


# --- the verdict ------------------------------------------------------------


@pytest.mark.parametrize(
    "noul, threshold, expected",
    [
        (0.97, 0.5, "PASS"),
        (0.02, 0.5, "FAIL"),
        # The boundary belongs to PASS, and it is a boundary the operator moved.
        (0.80, 0.80, "PASS"),
        (0.79, 0.80, "FAIL"),
        # A threshold of 0 passes everything and 1 fails everything short of
        # certainty. Both are legitimate settings and neither is special-cased.
        (0.0, 0.0, "PASS"),
        (0.999, 1.0, "FAIL"),
    ],
)
async def test_the_threshold_decides_and_the_config_owns_it(noul, threshold, expected) -> None:
    config = config_with(jev={"threshold": threshold})
    backend = JevBackend(config.tier("judge"), client=judging(noul=noul))

    result = await backend.complete(a_review(config))

    assert result.body["choices"][0]["message"]["content"].startswith(f"VERDICT: {expected}")


async def test_the_probability_survives_the_verdict_so_a_run_can_be_rethresholded() -> None:
    config = config_with()
    backend = JevBackend(config.tier("judge"), client=judging(noul=0.123))

    result = await backend.complete(a_review(config))
    content = result.body["choices"][0]["message"]["content"]

    # The number the decision was made from, not only the decision. A text judge
    # could never hand this back.
    assert "p(correct)=0.123" in content
    assert "threshold=0.50" in content


async def test_the_verdict_is_the_shape_the_loop_already_parses() -> None:
    from llm_router.verification import answer_text, parse_verdict

    config = config_with()
    backend = JevBackend(config.tier("judge"), client=judging(noul=0.10))

    result = await backend.complete(a_review(config))
    passed, reason = parse_verdict(answer_text(result.body))

    # verification.py is untouched by this backend, so the proof that they fit
    # is that its own parser reads the reply and keeps the evidence as a reason.
    assert passed is False
    assert reason is not None and "p(correct)=0.100" in reason


# --- the translation --------------------------------------------------------


async def test_the_reviewers_brief_becomes_the_question_and_the_rest_the_state() -> None:
    config = config_with()
    seen: list[dict[str, Any]] = []
    backend = JevBackend(config.tier("judge"), client=judging(seen=seen))

    await backend.complete(a_review(config, answer="4"))

    body = seen[0]
    # Required by the service, and this is the regression guard for it: the
    # written guide says the endpoint names its own model, so the first version
    # of this backend omitted the field and every live call came back 422.
    assert body["model"] == "jev-latest"
    question = body["questions"]["verdict"]
    assert question["type"] == "noul"
    # The brief travels as the question. It arrived as a system message, and Jev
    # has no roles to put it in.
    assert "Judge only whether the answer is correct" in question["instructions"]
    # What is read, rather than what is asked, is the conversation and answer.
    assert "=== ANSWER UNDER REVIEW ===" in body["state"]
    assert "4" in body["state"]
    assert "Judge only whether the answer is correct" not in body["state"]


async def test_config_instructions_override_the_brief_the_review_carried() -> None:
    config = config_with(jev={"instructions": "Is the arithmetic right?"})
    seen: list[dict[str, Any]] = []
    backend = JevBackend(config.tier("judge"), client=judging(seen=seen))

    await backend.complete(a_review(config))

    assert seen[0]["questions"]["verdict"]["instructions"] == "Is the arithmetic right?"


async def test_a_review_with_no_brief_falls_back_to_the_built_in_question() -> None:
    config = config_with()
    seen: list[dict[str, Any]] = []
    backend = JevBackend(config.tier("judge"), client=judging(seen=seen))

    await backend.complete(make_request(messages=[{"role": "user", "content": "just this"}]))

    assert seen[0]["questions"]["verdict"]["instructions"] == DEFAULT_INSTRUCTIONS
    assert seen[0]["state"] == "just this"


async def test_extra_body_overrides_the_question_when_the_wire_shape_moves() -> None:
    # The escape hatch the module docstring promises: TypeSafe's guide and
    # Cloudflare's model page disagree about the question shape, so a wire
    # change has to be reachable from config rather than from a patch.
    config = config_with(extra_body={"questions": {"verdict": {"instructions": "custom"}}})
    seen: list[dict[str, Any]] = []
    backend = JevBackend(config.tier("judge"), client=judging(seen=seen))

    await backend.complete(a_review(config))

    assert seen[0]["questions"] == {"verdict": {"instructions": "custom"}}
    assert seen[0]["state"]  # the state is still built the same way


async def test_the_question_key_is_used_on_both_sides() -> None:
    config = config_with(jev={"question_key": "ok"})
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"answers": {"ok": {"noul": 0.9}}, "usage": {}})

    backend = JevBackend(
        config.tier("judge"), client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    result = await backend.complete(a_review(config))

    assert "ok" in seen[0]["questions"]
    assert result.body["choices"][0]["message"]["content"].startswith("VERDICT: PASS")


# --- what it costs ----------------------------------------------------------


async def test_usage_is_read_in_jevs_spelling_and_the_free_output_is_a_real_zero() -> None:
    config = config_with()
    backend = JevBackend(
        config.tier("judge"),
        client=judging(usage={"input_tokens": 1500, "output_tokens": 0}),
    )

    result = await backend.complete(a_review(config))

    assert result.usage.prompt_tokens == 1500
    assert result.usage.completion_tokens == 0
    assert result.usage.total_tokens == 1500


# --- the hole this suite exists to keep shut --------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"answers": {}, "usage": {}},
        {"answers": {"verdict": {}}, "usage": {}},
        {"answers": {"verdict": {"choice": "yes"}}, "usage": {}},
        # A boolean is not a probability. Python would happily read True as 1.0
        # and call it certainty, which is a judgement nobody made.
        {"answers": {"verdict": {"noul": True}}, "usage": {}},
        {"answers": {"verdict": {"noul": "0.9"}}, "usage": {}},
        {"not": "the shape at all"},
    ],
)
async def test_a_response_with_no_probability_is_an_error_and_never_a_pass(payload) -> None:
    config = config_with()
    backend = JevBackend(config.tier("judge"), client=judging(payload=payload))

    with pytest.raises(BackendError) as raised:
        await backend.complete(a_review(config))

    # It reaches the loop as an error, where `on_verifier_error` decides. The
    # one outcome that must be unreachable is a silent PASS.
    assert "noul probability" in str(raised.value)


async def test_an_upstream_error_keeps_its_status_for_the_client() -> None:
    config = config_with()
    backend = JevBackend(config.tier("judge"), client=judging(status=429, payload={"e": "slow"}))

    with pytest.raises(BackendError) as raised:
        await backend.complete(a_review(config))

    assert raised.value.status == 429


async def test_a_transport_failure_is_a_backend_error_not_a_traceback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    config = config_with()
    backend = JevBackend(
        config.tier("judge"), client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )

    with pytest.raises(BackendError):
        await backend.complete(a_review(config))


async def test_a_jev_tier_cannot_stream_and_says_so() -> None:
    config = config_with()
    backend = JevBackend(config.tier("judge"), client=judging())

    with pytest.raises(BackendError) as raised:
        async for _ in backend.stream(a_review(config)):
            pass

    assert "cannot stream" in str(raised.value)


# --- config -----------------------------------------------------------------


def test_a_jev_block_on_a_tier_that_cannot_use_it_is_refused() -> None:
    tiers = {**BASE_CONFIG["tiers"]}
    tiers["mid"] = {**tiers["mid"], "jev": {"threshold": 0.9}}

    with pytest.raises(ConfigError) as raised:
        parse_config({**BASE_CONFIG, "tiers": tiers})

    assert "do nothing" in str(raised.value)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_a_threshold_outside_the_unit_interval_is_refused(bad) -> None:
    with pytest.raises(ConfigError):
        config_with(jev={"threshold": bad})


def test_an_unknown_jev_field_is_refused_rather_than_ignored() -> None:
    with pytest.raises(ConfigError):
        config_with(jev={"treshold": 0.9})


def test_a_jev_tier_needs_no_base_url() -> None:
    # One vendor, one endpoint. Still overridable for a gateway that fronts it.
    assert config_with().tier("judge").base_url == "https://api.typesafe.ai/v1"


# --- a judge is not a cheaper way of having answered ------------------------


def test_a_judge_is_never_eligible_to_serve_and_so_never_a_counterfactual() -> None:
    from llm_router.eligibility import RejectionReason, evaluate

    config = config_with()
    result = evaluate(config, make_request())

    # The number this protects: `stats` prints what each tier WOULD have cost,
    # and a judge at $0.042/1M would look like the cheapest way to have answered
    # every request in the log. It was not a way of answering them at all.
    assert not result.is_eligible("judge")
    rejection = next(r for r in result.rejections if r.tier == "judge")
    assert rejection.reason is RejectionReason.CANNOT_GENERATE
    # The tiers that really could have served it are untouched.
    assert result.is_eligible("cheap") and result.is_eligible("top")


def test_the_same_judge_is_eligible_for_the_review_it_exists_to_read() -> None:
    from llm_router.eligibility import check_tier
    from llm_router.tokens import estimate_request_budget

    config = config_with()
    review = a_review(config)

    assert (
        check_tier(
            config.tier("judge"),
            review,
            estimated_tokens=estimate_request_budget(review),
            serving=False,
        )
        is None
    )


def test_a_judge_that_cannot_fit_the_review_is_still_rejected_for_size() -> None:
    from llm_router.eligibility import RejectionReason, check_tier

    # `serving=False` relaxes one rule, not the gate. A review that does not fit
    # must still be a skip rather than a call that will be refused upstream.
    config = config_with(context_window=10)
    review = a_review(config)
    rejection = check_tier(config.tier("judge"), review, estimated_tokens=5000, serving=False)

    assert rejection is not None
    assert rejection.reason is RejectionReason.CONTEXT_WINDOW


@pytest.mark.parametrize(
    "router, message",
    [
        ({"kind": "static", "default_tier": "judge"}, "cannot answer"),
        ({"kind": "static", "default_tier": "cheap", "model_map": {"x": "judge"}}, "cannot answer"),
    ],
)
def test_a_judge_is_refused_anywhere_an_answer_is_expected(router, message) -> None:
    tiers = {**BASE_CONFIG["tiers"], "judge": JEV_TIER}

    with pytest.raises(ConfigError) as raised:
        parse_config({**BASE_CONFIG, "tiers": tiers, "router": router})

    assert message in str(raised.value)


def test_a_judge_verifier_with_no_escalate_to_is_refused_at_startup() -> None:
    # The trap this closes: `escalate_to` defaults to the verifier, because a
    # judge that has read the question is the obvious tier to re-answer it. For
    # a judge that cannot answer anything that default is wrong, and it would
    # only break on the first FAILED verdict -- in production, on the requests
    # that had already gone wrong.
    with pytest.raises(ConfigError) as raised:
        verifying_with_jev(escalate_to=None)

    assert "has no default here" in str(raised.value)


def test_a_judge_named_as_escalate_to_is_refused_with_its_own_message() -> None:
    with pytest.raises(ConfigError) as raised:
        verifying_with_jev(verifier_tier="top", escalate_to="judge")

    assert "escalate_to" in str(raised.value)


# --- end to end, through the loop that was not changed ----------------------


def verifying_with_jev(**verification: Any) -> Config:
    settings = {"enabled": True, "verifier_tier": "judge", "escalate_to": "top"}
    settings.update(verification)
    tiers = {**BASE_CONFIG["tiers"], "judge": JEV_TIER}
    return parse_config({**BASE_CONFIG, "tiers": tiers, "verification": settings})


def build_app(config: Config, log: RequestLog, *, client: httpx.AsyncClient, answers: dict):
    backends: dict[str, Any] = {}

    def factory(tier):
        if tier.backend == "jev":
            backend: Any = JevBackend(tier, client=client)
        else:
            backend = FakeBackend(tier, content=answers.get(tier.name, "hello"))
        backends[tier.name] = backend
        return backend

    return create_app(config, backend_factory=factory, log=log), backends


def post(client: TestClient):
    return client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "what is 2+2?"}]},
    )


def test_jev_fails_a_cheap_answer_and_the_loop_escalates_untouched() -> None:
    log = RequestLog(":memory:")
    config = verifying_with_jev()
    app, _ = build_app(
        config,
        log,
        client=judging(noul=0.04, usage={"input_tokens": 1000, "output_tokens": 0}),
        answers={"top": "the proper answer"},
    )

    with TestClient(app) as http:
        response = post(http)
        row = log.query("SELECT * FROM requests")[0]
        check = log.query("SELECT * FROM verifications")[0]

        # The client never sees the answer the judge rejected.
        assert response.json()["choices"][0]["message"]["content"] == "the proper answer"
        assert check["verdict"] == "fail"
        assert check["verifier_tier"] == "judge"
        # The probability reached the log as the reason, through a parser this
        # change did not touch.
        assert "p(correct)=0.040" in check["reason"]
        assert check["escalated"] == 1 and check["escalated_to"] == "top"
        # 1000 input tokens at $0.042/1M, and not a cent for the output.
        assert check["verifier_cost_usd"] == pytest.approx(1000 * 0.042 / 1_000_000)
        assert row["final_tier"] == "top"

    log.close()


def test_jev_passing_leaves_the_cheap_answer_and_still_bills_the_review() -> None:
    log = RequestLog(":memory:")
    config = verifying_with_jev()
    app, _ = build_app(config, log, client=judging(noul=0.98), answers={})

    with TestClient(app) as http:
        response = post(http)
        check = log.query("SELECT * FROM verifications")[0]

        assert response.json()["choices"][0]["message"]["content"] == "hello"
        assert check["verdict"] == "pass"
        assert check["escalated"] == 0
        # A review that costs nothing is the bug this line exists to catch --
        # cheap is not free, and the savings column has to carry it.
        assert check["verifier_cost_usd"] > 0

    log.close()


def test_a_judge_that_answers_nothing_never_becomes_a_pass_in_the_log() -> None:
    log = RequestLog(":memory:")
    # `on_verifier_error` defaults to `accept`, so the cheap answer stands --
    # but it must stand as a recorded ERROR, not as a verdict.
    config = verifying_with_jev()
    app, _ = build_app(config, log, client=judging(payload={"answers": {}}), answers={})

    with TestClient(app) as http:
        post(http)
        check = log.query("SELECT * FROM verifications")[0]

        assert check["verdict"] == "error"
        assert check["escalated"] == 0

    log.close()


# --- several questions in one call ------------------------------------------
#
# Measured on 434 reviewed answers (227 natural, 207 with one planted defect):
# eight narrow questions asked together and averaged in logit space beat the
# single question by +0.023 AUC, 95% CI [0.008, 0.038] -- and cost one call,
# because the state is read once however many questions ride on it.


def answering(probs: dict[str, float], seen: list[dict[str, Any]] | None = None) -> httpx.AsyncClient:
    """A client that answers each question it is asked from `probs`."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if seen is not None:
            seen.append(body)
        answers = {k: {"type": "noul", "noul": probs[k]} for k in body["questions"] if k in probs}
        return httpx.Response(
            200,
            json={"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 1300, "output_tokens": 0}},
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


MANY = {"questions": {"right": "Is it right?", "coherent": "Does it hang together?"},
        "error_questions": {"flawed": "Does it contain an error?"}}


async def test_every_configured_question_rides_on_one_call_over_the_same_state() -> None:
    config = config_with(jev=MANY)
    seen: list[dict[str, Any]] = []
    backend = JevBackend(config.tier("judge"), client=answering({"right": 0.9, "coherent": 0.9, "flawed": 0.1}, seen))

    await backend.complete(a_review(config, answer="4"))

    assert len(seen) == 1
    assert seen[0]["questions"] == {
        "right": {"type": "noul", "instructions": "Is it right?"},
        "coherent": {"type": "noul", "instructions": "Does it hang together?"},
        "flawed": {"type": "noul", "instructions": "Does it contain an error?"},
    }
    # The state is what a single question reads: the review, not the brief.
    assert "=== ANSWER UNDER REVIEW ===" in seen[0]["state"]


async def test_the_answers_are_averaged_in_logit_space_with_error_questions_flipped() -> None:
    config = config_with(jev={**MANY, "threshold": 0.75})
    probs = {"right": 0.9, "coherent": 0.5, "flawed": 0.2}
    backend = JevBackend(config.tier("judge"), client=answering(probs))

    result = await backend.complete(a_review(config))
    content = result.body["choices"][0]["message"]["content"]

    # logit(0.9)=2.197, logit(0.5)=0, and "flawed" at 0.2 is p(correct)=0.8,
    # logit 1.386. Mean 1.195, back through the sigmoid: 0.768.
    assert content.startswith("VERDICT: PASS")
    assert "p(correct)=0.768" in content
    # The weakest question is named, so a failure says where it came from.
    assert "weakest coherent=0.500" in content


async def test_one_confident_doubt_can_fail_the_answer_on_its_own() -> None:
    config = config_with(jev={**MANY, "threshold": 0.75})
    backend = JevBackend(config.tier("judge"), client=answering({"right": 0.9, "coherent": 0.9, "flawed": 0.97}))

    result = await backend.complete(a_review(config))

    assert result.body["choices"][0]["message"]["content"].startswith("VERDICT: FAIL")


async def test_a_question_left_unanswered_is_an_error_and_never_a_pass() -> None:
    config = config_with(jev=MANY)
    backend = JevBackend(config.tier("judge"), client=answering({"right": 0.99, "coherent": 0.99}))

    with pytest.raises(BackendError, match="flawed"):
        await backend.complete(a_review(config))


@pytest.mark.parametrize(
    "jev, message",
    [
        ({"questions": {}}, "questions"),
        ({"questions": {"a": "  "}}, "empty"),
        ({"questions": {"a": "x"}, "error_questions": {"a": "y"}}, "both"),
        ({"questions": {"a": "x"}, "instructions": "z"}, "instructions"),
        ({"questions": {"a": "x"}, "question_key": "k"}, "question_key"),
        ({"questions": ["a"]}, "mapping"),
    ],
    ids=["empty-map", "empty-text", "key-in-both", "with-instructions", "with-question-key", "not-a-map"],
)
def test_an_ambiguous_question_set_is_refused_at_startup(jev, message) -> None:
    with pytest.raises(ConfigError, match=message):
        config_with(jev=jev)


# --- Jev as a prefilter: it may pass an answer, and never fails one ---------
#
# Measured on the same 434 answers: as the only judge Jev reaches AUC ~0.78
# against an Opus 5 reference, which is not a verdict to escalate on. Asked
# first, and trusted only when it passes, it keeps a share of answers away from
# the expensive judge -- and a Jev that is wrong, unsure or down costs only
# that saving, never a verdict.


def prefiltered(**verification: Any) -> Config:
    settings = {"enabled": True, "verifier_tier": "top", "prefilter_tier": "judge", "escalate_to": "mid"}
    settings.update(verification)
    tiers = {**BASE_CONFIG["tiers"], "judge": JEV_TIER}
    return parse_config({**BASE_CONFIG, "tiers": tiers, "verification": settings})


def test_a_confident_prefilter_pass_never_calls_the_verifier() -> None:
    log = RequestLog(":memory:")
    app, backends = build_app(prefiltered(), log, client=judging(noul=0.97), answers={"top": "VERDICT: FAIL - no"})

    with TestClient(app) as http:
        response = post(http)
        check = log.query("SELECT * FROM verifications")[0]

        assert response.json()["choices"][0]["message"]["content"] == "hello"
        assert backends["top"].calls == []
        assert check["verdict"] == "pass"
        # Who decided, in the column that says who decided.
        assert check["verifier_tier"] == "judge"
        assert "p(correct)=0.970" in check["reason"]
        assert check["verifier_cost_usd"] == pytest.approx(1000 * 0.042 / 1_000_000)

    log.close()


def test_a_prefilter_fail_is_only_a_referral_and_the_verifier_decides() -> None:
    log = RequestLog(":memory:")
    app, backends = build_app(
        prefiltered(), log, client=judging(noul=0.10),
        answers={"top": "VERDICT: FAIL - the sum is wrong", "mid": "the proper answer"},
    )

    with TestClient(app) as http:
        response = post(http)
        check = log.query("SELECT * FROM verifications")[0]

        assert len(backends["top"].calls) == 1
        assert response.json()["choices"][0]["message"]["content"] == "the proper answer"
        assert check["verdict"] == "fail" and check["verifier_tier"] == "top"
        assert check["escalated"] == 1 and check["escalated_to"] == "mid"
        # Both reviews were bought, and both are on the bill.
        top = 100 * 15.0 / 1_000_000 + 50 * 75.0 / 1_000_000
        assert check["verifier_cost_usd"] == pytest.approx(1000 * 0.042 / 1_000_000 + top)
        # What the prefilter said survives beside the verdict, so its leaks and
        # its savings can both be counted from the log later.
        assert "p(correct)=0.100" in check["reason"]
        assert "the sum is wrong" in check["reason"]

    log.close()


def test_the_verifier_can_overrule_a_prefilter_that_doubted() -> None:
    log = RequestLog(":memory:")
    app, _ = build_app(prefiltered(), log, client=judging(noul=0.10), answers={"top": "VERDICT: PASS"})

    with TestClient(app) as http:
        response = post(http)
        check = log.query("SELECT * FROM verifications")[0]

        assert response.json()["choices"][0]["message"]["content"] == "hello"
        assert check["verdict"] == "pass" and check["verifier_tier"] == "top"
        assert check["escalated"] == 0

    log.close()


def test_a_prefilter_that_breaks_hands_the_review_to_the_verifier() -> None:
    log = RequestLog(":memory:")
    app, backends = build_app(
        prefiltered(), log, client=judging(payload={"answers": {}}), answers={"top": "VERDICT: PASS"}
    )

    with TestClient(app) as http:
        post(http)
        check = log.query("SELECT * FROM verifications")[0]

        # Not an ERROR row: the review happened, just not where it was tried first.
        assert len(backends["top"].calls) == 1
        assert check["verdict"] == "pass" and check["verifier_tier"] == "top"
        assert "prefilter" in check["reason"]

    log.close()


@pytest.mark.parametrize(
    "verification, message",
    [
        ({"prefilter_tier": "nowhere"}, "not a configured tier"),
        ({"prefilter_tier": "top"}, "same tier"),
        ({"enabled": False}, "requires"),
    ],
    ids=["unknown", "same-as-verifier", "verification-off"],
)
def test_a_prefilter_that_could_not_work_is_refused_at_startup(verification, message) -> None:
    with pytest.raises(ConfigError, match=message):
        prefiltered(**verification)
