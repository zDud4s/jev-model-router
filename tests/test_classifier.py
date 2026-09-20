"""The difficulty classifier: features, fitting, the split, the router, the report.

The tests that matter most here are not the ones asserting the model works. They
are the ones asserting it cannot pretend to: that the same request produces the
same features through both doors, that a conversation never straddles the split,
that a score nobody acted on is still recorded, and that the wrong split reads
better than the right one on data where the difference is the whole story.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from conftest import BASE_CONFIG, make_request
from llm_router.app import create_app
from llm_router.classifier import (
    MODEL_FORMAT,
    DifficultyModel,
    Example,
    ModelError,
    build_model,
    conversation_key,
    features_from_messages,
    features_from_prompt_text,
    features_from_request,
    fit,
    split_by_group,
)
from llm_router.config import ConfigError, parse_config
from llm_router.db import SCHEMA_VERSION, LogEntry, RequestLog
from llm_router.pricing import Counterfactual
from llm_router.routing import ClassifierRouter, RouteDecision, StaticRouter, build_router
from llm_router.schemas import Usage
from llm_router.stats import collect, format_text
from llm_router.training import TrainingError, format_report, load_rows, train_from_log
from llm_router.verification import VerificationOutcome, Verdict


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def a_model(**overrides) -> DifficultyModel:
    """A hand-built model, so a routing test asserts routing and not fitting."""
    fields = {
        "weights": {"w:integral": 5.0},
        "bias": -3.0,
        "predicts_tier": "cheap",
        "threshold": 0.5,
    }
    fields.update(overrides)
    return DifficultyModel(**fields)


def config_with(**router_overrides):
    router = {"kind": "static", "default_tier": "cheap", "strong_tier": "top"}
    router.update(router_overrides)
    return parse_config({**BASE_CONFIG, "router": router})


def prompt_text(*turns: str, opening: str = "let us begin") -> str:
    """A conversation as the log stores it: the opening fixes the group."""
    messages = [{"role": "user", "content": opening}]
    for index, turn in enumerate(turns):
        if index:
            messages.append({"role": "assistant", "content": "ok"})
        messages.append({"role": "user", "content": turn})
    return json.dumps(messages, sort_keys=True, ensure_ascii=False)


def record(
    log: RequestLog,
    text: str,
    verdict: Verdict | None,
    *,
    opening: str = "let us begin",
    tier: str = "cheap",
    unparseable: bool = False,
    reviewed: bool = True,
    status: int = 200,
    strong_cost: float = 0.01,
    route_cost: float = 0.0,
    score: float | None = None,
    model: str | None = None,
) -> None:
    verification = (
        VerificationOutcome(
            verdict=verdict,
            # None is how the loop records a verdict no judge made: an empty or
            # cut-off answer, failed by a rule for nothing.
            verifier_tier="top" if reviewed else None,
            unparseable=unparseable,
        )
        if verdict is not None
        else None
    )
    log.record(
        LogEntry(
            request_id="req_test",
            prompt_text=prompt_text(text, opening=opening),
            tier=tier,
            router="classifier",
            usage=Usage(prompt_tokens=100, completion_tokens=50),
            cost_usd=route_cost,
            http_status=status,
            counterfactuals=[
                Counterfactual(tier="top", cost_usd=strong_cost, priced=True, eligible=True),
                Counterfactual(tier="cheap", cost_usd=0.0, priced=False, eligible=True),
            ],
            verification=verification,
            route_score=score,
            route_model=model,
        )
    )


def a_corpus(log: RequestLog, *, conversations: int = 12, per_conversation: int = 6) -> None:
    """A corpus where difficulty is NOT in the words -- only the conversation is.

    Each conversation carries a marker word of its own and one label for all of
    its requests. Nothing else distinguishes a failing request from a passing
    one, so a model can only score better than chance by recognising which
    conversation a prompt came from. That is precisely the mistake a by-row
    split rewards, which is what makes this corpus the right one to measure the
    two splits against each other.
    """
    for index in range(conversations):
        marker = f"marker{index}word"
        verdict = Verdict.FAIL if index % 2 == 0 else Verdict.PASS
        for turn in range(per_conversation):
            record(
                log,
                f"please handle the {marker} case number {turn} carefully now",
                verdict,
                opening=f"conversation {index}",
            )


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------


def test_the_same_conversation_produces_the_same_features_through_both_doors():
    # The single most important property in the module: if the serving path and
    # the training path could disagree, a model would be fitted on one thing and
    # asked about another, and the symptom -- scores near the base rate -- looks
    # like a hard problem rather than a bug.
    request = make_request(
        messages=[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "compute the integral of x squared"},
        ]
    )
    stored = json.dumps(
        [m.model_dump(exclude_none=True) for m in request.messages],
        sort_keys=True,
        ensure_ascii=False,
    )
    assert features_from_request(request) == features_from_prompt_text(stored)


def test_words_come_from_the_last_user_turn_and_size_from_the_whole():
    vector = features_from_messages(
        [
            {"role": "user", "content": "earlierword in the history"},
            {"role": "assistant", "content": "noted"},
            {"role": "user", "content": "laterword is the question"},
        ]
    )
    assert "w:laterword" in vector
    assert "w:earlierword" not in vector
    # But the history still counts toward the size signals.
    assert vector["n:log_chars"] > vector["n:log_last_chars"]
    assert vector["n:multi_turn"] == 1.0


def test_a_prompt_that_is_not_a_message_list_is_skipped_rather_than_guessed_at():
    assert features_from_prompt_text("not json at all") is None
    assert features_from_prompt_text("[]") is None
    assert features_from_prompt_text('{"role": "user"}') is None


def test_a_tool_only_request_still_produces_a_vector():
    # No user turn anywhere. Falling through to the last message keeps such a
    # request scoreable instead of collapsing every one of them onto the bias.
    vector = features_from_messages([{"role": "tool", "content": "distinctiveresult here"}])
    assert "w:distinctiveresult" in vector


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


def test_a_prompt_of_entirely_unseen_words_scores_the_base_rate():
    model = a_model(bias=-1.0)
    score = model.score_request(make_request(messages=[{"role": "user", "content": "zzz"}]))
    # Only the numeric features move it at all, and they carry no weight here.
    assert score == pytest.approx(model.base_rate, abs=1e-9)


def test_the_fingerprint_survives_a_round_trip_and_changes_with_the_weights(tmp_path):
    model = a_model()
    path = tmp_path / "model.json"
    model.save(path)
    loaded = DifficultyModel.load(path)
    assert loaded.fingerprint == model.fingerprint
    assert a_model(weights={"w:integral": 4.9}).fingerprint != model.fingerprint


def test_a_model_file_from_a_future_format_is_refused(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"format": MODEL_FORMAT + 1, "predicts_tier": "cheap"}))
    with pytest.raises(ModelError, match="format"):
        DifficultyModel.load(path)


def test_a_missing_model_file_names_the_command_that_makes_one(tmp_path):
    with pytest.raises(ModelError, match="llm-router train"):
        DifficultyModel.load(tmp_path / "absent.json")


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------


def _examples(pairs, *, group_prefix="g") -> list[Example]:
    return [
        Example(
            vector=features_from_messages([{"role": "user", "content": text}]),
            label=label,
            group=f"{group_prefix}{index}",
        )
        for index, (text, label) in enumerate(pairs)
    ]


def test_it_learns_a_signal_that_is_actually_there():
    pairs = []
    for index in range(30):
        pairs.append((f"prove the integral converges for case {index}", 1))
        pairs.append((f"say hello to the user number {index}", 0))
    weights, bias = fit(_examples(pairs), seed=0)
    model = DifficultyModel(weights=weights, bias=bias, predicts_tier="cheap")
    hard = model.score(features_from_messages([{"role": "user", "content": "prove the integral converges for case 99"}]))
    easy = model.score(features_from_messages([{"role": "user", "content": "say hello to the user number 99"}]))
    assert hard > 0.5 > easy


def test_the_minority_class_is_not_optimised_away():
    # Nine easy prompts for every hard one. An unweighted fit answers "never
    # fails" to everything, scores 90% accuracy, and routes nothing anywhere --
    # which is why `balance` exists and why it is on.
    pairs = [(f"prove the integral converges for case {i}", 1) for i in range(6)]
    pairs += [(f"say hello to the user number {i}", 0) for i in range(54)]
    weights, bias = fit(_examples(pairs), seed=0)
    model = DifficultyModel(weights=weights, bias=bias, predicts_tier="cheap")
    assert model.score(features_from_messages([{"role": "user", "content": "prove the integral converges for case 7"}])) > 0.5


def test_the_same_corpus_fits_the_same_weights_every_time():
    # Not a style preference. Python randomises string hashing per process, so a
    # feature vector built from a `set` iterates differently in every run, the
    # fit sums the same floats in a different order, and identical data produces
    # a different fingerprint -- measured, before the vectors were sorted, as
    # 232c6cb5a88a one run and 35b0f95c0bb1 the next. A fingerprint that is not
    # reproducible cannot identify the weights a score came from, which is the
    # only job it has.
    pairs = [(f"prove the integral converges {i}", 1) for i in range(20)]
    pairs += [(f"say hello politely {i}", 0) for i in range(20)]
    first = build_model(_examples(pairs), predicts_tier="cheap")
    second = build_model(_examples(pairs), predicts_tier="cheap")
    assert first.fingerprint == second.fingerprint
    assert list(first.to_dict()["weights"]) == sorted(first.weights)


def test_a_word_seen_once_is_pruned_rather_than_fitted():
    pairs = [(f"routine request number {i}", i % 2) for i in range(40)]
    pairs.append(("routine request number 40 with hapaxlegomenon", 1))
    weights, _ = fit(_examples(pairs), min_df=3, seed=0)
    assert "w:hapaxlegomenon" not in weights


# --------------------------------------------------------------------------
# the split
# --------------------------------------------------------------------------


def test_a_conversation_never_straddles_the_split():
    examples = [
        Example(vector={}, label=index % 2, group=f"conv{index % 10}") for index in range(200)
    ]
    train, test = split_by_group(examples)
    assert {e.group for e in train}.isdisjoint({e.group for e in test})


def test_the_split_is_stable_when_the_corpus_grows():
    # Hashing the group rather than shuffling a list is what makes two training
    # runs a week apart comparable instead of merely similar.
    first = [Example(vector={}, label=0, group=f"conv{i}") for i in range(50)]
    grown = first + [Example(vector={}, label=0, group=f"conv{i}") for i in range(50, 80)]
    held_before = {e.group for e in split_by_group(first)[1]}
    held_after = {e.group for e in split_by_group(grown)[1]}
    assert held_before <= held_after


def test_requests_from_one_conversation_share_a_group_key():
    a = prompt_text("first question", opening="shared opening")
    b = prompt_text("second question", opening="shared opening")
    c = prompt_text("first question", opening="different opening")
    assert conversation_key(a) == conversation_key(b)
    assert conversation_key(a) != conversation_key(c)


# --------------------------------------------------------------------------
# training from the log
# --------------------------------------------------------------------------


@pytest.fixture
def corpus_log():
    log = RequestLog(":memory:", store_prompts=True)
    yield log
    log.close()


def test_a_log_that_stored_no_prompts_says_exactly_that(corpus_log):
    silent = RequestLog(":memory:", store_prompts=False)
    try:
        for _ in range(10):
            record(silent, "some question", Verdict.PASS)
        with pytest.raises(TrainingError, match="store_prompts"):
            load_rows(silent, predicts_tier="cheap", strong_tier="top")
    finally:
        silent.close()


def test_an_empty_log_says_how_labels_are_produced(corpus_log):
    with pytest.raises(TrainingError, match="verification loop"):
        load_rows(corpus_log, predicts_tier="cheap", strong_tier="top")


def test_a_verdict_the_fallback_decided_is_not_a_label(corpus_log):
    record(corpus_log, "genuinely reviewed request", Verdict.PASS)
    record(corpus_log, "verifier returned prose", Verdict.PASS, unparseable=True)
    rows = load_rows(corpus_log, predicts_tier="cheap", strong_tier="top")
    assert len(rows) == 1


def test_only_rows_served_by_the_modelled_tier_are_used(corpus_log):
    record(corpus_log, "answered by the cheap tier", Verdict.PASS, tier="cheap")
    record(corpus_log, "answered by the middle tier", Verdict.FAIL, tier="mid")
    rows = load_rows(corpus_log, predicts_tier="cheap", strong_tier="top")
    assert len(rows) == 1
    assert rows[0].example.label == 0


def test_a_failed_request_has_no_answer_to_have_judged(corpus_log):
    record(corpus_log, "this one errored", Verdict.PASS, status=502)
    with pytest.raises(TrainingError):
        load_rows(corpus_log, predicts_tier="cheap", strong_tier="top")


def test_one_class_is_refused_with_the_router_that_is_already_correct(corpus_log):
    for index in range(60):
        record(corpus_log, f"everything passes here {index}", Verdict.PASS, opening=f"c{index}")
    with pytest.raises(TrainingError, match="always cheap"):
        train_from_log(corpus_log, predicts_tier="cheap", strong_tier="top")


def test_too_few_labels_is_refused_rather_than_memorised(corpus_log):
    for index in range(10):
        verdict = Verdict.FAIL if index % 2 else Verdict.PASS
        record(corpus_log, f"request {index}", verdict, opening=f"c{index}")
    with pytest.raises(TrainingError, match="memorises"):
        train_from_log(corpus_log, predicts_tier="cheap", strong_tier="top")


def test_the_wrong_split_reads_better_than_the_right_one(corpus_log):
    # The expensive lesson, measured. On a corpus where only the conversation
    # is learnable, the by-row split scores well and the grouped split does not.
    # Both numbers come from the same data and the same fitting procedure; the
    # only difference is which rows were held out.
    a_corpus(corpus_log)
    report = train_from_log(corpus_log, predicts_tier="cheap", strong_tier="top", min_examples=40)
    assert report.ungrouped.accuracy > report.grouped.accuracy + 0.2
    assert "the model recognising conversations" in format_report(report)


def test_the_shipped_model_is_the_one_the_report_describes(corpus_log):
    a_corpus(corpus_log)
    report = train_from_log(corpus_log, predicts_tier="cheap", strong_tier="top", min_examples=40)
    # Refitting on the full corpus after evaluating would leave a report
    # attached to weights nobody measured.
    assert report.model.examples == report.train_size
    assert report.model.metrics["grouped"]["accuracy"] == report.grouped.accuracy


def test_the_report_keeps_money_and_bad_answers_in_separate_columns(corpus_log):
    a_corpus(corpus_log)
    report = train_from_log(corpus_log, predicts_tier="cheap", strong_tier="top", min_examples=40)
    names = [p.name for p in report.policies]
    assert "always cheap, no verifier" in names
    assert "always strong" in names
    floor = next(p for p in report.policies if p.name == "always cheap, no verifier")
    # The cheapest policy is also the one that ships every failure. If a report
    # ever shows it winning outright, the report has stopped counting something.
    assert floor.cost_usd == 0.0
    assert floor.bad_answers > 0
    text = format_report(report)
    assert "will not invent a price for it" in text


def test_the_threshold_sweep_is_priced_at_every_step(corpus_log):
    a_corpus(corpus_log)
    report = train_from_log(corpus_log, predicts_tier="cheap", strong_tier="top", min_examples=40)
    # The points are shares of the training split's own scores, not a fixed
    # grid, so two shares that land on equal scores collapse into one point
    # instead of printing the same policy twice under different numbers.
    assert 2 <= len(report.sweep) <= 6
    assert any(threshold == report.model.threshold for threshold, _, _ in report.sweep)
    # Escalating more can only cost more and ship fewer bad answers; a sweep
    # that is not monotone in both is an arithmetic error somewhere.
    costs = [cost for _, cost, _ in report.sweep]
    bad = [count for _, _, count in report.sweep]
    assert costs == sorted(costs, reverse=True)
    assert bad == sorted(bad)


# --------------------------------------------------------------------------
# the router
# --------------------------------------------------------------------------


def test_a_prompt_the_model_dislikes_goes_to_the_strong_tier():
    router = ClassifierRouter(config_with(), a_model())
    decision = router.decide(
        make_request(messages=[{"role": "user", "content": "the integral please"}]),
        ["cheap", "mid", "top"],
    )
    assert decision.tier == "top"
    assert decision.score > 0.5
    assert decision.model == a_model().fingerprint


def test_an_ordinary_prompt_stays_cheap_and_is_still_scored():
    router = ClassifierRouter(config_with(), a_model())
    decision = router.decide(make_request(), ["cheap", "mid", "top"])
    assert decision.tier == "cheap"
    assert decision.score is not None and decision.score < 0.5


def test_an_explicit_request_is_never_overruled_and_never_scored():
    router = ClassifierRouter(config_with(), a_model())
    decision = router.decide(
        make_request(model="mid", messages=[{"role": "user", "content": "the integral please"}]),
        ["cheap", "mid", "top"],
    )
    assert decision.tier == "mid"
    # None rather than a score nobody acted on: the row must not read as though
    # the model chose `mid`.
    assert decision.score is None


def test_a_score_that_could_not_be_acted_on_is_recorded_anyway():
    # `top` filtered out by the gate. The request stays cheap -- and the fact
    # that the model wanted to escalate stays countable.
    router = ClassifierRouter(config_with(), a_model())
    decision = router.decide(
        make_request(messages=[{"role": "user", "content": "the integral please"}]),
        ["cheap"],
    )
    assert decision.tier == "cheap"
    assert decision.score > 0.5
    assert "ineligible" in decision.reason


def test_a_request_the_cheap_tier_cannot_take_is_not_scored():
    router = ClassifierRouter(config_with(), a_model())
    decision = router.decide(
        make_request(messages=[{"role": "user", "content": "the integral please"}]),
        ["mid", "top"],
    )
    assert decision.tier == "top"
    assert decision.score is None


def test_exploration_keeps_a_share_of_escalations_cheap():
    config = config_with(explore_rate=0.05)
    hard = make_request(messages=[{"role": "user", "content": "the integral please"}])

    explored = ClassifierRouter(config, a_model(), rng=lambda: 0.01).decide(hard, ["cheap", "top"])
    assert explored.tier == "cheap"
    assert explored.reason.startswith("explore")
    assert explored.score > 0.5

    obeyed = ClassifierRouter(config, a_model(), rng=lambda: 0.99).decide(hard, ["cheap", "top"])
    assert obeyed.tier == "top"


def test_exploration_is_off_unless_asked_for():
    router = ClassifierRouter(config_with(), a_model(), rng=lambda: 0.0)
    decision = router.decide(
        make_request(messages=[{"role": "user", "content": "the integral please"}]),
        ["cheap", "top"],
    )
    assert decision.tier == "top"


def test_a_model_trained_on_another_tier_is_refused():
    with pytest.raises(ConfigError, match="says nothing about another"):
        ClassifierRouter(config_with(), a_model(predicts_tier="mid"))


def test_the_config_threshold_overrides_the_model_s_own():
    router = ClassifierRouter(config_with(threshold=0.99), a_model())
    decision = router.decide(
        make_request(messages=[{"role": "user", "content": "the integral please"}]),
        ["cheap", "top"],
    )
    assert decision.tier == "cheap"
    assert router.threshold == 0.99


def test_a_classifier_router_with_no_model_refuses_to_start(tmp_path):
    config = parse_config(
        {
            **BASE_CONFIG,
            "router": {
                "kind": "classifier",
                "default_tier": "cheap",
                "strong_tier": "top",
                "model_path": str(tmp_path / "absent.json"),
            },
        }
    )
    # Not a fallback to static: a router that looks like it is classifying and
    # is not would be indistinguishable from a working one in every report.
    with pytest.raises(ConfigError, match="llm-router train"):
        build_router(config)


def test_a_trained_model_round_trips_into_a_running_router(tmp_path):
    pairs = [(f"prove the integral converges {i}", 1) for i in range(20)]
    pairs += [(f"say hello politely {i}", 0) for i in range(20)]
    model = build_model(_examples(pairs), predicts_tier="cheap")
    path = tmp_path / "model.json"
    model.save(path)
    config = parse_config(
        {
            **BASE_CONFIG,
            "router": {
                "kind": "classifier",
                "default_tier": "cheap",
                "strong_tier": "top",
                "model_path": str(path),
            },
        }
    )
    router = build_router(config)
    assert router.name == "classifier"
    assert router.model.fingerprint == model.fingerprint


def test_the_static_router_still_answers_the_same_questions():
    router = StaticRouter(config_with())
    assert router.decide(make_request(), ["cheap", "top"]) == RouteDecision(
        tier="cheap", reason="default"
    )
    assert router.decide(make_request(model="top"), ["cheap", "top"]).reason == "requested"
    assert router.decide(make_request(), ["top"]).tier == "top"


def test_the_router_never_returns_a_tier_the_gate_removed():
    router = ClassifierRouter(config_with(), a_model())
    for candidates in (["cheap"], ["mid"], ["top"], ["cheap", "top"], ["mid", "top"]):
        for text in ("the integral please", "hello"):
            decision = router.decide(
                make_request(messages=[{"role": "user", "content": text}]), candidates
            )
            assert decision.tier in candidates


# --------------------------------------------------------------------------
# the request path
# --------------------------------------------------------------------------


@pytest.fixture
def scored_app(backend_factory, fake_backends):
    log = RequestLog(":memory:")
    config = config_with()
    app = create_app(
        config, backend_factory=backend_factory, log=log, router=ClassifierRouter(config, a_model())
    )
    yield app, log, fake_backends


def test_the_score_reaches_the_log_row_and_the_response_header(scored_app):
    app, log, backends = scored_app
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "the integral please"}]},
        )
        # Read inside the block: leaving it closes the log with the app.
        row = log.query("SELECT tier, route_score, route_model, route_reason FROM requests")[0]
    assert response.status_code == 200
    assert response.headers["X-Router-Tier"] == "top"
    assert float(response.headers["X-Router-Score"]) > 0.5
    assert row["tier"] == "top"
    assert row["route_score"] > 0.5
    assert row["route_model"] == a_model().fingerprint
    assert ">=" in row["route_reason"]
    assert backends["top"].calls and not backends["cheap"].calls


def test_an_unscored_request_stores_null_and_not_zero(scored_app):
    # NULL is the fact. A 0.0 in this column would read as "the model was sure
    # it was easy", which is a different claim from "nothing scored it".
    app, log, _ = scored_app
    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "mid", "messages": [{"role": "user", "content": "the integral please"}]},
        )
        row = log.query("SELECT route_score, route_model FROM requests")[0]
    assert row["route_score"] is None
    assert row["route_model"] is None


def test_healthz_publishes_the_deployed_model(scored_app):
    app, _, _ = scored_app
    with TestClient(app) as client:
        body = client.get("/healthz").json()
    assert body["router"] == "classifier"
    assert body["classifier"]["fingerprint"] == a_model().fingerprint
    assert body["classifier"]["predicts_tier"] == "cheap"
    assert body["schema_version"] == SCHEMA_VERSION


def test_the_static_path_writes_no_score_at_all(backend_factory):
    log = RequestLog(":memory:")
    app = create_app(config_with(), backend_factory=backend_factory, log=log)
    with TestClient(app) as client:
        client.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
        row = log.query("SELECT route_score, route_reason FROM requests")[0]
    assert row["route_score"] is None
    assert row["route_reason"] == "default"


# --------------------------------------------------------------------------
# reading it back
# --------------------------------------------------------------------------


def test_a_log_with_no_scores_prints_no_classifier_block(corpus_log):
    record(corpus_log, "unscored request", Verdict.PASS)
    stats = collect(corpus_log)
    assert stats.classifier is None
    assert "classifier" not in format_text(stats)


def test_the_bands_report_the_actual_failure_rate_per_score(corpus_log):
    for _ in range(4):
        record(corpus_log, "easy request", Verdict.PASS, score=0.1, model="abc123")
    for _ in range(2):
        record(corpus_log, "harder request", Verdict.FAIL, score=0.9, model="abc123")
    stats = collect(corpus_log)
    assert stats.classifier.scored == 6
    assert stats.classifier.reviewed == 6
    assert dict((label, (rows, fails)) for label, rows, fails in stats.classifier.bands) == {
        "0.0-0.2": (4, 0),
        "0.8-1.0": (2, 2),
    }
    assert stats.classifier.separation > 0


def test_scores_that_were_never_reviewed_are_named_rather_than_dropped(corpus_log):
    record(corpus_log, "sent to the strong tier", None, tier="top", score=0.9, model="abc123")
    record(corpus_log, "kept cheap", Verdict.PASS, score=0.1, model="abc123")
    stats = collect(corpus_log)
    assert stats.classifier.unreviewed == 1
    text = format_text(stats)
    assert "cannot contradict the model" in text
    # And with exploration off, the report says why that number will keep growing.
    assert "explore_rate is 0" in text


def test_a_model_that_scores_backwards_is_called_out(corpus_log):
    record(corpus_log, "scored easy and failed", Verdict.FAIL, score=0.1, model="abc123")
    record(corpus_log, "scored hard and passed", Verdict.PASS, score=0.9, model="abc123")
    stats = collect(corpus_log)
    assert stats.classifier.separation < 0
    assert "BACKWARDS OR NOISE" in format_text(stats)


def test_two_models_in_one_column_are_not_averaged_quietly(corpus_log):
    record(corpus_log, "scored by the old model", Verdict.PASS, score=0.2, model="old111")
    record(corpus_log, "scored by the new model", Verdict.FAIL, score=0.8, model="new222")
    stats = collect(corpus_log)
    assert set(stats.classifier.models) == {"old111", "new222"}
    assert "MORE THAN ONE MODEL" in format_text(stats)


def test_an_unparseable_verdict_is_not_evidence_about_the_score(corpus_log):
    record(corpus_log, "fallback decided this", Verdict.PASS, score=0.2, model="abc123", unparseable=True)
    stats = collect(corpus_log)
    assert stats.classifier.scored == 1
    assert stats.classifier.reviewed == 0
    assert stats.classifier.bands == []


def test_a_target_in_traffic_and_a_target_in_quality_cannot_both_be_set(corpus_log):
    # They set the same threshold from opposite ends. Honouring one silently
    # would give the operator a policy they did not ask for and no sign of it.
    a_corpus(corpus_log)
    with pytest.raises(TrainingError, match="not both"):
        train_from_log(
            corpus_log,
            predicts_tier="cheap",
            strong_tier="top",
            min_examples=40,
            target_escalation=0.2,
            target_recall=0.8,
        )


def test_a_corpus_of_failures_no_judge_ever_saw_is_called_out(corpus_log):
    # The verification loop fails an empty or cut-off answer without paying a
    # judge. Those verdicts are true, and they are also the cheapest thing in
    # the system to detect -- so a classifier fitted mostly on them is being
    # trained to predict something already handled without it.
    for i in range(60):
        record(corpus_log, f"short {i}", Verdict.PASS, opening=f"c{i}")
    for i in range(20):
        record(
            corpus_log,
            f"a much longer question {i} " * 20,
            Verdict.FAIL,
            opening=f"u{i}",
            reviewed=False,
        )
    for i in range(4):
        record(corpus_log, f"wrong but finished {i} " * 20, Verdict.FAIL, opening=f"w{i}")

    report = train_from_log(
        corpus_log, predicts_tier="cheap", strong_tier="top", min_examples=40
    )

    assert any("failed without a review" in w for w in report.warnings)
    assert any("output budget" in w for w in report.warnings)


def test_a_corpus_a_judge_actually_read_draws_no_such_warning(corpus_log):
    for i in range(60):
        record(corpus_log, f"short {i}", Verdict.PASS, opening=f"c{i}")
    for i in range(24):
        record(corpus_log, f"a much longer question {i} " * 20, Verdict.FAIL, opening=f"w{i}")

    report = train_from_log(
        corpus_log, predicts_tier="cheap", strong_tier="top", min_examples=40
    )

    assert not any("failed without a review" in w for w in report.warnings)


def test_a_price_for_a_bad_answer_turns_the_table_into_a_decision(corpus_log):
    # The report refuses to invent what a wrong answer costs. Supplied with the
    # number, it does the arithmetic and names the winner -- and says whose
    # number it was, because the ranking is worth exactly as much as that.
    a_corpus(corpus_log)
    priced = train_from_log(
        corpus_log,
        predicts_tier="cheap",
        strong_tier="top",
        min_examples=40,
        bad_answer_cost=0.05,
    )
    unpriced = train_from_log(
        corpus_log, predicts_tier="cheap", strong_tier="top", min_examples=40
    )

    text = format_report(priced)
    assert "<- cheapest" in text
    assert "the price YOU gave" in text
    cheapest = min(p.cost_usd + p.bad_answers * 0.05 for p in priced.policies)
    marked = [line for line in text.splitlines() if "<- cheapest" in line]
    assert any(f"${cheapest:>11.6f}" in line for line in marked)

    # Without it, nothing is ranked and the refusal is still printed.
    plain = format_report(unpriced)
    assert "<- cheapest" not in plain
    assert "will not invent a price for it" in plain
