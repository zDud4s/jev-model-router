"""Labels from an answer key: graded, written like traffic, and kept apart from it."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from llm_router.benchmark import INSTRUCTION, Item, final_answer, grade, label, load_gsm8k
from llm_router.config import parse_config
from llm_router.db import RequestLog
from llm_router.training import train_from_log

from conftest import BASE_CONFIG, FakeBackend


def test_the_committed_answer_is_the_last_answer_line() -> None:
    assert final_answer("x = 3\nANSWER: 1,250") == Decimal("1250")
    assert final_answer("ANSWER: 4\nwait, no.\nANSWER: $5.") == Decimal("5")
    assert final_answer("**ANSWER:** 18") == Decimal("18")


def test_a_reply_that_never_answers_fails_rather_than_being_guessed() -> None:
    # Grading "the last number in the text" would score this a pass by luck.
    ok, reason = grade("so the total is 18 eggs", "18")

    assert not ok
    assert "no ANSWER line" in reason


def test_numbers_are_compared_as_numbers() -> None:
    assert grade("ANSWER: 18.0", "18")[0]
    assert not grade("ANSWER: 17", "18")[0]


def test_gsm8k_answers_are_read_from_the_last_line(tmp_path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text(
        json.dumps({"question": "q", "answer": "work\n#### 1,000"}) + "\n", encoding="utf-8"
    )

    assert load_gsm8k(path) == [Item("q", "1,000")]


# --- the run ---------------------------------------------------------------------


class AnswerKeyBackend(FakeBackend):
    """Answers right for questions mentioning `easy`, wrong otherwise."""

    async def complete(self, request):
        text = request.messages[-1].content
        self.content = "ANSWER: 1" if "easy" in text else "ANSWER: 2"
        return await super().complete(request)


def _items(n: int) -> list[Item]:
    # Varying length, so the features have something to separate.
    return [
        Item("easy " * (1 + i % 3), "1") if i % 4 else Item(f"hard {i} " * 40, "1")
        for i in range(n)
    ]


def _label(tmp_path, items, **kwargs):
    config = parse_config({**BASE_CONFIG, "router": {"kind": "static", "default_tier": "cheap", "strong_tier": "top"}})
    db = str(tmp_path / "bench.db")
    report = label(
        config,
        items,
        db=db,
        tier="cheap",
        source="gsm8k",
        backend_factory=lambda tier: AnswerKeyBackend(tier),
        **kwargs,
    )
    return report, db


def test_each_answer_becomes_a_labelled_row_that_says_where_its_label_came_from(tmp_path) -> None:
    report, db = _label(tmp_path, [Item("easy one", "1"), Item("hard one", "1")])

    assert (report.passed, report.failed) == (1, 1)
    log = RequestLog(db)
    rows = log.query(
        "SELECT r.tier, r.prompt_text, v.verdict, v.verifier_tier FROM requests r "
        "JOIN verifications v ON v.request_row_id = r.id ORDER BY r.id"
    )
    assert [(r["tier"], r["verdict"], r["verifier_tier"]) for r in rows] == [
        ("cheap", "pass", "ground_truth:gsm8k"),
        ("cheap", "fail", "ground_truth:gsm8k"),
    ]
    # Stored, and in the router's own shape, or training could not read it.
    assert json.loads(rows[0]["prompt_text"]) == [{"role": "user", "content": "easy one" + INSTRUCTION}]


def test_a_second_run_resumes_instead_of_asking_again(tmp_path) -> None:
    items = [Item("easy a", "1"), Item("easy b", "1"), Item("easy c", "1")]
    _label(tmp_path, items, limit=2)

    report, db = _label(tmp_path, items)

    assert (report.asked, report.skipped_done) == (1, 2)
    assert len(RequestLog(db).query("SELECT id FROM requests")) == 3


def test_the_serving_log_is_refused(tmp_path) -> None:
    serving = str(tmp_path / "serving.db")
    config = parse_config({**BASE_CONFIG, "log": {"path": serving}})

    with pytest.raises(ValueError, match="serving log"):
        label(config, [], db=serving, tier="cheap", source="gsm8k")


def test_a_labelled_database_trains(tmp_path) -> None:
    _, db = _label(tmp_path, _items(80))

    report = train_from_log(
        RequestLog(db), predicts_tier="cheap", strong_tier="top", judged_by="ground_truth:gsm8k"
    )

    assert report.rows == 80
    assert report.model.judged_by == "ground_truth:gsm8k"


# --- unfinished is not the same failure as wrong ----------------------------
#
# 400 GSM8K questions through qwen3.5:4b, 2026-09-20: 70 failures, 60 of them
# stopped at exactly the tier's 2048-token budget with no answer stated. Both
# kinds had been recorded as "the reply states no ANSWER line", so the corpus
# read as a hard benchmark when most of it was one number in the config.


class CutOffBackend(FakeBackend):
    """Answers `easy` questions; runs out of budget on everything else."""

    async def complete(self, request):
        text = request.messages[-1].content
        if "easy" in text:
            self.content = "ANSWER: 1"
            return await super().complete(request)
        self.content = "First I work out how many there are, which means"
        response = await super().complete(request)
        response.body["choices"][0]["finish_reason"] = "length"
        return response


def test_an_unfinished_reply_is_graded_apart_from_a_wrong_one() -> None:
    cut_off, reason = grade("I start by adding", "4", "length")
    silent, quiet_reason = grade("The answer is obvious.", "4", "stop")

    assert cut_off is False and silent is False
    assert "unfinished" in reason and "length" in reason
    assert "unfinished" not in quiet_reason


def test_a_corpus_that_is_mostly_budget_says_so_before_it_is_trained_on(tmp_path) -> None:
    from llm_router.benchmark import format_label_report

    config = parse_config(
        {**BASE_CONFIG, "router": {"kind": "static", "default_tier": "cheap",
                                   "strong_tier": "top"}}
    )
    items = [Item("easy one", "1"), Item("long one", "1"), Item("long two", "1")]
    report = label(
        config,
        items,
        db=str(tmp_path / "bench.db"),
        tier="cheap",
        source="gsm8k",
        backend_factory=lambda tier: CutOffBackend(tier),
    )

    assert (report.failed, report.unfinished) == (2, 2)
    text = format_label_report(report, db="bench.db")
    assert "UNFINISHED rather than wrong" in text
    # The advice is the point: fix the run, do not train on it.
    assert "raise the tier's output budget, and check the temperature" in text


def test_a_label_is_drawn_at_temperature_zero_unless_asked_otherwise(tmp_path) -> None:
    # Measured: at the provider's default of 0.8, asking the same 60 questions
    # again recovered 27 of them inside the original budget. A label drawn hot
    # records what the sample did, which is not what the classifier is for.
    config = parse_config(
        {**BASE_CONFIG, "router": {"kind": "static", "default_tier": "cheap",
                                   "strong_tier": "top"}}
    )
    def asked_at(name: str, **kwargs) -> list[float | None]:
        seen: list[FakeBackend] = []

        def factory(tier):
            seen.append(AnswerKeyBackend(tier))
            return seen[-1]

        label(
            config,
            [Item("easy one", "1")],
            db=str(tmp_path / name),
            tier="cheap",
            source="gsm8k",
            backend_factory=factory,
            **kwargs,
        )
        return [call.temperature for backend in seen for call in backend.calls]

    assert asked_at("zero.db") == [0.0]
    # None leaves it to the provider, for whoever wants the production spread.
    assert asked_at("hot.db", temperature=None) == [None]
