"""Labels from questions whose answers are known, instead of from a judge.

The classifier learns from `pass`/`fail` verdicts, and in production those come
from the verifier: a paid call per label. A question set with an answer key
labels the cheap tier's answers for nothing -- and more reliably than a judge,
which was measured missing arithmetic when it could not afford to reason.

Each question goes through the router itself, in process, with an explicit
`model:` so the tier is the one asked for. The row it writes is therefore the
row real traffic would write: same `prompt_text`, same costs, same
counterfactuals. Only the verdict differs in origin, and it says so:
`verifier_tier` is `ground_truth:<source>`, never a tier name.

What this does NOT give is a model of *your* traffic. A classifier trained here
learns what makes one grade-school word problem harder than another. It tells
you whether the features carry any signal at all; routing real requests on it
is a different claim, and only real verdicts can support it.

Rows go to their own database, never the serving log. Mixed in, they would be
counted by `stats` as traffic and as savings.

Labelling asks at temperature 0 unless told otherwise, and that default was
bought the expensive way. The first 400-question run used the provider's
default -- 0.8 on Ollama -- and 60 answers ran past the output budget without
finishing. Asked again, 27 of those same questions were answered in under 2048
tokens, several in under 300: nothing about the question had changed, only the
sample. A label drawn at temperature 0.8 records what the dice did, and a
classifier fitted on those labels is being asked to predict a coin from the
text of the question.

If your production traffic runs hot, that variance does not disappear -- it
becomes a ceiling on what any router reading only the prompt can achieve, and
the honest way to see it is to label at 0 and measure the spread separately.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import Config, VerificationConfig
from .db import RequestLog, sha256_hex

# Appended to the question rather than sent as a system message: the train/test
# split groups by the FIRST message, and one shared system prompt would put
# every question in a single group.
INSTRUCTION = "\n\nSolve it, then end your reply with a final line of the form ANSWER: <number>."

_ANSWER = re.compile(r"ANSWER:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


@dataclass(frozen=True)
class Item:
    question: str
    answer: str


def load_gsm8k(path: str | Path) -> list[Item]:
    """GSM8K's JSONL: the answer's last line is `#### <number>`."""
    items = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                raw = json.loads(line)
                items.append(Item(raw["question"], raw["answer"].split("####")[-1].strip()))
    return items


def _number(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace(",", "").rstrip("."))
    except InvalidOperation:
        return None


def final_answer(reply: str) -> Decimal | None:
    """The number the reply commits to: the last `ANSWER:` line, else nothing.

    No fallback to "the last number in the text". A reply that never states an
    answer has not given one, and grading whatever number it happened to end
    near would turn a failure to answer into a coin flip.
    """
    found = _ANSWER.findall(reply)
    return _number(found[-1]) if found else None


def grade(reply: str, expected: str, finish_reason: str | None = None) -> tuple[bool, str]:
    """(passed, reason). `finish_reason` separates two failures that look alike.

    A reply with no ANSWER line in it has not answered, but WHY it has not is
    the difference between a model that cannot do the arithmetic and one that
    was cut off before it finished. Measured over 400 questions: 60 of 70
    failures were the second, and a corpus that calls them the same thing
    teaches a classifier to predict the output budget.
    """
    want = _number(expected)
    got = final_answer(reply)
    if got is None:
        if finish_reason == "length":
            return False, f"expected {expected}; unfinished (finish_reason=length)"
        return False, f"expected {expected}; the reply states no ANSWER line"
    return got == want, f"expected {expected}, answered {got}"


def prompt_text_for(question: str) -> str:
    # Must match app._prompt_text for the same request, or resuming would redo
    # every question.
    return json.dumps(
        [{"role": "user", "content": question + INSTRUCTION}],
        sort_keys=True,
        ensure_ascii=False,
    )


@dataclass
class LabelReport:
    asked: int = 0
    passed: int = 0
    failed: int = 0
    # Failures that were cut off at the output budget rather than answered
    # wrongly. Counted apart because the fix is a number in the config, and
    # because training on them models the budget instead of the difficulty.
    unfinished: int = 0
    skipped_done: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def label(
    config: Config,
    items: Iterable[Item],
    *,
    db: str,
    tier: str,
    source: str,
    limit: int | None = None,
    temperature: float | None = 0.0,
    backend_factory: Callable | None = None,
    progress: Callable[[int, bool, str], None] | None = None,
) -> LabelReport:
    from fastapi.testclient import TestClient

    from .app import create_app

    if tier not in config.tiers:
        raise ValueError(f"unknown tier {tier!r}")
    if Path(db).resolve() == Path(config.log.path).resolve():
        raise ValueError(
            f"{db} is the serving log. Benchmark rows would be counted as traffic "
            "and as savings; give them a database of their own."
        )

    # Public questions, not user data, so the text is kept: training cannot
    # read a hash. Verification is off because the answer key is the verdict.
    log = RequestLog(db, store_prompts=True)
    run_config = replace(config, verification=VerificationConfig())
    done = {r["prompt_sha256"] for r in log.query("SELECT prompt_sha256 FROM requests")}
    report = LabelReport()
    verifier = f"ground_truth:{source}"

    kwargs = {"backend_factory": backend_factory} if backend_factory else {}
    app = create_app(run_config, log=log, **kwargs)
    try:
        with TestClient(app) as client:
            for item in items:
                if limit is not None and report.asked >= limit:
                    break
                if sha256_hex(prompt_text_for(item.question)) in done:
                    report.skipped_done += 1
                    continue
                report.asked += 1
                payload: dict[str, Any] = {
                    "model": tier,
                    "messages": [{"role": "user", "content": item.question + INSTRUCTION}],
                }
                if temperature is not None:
                    payload["temperature"] = temperature
                response = client.post("/v1/chat/completions", json=payload)
                request_id = response.headers.get("X-Request-Id")
                if response.status_code != 200 or not request_id:
                    report.errors.append(f"{response.status_code}: {response.text[:200]}")
                    continue
                choice = response.json()["choices"][0]
                reply = choice["message"].get("content") or ""
                ok, reason = grade(reply, item.answer, choice.get("finish_reason"))
                log.record_label(
                    request_id, verdict="pass" if ok else "fail", reason=reason, source=verifier
                )
                if ok:
                    report.passed += 1
                else:
                    report.failed += 1
                    if choice.get("finish_reason") == "length":
                        report.unfinished += 1
                if progress:
                    progress(report.asked, ok, reason)
    finally:
        log.close()
    return report


def format_label_report(report: LabelReport, *, db: str) -> str:
    graded = report.passed + report.failed
    rate = f"{report.failed / graded:.1%}" if graded else "n/a"
    lines = [
        f"asked {report.asked} question(s)  ->  {db}",
        f"  pass {report.passed}   fail {report.failed}   failure rate {rate}",
    ]
    if report.unfinished:
        share = report.unfinished / report.failed
        lines.append(
            f"  of those failures, {report.unfinished} ({share:.0%}) were UNFINISHED rather "
            "than wrong: cut off at the tier's output budget"
        )
        if share >= 0.5:
            lines.append(
                "  most of this corpus's failures are unfinished answers, which is not a "
                "difficulty signal -- the router already fails one without a model. Before "
                "training here: raise the tier's output budget, and check the temperature. "
                "Measured on this benchmark, asking the same questions again recovered 27 "
                "of 60 within the OLD budget, so most of what looked like difficulty was "
                "the sample rather than the question."
            )
    lines.append(f"  already labelled, skipped {report.skipped_done}")
    lines += [f"  ERROR {e}" for e in report.errors]
    return "\n".join(lines)
