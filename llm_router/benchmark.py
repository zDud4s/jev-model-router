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


def grade(reply: str, expected: str) -> tuple[bool, str]:
    want = _number(expected)
    got = final_answer(reply)
    if got is None:
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
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": tier,
                        "messages": [{"role": "user", "content": item.question + INSTRUCTION}],
                    },
                )
                request_id = response.headers.get("X-Request-Id")
                if response.status_code != 200 or not request_id:
                    report.errors.append(f"{response.status_code}: {response.text[:200]}")
                    continue
                reply = response.json()["choices"][0]["message"].get("content") or ""
                ok, reason = grade(reply, item.answer)
                log.record_label(
                    request_id, verdict="pass" if ok else "fail", reason=reason, source=verifier
                )
                if ok:
                    report.passed += 1
                else:
                    report.failed += 1
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
        f"  already labelled, skipped {report.skipped_done}",
    ]
    lines += [f"  ERROR {e}" for e in report.errors]
    return "\n".join(lines)
