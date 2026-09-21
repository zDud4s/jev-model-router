"""Measure a configured verifier tier against answers whose truth is known.

The project has twice reached the same conclusion about judges the expensive
way: a judge is only as good as the checking it can afford to do, and the way it
fails is by passing things rather than by breaking. `max_verdict_tokens` exists
because a thinking judge ran out of budget and `on_unparseable: accept` turned
silence into a pass. Turning reasoning off instead bought six-token verdicts
that passed *"strawberry has two r's"*.

So before a new judge is believed, it answers the same kind of question. This
script asks one -- any tier the config names -- to review answers that are
already known to be right or wrong, and prints what it got, what it cost and how
long it took.

It is deliberately NOT specific to Jev. A number with nothing to compare it to
decides nothing, and the comparison that matters is Jev against the text judges
already measured here:

    python scripts/judge_eval.py -c config.yaml --tier judge     # jev
    python scripts/judge_eval.py -c config.yaml --tier sonnet    # a text judge

Everything on the path is the real thing: `build_review_request` builds the
prompt the loop would build, the configured backend answers it, and
`parse_verdict` reads the reply. A harness that mocked any of those would be
measuring itself.

Three things it prints that a bare accuracy number would hide.

**The always-PASS baseline.** A judge that approves everything scores the share
of the corpus that is correct. That is the number to beat, and it is the same
discipline the classifier report uses -- the majority baseline printed beside
the accuracy, so a result that does not clear it cannot be read as one that
does.

**Recall on the wrong answers, separately.** Catching failures is the entire
job. An accuracy that looks respectable while recall is near zero is a judge
that has learned to say PASS, which is exactly what a cheap judge degrades into.

**The probability, when the judge reports one.** A Jev tier writes its
probability into the verdict line, so this can sweep the threshold afterwards
and show where the operator's line should have been. That is a rerun a text
judge cannot give you: PASS/FAIL keeps no evidence, and moving the threshold
means paying for the whole run again.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_router.backends import build_backend  # noqa: E402
from llm_router.backends.base import BackendError  # noqa: E402
from llm_router.config import ConfigError, load_config  # noqa: E402
from llm_router.pricing import cost_usd  # noqa: E402
from llm_router.schemas import ChatCompletionRequest  # noqa: E402
from llm_router.verification import (  # noqa: E402
    answer_text,
    build_review_request,
    finish_reason,
    parse_verdict,
)


@dataclass(frozen=True)
class Case:
    question: str
    answer: str
    correct: bool
    note: str


# Planted answers. The wrong ones are wrong in the ways that actually reach a
# verifier: a plausible arithmetic slip, a confident false fact, a half-answered
# request, a contradiction, and an answer to a question nobody asked. Three are
# the specific ones that beat a judge in this project's own measurements, and
# they are marked.
CASES: list[Case] = [
    Case("What is 17 x 4?", "17 x 4 = 68.", True, "plain arithmetic"),
    Case(
        "What is 17 x 4?",
        "17 x 4 = 78.",
        False,
        "arithmetic slip, stated confidently",
    ),
    Case(
        "How many r's are in the word strawberry?",
        "The word strawberry has two r's.",
        False,
        "PASSED by a reasoning-off judge in the README's measurement",
    ),
    Case(
        "What is the capital of Australia?",
        "Australia has no single official capital; power is shared between "
        "Sydney and Melbourne.",
        False,
        "confident false fact; the cheap model really produced this one",
    ),
    Case("What is the capital of Australia?", "Canberra.", True, "plain fact"),
    Case(
        "What HTTP status code means the request was not authenticated?",
        "401 Unauthorized.",
        True,
        "the case a reasoning-off judge got right",
    ),
    Case(
        "What HTTP status code means the request was not authenticated?",
        "403 Forbidden, which is returned when no credentials were supplied.",
        False,
        "the adjacent-and-wrong answer",
    ),
    Case(
        "List the first four prime numbers and then add them up.",
        "The first four primes are 2, 3, 5 and 7.",
        False,
        "ignores half the request",
    ),
    Case(
        "List the first four prime numbers and then add them up.",
        "2, 3, 5 and 7. Their sum is 17.",
        True,
        "both halves answered",
    ),
    Case(
        "A shop sells pens at 3 for 2 euros. What do 12 pens cost?",
        "12 pens is 4 groups of 3, so 4 x 2 = 8 euros.",
        True,
        "multi-step and right",
    ),
    Case(
        "A shop sells pens at 3 for 2 euros. What do 12 pens cost?",
        "12 pens is 4 groups of 3, so 4 x 2 = 6 euros.",
        False,
        "right method, wrong last step -- the hardest kind to catch",
    ),
    Case(
        "Is 91 a prime number?",
        "Yes, 91 is prime: it has no divisors other than 1 and itself.",
        False,
        "91 = 7 x 13; a judge that cannot factor cannot catch it",
    ),
    Case(
        "Summarise what a mutex does in one sentence.",
        "A mutex ensures that only one thread at a time can hold a lock on a "
        "shared resource.",
        True,
        "open-ended and correct",
    ),
    Case(
        "Summarise what a mutex does in one sentence.",
        "A mutex is a message queue that threads use to send each other values "
        "in order.",
        False,
        "open-ended and wrong -- no arithmetic to check it against",
    ),
    Case(
        "Explain in two sentences why the sky is blue.",
        "Sunlight contains every colour, and the short blue wavelengths scatter "
        "most strongly off air molecules. That scattered blue light arrives at "
        "your eye from every direction, so the",
        False,
        "stops mid-thought; the free rule would catch this one before any judge",
    ),
    Case(
        "What year did the Berlin Wall fall?",
        "1989.",
        True,
        "plain fact, short answer",
    ),
]

_PROBABILITY_RE = re.compile(r"p\(correct\)=([0-9.]+)")


@dataclass
class Result:
    case: Case
    passed: bool | None
    reason: str | None
    probability: float | None
    cost: float
    latency_ms: int
    error: str | None = None
    # Only read when the verdict was unparseable, and then it is the whole
    # diagnosis. `length` means the judge was cut off before it wrote one, which
    # is a budget to raise; anything else means it wrote something the parser
    # could not read, which is a prompt to fix. Without this the two arrive as
    # the same word and send the reader to the regex for the wrong one.
    finish: str | None = None
    tail: str = ""

    @property
    def right(self) -> bool:
        """A verdict is right when it agrees with what we planted."""
        return self.passed is not None and self.passed == self.case.correct


async def run(
    config, tier_name: str, limit: int | None, only: list[int] | None = None
) -> list[Result]:
    tier = config.tier(tier_name)
    backend = build_backend(tier)
    results: list[Result] = []
    numbered = list(enumerate(CASES, 1))
    # `--only` keeps the original case numbers, so a re-run of the one case that
    # went wrong is readable beside the run it came from.
    selected = [(n, c) for n, c in numbered if n in only] if only else numbered[: limit or len(CASES)]
    cases = [c for _, c in selected]
    # Printed as they land rather than collected and printed at the end. A
    # thinking judge takes minutes a verdict -- measured at ~145s on the local
    # qwen3.5:4b -- and a run that shows nothing for forty of them is
    # indistinguishable from one that has hung.
    print(f"\njudge: {tier_name}   cases: {len(cases)}\n")
    try:
        for case in cases:
            request = ChatCompletionRequest.model_validate(
                {"model": "auto", "messages": [{"role": "user", "content": case.question}]}
            )
            review = build_review_request(request, case.answer, config.verification)
            started = time.perf_counter()
            try:
                response = await backend.complete(review)
            except BackendError as exc:
                result = Result(case, None, None, None, 0.0, _ms(started), error=str(exc))
            else:
                said = answer_text(response.body)
                passed, reason = parse_verdict(said)
                match = _PROBABILITY_RE.search(reason or "")
                result = Result(
                    case=case,
                    passed=passed,
                    reason=reason,
                    probability=float(match.group(1)) if match else None,
                    cost=cost_usd(tier.prices, response.usage),
                    latency_ms=_ms(started),
                    finish=finish_reason(response.body),
                    tail=said[-160:].replace("\n", " ") if passed is None else "",
                )
            results.append(result)
            print(_line(selected[len(results) - 1][0], result), flush=True)
    finally:
        await backend.aclose()
    return results


def _line(index: int, result: Result) -> str:
    if result.error:
        mark, said = "ERR ", result.error[:60]
    else:
        # `None` is the failure this project cares most about: a reply the
        # parser could not read, which `on_unparseable` would have decided.
        said = {True: "PASS", False: "FAIL", None: "UNPARSEABLE"}[result.passed]
        mark = "ok  " if result.right else "MISS"
    want = "PASS" if result.case.correct else "FAIL"
    probability = f" p={result.probability:.3f}" if result.probability is not None else ""
    line = (
        f"  {index:2}  {mark} said {said:<11} wanted {want:<4}"
        f"{probability}  {result.latency_ms:>5}ms   {result.case.note}"
    )
    if result.passed is None and not result.error:
        cause = (
            "cut off before it wrote a verdict -- raise max_verdict_tokens"
            if result.finish == "length"
            else f"finish_reason={result.finish!r}, no verdict token in the reply"
        )
        line += f"\n          ^ {cause}\n          ^ ...{result.tail}"
    return line


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def report(results: list[Result], tier_name: str) -> None:
    answered = [r for r in results if r.error is None]
    if not answered:
        print("\nno verdicts: every call failed\n")
        return

    unparseable = [r for r in answered if r.passed is None]
    wrong_answers = [r for r in answered if not r.case.correct]
    caught = [r for r in wrong_answers if r.passed is False]
    flagged = [r for r in answered if r.passed is False]
    right = [r for r in answered if r.right]
    baseline = sum(1 for r in answered if r.case.correct) / len(answered)

    print(f"\n  accuracy            {len(right) / len(answered):.3f}  ({len(right)}/{len(answered)})")
    print(f"  always-PASS baseline {baseline:.3f}  <- the number to beat")
    if wrong_answers:
        print(
            f"  recall on wrong     {len(caught) / len(wrong_answers):.3f}  "
            f"({len(caught)}/{len(wrong_answers)} bad answers caught)"
        )
    if flagged:
        print(
            f"  precision on FAIL   {len(caught) / len(flagged):.3f}  "
            f"({len(caught)}/{len(flagged)} flags were real)"
        )
    if unparseable:
        # The loud one. Every unparseable verdict was decided by the fallback,
        # not by the judge, and a pass rate that includes them is not a
        # measurement of anything.
        print(f"  UNPARSEABLE         {len(unparseable)}  decided by on_unparseable, not by the judge")
    print(f"  cost                ${sum(r.cost for r in answered):.6f} total")
    latencies = sorted(r.latency_ms for r in answered)
    print(f"  latency             {latencies[len(latencies) // 2]}ms median, {latencies[-1]}ms max")

    sweep(answered)


def sweep(results: list[Result]) -> None:
    """Where the threshold should have been, for a judge that reports one."""
    scored = [r for r in results if r.probability is not None]
    if len(scored) < len(results):
        return
    print("\n  threshold sweep (this run, re-scored -- no second call):")
    best: tuple[float, float] = (-1.0, 0.0)
    for step in range(0, 21):
        threshold = step / 20
        right = sum(1 for r in scored if (r.probability >= threshold) == r.case.correct)
        accuracy = right / len(scored)
        if accuracy > best[0]:
            best = (accuracy, threshold)
        bar = "#" * int(accuracy * 40)
        print(f"    {threshold:.2f}  {accuracy:.3f}  {bar}")
    print(f"\n  best on this corpus: threshold {best[1]:.2f} at accuracy {best[0]:.3f}")
    print("  (16 cases is a demonstration, not a tuning set -- fit it on real traffic)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", required=True, help="path to the router config")
    parser.add_argument("--tier", required=True, help="which configured tier judges")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N cases")
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated case numbers, to re-run the ones that went wrong",
    )
    args = parser.parse_args()
    only = [int(n) for n in args.only.split(",")] if args.only else None

    try:
        config = load_config(args.config)
        results = asyncio.run(run(config, args.tier, args.limit, only))
    except ConfigError as exc:
        print(f"config: {exc}", file=sys.stderr)
        return 2
    report(results, args.tier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
