"""The verification loop: a stronger tier checks a cheaper tier's answer.

This is the component that decides whether a router's savings are real. Routing
alone is a bet -- the cheap tier is assumed good enough and nobody ever finds
out. Verification turns the bet into a measurement: the strong model reads the
cheap model's answer, says pass or fail, and a failure is re-answered properly.
The same verdicts are the labelled data a difficulty classifier needs, which is
why this is built before the classifier rather than after it.

It is also the component that can destroy the economics, so three things are
non-negotiable here:

1. **The verifier's own cost is charged to the request.** A savings number that
   counts the cheap call and quietly omits the review is not a savings number.
   `stats` prints verification spend as a share of the bill for this reason.

2. **The verifier passes through the eligibility gate like anything else.** The
   review prompt is a DIFFERENT prompt from the original -- it carries the
   transcript AND the answer -- so eligibility is re-evaluated against it. A
   verifier that cannot fit the review is recorded as skipped, never silently
   dropped and never sent to a backend that will reject it.

3. **Nothing here is silent.** Every skip carries its reason, every unparseable
   verdict is counted, and a verifier that fails is a recorded outcome rather
   than an absence of one.

Streaming is not verified, and that is a property of streaming rather than a
gap: once the first byte is on the wire the answer cannot be retracted. Those
requests are recorded as skipped with that reason.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from .backends.base import Backend
from .config import Config, VerificationConfig
from .eligibility import check_tier
from .pricing import cost_usd
from .schemas import ChatCompletionRequest, Usage
from .tokens import estimate_request_budget


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    # The verifier was asked and could not answer: the backend failed. Distinct
    # from FAIL, which is the verifier working correctly and disliking what it
    # read. Collapsing the two would make a broken verifier look like a bad
    # cheap model.
    ERROR = "error"
    # The verifier was never asked. `reason` says why.
    SKIPPED = "skipped"


class SkipReason(str, Enum):
    STREAMING = "streaming"
    TIER_NOT_VERIFIED = "tier_not_verified"
    SAMPLED_OUT = "sampled_out"
    NO_ANSWER_TEXT = "no_answer_text"
    VERIFIER_INELIGIBLE = "verifier_ineligible"
    UPSTREAM_ERROR = "upstream_error"


_DEFAULT_SYSTEM_PROMPT = """\
You are reviewing another assistant's answer. You are shown the conversation and \
the answer that was produced for it.

Judge only whether the answer is correct, complete and responsive to what was \
asked. Do not judge style, tone or length. Do not write your own answer.

Reply with exactly one line in this form:

VERDICT: PASS
VERDICT: FAIL - one short sentence saying what is wrong

Answer FAIL when the answer is factually wrong, ignores part of the request, \
contradicts itself, or stops mid-thought. Answer PASS otherwise. An answer you \
cannot fully check, but which contains nothing you can point to as wrong, is a \
PASS -- failing on uncertainty turns every request into two."""

# Tolerant on purpose: models wrap the line in bold, swap the dash for a colon,
# or lead with a sentence of preamble. What it will NOT do is guess -- a reply
# with no verdict token in it is unparseable and handled as such.
_VERDICT_RE = re.compile(r"\bVERDICT\b\s*[:\-—]?\s*\**\s*(PASS|FAIL)\b", re.IGNORECASE)
_BARE_RE = re.compile(r"^\W*(PASS|FAIL)\b", re.IGNORECASE)


@dataclass
class VerificationOutcome:
    """What the loop did to one request, including what it cost."""

    verdict: Verdict
    reason: str | None = None
    verifier_tier: str | None = None
    verifier_usage: Usage = field(default_factory=Usage)
    verifier_cost_usd: float = 0.0
    escalated: bool = False
    escalated_to: str | None = None
    escalation_usage: Usage = field(default_factory=Usage)
    escalation_cost_usd: float = 0.0
    latency_ms: int = 0
    # True when the verifier replied but the reply contained no verdict. The
    # configured fallback then decided the outcome, so `verdict` alone would
    # hide a broken verifier behind a plausible-looking pass rate.
    unparseable: bool = False
    # The replacement response body, present only when an escalation produced one.
    body: dict[str, Any] | None = None

    @property
    def extra_cost_usd(self) -> float:
        """Everything this loop added to the request's bill."""
        return self.verifier_cost_usd + self.escalation_cost_usd


def answer_text(body: Mapping[str, Any] | None) -> str:
    """The assistant text of a completion body, or '' when there is none.

    An answer that is only tool calls has no text to review. That is a real case
    rather than an error, and it is why this returns '' instead of raising.
    """
    if not body:
        return ""
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        # Multimodal shape: keep the text parts, drop the rest.
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(parts).strip()
    return ""


def finish_reason(body: Mapping[str, Any] | None) -> str | None:
    """Why the backend stopped generating, as it reported it."""
    choices = (body or {}).get("choices") or []
    return choices[0].get("finish_reason") if choices else None


def has_tool_calls(body: Mapping[str, Any] | None) -> bool:
    choices = (body or {}).get("choices") or []
    if not choices:
        return False
    return bool((choices[0].get("message") or {}).get("tool_calls"))


def parse_verdict(text: str) -> tuple[bool | None, str | None]:
    """(passed, reason). `passed` is None when the reply carries no verdict."""
    if not text:
        return None, None
    match = _VERDICT_RE.search(text) or _BARE_RE.search(text.strip())
    if not match:
        return None, None
    passed = match.group(1).upper() == "PASS"
    tail = text[match.end():].strip(" \t\n:-—*")
    reason = tail.splitlines()[0].strip() if tail else None
    return passed, reason or None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    # Keep the head and the tail: a truncated answer's ending is where "stops
    # mid-thought" is visible, and the middle is the cheapest part to lose.
    head = limit * 2 // 3
    tail = limit - head
    omitted = len(text) - limit
    return f"{text[:head]}\n\n[... {omitted} characters omitted ...]\n\n{text[-tail:]}"


def render_transcript(request: ChatCompletionRequest, limit: int) -> str:
    lines: list[str] = []
    for message in request.messages:
        content = message.content
        if not isinstance(content, str):
            content = str(content)
        lines.append(f"[{message.role}] {content}")
    return _truncate("\n\n".join(lines), limit)


def build_review_request(
    request: ChatCompletionRequest, answer: str, settings: VerificationConfig
) -> ChatCompletionRequest:
    """The prompt the verifier is actually sent.

    The conversation is flattened into one user message rather than replayed as
    real turns. A verifier handed the conversation as turns tends to answer it
    instead of reviewing it, and a single block also makes the token estimate --
    which the eligibility gate depends on -- match what is sent.

    `tools` is deliberately absent: a review is a text judgement, and carrying
    the original tool schemas would both invite tool calls and inflate the
    prompt by the largest part of an agentic request.
    """
    transcript = render_transcript(request, settings.max_transcript_chars)
    reviewed = _truncate(answer, settings.max_answer_chars)
    return ChatCompletionRequest(
        model=settings.verifier_tier or "",
        messages=[
            {"role": "system", "content": settings.system_prompt or _DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "=== CONVERSATION ===\n"
                    f"{transcript}\n\n"
                    "=== ANSWER UNDER REVIEW ===\n"
                    f"{reviewed}\n\n"
                    "=== YOUR VERDICT ==="
                ),
            },
        ],
        max_tokens=settings.max_verdict_tokens,
        temperature=0.0,
    )


class Verifier:
    """Runs the loop. Holds no state between requests beyond its collaborators."""

    def __init__(
        self,
        config: Config,
        backends: Mapping[str, Backend],
        *,
        rng: Callable[[], float] | None = None,
    ) -> None:
        self._config = config
        self._settings = config.verification
        self._backends = backends
        # Injected so a test can decide sampling instead of hoping.
        self._rng = rng or random.random

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    def skip(self, reason: SkipReason, detail: str | None = None) -> VerificationOutcome:
        return VerificationOutcome(
            verdict=Verdict.SKIPPED,
            reason=f"{reason.value}: {detail}" if detail else reason.value,
            verifier_tier=self._settings.verifier_tier,
        )

    async def check(
        self,
        request: ChatCompletionRequest,
        *,
        served_tier: str,
        body: Mapping[str, Any] | None,
        eligible: list[str],
    ) -> VerificationOutcome:
        """Review the answer in `body`, escalating if it fails. Never raises."""
        settings = self._settings
        if not settings.verifies(served_tier):
            return self.skip(SkipReason.TIER_NOT_VERIFIED, served_tier)

        answer = answer_text(body)
        if not answer:
            if has_tool_calls(body):
                # A tool call is a complete answer with no prose in it.
                return self.skip(SkipReason.NO_ANSWER_TEXT)
            # No text and no tool calls is not an answer at all. This used to be
            # the same skip as the tool-call case, and on the first run against
            # a real model it let an empty 200 through to the client: a thinking
            # model spent its whole window reasoning and said nothing. Failing it
            # needs no reviewer, costs nothing, and so ignores sampling --
            # sampling rations a paid review, and there is nothing here to pay.
            return await self._fail_empty(request, served_tier, body, eligible)

        if self._rng() >= settings.sample_rate:
            return self.skip(SkipReason.SAMPLED_OUT)

        verifier_name = settings.verifier_tier
        assert verifier_name is not None  # guaranteed by config validation
        review = build_review_request(request, answer, settings)

        # The gate again, against the review prompt rather than the original.
        rejection = check_tier(
            self._config.tier(verifier_name),
            review,
            estimated_tokens=estimate_request_budget(review),
        )
        if rejection is not None:
            return self.skip(SkipReason.VERIFIER_INELIGIBLE, rejection.detail)

        started = time.perf_counter()
        outcome = VerificationOutcome(verdict=Verdict.ERROR, verifier_tier=verifier_name)
        try:
            result = await self._backends[verifier_name].complete(review)
        except Exception as exc:  # noqa: BLE001 - a broken verifier must not break the request
            outcome.reason = f"verifier failed: {type(exc).__name__}: {exc}"
            if settings.on_verifier_error == "escalate":
                await self._escalate(request, served_tier, eligible, outcome)
            outcome.latency_ms = _elapsed_ms(started)
            return outcome

        outcome.verifier_usage = result.usage
        outcome.verifier_cost_usd = cost_usd(
            self._config.tier(verifier_name).prices, result.usage
        )

        passed, reason = parse_verdict(answer_text(result.body))
        if passed is None:
            outcome.unparseable = True
            # The fallback decides, and says so: a pass rate that includes
            # unparseable replies must be readable as such in the log.
            passed = settings.on_unparseable == "accept"
            cause = ""
            if finish_reason(result.body) == "length":
                # Measured on qwen3.5:4b: every review came back with exactly
                # max_verdict_tokens generated and no verdict, because a
                # thinking model spends that budget reasoning before it writes a
                # word. A bare "unparseable" sends the reader to the regex; the
                # cause is the budget, and the reason says so.
                cause = (
                    f"; the verifier stopped at max_verdict_tokens "
                    f"({settings.max_verdict_tokens}) before giving one -- a thinking "
                    "model spends that budget reasoning: set extra_body think: false "
                    "on the verifier tier, or raise max_verdict_tokens"
                )
            outcome.reason = (
                f"unparseable verdict{cause}; on_unparseable={settings.on_unparseable}"
            )
        else:
            outcome.reason = reason

        outcome.verdict = Verdict.PASS if passed else Verdict.FAIL
        if not passed:
            await self._escalate(request, served_tier, eligible, outcome)

        outcome.latency_ms = _elapsed_ms(started)
        return outcome

    async def _fail_empty(
        self,
        request: ChatCompletionRequest,
        served_tier: str,
        body: Mapping[str, Any] | None,
        eligible: list[str],
    ) -> VerificationOutcome:
        started = time.perf_counter()
        outcome = VerificationOutcome(
            verdict=Verdict.FAIL,
            # Nobody judged it, and the column says so rather than crediting a
            # verifier that was never called.
            verifier_tier=None,
            reason=(
                f"empty answer (finish_reason={finish_reason(body)}); "
                "failed without a review"
            ),
        )
        await self._escalate(request, served_tier, eligible, outcome)
        outcome.latency_ms = _elapsed_ms(started)
        return outcome

    async def _escalate(
        self,
        request: ChatCompletionRequest,
        served_tier: str,
        eligible: list[str],
        outcome: VerificationOutcome,
    ) -> None:
        """Re-answer the ORIGINAL request at the stronger tier. Mutates `outcome`."""
        target = self._settings.escalate_to
        if not target or target == served_tier:
            outcome.reason = _note(
                outcome.reason, "no escalation target distinct from the served tier"
            )
            return
        if target not in eligible:
            # Known bad and unfixable: the strong tier could not have served this
            # request either. Saying so is worth more than a silent pass.
            outcome.reason = _note(
                outcome.reason, f"escalation tier {target!r} ineligible for this request"
            )
            return

        try:
            result = await self._backends[target].complete(request)
        except Exception as exc:  # noqa: BLE001
            outcome.reason = _note(
                outcome.reason, f"escalation failed: {type(exc).__name__}: {exc}"
            )
            return

        outcome.escalated = True
        outcome.escalated_to = target
        outcome.escalation_usage = result.usage
        outcome.escalation_cost_usd = cost_usd(self._config.tier(target).prices, result.usage)
        outcome.body = result.body


def _note(existing: str | None, addition: str) -> str:
    return f"{existing}; {addition}" if existing else addition


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def build_verifier(config: Config, backends: Mapping[str, Backend]) -> Verifier | None:
    """None when verification is off, so the caller has nothing to branch on."""
    if not config.verification.enabled:
        return None
    return Verifier(config, backends)


__all__ = [
    "SkipReason",
    "Verdict",
    "VerificationOutcome",
    "Verifier",
    "answer_text",
    "build_review_request",
    "build_verifier",
    "finish_reason",
    "has_tool_calls",
    "parse_verdict",
]
