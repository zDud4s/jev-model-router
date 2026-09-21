"""Backend for TypeSafe's Jev, a judge that answers with a probability.

Jev is not an LLM and this is not an LLM adapter. It takes program state plus
typed questions and returns typed answers -- for a yes/no question ("noul"), one
calibrated probability. It cannot generate text at all, which is why a Jev tier
is a *verifier* and never a serving tier: `stream` refuses, and a tier that
cannot write prose has nothing to say to a client that asked for some.

Why a backend rather than a new verification path: the loop already knows how to
pick a judge, gate it on eligibility, bill it to the request that caused it, and
record what it said. All of that is keyed on a tier, so the smallest correct way
to let Jev judge is to make it a tier whose backend happens to answer in one
number. Pricing, counterfactuals, `stats` and the labelled rows `train` reads
then work untouched.

The translation has two moving parts, and both are the kind that lie quietly if
left undocumented.

**The system message becomes the question's instructions.** `build_review_request`
puts the reviewer's brief in a system message and the conversation plus the
answer under review in a user message. Jev has no roles: the brief is what it is
asked, the rest is what it reads. The brief's closing paragraph asks for a line
reading `VERDICT: PASS`, which Jev structurally cannot write and simply ignores.
That paragraph is inert here rather than honoured -- an operator who edited
`verification.system_prompt` to change the reply *format* has changed nothing,
while the part that carries meaning, what counts as a failure, is honoured in
full.

**The probability becomes a verdict, and the threshold is ours.** Jev answers
"how likely is it that this answer is correct"; the loop needs PASS or FAIL.
Somebody has to draw the line, and where to draw it is a routing decision rather
than a provider knob -- which is why it is `jev.threshold` in config and not a
key in `extra_body`. The probability is written into the verdict line, so the
log keeps the number the decision was made from and not only the decision. That
is the one thing a text judge could never give: a recorded run can be
re-thresholded after the fact, because the evidence survived the verdict.

One thing this removes rather than adds. There is no unparseable verdict here. A
text judge can spend its whole budget reasoning and never write a verdict, and
`on_unparseable: accept` then passes an answer nobody judged -- the failure this
project measured twice, on two different judges. Jev returns a probability or an
error, and a missing probability is raised as a backend error rather than
rendered as a pass.

**The wire shape is not settled.** TypeSafe's own guide and Cloudflare's model
page disagree on whether a noul question carries an explicit `type`, and on
whether the context limit is 32k or 64k. This sends `type` (the unambiguous
reading) and takes the window from the tier's `context_window` like every other
tier. If the shape moves, `extra_body` overrides any top-level key of the
request body -- including `questions` wholesale -- so a wire change is a config
edit rather than a patch.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

import httpx

from ..config import TierConfig
from ..schemas import ChatCompletionRequest, Usage, build_completion
from .base import BackendError, BackendResponse, StreamEvent

# Asked when the review carries no system message and the config names no
# instructions. It is the semantic half of the default reviewer brief with the
# format paragraph dropped: a noul question wants the criterion, not the shape
# of the reply.
DEFAULT_INSTRUCTIONS = (
    "The state is a conversation and an answer that another assistant produced "
    "for it. Is that answer correct, complete and responsive to what was asked? "
    "Judge correctness only, not style, tone or length. An answer that is "
    "factually wrong, ignores part of the request, contradicts itself, or stops "
    "mid-thought is not correct."
)


class JevBackend:
    """Asks Jev one yes/no question and renders the probability as a verdict."""

    def __init__(self, tier: TierConfig, client: httpx.AsyncClient | None = None) -> None:
        self.tier = tier
        self._client = client or httpx.AsyncClient(timeout=tier.timeout_s)
        self._owns_client = client is None

    @property
    def _url(self) -> str:
        return f"{self.tier.base_url}/systemone"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self.tier.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _body(self, request: ChatCompletionRequest) -> dict[str, Any]:
        settings = self.tier.jev
        instructions = settings.instructions
        state: list[str] = []
        for message in request.messages:
            content = _text_of(message.content)
            if message.role == "system":
                # The reviewer's brief, unless config already named one. First
                # system message wins; a review request carries exactly one.
                if instructions is None:
                    instructions = content
                continue
            if content:
                state.append(content)
        body: dict[str, Any] = {
            # Required, and the published schema is what says so -- TypeSafe's
            # written guide claims the endpoint names its own model, and the
            # live API answers 422 `missing: body.model` to a request that
            # believes it. Names come from GET /v1/models (`jev-latest`,
            # `jev-preview`); the response reports which one actually answered,
            # so an alias here still leaves a specific version in the log.
            "model": self.tier.model,
            "state": "\n\n".join(state),
            "questions": {
                settings.question_key: {
                    "type": "noul",
                    "instructions": instructions or DEFAULT_INSTRUCTIONS,
                }
            },
        }
        body.update(self.tier.extra_body)
        return body

    async def complete(self, request: ChatCompletionRequest) -> BackendResponse:
        try:
            response = await self._client.post(
                self._url, json=self._body(request), headers=self._headers()
            )
        except httpx.HTTPError as exc:
            raise BackendError(f"{self.tier.name}: {exc}", status=502) from exc

        if response.status_code >= 400:
            raise BackendError(
                f"{self.tier.name}: upstream returned {response.status_code}",
                status=response.status_code,
                body=_safe_json(response),
            )

        payload = _safe_json(response)
        if not isinstance(payload, dict):
            raise BackendError(f"{self.tier.name}: upstream sent no JSON object", status=502)

        key = self.tier.jev.question_key
        answer = (payload.get("answers") or {}).get(key)
        probability = answer.get("noul") if isinstance(answer, dict) else None
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            # No probability means the call did not work, whatever the status
            # said. Rendering a verdict from it would invent a judgement nobody
            # made, and `on_verifier_error` exists precisely so that case has
            # somewhere to go that is not a silent pass.
            raise BackendError(
                f"{self.tier.name}: response carried no noul probability for {key!r}",
                status=502,
                body=payload,
            )

        probability = float(probability)
        threshold = self.tier.jev.threshold
        verdict = "PASS" if probability >= threshold else "FAIL"
        detail = f"jev p(correct)={probability:.3f} threshold={threshold:.2f}"
        usage = _usage_of(payload.get("usage"))
        return BackendResponse(
            body=build_completion(
                model=self.tier.name, content=f"VERDICT: {verdict} - {detail}", usage=usage
            ),
            usage=usage,
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        # Unreachable through the verification loop, which never streams a
        # review. It is here so that configuring a Jev tier as a SERVING tier
        # fails loudly at the first streamed request instead of half-working.
        raise BackendError(
            f"{self.tier.name}: jev returns a typed decision and cannot stream", status=400
        )
        yield  # pragma: no cover - makes this an async generator, as the protocol requires

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _text_of(content: Any) -> str:
    """Flatten a message's content to the text Jev should read."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _usage_of(raw: Any) -> Usage:
    """Jev reports `input_tokens` / `output_tokens`, not the OpenAI spelling.

    Output is free and reported as zero, so the completion column of a Jev tier
    is genuinely empty rather than unmeasured -- and the price table says the
    same thing with `output: 0.0`, a written-down zero rather than an omitted
    field. That distinction is the one the `cache_read` bug taught.
    """
    raw = raw if isinstance(raw, dict) else {}
    prompt = int(raw.get("input_tokens", 0) or 0)
    completion = int(raw.get("output_tokens", 0) or 0)
    return Usage(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
    )


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:  # noqa: BLE001 - an error body that is not JSON is still useful
        return response.text
