"""Backend for a local Ollama server (`POST /api/chat`).

Ollama has its own request and response shape, so this adapter translates in
both directions. It also reports token counts under its own names
(`prompt_eval_count` / `eval_count`), which is the only reason costing a local
model works at all -- and costing a free model still matters, because the
counterfactual columns need its token counts to price the request against the
paid tiers.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from ..config import TierConfig
from ..schemas import (
    ChatCompletionRequest,
    Usage,
    build_chunk,
    build_completion,
    new_completion_id,
)
from .base import BackendError, BackendResponse, StreamChunk, StreamEnd, StreamEvent

# Ollama's finish reasons do not always match OpenAI's vocabulary.
_FINISH_REASONS = {"stop": "stop", "length": "length"}


class OllamaBackend:
    def __init__(self, tier: TierConfig, client: httpx.AsyncClient | None = None) -> None:
        self.tier = tier
        self._client = client or httpx.AsyncClient(timeout=tier.timeout_s)
        self._owns_client = client is None

    @property
    def _url(self) -> str:
        return f"{self.tier.base_url}/api/chat"

    def _body(self, request: ChatCompletionRequest, *, stream: bool) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.output_budget is not None:
            options["num_predict"] = request.output_budget
        if request.stop is not None:
            options["stop"] = request.stop if isinstance(request.stop, list) else [request.stop]

        body: dict[str, Any] = {
            "model": self.tier.model,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "stream": stream,
        }
        if options:
            body["options"] = options
        if request.tools:
            # Ollama accepts OpenAI-shaped tool definitions. Whether the model
            # honours them is a property of the model, which is what the tier's
            # `supports_tools` flag declares to the eligibility gate.
            body["tools"] = request.tools
        body.update(self.tier.extra_body)
        return body

    def _usage(self, payload: dict[str, Any]) -> Usage:
        prompt = int(payload.get("prompt_eval_count", 0) or 0)
        completion = int(payload.get("eval_count", 0) or 0)
        # Ollama reports no prompt cache, so cached_tokens stays 0 rather than
        # being guessed from its prompt-reuse behaviour.
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )

    async def complete(self, request: ChatCompletionRequest) -> BackendResponse:
        try:
            response = await self._client.post(self._url, json=self._body(request, stream=False))
        except httpx.HTTPError as exc:
            raise BackendError(f"{self.tier.name}: {exc}", status=502) from exc

        if response.status_code >= 400:
            raise BackendError(
                f"{self.tier.name}: ollama returned {response.status_code}",
                status=response.status_code,
                body=_safe_json(response),
            )

        payload = response.json()
        message = payload.get("message") or {}
        return BackendResponse(
            body=build_completion(
                model=self.tier.name,
                content=message.get("content", ""),
                usage=self._usage(payload),
                finish_reason=_FINISH_REASONS.get(payload.get("done_reason", "stop"), "stop"),
                tool_calls=message.get("tool_calls"),
            ),
            usage=self._usage(payload),
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        completion_id = new_completion_id()
        usage = Usage()
        try:
            async with self._client.stream(
                "POST", self._url, json=self._body(request, stream=True)
            ) as response:
                if response.status_code >= 400:
                    raw = await response.aread()
                    raise BackendError(
                        f"{self.tier.name}: ollama returned {response.status_code}",
                        status=response.status_code,
                        body=_safe_loads(raw),
                    )
                first = True
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    message = payload.get("message") or {}
                    delta: dict[str, Any] = {}
                    if first:
                        # OpenAI clients expect the role on the opening chunk.
                        delta["role"] = "assistant"
                        first = False
                    if message.get("content"):
                        delta["content"] = message["content"]
                    if message.get("tool_calls"):
                        delta["tool_calls"] = message["tool_calls"]

                    done = bool(payload.get("done"))
                    if done:
                        usage = self._usage(payload)
                    if delta or done:
                        yield StreamChunk(
                            data=build_chunk(
                                model=self.tier.name,
                                completion_id=completion_id,
                                delta=delta,
                                finish_reason=(
                                    _FINISH_REASONS.get(payload.get("done_reason", "stop"), "stop")
                                    if done
                                    else None
                                ),
                            )
                        )
        except httpx.HTTPError as exc:
            raise BackendError(f"{self.tier.name}: {exc}", status=502) from exc
        yield StreamEnd(usage=usage)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:  # noqa: BLE001
        return response.text


def _safe_loads(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return raw.decode("utf-8", errors="replace")
