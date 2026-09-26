"""Backend for any endpoint that already speaks the OpenAI chat API.

This one adapter covers OpenAI itself, OpenRouter, vLLM, LM Studio, and
Anthropic behind an OpenAI-compatible proxy. That is why the config's backend
kind is `openai_compatible` rather than a vendor name: the wire protocol is the
thing this code knows about, and a vendor-named adapter per provider would be
four copies of the same file.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from ..config import TierConfig
from ..schemas import ChatCompletionRequest, Usage
from .base import BackendError, BackendResponse, StreamChunk, StreamEnd, StreamEvent


class OpenAICompatibleBackend:
    """Forwards the request nearly untouched; only `model` is rewritten."""

    def __init__(self, tier: TierConfig, client: httpx.AsyncClient | None = None) -> None:
        self.tier = tier
        self._client = client or httpx.AsyncClient(timeout=tier.timeout_s)
        self._owns_client = client is None

    @property
    def _url(self) -> str:
        return f"{self.tier.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self.tier.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _body(self, request: ChatCompletionRequest, *, stream: bool) -> dict[str, Any]:
        body = request.forwardable()
        # The client asked for a tier; the provider needs its own model name.
        body["model"] = self.tier.model
        body.update(self.tier.extra_body)
        if stream:
            body["stream"] = True
            # Without this, most providers omit usage from streamed responses
            # entirely, and the log would have nothing to cost the request with.
            body["stream_options"] = {"include_usage": True}
        return body

    async def complete(self, request: ChatCompletionRequest) -> BackendResponse:
        try:
            response = await self._client.post(
                self._url, json=self._body(request, stream=False), headers=self._headers()
            )
        except httpx.HTTPError as exc:
            raise BackendError(f"{self.tier.name}: {exc}", status=502) from exc

        if response.status_code >= 400:
            raise BackendError(
                f"{self.tier.name}: upstream returned {response.status_code}",
                status=response.status_code,
                body=_safe_json(response),
            )

        body = response.json()
        usage = Usage.from_openai(body.get("usage"))
        # Report the tier, not the upstream model name. The client asked the
        # router for a tier and should not have to learn what sits behind it.
        body["model"] = self.tier.name
        return BackendResponse(body=body, usage=usage)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        usage = Usage()
        try:
            async with self._client.stream(
                "POST", self._url, json=self._body(request, stream=True), headers=self._headers()
            ) as response:
                if response.status_code >= 400:
                    raw = await response.aread()
                    raise BackendError(
                        f"{self.tier.name}: upstream returned {response.status_code}",
                        status=response.status_code,
                        body=_safe_loads(raw),
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        # A malformed frame is the provider's problem, not a
                        # reason to drop a stream the client is already reading.
                        continue
                    if chunk.get("usage"):
                        usage = Usage.from_openai(chunk["usage"])
                    chunk["model"] = self.tier.name
                    # A usage-only frame carries no choices; forwarding it is
                    # harmless and clients that asked for usage expect it.
                    yield StreamChunk(data=chunk)
        except httpx.HTTPError as exc:
            raise BackendError(f"{self.tier.name}: {exc}", status=502) from exc
        yield StreamEnd(usage=usage)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:  # noqa: BLE001 - an error body that is not JSON is still useful
        return response.text


def _safe_loads(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return raw.decode("utf-8", errors="replace")
