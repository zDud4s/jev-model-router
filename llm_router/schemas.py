"""OpenAI-compatible request and response shapes.

The request model is deliberately permissive (`extra="allow"`): clients send
fields this router does not understand, and the correct behaviour for a proxy is
to forward them rather than to reject the request.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    # `content` is a string in the classic shape and a list of parts in the
    # multimodal one; both are forwarded untouched.
    content: Any = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: Any = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    stream_options: dict[str, Any] | None = None
    user: str | None = None

    @property
    def output_budget(self) -> int | None:
        """The caller's declared output cap, under either spelling."""
        return self.max_completion_tokens or self.max_tokens

    def forwardable(self) -> dict[str, Any]:
        """The request body to pass on, minus fields the router owns itself."""
        body = self.model_dump(exclude_none=True)
        for owned in ("model", "stream", "stream_options"):
            body.pop(owned, None)
        return body


class Usage(BaseModel):
    """Token counts as reported by the backend.

    `cached_tokens` and `cache_write_tokens` are zero for backends that do not
    report caching; they are not guesses.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    # What the provider says it CHARGED for this call, in USD, when it says.
    # None means "not reported", never "free": the estimate from the price table
    # is a different number, kept beside this one rather than replaced by it, so
    # the gap between them is something `stats` can show. OpenRouter reports it
    # on every response as `usage.cost` (its credits are dollars); OpenAI and
    # Anthropic report nothing per request.
    billed_usd: float | None = None

    @classmethod
    def from_openai(cls, raw: dict[str, Any] | None) -> "Usage":
        raw = raw or {}
        details = raw.get("prompt_tokens_details") or {}
        prompt = int(raw.get("prompt_tokens", 0) or 0)
        completion = int(raw.get("completion_tokens", 0) or 0)
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=int(raw.get("total_tokens", prompt + completion) or 0),
            cached_tokens=int(details.get("cached_tokens", 0) or 0),
            # Three spellings of one number. Anthropic-style proxies use the
            # `cache_creation` pair; OpenRouter uses `cache_write_tokens`, and
            # until it was read here every cache write through OpenRouter was
            # billed at nothing.
            cache_write_tokens=int(
                details.get("cache_write_tokens")
                or details.get("cache_creation_tokens")
                or raw.get("cache_creation_input_tokens")
                or 0
            ),
            billed_usd=_optional_float(raw.get("cost")),
        )

    def to_openai(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens or (self.prompt_tokens + self.completion_tokens),
            "prompt_tokens_details": {"cached_tokens": self.cached_tokens},
        }


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def build_completion(
    *,
    model: str,
    content: str,
    usage: Usage,
    completion_id: str | None = None,
    finish_reason: str = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble a non-streaming `chat.completion` body."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": completion_id or new_completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage.to_openai(),
    }


def build_chunk(
    *,
    model: str,
    completion_id: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """Assemble one `chat.completion.chunk` of a streamed response."""
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def error_body(message: str, *, kind: str = "invalid_request_error", code: str | None = None) -> dict[str, Any]:
    """The OpenAI error envelope, so client SDKs surface our errors normally."""
    return {"error": {"message": message, "type": kind, "param": None, "code": code}}


ObjectKind = Literal["model", "list"]


def model_card(tier_name: str, model: str) -> dict[str, Any]:
    """One entry of `GET /v1/models`.

    The id is the TIER name, not the upstream model name: a tier is what a
    client can meaningfully ask for, and the mapping behind it is the router's.
    """
    return {
        "id": tier_name,
        "object": "model",
        "created": 0,
        "owned_by": "llm-router",
        # Non-standard, but harmless to clients and useful to a human reading it.
        "root": model,
    }


class ModelList(BaseModel):
    object: str = "list"
    data: list[dict[str, Any]] = Field(default_factory=list)
