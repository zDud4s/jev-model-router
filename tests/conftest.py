"""Shared fixtures.

No test in this suite touches the network. Backends are replaced through
`create_app(backend_factory=...)`, which is the single injection point the
application exposes for exactly this.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Callable

import pytest

from jev_model_router.backends.base import (
    BackendError,
    BackendResponse,
    StreamChunk,
    StreamEnd,
    StreamEvent,
)
from jev_model_router.config import Config, TierConfig, parse_config
from jev_model_router.db import RequestLog
from jev_model_router.schemas import ChatCompletionRequest, Usage, build_chunk, build_completion

BASE_CONFIG: dict[str, Any] = {
    "log": {"path": ":memory:"},
    # The startup check reads this machine's CLI caches; tests that want it
    # pass `catalog_check` to create_app.
    "catalog": {"check_on_start": False},
    "router": {"kind": "static", "default_tier": "cheap"},
    "tiers": {
        # Local, free, small window, no tools: the tier the eligibility gate
        # exists to filter out. The window is deliberately larger than the
        # gate's default output reservation, or this tier would be ineligible
        # for every request and the routing tests would prove nothing.
        "cheap": {
            "backend": "ollama",
            "model": "llama3.1:8b",
            "base_url": "http://localhost:11434",
            "context_window": 4096,
            "supports_tools": False,
        },
        "mid": {
            "backend": "openai_compatible",
            "model": "mid-model",
            "base_url": "https://example.invalid/v1",
            "context_window": 200000,
            "supports_tools": True,
            "prices": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
        },
        "top": {
            "backend": "openai_compatible",
            "model": "top-model",
            "base_url": "https://example.invalid/v1",
            "context_window": 200000,
            "supports_tools": True,
            "prices": {"input": 15.0, "output": 75.0, "cache_read": 1.5, "cache_write": 18.75},
        },
    },
}


@pytest.fixture
def config() -> Config:
    return parse_config(BASE_CONFIG)


@pytest.fixture
def request_log() -> RequestLog:
    log = RequestLog(":memory:")
    yield log
    log.close()


class FakeBackend:
    """A backend that answers from a script instead of over HTTP."""

    def __init__(
        self,
        tier: TierConfig,
        *,
        content: str = "hello",
        usage: Usage | None = None,
        error: BackendError | None = None,
        stream_error_after: int | None = None,
    ) -> None:
        self.tier = tier
        self.content = content
        self.usage = usage or Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        self.error = error
        self.stream_error_after = stream_error_after
        self.calls: list[ChatCompletionRequest] = []
        self.closed = False

    async def complete(self, request: ChatCompletionRequest) -> BackendResponse:
        self.calls.append(request)
        if self.error:
            raise self.error
        return BackendResponse(
            body=build_completion(model=self.tier.name, content=self.content, usage=self.usage),
            usage=self.usage,
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        self.calls.append(request)
        if self.error and self.stream_error_after is None:
            raise self.error
        emitted = 0
        for index, piece in enumerate(self.content.split(" ")):
            delta: dict[str, Any] = {"content": (" " if index else "") + piece}
            if index == 0:
                delta["role"] = "assistant"
            yield StreamChunk(
                data=build_chunk(model=self.tier.name, completion_id="chatcmpl-test", delta=delta)
            )
            emitted += 1
            if self.stream_error_after is not None and emitted >= self.stream_error_after:
                assert self.error is not None
                raise self.error
        yield StreamChunk(
            data=build_chunk(
                model=self.tier.name,
                completion_id="chatcmpl-test",
                delta={},
                finish_reason="stop",
            )
        )
        yield StreamEnd(usage=self.usage)

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def fake_backends() -> dict[str, FakeBackend]:
    """Populated by `backend_factory` so a test can inspect what was called."""
    return {}


@pytest.fixture
def backend_factory(fake_backends: dict[str, FakeBackend]) -> Callable[[TierConfig], FakeBackend]:
    def factory(tier: TierConfig) -> FakeBackend:
        backend = FakeBackend(tier)
        fake_backends[tier.name] = backend
        return backend

    return factory


def make_request(**overrides: Any) -> ChatCompletionRequest:
    payload: dict[str, Any] = {
        "model": "auto",
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return ChatCompletionRequest.model_validate(payload)
