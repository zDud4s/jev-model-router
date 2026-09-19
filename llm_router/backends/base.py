"""The backend interface.

Kept deliberately small: two methods and one error type. A backend translates an
OpenAI-shaped request into whatever the provider wants, and translates the
answer back. It does not route, price, or log -- adding a provider should not
require reading any other module in this package.

Streaming yields a union of events rather than raw dicts so the final token
usage has somewhere to arrive. Providers report usage in a trailing frame, after
the last content delta, and the log needs it; a bare `AsyncIterator[dict]` would
have no place to put it that is not a side channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from ..config import TierConfig
from ..schemas import ChatCompletionRequest, Usage


class BackendError(Exception):
    """A backend failed to serve a request.

    `status` is the HTTP status to hand the client. It carries the upstream
    status when there was one, so a client's own retry logic keeps working
    (a 429 must not reach it as a 500).
    """

    def __init__(self, message: str, *, status: int = 502, body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class BackendResponse:
    """A non-streaming answer: an OpenAI-shaped body plus the usage we billed."""

    body: dict[str, Any]
    usage: Usage = field(default_factory=Usage)


@dataclass
class StreamChunk:
    """One `chat.completion.chunk`, ready to be serialized as an SSE frame."""

    data: dict[str, Any]


@dataclass
class StreamEnd:
    """Terminal event carrying the usage totals for the finished stream."""

    usage: Usage = field(default_factory=Usage)


StreamEvent = StreamChunk | StreamEnd


@runtime_checkable
class Backend(Protocol):
    """What the router requires of a provider adapter."""

    tier: TierConfig

    async def complete(self, request: ChatCompletionRequest) -> BackendResponse:
        """Serve a non-streaming request. Raises BackendError on failure."""
        ...

    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        """Serve a streaming request, ending with exactly one StreamEnd."""
        ...

    async def aclose(self) -> None:
        """Release connections. Safe to call more than once."""
        ...
