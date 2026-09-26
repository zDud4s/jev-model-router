"""Backend registry.

One dict, keyed by the `backend:` value in config. A new provider is a new
module plus one line here -- and if it already speaks the OpenAI wire format,
not even that: it is a config entry with `backend: openai_compatible`.
"""

from __future__ import annotations

from typing import Callable

from ..config import Config, TierConfig
from .base import (
    Backend,
    BackendError,
    BackendResponse,
    StreamChunk,
    StreamEnd,
    StreamEvent,
)
from .claude_cli import ClaudeCliBackend
from .codex_cli import CodexCliBackend
from .jev import JevBackend
from .ollama import OllamaBackend
from .openai_compatible import OpenAICompatibleBackend

BackendFactory = Callable[[TierConfig], Backend]

_REGISTRY: dict[str, BackendFactory] = {
    "claude_cli": ClaudeCliBackend,
    "codex_cli": CodexCliBackend,
    "jev": JevBackend,
    "ollama": OllamaBackend,
    "openai_compatible": OpenAICompatibleBackend,
}


def build_backend(tier: TierConfig) -> Backend:
    try:
        factory = _REGISTRY[tier.backend]
    except KeyError:
        raise ValueError(f"unknown backend kind: {tier.backend!r}") from None
    return factory(tier)


def build_backends(config: Config, factory: BackendFactory | None = None) -> dict[str, Backend]:
    """One backend instance per tier, built once at startup.

    `factory` is the injection point the tests use: it replaces every adapter
    with a fake, which is how the suite runs with no network at all.
    """
    make = factory or build_backend
    return {name: make(tier) for name, tier in config.tiers.items()}


__all__ = [
    "Backend",
    "BackendError",
    "BackendFactory",
    "BackendResponse",
    "ClaudeCliBackend",
    "CodexCliBackend",
    "JevBackend",
    "OllamaBackend",
    "OpenAICompatibleBackend",
    "StreamChunk",
    "StreamEnd",
    "StreamEvent",
    "build_backend",
    "build_backends",
]
