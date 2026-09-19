"""Routing: pick one tier from the eligible candidates.

The seam is the product of this step, not the rule. `Router.choose` receives the
request and the already-filtered candidate list, and returns a tier name from
that list. The difficulty classifier is a later step and will implement exactly
this interface, so nothing outside this module needs to change when it lands.

Contract for any implementation:
  * the return value MUST be a member of `candidates` (never a tier the
    eligibility gate removed);
  * `candidates` may be empty, in which case there is nothing to choose and the
    caller — not the router — turns that into an error.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .config import Config
from .schemas import ChatCompletionRequest


@runtime_checkable
class Router(Protocol):
    name: str

    def choose(self, request: ChatCompletionRequest, candidates: list[str]) -> str:
        """Return one of `candidates`. Only called with a non-empty list."""
        ...


class StaticRouter:
    """Fixed rules: an explicit model_map entry, else a configured default.

    A client that asks for a tier by name (`model: "top"`) gets it; a client
    that hardcodes a vendor model string can be mapped via `router.model_map`.
    Everything else takes `router.default_tier`.
    """

    name = "static"

    def __init__(self, config: Config) -> None:
        self._config = config
        self._default = config.router.default_tier or next(iter(config.tiers))
        self._model_map = config.router.model_map

    def choose(self, request: ChatCompletionRequest, candidates: list[str]) -> str:
        preferred = self._preference(request)
        if preferred in candidates:
            return preferred
        # The preference was filtered out by the eligibility gate. Falling back
        # to the first surviving candidate in configuration order keeps the
        # request serviceable; the rejection that caused it is already logged,
        # so this substitution is visible rather than silent.
        return candidates[0]

    def _preference(self, request: ChatCompletionRequest) -> str:
        requested = request.model
        if requested in self._config.tiers:
            return requested
        if requested in self._model_map:
            return self._model_map[requested]
        return self._default


def build_router(config: Config) -> Router:
    kind = config.router.kind
    if kind == "static":
        return StaticRouter(config)
    # The classifier will register here. Failing loudly on an unknown kind is
    # better than silently serving every request from the default tier.
    raise ValueError(f"unknown router kind: {kind!r}")
