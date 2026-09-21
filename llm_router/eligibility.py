"""The eligibility gate: a hard filter that runs BEFORE any routing decision.

Ordering is the whole point. If the router chose first and eligibility merely
adjusted the choice afterwards, a preferred-but-impossible tier would still
produce a request the backend rejects. Filtering first means the router only
ever sees tiers that can actually serve the request, so the later classifier
inherits the same guarantee for free without knowing anything about it.

A rejection is a fact about the request, not a preference, so every one is
recorded with its reason and ends up in the log row.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import Config, TierConfig
from .schemas import ChatCompletionRequest
from .tokens import estimate_request_budget


class RejectionReason(str, Enum):
    CONTEXT_WINDOW = "context_window"
    TOOLS_UNSUPPORTED = "tools_unsupported"
    # The tier is a judge, not a model: it returns a typed decision and cannot
    # generate an answer at all. Distinct from the other two, which are about
    # THIS request being too large or too toolful for a tier that could
    # otherwise have served it.
    CANNOT_GENERATE = "cannot_generate"


@dataclass(frozen=True)
class Rejection:
    tier: str
    reason: RejectionReason
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"tier": self.tier, "reason": self.reason.value, "detail": self.detail}


@dataclass(frozen=True)
class EligibilityResult:
    eligible: list[str]
    rejections: list[Rejection]

    def is_eligible(self, tier: str) -> bool:
        return tier in self.eligible


def check_tier(
    tier: TierConfig,
    request: ChatCompletionRequest,
    *,
    estimated_tokens: int,
    serving: bool = True,
) -> Rejection | None:
    """Return the reason this tier cannot take the request, or None.

    `serving` is the difference between the two questions this function is
    asked. Routing asks "could this tier have answered the request", and a
    judge could not -- so a Jev tier is rejected for every request, which is
    what keeps it out of the counterfactual table as a cheaper way of having
    answered. The verification loop asks the narrower "can this tier take the
    review prompt", where a judge is exactly what is wanted, and passes
    `serving=False`.

    Collapsing the two would break one of them whichever way it went: a single
    permissive check invents a baseline nobody could have used, and a single
    strict one makes every Jev verdict a skip.
    """
    if serving and not tier.can_serve:
        return Rejection(
            tier=tier.name,
            reason=RejectionReason.CANNOT_GENERATE,
            detail=(
                f"backend {tier.backend!r} returns a typed decision and cannot "
                f"generate an answer"
            ),
        )
    if estimated_tokens > tier.context_window:
        return Rejection(
            tier=tier.name,
            reason=RejectionReason.CONTEXT_WINDOW,
            detail=(
                f"estimated {estimated_tokens} tokens exceeds context_window "
                f"{tier.context_window}"
            ),
        )
    if request.tools and not tier.supports_tools:
        return Rejection(
            tier=tier.name,
            reason=RejectionReason.TOOLS_UNSUPPORTED,
            detail=f"request declares {len(request.tools)} tool(s); backend declares supports_tools: false",
        )
    return None


def evaluate(config: Config, request: ChatCompletionRequest) -> EligibilityResult:
    """Partition the configured tiers into those that can serve this request and those that cannot."""
    estimated = estimate_request_budget(request)
    eligible: list[str] = []
    rejections: list[Rejection] = []
    # Configuration order is preserved so that "first eligible" is a stable,
    # operator-controlled fallback rather than dictionary luck.
    for name, tier in config.tiers.items():
        rejection = check_tier(tier, request, estimated_tokens=estimated)
        if rejection is None:
            eligible.append(name)
        else:
            rejections.append(rejection)
    return EligibilityResult(eligible=eligible, rejections=rejections)
