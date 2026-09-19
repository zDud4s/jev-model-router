"""Routing: pick one tier from the eligible candidates, and say why.

The seam is the product of this step, not the rule. `Router.decide` receives the
request and the already-filtered candidate list, and returns a decision naming a
tier from that list.

Contract for any implementation:
  * the tier returned MUST be a member of `candidates` (never a tier the
    eligibility gate removed);
  * `candidates` may be empty, in which case there is nothing to choose and the
    caller -- not the router -- turns that into an error;
  * the decision carries whatever the router used to make it, so the log can
    hold the reason next to the outcome.

That last clause is a correction. This module used to promise that a learned
router would implement the existing interface and nothing else would change, and
that was wrong in one specific way: a router that computes a number and does not
report it cannot be evaluated after the fact. Its score has to reach the log row
beside the verdict, or the deployed model is unfalsifiable -- which is the exact
failure mode the rest of this project exists to avoid. So `choose` became
`decide`, and the return type grew three fields.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from .classifier import DifficultyModel, ModelError
from .config import Config, ConfigError
from .schemas import ChatCompletionRequest


@dataclass(frozen=True)
class RouteDecision:
    """The tier, and the evidence behind it."""

    tier: str
    # P(this prompt's cheap answer fails review), or None when nothing scored
    # it -- an explicit request, or a cheap tier the gate had already removed.
    # None is not 0.0: "not scored" and "scored as easy" are different rows.
    score: float | None = None
    # Fingerprint of the model that produced the score. Without it, a column of
    # scores spanning a retraining is two models' numbers in one histogram.
    model: str | None = None
    reason: str = ""


@runtime_checkable
class Router(Protocol):
    name: str

    def decide(
        self, request: ChatCompletionRequest, candidates: list[str]
    ) -> RouteDecision:
        """Return a decision naming one of `candidates`. Never called with an empty list."""
        ...


def _explicit_preference(config: Config, request: ChatCompletionRequest) -> str | None:
    """The tier the client actually asked for, or None if it left it to us.

    Kept separate from the default so a router can tell "the client said `top`"
    from "the client said nothing and `top` happens to be the default". A
    classifier must never overrule the first, and exists entirely for the second.
    """
    requested = request.model
    if requested in config.tiers:
        return requested
    return config.router.model_map.get(requested)


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

    def decide(
        self, request: ChatCompletionRequest, candidates: list[str]
    ) -> RouteDecision:
        explicit = _explicit_preference(self._config, request)
        preferred = explicit or self._default
        if preferred in candidates:
            return RouteDecision(
                tier=preferred, reason="requested" if explicit else "default"
            )
        # The preference was filtered out by the eligibility gate. Falling back
        # to the first surviving candidate in configuration order keeps the
        # request serviceable; the rejection that caused it is already logged,
        # so this substitution is visible rather than silent.
        return RouteDecision(tier=candidates[0], reason=f"{preferred} ineligible")


class ClassifierRouter:
    """The learned router: escalate the prompts the cheap tier is likely to fail.

    One model, one question: given this prompt, what is the probability that
    `default_tier`'s answer would not survive review? Above the threshold the
    request goes straight to `strong_tier` and no cheap answer is produced at
    all; below it, the cheap tier answers as usual.

    Four things it deliberately does not do:

    * **Overrule an explicit request.** A client that asked for a tier by name
      gets it, unscored. Guessing over a stated preference is not routing.
    * **Score a request the cheap tier could not have taken.** The gate has
      already removed it; there is no bet to place, and the row records no score
      rather than a score nobody acted on.
    * **Hide that it was overruled.** When the strong tier is itself ineligible
      the request stays cheap and the score is logged anyway, so "wanted to
      escalate and could not" stays countable instead of invisible.
    * **Run on a model trained for a different tier.** The file says which tier
      its labels describe; a mismatch is a configuration error rather than a
      warning, because the resulting scores would be confident and meaningless.

    And one thing it does that looks like a bug and is not: `explore_rate`
    sends a fraction of would-be escalations to the cheap tier anyway. A
    classifier that is always obeyed destroys the evidence that would test it --
    every prompt it scores high skips the cheap tier, is never reviewed, and
    never becomes a label, so the next model is fitted only on prompts the
    current one already believed were easy, and its failure rate looks
    wonderful for as long as nobody checks. The explored requests are the only
    rows that can ever contradict the model, which makes a few percent of them
    the cheapest insurance in the system.
    """

    name = "classifier"

    def __init__(
        self,
        config: Config,
        model: DifficultyModel,
        *,
        rng: Callable[[], float] | None = None,
    ) -> None:
        router = config.router
        cheap = router.default_tier or next(iter(config.tiers))
        strong = router.strong_tier
        if not strong:
            raise ConfigError("router.kind 'classifier' requires router.strong_tier")
        if model.predicts_tier != cheap:
            raise ConfigError(
                f"model was trained on tier {model.predicts_tier!r} but "
                f"router.default_tier is {cheap!r}. A model of one cheap model's "
                "failures says nothing about another's -- retrain it, or point "
                "the router at the tier the model describes."
            )
        self._config = config
        self._model = model
        self._cheap = cheap
        self._strong = strong
        # The config may override the threshold the model was trained with; the
        # model's own value is the default, so a tuned file carries its tuning.
        self._threshold = (
            router.threshold if router.threshold is not None else model.threshold
        )
        self._explore_rate = router.explore_rate
        # Injected so a test can decide the coin instead of hoping.
        self._rng = rng or random.random

    @property
    def model(self) -> DifficultyModel:
        return self._model

    @property
    def threshold(self) -> float:
        return self._threshold

    def decide(
        self, request: ChatCompletionRequest, candidates: list[str]
    ) -> RouteDecision:
        explicit = _explicit_preference(self._config, request)
        if explicit and explicit in candidates:
            return RouteDecision(tier=explicit, reason="requested")

        if self._cheap not in candidates:
            # Nothing to predict: the bet this model was trained to make is not
            # on the table for this request.
            tier = self._strong if self._strong in candidates else candidates[0]
            return RouteDecision(tier=tier, reason=f"{self._cheap} ineligible")

        score = self._model.score_request(request)
        fingerprint = self._model.fingerprint
        if score < self._threshold:
            return RouteDecision(
                tier=self._cheap,
                score=score,
                model=fingerprint,
                reason=f"score {score:.3f} < {self._threshold:.2f}",
            )
        if self._strong not in candidates:
            return RouteDecision(
                tier=self._cheap,
                score=score,
                model=fingerprint,
                reason=f"{self._strong} ineligible; kept {self._cheap} at score {score:.3f}",
            )
        if self._explore_rate and self._rng() < self._explore_rate:
            # Overruled on purpose. This row will be verified, and it is the
            # only kind of row that can ever show the model was wrong to want
            # to escalate.
            return RouteDecision(
                tier=self._cheap,
                score=score,
                model=fingerprint,
                reason=f"explore: score {score:.3f} >= {self._threshold:.2f}, kept cheap",
            )
        return RouteDecision(
            tier=self._strong,
            score=score,
            model=fingerprint,
            reason=f"score {score:.3f} >= {self._threshold:.2f}",
        )


def build_router(config: Config) -> Router:
    kind = config.router.kind
    if kind == "static":
        return StaticRouter(config)
    if kind == "classifier":
        path = config.router.model_path
        if not path:
            raise ConfigError("router.kind 'classifier' requires router.model_path")
        try:
            model = DifficultyModel.load(path)
        except ModelError as exc:
            # Refusing to start is the point. The alternative -- fall back to
            # the static router and carry on -- is a system that looks like it
            # is classifying and is not, which is worse than an outage because
            # it is quiet. `kind: static` is how you ask for that behaviour.
            raise ConfigError(f"router.kind 'classifier': {exc}") from None
        return ClassifierRouter(config, model)
    # Failing loudly on an unknown kind is better than silently serving every
    # request from the default tier.
    raise ValueError(f"unknown router kind: {kind!r}")


__all__ = [
    "ClassifierRouter",
    "RouteDecision",
    "Router",
    "StaticRouter",
    "build_router",
]
