"""Configuration model, loaded from YAML.

The model catalog lives entirely in the config file. Nothing in this package may
name a concrete model: adding a backend must be one line of YAML and no code
change, so every model-specific fact (price, context window, tool support) is a
field here rather than a table in source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

BackendKind = Literal["ollama", "openai_compatible", "jev", "claude_cli", "codex_cli"]

# Reasoning levels the two subscription CLIs accept. Not every model takes every
# level (gpt-5.5 stops at xhigh, only some Codex models have ultra); the CLI is
# the one that refuses, and it says so in its own words.
EFFORTS = ("low", "medium", "high", "xhigh", "max", "ultra")

# The backends that run on a subscription login rather than a per-token key, and
# the subscription each one draws on unless a tier names another.
_SUBSCRIPTION_BACKENDS = {"claude_cli": "claude", "codex_cli": "codex"}


@dataclass(frozen=True)
class Prices:
    """USD per 1M tokens. Every field is optional.

    A tier with no prices costs zero. That is deliberate rather than a missing
    value: the headline use case is a local model, which really is free at the
    margin, and requiring the operator to write `input: 0.0` to express that
    would be a configuration trap.
    """

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0

    # Whether the operator supplied any price at all. Stats needs to tell "this
    # baseline is genuinely free" apart from "nobody configured this baseline",
    # because only the first supports a savings claim.
    configured: bool = False

    @classmethod
    def parse(cls, raw: dict[str, Any] | None) -> "Prices":
        if not raw:
            return cls()
        unknown = set(raw) - {"input", "output", "cache_read", "cache_write"}
        if unknown:
            raise ConfigError(f"unknown price fields: {sorted(unknown)}")
        input_price = float(raw.get("input", 0.0))
        return cls(
            input=input_price,
            output=float(raw.get("output", 0.0)),
            # Omitted means "I did not say", not "free". It used to default to
            # 0, and the first remote provider (OpenRouter) reported every
            # prompt token of a repeated question as cached -- so a tier priced
            # on input cost nothing for its input, and every counterfactual
            # against it shrank. The input rate never flatters a saving; a
            # cache discount has to be written down to be claimed.
            cache_read=float(raw.get("cache_read", input_price)),
            cache_write=float(raw.get("cache_write", 0.0)),
            configured=True,
        )


@dataclass(frozen=True)
class JevConfig:
    """How a Jev tier turns one probability into one verdict.

    These are not provider knobs and that is why they are not `extra_body`.
    Jev answers "how likely is it that this answer is correct" and the
    verification loop needs PASS or FAIL; `threshold` is where the operator
    draws that line, which is a decision about this router's economics rather
    than about TypeSafe's API. A threshold buried in a dict documented as
    "passed verbatim to the backend request body" would be sent to the provider
    by anyone reading the field it sat in.

    The default of 0.5 is the neutral reading of a calibrated probability and
    not a recommendation. Which way to lean is arithmetic the operator has to
    do: a false FAIL buys an escalation that was not needed, a false PASS ships
    a wrong answer. Those cost different amounts, so the line between them is
    not naturally in the middle.
    """

    threshold: float = 0.5
    # Overrides the reviewer brief that `build_review_request` puts in the
    # system message. None means "use whatever the review carried", which keeps
    # `verification.system_prompt` working for a Jev judge too.
    instructions: str | None = None
    # The key the question is asked and answered under. It appears in the
    # request and in the response, so it only matters that both agree.
    question_key: str = "verdict"
    # Several questions in one call, key -> instructions. `questions` ask "is
    # the answer right in this respect", `error_questions` ask "is there a
    # defect", and are read as 1 - p. Their p(correct) are averaged in logit
    # space and `threshold` applies to that. Empty means the single question.
    #
    # Measured before it was built: on 434 reviewed answers, eight such
    # questions beat the single one by +0.023 AUC, 95% CI [0.008, 0.038], at
    # the price of one call -- the state is read once however many ride on it.
    questions: dict[str, str] = field(default_factory=dict)
    error_questions: dict[str, str] = field(default_factory=dict)

    @property
    def asks_many(self) -> bool:
        return bool(self.questions or self.error_questions)

    @classmethod
    def parse(cls, raw: dict[str, Any] | None, *, tier: str) -> "JevConfig":
        if not raw:
            return cls()
        if not isinstance(raw, dict):
            raise ConfigError(f"tier {tier!r}: 'jev' must be a mapping")
        unknown = set(raw) - {"threshold", "instructions", "question_key", "questions", "error_questions"}
        if unknown:
            raise ConfigError(f"tier {tier!r}: unknown jev fields: {sorted(unknown)}")
        threshold = float(raw.get("threshold", 0.5))
        if not 0.0 <= threshold <= 1.0:
            raise ConfigError(
                f"tier {tier!r}: jev.threshold must be between 0 and 1, got {threshold}"
            )
        instructions = raw.get("instructions")
        if instructions is not None and not str(instructions).strip():
            # An empty string would silently fall through to the built-in
            # default, which is the opposite of what writing one says.
            raise ConfigError(f"tier {tier!r}: jev.instructions is empty")
        question_key = str(raw.get("question_key", "verdict"))
        if not question_key:
            raise ConfigError(f"tier {tier!r}: jev.question_key is empty")
        questions = _jev_questions(raw, "questions", tier=tier)
        error_questions = _jev_questions(raw, "error_questions", tier=tier)
        if "questions" in raw or "error_questions" in raw:
            if not questions and not error_questions:
                raise ConfigError(f"tier {tier!r}: jev.questions and jev.error_questions are both empty")
            both = set(questions) & set(error_questions)
            if both:
                # One key, two meanings: the answer under it could be read
                # either way round, and the combination would be a coin toss.
                raise ConfigError(
                    f"tier {tier!r}: jev keys in both questions and error_questions: {sorted(both)}"
                )
            # Each of these names ONE question. Beside a question set they would
            # be silently ignored, which is the opposite of what writing them says.
            for single in ("instructions", "question_key"):
                if single in raw:
                    raise ConfigError(
                        f"tier {tier!r}: jev.{single} names the single question and cannot "
                        "be combined with jev.questions / jev.error_questions"
                    )
        return cls(
            threshold=threshold,
            instructions=str(instructions) if instructions is not None else None,
            question_key=question_key,
            questions=questions,
            error_questions=error_questions,
        )


def _jev_questions(raw: dict[str, Any], name: str, *, tier: str) -> dict[str, str]:
    value = raw.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"tier {tier!r}: jev.{name} must be a mapping of key to instructions")
    out: dict[str, str] = {}
    for key, text in value.items():
        if not str(key).strip() or text is None or not str(text).strip():
            raise ConfigError(f"tier {tier!r}: jev.{name} has an empty key or empty instructions")
        out[str(key)] = str(text)
    return out


@dataclass(frozen=True)
class TierConfig:
    """One logical tier, mapped to one concrete backend model."""

    name: str
    backend: BackendKind
    model: str
    base_url: str
    api_key_env: str | None = None
    # Total budget in tokens (prompt + generated). Used by the eligibility gate.
    context_window: int = 8192
    supports_tools: bool = False
    prices: Prices = field(default_factory=Prices)
    timeout_s: float = 600.0
    # Passed verbatim to the backend request body, for provider-specific knobs.
    extra_body: dict[str, Any] = field(default_factory=dict)
    # Only meaningful when `backend` is `jev`; parse_config rejects it elsewhere
    # rather than letting a block that does nothing look like it does something.
    jev: JevConfig = field(default_factory=JevConfig)
    # Reasoning level handed to the CLI (`--effort` for claude, `model_reasoning_effort`
    # for codex). None leaves the CLI's own default. One model at two levels is two
    # tiers: the level changes what the answer costs and how good it is, so it is
    # part of the destination a router picks, not a detail of the call.
    effort: str | None = None
    # The subscription whose quota this tier spends, or None for a tier billed per
    # token (or free). Defaults from the backend; two CLI tiers on different
    # accounts can name different pools.
    subscription: str | None = None

    @property
    def can_serve(self) -> bool:
        """Whether this tier can answer a client at all.

        False for a judge like Jev, which returns a typed decision and cannot
        generate text. This is not an eligibility detail, it is the difference
        between the two kinds of number this project prints: a tier that could
        not have served the request must never appear in a counterfactual as a
        cheaper way of having answered it, because it was not a way of
        answering it. An omitted baseline is a gap and an invented one is a
        lie, and the whole claim rests on which of the two these are.
        """
        return self.backend != "jev"

    @property
    def api_key(self) -> str | None:
        """Read the key from the environment at call time.

        Keys are named in config but never stored in it, so a config file can be
        committed and shared without carrying a credential.
        """
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)


@dataclass(frozen=True)
class RouterConfig:
    kind: str = "static"
    default_tier: str | None = None
    # Optional map of client-requested model name -> tier, so an existing client
    # that hardcodes a model string can still steer the router.
    model_map: dict[str, str] = field(default_factory=dict)
    # --- kind: classifier -------------------------------------------------
    # Where a request goes when the model predicts the cheap tier would fail.
    strong_tier: str | None = None
    # Path to the trained model. There is no bundled default and no fallback:
    # a classifier router with no model refuses to start, because one that
    # quietly served every request from the default tier would be
    # indistinguishable from a working one in every report it produced.
    model_path: str | None = None
    # None means "use the threshold the model was trained with". An explicit
    # value overrides it, which is how a tuned threshold is deployed without
    # retraining.
    threshold: float | None = None
    # Fraction of would-be escalations sent to the cheap tier anyway, so they
    # get verified and produce labels. Without it the model destroys its own
    # evidence: every prompt it scores high goes to the strong tier, is never
    # reviewed, and never appears in the next training set -- so the next model
    # is fitted only on prompts the current one already believed were easy. A
    # few percent is the cost of still being able to tell whether it is right.
    explore_rate: float = 0.0
    # --- kind: capabilities -----------------------------------------------
    capabilities: "CapabilitiesConfig | None" = None


# P(the task fails | it needs this, and the model's level at it is 0, 1, 2, 3).
# A prior, written down so it can be argued with; the logged decisions are what
# replace it.
_DEFAULT_MISS = (0.9, 0.5, 0.2, 0.05)


@dataclass(frozen=True)
class ModelCard:
    """What the capabilities router believes about one destination tier."""

    # requirement key -> 0 (cannot), 1 (weak), 2 (good), 3 (excellent), and
    # anything between: `miss` is interpolated. A key the card leaves out is 0:
    # an unexamined claim is not a capability.
    levels: dict[str, float]
    # Output tokens a typical answer costs on this tier. Effort moves it a lot,
    # which is most of what a higher level costs.
    output_tokens: int = 1500
    # Input tokens the backend adds to every call on its own account, in
    # full-price equivalents. `codex exec` sends ~12.5k of its own instructions
    # with a one-word question (measured 2026-09-24: 12499 in, 8960 cached),
    # which costs more than the question on a cheap model.
    input_overhead: int = 0
    # Per 1M tokens, used ONLY by the router, for a tier whose own `prices` are
    # empty: a subscription's quota weight, or a local model's shadow price. Kept
    # off `prices` on purpose: those feed the request log's cost, and neither
    # kind of call costs money at the margin -- a weight written there would be
    # logged as money spent.
    list_prices: Prices = field(default_factory=Prices)
    # The profile it was built from (its `match`), so calibration learnt from
    # outcomes can be kept per family rather than one number for every model.
    family: str | None = None


@dataclass(frozen=True)
class ModelProfile:
    """What a family of models is believed to do, before any effort is applied.

    Matched by glob against a discovered model id, first match wins. Levels are
    at effort `high`; `EffortRule` moves them for the other levels.
    """

    match: str
    levels: dict[str, float]
    output_tokens: int
    list_prices: Prices


@dataclass(frozen=True)
class EffortRule:
    """How one effort level moves a profile: thinking levels by `shift`, output by `output`."""

    shift: float
    output: float


# Relative to `high`, where profiles are written. Thinking requirements only: a
# higher effort reasons longer, it does not know more about a niche framework.
# Output includes the thinking, and thinking is what effort buys: `high` spends
# far more of it than `medium`, which is why their output factors are far apart.
_DEFAULT_EFFORT_RULES = {
    "low": EffortRule(-0.75, 0.2),
    "medium": EffortRule(-0.3, 0.45),
    "high": EffortRule(0.0, 1.0),
    "xhigh": EffortRule(0.25, 1.6),
    "max": EffortRule(0.4, 2.5),
    "ultra": EffortRule(0.5, 3.0),
}
_DEFAULT_THINKING = ("reasoning", "debugging", "math", "ambiguity", "precision")


@dataclass(frozen=True)
class BenchmarksConfig:
    """Where the benchmark evidence lives, and how it becomes a level. See `scores.py`."""

    path: str
    # The level at ability 0 (50% above chance on a benchmark of average
    # difficulty) and the levels per unit of ability. None: the line through the
    # profiles' own levels, so the evidence reorders models without moving their
    # average or spread; `benchmarks fit` then moves it by offsets.
    a: float | None = None
    k: float | None = None
    # The profile's weight in the blend, in evidence units: a level is
    # C/(C + profile_weight) measured and the rest profile. Small, because a
    # written guess is weak next to a measured point.
    profile_weight: float = 0.25
    # A vendor-reported point's evidence, relative to an independent one.
    vendor_weight: float = 0.5
    # Startup runs `benchmarks import` when the import is older than this. None:
    # only on command.
    refresh_hours: float | None = 24.0
    refresh_timeout_s: float = 30.0


def _parse_benchmarks(raw: Any) -> BenchmarksConfig | None:
    if raw is None:
        return None
    where = "router.capabilities.benchmarks"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping")
    extra = set(raw) - {"path", "a", "k", "profile_weight", "vendor_weight", "refresh_hours", "refresh_timeout_s"}
    if extra:
        raise ConfigError(f"{where}: unknown fields {sorted(extra)}")
    if not raw.get("path"):
        raise ConfigError(f"{where}.path is required")
    try:
        a = None if raw.get("a") is None else float(raw["a"])
        k = None if raw.get("k") is None else float(raw["k"])
        weight = float(raw.get("profile_weight", 0.25))
        vendor = float(raw.get("vendor_weight", 0.5))
        refresh = None if raw.get("refresh_hours", 24.0) is None else float(raw.get("refresh_hours", 24.0))
        timeout = float(raw.get("refresh_timeout_s", 30.0))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where}: {exc}") from None
    if a is not None and not 0.0 <= a <= 3.0:
        raise ConfigError(f"{where}.a must be in [0, 3], got {a}")
    if k is not None and not 0.1 <= k <= 4.0:
        raise ConfigError(f"{where}.k must be in [0.1, 4], got {k}")
    if weight <= 0:
        raise ConfigError(f"{where}.profile_weight must be positive, got {weight}")
    if not 0.0 < vendor <= 1.0:
        raise ConfigError(f"{where}.vendor_weight must be in (0, 1], got {vendor}")
    if refresh is not None and refresh <= 0:
        raise ConfigError(f"{where}.refresh_hours must be positive or null, got {refresh}")
    if timeout <= 0:
        raise ConfigError(f"{where}.refresh_timeout_s must be positive, got {timeout}")
    return BenchmarksConfig(str(raw["path"]), a, k, weight, vendor, refresh, timeout)


@dataclass(frozen=True)
class DiscoverSource:
    """A provider whose every served model, at every effort it takes, becomes a tier."""

    name: str
    backend: str
    base_url: str
    context_window: int = 200000
    timeout_s: float = 600.0
    supports_tools: bool = False
    # See ModelCard.input_overhead: the CLI's own prompt, on every call.
    input_overhead: int = 0
    subscription: str | None = None


def _parse_levels(raw: Any, requirements: dict[str, str], where: str, default: float = 0.0) -> dict[str, float]:
    levels = {key: float(default) for key in requirements} if default else {}
    for key, value in (raw or {}).items():
        levels[str(key)] = float(value)
    stray = set(levels) - set(requirements)
    if stray:
        # A level for a requirement nobody asks about is a typo that would
        # otherwise pass as a capability.
        raise ConfigError(f"{where}: levels for unknown requirements {sorted(stray)}")
    if any(not 0 <= v <= 3 for v in levels.values()):
        raise ConfigError(f"{where}: levels must be between 0 and 3")
    return levels


def capped(levels: dict[str, float], ceilings: dict[str, float] | None) -> dict[str, float]:
    """`levels` with each capped requirement lowered to its ceiling; a cap never raises a level."""
    if not ceilings:
        return levels
    return {key: min(value, ceilings[key]) if key in ceilings else value for key, value in levels.items()}


def _parse_level_caps(raw: Any, requirements: dict[str, str]) -> dict[str, dict[str, float]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError("router.capabilities.level_caps must map a family to {requirement: ceiling}")
    out: dict[str, dict[str, float]] = {}
    for family, ceilings in raw.items():
        where = f"router.capabilities.level_caps[{family!r}]"
        if not isinstance(ceilings, dict):
            raise ConfigError(f"{where} must map requirements to ceilings")
        stray = set(ceilings) - set(requirements)
        if stray:
            raise ConfigError(f"{where}: unknown requirements {sorted(stray)}")
        try:
            row = {str(k): float(v) for k, v in ceilings.items()}
        except (TypeError, ValueError):
            raise ConfigError(f"{where}: ceilings must be numbers between 0 and 3") from None
        if any(not 0 <= v <= 3 for v in row.values()):
            raise ConfigError(f"{where}: ceilings must be between 0 and 3")
        out[str(family)] = row
    return out


def _parse_profile(raw: Any, requirements: dict[str, str], where: str, match: str | None = None) -> ModelProfile:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping")
    extra = set(raw) - {"match", "level", "levels", "output_tokens", "list_prices"}
    if extra:
        raise ConfigError(f"{where}: unknown fields {sorted(extra)}")
    pattern = match or raw.get("match")
    if not pattern:
        raise ConfigError(f"{where}: 'match' is required")
    level = float(raw.get("level", 0.0))
    if not 0 <= level <= 3:
        raise ConfigError(f"{where}: level must be between 0 and 3")
    prices = Prices.parse(raw.get("list_prices"))
    if not prices.configured:
        raise ConfigError(f"{where}: list_prices is required; it is the only cost a discovered tier has")
    output_tokens = int(raw.get("output_tokens", 2000))
    if output_tokens <= 0:
        raise ConfigError(f"{where}: output_tokens must be positive")
    return ModelProfile(
        match=str(pattern),
        levels=_parse_levels(raw.get("levels"), requirements, where, default=level),
        output_tokens=output_tokens,
        list_prices=prices,
    )


@dataclass(frozen=True)
class SubscriptionBudget:
    """A subscription's rolling window: how long a 429 keeps it off the table."""

    window_hours: float = 5.0


@dataclass(frozen=True)
class FailureCost:
    """What a wrong answer costs, in multiples of redoing the task on the tier `target` would pick.

    `detected`: the failure is caught -- the client's tests, a verifier -- and
    the price is the redo. `undetected`: it is not, it ships, and the price is the
    redo when it is found later plus what it broke meanwhile. `stakes` scales
    both, from the packet's `stakes` or, failing that, Jev's reading of the
    `precision` requirement.
    """

    detected: float = 1.0
    undetected: float = 4.0
    stakes: dict[str, float] = field(default_factory=lambda: {"low": 0.5, "normal": 1.0, "high": 3.0})


_RULES = ("target", "expected_cost")


@dataclass(frozen=True)
class CapabilitiesConfig:
    """Jev reads a task packet; cards say which tier covers what it needs.

    Jev answers yes/no questions with a probability, so it cannot name a model.
    It is asked what the task REQUIRES, one question per requirement, and the
    choice is arithmetic on those answers and the cards: estimated success
    P(m) = prod_r (1 - p_r * miss[level_{m,r}]), and the cheapest tier with
    P(m) >= target wins. Measured before it was built (llm-router-lab, A1, 11
    models): routing on Jev's reading of the task beat every fixed model and
    mixture at equal cost by +0.083 [+0.067, +0.099].
    """

    jev_tier: str
    requirements: dict[str, str]
    cards: dict[str, ModelCard]
    target: float = 0.8
    miss: tuple[float, float, float, float] = _DEFAULT_MISS
    # Jev's "no" band. A requirement the task does not have still comes back at
    # 0.02-0.15 (measured 2026-09-24 on the dry-run sample), and nine of those
    # multiplied sank every weak model on a rename. p <= floor counts as absent;
    # above it the need is rescaled to (p - floor) / (1 - floor).
    floor: float = 0.2
    subscriptions: dict[str, SubscriptionBudget] = field(default_factory=dict)
    max_packet_chars: int = 6000
    # --- discovery: every model a provider serves, at every effort, is a tier.
    discover: dict[str, DiscoverSource] = field(default_factory=dict)
    profiles: tuple[ModelProfile, ...] = ()
    # For a served model no profile matches. It is still offered -- the
    # catalog is all of what is served -- at levels and prices that make it the
    # last choice rather than the first, and it is reported.
    fallback_profile: ModelProfile | None = None
    effort_rules: dict[str, EffortRule] = field(default_factory=lambda: dict(_DEFAULT_EFFORT_RULES))
    thinking: tuple[str, ...] = ()
    # Every `miss` entry is multiplied by it. Set by `llm-router calibrate` from
    # anchors -- tasks with a tier the operator knows is (or is not) enough.
    miss_scale: float = 1.0
    # The highest effort discovery turns into a tier. The operator's rule, not
    # the catalog's: a level nobody would pay for is not an option to weigh.
    max_effort: str | None = None
    # How a tier is chosen from the estimates. `target`: the cheapest whose
    # success reaches `target`. `expected_cost`: the least cost + (1 - P) x the
    # cost of failing, so the bar moves with what a failure costs on THIS task --
    # low for a caught, cheap-to-redo one, high for one that ships unnoticed.
    rule: str = "target"
    failure: FailureCost = field(default_factory=FailureCost)
    # Per-family `miss_scale`, fitted by `calibrate --from-log` from verified
    # outcomes. A family listed here ignores the global `miss_scale`.
    family_scales: dict[str, float] = field(default_factory=dict)
    # Benchmark evidence that derives discovered cards' levels. None: cards are
    # the profiles and the effort rule alone.
    benchmarks: BenchmarksConfig | None = None
    # family key (a profile's `match`, or a hand-written card's tier name) ->
    # requirement -> the highest level any of its cards may have. Written by
    # `calibrate --anchors` from `insufficient` anchors; applied last to every card.
    level_caps: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Any, tiers: dict[str, "TierConfig"], must_serve: Any) -> "CapabilitiesConfig":
        if not isinstance(raw, dict):
            raise ConfigError("router.kind 'capabilities' requires a router.capabilities mapping")
        unknown = set(raw) - {
            "jev_tier", "requirements", "cards", "target", "miss", "floor", "subscriptions", "max_packet_chars",
            "discover", "profiles", "fallback_profile", "effort_rules", "thinking", "miss_scale", "max_effort",
            "rule", "failure", "family_scales", "benchmarks", "level_caps",
        }
        if unknown:
            raise ConfigError(f"unknown router.capabilities fields: {sorted(unknown)}")
        jev_tier = raw.get("jev_tier")
        if jev_tier not in tiers or tiers[jev_tier].backend != "jev":
            raise ConfigError(f"router.capabilities.jev_tier {jev_tier!r} must name a tier with backend 'jev'")
        requirements = _jev_questions(raw, "requirements", tier="router.capabilities")
        if not requirements:
            raise ConfigError("router.capabilities.requirements is empty")
        level_caps = _parse_level_caps(raw.get("level_caps"), requirements)
        discover = _parse_discover(raw.get("discover"))
        raw_cards = raw.get("cards") or {}
        if not isinstance(raw_cards, dict):
            raise ConfigError("router.capabilities.cards must map tiers to their cards")
        if not raw_cards and not discover:
            raise ConfigError(
                "router.capabilities needs cards for configured tiers, or `discover` sources to build them"
            )
        cards: dict[str, ModelCard] = {}
        for name, card in raw_cards.items():
            if name not in tiers:
                raise ConfigError(f"router.capabilities.cards names unknown tier {name!r}")
            must_serve(name, "router.capabilities.cards")
            card = card or {}
            extra = set(card) - {"levels", "output_tokens", "input_overhead", "list_prices"}
            if extra:
                raise ConfigError(f"router.capabilities.cards[{name!r}]: unknown fields {sorted(extra)}")
            levels = _parse_levels(card.get("levels"), requirements, f"router.capabilities.cards[{name!r}]")
            output_tokens = int(card.get("output_tokens", 1500))
            if output_tokens <= 0:
                raise ConfigError(f"router.capabilities.cards[{name!r}]: output_tokens must be positive")
            input_overhead = int(card.get("input_overhead", 0))
            if input_overhead < 0:
                raise ConfigError(f"router.capabilities.cards[{name!r}]: input_overhead cannot be negative")
            list_prices = Prices.parse(card.get("list_prices"))
            if tiers[name].subscription and not list_prices.configured:
                # Without it every subscription tier weighs zero and the router
                # sends everything to the most capable one.
                raise ConfigError(
                    f"router.capabilities.cards[{name!r}]: a subscription tier needs list_prices "
                    "to weigh its quota"
                )
            cards[name] = ModelCard(
                levels=capped(levels, level_caps.get(name)), output_tokens=output_tokens, input_overhead=input_overhead, list_prices=list_prices
            )
        target = float(raw.get("target", 0.8))
        if not 0.0 < target <= 1.0:
            raise ConfigError(f"router.capabilities.target must be in (0, 1], got {target}")
        miss = tuple(float(v) for v in raw.get("miss", _DEFAULT_MISS))
        if len(miss) != 4 or any(not 0.0 <= v <= 1.0 for v in miss):
            raise ConfigError("router.capabilities.miss must be four probabilities, for levels 0..3")
        floor = float(raw.get("floor", 0.2))
        if not 0.0 <= floor < 1.0:
            raise ConfigError(f"router.capabilities.floor must be in [0, 1), got {floor}")
        subscriptions: dict[str, SubscriptionBudget] = {}
        for name, sub in (raw.get("subscriptions") or {}).items():
            sub = sub or {}
            extra = set(sub) - {"window_hours"}
            if extra:
                raise ConfigError(f"router.capabilities.subscriptions[{name!r}]: unknown fields {sorted(extra)}")
            budget = SubscriptionBudget(
                window_hours=float(sub.get("window_hours", 5.0)),
            )
            if budget.window_hours <= 0:
                raise ConfigError(
                    f"router.capabilities.subscriptions[{name!r}]: window must be positive, budget not negative"
                )
            subscriptions[str(name)] = budget
        max_packet_chars = int(raw.get("max_packet_chars", 6000))
        if max_packet_chars < 500:
            raise ConfigError("router.capabilities.max_packet_chars must be at least 500")
        profiles = tuple(
            _parse_profile(item, requirements, f"router.capabilities.profiles[{i}]")
            for i, item in enumerate(raw.get("profiles") or [])
        )
        fallback = raw.get("fallback_profile")
        fallback_profile = (
            _parse_profile(fallback, requirements, "router.capabilities.fallback_profile", match="*")
            if fallback is not None
            else None
        )
        if discover and fallback_profile is None:
            raise ConfigError(
                "router.capabilities.discover needs a fallback_profile: a served model no profile "
                "matches is still offered, and needs levels and prices to be weighed at all"
            )
        effort_rules = dict(_DEFAULT_EFFORT_RULES)
        for effort, rule in (raw.get("effort_rules") or {}).items():
            if effort not in EFFORTS:
                raise ConfigError(f"router.capabilities.effort_rules: unknown effort {effort!r}")
            rule = rule or {}
            effort_rules[effort] = EffortRule(
                shift=float(rule.get("shift", 0.0)), output=float(rule.get("output", 1.0))
            )
            if effort_rules[effort].output <= 0:
                raise ConfigError(f"router.capabilities.effort_rules[{effort!r}].output must be positive")
        miss_scale = float(raw.get("miss_scale", 1.0))
        if not 0.0 < miss_scale <= 4.0:
            raise ConfigError(f"router.capabilities.miss_scale must be in (0, 4], got {miss_scale}")
        max_effort = raw.get("max_effort")
        if max_effort is not None and max_effort not in EFFORTS:
            raise ConfigError(f"router.capabilities.max_effort must be one of {list(EFFORTS)}, got {max_effort!r}")
        rule = raw.get("rule", "target")
        if rule not in _RULES:
            raise ConfigError(f"router.capabilities.rule must be one of {list(_RULES)}, got {rule!r}")
        failure = _parse_failure(raw.get("failure"))
        family_scales = {str(k): float(v) for k, v in (raw.get("family_scales") or {}).items()}
        if any(not 0.0 < v <= 4.0 for v in family_scales.values()):
            raise ConfigError("router.capabilities.family_scales values must be in (0, 4]")
        thinking = tuple(raw.get("thinking") or [k for k in _DEFAULT_THINKING if k in requirements])
        stray = set(thinking) - set(requirements)
        if stray:
            raise ConfigError(f"router.capabilities.thinking names unknown requirements {sorted(stray)}")
        return cls(
            jev_tier=str(jev_tier),
            requirements=requirements,
            cards=cards,
            target=target,
            miss=miss,  # type: ignore[arg-type]
            floor=floor,
            subscriptions=subscriptions,
            max_packet_chars=max_packet_chars,
            discover=discover,
            profiles=profiles,
            fallback_profile=fallback_profile,
            effort_rules=effort_rules,
            thinking=thinking,
            miss_scale=miss_scale,
            max_effort=max_effort,
            rule=rule,
            failure=failure,
            family_scales=family_scales,
            benchmarks=_parse_benchmarks(raw.get("benchmarks")),
            level_caps=level_caps,
        )


def _parse_failure(raw: Any) -> FailureCost:
    if raw is None:
        return FailureCost()
    if not isinstance(raw, dict) or set(raw) - {"detected", "undetected", "stakes"}:
        raise ConfigError("router.capabilities.failure takes only 'detected', 'undetected' and 'stakes'")
    base = FailureCost()
    stakes = dict(base.stakes)
    stakes.update({str(k): float(v) for k, v in (raw.get("stakes") or {}).items()})
    failure = FailureCost(
        detected=float(raw.get("detected", base.detected)),
        undetected=float(raw.get("undetected", base.undetected)),
        stakes=stakes,
    )
    if failure.detected < 0 or failure.undetected < 0 or any(v < 0 for v in stakes.values()):
        raise ConfigError("router.capabilities.failure: costs cannot be negative")
    if not {"low", "normal", "high"} <= set(stakes):
        raise ConfigError("router.capabilities.failure.stakes needs low, normal and high")
    return failure


def _parse_discover(raw: Any) -> dict[str, DiscoverSource]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError("router.capabilities.discover must map a name to a provider")
    out: dict[str, DiscoverSource] = {}
    for name, entry in raw.items():
        where = f"router.capabilities.discover[{name!r}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"{where} must be a mapping")
        extra = set(entry) - {
            "backend", "base_url", "context_window", "timeout_s", "supports_tools", "input_overhead", "subscription",
        }
        if extra:
            raise ConfigError(f"{where}: unknown fields {sorted(extra)}")
        backend = entry.get("backend")
        if backend not in ("claude_cli", "codex_cli", "ollama"):
            # The three that publish a catalog this router can read for free.
            raise ConfigError(f"{where}: backend must be claude_cli, codex_cli or ollama, got {backend!r}")
        if ":" in str(name):
            raise ConfigError(f"{where}: the name prefixes tier names and cannot contain ':'")
        out[str(name)] = DiscoverSource(
            name=str(name),
            backend=backend,
            base_url=str(entry.get("base_url") or _DEFAULT_BASE_URLS[backend]).rstrip("/"),
            context_window=int(entry.get("context_window", 200000)),
            timeout_s=float(entry.get("timeout_s", 600.0)),
            supports_tools=bool(entry.get("supports_tools", False)),
            input_overhead=int(entry.get("input_overhead", 0)),
            subscription=entry.get("subscription", _SUBSCRIPTION_BACKENDS.get(backend)),
        )
    return out


@dataclass(frozen=True)
class VerificationConfig:
    """When a stronger tier checks a cheaper tier's answer.

    Off by default, and that default is not shyness: verification adds a second
    call to every request it touches, so turning it on changes the bill. The
    knobs below exist so the operator decides how much, rather than discovering
    it in an invoice.
    """

    enabled: bool = False
    # Who judges. Required when enabled.
    verifier_tier: str | None = None
    # A cheaper judge asked FIRST, whose PASS is final and whose anything-else
    # hands the review to `verifier_tier`. It can spare the verifier a call and
    # can never fail an answer by itself: wrong, unsure or down, it costs only
    # the saving. Meant for a Jev tier, which answers in ~300ms for a
    # hundredth of a cent -- and which, measured as the only judge against an
    # Opus 5 reference, reached AUC ~0.78: good enough to wave through what it
    # is sure of, not good enough to escalate on. How much it passes is its
    # own `jev.threshold`, and that line is the leak rate the operator accepts.
    prefilter_tier: str | None = None
    # Which served tiers get checked. Empty means "every tier except the
    # verifier" -- asking a tier to review its own answer measures nothing.
    verify_tiers: frozenset[str] = frozenset()
    # Who re-answers when the verdict is FAIL. Defaults to the verifier, which
    # has already read the question.
    escalate_to: str | None = None
    # Fraction of eligible answers actually checked. The loop is most useful at
    # 1.0 while it is producing training labels, and cheapest sampled down once
    # the failure rate is known.
    sample_rate: float = 1.0
    # What to do when the verifier's reply cannot be parsed, and when the
    # verifier call itself fails. `accept` keeps the cheap answer; `escalate`
    # pays for a second answer. The default refuses to spend money on a broken
    # verifier -- but `stats` counts both, so the choice stays visible.
    on_unparseable: str = "accept"
    on_verifier_error: str = "accept"
    # Ask the SAME tier once more before paying a stronger one, when the answer
    # was failed without a review -- empty, or cut off at the output budget.
    # Measured on 400 GSM8K questions through qwen3.5:4b: of 60 answers that ran
    # past the budget without finishing, 27 were answered inside that same
    # budget on the next ask, several in under 300 tokens. Nothing about the
    # question had changed; the first sample simply ran away.
    #
    # Off by default because whether it pays is arithmetic, not a preference. A
    # retry is worth it when the cheap tier costs less than (retry success rate)
    # x (the strong tier): at 45% success, a cheap tier at a fiftieth of the
    # strong tier's price is free money, and one at half the price is a loss.
    retry_unfinished: bool = False
    # Truncation bounds for what the verifier is shown. A review prompt that
    # grows without limit is how a verification loop silently becomes the most
    # expensive call in the system.
    max_transcript_chars: int = 12000
    max_answer_chars: int = 8000
    # A ceiling, not a spend: a judge that answers directly stops by itself at
    # 6-26 tokens whatever this says. It only binds on a THINKING judge, which
    # spends it reasoning before it writes a word -- and a judge cut off there
    # gives no verdict, which `on_unparseable: accept` turns into a pass the
    # judge never made. So the ceiling is set for the thinking judge.
    #
    # Measured 2026-09-19. qwen3.5:4b (local): at 200, no verdict at all, every
    # time; with thinking on it needed 490-1054. deepseek-v4-flash (remote, via
    # OpenRouter): at 200, 10/10 planted cases right but one real answer cut
    # off unjudged; at 1024 every verdict finished, in 54-183. With reasoning
    # switched off both models answer in ~6 tokens and both waved a wrong
    # answer through (17 * 23 = 401 on one, "strawberry has two r's" on the
    # other) -- a judge that cannot think cannot check. 1024 at the remote
    # model's list price ($0.08/M out) is under a hundredth of a cent a review.
    max_verdict_tokens: int = 1024
    # Override the reviewer instructions. None uses the built-in prompt.
    system_prompt: str | None = None

    def verifies(self, tier: str) -> bool:
        if not self.enabled or tier in (self.verifier_tier, self.prefilter_tier):
            return False
        return not self.verify_tiers or tier in self.verify_tiers


@dataclass(frozen=True)
class LogConfig:
    path: str = "llm-router.db"
    # Prompts are user data. The default stores only a SHA-256 hash, which is
    # enough to spot repeats and correlate reports without holding the content.
    store_prompts: bool = False


@dataclass(frozen=True)
class CatalogConfig:
    """The startup check of configured tiers against what each provider serves.

    On by default: it reads local caches and asks each CLI for its catalog, and
    none of that spends quota. See `catalog.py`.
    """

    check_on_start: bool = True
    # Where the report is kept, rewritten only when it changes. None: not kept.
    path: str | None = "llm-router.catalog.json"
    check_ollama: bool = True
    # Overrides for where the CLIs keep their state; default ~/.claude, ~/.codex.
    claude_dir: str | None = None
    codex_home: str | None = None


@dataclass(frozen=True)
class TraceConfig:
    """The live routing view at /routing: the last `keep` requests, in memory only."""

    enabled: bool = True
    keep: int = 200


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080


@dataclass(frozen=True)
class Config:
    tiers: dict[str, TierConfig]
    router: RouterConfig = field(default_factory=RouterConfig)
    log: LogConfig = field(default_factory=LogConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)

    def tier(self, name: str) -> TierConfig:
        try:
            return self.tiers[name]
        except KeyError:
            raise ConfigError(f"unknown tier: {name!r}") from None


class ConfigError(Exception):
    """Raised for a config file the router cannot act on."""


_DEFAULT_BASE_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434",
    # Jev has one endpoint and one vendor, so naming it here saves every config
    # a line. Still overridable: Cloudflare Workers AI fronts the same model.
    "jev": "https://api.typesafe.ai/v1",
    # Not a URL: for this backend `base_url` names the executable to run.
    "claude_cli": "claude",
    "codex_cli": "codex",
}


def parse_config(raw: dict[str, Any]) -> Config:
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")

    raw_tiers = raw.get("tiers") or {}
    if not raw_tiers:
        raise ConfigError("config must declare at least one tier under 'tiers'")

    tiers: dict[str, TierConfig] = {}
    for name, entry in raw_tiers.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"tier {name!r} must be a mapping")
        backend = entry.get("backend")
        if backend not in ("ollama", "openai_compatible", "jev", "claude_cli", "codex_cli"):
            raise ConfigError(
                f"tier {name!r}: backend must be 'ollama', 'openai_compatible', "
                f"'jev', 'claude_cli' or 'codex_cli', got {backend!r}"
            )
        effort = entry.get("effort")
        if effort is not None:
            if backend not in _SUBSCRIPTION_BACKENDS:
                # Other backends take reasoning knobs through `extra_body`, in
                # each provider's own spelling; a field that did nothing there
                # would read as if it did.
                raise ConfigError(
                    f"tier {name!r}: 'effort' applies to claude_cli and codex_cli tiers; "
                    f"use extra_body for {backend!r}"
                )
            if effort not in EFFORTS:
                raise ConfigError(f"tier {name!r}: effort must be one of {list(EFFORTS)}, got {effort!r}")
        subscription = entry.get("subscription", _SUBSCRIPTION_BACKENDS.get(backend))
        if "jev" in entry and backend != "jev":
            raise ConfigError(
                f"tier {name!r}: 'jev' settings on a {backend!r} tier do nothing"
            )
        model = entry.get("model")
        if not model:
            raise ConfigError(f"tier {name!r}: 'model' is required")
        base_url = entry.get("base_url") or _DEFAULT_BASE_URLS.get(backend)
        if not base_url:
            raise ConfigError(f"tier {name!r}: 'base_url' is required for {backend}")
        tiers[name] = TierConfig(
            name=name,
            backend=backend,
            model=str(model),
            base_url=str(base_url).rstrip("/"),
            api_key_env=entry.get("api_key_env"),
            context_window=int(entry.get("context_window", 8192)),
            supports_tools=bool(entry.get("supports_tools", False)),
            prices=Prices.parse(entry.get("prices")),
            timeout_s=float(entry.get("timeout_s", 600.0)),
            extra_body=dict(entry.get("extra_body") or {}),
            jev=JevConfig.parse(entry.get("jev"), tier=name),
            effort=effort,
            subscription=str(subscription) if subscription else None,
        )

    def must_serve(tier_name: str, where: str) -> None:
        """Refuse a judge-only tier anywhere an answer is expected.

        Caught here rather than at the first request, because the failure it
        prevents is silent: a Jev tier named as `default_tier` would be asked
        to write prose it structurally cannot write, once per request, forever.
        """
        if not tiers[tier_name].can_serve:
            raise ConfigError(
                f"{where} {tier_name!r} uses backend {tiers[tier_name].backend!r}, "
                f"which returns a typed decision and cannot answer a request; "
                f"it can only verify"
            )

    raw_router = raw.get("router") or {}
    default_tier = raw_router.get("default_tier") or next(iter(tiers))
    if default_tier not in tiers:
        raise ConfigError(f"router.default_tier {default_tier!r} is not a configured tier")
    must_serve(default_tier, "router.default_tier")
    model_map = dict(raw_router.get("model_map") or {})
    for requested, target in model_map.items():
        if target not in tiers:
            raise ConfigError(
                f"router.model_map[{requested!r}] points at unknown tier {target!r}"
            )
        must_serve(target, f"router.model_map[{requested!r}]")
    kind = str(raw_router.get("kind", "static"))
    if kind not in ("static", "classifier", "capabilities"):
        raise ConfigError(
            f"router.kind must be 'static', 'classifier' or 'capabilities', got {kind!r}"
        )
    if "capabilities" in raw_router and kind != "capabilities":
        raise ConfigError(f"router.capabilities does nothing with router.kind {kind!r}")
    capabilities = (
        CapabilitiesConfig.parse(raw_router.get("capabilities"), tiers, must_serve)
        if kind == "capabilities"
        else None
    )

    strong_tier = raw_router.get("strong_tier")
    if strong_tier and strong_tier not in tiers:
        raise ConfigError(f"router.strong_tier {strong_tier!r} is not a configured tier")
    if strong_tier:
        must_serve(strong_tier, "router.strong_tier")
    threshold = raw_router.get("threshold")
    if threshold is not None:
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ConfigError(f"router.threshold must be in [0, 1], got {threshold}")
    explore_rate = float(raw_router.get("explore_rate", 0.0))
    if not 0.0 <= explore_rate <= 1.0:
        raise ConfigError(f"router.explore_rate must be in [0, 1], got {explore_rate}")
    if kind == "classifier":
        # Caught here rather than at the first request: a proxy that accepts a
        # config it cannot route with has already started answering by the time
        # anyone finds out.
        if not strong_tier:
            raise ConfigError("router.kind 'classifier' requires router.strong_tier")
        if strong_tier == default_tier:
            raise ConfigError(
                f"router.strong_tier and router.default_tier are both "
                f"{strong_tier!r}; there is nothing to escalate to"
            )
        if not raw_router.get("model_path"):
            raise ConfigError("router.kind 'classifier' requires router.model_path")

    router = RouterConfig(
        kind=kind,
        default_tier=default_tier,
        model_map=model_map,
        strong_tier=strong_tier,
        model_path=(str(raw_router["model_path"]) if raw_router.get("model_path") else None),
        threshold=threshold,
        explore_rate=explore_rate,
        capabilities=capabilities,
    )

    raw_log = raw.get("log") or {}
    log = LogConfig(
        path=str(raw_log.get("path", "llm-router.db")),
        store_prompts=bool(raw_log.get("store_prompts", False)),
    )

    raw_server = raw.get("server") or {}
    server = ServerConfig(
        host=str(raw_server.get("host", "127.0.0.1")),
        port=int(raw_server.get("port", 8080)),
    )

    verification = _parse_verification(raw.get("verification"), tiers)

    raw_catalog = raw.get("catalog") or {}
    if not isinstance(raw_catalog, dict):
        raise ConfigError("'catalog' must be a mapping")
    unknown = set(raw_catalog) - {"check_on_start", "path", "check_ollama", "claude_dir", "codex_home"}
    if unknown:
        raise ConfigError(f"unknown catalog fields: {sorted(unknown)}")
    catalog = CatalogConfig(
        check_on_start=bool(raw_catalog.get("check_on_start", True)),
        # An explicit empty path turns the file off; an omitted one keeps the default.
        path=(str(raw_catalog["path"]) or None) if raw_catalog.get("path") is not None else (
            None if "path" in raw_catalog else CatalogConfig.path
        ),
        check_ollama=bool(raw_catalog.get("check_ollama", True)),
        claude_dir=raw_catalog.get("claude_dir"),
        codex_home=raw_catalog.get("codex_home"),
    )

    raw_trace = raw.get("trace") or {}
    if not isinstance(raw_trace, dict) or set(raw_trace) - {"enabled", "keep"}:
        raise ConfigError("'trace' takes only 'enabled' and 'keep'")
    trace = TraceConfig(enabled=bool(raw_trace.get("enabled", True)), keep=int(raw_trace.get("keep", 200)))
    if trace.keep < 1:
        raise ConfigError("trace.keep must be at least 1")

    return Config(
        tiers=tiers,
        router=router,
        log=log,
        server=server,
        verification=verification,
        catalog=catalog,
        trace=trace,
    )


_FALLBACK_POLICIES = ("accept", "escalate")


def _parse_verification(
    raw: dict[str, Any] | None, tiers: dict[str, TierConfig]
) -> VerificationConfig:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError("'verification' must be a mapping")
    unknown = set(raw) - {
        "enabled",
        "verifier_tier",
        "prefilter_tier",
        "verify_tiers",
        "escalate_to",
        "sample_rate",
        "on_unparseable",
        "on_verifier_error",
        "retry_unfinished",
        "max_transcript_chars",
        "max_answer_chars",
        "max_verdict_tokens",
        "system_prompt",
    }
    if unknown:
        raise ConfigError(f"unknown verification fields: {sorted(unknown)}")

    enabled = bool(raw.get("enabled", False))
    verifier_tier = raw.get("verifier_tier")
    if enabled and not verifier_tier:
        raise ConfigError("verification.enabled requires 'verifier_tier'")
    if verifier_tier and verifier_tier not in tiers:
        raise ConfigError(
            f"verification.verifier_tier {verifier_tier!r} is not a configured tier"
        )

    prefilter_tier = raw.get("prefilter_tier")
    if prefilter_tier:
        if not enabled:
            raise ConfigError("verification.prefilter_tier requires verification.enabled")
        if prefilter_tier not in tiers:
            raise ConfigError(
                f"verification.prefilter_tier {prefilter_tier!r} is not a configured tier"
            )
        if prefilter_tier == verifier_tier:
            # It would ask the same judge twice and call the second answer a
            # second opinion.
            raise ConfigError(
                f"verification.prefilter_tier {prefilter_tier!r} is the same tier as "
                "verifier_tier; a prefilter is a different, cheaper judge"
            )

    verify_tiers = frozenset(raw.get("verify_tiers") or ())
    unknown_tiers = verify_tiers - set(tiers)
    if unknown_tiers:
        raise ConfigError(
            f"verification.verify_tiers names unknown tier(s): {sorted(unknown_tiers)}"
        )

    escalate_to = raw.get("escalate_to") or verifier_tier
    # `auto`: the router picks the tier, among those it believes stronger than
    # the one that failed -- the cheap-first cascade. Needs a router that can.
    if escalate_to == "auto":
        pass
    elif escalate_to and escalate_to not in tiers:
        raise ConfigError(
            f"verification.escalate_to {escalate_to!r} is not a configured tier"
        )
    if escalate_to and escalate_to != "auto" and not tiers[escalate_to].can_serve:
        # `escalate_to` defaults to the verifier because a judge that has
        # already read the question is the obvious tier to re-answer it. That
        # default is wrong for a judge which cannot answer anything, and the
        # failure would only appear on the first FAILED verdict -- long after
        # startup, and only for the requests that went wrong. So it is refused
        # here, and the message says which of the two mistakes was made.
        if not raw.get("escalate_to"):
            raise ConfigError(
                f"verification.verifier_tier {escalate_to!r} returns a typed decision "
                f"and cannot re-answer a request, so 'escalate_to' has no default "
                f"here: name the tier that answers a failed verdict"
            )
        raise ConfigError(
            f"verification.escalate_to {escalate_to!r} uses backend "
            f"{tiers[escalate_to].backend!r}, which returns a typed decision and "
            f"cannot answer a request"
        )

    sample_rate = float(raw.get("sample_rate", 1.0))
    if not 0.0 <= sample_rate <= 1.0:
        raise ConfigError(f"verification.sample_rate must be in [0, 1], got {sample_rate}")

    for key in ("on_unparseable", "on_verifier_error"):
        value = str(raw.get(key, "accept"))
        if value not in _FALLBACK_POLICIES:
            raise ConfigError(
                f"verification.{key} must be one of {list(_FALLBACK_POLICIES)}, got {value!r}"
            )

    return VerificationConfig(
        enabled=enabled,
        verifier_tier=verifier_tier,
        prefilter_tier=prefilter_tier or None,
        verify_tiers=verify_tiers,
        escalate_to=escalate_to,
        sample_rate=sample_rate,
        on_unparseable=str(raw.get("on_unparseable", "accept")),
        on_verifier_error=str(raw.get("on_verifier_error", "accept")),
        retry_unfinished=bool(raw.get("retry_unfinished", False)),
        max_transcript_chars=int(raw.get("max_transcript_chars", 12000)),
        max_answer_chars=int(raw.get("max_answer_chars", 8000)),
        max_verdict_tokens=int(raw.get("max_verdict_tokens", 1024)),
        system_prompt=raw.get("system_prompt"),
    )


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return parse_config(raw)
