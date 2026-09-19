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

BackendKind = Literal["ollama", "openai_compatible"]


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
        return cls(
            input=float(raw.get("input", 0.0)),
            output=float(raw.get("output", 0.0)),
            cache_read=float(raw.get("cache_read", 0.0)),
            cache_write=float(raw.get("cache_write", 0.0)),
            configured=True,
        )


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
    # Truncation bounds for what the verifier is shown. A review prompt that
    # grows without limit is how a verification loop silently becomes the most
    # expensive call in the system.
    max_transcript_chars: int = 12000
    max_answer_chars: int = 8000
    # The verdict is one line plus a reason; it does not need a large budget.
    max_verdict_tokens: int = 200
    # Override the reviewer instructions. None uses the built-in prompt.
    system_prompt: str | None = None

    def verifies(self, tier: str) -> bool:
        if not self.enabled or tier == self.verifier_tier:
            return False
        return not self.verify_tiers or tier in self.verify_tiers


@dataclass(frozen=True)
class LogConfig:
    path: str = "llm-router.db"
    # Prompts are user data. The default stores only a SHA-256 hash, which is
    # enough to spot repeats and correlate reports without holding the content.
    store_prompts: bool = False


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

    def tier(self, name: str) -> TierConfig:
        try:
            return self.tiers[name]
        except KeyError:
            raise ConfigError(f"unknown tier: {name!r}") from None


class ConfigError(Exception):
    """Raised for a config file the router cannot act on."""


_DEFAULT_BASE_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434",
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
        if backend not in ("ollama", "openai_compatible"):
            raise ConfigError(
                f"tier {name!r}: backend must be 'ollama' or 'openai_compatible', "
                f"got {backend!r}"
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
        )

    raw_router = raw.get("router") or {}
    default_tier = raw_router.get("default_tier") or next(iter(tiers))
    if default_tier not in tiers:
        raise ConfigError(f"router.default_tier {default_tier!r} is not a configured tier")
    model_map = dict(raw_router.get("model_map") or {})
    for requested, target in model_map.items():
        if target not in tiers:
            raise ConfigError(
                f"router.model_map[{requested!r}] points at unknown tier {target!r}"
            )
    kind = str(raw_router.get("kind", "static"))
    if kind not in ("static", "classifier"):
        raise ConfigError(f"router.kind must be 'static' or 'classifier', got {kind!r}")

    strong_tier = raw_router.get("strong_tier")
    if strong_tier and strong_tier not in tiers:
        raise ConfigError(f"router.strong_tier {strong_tier!r} is not a configured tier")
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

    return Config(
        tiers=tiers, router=router, log=log, server=server, verification=verification
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
        "verify_tiers",
        "escalate_to",
        "sample_rate",
        "on_unparseable",
        "on_verifier_error",
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

    verify_tiers = frozenset(raw.get("verify_tiers") or ())
    unknown_tiers = verify_tiers - set(tiers)
    if unknown_tiers:
        raise ConfigError(
            f"verification.verify_tiers names unknown tier(s): {sorted(unknown_tiers)}"
        )

    escalate_to = raw.get("escalate_to") or verifier_tier
    if escalate_to and escalate_to not in tiers:
        raise ConfigError(
            f"verification.escalate_to {escalate_to!r} is not a configured tier"
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
        verify_tiers=verify_tiers,
        escalate_to=escalate_to,
        sample_rate=sample_rate,
        on_unparseable=str(raw.get("on_unparseable", "accept")),
        on_verifier_error=str(raw.get("on_verifier_error", "accept")),
        max_transcript_chars=int(raw.get("max_transcript_chars", 12000)),
        max_answer_chars=int(raw.get("max_answer_chars", 8000)),
        max_verdict_tokens=int(raw.get("max_verdict_tokens", 200)),
        system_prompt=raw.get("system_prompt"),
    )


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return parse_config(raw)
