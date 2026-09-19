"""Compare the configured prices with the provider's own price list.

Every cost the router reports -- the estimate, every counterfactual, every
saving -- is the token count times a price somebody typed into the config.
`stats` can show drift against the invoice only after the money is spent; this
catches a wrong or stale price before any traffic runs on it.

Only OpenRouter publishes a machine-readable list (`GET /models`, public, no
key, no quota). Other providers are reported as unchecked rather than passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from .config import Config, TierConfig
from .reconcile import supports_lookup

# base_url -> {model id -> pricing block}
Catalog = Callable[[str], "dict[str, dict[str, Any]]"]



def openrouter_catalog(base_url: str) -> dict[str, dict[str, Any]]:
    response = httpx.get(f"{base_url}/models", timeout=30)
    response.raise_for_status()
    return {m["id"]: m.get("pricing") or {} for m in response.json()["data"]}


def _per_million(value: Any) -> float:
    # The catalog prices per token, as strings.
    return round(float(value) * 1_000_000, 6)


def listed_prices(pricing: dict[str, Any]) -> dict[str, float]:
    """The catalog's prices in the config's unit, USD per 1M tokens."""
    prompt = _per_million(pricing.get("prompt", 0))
    return {
        "input": prompt,
        "output": _per_million(pricing.get("completion", 0)),
        # No cache price listed means the provider bills a cached token as an
        # ordinary one -- the same default `Prices.parse` applies.
        "cache_read": _per_million(pricing["input_cache_read"])
        if "input_cache_read" in pricing
        else prompt,
        "cache_write": _per_million(pricing.get("input_cache_write", 0)),
    }


@dataclass
class TierCheck:
    tier: str
    model: str
    status: str  # ok | mismatch | unpriced | not_listed | unchecked
    # field -> (configured, listed), only for fields that disagree
    differences: dict[str, tuple[float, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class PriceReport:
    tiers: list[TierCheck] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and all(t.status in ("ok", "unchecked") for t in self.tiers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tiers": [t.__dict__ for t in self.tiers],
            "errors": self.errors,
        }


def _compare(tier: TierConfig, pricing: dict[str, Any]) -> TierCheck:
    listed = listed_prices(pricing)
    check = TierCheck(tier=tier.name, model=tier.model, status="ok")
    if not tier.prices.configured:
        # A tier with no prices costs zero by design -- right for a local model,
        # wrong for one the provider charges for.
        if any(listed.values()):
            check.status = "unpriced"
            check.differences = {k: (0.0, v) for k, v in listed.items() if v}
    else:
        for ours in listed:
            configured = round(getattr(tier.prices, ours), 6)
            if configured != listed[ours]:
                check.differences[ours] = (configured, listed[ours])
        if check.differences:
            check.status = "mismatch"
    if pricing.get("overrides"):
        # A long prompt is billed at a higher rate the config cannot express.
        thresholds = sorted(o.get("min_prompt_tokens", 0) for o in pricing["overrides"])
        check.notes.append(
            f"the provider charges more above {thresholds[0]} prompt tokens; "
            "the config prices every request at the base rate"
        )
    reasoning = pricing.get("internal_reasoning")
    if reasoning and float(reasoning) and float(reasoning) != float(pricing.get("completion", 0)):
        check.notes.append(
            f"reasoning tokens are billed at {_per_million(reasoning)}/M, "
            "not at the output rate"
        )
    return check


def check_prices(config: Config, *, catalog: Catalog = openrouter_catalog) -> PriceReport:
    report = PriceReport()
    fetched: dict[str, dict[str, dict[str, Any]] | None] = {}
    for tier in config.tiers.values():
        if not supports_lookup(tier):
            report.tiers.append(TierCheck(tier=tier.name, model=tier.model, status="unchecked"))
            continue
        if tier.base_url not in fetched:
            try:
                fetched[tier.base_url] = catalog(tier.base_url)
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                fetched[tier.base_url] = None
                report.errors.append(f"{tier.base_url}/models: {type(exc).__name__}: {exc}")
        models = fetched[tier.base_url]
        if models is None:
            continue
        if tier.model not in models:
            report.tiers.append(TierCheck(tier=tier.name, model=tier.model, status="not_listed"))
            continue
        report.tiers.append(_compare(tier, models[tier.model]))
    return report


_LABELS = {
    "ok": "ok",
    "mismatch": "MISMATCH",
    "unpriced": "UNPRICED",
    "not_listed": "NOT LISTED",
    "unchecked": "unchecked",
}


def format_report(report: PriceReport) -> str:
    lines = ["prices against the provider's list (USD per 1M tokens):"]
    for t in report.tiers:
        line = f"  {_LABELS[t.status]:<10} {t.tier} ({t.model})"
        if t.status == "unchecked":
            line += " -- this provider publishes no price list"
        elif t.status == "not_listed":
            line += " -- no such model id in the provider's list"
        elif t.status == "unpriced":
            line += " -- no prices configured, so every call is costed at zero"
        lines.append(line)
        for name, (configured, listed) in t.differences.items():
            lines.append(f"             {name}: config {configured:g}, provider {listed:g}")
        for note in t.notes:
            lines.append(f"             note: {note}")
    for error in report.errors:
        lines.append(f"  ERROR {error}")
    return "\n".join(lines)

