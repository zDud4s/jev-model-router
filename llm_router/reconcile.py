"""Ask the provider what it charged, after the fact.

The request path records the provider's bill when the response carries one
(`usage.cost` on OpenRouter). Two cases leave that column empty and this fills
them:

- **A stream the client abandoned.** Usage arrives in the last frame, so a
  client that hung up first leaves a row with zero tokens -- costed at zero,
  and counterfactually free at every tier -- for a call the provider billed.
- **A response that simply did not say.** Rare on OpenRouter, but the column
  is only trustworthy if a gap in it can be closed rather than explained away.

Only providers with a per-request lookup can be reconciled. Today that is
OpenRouter (`GET /generation?id=`). OpenAI and Anthropic report cost per DAY
through admin-key endpoints, which can check a day's total and never a row;
their rows are counted as unsupported rather than skipped silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from .config import Config, TierConfig
from .db import RequestLog
from .pricing import cost_usd
from .schemas import Usage

# Returns the provider's record for one call, or None when it has none yet --
# OpenRouter answers 404 for a few seconds after a generation finishes.
Fetch = Callable[[TierConfig, str], "dict[str, Any] | None"]


class ReconcileError(Exception):
    pass


def supports_lookup(tier: TierConfig) -> bool:
    return tier.backend == "openai_compatible" and "openrouter.ai" in tier.base_url


def openrouter_fetch(tier: TierConfig, upstream_id: str) -> dict[str, Any] | None:
    response = httpx.get(
        f"{tier.base_url}/generation",
        params={"id": upstream_id},
        headers={"Authorization": f"Bearer {tier.api_key}"} if tier.api_key else {},
        timeout=tier.timeout_s,
    )
    if response.status_code == 404:
        return None
    if response.status_code >= 400:
        raise ReconcileError(f"{tier.name}: generation lookup returned {response.status_code}")
    return response.json().get("data")


@dataclass
class ReconcileReport:
    checked: int = 0
    filled: int = 0
    # Rows whose usage never arrived and whose tokens came from the provider.
    tokens_recovered: int = 0
    not_yet: int = 0
    unsupported: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# A row qualifies only when the routed call was the ONLY call. A reviewed or
# escalated request made calls this lookup knows nothing about (their ids were
# never kept), and filling its bill from one of them would be the partial bill
# `billed_total` refuses to write on the request path.
_CANDIDATES = """
    SELECT r.id, r.tier, r.upstream_id, r.input_tokens, r.output_tokens
    FROM requests r
    LEFT JOIN verifications v ON v.request_row_id = r.id
    WHERE r.billed_cost_usd IS NULL
      AND r.upstream_id IS NOT NULL
      AND (v.request_row_id IS NULL OR (v.verdict = 'skipped' AND v.escalated = 0))
    ORDER BY r.id
    LIMIT ?
"""


def reconcile(
    config: Config,
    log: RequestLog,
    *,
    fetch: Fetch = openrouter_fetch,
    limit: int = 500,
    dry_run: bool = False,
) -> ReconcileReport:
    report = ReconcileReport()
    for row in log.query(_CANDIDATES, (limit,)):
        tier = config.tiers.get(row["tier"])
        if tier is None or not supports_lookup(tier):
            report.unsupported[row["tier"]] = report.unsupported.get(row["tier"], 0) + 1
            continue
        report.checked += 1
        try:
            data = fetch(tier, row["upstream_id"])
        except Exception as exc:  # noqa: BLE001 - one bad row must not stop the rest
            report.errors.append(f"row {row['id']}: {type(exc).__name__}: {exc}")
            continue
        if not data or data.get("total_cost") is None:
            report.not_yet += 1
            continue

        usage: Usage | None = None
        if row["input_tokens"] == 0 and row["output_tokens"] == 0:
            usage = Usage(
                prompt_tokens=int(data.get("native_tokens_prompt") or 0),
                completion_tokens=int(data.get("native_tokens_completion") or 0),
                cached_tokens=int(data.get("native_tokens_cached") or 0),
            )
            report.tokens_recovered += 1
        report.filled += 1
        if not dry_run:
            log.apply_billing(
                row["id"],
                billed_usd=float(data["total_cost"]),
                usage=usage,
                costs=None
                if usage is None
                else {name: cost_usd(t.prices, usage) for name, t in config.tiers.items()},
                served_tier=row["tier"],
            )
    return report


def format_report(report: ReconcileReport, *, dry_run: bool) -> str:
    lines = [
        f"checked {report.checked} row(s) with the provider"
        + (" (dry run: nothing written)" if dry_run else ""),
        f"  billed cost filled   {report.filled}",
        f"  tokens recovered     {report.tokens_recovered}"
        "  (usage never arrived: an abandoned stream)",
        f"  not yet available    {report.not_yet}  (run again in a minute)",
    ]
    for tier, count in sorted(report.unsupported.items()):
        lines.append(
            f"  UNSUPPORTED {tier}: {count} row(s) -- this provider has no per-request "
            "lookup; only a daily total can check it"
        )
    for error in report.errors:
        lines.append(f"  ERROR {error}")
    return "\n".join(lines)
