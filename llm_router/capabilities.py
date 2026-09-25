"""The capabilities router: Jev reads what a task needs, the cards say who covers it.

Jev answers yes/no questions with a calibrated probability and cannot name a
model, so the choice is split in two, and each half can be checked alone:

1. **Read the task.** A short packet -- the task itself, trimmed, plus what the
   request's shape says about it -- goes to Jev with one question per
   requirement ("does this need deep reasoning?", "several files?"). One call,
   every requirement answered.
2. **Pick a destination.** Each tier the router may use has a card: a level from
   0 to 3 per requirement. With n_r = max(0, p_r - floor) / (1 - floor), the
   strength of each need once Jev's "no" band is taken off, estimated success
   on tier m is P(m) = prod_r (1 - n_r * miss[level_{m,r}]), and the cheapest
   tier with P(m) >= target wins; when none reaches it, the most likely one does.

The lab result this rests on (llm-router-lab, A1, 11 destination models, 2000
held-out tasks): routing on Jev's reading beat every fixed model and mixture at
equal cost by +0.083 [+0.067, +0.099], and nearly all of that gain was Jev
recognising the KIND of task. So the requirements name kinds of work, and the
cards -- hand-written priors today -- are what the logged decisions will
recalibrate. Every decision logs Jev's answers and the card fingerprint, which is
exactly what that needs.

Cost is one currency, the price per token: tokens in (estimated) plus the
card's typical output, at the tier's `prices` or, where it bills nothing, the
card's `list_prices`. A subscription's quota is a share of those same tokens,
so pricing it any other way would weigh the same model twice, by two methods.
A local model's list price is a shadow price for the machine and the wait, so
"free" does not win every task it could scrape through.

A subscription is still watched, for availability only: one that has answered
429 is off the table until its window turns.

A tier's family scale (`family_scales`) and level caps (`level_caps`) are kept
under its family key: the profile glob for a discovered card, or the tier name
for a hand-written one. `state()` reports the tiers no need vector can ever
send work to: those another tier beats on every requirement at no more cost
(see `dominance.py`).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable

import httpx

from .config import CapabilitiesConfig, Config, Prices, TierConfig
from .routing import RouteDecision, _explicit_preference
from .schemas import ChatCompletionRequest, Usage
from .tokens import estimate_prompt_tokens

# (packet, questions) -> requirement key -> P(yes). Injected by tests.
Ask = Callable[[str, dict[str, str]], Awaitable[dict[str, float]]]

_FILE_RE = re.compile(r"(?<![\w/.-])(?:[\w.-]+/)*[\w-]+\.(?:py|rs|ts|tsx|js|jsx|go|java|kt|rb|php|cs|cpp|cc|c|h|hpp|md|yaml|yml|toml|json|sql|sh|ps1)\b")
_FENCE_RE = re.compile(r"```([\w+-]*)")


# ---------------------------------------------------------------- the packet
def build_packet(request: ChatCompletionRequest, *, max_chars: int) -> str:
    """What Jev reads: the task, trimmed, and the request's shape, measured.

    A client that knows more than the transcript shows -- the NucleOS planner
    knows a task's size, risk and files -- sends it as a top-level `packet`
    object, and its fields lead. Nothing here asks a model to summarise: the
    packet is built by rules, so it costs nothing and says the same thing twice.
    """
    system: list[str] = []
    turns: list[tuple[str, str]] = []
    for message in request.messages:
        text = _text(message.content)
        if not text:
            continue
        if message.role == "system":
            system.append(text)
        else:
            turns.append((message.role, text))
    goal = next((text for role, text in reversed(turns) if role == "user"), "")
    everything = "\n".join(system + [text for _, text in turns])

    lines = ["TASK PACKET"]
    supplied = _supplied(request)
    for key, value in supplied.items():
        if key in ("failed_tiers",):
            continue  # routing instructions, not a description of the task
        lines.append(f"{key}: {_flat(value)}")
    files = sorted(set(_FILE_RE.findall(everything)))
    langs = sorted({lang for lang in _FENCE_RE.findall(everything) if lang})
    lines += [
        f"prompt_tokens: ~{estimate_prompt_tokens(request)}",
        f"conversation_turns: {len(turns)}",
        f"tools_offered: {', '.join(_tool_names(request)) or 'none'}",
        f"files_mentioned: {len(files)}" + (f" ({', '.join(files[:8])})" if files else ""),
        f"code_blocks: {everything.count('```') // 2}" + (f" ({', '.join(langs)})" if langs else ""),
    ]
    head = "\n".join(lines)
    budget = max(0, max_chars - len(head) - 40)
    instructions = system[0] if system else ""
    part = budget // 5 if instructions else 0
    body = []
    if instructions:
        body.append("instructions:\n" + _trim(instructions, part))
    body.append("task:\n" + _trim(goal, budget - part))
    return head + "\n\n" + "\n\n".join(body)


def _supplied(request: ChatCompletionRequest) -> dict[str, Any]:
    extra = request.model_extra or {}
    packet = extra.get("packet")
    if isinstance(packet, dict):
        return {str(k): v for k, v in packet.items()}
    if isinstance(packet, str) and packet.strip():
        return {"summary": packet.strip()}
    return {}


def failed_tiers(request: ChatCompletionRequest) -> set[str]:
    """Tiers the client says already failed this task; they are not offered again."""
    value = _supplied(request).get("failed_tiers")
    return {str(v) for v in value} if isinstance(value, list) else set()


def _tool_names(request: ChatCompletionRequest) -> list[str]:
    names = []
    for tool in request.tools or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else (tool.get("name") if isinstance(tool, dict) else None)
        if name:
            names.append(str(name))
    return names


def _trim(text: str, limit: int) -> str:
    """Head and tail: a task states its goal first and its constraints last."""
    if len(text) <= limit:
        return text
    keep = max(limit - 20, 0)
    return text[: keep * 2 // 3] + "\n[...]\n" + text[len(text) - keep // 3 :]


def _flat(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return "" if content is None else str(content)


# ---------------------------------------------------------------- asking Jev
def jev_asker(tier: TierConfig, client: httpx.AsyncClient | None = None) -> Ask:
    """One POST to /systemone, every requirement as a noul question."""
    http = client or httpx.AsyncClient(timeout=tier.timeout_s)

    async def ask(packet: str, questions: dict[str, str]) -> dict[str, float]:
        headers = {"Content-Type": "application/json"}
        if tier.api_key:
            headers["Authorization"] = f"Bearer {tier.api_key}"
        body: dict[str, Any] = {
            "model": tier.model,
            "state": packet,
            "questions": {k: {"type": "noul", "instructions": v} for k, v in questions.items()},
        }
        body.update(tier.extra_body)
        response = await http.post(f"{tier.base_url}/systemone", json=body, headers=headers)
        response.raise_for_status()
        answers = (response.json() or {}).get("answers") or {}
        out: dict[str, float] = {}
        for key in questions:
            answer = answers.get(key)
            p = answer.get("noul") if isinstance(answer, dict) else None
            if isinstance(p, bool) or not isinstance(p, (int, float)):
                # A requirement with no answer is not a requirement with p=0:
                # that would route as if the task were easy.
                raise ValueError(f"jev gave no probability for {key!r}")
            out[key] = float(p)
        return out

    ask.client = http  # type: ignore[attr-defined]
    return ask


# ---------------------------------------------------------------- quota
class QuotaLedger:
    """Per subscription: 429 lockouts, and list-price spend over the window for display.

    The spend never moves a decision -- cost is the price per token, the same
    for a subscription as for the API. A 429 does: that subscription is not
    available until its window turns.
    """

    def __init__(self, config: CapabilitiesConfig, clock: Callable[[], float] = time.time) -> None:
        self._budgets = config.subscriptions
        self._clock = clock
        self._spent: dict[str, deque[tuple[float, float]]] = {}
        self._locked_until: dict[str, float] = {}

    def record(self, subscription: str, usd: float) -> None:
        self._spent.setdefault(subscription, deque()).append((self._clock(), usd))

    def lock(self, subscription: str) -> None:
        budget = self._budgets.get(subscription)
        hours = budget.window_hours if budget else 1.0
        self._locked_until[subscription] = self._clock() + hours * 3600

    def used(self, subscription: str) -> float:
        budget = self._budgets.get(subscription)
        entries = self._spent.get(subscription)
        if not entries:
            return 0.0
        horizon = self._clock() - (budget.window_hours if budget else 5.0) * 3600
        while entries and entries[0][0] < horizon:
            entries.popleft()
        return sum(usd for _, usd in entries)

    def locked(self, subscription: str) -> bool:
        return self._clock() < self._locked_until.get(subscription, 0.0)

    def state(self) -> dict[str, Any]:
        names = set(self._budgets) | set(self._spent) | set(self._locked_until)
        return {
            name: {
                "used_usd": round(self.used(name), 4),
                "locked": self.locked(name),
            }
            for name in sorted(names)
        }


def _usd(prices: Prices, prompt_tokens: int, output_tokens: int) -> float:
    return (prompt_tokens * prices.input + output_tokens * prices.output) / 1_000_000


# ---------------------------------------------------------------- the router
@dataclass(frozen=True)
class Option:
    tier: str
    success: float
    cost: float


class CapabilityRouter:
    """Jev reads the packet; the cheapest card that covers it at `target` wins."""

    name = "capabilities"

    def __init__(
        self,
        config: Config,
        *,
        ask: Ask | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        caps = config.router.capabilities
        assert caps is not None, "parse_config guarantees a capabilities block for this kind"
        self._config = config
        self._caps = caps
        self._ask = ask or jev_asker(config.tier(caps.jev_tier))
        self._default = config.router.default_tier or next(iter(config.tiers))
        self.ledger = QuotaLedger(caps, clock)
        # Everything a decision depends on besides the task, so two log rows
        # with the same fingerprint were decided by the same rules.
        spec = {
            "requirements": caps.requirements,
            "cards": {k: [v.levels, v.output_tokens] for k, v in sorted(caps.cards.items())},
            "miss": caps.miss,
            "floor": caps.floor,
            "miss_scale": caps.miss_scale,
            "target": caps.target,
            "rule": caps.rule,
            "failure": asdict(caps.failure),
            "family_scales": caps.family_scales,
        }
        self.fingerprint = "capabilities:" + hashlib.sha256(
            json.dumps(spec, sort_keys=True).encode()
        ).hexdigest()[:12]
        # Cards do not change after construction, so neither does which of them can never win.
        from .dominance import dominated  # local: dominance imports this module's types

        try:
            self.dominated = dominated(self)
            self.dominance_error: str | None = None
        except Exception as exc:  # noqa: BLE001 - a report must not stop the router
            self.dominated, self.dominance_error = {}, f"{type(exc).__name__}: {exc}"

    def success(
        self, needs: dict[str, float], tier: str, *, scale: float | None = None, levels: dict[str, float] | None = None
    ) -> float:
        """Estimated success on `tier`. `scale` and `levels` stand in for the configured ones, for one answer."""
        card = self._caps.cards[tier]
        floor = self._caps.floor
        if scale is None:
            scale = self.scale_for(tier)
        table = card.levels if levels is None else levels
        p = 1.0
        for key, need in needs.items():
            strength = max(0.0, need - floor) / (1.0 - floor)
            p *= 1.0 - strength * min(1.0, scale * self._miss(table.get(key, 0.0)))
        return p

    def family_key(self, tier: str) -> str:
        """The key a family scale or a level cap is kept under: the profile glob, or the tier for a hand-written card."""
        return self._caps.cards[tier].family or tier

    def scale_for(self, tier: str, default: float | None = None) -> float:
        """The tier's family scale if one is fitted, else `default` (the configured `miss_scale` when None)."""
        fallback = self._caps.miss_scale if default is None else default
        return self._caps.family_scales.get(self.family_key(tier), fallback)

    def prices(self, tier_name: str) -> Prices:
        """What a tier really bills wins; the card's list prices stand in where it bills nothing."""
        tier = self._config.tier(tier_name)
        return tier.prices if tier.prices.configured else self._caps.cards[tier_name].list_prices

    def _miss(self, level: float) -> float:
        """`miss` at a level between the table's points, linearly."""
        level = min(3.0, max(0.0, float(level)))
        low = min(int(level), 2)
        frac = level - low
        return self._caps.miss[low] * (1 - frac) + self._caps.miss[low + 1] * frac

    def cost(self, tier_name: str, prompt_tokens: int) -> float:
        tier = self._config.tier(tier_name)
        card = self._caps.cards[tier_name]
        if tier.subscription and self.ledger.locked(tier.subscription):
            return math.inf  # answered 429: not available until the window turns
        return _usd(self.prices(tier_name), prompt_tokens + card.input_overhead, card.output_tokens)

    async def decide(
        self, request: ChatCompletionRequest, candidates: list[str], *, also_failed: frozenset[str] = frozenset()
    ) -> RouteDecision:
        explicit = _explicit_preference(self._config, request)
        if explicit and explicit in candidates and not also_failed:
            return RouteDecision(tier=explicit, reason="requested")

        failed = failed_tiers(request) | set(also_failed)
        pool = [t for t in candidates if t in self._caps.cards and t not in failed]
        if not pool:
            return self._fallback(candidates, "no carded tier is eligible")

        packet = build_packet(request, max_chars=self._caps.max_packet_chars)
        started = time.perf_counter()
        try:
            needs = await self._ask(packet, self._caps.requirements)
        except Exception as exc:  # noqa: BLE001 - any failure to read the task routes by default
            # Jev down is a routing outage, not a request failure: the default
            # tier still answers, and the reason says why it was chosen.
            return self._fallback(
                candidates, f"jev error: {type(exc).__name__}: {exc}"[:300], packet=packet,
                jev_ms=_ms(started),
            )
        jev_ms = _ms(started)

        tokens = estimate_prompt_tokens(request)
        options = [Option(t, self.success(needs, t), self.cost(t, tokens)) for t in pool]
        live = [o for o in options if math.isfinite(o.cost)]
        if not live:
            return self._fallback(
                candidates, "every carded tier's subscription is locked (429)", needs, packet=packet, jev_ms=jev_ms
            )
        # A retry after a failure never goes DOWN: a tier rated below one that
        # already failed this task is a worse guess. An equal one from another
        # family is fair -- two models do not fail the same way.
        failed_bar = max((self.success(needs, t) for t in failed if t in self._caps.cards), default=None)
        if failed_bar is not None:
            stronger = [o for o in live if o.success >= failed_bar - 1e-9]
            if not stronger:
                return self._fallback(
                    candidates, "no tier is rated at or above the one(s) that failed", needs, packet=packet,
                    jev_ms=jev_ms, escalation=True,
                )
            live = stronger
        pick, rule, weighed = self._pick(live, needs, request)
        expected = weighed.pop("expected", {})
        # The cheaper tiers it passed over, and why: that is what a wrong card
        # looks like in the log, and what recalibration reads.
        cheaper = sorted((o for o in live if o.cost < pick.cost), key=lambda o: -o.cost)[:3]
        reason = {
            "rule": rule,
            "need": {k: round(v, 3) for k, v in needs.items()},
            "pick": [pick.tier, round(pick.success, 3), _money(pick.cost)],
            **({"expected": _money(expected[pick.tier])} if expected else {}),
            "passed_over": [[o.tier, round(o.success, 3), _money(o.cost)] for o in cheaper],
        }
        if failed:
            reason["skipped_failed"] = sorted(failed)
        return RouteDecision(
            tier=pick.tier,
            score=pick.success,
            model=self.fingerprint,
            reason=json.dumps(reason, separators=(",", ":")),
            detail={
                "rule": rule,
                "packet": packet,
                "needs": needs,
                "jev_ms": jev_ms,
                "target": self._caps.target,
                "floor": self._caps.floor,
                "prompt_tokens": tokens,
                "pick": pick.tier,
                "skipped_failed": sorted(failed),
                "failed_bar": failed_bar,
                **weighed,
                "options": [self._option_view(o, expected.get(o.tier)) for o in sorted(options, key=lambda o: o.cost)],
            },
        )

    async def escalation(self, request: ChatCompletionRequest, served: str, candidates: list[str]) -> str | None:
        """The tier to re-answer a request `served` failed, or None if none is rated as strong."""
        decision = await self.decide(request, candidates, also_failed=frozenset({served}))
        if (decision.detail or {}).get("escalation") is False or decision.tier == served:
            return None
        return decision.tier

    def _pick(
        self, live: list[Option], needs: dict[str, float], request: ChatCompletionRequest
    ) -> tuple[Option, str, dict[str, Any]]:
        caps = self._caps
        if caps.rule == "expected_cost":
            # What failing costs: redoing the task on the safe choice -- the
            # cheapest option that reaches `target`, as the target rule would
            # have picked -- times how much worse it is when nobody catches
            # it, times stakes. Not the most likely option: that is the dearest
            # model there is, and nobody redoes a docstring on it.
            covered = [o for o in live if o.success >= caps.target]
            redo = (min(covered, key=lambda o: (o.cost, -o.success)) if covered
                    else max(live, key=lambda o: (o.success, -o.cost)))
            stakes, stakes_from = self._stakes(needs, request)
            verifiable = _supplied(request).get("verifiable") is True
            expected: dict[str, float] = {}
            for o in live:
                caught = 1.0 if verifiable else self._verified_share(o.tier)
                factor = caught * caps.failure.detected + (1 - caught) * caps.failure.undetected
                expected[o.tier] = o.cost + (1 - o.success) * redo.cost * factor * stakes
            pick = min(live, key=lambda o: (expected[o.tier], -o.success))
            return pick, "least expected cost", {
                "expected": expected, "stakes": round(stakes, 3), "stakes_from": stakes_from,
                "verifiable": verifiable, "redo_tier": redo.tier,
            }
        covered = [o for o in live if o.success >= caps.target]
        if covered:
            return min(covered, key=lambda o: (o.cost, -o.success)), f">= {caps.target:.2f}, cheapest", {}
        return max(live, key=lambda o: (o.success, -o.cost)), f"none >= {caps.target:.2f}, most likely", {}

    def _stakes(self, needs: dict[str, float], request: ChatCompletionRequest) -> tuple[float, str]:
        table = self._caps.failure.stakes
        said = _supplied(request).get("stakes")
        if isinstance(said, str) and said in table:
            return table[said], "packet"
        if "precision" in needs:
            # Jev's reading of "would a small mistake be costly": 0 -> low,
            # 0.5 -> normal, 1 -> high, linearly between.
            p = min(1.0, max(0.0, needs["precision"]))
            if p <= 0.5:
                return table["low"] + (table["normal"] - table["low"]) * p / 0.5, "jev precision"
            return table["normal"] + (table["high"] - table["normal"]) * (p - 0.5) / 0.5, "jev precision"
        return table["normal"], "default"

    def _verified_share(self, tier: str) -> float:
        """How often a failure on `tier` is caught by the router's own verifier."""
        settings = self._config.verification
        return settings.sample_rate if settings.verifies(tier) else 0.0

    def _option_view(self, option: Option, expected: float | None = None) -> dict[str, Any]:
        tier = self._config.tier(option.tier)
        return {
            "tier": option.tier,
            "model": tier.model,
            "effort": tier.effort,
            "subscription": tier.subscription,
            "success": round(option.success, 4),
            "cost": option.cost if math.isfinite(option.cost) else None,
            "covered": option.success >= self._caps.target,
            "expected": expected if expected is None or math.isfinite(expected) else None,
        }

    def _fallback(
        self,
        candidates: list[str],
        why: str,
        needs: dict[str, float] | None = None,
        *,
        packet: str | None = None,
        jev_ms: int | None = None,
        escalation: bool | None = None,
    ) -> RouteDecision:
        tier = self._default if self._default in candidates else candidates[0]
        reason: dict[str, Any] = {"rule": "fallback", "why": why}
        if needs:
            reason["need"] = {k: round(v, 3) for k, v in needs.items()}
        return RouteDecision(
            tier=tier,
            model=self.fingerprint,
            reason=json.dumps(reason, separators=(",", ":")),
            detail={"rule": "fallback", "why": why, "packet": packet, "needs": needs or {}, "jev_ms": jev_ms,
                    "target": self._caps.target, "floor": self._caps.floor, "pick": tier, "options": [],
                    **({"escalation": False} if escalation else {})},
        )

    def observe(self, tier_name: str, usage: Usage, status: int) -> None:
        """Charge a finished call to its subscription, or lock it on a 429."""
        tier = self._config.tiers.get(tier_name)
        if tier is None or not tier.subscription:
            return
        if status == 429:
            self.ledger.lock(tier.subscription)
            return
        card = self._caps.cards.get(tier_name)
        if card is not None and status < 400:
            self.ledger.record(
                tier.subscription, _usd(card.list_prices, usage.prompt_tokens, usage.completion_tokens)
            )

    def state(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "target": self._caps.target,
            "rule": self._caps.rule,
            "miss_scale": self._caps.miss_scale,
            "family_scales": self._caps.family_scales,
            "cards": sorted(self._caps.cards),
            "dominated": self.dominated,
            "subscriptions": self.ledger.state(),
        }

    async def aclose(self) -> None:
        client = getattr(self._ask, "client", None)
        if client is not None:
            await client.aclose()


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _money(usd: float) -> float | str:
    return "inf" if not math.isfinite(usd) else round(usd, 6)


__all__ = ["CapabilityRouter", "QuotaLedger", "build_packet", "failed_tiers", "jev_asker"]
