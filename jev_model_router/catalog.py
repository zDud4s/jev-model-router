"""The model catalog: which configured tiers can actually be served right now.

A tier names a model and an effort, and both go stale without anyone editing the
config: a vendor withdraws a model from a plan, a CLI too old for a new model
rejects it, a local model gets deleted. Found on 2026-09-24, the day this was
written: the Codex catalog listed gpt-6-luna/sol/astra at 15:08 UTC and none of
them at 17:43 -- seven configured tiers that would each have failed on their
first request, and a router that would have kept picking them.

So at startup every tier is checked against what its provider says it serves,
from sources that cost no quota:

* **Codex** -- `codex debug models`, which refreshes the account's catalog and
  prints it (~1 s); `~/.codex/models_cache.json` if that fails.
* **Claude Code** -- the catalog the CLI caches in
  `~/.claude/cache/model-catalog/`, plus `claude --version`, because a model can
  require a newer CLI than the one a tier points at.
* **Ollama** -- `GET /api/tags` on the tier's server.
* **An OpenAI-compatible API** named by a `discover` source -- `GET /models`.
  Where the listing says more than ids (context length, per-token prices, tool
  support, reasoning efforts, which id is a variant of which), that is read too:
  a discovered tier is built from what its provider states, not from a default.

A tier whose model or effort is not served is UNAVAILABLE: removed from
eligibility, with the reason in every request's rejections. A tier whose source
could not be read is UNVERIFIED and stays in -- a missing cache is not evidence
that a model is gone. Models a provider serves that no tier uses are listed as
unconfigured: routing to one needs a card, and a card is a judgement this module
does not make.

The report is written to `catalog.path` when it changed, so the file's history is
the history of the catalog, and served on /healthz.
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import Config, TierConfig

OK, UNAVAILABLE, UNVERIFIED, NOT_CHECKED = "ok", "unavailable", "unverified", "not_checked"

# argv, timeout -> stdout. Injected by tests.
Run = Callable[[list[str], float], str]


@dataclass(frozen=True)
class Offered:
    """One model a provider serves, and what it accepts."""

    id: str
    efforts: tuple[str, ...] | None = None  # None: the source does not say
    min_cli: str | None = None
    aliases: tuple[str, ...] = ()
    # Served, but not offered to people (Codex's `visibility: hide`): a utility
    # model, never reported as one a tier could be added for.
    hidden: bool = False
    # The id this one is a variant or alias of, when the listing says so and
    # lists it too: the same model, so the same benchmark evidence.
    same_as: str | None = None
    # What the listing states per model. None: it does not say, and the
    # discover source's own value holds.
    context_window: int | None = None
    supports_tools: bool | None = None
    prices: dict[str, float] | None = None  # USD per 1M tokens: input, output, cache_read, cache_write
    # Priced per call by something the listing cannot state up front (a router
    # model, a negative sentinel): a tier could never be costed before the call.
    unpriceable: bool = False


@dataclass
class Source:
    name: str
    models: dict[str, Offered] = field(default_factory=dict)
    origin: str = ""
    fetched_at: str | None = None
    error: str | None = None
    stale: bool = False
    cli_version: str | None = None


@dataclass
class TierStatus:
    status: str
    reason: str = ""


@dataclass
class CatalogReport:
    checked_at: str
    tiers: dict[str, TierStatus]
    sources: dict[str, Source]
    unconfigured: dict[str, list[str]]
    # discover source name -> what that provider serves; tiers are built from it.
    discovered: dict[str, Source] = field(default_factory=dict)

    @property
    def unavailable(self) -> dict[str, str]:
        return {name: s.reason for name, s in self.tiers.items() if s.status == UNAVAILABLE}

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_at": self.checked_at,
            "tiers": {name: {"status": s.status, "reason": s.reason} for name, s in sorted(self.tiers.items())},
            "sources": {
                name: {
                    "origin": src.origin,
                    "fetched_at": src.fetched_at,
                    "stale": src.stale,
                    "cli_version": src.cli_version,
                    "error": src.error,
                    "models": sorted(src.models),
                }
                for name, src in sorted(self.sources.items())
            },
            "unconfigured": {k: sorted(v) for k, v in sorted(self.unconfigured.items())},
            "discovered": {
                name: sorted(
                    f"{m.id}@{e}" if e else m.id for m in src.models.values() for e in (m.efforts or (None,))
                )
                for name, src in sorted(self.discovered.items())
            },
        }

    def summary(self) -> str:
        lines = []
        bad = self.unavailable
        counts = {st: sum(1 for s in self.tiers.values() if s.status == st) for st in (OK, UNAVAILABLE, UNVERIFIED)}
        lines.append(
            f"catalog: {counts[OK]} tier(s) ok, {counts[UNAVAILABLE]} unavailable, {counts[UNVERIFIED]} unverified"
        )
        for name, reason in sorted(bad.items()):
            lines.append(f"  unavailable {name}: {reason}")
        for name, s in sorted(self.tiers.items()):
            if s.status == UNVERIFIED:
                lines.append(f"  unverified {name}: {s.reason}")
        for provider, models in sorted(self.unconfigured.items()):
            if models:
                lines.append(f"  {provider} serves models no tier uses: {', '.join(sorted(models))}")
        for src in self.sources.values():
            if src.stale:
                lines.append(f"  {src.name} catalog is stale (fetched {src.fetched_at})")
        return "\n".join(lines)


# ---------------------------------------------------------------- sources
def _run(argv: list[str], timeout: float) -> str:
    proc = subprocess.run(
        argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}: {(proc.stderr or proc.stdout)[-300:]}")
    return proc.stdout


def _version(text: str) -> str | None:
    match = re.search(r"\d+\.\d+\.\d+", text or "")
    return match.group(0) if match else None


def _older(have: str, need: str) -> bool:
    as_tuple = lambda v: tuple(int(x) for x in v.split("."))  # noqa: E731
    return as_tuple(have) < as_tuple(need)


def codex_source(executable: str, codex_home: str | None, run: Run = _run) -> Source:
    source = Source("codex")
    payload: Any = None
    try:
        payload = json.loads(run([executable, "debug", "models"], 30.0))
        source.origin = f"{executable} debug models"
        source.fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    except Exception as exc:  # noqa: BLE001 - fall back to the cache the CLI keeps
        cache = Path(codex_home or os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "models_cache.json"
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            source.origin = str(cache)
            source.fetched_at = payload.get("fetched_at")
            source.stale = True  # the live refresh failed, so this is at best the last one
            source.error = f"refresh failed ({type(exc).__name__}: {exc}); read the cache"[:300]
        except Exception as cache_exc:  # noqa: BLE001
            source.error = f"refresh failed ({exc}); no cache ({cache_exc})"[:300]
            return source
    for model in (payload or {}).get("models") or []:
        slug = model.get("slug")
        if not slug:
            continue
        levels = model.get("supported_reasoning_levels")
        efforts = None
        if isinstance(levels, list):
            efforts = tuple(
                (level.get("effort") if isinstance(level, dict) else str(level)) for level in levels
            )
        source.models[slug] = Offered(id=slug, efforts=efforts, hidden=model.get("visibility") == "hide")
    return source


def claude_source(executable: str, claude_dir: str | None, run: Run = _run, now: float | None = None) -> Source:
    source = Source("claude")
    root = Path(claude_dir or os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
    files = sorted(glob.glob(str(root / "cache" / "model-catalog" / "*.json")), key=os.path.getmtime)
    try:
        source.cli_version = _version(run([executable, "--version"], 30.0))
    except Exception as exc:  # noqa: BLE001
        source.error = f"cannot read the CLI version ({type(exc).__name__}: {exc})"[:300]
    if not files:
        source.error = (source.error + "; " if source.error else "") + f"no model catalog under {root}"
        return source
    path = files[-1]
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        source.error = f"unreadable catalog {path}: {exc}"[:300]
        return source
    source.origin = path
    fetched, stale_at = payload.get("fetchedAt"), payload.get("staleAt")
    if isinstance(fetched, (int, float)):
        source.fetched_at = datetime.fromtimestamp(fetched / 1000, timezone.utc).isoformat(timespec="seconds")
    if isinstance(stale_at, (int, float)):
        # Refreshed by Claude Code itself whenever it runs; stale means nobody
        # has used it for a while, not that the models changed.
        source.stale = (now if now is not None else time.time()) * 1000 > stale_at
    catalog = payload.get("catalog") or {}
    for model in (catalog.get("config") or {}).get("models") or []:
        model_id = model.get("id")
        if not model_id:
            continue
        thinking = model.get("thinking") or {}
        efforts: tuple[str, ...] = tuple(
            opt.get("id") for opt in thinking.get("effort_options") or [] if isinstance(opt, dict)
        )
        aliases = []
        undated = re.sub(r"-\d{8}$", "", model_id)
        if undated != model_id:
            aliases.append(undated)
        if model.get("section") == "main" and model.get("short_name"):
            aliases.append(str(model["short_name"]).lower())
        source.models[model_id] = Offered(
            id=model_id, efforts=efforts, min_cli=model.get("min_claude_code_version"), aliases=tuple(aliases)
        )
    return source


def ollama_source(base_url: str, get: Callable[[str], Any] | None = None) -> Source:
    source = Source(f"ollama {base_url}", origin=f"{base_url}/api/tags")
    try:
        payload = (get or _get_json)(f"{base_url}/api/tags")
    except Exception as exc:  # noqa: BLE001
        source.error = f"unreachable ({type(exc).__name__}: {exc})"[:300]
        return source
    source.fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for model in payload.get("models") or []:
        name = model.get("name") or model.get("model")
        if name:
            aliases = (name[: -len(":latest")],) if name.endswith(":latest") else ()
            source.models[name] = Offered(id=name, aliases=aliases)
    return source


def _get_json(url: str, headers: dict[str, str] | None = None, timeout: float = 5.0) -> Any:
    response = httpx.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    return response.json()


# Listing price fields (USD per token, as strings) -> Prices fields (per 1M).
_LISTED_PRICES = {"prompt": "input", "completion": "output", "input_cache_read": "cache_read",
                  "input_cache_write": "cache_write"}


def models_source(base_url: str, get: Callable[[str], Any] | None = None, api_key: str | None = None) -> Source:
    """`GET {base_url}/models`, the OpenAI-compatible listing, read as far as it goes.

    Every such API returns `data[].id`. Richer listings also state each model's
    context length, per-token prices, supported parameters, output modalities
    and reasoning efforts, and which id is an alias or variant of which; each is
    read when present and left to the discover source's defaults when not.
    """
    source = Source(f"models {base_url}", origin=f"{base_url}/models")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        payload = get(f"{base_url}/models") if get else _get_json(f"{base_url}/models", headers, timeout=30.0)
    except Exception as exc:  # noqa: BLE001
        source.error = f"unreachable ({type(exc).__name__}: {exc})"[:300]
        return source
    source.fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entries = [m for m in (payload.get("data") if isinstance(payload, dict) else None) or [] if isinstance(m, dict)]
    listed = {str(m["id"]) for m in entries if m.get("id")}
    for model in entries:
        model_id = str(model.get("id") or "")
        if not model_id:
            continue
        outputs = (model.get("architecture") or {}).get("output_modalities")
        if isinstance(outputs, list) and "text" not in outputs:
            continue  # it cannot write an answer
        params = model.get("supported_parameters")
        reasoning = model.get("reasoning") or {}
        efforts = reasoning.get("supported_efforts") if isinstance(reasoning, dict) else None
        prices, unpriceable = _listed_prices(model.get("pricing"))
        context = model.get("context_length") or (model.get("top_provider") or {}).get("context_length")
        aliases = tuple(a for a in (model.get("canonical_slug"),) if a and a != model_id)
        source.models[model_id] = Offered(
            id=model_id,
            efforts=tuple(str(e) for e in efforts) if isinstance(efforts, list) and efforts else None,
            aliases=aliases,
            same_as=_same_as(model_id, model.get("alias_target"), listed),
            context_window=int(context) if isinstance(context, (int, float)) and context > 0 else None,
            supports_tools=("tools" in params) if isinstance(params, list) else None,
            prices=prices,
            unpriceable=unpriceable,
        )
    return source


def _listed_prices(raw: Any) -> tuple[dict[str, float] | None, bool]:
    if not isinstance(raw, dict):
        return None, False
    out: dict[str, float] = {}
    for listed, field_name in _LISTED_PRICES.items():
        try:
            value = float(raw[listed])
        except (KeyError, TypeError, ValueError):
            continue
        if value < 0:
            return None, True
        out[field_name] = round(value * 1_000_000, 6)
    return (out or None), False


def _same_as(model_id: str, alias_target: Any, listed: set[str]) -> str | None:
    """The listed id this one is another name for: its alias target, or the id before a `:variant` suffix."""
    if isinstance(alias_target, str) and alias_target in listed and alias_target != model_id:
        return alias_target
    base = model_id.split(":", 1)[0]
    return base if base != model_id and base in listed else None


# ---------------------------------------------------------------- the check
def _find(source: Source, model: str) -> Offered | None:
    if model in source.models:
        return source.models[model]
    return next((m for m in source.models.values() if model in m.aliases), None)


def _judge(tier: TierConfig, source: Source) -> TierStatus:
    if not source.models:
        # Nothing read at all: an unreachable Ollama really cannot answer; a
        # missing CLI cache says nothing about the models.
        if tier.backend == "ollama":
            return TierStatus(UNAVAILABLE, f"ollama: {source.error}")
        return TierStatus(UNVERIFIED, source.error or f"the catalog of {source.name} is empty")
    offered = _find(source, tier.model)
    if offered is None:
        return TierStatus(UNAVAILABLE, f"{tier.model!r} is not in the catalog of {source.name}")
    if tier.effort is not None and offered.efforts is not None and tier.effort not in offered.efforts:
        takes = ", ".join(offered.efforts) or "none"
        return TierStatus(UNAVAILABLE, f"{tier.model!r} does not take effort {tier.effort!r} (takes: {takes})")
    if offered.min_cli and source.cli_version and _older(source.cli_version, offered.min_cli):
        return TierStatus(
            UNAVAILABLE, f"{tier.model!r} needs Claude Code >= {offered.min_cli}; {tier.base_url} is {source.cli_version}"
        )
    return TierStatus(OK)


def check_catalog(
    config: Config,
    *,
    run: Run = _run,
    get: Callable[[str], Any] | None = None,
    now: float | None = None,
) -> CatalogReport:
    settings = config.catalog
    sources: dict[str, Source] = {}
    tiers: dict[str, TierStatus] = {}
    source_of: dict[str, str] = {}

    def read(backend: str, base_url: str, api_key: str | None = None, listing: bool = False) -> str | None:
        key = f"{backend} {base_url}"
        if key not in sources:
            if backend == "codex_cli":
                sources[key] = codex_source(base_url, settings.codex_home, run)
            elif backend == "claude_cli":
                sources[key] = claude_source(base_url, settings.claude_dir, run, now)
            elif backend == "ollama" and settings.check_ollama:
                sources[key] = ollama_source(base_url, get)
            elif backend == "openai_compatible" and listing:
                # Read only for a discover source: an explicit tier on an API
                # nobody asked to list stays NOT_CHECKED, as it always was.
                sources[key] = models_source(base_url, get, api_key)
            else:
                return None
        return key

    caps = config.router.capabilities
    discover_keys: dict[str, str] = {}
    for name, provider in (caps.discover if caps else {}).items():
        from . import keystore

        api_key = keystore.lookup(provider.api_key_env) if provider.api_key_env else None
        key = read(provider.backend, provider.base_url, api_key, listing=True)
        if key is not None:
            discover_keys[name] = key

    for name, tier in config.tiers.items():
        key = read(tier.backend, tier.base_url)
        if key is None:
            tiers[name] = TierStatus(NOT_CHECKED, f"no catalog source for backend {tier.backend!r}")
            continue
        source_of[name] = key

    families = [key.split(" ", 1)[0] for key in sources]
    for key, source in sources.items():
        backend, where = key.split(" ", 1)
        if backend != "ollama" and families.count(backend) > 1:
            source.name = f"{source.name} ({where})"
    for name, key in source_of.items():
        tiers[name] = _judge(config.tiers[name], sources[key])

    unconfigured: dict[str, list[str]] = {}
    for key, source in sources.items():
        if key in discover_keys.values():
            continue  # every model it serves becomes a tier; none is left out
        used = {
            offered.id
            for name, tier in config.tiers.items()
            if source_of.get(name) == key and (offered := _find(source, tier.model)) is not None
        }
        unconfigured[source.name] = [m for m, o in source.models.items() if m not in used and not o.hidden]
    return CatalogReport(
        checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        tiers=tiers,
        sources={source.name: source for source in sources.values()},
        unconfigured=unconfigured,
        discovered={name: sources[key] for name, key in discover_keys.items()},
    )


def write_if_changed(report: CatalogReport, path: str | Path) -> bool:
    """Write the report when anything but the check time differs. Returns whether it wrote."""
    target = Path(path)
    fresh = report.to_dict()
    try:
        old = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        old = None
    strip = lambda d: {**d, "checked_at": None, "sources": {  # noqa: E731
        k: {**v, "fetched_at": None} for k, v in (d.get("sources") or {}).items()
    }}
    if old is not None and strip(old) == strip(fresh):
        return False
    target.write_text(json.dumps(fresh, indent=2) + "\n", encoding="utf-8")
    return True


def startup_check(config: Config) -> CatalogReport:
    """Run the check, persist it, and say what it found on stderr."""
    report = check_catalog(config)
    if config.catalog.path:
        try:
            if write_if_changed(report, config.catalog.path):
                print(f"catalog updated: {config.catalog.path}", file=sys.stderr)
        except OSError as exc:
            print(f"catalog: cannot write {config.catalog.path}: {exc}", file=sys.stderr)
    print(report.summary(), file=sys.stderr)
    return report


__all__ = ["CatalogReport", "check_catalog", "startup_check", "write_if_changed"]
