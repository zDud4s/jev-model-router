"""The HTTP surface: an OpenAI-compatible proxy.

Request lifecycle, in the order it must happen:

    parse -> eligibility gate -> route -> call backend -> cost -> log

The gate runs before the router, never after. See `eligibility.py` for why that
ordering is the point rather than an implementation detail.

Logging is the last step and is allowed to fail. Nothing in this module lets a
log failure change what the client receives.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable
from urllib.parse import urlsplit

import anyio
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import ValidationError

from .backends import (
    Backend,
    BackendError,
    BackendFactory,
    StreamChunk,
    StreamEnd,
    build_backends,
)
from .config import Config, ConfigError
from .db import LogEntry, RequestLog
from .eligibility import evaluate
from .pricing import cost_usd, counterfactuals
from .routing import Router, build_router
from .schemas import ChatCompletionRequest, ModelList, Usage, error_body, model_card
from .route_api import OUTCOMES, parse_route_ask, runner_of
from .trace import PREVIEW_CHARS, TraceStore, prompt_preview
from .verification import (
    SkipReason,
    Verdict,
    VerificationOutcome,
    Verifier,
    build_verifier,
)

# Sent verbatim by the client's SDK when it asked for usage on a stream.
_DONE_FRAME = "data: [DONE]\n\n"


def _prompt_text(request: ChatCompletionRequest) -> str:
    """The canonical text whose SHA-256 identifies this prompt.

    Serialized deterministically (sorted keys) so the same conversation hashes
    the same way across runs -- a hash that changed with dict ordering would be
    useless for spotting repeats, which is most of what the hash is for.
    """
    return json.dumps(
        [m.model_dump(exclude_none=True) for m in request.messages],
        sort_keys=True,
        ensure_ascii=False,
    )


def _sse(chunk: dict[str, Any]) -> str:
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _takes_task(router: Any) -> bool:
    """Whether this router's `decide` can be told the request is a whole task (injected test routers may not)."""
    try:
        return "task" in inspect.signature(router.decide).parameters
    except (TypeError, ValueError):
        return False


def _benchmark_scores(config: Config, fetch: Callable[[str, dict[str, str], float], bytes] | None) -> Any:
    """The benchmark evidence for discovered cards, refreshed when stale. None: cards are the profiles alone."""
    caps = config.router.capabilities
    if caps is None or caps.benchmarks is None:
        return None
    from .scores import load_scores
    from .scores_import import refresh

    try:
        scores = load_scores(config)
        refreshed, lines = refresh(config, scores, fetch)
        for line in lines:
            print(line, file=sys.stderr)
        return load_scores(config) if refreshed else scores
    except Exception as exc:  # noqa: BLE001 - the profiles alone still route
        print(f"benchmarks failed to load, cards are the profiles alone: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return None


def create_app(
    config: Config,
    *,
    backend_factory: BackendFactory | None = None,
    log: RequestLog | None = None,
    router: Router | None = None,
    verifier: Verifier | None = None,
    catalog_check: Callable[[Config], Any] | None = None,
    benchmark_fetch: Callable[[str, dict[str, str], float], bytes] | None = None,
    config_path: str | Path | None = None,
) -> FastAPI:
    """Build the app. Every collaborator is injectable, which is how tests avoid the network.

    `config_path` is the file `config` was read from; with it the routing page
    can edit that file (the running app keeps the config it started with).
    """

    # The catalog check runs first: discovered tiers must exist before the
    # backends, router and verifier that serve them are built.
    # "configured" and "scores": what `expand` was given, kept for the tiers view.
    catalog: dict[str, Any] = {"report": None, "unavailable": {}, "discovered": 0, "unprofiled": [],
                               "configured": None, "scores": None}
    if config.catalog.check_on_start:
        from .catalog import startup_check
        from .discovery import expand

        try:
            report = (catalog_check or startup_check)(config)
            catalog["report"] = report
            catalog["unavailable"] = dict(report.unavailable)
            if getattr(report, "discovered", None):
                scores = _benchmark_scores(config, benchmark_fetch)
                configured = config
                try:
                    config, found, unprofiled = expand(config, report, scores)
                except Exception as exc:  # noqa: BLE001 - the evidence broke: the profiles alone still route
                    if scores is None:
                        raise
                    print(f"benchmarks: deriving cards failed, cards are the profiles alone: "
                          f"{type(exc).__name__}: {exc}", file=sys.stderr)
                    scores = None
                    config, found, unprofiled = expand(config, report, None)
                catalog["discovered"] = len(found)
                catalog["configured"], catalog["scores"] = configured, scores
                catalog["unprofiled"] = unprofiled
                print(f"catalog: {len(found)} tier(s) discovered", file=sys.stderr)
                if unprofiled:
                    from .scores_derive import some

                    print(f"  on the fallback profile (write a profile): {len(unprofiled)}: {some(unprofiled)}",
                          file=sys.stderr)
                from .discovery import stale_profiles

                for line in stale_profiles(config):
                    print(f"  {line}", file=sys.stderr)
                from .discovery import unused_caps

                unused = unused_caps(config)
                if unused:
                    print(f"  level_caps for no card (a model left the catalog?): {', '.join(unused)}", file=sys.stderr)
                if scores is not None:
                    from .discovery import served_ids
                    from .scores_derive import startup_lines

                    try:  # a report only: its failure must not change what is served
                        lines = startup_lines(scores, config.router.capabilities, served_ids(found, report, configured))
                    except Exception as exc:  # noqa: BLE001
                        lines = [f"benchmarks: startup report failed: {type(exc).__name__}: {exc}"]
                    for line in lines:
                        print(line, file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - a broken check must not stop the proxy
            print(f"catalog check failed, serving the configured tiers: {type(exc).__name__}: {exc}", file=sys.stderr)

    request_log = log or RequestLog(config.log.path, store_prompts=config.log.store_prompts)
    backends: dict[str, Backend] = build_backends(config, backend_factory)
    active_router: Router = router or build_router(config)
    if catalog["unavailable"] and hasattr(active_router, "note_unavailable"):
        active_router.note_unavailable(catalog["unavailable"])
    if getattr(active_router, "dominance_error", None):
        print(f"dominance check failed: {active_router.dominance_error}", file=sys.stderr)
    elif getattr(active_router, "dominated", None):
        from .dominance import summary

        for line in summary(active_router.dominated, len(config.router.capabilities.cards)):
            print(line, file=sys.stderr)
    # A route-only decision is a whole task; a router that can price one is told so.
    route_task = {"task": True} if _takes_task(active_router) else {}
    calls = getattr(active_router, "calls", None)
    if calls is not None:
        try:  # the log's outcomes seed each tier's shape; a log it cannot read leaves the config's
            rows = request_log.task_usage()
        except Exception as exc:  # noqa: BLE001
            rows = []
            print(f"task shapes: cannot read them from the log: {type(exc).__name__}: {exc}", file=sys.stderr)
        for row in rows:
            try:  # one bad row must not cost every other row its evidence
                calls.record_task(row["tier"], Usage(
                    prompt_tokens=row["input"] or 0, completion_tokens=row["output"] or 0,
                    cached_tokens=row["cached"] or 0, cache_write_tokens=row["written"] or 0,
                ))
            except Exception as exc:  # noqa: BLE001
                print(f"task shapes: cannot read a row for {row['tier']!r}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
        uncached = calls.uncached()
        if uncached:
            from .scores_derive import some

            print(f"task_shape: {len(uncached)} tier(s) have no cache price and are priced as if they cached "
                  f"nothing: {some(uncached)}", file=sys.stderr)
    # None when verification is off, so the request path has one branch rather
    # than a cascade of `if config.verification.enabled` checks.
    active_verifier: Verifier | None = verifier or build_verifier(config, backends, active_router)
    traces: TraceStore | None = TraceStore(config.trace.keep) if config.trace.enabled else None
    tiers: dict[str, Any] = {"error": None, "global": None, "tiers": []}
    if traces is not None:
        from . import tiers_view

        try:  # a report only: its failure must not change what is served
            tiers = tiers_view.build(config, active_router, report=catalog["report"],
                                     configured=catalog["configured"], scores=catalog["scores"])
        except Exception as exc:  # noqa: BLE001
            tiers["error"] = f"{type(exc).__name__}: {exc}"
            print(f"tiers view failed: {tiers['error']}", file=sys.stderr)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        for backend in backends.values():
            await backend.aclose()
        closer = getattr(active_router, "aclose", None)
        if closer is not None:
            await closer()
        request_log.close()

    app = FastAPI(title="jev-model-router", version="0.1.0", lifespan=lifespan)
    app.state.config = config
    app.state.log = request_log
    app.state.backends = backends
    app.state.router = active_router
    app.state.verifier = active_verifier

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        body: dict[str, Any] = {
            "status": "ok",
            "tiers": sorted(config.tiers),
            "router": active_router.name,
            "verification": active_verifier is not None,
            "schema_version": request_log.schema_version,
            # Surfaced here so a silently broken log is discoverable without
            # reading the database.
            "log_failures": request_log.failed_writes,
        }
        model = getattr(active_router, "model", None)
        if model is not None:
            # The deployed model's own held-out numbers, served next to its
            # fingerprint. A model whose evidence is only in the terminal where
            # it was trained gets deployed on memory of a good result.
            body["classifier"] = {
                "fingerprint": model.fingerprint,
                "predicts_tier": model.predicts_tier,
                "trained_at": model.trained_at,
                "examples": model.examples,
                "threshold": getattr(active_router, "threshold", model.threshold),
                "base_rate": model.base_rate,
                "metrics": model.metrics,
            }
        if catalog["report"] is not None:
            body["catalog"] = {
                "checked_at": catalog["report"].checked_at,
                "unavailable": catalog["unavailable"],
                "unconfigured": catalog["report"].unconfigured,
                "discovered_tiers": catalog["discovered"],
                "unprofiled": catalog["unprofiled"],
            }
        state = getattr(active_router, "state", None)
        if callable(state):
            # The capabilities router's rules fingerprint and each
            # subscription's window, so a router that has started refusing a
            # subscription says so here rather than in a pattern of tiers.
            body["routing"] = state()
        return body

    if traces is not None:

        @app.get("/routing", response_class=HTMLResponse)
        async def routing_page() -> str:
            from .routing_page import PAGE

            return PAGE

        @app.get("/routing/tiers")
        async def routing_tiers() -> dict[str, Any]:
            """Every carded tier's levels, their sources and evidence, and what dominates it; built at startup."""
            return tiers

        @app.get("/routing/traces")
        async def routing_traces(after: int = 0) -> dict[str, Any]:
            return {"seq": traces.seq, "traces": traces.since(after)}

        @app.post("/routing/dry-run")
        async def routing_dry_run(raw_request: Request) -> Any:
            """Route a request without answering it: eligibility and the router only.

            With the capabilities router that is one Jev call and nothing else --
            no destination model runs, no quota is spent.
            """
            try:
                payload = await raw_request.json()
                request = ChatCompletionRequest.model_validate({"model": "auto", **payload})
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                return JSONResponse(status_code=400, content=error_body(f"invalid request: {exc}"))
            request_id = f"dry_{uuid.uuid4().hex[:12]}"
            trace = traces.start(request_id, kind="dry-run", preview=prompt_preview(request.messages),
                                 requested_model=request.model)
            eligibility = evaluate(config, request, catalog["unavailable"])
            traces.stage(trace, "eligibility", eligible=list(eligibility.eligible),
                         rejected=[r.as_dict() for r in eligibility.rejections])
            if not eligibility.eligible:
                traces.finish(trace, "error")
                return {"trace": trace["id"], "tier": None}
            began = time.perf_counter()
            decision = active_router.decide(request, list(eligibility.eligible))
            if inspect.isawaitable(decision):
                decision = await decision
            traces.stage(trace, "route", **_route_view(decision, config, began))
            traces.finish(trace, "ok")
            return {"trace": trace["id"], "tier": decision.tier, "score": decision.score}

        from . import config_edit

        # What this process was started with, so the page can say a saved file
        # is not yet the one being served.
        started_digest = None
        if config_path is not None:
            try:
                started_digest = config_edit.digest(Path(config_path).read_text(encoding="utf-8"))
            except OSError:
                pass

        def config_guard(raw_request: Request) -> JSONResponse | None:
            """Only this page may write the config: JSON (so a cross-site form cannot), same origin, a known file."""
            if config_path is None:
                return JSONResponse(status_code=404, content=error_body("the proxy was not started from a config file"))
            origin = raw_request.headers.get("origin")
            if origin and urlsplit(origin).netloc != raw_request.headers.get("host"):
                return JSONResponse(status_code=403, content=error_body("config edits are accepted from this page only"))
            if not raw_request.headers.get("content-type", "").startswith("application/json"):
                return JSONResponse(status_code=415, content=error_body("send the edit as application/json"))
            return None

        async def edited(raw_request: Request) -> tuple[str, str, str | None]:
            """(the file now, the text the edit makes of it, the digest the page read)."""
            payload = await raw_request.json()
            if not isinstance(payload, dict):
                raise config_edit.EditError("send {text} or {changes}")
            current = Path(config_path).read_text(encoding="utf-8")
            if isinstance(payload.get("text"), str):
                text = payload["text"]
            elif isinstance(payload.get("changes"), list):
                text = config_edit.apply_changes(current, payload["changes"])
            else:
                raise config_edit.EditError("send {text} or {changes}")
            return current, text, payload.get("base")

        @app.get("/routing/config")
        async def routing_config() -> dict[str, Any]:
            """The config file as text and as data, whether it validates, and what the form may offer."""
            if config_path is None:
                return {"path": None, "options": config_edit.options()}
            try:
                text = Path(config_path).read_text(encoding="utf-8")
                data = yaml.safe_load(text) or {}
            except (OSError, yaml.YAMLError) as exc:
                return {"path": str(config_path), "error": f"{type(exc).__name__}: {exc}", "options": config_edit.options()}
            digest = config_edit.digest(text)
            return {"path": str(config_path), "text": text, "data": data, "digest": digest,
                    "served": digest == started_digest, "check": config_edit.check(text),
                    "options": config_edit.options()}

        @app.get("/routing/config/sources")
        async def routing_config_sources() -> Any:
            """Providers the file could discover models from and does not yet: the list `init` offers."""
            if config_path is None:
                return JSONResponse(status_code=404, content=error_body("the proxy was not started from a config file"))
            try:
                data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError) as exc:
                return JSONResponse(status_code=400, content=error_body(f"{type(exc).__name__}: {exc}"))
            # detect() looks for executables and asks Ollama for its tags: blocking, so off the loop.
            return {"sources": await anyio.to_thread.run_sync(config_edit.sources_on_offer, data)}

        @app.post("/routing/config/check")
        async def routing_config_check(raw_request: Request) -> Any:
            """What a save would write: the new text, its diff, and whether it validates. Writes nothing."""
            refused = config_guard(raw_request)
            if refused is not None:
                return refused
            try:
                current, text, _ = await edited(raw_request)
            except (config_edit.EditError, json.JSONDecodeError, OSError) as exc:
                return JSONResponse(status_code=400, content=error_body(str(exc)))
            return {**config_edit.check(text), "text": text, "diff": config_edit.diff(current, text, Path(config_path).name)}

        @app.put("/routing/config")
        async def routing_config_save(raw_request: Request) -> Any:
            """Validate and write the config atomically. It is served from the next start."""
            refused = config_guard(raw_request)
            if refused is not None:
                return refused
            try:
                _, text, base = await edited(raw_request)
                config_edit.write(config_path, text, base)
            except config_edit.EditError as exc:
                return JSONResponse(status_code=409 if "changed on disk" in str(exc) else 400, content=error_body(str(exc)))
            except (ConfigError, json.JSONDecodeError, OSError) as exc:
                return JSONResponse(status_code=400, content=error_body(str(exc)))
            digest = config_edit.digest(text)
            return {"saved": True, "path": str(config_path), "digest": digest, "served": digest == started_digest}

    @app.post("/v1/route")
    async def route_only(raw_request: Request) -> Any:
        """Which model and effort should run this task. Nothing is run; see route_api.py."""
        try:
            ask = parse_route_ask(await raw_request.json(), config)
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            return JSONResponse(status_code=400, content=error_body(f"invalid route request: {exc}"))
        request = ask.request
        decision_id = f"rt_{uuid.uuid4().hex[:16]}"
        trace = (
            traces.start(decision_id, kind="route", preview=prompt_preview(request.messages),
                         requested_model=ask.stage or "route")
            if traces is not None else None
        )
        eligibility = evaluate(config, request, catalog["unavailable"])
        eligible = [t for t in eligibility.eligible if ask.can_run(config.tiers[t])]
        if trace is not None:
            traces.stage(trace, "eligibility", eligible=eligible,
                         rejected=[r.as_dict() for r in eligibility.rejections])
        if not eligible:
            if trace is not None:
                traces.finish(trace, "error")
            runners = sorted(ask.runners) or "any runner"
            left_out = (f"; the request's models/exclude left out {eligibility.excluded} tier(s)"
                        if eligibility.excluded else "")
            return JSONResponse(status_code=422, content=error_body(
                f"no eligible tier runs on {runners}{left_out}"))
        began = time.perf_counter()
        decision = active_router.decide(request, eligible, **route_task)
        if inspect.isawaitable(decision):
            decision = await decision
        tier = config.tiers[decision.tier]
        detail = decision.detail or {}
        picked = next((o for o in detail.get("options", []) if o.get("tier") == decision.tier), {})
        if trace is not None:
            traces.stage(trace, "route", **_route_view(decision, config, began))
            traces.finish(trace, "ok")
        await asyncio.to_thread(
            request_log.record_decision, decision_id,
            task=_prompt_text(request), tier=decision.tier, model=tier.model, effort=tier.effort,
            runner=runner_of(tier), stage=ask.stage, route_score=decision.score,
            route_model=decision.model, route_reason=decision.reason,
        )
        return {
            "decision_id": decision_id,
            "tier": decision.tier,
            "runner": runner_of(tier),
            "model": tier.model,
            "effort": tier.effort,
            "success": decision.score,
            "estimated_cost_usd": picked.get("cost"),
            # Whether that figure prices a whole task by its shape or one call. Either way it
            # compares tiers; it is not a budget.
            "cost_basis": detail.get("cost_basis", "one_call"),
            "rule": detail.get("rule") or decision.reason,
            # Why a fallback fell back (Jev unreachable, no key...): the caller
            # cannot see the router's stderr.
            **({"why": detail["why"]} if detail.get("why") else {}),
            # Jev could not read a requirement: which, and whether the floor raised the pick.
            **({"unsure": detail["unsure"]} if detail.get("unsure") else {}),
            "router": decision.model,
            "unknown_failed": list(ask.unknown_failed),
        }

    @app.post("/v1/route/{decision_id}/outcome")
    async def route_outcome(decision_id: str, raw_request: Request) -> Any:
        """What the caller's gate made of a routed task: the label calibration learns from."""
        try:
            payload = await raw_request.json()
        except json.JSONDecodeError as exc:
            return JSONResponse(status_code=400, content=error_body(f"invalid outcome: {exc}"))
        status = payload.get("status") if isinstance(payload, dict) else None
        if status not in OUTCOMES:
            return JSONResponse(status_code=400, content=error_body(f"'status' must be one of {list(OUTCOMES)}"))
        raw_usage = payload.get("usage")
        try:
            usage = Usage.model_validate(raw_usage) if isinstance(raw_usage, dict) else None
        except ValidationError as exc:
            return JSONResponse(status_code=400, content=error_body(f"invalid usage: {exc}"))
        detail = payload.get("detail")
        result = await asyncio.to_thread(
            request_log.set_outcome, decision_id, status, str(detail)[:2000] if detail else None, usage
        )
        if result == "missing":
            return JSONResponse(status_code=404, content=error_body(f"no decision {decision_id!r}"))
        if result == "exists":
            return JSONResponse(status_code=409, content=error_body(f"decision {decision_id!r} already has an outcome"))
        row = request_log.decision(decision_id)
        observe = getattr(active_router, "observe", None)
        if observe is not None and row is not None:
            try:
                if status == "rate_limited":
                    observe(row["tier"], Usage(), 429)
                elif usage is not None:
                    observe(row["tier"], usage, 200)
            except Exception:  # noqa: BLE001 - accounting must not fail the report
                pass
        calls = getattr(active_router, "calls", None)
        # A failed task's usage still counts toward the shape: it cost what it
        # cost regardless of the caller's gate verdict.
        if calls is not None and usage is not None and row is not None:
            try:
                calls.record_task(row["tier"], usage)
            except Exception:  # noqa: BLE001 - a shape must not fail the report
                pass
        return {"decision_id": decision_id, "status": status, "tier": row["tier"] if row else None}

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        # Tiers are what a client may ask for, so tiers are what is listed.
        return ModelList(data=[model_card(name, tier.model) for name, tier in config.tiers.items()])

    @app.post("/v1/chat/completions")
    async def chat_completions(raw_request: Request) -> Any:
        started = time.perf_counter()
        request_id = f"req_{uuid.uuid4().hex[:16]}"

        try:
            payload = await raw_request.json()
            request = ChatCompletionRequest.model_validate(payload)
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            # Malformed input never reached a backend and has no prompt to hash
            # beyond what arrived, so it is logged with an empty prompt.
            await request_log.record_async(
                LogEntry(
                    request_id=request_id,
                    prompt_text="",
                    http_status=400,
                    error=str(exc),
                    latency_ms=_elapsed_ms(started),
                )
            )
            return JSONResponse(status_code=400, content=error_body(f"invalid request: {exc}"))

        prompt = _prompt_text(request)
        trace = (
            traces.start(request_id, kind="request", preview=prompt_preview(request.messages),
                         requested_model=request.model)
            if traces is not None
            else None
        )

        def stage(name: str, **data: Any) -> None:
            if trace is not None:
                traces.stage(trace, name, **data)

        # --- eligibility gate, before any routing decision --------------------
        eligibility = evaluate(config, request, catalog["unavailable"])
        base_entry = LogEntry(
            request_id=request_id,
            prompt_text=prompt,
            requested_model=request.model,
            router=active_router.name,
            stream=request.stream,
            rejections=eligibility.rejections,
        )

        stage("eligibility", eligible=list(eligibility.eligible),
              rejected=[r.as_dict() for r in eligibility.rejections])
        if not eligibility.eligible:
            if trace is not None:
                traces.finish(trace, "error")
            reasons = "; ".join(f"{r.tier}: {r.detail}" for r in eligibility.rejections)
            if eligibility.excluded:
                reasons = "; ".join(filter(None, [
                    reasons, f"{eligibility.excluded} tier(s) left out by the request's models/exclude"]))
            base_entry.http_status = 400
            base_entry.error = f"no eligible tier ({reasons})"
            base_entry.latency_ms = _elapsed_ms(started)
            await request_log.record_async(base_entry)
            return JSONResponse(
                status_code=400,
                content=error_body(
                    f"no configured tier can serve this request: {reasons}",
                    code="no_eligible_tier",
                ),
            )

        # --- route ------------------------------------------------------------
        route_began = time.perf_counter()
        decision = active_router.decide(request, list(eligibility.eligible))
        if inspect.isawaitable(decision):
            decision = await decision
        stage("route", **_route_view(decision, config, route_began))
        tier_name = decision.tier
        tier = config.tier(tier_name)
        backend = backends[tier_name]
        base_entry.tier = tier_name
        base_entry.backend = tier.backend
        base_entry.model = tier.model
        # The score goes in the row whether or not it changed anything. Next to
        # the verdict the verifier will write, it is the only way to ask later
        # whether the deployed model's numbers meant anything on real traffic.
        base_entry.route_score = decision.score
        base_entry.route_model = decision.model
        base_entry.route_reason = decision.reason or None
        eligible_set = set(eligibility.eligible)

        async def finish(
            usage: Usage,
            status: int,
            error: str | None,
            verification: VerificationOutcome | None = None,
        ) -> None:
            """Cost the request and write its row. Never raises."""
            base_entry.usage = usage
            # The counterfactual is costed from the ROUTED call's usage alone,
            # and that is the comparison this project is about: "what would this
            # request have cost had it gone straight to tier X" means one call at
            # tier X, not one call plus a review. The review's price goes on the
            # actual side of the ledger, where it belongs.
            base_entry.cost_usd = cost_usd(tier.prices, usage)
            base_entry.counterfactuals = counterfactuals(
                config, usage, eligible_tiers=eligible_set
            )
            base_entry.verification = verification
            base_entry.final_tier = tier_name
            if verification is not None:
                base_entry.cost_usd += verification.extra_cost_usd
                if verification.escalated and verification.escalated_to:
                    base_entry.final_tier = verification.escalated_to
            base_entry.billed_cost_usd = billed_total(usage, verification)
            base_entry.billed_source = "inline" if base_entry.billed_cost_usd is not None else None
            base_entry.latency_ms = _elapsed_ms(started)
            base_entry.http_status = status
            base_entry.error = error
            observe = getattr(active_router, "observe", None)
            if observe is not None:
                # What each call really spent, charged to the tier that made
                # it: an escalation spends the quota of the tier it went to.
                try:
                    observe(tier_name, usage, status)
                    if verification is not None and verification.escalated and verification.escalated_to:
                        observe(verification.escalated_to, verification.escalation_usage, 200)
                except Exception:  # noqa: BLE001 - accounting must not fail a request
                    pass
            await request_log.record_async(base_entry)
            if trace is not None:
                state = getattr(active_router, "state", None)
                stage(
                    "done",
                    status=status,
                    error=error,
                    final_tier=base_entry.final_tier,
                    usage=usage.model_dump(),
                    cost_usd=base_entry.cost_usd,
                    verdict=verification.verdict.value if verification is not None else None,
                    subscriptions=(state() or {}).get("subscriptions") if callable(state) else None,
                )
                traces.finish(trace, "ok" if status < 400 else "error")

        # --- call -------------------------------------------------------------
        # A stream cannot be verified: by the time an answer could be judged the
        # client has already read it. Recorded as a skip, so the rate is visible
        # rather than inferred from a gap in the table.
        stream_skip = _skip(active_verifier, config, tier_name, SkipReason.STREAMING)

        # Known before the call, so both paths send them. The streaming path
        # used to send only X-Request-Id: found on the first remote stream, where
        # a client had no way to learn which tier had answered it.
        headers = {"X-Request-Id": request_id, "X-Router-Tier": tier_name}
        if decision.score is not None:
            headers["X-Router-Score"] = f"{decision.score:.4f}"
            if decision.model:
                headers["X-Router-Model"] = decision.model

        stage("call", tier=tier_name, backend=tier.backend, model=tier.model, effort=tier.effort,
              stream=request.stream)
        if request.stream:
            events = backend.stream(request)
            try:
                # Pull the first event before returning a StreamingResponse. Once
                # streaming starts the status code is already on the wire, so a
                # backend that fails at connect time must be caught here or the
                # client sees a 200 that carries an error.
                first = await events.__anext__()
            except BackendError as exc:
                await finish(Usage(), exc.status, str(exc), stream_skip)
                return JSONResponse(status_code=exc.status, content=error_body(str(exc), kind="upstream_error"))
            except StopAsyncIteration:
                await finish(Usage(), 502, "backend produced no events", stream_skip)
                return JSONResponse(status_code=502, content=error_body("backend produced no events", kind="upstream_error"))

            async def body() -> AsyncIterator[str]:
                usage = Usage()
                status = 200
                error: str | None = None
                try:
                    pending: Any = first
                    while True:
                        if isinstance(pending, StreamEnd):
                            usage = pending.usage
                            break
                        if isinstance(pending, StreamChunk):
                            if base_entry.upstream_id is None:
                                base_entry.upstream_id = _upstream_id(pending.data)
                            yield _sse(pending.data)
                        try:
                            pending = await events.__anext__()
                        except StopAsyncIteration:
                            break
                    yield _DONE_FRAME
                except BackendError as exc:
                    status, error = exc.status, str(exc)
                    # The status line is long gone; the only way to tell the
                    # client is an error frame inside the stream it is reading.
                    yield _sse(error_body(str(exc), kind="upstream_error"))
                    yield _DONE_FRAME
                except Exception as exc:  # noqa: BLE001
                    status, error = 500, f"{type(exc).__name__}: {exc}"
                    yield _sse(error_body(error, kind="internal_error"))
                    yield _DONE_FRAME
                except BaseException:
                    # Cancellation: the client hung up. 499 is nginx's "client
                    # closed request" -- the 200 already on the wire is not what
                    # happened, and a row that says 200 with no tokens reads as
                    # a free success.
                    status = 499
                    error = "client disconnected before the stream finished"
                    raise
                finally:
                    # Shielded, or the row is lost. On disconnect Starlette
                    # cancels this task group, and anyio's cancellation is
                    # level-triggered: every await here is cancelled again,
                    # including the one that writes the row. Measured against a
                    # real uvicorn + Ollama: an abandoned stream left no row at
                    # all, not even after a clean shutdown. Its tokens were
                    # still generated and, on a paid tier, still billed --
                    # `reconcile` recovers them through `upstream_id`.
                    with anyio.CancelScope(shield=True):
                        await finish(usage, status, error, stream_skip)

            return StreamingResponse(
                body(),
                media_type="text/event-stream",
                headers={
                    **headers,
                    "Cache-Control": "no-cache",
                    **({"X-Router-Verdict": stream_skip.verdict.value} if stream_skip else {}),
                },
            )

        try:
            result = await backend.complete(request)
        except BackendError as exc:
            # The request failed, but the row is still written: an error costs
            # latency and sometimes tokens, and a log that only holds successes
            # cannot answer what a tier is really costing.
            await finish(
                Usage(),
                exc.status,
                str(exc),
                _skip(active_verifier, config, tier_name, SkipReason.UPSTREAM_ERROR),
            )
            content = exc.body if isinstance(exc.body, dict) and "error" in exc.body else error_body(str(exc), kind="upstream_error")
            return JSONResponse(status_code=exc.status, content=content)
        except Exception as exc:  # noqa: BLE001
            await finish(
                Usage(),
                500,
                f"{type(exc).__name__}: {exc}",
                _skip(active_verifier, config, tier_name, SkipReason.UPSTREAM_ERROR),
            )
            return JSONResponse(status_code=500, content=error_body(str(exc), kind="internal_error"))

        # --- verify -----------------------------------------------------------
        body = result.body
        try:
            stage("answer", text=str(body["choices"][0]["message"]["content"])[:PREVIEW_CHARS])
        except (KeyError, IndexError, TypeError):
            pass
        base_entry.upstream_id = _upstream_id(body)
        verification: VerificationOutcome | None = None
        if active_verifier is not None:
            verification = await active_verifier.check(
                request,
                served_tier=tier_name,
                body=body,
                eligible=list(eligibility.eligible),
            )
            headers["X-Router-Verdict"] = verification.verdict.value
            if verification.body is not None:
                # The escalated answer replaces the cheap one, so the client
                # never sees the answer that failed review.
                body = verification.body
                headers["X-Router-Tier"] = verification.escalated_to or tier_name
                headers["X-Router-Escalated-From"] = tier_name

        await finish(result.usage, 200, None, verification)
        return JSONResponse(status_code=200, content=body, headers=headers)

    return app


def _route_view(decision: Any, config: Config, began: float) -> dict[str, Any]:
    tier = config.tiers.get(decision.tier)
    return {
        "tier": decision.tier,
        "model": tier.model if tier else None,
        "effort": tier.effort if tier else None,
        "score": decision.score,
        "reason": decision.reason,
        "detail": decision.detail,
        "router_ms": int((time.perf_counter() - began) * 1000),
    }


def _skip(
    verifier: Verifier | None, config: Config, tier_name: str, reason: SkipReason
) -> VerificationOutcome | None:
    """A skip row, but only for a request verification would otherwise have taken.

    Returning None for a tier the loop never watches keeps the table's
    denominator honest: a skip rate computed over requests that were never
    candidates measures nothing.
    """
    if verifier is None or not config.verification.verifies(tier_name):
        return None
    return verifier.skip(reason)


def _upstream_id(payload: Any) -> str | None:
    value = payload.get("id") if isinstance(payload, dict) else None
    return value if isinstance(value, str) and value else None


def billed_total(usage: Usage, verification: VerificationOutcome | None) -> float | None:
    """The provider's own total for every call this request made, or None.

    All or nothing. A request that bought a review and an escalation has three
    bills; summing the two that were reported and dropping the third would sit
    beside a three-call estimate and read as a saving.
    """
    parts = [usage.billed_usd]
    if verification is not None:
        reviewed = verification.verifier_tier is not None and verification.verdict in (
            Verdict.PASS,
            Verdict.FAIL,
            Verdict.ERROR,
        )
        if reviewed:
            parts.append(verification.verifier_usage.billed_usd)
        if verification.escalated:
            parts.append(verification.escalation_usage.billed_usd)
    if any(part is None for part in parts):
        return None
    return sum(parts)  # type: ignore[arg-type]


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def app_from_config_path(path: str) -> FastAPI:
    """Entry point for `uvicorn`-style deployment."""
    from .config import load_config

    return create_app(load_config(path), config_path=path)
