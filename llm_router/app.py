"""The HTTP surface: an OpenAI-compatible proxy.

Request lifecycle, in the order it must happen:

    parse -> eligibility gate -> route -> call backend -> cost -> log

The gate runs before the router, never after. See `eligibility.py` for why that
ordering is the point rather than an implementation detail.

Logging is the last step and is allowed to fail. Nothing in this module lets a
log failure change what the client receives.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from .backends import (
    Backend,
    BackendError,
    BackendFactory,
    StreamChunk,
    StreamEnd,
    build_backends,
)
from .config import Config
from .db import LogEntry, RequestLog
from .eligibility import evaluate
from .pricing import cost_usd, counterfactuals
from .routing import Router, build_router
from .schemas import ChatCompletionRequest, ModelList, Usage, error_body, model_card
from .verification import (
    SkipReason,
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


def create_app(
    config: Config,
    *,
    backend_factory: BackendFactory | None = None,
    log: RequestLog | None = None,
    router: Router | None = None,
    verifier: Verifier | None = None,
) -> FastAPI:
    """Build the app. Every collaborator is injectable, which is how tests avoid the network."""

    request_log = log or RequestLog(config.log.path, store_prompts=config.log.store_prompts)
    backends: dict[str, Backend] = build_backends(config, backend_factory)
    active_router: Router = router or build_router(config)
    # None when verification is off, so the request path has one branch rather
    # than a cascade of `if config.verification.enabled` checks.
    active_verifier: Verifier | None = verifier or build_verifier(config, backends)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        for backend in backends.values():
            await backend.aclose()
        request_log.close()

    app = FastAPI(title="llm-router", version="0.1.0", lifespan=lifespan)
    app.state.config = config
    app.state.log = request_log
    app.state.backends = backends
    app.state.router = active_router
    app.state.verifier = active_verifier

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "tiers": sorted(config.tiers),
            "router": active_router.name,
            "verification": active_verifier is not None,
            "schema_version": request_log.schema_version,
            # Surfaced here so a silently broken log is discoverable without
            # reading the database.
            "log_failures": request_log.failed_writes,
        }

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

        # --- eligibility gate, before any routing decision --------------------
        eligibility = evaluate(config, request)
        base_entry = LogEntry(
            request_id=request_id,
            prompt_text=prompt,
            requested_model=request.model,
            router=active_router.name,
            stream=request.stream,
            rejections=eligibility.rejections,
        )

        if not eligibility.eligible:
            reasons = "; ".join(f"{r.tier}: {r.detail}" for r in eligibility.rejections)
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
        tier_name = active_router.choose(request, list(eligibility.eligible))
        tier = config.tier(tier_name)
        backend = backends[tier_name]
        base_entry.tier = tier_name
        base_entry.backend = tier.backend
        base_entry.model = tier.model
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
            base_entry.latency_ms = _elapsed_ms(started)
            base_entry.http_status = status
            base_entry.error = error
            await request_log.record_async(base_entry)

        # --- call -------------------------------------------------------------
        # A stream cannot be verified: by the time an answer could be judged the
        # client has already read it. Recorded as a skip, so the rate is visible
        # rather than inferred from a gap in the table.
        stream_skip = _skip(active_verifier, config, tier_name, SkipReason.STREAMING)

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
                finally:
                    # Runs on client disconnect too, so an abandoned stream is
                    # still costed with the tokens it actually consumed.
                    await finish(usage, status, error, stream_skip)

            return StreamingResponse(
                body(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Request-Id": request_id},
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
        verification: VerificationOutcome | None = None
        headers = {"X-Request-Id": request_id, "X-Router-Tier": tier_name}
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


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def app_from_config_path(path: str) -> FastAPI:
    """Entry point for `uvicorn`-style deployment."""
    from .config import load_config

    return create_app(load_config(path))
