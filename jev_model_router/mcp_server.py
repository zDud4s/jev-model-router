"""MCP server over stdio: the route-only API as tools an agent can call.

An agent that delegates work -- a subagent in Claude Code, a `codex exec` in
Codex -- is exactly the caller `/v1/route` was built for: it runs the model
itself and only needs to be told which one. MCP is how those agents take a
tool, so this module speaks it and forwards each call to the two route
endpoints unchanged. There is one decision path, the HTTP one; this is a
transport, not a second router.

With delegation on (`mcp --delegate`, or JEV_MODEL_ROUTER_DELEGATE=1), a server with a runner of its
own also offers the tiers another agent's CLI on this machine can run, and each answer says how to run
it: `run: subagent` (this agent runs it), `delegate` (a `jev-model-router delegate` command to run in the
shell), or `unavailable`. A delegated run (JEV_MODEL_ROUTER_DEPTH=1) is offered its own runner only.

Two ways to reach the endpoints:

* **In process** (default): the app is built from the config here and called
  through an ASGI transport. Nothing else has to be running.
* **`--url`**: a proxy already started with `jev-model-router serve`, so the agent
  shares its log, subscription windows and routing view.

The protocol is JSON-RPC 2.0, one message per line. Only what a tool server
needs is implemented: `initialize`, `ping`, `tools/list`, `tools/call`.
Nothing but protocol messages may reach stdout, so the caller points
`sys.stdout` at stderr and hands `run` the real stdout.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import sys
from typing import IO, Any, Awaitable, Callable
from urllib.parse import quote

import httpx

from .delegates import Target, adapter_for
from .route_api import OUTCOMES

# Newest first. A client asking for one of these gets it back; any other gets the newest.
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "jev-model-router", "version": "0.1.0"}

_ROUTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task": {"type": "string", "description": "The work to route, as you would brief the agent that does it."},
        "stage": {"type": "string", "description": "Optional label, e.g. plan, implement, review. Logged."},
        "files": {"type": "array", "items": {"type": "string"}, "description": "Optional: files the task touches."},
        "attempt": {"type": "integer", "description": "Optional: 1 for the first try, 2 for the first retry..."},
        "gate_output": {"type": "string", "description": "Optional: why the last attempt failed (test output, review)."},
        "failed": {
            "type": "array", "items": {"type": "string"},
            "description": "Optional: what already failed this task, as tier names or model[@effort].",
        },
        "runners": {
            "type": "array", "items": {"type": "string"},
            "description": "Optional: only tiers these CLIs run, e.g. [\"claude\"]. Defaults to what this server can run.",
        },
        "packet": {"type": "object", "description": "Optional: any other context, passed to the router as-is."},
        "models": {
            "type": "array", "items": {"type": "string"},
            "description": "Optional: only these may be chosen. Globs over tier names and model ids, "
                           "e.g. [\"vendor/*\", \"*-mini\"]. Use when the user names which models are allowed.",
        },
        "exclude": {
            "type": "array", "items": {"type": "string"},
            "description": "Optional: none of these may be chosen. Globs, as for models.",
        },
    },
    "required": ["task"],
    "additionalProperties": False,
}

_OUTCOME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision_id": {"type": "string", "description": "The decision_id the route tool returned."},
        "status": {
            "type": "string", "enum": list(OUTCOMES),
            "description": "pass/fail: what your check said of the work. rate_limited: the subscription "
                           "refused. error: the run broke for a reason that says nothing about the model.",
        },
        "detail": {"type": "string", "description": "Optional: one line on why."},
        "usage": {
            "type": "object",
            "properties": {
                "prompt_tokens": {
                    "type": "integer",
                    "description": "All input tokens for the run, cached and cache-write tokens included, "
                                   "not only the fresh part.",
                },
                "completion_tokens": {"type": "integer"},
                "cached_tokens": {"type": "integer", "description": "Input tokens read from the prompt cache."},
                "cache_write_tokens": {"type": "integer", "description": "Input tokens written to the prompt cache."},
            },
            "description": "Optional: tokens the whole run used, if known. Leave out what you do not know.",
        },
    },
    "required": ["decision_id", "status"],
    "additionalProperties": False,
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "route",
        "description": (
            "Ask which model and effort should do a task before you delegate it. Nothing is run and no "
            "quota is spent. Returns runner, model, effort, the router's success estimate and a "
            "decision_id; `run` says how to run it from here (subagent, delegate with a command, or "
            "unavailable). Report what happened with report_outcome."
        ),
        "inputSchema": _ROUTE_SCHEMA,
    },
    {
        "name": "report_outcome",
        "description": (
            "Report how a routed task went, once, after your check (tests, review) ran. This is what the "
            "router learns each model's real success rate from."
        ),
        "inputSchema": _OUTCOME_SCHEMA,
    },
]

# JSON-RPC error codes.
_PARSE_ERROR, _INVALID_REQUEST, _METHOD_NOT_FOUND, _INVALID_PARAMS = -32700, -32600, -32601, -32602


class McpServer:
    """Answers one JSON-RPC message at a time; `client` reaches the route endpoints."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        runners: list[str] | None = None,
        delegate: bool = False,
        depth: int = 0,
        delegate_via: list[str] | None = None,
        powershell: bool = False,
        which: Callable[[str], str | None] = shutil.which,
    ) -> None:
        self._client = client
        self._own = frozenset(runners or [])
        self._delegate = delegate and depth < 1
        self._depth = depth
        # How `delegate` reaches this server's decisions: ["-c", <absolute config>] or ["--url", <url>].
        self._via = list(delegate_via or [])
        self._powershell = powershell
        self._which = which
        self._runnable: frozenset[str] | None = None

    async def handle(self, message: Any) -> dict[str, Any] | None:
        """The response to `message`, or None for a notification."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or "method" not in message:
            return _error(message.get("id") if isinstance(message, dict) else None,
                          _INVALID_REQUEST, "not a JSON-RPC 2.0 request")
        if "id" not in message:
            return None  # notifications/initialized, cancellations: nothing to answer
        msg_id, method = message["id"], message["method"]
        params = message.get("params") or {}
        if method == "initialize":
            asked = params.get("protocolVersion")
            return _result(msg_id, {
                "protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            })
        if method == "ping":
            return _result(msg_id, {})
        if method == "tools/list":
            return _result(msg_id, {"tools": TOOLS})
        if method == "tools/call":
            name, arguments = params.get("name"), params.get("arguments") or {}
            if not isinstance(arguments, dict):
                return _error(msg_id, _INVALID_PARAMS, "arguments must be an object")
            if name == "route":
                return _result(msg_id, await self._route(arguments))
            if name == "report_outcome":
                return _result(msg_id, await self._outcome(arguments))
            return _error(msg_id, _INVALID_PARAMS, f"unknown tool {name!r}")
        return _error(msg_id, _METHOD_NOT_FOUND, f"unknown method {method!r}")

    async def _route(self, arguments: dict[str, Any]) -> dict[str, Any]:
        body = dict(arguments)
        asked = body.get("runners") or []
        allowed = await self._allowed()
        if allowed is not None:
            runners = [r for r in asked if r in allowed] if asked else sorted(allowed)
            if not runners:
                # Forwarding an empty list would mean "any runner".
                return _tool_error(f"none of {asked} can run from here; this server can run {sorted(allowed)}")
            body["runners"] = runners
        elif self._own and not asked:
            body["runners"] = sorted(self._own)
        payload, error = await self._request("POST", "/v1/route", body)
        if error is not None:
            return error
        payload.update(await self._how_to_run(payload))
        return _ok(payload)

    async def _allowed(self) -> frozenset[str] | None:
        """The runners this server may offer; None for the old rule (an explicit list, else --runner)."""
        if not self._own:
            return None  # no runner of its own: never widens or intersects
        if self._depth >= 1:
            return self._own
        if not self._delegate:
            return None
        return self._own | await self._delegable()

    async def _delegable(self) -> frozenset[str]:
        """Runners of tiers an adapter can run on this machine. Asked once: PATH does not change under a session."""
        if self._runnable is None:
            payload, error = await self._request("GET", "/v1/route/targets", None)
            found: set[str] = set()
            for raw in (payload or {}).get("targets", []) if error is None else []:
                try:
                    target = Target.from_dict(raw)
                except (KeyError, TypeError, ValueError):
                    continue
                adapter = adapter_for(target.runner)
                if adapter is not None and adapter.available(target, self._which):
                    found.add(target.runner)
            self._runnable = frozenset(found)
        return self._runnable

    async def _how_to_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        runner = payload.get("runner")
        if runner in self._own:
            return {"run": "subagent"}
        if self._delegate and self._via and runner in await self._delegable():
            # The interpreter running this server: the entry point may not be on the agent shell's PATH.
            command = [sys.executable, "-m", "jev_model_router", "delegate", str(payload["decision_id"]), *self._via]
            return {"run": "delegate", "command": command,
                    "command_line": command_line(command, powershell=self._powershell)}
        return {"run": "unavailable"}

    async def _outcome(self, arguments: dict[str, Any]) -> dict[str, Any]:
        body = dict(arguments)
        decision_id = body.pop("decision_id", None)
        if not isinstance(decision_id, str) or not decision_id:
            return _tool_error("'decision_id' is required: the one the route tool returned")
        payload, error = await self._request("POST", f"/v1/route/{quote(decision_id, safe='')}/outcome", body)
        return error if error is not None else _ok(payload)

    async def _request(self, method: str, path: str, body: dict[str, Any] | None) -> tuple[Any, dict[str, Any] | None]:
        """(payload, None), or (None, a tool error the agent reads)."""
        try:
            response = await self._client.request(method, path, json=body)
        except httpx.HTTPError as exc:
            return None, _tool_error(f"cannot reach the router at {self._client.base_url}: {type(exc).__name__}: "
                                     f"{exc}. Is `jev-model-router serve` running?")
        try:
            payload = response.json()
        except ValueError:
            payload = {"error": {"message": response.text[:500]}}
        if response.status_code >= 400:
            message = payload.get("error", {}).get("message") if isinstance(payload, dict) else None
            return None, _tool_error(f"HTTP {response.status_code}: {message or payload}")
        return payload, None


def _result(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _tool_error(message: str) -> dict[str, Any]:
    # A failed call is a tool result the agent reads, not a protocol error.
    return {"content": [{"type": "text", "text": message}], "isError": True}


def _ok(payload: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}], "isError": False}


def command_line(argv: list[str], *, powershell: bool) -> str:
    """`argv` as one line for the host agent's shell: PowerShell's call operator, or POSIX quoting."""
    if powershell:
        return "& " + " ".join("'" + arg.replace("'", "''") + "'" for arg in argv)
    return shlex.join(argv)


async def serve_stdio(
    handle: Callable[[Any], Awaitable[dict[str, Any] | None]],
    stdin: IO[bytes],
    stdout: IO[bytes],
) -> None:
    """Read messages until EOF. Each is answered as its own task, so a slow route never blocks a ping."""
    lock = asyncio.Lock()
    pending: set[asyncio.Task[None]] = set()

    async def write(response: dict[str, Any]) -> None:
        data = json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n"
        async with lock:
            stdout.write(data)
            stdout.flush()

    async def answer(message: Any) -> None:
        try:
            response = await handle(message)
        except Exception as exc:  # noqa: BLE001 - one bad call must not end the session
            msg_id = message.get("id") if isinstance(message, dict) else None
            response = _error(msg_id, -32603, f"{type(exc).__name__}: {exc}") if msg_id is not None else None
        if response is not None:
            await write(response)

    while True:
        line = await asyncio.to_thread(stdin.readline)
        if not line:
            break
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            await write(_error(None, _PARSE_ERROR, f"invalid JSON: {exc}"))
            continue
        task = asyncio.create_task(answer(message))
        pending.add(task)
        task.add_done_callback(pending.discard)
    if pending:
        await asyncio.gather(*pending)


async def run(*, app: Any = None, url: str | None = None, runners: list[str] | None = None,
              stdin: IO[bytes] | None = None, stdout: IO[bytes] | None = None, **options: Any) -> None:
    """Serve until stdin closes: in process when given `app`, else against the proxy at `url`.

    `options` are McpServer's delegation settings (delegate, depth, delegate_via, powershell).
    """
    stdin = stdin or sys.stdin.buffer
    stdout = stdout or sys.stdout.buffer
    if app is not None:
        # The app's lifespan closes its backends and log; ASGITransport does not run it.
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://jev-model-router", timeout=None) as client:
                await serve_stdio(McpServer(client, runners=runners, **options).handle, stdin, stdout)
        return
    if not url:
        raise ValueError("run() needs an app or a url")
    async with httpx.AsyncClient(base_url=url.rstrip("/"), timeout=httpx.Timeout(600.0, connect=5.0)) as client:
        await serve_stdio(McpServer(client, runners=runners, **options).handle, stdin, stdout)


__all__ = ["McpServer", "PROTOCOL_VERSIONS", "TOOLS", "command_line", "run", "serve_stdio"]
