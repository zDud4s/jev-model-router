"""The MCP server: the route-only API as tools, over JSON-RPC on stdio.

The tools go through the real `/v1/route` endpoints of an in-process app, so
these tests check the transport, not a second copy of the routing. Jev is never
called: the router answers from a scripted `Ask`.
"""

from __future__ import annotations

import io
import json

import httpx
import pytest

from jev_model_router import cli
from jev_model_router.capabilities import CapabilityRouter
from jev_model_router.config import parse_config
from jev_model_router.db import RequestLog
from jev_model_router.mcp_server import PROTOCOL_VERSIONS, TOOLS, McpServer, run

from test_capabilities import HARD, Ask, raw_config


def app_for(backend_factory, log=None):
    from jev_model_router.app import create_app

    config = parse_config(raw_config())
    router = CapabilityRouter(config, ask=Ask(HARD))
    return create_app(config, backend_factory=backend_factory, log=log or RequestLog(":memory:"), router=router)


def call(msg_id, name, arguments):
    return {"jsonrpc": "2.0", "id": msg_id, "method": "tools/call", "params": {"name": name, "arguments": arguments}}


def text_of(response):
    return response["result"]["content"][0]["text"]


@pytest.fixture
async def server(backend_factory):
    app = app_for(backend_factory)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://jev-model-router") as client:
            yield McpServer(client, runners=["claude"])


async def test_initialize_answers_the_version_asked_for_when_it_knows_it(server):
    known = await server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                 "params": {"protocolVersion": PROTOCOL_VERSIONS[-1]}})
    unknown = await server.handle({"jsonrpc": "2.0", "id": 2, "method": "initialize",
                                   "params": {"protocolVersion": "1999-01-01"}})
    assert known["result"]["protocolVersion"] == PROTOCOL_VERSIONS[-1]
    assert unknown["result"]["protocolVersion"] == PROTOCOL_VERSIONS[0]
    assert known["result"]["capabilities"] == {"tools": {}}


async def test_notifications_get_no_answer_and_unknown_methods_an_error(server):
    assert await server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert (await server.handle({"jsonrpc": "2.0", "id": 3, "method": "resources/list"}))["error"]["code"] == -32601
    assert (await server.handle({"id": 4, "method": "ping"}))["error"]["code"] == -32600
    assert (await server.handle({"jsonrpc": "2.0", "id": 5, "method": "ping"}))["result"] == {}


async def test_tools_list_names_route_and_report_outcome(server):
    tools = (await server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}))["result"]["tools"]
    assert [t["name"] for t in tools] == ["route", "report_outcome"]
    assert all(t["inputSchema"]["type"] == "object" for t in tools)


async def test_route_is_the_http_decision_limited_to_the_servers_runner(server, fake_backends):
    response = await server.handle(call(1, "route", {"task": "Fix the race in the scheduler"}))
    body = json.loads(text_of(response))
    assert response["result"]["isError"] is False
    assert body["runner"] == "claude" and body["tier"] == "sub"
    assert body["decision_id"].startswith("rt_")
    assert all(not b.calls for b in fake_backends.values())  # nothing was run


async def test_runners_given_by_the_agent_win_over_the_default(server):
    response = await server.handle(call(1, "route", {"task": "Fix it", "runners": ["codex"]}))
    assert json.loads(text_of(response))["runner"] == "codex"


async def test_an_endpoint_refusal_is_a_tool_error_the_agent_reads(server):
    response = await server.handle(call(1, "route", {"task": "Fix it", "runners": ["nobody"]}))
    assert response["result"]["isError"] is True
    assert "HTTP 422" in text_of(response) and "no eligible tier" in text_of(response)


async def test_an_outcome_is_recorded_once(server):
    routed = json.loads(text_of(await server.handle(call(1, "route", {"task": "Fix it"}))))
    first = await server.handle(call(2, "report_outcome", {"decision_id": routed["decision_id"], "status": "pass"}))
    again = await server.handle(call(3, "report_outcome", {"decision_id": routed["decision_id"], "status": "fail"}))
    assert json.loads(text_of(first))["status"] == "pass"
    assert again["result"]["isError"] is True and "HTTP 409" in text_of(again)


async def test_an_outcome_without_a_decision_id_is_refused_before_any_call(server):
    response = await server.handle(call(1, "report_outcome", {"status": "pass"}))
    assert response["result"]["isError"] is True and "decision_id" in text_of(response)


async def test_an_unknown_tool_is_an_invalid_params_error(server):
    assert (await server.handle(call(1, "nope", {})))["error"]["code"] == -32602


async def test_stdio_answers_every_request_and_survives_bad_json(backend_factory):
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": PROTOCOL_VERSIONS[0]}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        call(2, "route", {"task": "Fix the race in the scheduler"}),
    ]
    stdin = io.BytesIO(b"not json\n" + b"".join(json.dumps(m).encode() + b"\n" for m in lines))
    stdout = io.BytesIO()
    await run(app=app_for(backend_factory), runners=["claude"], stdin=stdin, stdout=stdout)
    answers = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert sorted(a.get("id") or 0 for a in answers) == [0, 1, 2]
    assert next(a for a in answers if a["id"] is None)["error"]["code"] == -32700
    assert json.loads(text_of(next(a for a in answers if a["id"] == 2)))["runner"] == "claude"


async def test_a_proxy_that_is_not_running_is_named_in_the_tool_error():
    transport = httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:9") as client:
        response = await McpServer(client).handle(call(1, "route", {"task": "Fix it"}))
    assert response["result"]["isError"] is True and "jev-model-router serve" in text_of(response)


def test_the_cli_names_the_env_var_when_the_config_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["-c", str(tmp_path / "absent.yaml"), "mcp"]) == 2
    assert "JEV_MODEL_ROUTER_CONFIG" in capsys.readouterr().err


def test_the_config_defaults_to_the_env_var(monkeypatch):
    monkeypatch.setenv(cli.CONFIG_ENV, "/somewhere/router.yaml")
    assert cli._build_parser().parse_args(["check"]).config == "/somewhere/router.yaml"
    monkeypatch.delenv(cli.CONFIG_ENV)
    assert cli._build_parser().parse_args(["check"]).config == cli.DEFAULT_CONFIG


def test_mcp_uses_a_running_proxy_when_the_env_var_names_one(monkeypatch):
    monkeypatch.setenv(cli.URL_ENV, "http://127.0.0.1:8080")
    assert cli._build_parser().parse_args(["mcp"]).url == "http://127.0.0.1:8080"
    monkeypatch.delenv(cli.URL_ENV)
    assert cli._build_parser().parse_args(["mcp"]).url is None


async def test_a_jev_refusal_reaches_the_agent_as_a_tool_error(backend_factory):
    from jev_model_router.app import create_app

    config = parse_config(raw_config(on_jev_failure="reject"))
    router = CapabilityRouter(config, ask=Ask(error=RuntimeError("jev timed out")))
    app = create_app(config, backend_factory=backend_factory, log=RequestLog(":memory:"), router=router)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://jev-model-router") as client:
            response = await McpServer(client, runners=["claude"]).handle(call(1, "route", {"task": "Fix it"}))
    assert response["result"]["isError"] is True
    assert "HTTP 503" in text_of(response) and "on_jev_failure is reject" in text_of(response)


async def test_an_outcome_carries_cache_tokens_through(server):
    props = next(t for t in TOOLS if t["name"] == "report_outcome")["inputSchema"]["properties"]["usage"]["properties"]
    assert {"cached_tokens", "cache_write_tokens"} <= set(props)
    assert "cache" in props["prompt_tokens"]["description"].lower()
    routed = json.loads(text_of(await server.handle(call(1, "route", {"task": "Fix it"}))))
    assert routed["cost_basis"] in ("one_call", "task_shape")
    usage = {"prompt_tokens": 3000, "completion_tokens": 10, "cached_tokens": 2900, "cache_write_tokens": 60}
    response = await server.handle(call(2, "report_outcome",
                                        {"decision_id": routed["decision_id"], "status": "pass", "usage": usage}))
    assert response["result"]["isError"] is False
