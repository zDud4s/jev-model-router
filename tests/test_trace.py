"""The live routing view: traces of recent requests, and the page that draws them."""

from __future__ import annotations

import copy

from fastapi.testclient import TestClient

from jev_model_router.app import create_app
from jev_model_router.capabilities import CapabilityRouter
from jev_model_router.config import parse_config
from jev_model_router.db import RequestLog
from jev_model_router.trace import TraceStore

from conftest import BASE_CONFIG
from test_capabilities import Ask, HARD, raw_config

CHAT = {"model": "auto", "messages": [{"role": "user", "content": "Refactor the parser\nacross modules"}]}


def app_with(backend_factory, raw=None, router_ask=None):
    config = parse_config(raw or raw_config())
    router = CapabilityRouter(config, ask=router_ask or Ask(HARD))
    return create_app(config, backend_factory=backend_factory, log=RequestLog(":memory:"), router=router)


def test_a_request_is_traced_from_arrival_to_answer(backend_factory):
    with TestClient(app_with(backend_factory)) as client:
        assert client.post("/v1/chat/completions", json=CHAT).status_code == 200
        data = client.get("/routing/traces").json()
    [trace] = data["traces"]
    assert trace["kind"] == "request" and trace["status"] == "ok"
    assert trace["preview"].startswith("Refactor the parser")
    assert [s["name"] for s in trace["stages"]] == ["eligibility", "route", "call", "answer", "done"]
    route = next(s for s in trace["stages"] if s["name"] == "route")
    assert route["tier"] == "sub"
    detail = route["detail"]
    assert detail["needs"] == HARD and detail["pick"] == "sub" and "TASK PACKET" in detail["packet"]
    # Every option weighed, with its estimate and whether it reached the target.
    assert {o["tier"] for o in detail["options"]} == {"cheap", "mid", "top", "sub", "cx"}
    assert next(o for o in detail["options"] if o["tier"] == "sub")["covered"] is True
    done = trace["stages"][-1]
    assert done["status"] == 200 and done["usage"]["completion_tokens"] == 50
    assert "claude" in done["subscriptions"]
    # Stages are stamped in order.
    stamps = [s["t_ms"] for s in trace["stages"]]
    assert stamps == sorted(stamps)


def test_route_only_asks_the_router_and_runs_no_model(backend_factory, fake_backends):
    ask = Ask(HARD)
    with TestClient(app_with(backend_factory, router_ask=ask)) as client:
        response = client.post("/routing/dry-run", json={"messages": CHAT["messages"], "packet": {"size": "large"}})
        assert response.status_code == 200 and response.json()["tier"] == "sub"
        [trace] = client.get("/routing/traces").json()["traces"]
    assert trace["kind"] == "dry-run"
    assert [s["name"] for s in trace["stages"]] == ["eligibility", "route"]
    assert len(ask.calls) == 1 and "size: large" in ask.calls[0][0]
    assert all(not backend.calls for backend in fake_backends.values())


def test_a_dry_run_under_reject_with_jev_down_shows_the_refusal(backend_factory):
    app = app_with(backend_factory, raw=raw_config(on_jev_failure="reject"),
                   router_ask=Ask(error=RuntimeError("jev timed out")))
    with TestClient(app) as client:
        body = client.post("/routing/dry-run", json={"messages": CHAT["messages"]}).json()
        [trace] = client.get("/routing/traces").json()["traces"]
    assert body["tier"] is None and "jev timed out" in body["error"]
    assert trace["status"] == "error"


def test_polling_after_a_sequence_returns_only_what_moved(backend_factory):
    with TestClient(app_with(backend_factory)) as client:
        client.post("/routing/dry-run", json={"messages": CHAT["messages"]})
        first = client.get("/routing/traces").json()
        assert client.get(f"/routing/traces?after={first['seq']}").json()["traces"] == []
        client.post("/routing/dry-run", json={"messages": CHAT["messages"]})
        second = client.get(f"/routing/traces?after={first['seq']}").json()
    assert len(second["traces"]) == 1 and second["traces"][0]["id"] != first["traces"][0]["id"]


def test_the_page_is_served_and_turned_off_with_the_traces(backend_factory):
    with TestClient(app_with(backend_factory)) as client:
        page = client.get("/routing")
        assert page.status_code == 200 and "<title>Routing" in page.text
    raw = raw_config()
    raw["trace"] = {"enabled": False}
    with TestClient(app_with(backend_factory, raw=raw)) as client:
        assert client.get("/routing").status_code == 404
        assert client.get("/routing/traces").status_code == 404
        assert client.post("/v1/chat/completions", json=CHAT).status_code == 200


def test_an_ineligible_request_ends_its_trace_as_an_error(backend_factory):
    raw = copy.deepcopy(BASE_CONFIG)
    app = create_app(parse_config(raw), backend_factory=backend_factory, log=RequestLog(":memory:"))
    too_big = {"model": "auto", "messages": [{"role": "user", "content": "x" * 3_000_000}]}
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=too_big).status_code == 400
        [trace] = client.get("/routing/traces").json()["traces"]
    assert trace["status"] == "error"
    assert trace["stages"][0]["eligible"] == []


def test_the_store_keeps_only_the_newest():
    store = TraceStore(keep=2)
    for i in range(3):
        store.start(f"r{i}", kind="request", preview="", requested_model="auto")
    assert [t["id"] for t in store.since(0)] == ["r1", "r2"]
