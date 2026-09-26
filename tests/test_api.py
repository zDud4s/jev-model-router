"""The HTTP surface: OpenAI response shape, streaming, models, health."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from jev_model_router.app import create_app
from jev_model_router.backends.base import BackendError
from jev_model_router.db import RequestLog
from jev_model_router.schemas import Usage


@pytest.fixture
def client(config, backend_factory):
    app = create_app(config, backend_factory=backend_factory, log=RequestLog(":memory:"))
    with TestClient(app) as test_client:
        yield test_client


def test_healthz_reports_the_configured_tiers(client) -> None:
    body = client.get("/healthz").json()

    assert body["status"] == "ok"
    assert body["tiers"] == ["cheap", "mid", "top"]
    assert body["router"] == "static"
    assert body["schema_version"] >= 1


def test_models_lists_tiers_not_upstream_model_names(client) -> None:
    body = client.get("/v1/models").json()

    assert body["object"] == "list"
    assert [entry["id"] for entry in body["data"]] == ["cheap", "mid", "top"]
    assert all(entry["object"] == "model" for entry in body["data"])
    # The upstream name is exposed as a hint, but the id is what a client asks for.
    assert body["data"][0]["root"] == "llama3.1:8b"


def test_a_completion_has_the_openai_response_shape(client) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    assert body["model"] == "cheap"  # the tier, which is what the client can ask for
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "hello"
    assert choice["finish_reason"] == "stop"
    assert body["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "prompt_tokens_details": {"cached_tokens": 0},
    }


def test_a_streamed_completion_is_a_well_formed_sse_sequence(client) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = [line for line in response.text.split("\n\n") if line.strip()]
    assert frames[-1] == "data: [DONE]"

    chunks = [json.loads(frame[len("data: ") :]) for frame in frames[:-1]]
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "hello"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_a_stream_says_which_tier_answered_it(client) -> None:
    # Found on the first stream from a real remote provider: the non-streaming
    # path sent X-Router-Tier and the streaming path did not, so a streaming
    # client could not learn which tier had answered. The tier is decided before
    # the first byte, so nothing stops a stream from saying it.
    response = client.post(
        "/v1/chat/completions",
        json={"model": "top", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    assert response.headers["X-Router-Tier"] == "top"
    assert "X-Request-Id" in response.headers


def test_the_request_goes_to_the_default_tier(client, fake_backends) -> None:
    client.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})

    assert len(fake_backends["cheap"].calls) == 1
    assert fake_backends["mid"].calls == []


def test_an_explicitly_requested_tier_is_used(client, fake_backends) -> None:
    client.post("/v1/chat/completions", json={"model": "top", "messages": [{"role": "user", "content": "hi"}]})

    assert len(fake_backends["top"].calls) == 1


def test_a_request_no_tier_can_serve_is_refused_with_the_reasons(config, backend_factory) -> None:
    # Shrink every window so nothing can take the request; the gate must refuse
    # rather than send it to a backend that would reject it.
    from dataclasses import replace

    narrow = replace(
        config,
        tiers={name: replace(tier, context_window=10) for name, tier in config.tiers.items()},
    )
    app = create_app(narrow, backend_factory=backend_factory, log=RequestLog(":memory:"))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "x" * 5000}]},
        )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "no_eligible_tier"
    assert "context_window" in error["message"] or "exceeds context_window" in error["message"]


def test_an_upstream_error_reaches_the_client_with_its_own_status(config, fake_backends) -> None:
    from conftest import FakeBackend

    def factory(tier):
        backend = FakeBackend(tier, error=BackendError("rate limited", status=429))
        fake_backends[tier.name] = backend
        return backend

    app = create_app(config, backend_factory=factory, log=RequestLog(":memory:"))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
        )

    # A 429 must not reach the client as a 500, or its retry logic stops working.
    assert response.status_code == 429
    assert "rate limited" in response.json()["error"]["message"]


def test_a_malformed_body_is_a_400_in_the_openai_error_envelope(client) -> None:
    response = client.post("/v1/chat/completions", json={"messages": []})

    assert response.status_code == 400
    assert "error" in response.json()


def test_usage_reported_by_the_backend_is_passed_through(config, fake_backends) -> None:
    from conftest import FakeBackend

    def factory(tier):
        backend = FakeBackend(
            tier, usage=Usage(prompt_tokens=7, completion_tokens=3, total_tokens=10, cached_tokens=4)
        )
        fake_backends[tier.name] = backend
        return backend

    app = create_app(config, backend_factory=factory, log=RequestLog(":memory:"))
    with TestClient(app) as client:
        body = client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
        ).json()

    assert body["usage"]["prompt_tokens"] == 7
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 4
