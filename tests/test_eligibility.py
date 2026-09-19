"""The eligibility gate: the hard filter that runs before routing."""

from __future__ import annotations

from llm_router.eligibility import RejectionReason, evaluate
from llm_router.routing import StaticRouter
from llm_router.tokens import estimate_prompt_tokens, estimate_request_budget

from conftest import make_request


def test_an_over_window_request_is_rejected_by_the_small_tier(config) -> None:
    # `cheap` declares a 4096-token window; this prompt cannot fit in it.
    request = make_request(messages=[{"role": "user", "content": "x" * 200_000}])

    result = evaluate(config, request)

    assert "cheap" not in result.eligible
    rejection = next(r for r in result.rejections if r.tier == "cheap")
    assert rejection.reason is RejectionReason.CONTEXT_WINDOW
    # The reason is logged, not merely implied.
    assert str(config.tiers["cheap"].context_window) in rejection.detail
    assert result.eligible == ["mid", "top"]


def test_the_output_reservation_counts_against_the_window(config) -> None:
    # Input alone fits; input plus the caller's declared output does not. A gate
    # that only measured the prompt would let this through and the backend would
    # reject it, which is the failure the gate exists to prevent.
    # Sized against the fixture's 4096-token window: ~3600 input tokens leaves room
    # under it, and the 1000 the caller reserves for the answer takes it over.
    request = make_request(
        messages=[{"role": "user", "content": "x" * 12_600}],
        max_tokens=1_000,
    )

    window = config.tiers["cheap"].context_window
    # Both halves, or the test passes for the wrong reason: a prompt that overflows on
    # its own would satisfy the second assertion while proving nothing about output.
    assert estimate_prompt_tokens(request) < window
    assert estimate_request_budget(request) > window
    assert "cheap" not in evaluate(config, request).eligible


def test_a_tools_request_is_rejected_by_a_tool_less_backend(config) -> None:
    request = make_request(
        tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    )

    result = evaluate(config, request)

    assert "cheap" not in result.eligible
    rejection = next(r for r in result.rejections if r.tier == "cheap")
    assert rejection.reason is RejectionReason.TOOLS_UNSUPPORTED
    assert "supports_tools" in rejection.detail


def test_every_tier_is_eligible_for_an_ordinary_request(config) -> None:
    result = evaluate(config, make_request())

    assert result.eligible == ["cheap", "mid", "top"]
    assert result.rejections == []


def test_the_router_never_returns_a_filtered_out_tier(config) -> None:
    # The default tier is `cheap`, but a tools request makes it ineligible.
    # Routing must not hand back a tier the gate removed.
    request = make_request(
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}]
    )
    result = evaluate(config, request)

    chosen = StaticRouter(config).choose(request, list(result.eligible))

    assert chosen in result.eligible
    assert chosen != "cheap"


def test_an_explicit_tier_request_is_honoured_when_eligible(config) -> None:
    request = make_request(model="top")
    result = evaluate(config, request)

    assert StaticRouter(config).choose(request, list(result.eligible)) == "top"


def test_the_model_map_steers_a_client_that_hardcodes_a_model_name() -> None:
    from llm_router.config import parse_config

    from conftest import BASE_CONFIG

    raw = {**BASE_CONFIG, "router": {"kind": "static", "default_tier": "cheap", "model_map": {"gpt-4o": "top"}}}
    cfg = parse_config(raw)
    request = make_request(model="gpt-4o")

    assert StaticRouter(cfg).choose(request, ["cheap", "mid", "top"]) == "top"
