"""An API's `/models` listing as a discover source, and the caller's shortlist per decision.

The listing is scripted here: nothing touches the network. Its shape is the
richest one met so far (per-model context, prices, parameters, efforts, alias
targets, `:variant` ids); the reader must also take a bare `data[].id` list.
"""

from __future__ import annotations

import copy
import json

import pytest
from fastapi.testclient import TestClient

from jev_model_router.app import create_app
from jev_model_router.capabilities import CapabilityRouter
from jev_model_router.catalog import Offered, check_catalog, models_source
from jev_model_router.config import parse_config
from jev_model_router.db import RequestLog
from jev_model_router.discovery import expand
from jev_model_router.eligibility import evaluate
from jev_model_router.mcp_server import TOOLS
from jev_model_router.schemas import ChatCompletionRequest
from jev_model_router.scores_derive import served_keys

from test_capabilities import HARD, Ask
from test_capabilities import raw_config as caps_raw_config
from test_discovery import raw_config, report, served
from test_scores import linked, pt, scored

API = "https://api.example/v1"


def entry(model_id, **extra):
    return {"id": model_id, "context_length": 128000,
            "pricing": {"prompt": "0.000001", "completion": "0.000004", "input_cache_read": "0.0000001"},
            "supported_parameters": ["tools", "reasoning", "temperature"],
            "architecture": {"output_modalities": ["text"]}, **extra}


LISTING = {"data": [
    entry("vendor/big", reasoning={"supported_efforts": ["high", "medium", "low"]},
          canonical_slug="vendor/big-20260901"),
    entry("vendor/big:free", pricing={"prompt": "0", "completion": "0"}),
    entry("vendor/small", supported_parameters=["temperature"], context_length=None,
          top_provider={"context_length": 32000}),
    entry("~vendor/big-latest", alias_target="vendor/big"),
    entry("router/auto", pricing={"prompt": "-1", "completion": "-1"}),
    entry("vendor/painter", architecture={"output_modalities": ["image"]}),
    entry("other/orphan:beta"),
]}


def listed(payload=LISTING):
    return models_source(API, get=lambda url: payload)


# ---------------------------------------------------------------- reading the listing
def test_each_model_carries_what_the_listing_states():
    big = listed().models["vendor/big"]
    assert big.efforts == ("high", "medium", "low") and big.aliases == ("vendor/big-20260901",)
    assert (big.context_window, big.supports_tools) == (128000, True)
    assert big.prices == {"input": 1.0, "output": 4.0, "cache_read": 0.1}
    small = listed().models["vendor/small"]
    assert (small.context_window, small.supports_tools, small.efforts) == (32000, False, None)


def test_variants_and_aliases_name_the_model_they_are_only_when_it_is_listed():
    models = listed().models
    assert models["vendor/big:free"].same_as == "vendor/big"
    assert models["~vendor/big-latest"].same_as == "vendor/big"
    assert models["other/orphan:beta"].same_as is None  # its base is not listed


def test_a_model_that_cannot_write_or_be_priced_is_marked():
    models = listed().models
    assert "vendor/painter" not in models
    assert models["router/auto"].unpriceable and models["router/auto"].prices is None


def test_a_bare_listing_is_ids_and_nothing_more():
    bare = listed({"object": "list", "data": [{"id": "m1"}, {"id": "m2"}]}).models
    assert bare["m1"] == Offered("m1")


def test_an_unreachable_listing_is_an_error_not_an_empty_catalog():
    def fail(url):
        raise OSError("refused")

    source = models_source(API, get=fail)
    assert source.models == {} and "refused" in source.error


def test_the_key_is_sent_when_the_source_names_one(monkeypatch):
    import jev_model_router.catalog as catalog

    seen = {}

    def fake(url, headers=None, timeout=5.0):
        seen.update(url=url, headers=headers)
        return {"data": []}

    monkeypatch.setattr(catalog, "_get_json", fake)
    models_source(API, api_key="sk-1")
    assert seen == {"url": f"{API}/models", "headers": {"Authorization": "Bearer sk-1"}}


def api_raw(**source):
    raw = raw_config()
    raw["router"]["capabilities"]["discover"] = {
        "api": {"backend": "openai_compatible", "base_url": API, "api_key_env": "API_KEY", **source},
    }
    return raw


def test_the_catalog_check_lists_a_discover_api_but_not_an_explicit_tier_elsewhere():
    raw = api_raw()
    raw["tiers"]["remote"] = {"backend": "openai_compatible", "model": "m", "base_url": "https://elsewhere/v1"}
    seen = []
    result = check_catalog(parse_config(raw), get=lambda url: seen.append(url) or LISTING)
    assert f"{API}/models" in seen and not any("elsewhere" in url for url in seen)
    assert "vendor/big" in result.discovered["api"].models
    assert result.tiers["remote"].status == "not_checked"


# ---------------------------------------------------------------- expand
def expanded(scores=None, **source):
    return expand(parse_config(api_raw(**source)), report(api=listed()), scores)


def test_a_discovered_api_tier_is_built_from_its_listing():
    config, found, _ = expanded()
    tier = config.tiers["api:vendor/small"]
    assert (tier.context_window, tier.supports_tools, tier.api_key_env) == (32000, False, "API_KEY")
    assert tier.prices.configured and tier.prices.output == 4.0
    assert "api:router/auto" not in config.tiers  # unpriceable
    assert {d.tier for d in found} >= {"api:vendor/big", "api:vendor/big:free"}


def test_without_an_effort_body_an_api_model_is_one_tier_at_its_default():
    config, _, _ = expanded()
    assert [n for n in config.tiers if n.startswith("api:vendor/big") and "@" in n] == []


def test_an_effort_body_makes_one_tier_per_effort_carrying_it():
    config, _, _ = expanded(effort_body={"reasoning": {"effort": "{effort}"}}, extra_body={"usage": {"include": True}})
    tier = config.tiers["api:vendor/big@medium"]
    assert tier.extra_body == {"usage": {"include": True}, "reasoning": {"effort": "medium"}}
    assert tier.effort == "medium" and "api:vendor/big" not in config.tiers
    assert config.tiers["api:vendor/small"].extra_body == {"usage": {"include": True}}


def test_include_and_exclude_narrow_the_listing_by_any_id():
    config, _, _ = expanded(include=["vendor/*"], exclude=["*:free", "vendor/big-2026*"])
    names = {n for n in config.tiers if n.startswith("api:")}
    assert names == {"api:vendor/small"}  # big is excluded through its canonical alias


def test_require_evidence_keeps_only_profiled_or_benchmarked_models():
    raw_profiles = [{"match": "vendor/small", "level": 1.0, "list_prices": {"input": 1, "output": 1}}]
    config, found, _ = expand(
        parse_config(_with_profiles(api_raw(require_evidence=True), raw_profiles)), report(api=listed())
    )
    assert {d.tier for d in found} == {"api:vendor/small"}
    assert "api:vendor/big" not in config.tiers and "api:vendor/big" not in config.router.capabilities.cards


def test_require_evidence_accepts_benchmark_points_under_any_of_a_models_ids():
    _, scores = scored(linked(pt("code", "big", "high", 70), pt("code", "acme-small", None, 40)))
    config, found, _ = expanded(scores, require_evidence=True)
    kept = {d.tier for d in found}
    assert {"api:vendor/big", "api:vendor/big:free", "api:~vendor/big-latest"} <= kept
    assert "api:vendor/small" not in kept


def _with_profiles(raw, profiles):
    raw["router"]["capabilities"]["profiles"] = profiles
    return raw


# ---------------------------------------------------------------- one model, two catalogs
def test_two_catalogs_spelling_one_model_differently_share_its_evidence():
    _, scores = scored(linked(pt("code", "acme-large", "high", 80)))
    keys, ambiguous = served_keys(scores, {"cli:acme-large": ("acme-large",),
                                           "api:vendor/acme.large": ("vendor/acme.large",)})
    assert ambiguous == set() and keys["cli:acme-large"] == keys["api:vendor/acme.large"] == ("acme-large",)


def test_one_catalog_listing_two_spellings_of_one_key_is_still_ambiguous():
    _, scores = scored(linked(pt("code", "acme-large", "high", 80)))
    _, ambiguous = served_keys(scores, {"api:vendor/acme-large": ("vendor/acme-large",),
                                        "api:vendor/acme_large": ("vendor/acme_large",)})
    assert ambiguous == {"acme-large"}


def test_a_variant_is_owned_by_its_base_and_is_not_an_ambiguity():
    from jev_model_router.discovery import ids_of

    models = listed().models
    ids = {f"api:{m}": ids_of(models[m]) for m in ("vendor/big", "vendor/big:free", "~vendor/big-latest")}
    assert ids["api:vendor/big:free"][0] == "vendor/big"
    _, scores = scored(linked(pt("code", "big", "high", 70)))
    keys, ambiguous = served_keys(scores, ids)
    assert ambiguous == set() and all("big" in k for k in keys.values())


# ---------------------------------------------------------------- the caller's shortlist
def request(**extra):
    return ChatCompletionRequest.model_validate({"model": "auto", "messages": [{"role": "user", "content": "hi"}],
                                                 **extra})


def test_models_and_exclude_are_globs_over_tier_names_and_model_ids():
    r = request(models=["vendor/*", "cheap"], exclude=["*:free"])
    assert r.allows("api:vendor/big", "vendor/big") and r.allows("cheap", "llama")
    assert not r.allows("api:vendor/big:free", "vendor/big:free") and not r.allows("top", "other")
    assert request().allows("anything", "at-all")
    assert request(models=["VENDOR/*"]).allows("x", "vendor/big")  # case does not matter


def test_the_shortlist_is_the_routers_and_never_forwarded():
    body = request(models=["a"], exclude=["b"], temperature=0.2).forwardable()
    assert "models" not in body and "exclude" not in body and body["temperature"] == 0.2


def test_tiers_the_request_left_out_are_counted_not_rejected():
    config = parse_config(caps_raw_config())
    result = evaluate(config, request(models=["cheap", "mid"]))
    assert set(result.eligible) <= {"cheap", "mid"} and result.excluded == len(config.tiers) - 2
    assert not any(r.tier not in {"cheap", "mid"} for r in result.rejections)


def route_client(backend_factory):
    config = parse_config(caps_raw_config())
    app = create_app(config, backend_factory=backend_factory, log=RequestLog(":memory:"),
                     router=CapabilityRouter(config, ask=Ask(HARD)))
    return TestClient(app)


def test_route_chooses_only_among_what_the_request_allows(backend_factory):
    with route_client(backend_factory) as client:
        free = client.post("/v1/route", json={"task": "Fix the race"}).json()
        pinned = client.post("/v1/route", json={"task": "Fix the race", "models": ["cx"]}).json()
        barred = client.post("/v1/route", json={"task": "Fix the race", "exclude": [free["tier"]]}).json()
    assert pinned["tier"] == "cx"
    assert barred["tier"] != free["tier"]


def test_route_says_when_the_shortlist_left_nothing(backend_factory):
    with route_client(backend_factory) as client:
        r = client.post("/v1/route", json={"task": "Fix it", "models": ["nobody/*"]})
        bad = client.post("/v1/route", json={"task": "Fix it", "models": "cx"})
    assert r.status_code == 422 and "models/exclude left out" in r.json()["error"]["message"]
    assert bad.status_code == 400 and "list of globs" in bad.json()["error"]["message"]


def test_the_mcp_route_tool_offers_the_shortlist():
    props = next(t for t in TOOLS if t["name"] == "route")["inputSchema"]["properties"]
    assert props["models"]["type"] == props["exclude"]["type"] == "array"
