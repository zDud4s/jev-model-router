"""Benchmark evidence in discovery, at startup and on the command line."""

from __future__ import annotations

import pytest

from llm_router.catalog import CatalogReport, Offered, Source
from llm_router.config import parse_config
from llm_router.discovery import expand, model_ids, served_ids
from llm_router.scores_derive import served_keys, startup_lines

from test_scores import BENCHES, linked, pt, raw_config, scored


def catalog(**models: Offered) -> CatalogReport:
    source = Source(name="cli", models=dict(models), cli_version="9.9.9")
    return CatalogReport(checked_at="now", tiers={}, sources={}, unconfigured={}, discovered={"cli": source})


REPORT = catalog(**{
    "acme-large": Offered("acme-large", efforts=("medium", "high"), aliases=("acme-large-latest",)),
    "acme-small": Offered("acme-small"),
    "unknown-1": Offered("unknown-1", efforts=("high",)),
})
POINTS = linked(pt("code", "acme-large-latest", "high", 80), pt("code", "acme-small", None, 40),
                pt("code", "zeta-1", "high", 60))


def test_expand_without_scores_builds_todays_cards():
    expanded, _, _ = expand(parse_config(raw_config()), REPORT)
    assert expanded.router.capabilities.cards["cli:acme-large@medium"].levels == {
        "reasoning": pytest.approx(1.7), "niche": 2.0}


def test_expand_with_scores_derives_the_discovered_cards_through_their_aliases():
    _, scores = scored(POINTS)
    expanded, _, _ = expand(parse_config(raw_config()), REPORT, scores)
    cards = expanded.router.capabilities.cards
    assert cards["cli:acme-large@high"].levels["reasoning"] > cards["cli:acme-small"].levels["reasoning"]
    assert cards["cli:acme-large@high"].family == "acme-large"
    assert cards["cli:unknown-1@high"].levels == {"reasoning": 0.5, "niche": 0.5}  # no evidence: the profile


def test_an_explicit_tier_without_a_card_is_derived_too():
    raw = raw_config()
    raw["tiers"]["large"] = {"backend": "claude_cli", "model": "acme-large", "effort": "high",
                             "base_url": "C:/bin/cli.exe"}
    _, scores = scored(POINTS)
    expanded, _, _ = expand(parse_config(raw), REPORT, scores)
    assert expanded.router.capabilities.cards["large"].levels["reasoning"] > 2.0


def test_one_model_at_several_efforts_is_not_ambiguous_but_two_models_on_one_key_are():
    _, scores = scored(POINTS)
    keys, ambiguous = served_keys(scores, {"cli:a@low": ("acme-large",), "cli:a@high": ("acme-large",),
                                           "cli:b": ("acme_large",), "cli:c": ("zeta-1",)})
    assert ambiguous == {"acme-large"} and keys["cli:a@high"] == () and keys["cli:c"] == ("zeta-1",)


def test_model_ids_and_served_ids_name_every_id_a_point_may_use():
    _, found, _ = expand(parse_config(raw_config()), REPORT)
    assert model_ids(REPORT)["acme-large"] == ("acme-large", "acme-large-latest")
    assert served_ids(found, REPORT)["cli:acme-large"] == ("acme-large", "acme-large-latest")


def test_startup_lines_name_bare_and_vendor_only_models_and_unread_benchmarks():
    benches = {**BENCHES, "fresh": {"description": "never read"}}
    caps, scores = scored(linked(pt("code", "acme-large", "high", 80), pt("code", "zeta-1", "high", 60,
                                                                           origin="independent")), benches)
    served = {"cli:acme-large": ("acme-large",), "cli:acme-small": ("acme-small",), "cli:zeta-1": ("zeta-1",)}
    lines = startup_lines(scores, caps, served)
    assert "benchmarks: 1 served model(s) with no evidence: cli:acme-small" in lines
    assert not any("vendor evidence only" in line for line in lines)  # linked() adds independent base points
    assert any(line.startswith("benchmarks: unread") and "fresh" in line for line in lines)
    ambiguous = {**served, "cli:acme_large": ("acme_large",)}
    assert "benchmarks: ambiguous model key(s), used for no tier: acme-large" in         startup_lines(scores, caps, ambiguous)
    caps, vendor = scored([pt("code", "acme-large", "high", 80), pt("code", "zeta-1", "high", 60)])
    assert "benchmarks: 2 served model(s) with vendor evidence only: cli:acme-large, cli:zeta-1" in \
        startup_lines(vendor, caps, served)
