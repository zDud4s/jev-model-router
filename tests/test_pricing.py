"""Cost arithmetic, including the zero-price local case and the counterfactuals."""

from __future__ import annotations

from jev_model_router.config import Prices
from jev_model_router.pricing import cost_usd, counterfactuals
from jev_model_router.schemas import Usage


def test_a_local_model_with_no_prices_costs_zero(config) -> None:
    # `cheap` configures no prices at all. That must not raise, and must not
    # require the operator to write `input: 0.0`.
    tier = config.tiers["cheap"]

    assert tier.prices.configured is False
    assert cost_usd(tier.prices, Usage(prompt_tokens=10_000, completion_tokens=5_000)) == 0.0


def test_cost_is_per_million_tokens(config) -> None:
    prices = config.tiers["mid"].prices  # input 3.00, output 15.00 per 1M
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)

    assert cost_usd(prices, usage) == 18.0


def test_cached_tokens_replace_input_tokens_rather_than_adding_to_them() -> None:
    prices = Prices(input=10.0, output=0.0, cache_read=1.0, configured=True)
    usage = Usage(prompt_tokens=1_000_000, cached_tokens=900_000, completion_tokens=0)

    # 100k billed at 10.00/M plus 900k billed at 1.00/M.
    assert cost_usd(prices, usage) == round(1.0 + 0.9, 10)


def test_an_unpriced_cache_read_costs_the_input_rate_not_nothing() -> None:
    # Measured on OpenRouter: a repeated question came back with every prompt
    # token reported as cached. With cache_read defaulting to 0, a tier priced
    # only on input and output charged nothing for its input at all.
    prices = Prices.parse({"input": 10.0, "output": 0.0})
    usage = Usage(prompt_tokens=1_000_000, cached_tokens=1_000_000)

    assert cost_usd(prices, usage) == 10.0


def test_a_cache_discount_is_claimed_only_when_written_down() -> None:
    assert Prices.parse({"input": 10.0, "cache_read": 0.0}).cache_read == 0.0
    assert Prices.parse({"input": 10.0, "cache_read": 1.0}).cache_read == 1.0


def test_cache_writes_are_billed_on_top() -> None:
    prices = Prices(input=0.0, output=0.0, cache_write=10.0, configured=True)

    assert cost_usd(prices, Usage(cache_write_tokens=500_000)) == 5.0


def test_a_counterfactual_is_produced_for_every_configured_tier(config) -> None:
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)

    results = {cf.tier: cf for cf in counterfactuals(config, usage)}

    assert set(results) == {"cheap", "mid", "top"}
    assert results["cheap"].cost_usd == 0.0
    assert results["mid"].cost_usd == 18.0
    assert results["top"].cost_usd == 90.0


def test_an_unpriced_counterfactual_is_flagged_as_unpriced_not_free(config) -> None:
    results = {cf.tier: cf for cf in counterfactuals(config, Usage(prompt_tokens=1000))}

    # Zero cost and "no price configured" are different claims, and stats needs
    # to be able to tell them apart.
    assert results["cheap"].priced is False
    assert results["mid"].priced is True


def test_a_counterfactual_records_whether_that_tier_could_have_served_the_request(config) -> None:
    results = {
        cf.tier: cf
        for cf in counterfactuals(config, Usage(prompt_tokens=1000), eligible_tiers={"mid", "top"})
    }

    assert results["cheap"].eligible is False
    assert results["mid"].eligible is True
