"""The Ollama adapter's request body.

Found on the first run against a real model: the config said
`context_window: 32768`, the eligibility gate reasoned about 32768, and Ollama
ran the model at 4096 -- its VRAM-based default -- because nothing ever told it
otherwise. A thinking model filled those 4096 tokens with reasoning and the
client received an empty answer. The gate's number was fiction for this backend.
"""

from __future__ import annotations

from jev_model_router.backends.ollama import OllamaBackend
from jev_model_router.config import parse_config

from conftest import BASE_CONFIG, make_request


def test_the_configured_window_is_the_window_ollama_runs_with() -> None:
    config = parse_config(BASE_CONFIG)
    body = OllamaBackend(config.tiers["cheap"])._body(make_request(), stream=False)
    assert body["options"]["num_ctx"] == config.tiers["cheap"].context_window


def test_extra_body_options_are_merged_rather_than_replacing_the_window() -> None:
    # `extra_body` is a top-level update. Given an `options` key of its own it
    # used to replace the whole options dict, taking num_ctx with it.
    raw = {**BASE_CONFIG, "tiers": {**BASE_CONFIG["tiers"]}}
    raw["tiers"]["cheap"] = {**raw["tiers"]["cheap"], "extra_body": {"options": {"seed": 7}, "think": False}}
    tier = parse_config(raw).tiers["cheap"]
    body = OllamaBackend(tier)._body(make_request(max_tokens=50), stream=False)
    assert body["options"] == {"num_predict": 50, "num_ctx": tier.context_window, "seed": 7}
    assert body["think"] is False
