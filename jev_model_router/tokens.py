"""Token estimation for the eligibility gate.

This is a heuristic, not a tokenizer. It exists only to answer "will this
request plausibly fit in that backend's context window", and it runs before any
backend is contacted, so a real tokenizer for the eventual target is not
available yet: the target has not been chosen.

The estimate deliberately errs HIGH. An over-estimate costs a request the
cheapest tier; an under-estimate sends a request to a backend that rejects it,
which is the failure this gate exists to prevent.

Replace `estimate_tokens` with a tokenizer-backed implementation if you need
tighter packing; the rest of the package only calls this function.
"""

from __future__ import annotations

import json
from typing import Any

from .schemas import ChatCompletionRequest

# English text runs near 4 characters per token; 3.5 keeps the estimate on the
# high side for code and non-Latin scripts, which tokenize worse.
_CHARS_PER_TOKEN = 3.5

# Every message carries role and delimiter tokens the text itself does not show.
_PER_MESSAGE_OVERHEAD = 4


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Multimodal parts, tool payloads and anything else: serialize and count it.
    # Crude for images, but an image part is small as JSON and large as tokens,
    # so this under-counts there. Noted rather than hidden.
    return json.dumps(content, ensure_ascii=False)


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def estimate_prompt_tokens(request: ChatCompletionRequest) -> int:
    """Estimated input tokens: messages plus any tool definitions."""
    total = 0
    for message in request.messages:
        total += _PER_MESSAGE_OVERHEAD
        total += estimate_text_tokens(str(message.role))
        total += estimate_text_tokens(_text_of(message.content))
    if request.tools:
        # Tool schemas are part of the prompt and are routinely the largest part
        # of an agentic request.
        total += estimate_text_tokens(json.dumps(request.tools, ensure_ascii=False))
    return total


def estimate_request_budget(request: ChatCompletionRequest, *, default_output: int = 1024) -> int:
    """Input tokens plus the output the caller reserved.

    A context window covers both, so a request that fits on input alone can
    still overflow once the model answers. When the caller declares no cap we
    reserve `default_output`, because assuming zero output would let every
    borderline request through.
    """
    return estimate_prompt_tokens(request) + (request.output_budget or default_output)
