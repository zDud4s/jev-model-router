"""One short real call per named tier, straight to its backend (no router).

    python scripts/cli_smoke.py config.capabilities.yaml c-haiku c-sonnet-low x-luna56-low

Spends subscription quota: one tiny prompt per tier.
"""

from __future__ import annotations

import asyncio
import sys
import time

from jev_model_router.backends import build_backend
from jev_model_router.backends.base import BackendError
from jev_model_router.config import load_config
from jev_model_router.schemas import ChatCompletionRequest

PROMPT = "Reply with exactly the word: pong"


async def main() -> None:
    config = load_config(sys.argv[1])
    for name in sys.argv[2:]:
        tier = config.tier(name)
        backend = build_backend(tier)
        request = ChatCompletionRequest.model_validate(
            {"model": name, "messages": [{"role": "user", "content": PROMPT}]}
        )
        started = time.perf_counter()
        try:
            result = await backend.complete(request)
            text = result.body["choices"][0]["message"]["content"].strip()
            u = result.usage
            print(f"{name:14s} OK  {time.perf_counter() - started:5.1f}s  {text[:40]!r}  "
                  f"in={u.prompt_tokens} cached={u.cached_tokens} out={u.completion_tokens} billed={u.billed_usd}")
        except BackendError as exc:
            print(f"{name:14s} ERR {time.perf_counter() - started:5.1f}s  status={exc.status} {exc}  body={str(exc.body)[:300]!r}")
        finally:
            await backend.aclose()


if __name__ == "__main__":
    asyncio.run(main())
