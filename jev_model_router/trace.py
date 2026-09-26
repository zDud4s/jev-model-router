"""Live traces of recent requests, for the routing view at /routing.

Each request becomes one trace: a list of stages (received, eligibility, route,
answer, done), each stamped with milliseconds since the request arrived. The
store keeps the last `keep` traces in memory and nothing else -- it is a window
onto what the proxy is doing now, not a log; the request log is the record.

Every change to a trace takes the next sequence number, so a page that polls
`?after=<seq>` receives exactly the traces that moved since it last asked.

What it holds is the prompt's first lines and the packet Jev read. The proxy
binds to 127.0.0.1 by default, which is the only reason that is acceptable;
`trace.enabled: false` turns the store and the page off.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

PREVIEW_CHARS = 1500


class TraceStore:
    def __init__(self, keep: int = 200) -> None:
        self._keep = keep
        self._traces: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._seq = 0

    @property
    def seq(self) -> int:
        return self._seq

    def start(self, request_id: str, *, kind: str, preview: str, requested_model: str | None) -> dict[str, Any]:
        trace = {
            "id": request_id,
            "kind": kind,  # "request" or "dry-run"
            "started_at": time.time(),
            "requested_model": requested_model,
            "preview": preview[:PREVIEW_CHARS],
            "stages": [],
            "status": "running",
            "_t0": time.perf_counter(),
        }
        self._traces[request_id] = trace
        while len(self._traces) > self._keep:
            self._traces.popitem(last=False)
        self._bump(trace)
        return trace

    def stage(self, trace: dict[str, Any], name: str, **data: Any) -> None:
        trace["stages"].append({"name": name, "t_ms": int((time.perf_counter() - trace["_t0"]) * 1000), **data})
        self._bump(trace)

    def finish(self, trace: dict[str, Any], status: str) -> None:
        trace["status"] = status
        trace["total_ms"] = int((time.perf_counter() - trace["_t0"]) * 1000)
        self._bump(trace)

    def since(self, after: int) -> list[dict[str, Any]]:
        return [self._public(t) for t in self._traces.values() if t["seq"] > after]

    def _bump(self, trace: dict[str, Any]) -> None:
        self._seq += 1
        trace["seq"] = self._seq

    @staticmethod
    def _public(trace: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in trace.items() if not k.startswith("_")}


def prompt_preview(messages: list[Any]) -> str:
    """The last user message, which is what a person recognises a request by."""
    for message in reversed(messages):
        if getattr(message, "role", None) == "user":
            content = message.content
            if isinstance(content, list):
                content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
            return str(content or "")
    return ""


__all__ = ["TraceStore", "prompt_preview"]
