"""A routed task's usage and rate limits, counted once across the processes that share a log.

`jev-model-router delegate` may run in another process than the app that routed the task, and
writes the run's tokens (and a rate limit) straight to the log. The app counts what it writes itself
when it writes it (`account_usage`); what other writers did, it reads from `route_events` before each
decision (`EventFollower.catch_up`). Startup reads history only for task shapes (app.py), so the
follower starts at the newest event: old spend is not charged again, old locks not replayed.
"""

from __future__ import annotations

import sys
import threading
from datetime import datetime
from typing import Any

from .db import RequestLog
from .schemas import Usage


def account_usage(router: Any, tier: str, usage: Usage) -> None:
    """One routed task's whole usage: spend on its subscription, and evidence of its tier's shape."""
    observe = getattr(router, "observe", None)
    if observe is not None:
        try:
            observe(tier, usage, 200)
        except Exception:  # noqa: BLE001 - accounting must not fail a report
            pass
    calls = getattr(router, "calls", None)
    if calls is not None:
        try:
            calls.record_task(tier, usage)
        except Exception:  # noqa: BLE001 - a shape must not fail a report
            pass


class EventFollower:
    """Reads the route events other writers appended since it last looked."""

    def __init__(self, log: RequestLog, writer: str) -> None:
        self._log = log
        self.writer = writer
        # Two decisions at once (the MCP server answers each message as its own task) must not both
        # read the same new events before either moves `last`: that would count them twice.
        self._lock = threading.Lock()
        try:
            self.last = log.last_event_id()
        except Exception:  # noqa: BLE001 - a log it cannot read leaves nothing to catch up on
            self.last = 0

    def catch_up(self, router: Any) -> int:
        """Apply the new events to `router`; how many there were. Never raises."""
        with self._lock:
            return self._catch_up(router)

    def _catch_up(self, router: Any) -> int:
        try:
            rows = self._log.events_after(self.last, self.writer)
        except Exception as exc:  # noqa: BLE001 - a decision must not fail on the log
            print(f"route events: cannot read them: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 0
        for row in rows:
            self.last = row["id"]
            if row["kind"] == "usage" and row["input"] is not None and row["output"] is not None:
                account_usage(router, row["tier"], Usage(
                    prompt_tokens=row["input"], completion_tokens=row["output"],
                    cached_tokens=row["cached"], cache_write_tokens=row["written"],
                ))
            elif row["kind"] == "outcome" and row["outcome"] == "rate_limited":
                calls = getattr(router, "calls", None)
                if calls is None:
                    continue
                try:
                    calls.lock(row["tier"], at=datetime.fromisoformat(row["ts"]).timestamp())
                except Exception:  # noqa: BLE001
                    pass
        return len(rows)


__all__ = ["EventFollower", "account_usage"]
