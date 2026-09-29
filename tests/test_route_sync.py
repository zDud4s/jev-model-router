"""Usage and rate limits another process wrote to the shared log reach this app's router once."""

from __future__ import annotations

from datetime import datetime, timezone

from jev_model_router.capabilities import CapabilityRouter
from jev_model_router.config import parse_config
from jev_model_router.db import RequestLog
from jev_model_router.route_sync import EventFollower, account_usage
from jev_model_router.schemas import Usage

from test_capabilities import HARD, Ask, raw_config

CACHED = Usage(prompt_tokens=300_000, completion_tokens=1000, cached_tokens=290_000, cache_write_tokens=8000)


def router(clock=None):
    kwargs = {"clock": clock} if clock else {}
    return CapabilityRouter(parse_config(raw_config()), ask=Ask(HARD), **kwargs)


def routed(log, i, tier="sub"):
    log.record_decision(f"rt_{i}", task="t", tier=tier, model=None, effort=None, runner=None, stage=None,
                        route_score=None, route_model=None, route_reason=None)


def seen(r, tier="sub"):
    return len(r.calls._tasks.get(tier, ()))


def test_account_usage_charges_the_subscription_and_feeds_the_shape():
    r = router()
    account_usage(r, "sub", CACHED)
    assert r.ledger.used("claude") > 0 and seen(r) == 1


def test_account_usage_never_raises():
    account_usage(object(), "sub", CACHED)  # a router with neither observe nor calls
    account_usage(router(), "gone", CACHED)


def test_another_writers_usage_is_counted_once_and_its_own_never():
    log = RequestLog(":memory:")
    r = router()
    follower = EventFollower(log, "app")
    routed(log, 1)
    routed(log, 2)
    log.set_usage("rt_1", CACHED, "delegate")
    log.set_usage("rt_2", CACHED, "app")
    assert follower.catch_up(r) == 1 and seen(r) == 1
    assert follower.catch_up(r) == 0 and seen(r) == 1


def test_a_rate_limit_another_writer_saw_locks_from_when_it_was_seen():
    log = RequestLog(":memory:")
    ts = datetime.now(timezone.utc).timestamp()
    now = [ts]
    r = router(clock=lambda: now[0])
    follower = EventFollower(log, "app")
    routed(log, 1)
    log.set_outcome("rt_1", "rate_limited", None, None, "delegate")
    follower.catch_up(r)
    assert r.ledger.locked("claude")
    now[0] = ts + 5 * 3600 + 60  # the window in CAPS is 5 h, counted from the event
    assert not r.ledger.locked("claude")


def test_two_decisions_at_once_count_an_event_once():
    import threading

    log = RequestLog(":memory:")
    r = router()
    follower = EventFollower(log, "app")
    routed(log, 1)
    log.set_usage("rt_1", CACHED, "delegate")
    threads = [threading.Thread(target=follower.catch_up, args=(r,)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen(r) == 1


def test_history_before_the_follower_started_is_not_replayed():
    log = RequestLog(":memory:")
    routed(log, 1)
    log.set_usage("rt_1", CACHED, "delegate")
    log.set_outcome("rt_1", "rate_limited", None, None, "delegate")
    r = router()
    follower = EventFollower(log, "app")
    assert follower.catch_up(r) == 0
    assert r.ledger.used("claude") == 0 and not r.ledger.locked("claude")
