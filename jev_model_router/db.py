"""The SQLite request log, with explicit versioned migrations.

Two rules shape this module.

1. A broken log must never fail a request. Every write is wrapped, and a write
   that fails is itself recorded (in `log_failures`, or failing that, on stderr
   and in an in-process counter). Losing observability is bad; losing the user's
   request because observability broke is worse.

2. Counterfactual costs live in their own table rather than in columns of
   `requests`. Adding a tier is a config edit, and a config edit must not
   require a schema migration -- one column per tier would guarantee the
   opposite.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .eligibility import Rejection
from .pricing import Counterfactual
from .schemas import Usage
from .verification import VerificationOutcome

SCHEMA_VERSION = 8

# Each entry is one forward migration, applied in order. Never edit a migration
# that has shipped; append a new one. The list index + 1 is its version.
_MIGRATIONS: list[str] = [
    # --- v1 -----------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS requests (
        id                     INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id             TEXT    NOT NULL,
        ts                     TEXT    NOT NULL,
        requested_model        TEXT,
        tier                   TEXT,
        backend                TEXT,
        model                  TEXT,
        router                 TEXT,
        stream                 INTEGER NOT NULL DEFAULT 0,
        prompt_sha256          TEXT    NOT NULL,
        prompt_text            TEXT,
        input_tokens           INTEGER NOT NULL DEFAULT 0,
        output_tokens          INTEGER NOT NULL DEFAULT 0,
        cached_tokens          INTEGER NOT NULL DEFAULT 0,
        cache_write_tokens     INTEGER NOT NULL DEFAULT 0,
        cost_usd               REAL    NOT NULL DEFAULT 0.0,
        latency_ms             INTEGER NOT NULL DEFAULT 0,
        http_status            INTEGER NOT NULL DEFAULT 0,
        error                  TEXT,
        eligibility_rejections TEXT    NOT NULL DEFAULT '[]'
    );

    CREATE INDEX IF NOT EXISTS idx_requests_ts   ON requests(ts);
    CREATE INDEX IF NOT EXISTS idx_requests_tier ON requests(tier);

    CREATE TABLE IF NOT EXISTS counterfactuals (
        request_row_id INTEGER NOT NULL REFERENCES requests(id) ON DELETE CASCADE,
        tier           TEXT    NOT NULL,
        cost_usd       REAL    NOT NULL,
        priced         INTEGER NOT NULL,
        eligible       INTEGER NOT NULL,
        PRIMARY KEY (request_row_id, tier)
    );

    CREATE TABLE IF NOT EXISTS log_failures (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ts         TEXT NOT NULL,
        request_id TEXT,
        error      TEXT NOT NULL
    );
    """,
    # --- v2: the verification loop ------------------------------------------
    #
    # `cost_usd` keeps its meaning -- the whole bill for the client's request --
    # and gains `route_cost_usd`, the first call alone. Both are needed and
    # neither can be derived from the other once a request can pay for two more
    # calls: savings must be measured against the total, while "what did tier X
    # cost us" must not be inflated by a verifier tier X never chose.
    #
    # The backfill is what keeps v1 rows readable: they had exactly one call, so
    # their route cost IS their total and the tier that answered IS the tier
    # that was routed to. Leaving the new columns at 0/NULL would make every
    # pre-existing request look like a free route served by nobody.
    """
    ALTER TABLE requests ADD COLUMN route_cost_usd REAL NOT NULL DEFAULT 0.0;
    ALTER TABLE requests ADD COLUMN final_tier TEXT;
    UPDATE requests SET route_cost_usd = cost_usd, final_tier = tier;

    CREATE TABLE IF NOT EXISTS verifications (
        request_row_id           INTEGER NOT NULL PRIMARY KEY
                                 REFERENCES requests(id) ON DELETE CASCADE,
        verdict                  TEXT    NOT NULL,
        reason                   TEXT,
        verifier_tier            TEXT,
        verifier_input_tokens    INTEGER NOT NULL DEFAULT 0,
        verifier_output_tokens   INTEGER NOT NULL DEFAULT 0,
        verifier_cost_usd        REAL    NOT NULL DEFAULT 0.0,
        unparseable              INTEGER NOT NULL DEFAULT 0,
        escalated                INTEGER NOT NULL DEFAULT 0,
        escalated_to             TEXT,
        escalation_input_tokens  INTEGER NOT NULL DEFAULT 0,
        escalation_output_tokens INTEGER NOT NULL DEFAULT 0,
        escalation_cost_usd      REAL    NOT NULL DEFAULT 0.0,
        latency_ms               INTEGER NOT NULL DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_verifications_verdict ON verifications(verdict);
    """,
    # --- v3: what the router decided, and on what evidence --------------------
    #
    # Note the backfill, or rather its absence. v2 backfilled because v1 rows
    # really did have a route cost and a final tier; the old columns were the
    # answer under a different name. Here they were not: no classifier scored
    # those requests, so NULL is the fact and any number written into it would
    # be an invention. `route_model` carries the fingerprint of the weights that
    # produced the score, because a score column spanning a retraining is two
    # models' numbers in one histogram and no way to separate them afterwards.
    """
    ALTER TABLE requests ADD COLUMN route_score  REAL;
    ALTER TABLE requests ADD COLUMN route_model  TEXT;
    ALTER TABLE requests ADD COLUMN route_reason TEXT;

    CREATE INDEX IF NOT EXISTS idx_requests_route_model ON requests(route_model);
    """,
    # --- v4: what the provider says it charged ------------------------------
    #
    # `cost_usd` is an ESTIMATE: price table times reported tokens. These are
    # the provider's own figure, kept beside it and never merged into it, so the
    # gap is visible. No backfill: nobody asked the provider about old rows,
    # and NULL is the fact. `upstream_id` is the provider's id for the routed
    # call, which is what `reconcile` needs to ask after the fact -- the only
    # way to cost a stream the client abandoned before its usage frame arrived.
    """
    ALTER TABLE requests ADD COLUMN upstream_id     TEXT;
    ALTER TABLE requests ADD COLUMN billed_cost_usd REAL;
    ALTER TABLE requests ADD COLUMN billed_source   TEXT;
    """,
    # --- v5: decisions for a caller that runs the model itself ---------------
    #
    # Their own table, not rows in `requests`: nothing was called and nothing
    # was billed here, and a row of zeros among real calls would read as a free
    # request in every cost report. The outcome arrives later, from the caller's
    # gate, and is the label calibration learns from.
    """
    CREATE TABLE IF NOT EXISTS route_decisions (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        decision_id           TEXT    NOT NULL UNIQUE,
        ts                    TEXT    NOT NULL,
        stage                 TEXT,
        tier                  TEXT    NOT NULL,
        model                 TEXT,
        effort                TEXT,
        runner                TEXT,
        route_score           REAL,
        route_model           TEXT,
        route_reason          TEXT,
        task_sha256           TEXT    NOT NULL,
        task_text             TEXT,
        outcome               TEXT,
        outcome_ts            TEXT,
        outcome_detail        TEXT,
        outcome_input_tokens  INTEGER,
        outcome_output_tokens INTEGER
    );

    CREATE INDEX IF NOT EXISTS idx_route_decisions_ts ON route_decisions(ts);
    """,
    # v6: the cache tokens a routed task's caller reports, the evidence its
    # tier's task shape is observed from (see calls.py).
    """
    ALTER TABLE route_decisions ADD COLUMN outcome_cached_tokens      INTEGER;
    ALTER TABLE route_decisions ADD COLUMN outcome_cache_write_tokens INTEGER;
    """,
    # v7: a route-only request refused because Jev did not answer and
    # `on_jev_failure: reject` says so. Not a decision -- no tier, no outcome --
    # so not a `route_decisions` row, which stats and calibration read as one.
    """
    CREATE TABLE IF NOT EXISTS route_rejections (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ts           TEXT NOT NULL,
        request_id   TEXT NOT NULL,
        stage        TEXT,
        route_model  TEXT,
        route_reason TEXT NOT NULL,
        task_sha256  TEXT NOT NULL,
        task_text    TEXT
    );
    """,
    # v8: what was written to a route decision, by whom, in order. A decision's usage and outcome
    # are UPDATEs to a row the app inserted itself, so a row id cannot tell another process sharing
    # this log (a `jev-model-router delegate` run) what changed. This table can: the app reads the
    # other writers' events before each decision (route_sync.py).
    """
    CREATE TABLE IF NOT EXISTS route_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        decision_id TEXT    NOT NULL,
        kind        TEXT    NOT NULL,
        writer      TEXT,
        ts          TEXT    NOT NULL
    );
    """,
]


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _token_values(usage: Usage | None) -> tuple[int | None, int | None, int | None, int | None]:
    if usage is None:
        return (None, None, None, None)
    return (usage.prompt_tokens, usage.completion_tokens, usage.cached_tokens, usage.cache_write_tokens)


@dataclass
class LogEntry:
    """One request, as it will be written."""

    request_id: str
    prompt_text: str
    requested_model: str | None = None
    tier: str | None = None
    backend: str | None = None
    model: str | None = None
    router: str | None = None
    stream: bool = False
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    latency_ms: int = 0
    http_status: int = 0
    error: str | None = None
    rejections: list[Rejection] = field(default_factory=list)
    counterfactuals: list[Counterfactual] = field(default_factory=list)
    # The tier whose answer the client actually received. Equal to `tier` unless
    # a failed verdict escalated the request.
    final_tier: str | None = None
    verification: VerificationOutcome | None = None
    # What the router decided and why. `route_score` stays None unless a model
    # actually scored this request -- an explicit `model:` from the client, or a
    # cheap tier the gate had already removed, leaves nothing to record and
    # records nothing.
    route_score: float | None = None
    route_model: str | None = None
    route_reason: str | None = None
    # The provider's id for the routed call, and the provider's own total for
    # every call this request made (route, review, escalation). None when any
    # one of those calls went unreported: a partial bill next to a full
    # estimate would show a saving that is only a missing row.
    upstream_id: str | None = None
    billed_cost_usd: float | None = None
    billed_source: str | None = None
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def route_cost_usd(self) -> float:
        """What the routed tier's own call cost, with the loop's calls removed."""
        extra = self.verification.extra_cost_usd if self.verification else 0.0
        return self.cost_usd - extra


class RequestLog:
    """Owns the SQLite connection and the write path."""

    def __init__(self, path: str | Path, *, store_prompts: bool = False) -> None:
        self.path = str(path)
        self.store_prompts = store_prompts
        self.failed_writes = 0
        # One connection guarded by a lock. Writes are single-row and sub-
        # millisecond; a pool would buy nothing, and WAL already lets readers
        # (the stats command) work while a request is being written.
        self._lock = threading.Lock()
        if self.path != ":memory:":
            parent = Path(self.path).parent
            if str(parent):
                parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._configure()
        self.migrate()

    def _configure(self) -> None:
        # WAL: a reader never blocks the request path's writer.
        # An in-memory database has no journal file, so WAL is skipped there.
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def migrate(self) -> None:
        with self._lock:
            current = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            for version in range(current, len(_MIGRATIONS)):
                self._conn.executescript(_MIGRATIONS[version])
                # user_version is SQLite's own slot for exactly this: it needs no
                # table of its own and cannot drift from the file it describes.
                self._conn.execute(f"PRAGMA user_version = {version + 1}")
            self._conn.commit()

    @property
    def schema_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def record(self, entry: LogEntry) -> int | None:
        """Write one request row. Returns its id, or None if the write failed.

        Never raises: the caller is on the request path.
        """
        try:
            return self._write(entry)
        except Exception as exc:  # noqa: BLE001 - swallowing it is the requirement
            self.failed_writes += 1
            self._record_failure(entry.request_id, exc)
            return None

    async def record_async(self, entry: LogEntry) -> int | None:
        """Write off the event loop, so a slow disk does not stall the server."""
        return await asyncio.to_thread(self.record, entry)

    def _write(self, entry: LogEntry) -> int:
        rejections = json.dumps([r.as_dict() for r in entry.rejections])
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO requests (
                    request_id, ts, requested_model, tier, backend, model, router,
                    stream, prompt_sha256, prompt_text, input_tokens, output_tokens,
                    cached_tokens, cache_write_tokens, cost_usd, latency_ms,
                    http_status, error, eligibility_rejections, route_cost_usd, final_tier,
                    route_score, route_model, route_reason,
                    upstream_id, billed_cost_usd, billed_source
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    entry.request_id,
                    entry.ts.isoformat(),
                    entry.requested_model,
                    entry.tier,
                    entry.backend,
                    entry.model,
                    entry.router,
                    int(entry.stream),
                    sha256_hex(entry.prompt_text),
                    # The hash always goes in; the text only when the operator
                    # opted in, because prompts carry user data.
                    entry.prompt_text if self.store_prompts else None,
                    entry.usage.prompt_tokens,
                    entry.usage.completion_tokens,
                    entry.usage.cached_tokens,
                    entry.usage.cache_write_tokens,
                    entry.cost_usd,
                    entry.latency_ms,
                    entry.http_status,
                    entry.error,
                    rejections,
                    entry.route_cost_usd,
                    entry.final_tier or entry.tier,
                    entry.route_score,
                    entry.route_model,
                    entry.route_reason,
                    entry.upstream_id,
                    entry.billed_cost_usd,
                    entry.billed_source,
                ),
            )
            row_id = int(cur.lastrowid)
            if entry.counterfactuals:
                self._conn.executemany(
                    """
                    INSERT INTO counterfactuals (request_row_id, tier, cost_usd, priced, eligible)
                    VALUES (?,?,?,?,?)
                    """,
                    [
                        (row_id, cf.tier, cf.cost_usd, int(cf.priced), int(cf.eligible))
                        for cf in entry.counterfactuals
                    ],
                )
            if entry.verification is not None:
                v = entry.verification
                self._conn.execute(
                    """
                    INSERT INTO verifications (
                        request_row_id, verdict, reason, verifier_tier,
                        verifier_input_tokens, verifier_output_tokens, verifier_cost_usd,
                        unparseable, escalated, escalated_to,
                        escalation_input_tokens, escalation_output_tokens,
                        escalation_cost_usd, latency_ms
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        row_id,
                        v.verdict.value,
                        v.reason,
                        v.verifier_tier,
                        v.verifier_usage.prompt_tokens,
                        v.verifier_usage.completion_tokens,
                        v.verifier_cost_usd,
                        int(v.unparseable),
                        int(v.escalated),
                        v.escalated_to,
                        v.escalation_usage.prompt_tokens,
                        v.escalation_usage.completion_tokens,
                        v.escalation_cost_usd,
                        v.latency_ms,
                    ),
                )
            self._conn.commit()
            return row_id

    def _record_failure(self, request_id: str | None, exc: Exception) -> None:
        """Record that a log write failed. Best effort by definition."""
        message = f"{type(exc).__name__}: {exc}"
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO log_failures (ts, request_id, error) VALUES (?,?,?)",
                    (datetime.now(timezone.utc).isoformat(), request_id, message),
                )
                self._conn.commit()
        except Exception:  # noqa: BLE001 - the log itself is what broke
            # Last resort. The in-process counter stays accurate and /healthz
            # reports it, so a silently broken log is still visible somewhere.
            print(
                f"jev-model-router: log write failed and could not be recorded: {message}",
                file=sys.stderr,
            )

    def record_label(self, request_id: str, *, verdict: str, reason: str, source: str) -> None:
        """Attach a verdict that came from an answer key, not from a verifier.

        `source` goes in `verifier_tier` and must not name a tier: it is how a
        reader tells these labels from a judge's. No tokens and no cost -- no
        call was made to produce them.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM requests WHERE request_id = ?", (request_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"no request row for {request_id}")
            self._conn.execute(
                "INSERT INTO verifications (request_row_id, verdict, reason, verifier_tier) "
                "VALUES (?,?,?,?)",
                (row["id"], verdict, reason, source),
            )
            self._conn.commit()

    def apply_billing(
        self,
        row_id: int,
        *,
        billed_usd: float,
        served_tier: str,
        usage: Usage | None = None,
        costs: dict[str, float] | None = None,
    ) -> None:
        """Write a provider's after-the-fact bill onto an existing row.

        `usage` and `costs` are given only when the row's own usage never
        arrived; the estimate and every counterfactual are then recomputed from
        the provider's tokens, because a row costed from zero tokens is zero at
        every tier and flatters none of them honestly.
        """
        with self._lock:
            self._conn.execute(
                "UPDATE requests SET billed_cost_usd = ?, billed_source = 'reconciled' "
                "WHERE id = ?",
                (billed_usd, row_id),
            )
            if usage is not None and costs is not None:
                served = costs.get(served_tier, 0.0)
                self._conn.execute(
                    "UPDATE requests SET input_tokens = ?, output_tokens = ?, "
                    "cached_tokens = ?, cost_usd = ?, route_cost_usd = ? WHERE id = ?",
                    (
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        usage.cached_tokens,
                        served,
                        served,
                        row_id,
                    ),
                )
                self._conn.executemany(
                    "UPDATE counterfactuals SET cost_usd = ? "
                    "WHERE request_row_id = ? AND tier = ?",
                    [(cost, row_id, tier) for tier, cost in costs.items()],
                )
            self._conn.commit()

    def record_decision(
        self,
        decision_id: str,
        *,
        task: str,
        tier: str,
        model: str | None,
        effort: str | None,
        runner: str | None,
        stage: str | None,
        route_score: float | None,
        route_model: str | None,
        route_reason: str | None,
    ) -> bool:
        """Write one route-only decision. Never raises: the caller is waiting on the answer."""
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO route_decisions (decision_id, ts, stage, tier, model, effort, runner, "
                    "route_score, route_model, route_reason, task_sha256, task_text) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        decision_id, datetime.now(timezone.utc).isoformat(), stage, tier, model, effort, runner,
                        route_score, route_model, route_reason, sha256_hex(task),
                        task if self.store_prompts else None,
                    ),
                )
                self._conn.commit()
            return True
        except Exception as exc:  # noqa: BLE001 - losing the row must not lose the answer
            self.failed_writes += 1
            self._record_failure(decision_id, exc)
            return False

    def record_rejection(
        self, request_id: str, *, task: str, stage: str | None, route_model: str | None, route_reason: str
    ) -> bool:
        """Write one refused route-only request. Never raises: the caller is waiting on the answer."""
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO route_rejections (ts, request_id, stage, route_model, route_reason, "
                    "task_sha256, task_text) VALUES (?,?,?,?,?,?,?)",
                    (
                        datetime.now(timezone.utc).isoformat(), request_id, stage, route_model, route_reason,
                        sha256_hex(task), task if self.store_prompts else None,
                    ),
                )
                self._conn.commit()
            return True
        except Exception as exc:  # noqa: BLE001 - losing the row must not lose the answer
            self.failed_writes += 1
            self._record_failure(request_id, exc)
            return False

    def decision(self, decision_id: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM route_decisions WHERE decision_id = ?", (decision_id,)
            ).fetchone()

    _TOKENS = ("outcome_input_tokens", "outcome_output_tokens", "outcome_cached_tokens", "outcome_cache_write_tokens")

    def set_outcome(
        self, decision_id: str, status: str, detail: str | None, usage: Usage | None, writer: str | None = None,
    ) -> str:
        """`ok`, `kept_existing`, `missing` or `exists` -- the first outcome reported is the one kept.

        Tokens already on the row (a delegated run's, which read them exactly) are kept too, as a
        block: the four columns take `usage` only when the row has none, never a mix of two sources.
        `kept_existing` means the outcome was written and `usage` was not.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if self._conn.in_transaction:
                # A write that failed half-way (and whose failure record failed too) left one open.
                self._conn.rollback()
            # IMMEDIATE: another process may write this row between the read and the update.
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT outcome, outcome_input_tokens FROM route_decisions WHERE decision_id = ?", (decision_id,)
                ).fetchone()
                if row is None or row["outcome"] is not None:
                    self._conn.rollback()
                    return "missing" if row is None else "exists"
                writes_usage = usage is not None and row["outcome_input_tokens"] is None
                values = _token_values(usage)
                self._conn.execute(
                    "UPDATE route_decisions SET outcome = ?, outcome_ts = ?, outcome_detail = ?, "
                    + ", ".join(f"{c} = CASE WHEN outcome_input_tokens IS NULL THEN ? ELSE {c} END" for c in self._TOKENS)
                    + " WHERE decision_id = ?",
                    (status, now, detail, *values, decision_id),
                )
                self._event(decision_id, "outcome", writer, now)
                if writes_usage:
                    self._event(decision_id, "usage", writer, now)
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return "kept_existing" if usage is not None and not writes_usage else "ok"

    def set_usage(self, decision_id: str, usage: Usage, writer: str | None = None) -> str:
        """`ok`, `kept_existing` (the row already has tokens: the first are kept) or `missing`.

        Written whether or not an outcome exists yet: the agent may report its verdict before a
        delegated run left in the background has finished. The `outcome_*_tokens` columns keep
        their names; they now also hold usage that arrived before, or without, an outcome.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE route_decisions SET " + ", ".join(f"{c} = ?" for c in self._TOKENS)
                + " WHERE decision_id = ? AND outcome_input_tokens IS NULL",
                (*_token_values(usage), decision_id),
            )
            if cursor.rowcount:
                self._event(decision_id, "usage", writer, now)
            self._conn.commit()
            if cursor.rowcount:
                return "ok"
            found = self._conn.execute(
                "SELECT 1 FROM route_decisions WHERE decision_id = ?", (decision_id,)
            ).fetchone()
            return "kept_existing" if found else "missing"

    def _event(self, decision_id: str, kind: str, writer: str | None, ts: str) -> None:
        # Called inside the caller's transaction, with the lock held.
        self._conn.execute(
            "INSERT INTO route_events (decision_id, kind, writer, ts) VALUES (?,?,?,?)",
            (decision_id, kind, writer, ts),
        )

    def last_event_id(self) -> int:
        return int(self.query("SELECT COALESCE(MAX(id), 0) AS n FROM route_events")[0]["n"])

    def events_after(self, last_id: int, writer: str) -> list[sqlite3.Row]:
        """Other writers' route events after `last_id`, oldest first, with their decision's tier, outcome and tokens."""
        return self.query(
            "SELECT e.id, e.kind, e.ts, e.decision_id, d.tier, d.outcome, "
            "d.outcome_input_tokens AS input, d.outcome_output_tokens AS output, "
            "COALESCE(d.outcome_cached_tokens, 0) AS cached, COALESCE(d.outcome_cache_write_tokens, 0) AS written "
            "FROM route_events e JOIN route_decisions d ON d.decision_id = e.decision_id "
            "WHERE e.id > ? AND (e.writer IS NULL OR e.writer != ?) ORDER BY e.id",
            (last_id, writer),
        )

    def task_usage(self) -> list[sqlite3.Row]:
        """Route decisions with usage, with or without an outcome, oldest first; record_task decides which count."""
        return self.query(
            "SELECT tier, outcome_input_tokens AS input, outcome_output_tokens AS output, "
            "COALESCE(outcome_cached_tokens, 0) AS cached, COALESCE(outcome_cache_write_tokens, 0) AS written "
            "FROM route_decisions "
            "WHERE outcome_input_tokens IS NOT NULL AND outcome_output_tokens IS NOT NULL ORDER BY id"
        )

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, tuple(params)).fetchall())

    def close(self) -> None:
        with self._lock:
            self._conn.close()
