"""`benchmarks read`: Jev reads what each benchmark measures, the way it reads a task.

A benchmark's weights on the requirements are, by default, Jev's answers to the
router's own questions about the benchmark's description (or its example
tasks, averaged). One call per benchmark, repeated only when its words or the
questions change: the sidecar keeps a hash of both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .capabilities import Ask, build_packet
from .config import CapabilitiesConfig, Config
from .schemas import ChatCompletionRequest
from .scores import Scores, content_hash, write_sidecar

MAX_EXAMPLES = 10


@dataclass
class ReadReport:
    asked: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    manual: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


def _packet(text: str, caps: CapabilitiesConfig) -> str:
    request = ChatCompletionRequest.model_validate({"model": "auto", "messages": [{"role": "user", "content": text}]})
    return build_packet(request, max_chars=caps.max_packet_chars)


async def read(config: Config, scores: Scores, ask: Ask) -> ReadReport:
    caps = config.router.capabilities
    assert caps is not None
    report = ReadReport()
    jev = {
        key: {"hash": r.hash, "needs": r.needs, "read_at": r.read_at, "jev_model": r.jev_model}
        for key, r in scores.jev.items()
    }
    for key, bench in scores.benchmarks.items():
        if bench.requirements is not None:
            report.manual.append(key)
            continue
        digest = content_hash(bench, caps.requirements)
        held = scores.jev.get(key)
        if held is not None and held.hash == digest:
            report.unchanged.append(key)
            continue
        texts = list(bench.example_tasks[:MAX_EXAMPLES]) or [bench.description]
        try:
            readings = [await ask(_packet(text, caps), caps.requirements) for text in texts]
        except Exception as exc:  # noqa: BLE001 - one benchmark's failure must not lose the others
            report.errors[key] = f"{type(exc).__name__}: {exc}"[:300]
            continue
        jev[key] = {
            "hash": digest,
            "needs": {r: sum(x[r] for x in readings) / len(readings) for r in caps.requirements},
            "read_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "jev_model": config.tier(caps.jev_tier).model,
        }
        report.asked.append(key)
    if report.asked:
        write_sidecar(scores.sidecar_path, jev=jev)
    return report


__all__ = ["MAX_EXAMPLES", "ReadReport", "read"]
