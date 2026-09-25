"""`benchmarks read`: Jev reads what each benchmark measures, once per change."""

from __future__ import annotations

import asyncio
import json

from llm_router.config import parse_config
from llm_router.scores import load_scores
from llm_router.scores_read import MAX_EXAMPLES, read

from test_scores import raw_config


class Ask:
    def __init__(self, answers=None, fail_on: str | None = None) -> None:
        self.answers = answers or (lambda i: {"reasoning": 0.9, "niche": 0.1})
        self.fail_on = fail_on
        self.calls: list[str] = []

    async def __call__(self, packet: str, questions: dict[str, str]) -> dict[str, float]:
        self.calls.append(packet)
        if self.fail_on and self.fail_on in packet:
            raise RuntimeError("jev is down")
        return self.answers(len(self.calls) - 1)


def setup(tmp_path, benchmarks: str):
    path = tmp_path / "b.yaml"
    path.write_text("benchmarks:\n" + benchmarks + "points: []\n", encoding="utf-8")
    config = parse_config(raw_config(benchmarks={"path": str(path)}))
    return config, path


def run(config, ask):
    return asyncio.run(read(config, load_scores(config), ask))


def test_a_benchmark_is_read_once_and_not_again_until_its_words_change(tmp_path):
    config, path = setup(tmp_path, "  code: {description: fix code}\n")
    ask = Ask()
    assert run(config, ask).asked == ["code"] and len(ask.calls) == 1
    side = json.loads((tmp_path / "b.derived.json").read_text(encoding="utf-8"))
    assert side["jev"]["code"]["needs"] == {"reasoning": 0.9, "niche": 0.1}
    assert side["jev"]["code"]["jev_model"] == "jev-latest"
    assert run(config, ask).unchanged == ["code"] and len(ask.calls) == 1
    path.write_text("benchmarks:\n  code: {description: fix other code}\npoints: []\n", encoding="utf-8")
    assert run(config, ask).asked == ["code"] and len(ask.calls) == 2


def test_example_tasks_are_read_instead_capped_and_averaged(tmp_path):
    tasks = "".join(f"      - task number {i}\n" for i in range(12))
    config, _ = setup(tmp_path, "  code:\n    description: fix code\n    example_tasks:\n" + tasks)
    ask = Ask(answers=lambda i: {"reasoning": 1.0 if i % 2 else 0.0, "niche": 0.5})
    run(config, ask)
    assert len(ask.calls) == MAX_EXAMPLES == 10
    needs = json.loads((tmp_path / "b.derived.json").read_text(encoding="utf-8"))["jev"]["code"]["needs"]
    assert needs == {"reasoning": 0.5, "niche": 0.5}


def test_a_manual_benchmark_is_never_read(tmp_path):
    config, _ = setup(tmp_path, "  code: {description: fix code, requirements: {reasoning: 1.0}}\n")
    ask = Ask()
    assert run(config, ask).manual == ["code"] and ask.calls == []


def test_a_jev_error_keeps_the_old_reading_and_the_others_are_still_written(tmp_path):
    config, path = setup(tmp_path, "  code: {description: fix code}\n  lore: {description: know things}\n")
    run(config, Ask())
    path.write_text("benchmarks:\n  code: {description: fix code v2}\n  lore: {description: know more things}\n"
                    "points: []\n", encoding="utf-8")
    report = run(config, Ask(answers=lambda i: {"reasoning": 0.3, "niche": 0.3}, fail_on="v2"))
    assert list(report.errors) == ["code"] and report.asked == ["lore"]
    side = json.loads((tmp_path / "b.derived.json").read_text(encoding="utf-8"))["jev"]
    assert side["code"]["needs"] == {"reasoning": 0.9, "niche": 0.1}  # the old one
    assert side["lore"]["needs"] == {"reasoning": 0.3, "niche": 0.3}
