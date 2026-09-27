"""Ask Jev about a few tasks and print what the capabilities router would pick.

    python scripts/capabilities_dry_run.py [config.capabilities.yaml] [tasks.jsonl]

Only Jev is called; no destination model runs. Each task is a JSON line with
`messages` and an optional `packet`, as a client would send it. Without a tasks
file a built-in sample runs.
"""

from __future__ import annotations

import asyncio
import json
import sys

from jev_model_router.capabilities import CapabilityRouter
from jev_model_router.catalog import check_catalog
from jev_model_router.config import load_config
from jev_model_router.discovery import expand
from jev_model_router.eligibility import evaluate
from jev_model_router.routing import JevUnavailable
from jev_model_router.schemas import ChatCompletionRequest

SAMPLE = [
    {"messages": [{"role": "user", "content": "Rename the variable `cnt` to `count` in utils.py."}]},
    {"messages": [{"role": "user", "content": "Write a commit message for: fixed a typo in README.md"}]},
    {"messages": [{"role": "user", "content": "What does the regex ^\\d{3}-\\d{4}$ match?"}]},
    {"messages": [{"role": "user", "content": "cargo clippy says `unused import: std::fmt` in core/src/http.rs line 12. Fix it."}],
     "packet": {"kind": "code_change", "size": "trivial"}},
    {"messages": [{"role": "user", "content": (
        "The Tauri shell tests fail intermittently on Windows with `os error 32` (file in use) when "
        "scripts/gates.sh runs clippy after the app was started with npm run tauri dev. The message names "
        "libresource.a. Find the real cause and propose a fix to the gate script and docs.")}],
     "packet": {"kind": "debug", "size": "medium", "repo": "nucleos", "languages": "rust, ts"}},
    {"messages": [{"role": "user", "content": (
        "Add a capabilities router to jev-model-router: a new router kind that sends a task packet to Jev, reads "
        "per-requirement probabilities, and picks the cheapest model card that covers them, with quota "
        "accounting per subscription, new codex_cli backend, config validation, app wiring and tests.")}],
     "packet": {"kind": "code_change", "size": "large", "files_in_scope": 9, "risk": "normal"}},
    {"messages": [{"role": "user", "content": (
        "Given 11 models with per-item costs, derive the upper concave envelope of (cost, accuracy) points "
        "allowing random mixtures, and prove a paired bootstrap CI for the gain of a routing policy is valid.")}]},
    {"messages": [{"role": "user", "content": (
        "Design the sync architecture between the Rust core daemon, the Go sidecars and the Tauri shell so "
        "that approvals survive a daemon restart without double-executing a git merge. Weigh the options.")}],
     "packet": {"kind": "plan", "size": "large", "risk": "elevated"}},
]


async def main() -> None:
    config = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.capabilities.yaml")
    config, found, _ = expand(config, check_catalog(config))
    print(f"{len(found)} tier(s) discovered")
    tasks = SAMPLE
    if len(sys.argv) > 2:
        tasks = [json.loads(line) for line in open(sys.argv[2], encoding="utf-8") if line.strip()]
    router = CapabilityRouter(config)
    try:
        for task in tasks:
            request = ChatCompletionRequest.model_validate({"model": "auto", **task})
            eligible = list(evaluate(config, request).eligible)
            goal = task["messages"][-1]["content"].replace("\n", " ")[:70]
            try:
                decision = await router.decide(request, eligible)
            except JevUnavailable as exc:  # on_jev_failure: reject
                print(f"\n{goal}\n  -> refused: {exc.why}")
                continue
            reason = json.loads(decision.reason)
            need = " ".join(f"{k[:5]}={v:.2f}" for k, v in reason.get("need", {}).items())
            print(f"\n{goal}\n  -> {decision.tier}  P={decision.score if decision.score is None else round(decision.score, 3)}  "
                  f"{reason.get('rule')}\n  need: {need}\n  passed over: {reason.get('passed_over')}")
    finally:
        await router.aclose()


if __name__ == "__main__":
    asyncio.run(main())
