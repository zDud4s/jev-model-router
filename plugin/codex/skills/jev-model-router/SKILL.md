---
name: jev-model-router
description: Use before delegating a substantial task to a subagent or a separate `codex exec` run (implementing, fixing, reviewing, planning), to ask jev-model-router which model and reasoning effort should do it, and after the work has been checked, to report whether it passed. Skip it for quick lookups and for work you do yourself in this turn.
---

# Routing delegated work through jev-model-router

jev-model-router estimates, per task, the cheapest model and effort likely to succeed, and learns
from what you report back. It never runs anything itself: you ask, you run, you report.

The `jev-model-router` MCP server gives you two tools.

## 1. Before you delegate: `route`

Call `route` with the task written as you would brief the agent that does it. Add what you
know: `stage` (plan, implement, review...), `files`, and on a retry `attempt`, `gate_output`
(the tail of the failing test or review) and `failed` (the model or `model@effort` that
failed it, as the previous `route` returned it).

When the user has said which models may do the work (or may not), pass it on every
`route` call as `models` / `exclude`: globs over model ids and tier names, such as
`["anthropic/*", "*gpt-6*"]`. The router only chooses among what they leave.

The answer has `model`, `effort`, `runner`, `success` (the router's estimate) and a
`decision_id`. Keep the `decision_id`. If it has a `why`, the router fell back to its
default (for instance the judge could not be reached): mention it to the user.

## 2. Run it on what was chosen

- Delegating to a subagent that takes a model and reasoning effort: give it the returned
  `model` and `effort` as they are.
- Otherwise, for a self-contained task, run it from the repository root:

  ```bash
  codex exec -m <model> -c model_reasoning_effort="<effort>" "<the brief>"
  ```

  Leave out the `-c` when `effort` is null.
- If the answer's `runner` is not `codex`, you cannot run it from here; use the default and
  report `error` with that reason.

## 3. After the work is checked: `report_outcome`

Once the task's own check has run (tests, build, your review of the result), report it once
with the `decision_id`:

- `pass` / `fail`: what the check said. This is the label the router learns from, so report
  what the check said, not how the work looked.
- `rate_limited`: the model refused for quota or rate limits.
- `error`: the run broke for a reason that says nothing about the model (tool crash,
  interruption, the choice could not be followed).

On `fail`, route the retry with `attempt`, `gate_output` and `failed` filled in: the router
never offers a choice it rates below the one that failed.

## If the tools fail

If the jev-model-router tools are missing, the server did not start, usually because it found no
config: the user must set `JEV_MODEL_ROUTER_CONFIG` to their jev-model-router config file (see the
project README, "Use it from Claude Code or Codex"). Carry on with the default model; do not
block the task on routing.
