# jev-model-router

OpenAI-compatible HTTP proxy that picks which model tier serves each request and logs
what that decision cost, with a counterfactual row per configured tier. The README is the
short front page; `DESIGN.md` is the long-form design record. Read the section you are
touching before changing behaviour, and record new measurements and design reasons there.

This file is the single set of agent instructions for the repo. Codex reads it directly;
Claude Code reads it through `CLAUDE.md`, which imports it. Edit this file, not a copy.

## Setup

Python 3.11+. From the repo root:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Windows
# .venv/bin/python -m pip install -e ".[dev]"     # macOS / Linux
```

Local config is per-machine and git-ignored (`config.yaml`, `config.capabilities.yaml`,
`anchors.capabilities.yaml`, `*.db`, `jev-model-router.catalog.json`, `*.imported.json`,
`*.derived.json`). Start from `config.example.yaml`; never commit a machine's config,
database or credential. Keys are read from the env var a tier names in `api_key_env`,
else from the user's secrets file (`jev-model-router keys set`).

## Commands

```bash
.venv/Scripts/python -m pytest                          # full suite, no network, ~10 s
.venv/Scripts/python -m pytest tests/test_config.py -q  # one file
python -m jev_model_router -c config.yaml check              # validate a config, start nothing
python -m jev_model_router -c config.yaml serve              # run the proxy
python -m jev_model_router -c config.yaml stats              # read the request log back
python -m jev_model_router -c config.yaml mcp                # route-only API as MCP tools on stdio
python -m jev_model_router -c config.yaml keys               # which keys each tier needs; `keys set` stores them
python -m jev_model_router measure-shape                      # task_shape from local transcripts (read-only)
python -m jev_model_router --help                            # train, label, reconcile, calibrate, benchmarks ...
```

Scripts in `scripts/` (`cli_smoke.py`, `capabilities_dry_run.py`, `judge_eval.py`) make
real calls and spend quota or money. Do not run them unless asked.

## Layout

- `jev_model_router/app.py`: FastAPI app. `/v1/chat/completions`, `/v1/models`, `/v1/route`,
  `/v1/route/{id}/outcome`, `/healthz`, `/routing` (live view, tiers, config editor).
- `jev_model_router/cli.py`: every subcommand; `__main__.py` makes `python -m jev_model_router` work.
- `jev_model_router/mcp_server.py`: `jev-model-router mcp`, a stdio MCP server forwarding to the
  `/v1/route` endpoints (in process, or `--url` / `JEV_MODEL_ROUTER_URL` to a running proxy).
- `jev_model_router/delegate.py`: runs a route decision on another agent CLI and records its usage.
- `jev_model_router/delegates/`: one adapter per agent CLI, registered in `delegates/__init__.py`.
- `jev_model_router/config.py`: config parsing and validation (`ConfigError` names the file).
  A tier's `endpoints:` lists providers for one model; the first with a key is used.
- `jev_model_router/config_edit.py`: the `/routing` Config tab's writes; changes land in the
  YAML text line by line, so comments survive, and are re-parsed to prove they landed.
- `jev_model_router/keystore.py`: API keys, from the environment or the per-user secrets file
  (`JEV_MODEL_ROUTER_HOME`). Tests get an empty one each (`conftest.py`); never read the real one.
- `jev_model_router/backends/`: one module per backend kind (`ollama`, `openai_compatible`,
  `claude_cli`, `codex_cli`, `jev`), registered in `backends/__init__.py`.
- `jev_model_router/routing.py`, `eligibility.py`, `dominance.py`, `capabilities.py`: the
  routing decision. `verification.py`: cheap answer checked by a stronger tier.
- `jev_model_router/calls.py`: the call router: prices, the subscription ledger, and a task's
  cache-aware shape. The capabilities router is given one. `shape.py`: `measure-shape`.
- `jev_model_router/scores*.py`, `benchmark.py`, `calibration.py`: benchmark evidence into
  card levels. `catalog.py`, `discovery.py`: find what models a machine can reach
  (CLI caches, Ollama, any API's `/models` listing such as OpenRouter's).
- `jev_model_router/db.py`, `route_sync.py`, `stats.py`, `pricing.py`, `reconcile.py`: the log,
  cross-process event catch-up, and its costs.
- `tests/`: pytest, one file per module. `conftest.py` fakes backends through
  `create_app(backend_factory=...)`; no test touches the network, keep it that way.
- `docs/superpowers/`: specs and plans for in-flight work.
- `plugin/`: the Claude Code and Codex plugin (MCP server + skill), listed by
  `.claude-plugin/marketplace.json` and `.agents/plugins/marketplace.json`. Claude reads
  `.claude-plugin/`, `.mcp.json`, `skills/`; Codex reads `.codex-plugin/` and `codex/`.
  Keep the two skills in step. Check with `claude plugin validate .` and `./plugin`.

## Model-agnostic by design

This project must keep working, unedited, however many models are released. A new
model is data, never a code change:

- No model, vendor or version is named in `jev_model_router/` except in comments that
  record a measurement. Everything model-specific lives in config and data files
  (profiles, cards, anchors, calibration output).
- A newly released model must have a clear, mechanical path to a card and into
  Jev routing: discovered by the catalog, given a prior from evidence, refined by
  anchors and logged outcomes. "Hand-edit the source when X ships" is a design bug.
- Mechanisms are judged by whether they generalise to model N+1: prefer rules
  over the requirements and the evidence (dominance, caps, fitted scales) to rules
  about particular models.

## Conventions

- Every behaviour change comes with a test in the matching `tests/test_<module>.py`.
- Errors a user can hit in config or data files raise `ConfigError` naming the file.
- Files the tool writes (config, calibration, catalog) are written atomically.
- Commits use Conventional Commits with a scope: `fix(scores): ...`, `docs(spec): ...`.
