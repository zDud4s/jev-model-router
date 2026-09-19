# llm-router

An OpenAI-compatible proxy that decides which model serves each request, and records what
that decision cost.

One API in, any model out. Point a client at this endpoint instead of a vendor's, and the
backend it reaches is a line of configuration rather than a code change.

It routes, it verifies, and it measures. It does not yet classify — see
[What this does not do yet](#what-this-does-not-do-yet).

## Why the log is the point

A router's whole claim is that it saves money without costing quality. That claim is only
worth as much as the baseline it is measured against, and the tempting baseline — "what if
every request had gone to the most expensive model" — flatters any router, because nobody
runs that way. Configuring a cheaper model is free and saves a great deal on its own.

So every request writes a counterfactual row for **each** configured tier: what this exact
request, with its actual token counts, would have cost there. Savings against any baseline
is then a query rather than an argument, and a reader can check the number that matters to
them instead of the one that flatters the author.

The same rows record whether a tier was even *eligible* for the request, so "cheaper" is
never confused with "could not have served it".

And when the verification loop is on, its calls are billed to the request that caused them.
A savings number that counts the cheap answer and omits the review that approved it is not a
savings number. `stats` prints the loop's spend as a share of the bill, because that share is
what decides whether the whole idea works.

## Install

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt     # macOS / Linux
```

Python 3.11 or newer.

## Configure

Copy `config.example.yaml` and edit it. That file **is** the model catalog — nothing in the
source names a model.

```yaml
tiers:
  cheap:
    backend: ollama                       # local; /api/chat
    model: llama3.1:8b
    base_url: http://localhost:11434
    context_window: 131072
    supports_tools: false
    # no `prices:` -- a local model costs zero, and that is an answer, not a gap

  top:
    backend: openai_compatible            # OpenAI, OpenRouter, vLLM, a proxy...
    model: claude-opus-5
    base_url: https://api.example-proxy.com/v1
    api_key_env: LLM_ROUTER_TOP_KEY       # read at call time; no credential in this file
    context_window: 200000
    supports_tools: true
    prices:                               # USD per 1M tokens, every field optional
      input: 15.00
      output: 75.00
      cache_read: 1.50
      cache_write: 18.75
```

Validate it without starting anything:

```bash
python -m llm_router -c config.yaml check
```

### Eligibility comes before routing

A tier is removed from consideration — before the router is asked anything — when the
request cannot fit its `context_window` (input plus the output the caller reserved), or when
the request carries `tools` and the tier declares `supports_tools: false`.

This is a hard filter, not a preference. Routing to a backend that will reject the request
is the failure it exists to prevent, and it is the reason `context_window` is worth setting
honestly for every tier. Each rejection is logged with its reason.

## Verification: have the strong model check the cheap one

Routing alone is a bet — the cheap tier is assumed good enough and nobody ever finds out.
Verification turns the bet into a measurement: after a cheap tier answers, a stronger tier
reads the question and the answer and returns `VERDICT: PASS` or `VERDICT: FAIL`. A failure
re-runs the request at the strong tier, and the client receives *that* answer. The cheap
answer that failed review is never sent.

```yaml
verification:
  enabled: true
  verifier_tier: top
  verify_tiers: [cheap]
  escalate_to: top
  sample_rate: 1.0
```

Those verdicts are also the labelled data a difficulty classifier needs, which is why this is
built before the classifier rather than after it: "cheap answered and it held" and "cheap
answered and it did not" are exactly the two classes, produced by ordinary traffic instead of
by hand.

### The arithmetic, stated plainly

A verified request costs **two** calls. A failed one costs **three**. With a free local tier
and a `top` verifier, a request that is routed cheap and then fails review costs *more* than
sending it straight to `top` would have — you paid `top` twice. The loop only pays for itself
when the pass rate is high enough that the reviews you buy cost less than the escalations you
avoid, and there is exactly one way to find out what your pass rate is, which is to run it at
`sample_rate: 1.0` for a while and read the number.

The report does not hide this. It is the one below, from four requests with one failure.

### What it declines to do, and why

- **Streams are never verified.** Once the first byte is on the wire the answer cannot be
  retracted. Those requests are recorded as `skipped: streaming`, not omitted.
- **The verifier never reviews its own answer.** A tier grading itself measures nothing.
- **The verifier passes the eligibility gate like anything else** — against the *review*
  prompt, which carries the transcript and the answer and is therefore bigger than the
  original. A verifier that cannot fit it is `skipped: verifier_ineligible`, never sent to a
  backend that would reject it.
- **An answer that is only tool calls has no text to review**, and is skipped rather than
  failed.
- **A verifier that breaks does not break the request.** The verdict is recorded as `error`,
  which is a different thing from `fail`: one is the reviewer malfunctioning, the other is the
  reviewer working and disliking what it read.
- **A reply with no verdict in it is never guessed at.** It is counted as unparseable, the
  configured fallback decides, and `stats` says how many of its verdicts came from that
  fallback instead of from the model.

`on_unparseable` and `on_verifier_error` default to `accept`: a misconfigured verifier should
not quietly double your bill. Set them to `escalate` if you would rather pay than ship an
unchecked answer.

## Run

```bash
python -m llm_router -c config.yaml serve
```

Then point any OpenAI-compatible client at it:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=unused          # the proxy holds the real keys
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "hello"}]}'
```

Endpoints: `POST /v1/chat/completions` (streaming and non-streaming), `GET /v1/models`,
`GET /healthz`.

## Read the log back

```bash
python -m llm_router -c config.yaml stats
```

```
requests        4
errors          0
input tokens    400
output tokens   200
actual spend    $0.026250

spend by tier
  cheap               4 req  $    0.000000  in       400  out       200  cached         0

verification
  reviewed            4   pass 3  fail 1  verifier errors 0
  fail rate        25.0%  of reviewed answers
  escalated           1   a second, dearer answer the client actually received
  verifier spend        $    0.021000
  escalation spend      $    0.005250
  loop overhead         $    0.026250  (100.0% of actual spend)

counterfactual baselines: what everything would have cost at one tier
  always top        $    0.021000  savings $   -0.005250
  always mid        $    0.004200  savings $   -0.022050
  always cheap      $    0.000000  savings $   -0.026250
      note: 4 row(s) had no price configured for this tier
```

*(fake backends with fixed token counts — the shape is real, the numbers are not a
measurement of any model)*

Read the savings column. Every one of them is **negative**: the local tier answered all four
requests for nothing, and then `top` was paid four times to review them and once more to
redo one. That is a router losing money, printed as a router losing money. A report that only
compared against `always top` and only counted the routed call would have shown a saving
instead, and it would have been the same four requests.

Two spend figures appear above and they are different numbers on purpose:

- **`actual spend`** is the whole bill — routed call, reviews, escalations. Every savings
  figure is measured against this one.
- **`spend by tier`** is the routed call alone, because "what is `cheap` costing us" must not
  be inflated by a verifier `cheap` never chose.

The counterfactual baselines stay one call at one tier. "What would this have cost had it
gone straight to `top`" means one `top` call, not one plus a review — charging the baseline
for a loop it would never have run is how a router flatters itself into a saving.

The log is SQLite. Query it directly for anything the summary does not cover; the schema is
in `llm_router/db.py`.

Prompts are stored as a SHA-256 hash by default, because prompts are user data. Set
`log.store_prompts: true` to keep the text as well.

A log write that fails is recorded in `log_failures` and never fails the request. A broken
log must not take the proxy down with it.

## Tests

```bash
.venv/Scripts/python -m pytest
```

No network: backends are faked.

## What this does not do yet

One piece of the design is still absent, behind a seam that already exists:

- **The difficulty classifier.** `router.kind` is `static` today. A learned router implements
  the same `Router` interface in `llm_router/routing.py` and nothing else changes. Two
  warnings for whoever builds it, both learned the expensive way: split the training data by
  **session or user**, never randomly by prompt — prompts from one conversation share
  vocabulary, and a random split lets the model recognise the conversation instead of the
  difficulty. And filter the corpus: transcripts are full of tool results, system notices and
  attachment placeholders that are not prompts at all, and a classifier will happily learn to
  tell those apart and report a score that means nothing.

  The labels it needs are already being written: `verifications.verdict` is a `pass`/`fail`
  per prompt, produced by real traffic. Turn the loop on at `sample_rate: 1.0`, let it run,
  and the training set builds itself.

It is not stubbed with a fake. Where it is missing, the code says so.

## A baseline worth beating

Before trusting any router, price the config change it is competing with. Two models from the
same family often sit at a fixed ratio across every price axis — if the cheaper one is 60% of
the dearer one on input, output, cache read and cache write alike, then *"always use the
cheaper one"* saves exactly 40%, on any traffic mix whatsoever, for one line of YAML and no
moving parts.

A router has to beat that, not beat `always top`. The counterfactual table prints every tier
for this reason: the baseline that makes the router look worst is the honest one, and it is
already in the report.
