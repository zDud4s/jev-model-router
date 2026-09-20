# llm-router

An OpenAI-compatible proxy that decides which model serves each request, and records what
that decision cost.

One API in, any model out. Point a client at this endpoint instead of a vendor's, and the
backend it reaches is a line of configuration rather than a code change.

It routes, it verifies, it learns which prompts the cheap model gets wrong, and it
measures all three. It has met one real model so far, and that broke it in three places — see
[What happened the first time it met a real model](#what-happened-the-first-time-it-met-a-real-model).

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
    prices:                               # USD per 1M tokens, every field optional;
      input: 15.00                        # an omitted cache_read bills at `input`
      output: 75.00
      cache_read: 1.50
      cache_write: 18.75
```

Validate it without starting anything:

```bash
python -m llm_router -c config.yaml check            # add --prices to compare with OpenRouter's list
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

### The failures a judge should never be paid to find

Some answers are indefensible without reading them. Two are checked before the reviewer is
called, cost nothing, and therefore ignore `sample_rate` — what sampling rations is a paid
review, and there is nothing here to pay for:

- **An empty answer.** A thinking model that spends its whole window reasoning returns a 200
  with nothing in it. An answer that is not there is not a thing to review.
- **An answer cut off at our own output budget** (`finish_reason: length`). The model was
  mid-sentence when the budget ran out.

The second one came out of labelling 400 GSM8K questions through `qwen3.5:4b`: **60 of the 70
failures were not wrong answers but unfinished ones**, every one stopped at exactly the tier's
2048-token `num_predict`, while no correct answer came within five tokens of it. A judge was
being bought to rediscover a fact already in the response body.

Two things keep the rule honest:

- **Only when the budget was ours.** A caller who sent `max_tokens: 20` asked for a short
  answer and got one. Failing it would buy them a second answer cut off at 20 tokens by the
  same cap, and charge them for it — so a caller-set budget disables the check.
- **An escalation that is also cut off says so.** A stronger tier capped just as low returns a
  second unfinished answer at the stronger tier's price. The reason names the tier and the
  budget, because the fix is that number and not a third call.

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

## The classifier: route on a prediction instead of a guess

Verification answers "was the cheap answer good?" — afterwards, having already bought two
calls. The classifier answers "will it be?" — beforehand, from the prompt alone, for free.
Above a threshold the request skips the cheap tier entirely and goes straight to the strong
one; below it, nothing changes.

```yaml
router:
  kind: classifier
  default_tier: cheap       # the tier whose failures the model predicts
  strong_tier: top          # where a predicted failure goes instead
  model_path: ./classifier.json
  threshold: 0.5
  explore_rate: 0.05        # see "the model poisons its own evidence", below
```

The model is a logistic regression over the words of the last user turn plus a handful of
size features. No new dependency, a readable JSON file, and weights you can print and argue
with. It predicts one thing: **P(this prompt's cheap answer fails review)**.

### Training it

```bash
python -m llm_router -c config.yaml train --out classifier.json
```

There is no bundled model and no shortcut to one. The labels come from the verification loop
and nowhere else, so the sequence is: turn verification on at `sample_rate: 1.0`, serve real
traffic, then train. Two defaults stand in the way on purpose, and both refuse loudly rather
than producing a model out of nothing:

- **`log.store_prompts` is false.** Prompts are user data and the log holds a SHA-256 by
  default. A hash cannot be read for difficulty. Turning it on is a decision about storing
  conversations, and rows already written cannot be recovered.
- **Verdicts the verifier did not actually give are not labels.** A row with
  `unparseable = 1` had its outcome chosen by the `on_unparseable` fallback — with the
  default `accept`, that manufactures `pass` labels, and training on them teaches the model
  that a broken verifier is a well-answered prompt. Those rows are dropped.

Below 40 labels, or fewer than 5 of the rare class, it refuses: a model fitted on less has
memorised the log, and "always cheap" is the correct router anyway.

### What the report says

From the test suite's fake backends, with a deliberately learnable signal planted — the cheap
tier fails every prompt about integrals and passes every one about meeting notes:

```
corpus          72 labelled request(s) in 12 conversation(s)
                30 failure(s) in the training split of 60
held out        12 request(s), whole conversations only
model           a0c20a6d6457  predicts cheap, judged by top
                17 weight(s); an unseen prompt scores 0.482, the base rate

held out by conversation -- the number to believe
  accuracy        1.000   majority-class baseline 0.500
  precision       1.000   of the requests it sent to the strong tier
  recall          1.000   of the failures it caught
  escalates       50.0% of requests

what the held-out traffic would have cost, per policy
  policy                                     cost  bad answers  escalations
  classifier @ 0.50, no verifier     $   0.037800            0            6
  always cheap, no verifier          $   0.000000            6            0
  always cheap + verify all          $   0.134100            0            6
  always strong                      $   0.075600            0            0

strongest weights (+ pushes toward the strong tier)
    -1.386  w:meeting
    -1.386  w:summarise
    +1.313  w:prove
    +1.313  w:integral
```

A perfect score on planted data proves the pipeline runs, not that the idea works. Real
prompts do not separate on four words, and the number that comes back will be lower — that is
the number worth having.

The policy table is the argument for the classifier over the loop, stated in money: the same
held-out traffic, zero bad answers either way, at **a third of the cost** of reviewing every
request. And it is printed next to `always cheap`, which costs nothing and ships every
failure. Money and bad answers are separate columns. They are never summed, because this code
does not know what a wrong answer costs you and will not invent a price to make its own table
come out well.

### Two ways to lie to yourself, both measured here

**The split.** Prompts from one conversation share vocabulary, so a split that scatters them
across train and test lets the model recognise the *conversation* and report that as
difficulty. The report fits the same data twice — grouped by conversation, and split by row —
and prints both, so the gap between the honest number and the flattering one is a line of
output rather than a warning nobody heeds. On a corpus where only the conversation is
learnable, the wrong split reads perfect and the right one reads like a coin toss;
`tests/test_classifier.py` asserts exactly that.

**The class balance.** If 10% of requests fail, answering "never fails" scores 90%. Every
accuracy in the report is printed beside the majority-class baseline that needs no model at
all, and a model that fails to beat it is told so, in the report, in those words.

### The model poisons its own evidence

This one has no fix, only a price. A classifier that is always obeyed sends every prompt it
scores high to the strong tier — where no verifier looks at it, so it never becomes a label.
The next model is fitted only on prompts the current one already believed were easy, and its
measured failure rate looks wonderful for exactly as long as nobody checks.

`router.explore_rate` is the price: a few percent of would-be escalations go to the cheap tier
anyway, are answered, and are reviewed. Those rows are the only ones that can ever contradict
the model. Here is the same fake traffic again, this time served by the trained model with
`explore_rate: 0.10`:

```
classifier
  scored             72   by a0c20a6d6457 (72)
  reviewed           44   scored requests a verifier also judged
  unreviewed         28   scored and never checked -- these rows cannot contradict the model
  explored            8   would-be escalations kept cheap on purpose, so they could be judged
  mean score     0.991 on answers that failed, 0.009 on answers that passed
  score band      rows   failed   actual fail rate
    0.0-0.2         36        0      0.0%
    0.8-1.0          8        8    100.0%
```

Read the last line and then the one above it. **Those eight rows are the explored ones** — the
only requests the model wanted to escalate and did not. Without exploration that band is
empty, the table shows nothing but the easy half of the traffic, and it agrees with the model
forever.

`mean score` on failures minus `mean score` on passes is the whole model in one number. At or
below zero the scores are noise, whatever the training report said, and `stats` prints that
verdict in capitals rather than leaving it to be noticed.

**The two sampling knobs multiply.** `explore_rate` decides which requests stay cheap;
`sample_rate` then decides, independently, which of those get reviewed. At 0.10 and 0.25 the
labelled share of would-be escalations is 2.5%, not 10% — in the sampled run below, eight
explored requests produced four labels rather than eight. If exploration is there to collect the labels the model
cannot otherwise get, the sampling has to leave enough of them alive to matter.

### Does it actually save anything?

The same 72 fake requests, three ways, read off the counterfactual table. Savings are against
`always top`, the flattering baseline:

| | actual spend | vs always top | loop overhead |
|---|---|---|---|
| static router, verify everything | $0.804600 | **−$0.351000** | 100.0% |
| classifier, verify everything | $0.611400 | **−$0.094800** | 60.8% |
| classifier, `sample_rate: 0.25` | $0.356400 | **+$0.160200** | 32.8% |

That last row is the first positive number this project has ever printed, and it arrives only
once the classifier is doing the deciding and the verifier has been sampled down to spot
checks. It is also, still, a loss against `always mid` (−$0.253080) and `always cheap`
(−$0.356400) — the baselines that need no router at all. Which is the point of printing every
baseline: the router beat the comparison that flatters it and lost to the two that do not, and
you can read both in the same table.

(Fake backends throughout. The shapes are real; the numbers are not a measurement of any
model.)

### The first real corpus said no

400 GSM8K questions through `qwen3.5:4b` on a laptop GPU, labelled from the answer key:
6.4 hours, 295k generated tokens, $0. It failed 70 of them — 60 by running past a 2048-token
cap without ever stating an answer, 10 by answering wrongly. Then `train`:

| | accuracy | majority baseline |
|---|---|---|
| words + length features | 0.760 | 0.823 |
| length features only | 0.812 | 0.823 |

**Neither beats answering "cheap" every time**, and the report says so itself rather than
printing the cost table as a result. The strongest weights were `runs`, `equally`, `can` —
523 weights fitted on 307 examples, memorising vocabulary. The only real signal in the corpus
is weak and blunt: the longest quarter of the questions failed 26.2% of the time against
about 14% for the rest, which 523 word features drown rather than exploit.

Two things this does establish. The labelling path works end to end, and it is free. And the
negative result is legible: a router that could not tell you it had learned nothing would
have shipped these weights, because its cost column looks like a saving ($0.17 to catch 4 of
17 failures) right up until you read the line above it.

### What the corpus then taught the trainer

Three of those numbers were the trainer's fault rather than the data's, and each one is now
a flag on `train`:

- **The words were not merely useless, they were harmful.** Cross-validated on the training
  split alone, `words + length` scored **0.40 AUC — below chance** — while the same fit
  without them scored 0.618. A vocabulary fitted on a few hundred prompts describes the
  corpus. `--tune` cross-validates a small grid (words on/off, `min_df`, `l2`) on the
  training split and reports every point, so a grid that found nothing says so.
- **Accuracy was the wrong headline.** At a 17% failure rate, any model that escalates
  anything scores below "never escalate". The report now prints **AUC**, which needs no
  threshold, and **lift** — how much likelier an escalated request was to be a failure than a
  random one. The tuned model: AUC 0.590, lift 1.66x, catching 29.4% of failures for 17.7% of
  traffic. Weak, and no longer invisible under an accuracy column.
- **The threshold was a cliff, not a knob.** Every score this model produces falls between
  0.36 and 0.45, so `0.5` escalated nothing and `0.4` escalated everything — and the sweep
  printed seven rows of two policies. Thresholds are now read off the training split's own
  scores: `--target-escalation 0.15` means *escalate the hardest 15%*, and the sweep spans the
  range the model actually occupies. A threshold that escalates all or none of the held-out
  set is called out as such.

None of this rescues the result — a 1.66x lift on one benchmark is not a router. It makes the
next negative result cheaper to read.

One caveat about that cost table on a labelled database: the `always cheap + verify all` row
prices verification at zero, because here the labels came from an answer key and no verifier
was called. In production that row costs a judge call per request.

### What it will not do

- **Overrule an explicit request.** `model: "top"` gets `top`, unscored. Guessing over a
  stated preference is not routing.
- **Start without a model.** `kind: classifier` with a missing or unreadable model file
  refuses to boot. Falling back to the static router would produce a system that looks like it
  is classifying and is not — indistinguishable from a working one in every report.
- **Run on a model trained for another tier.** The file records which tier's failures it
  describes; a mismatch is a config error, because the scores would be confident and
  meaningless.
- **Reload on its own.** The model is read once at startup, so a running server keeps its
  weights until restarted. `train` says so after it writes.

One interface changed to make this possible, and an earlier draft of this README promised it
would not: `Router.choose` returned a tier name, and `Router.decide` now returns a decision
carrying the score, the model fingerprint and the reason. A router that computes a number and
does not report it cannot be checked afterwards, which is the one thing this project is not
willing to ship.

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

## What happened the first time it met a real model

Everything above was built against fake backends. The first run against a real one —
`qwen3.5:4b` through a local Ollama, a thinking model, three questions — broke in three
places, none of them reachable with a backend that always answers politely:

1. **The context window was fiction.** The config said `context_window: 32768` and the
   eligibility gate reasoned about 32768. Nothing ever told Ollama, which picks its own default
   from free VRAM — 4096 on that machine — and truncates silently past it. The Ollama backend
   now sends `num_ctx` from the tier's `context_window`, so the number the gate enforces is the
   number the backend runs with.
2. **An empty answer went to the client as a 200.** The model spent its whole window reasoning
   and wrote nothing. The loop recorded `no_answer_text` — a skip meant for answers that are
   only tool calls — and let it through. An answer with no text and no tool calls is now a
   `fail`, decided without paying a reviewer, regardless of sampling, and escalated.
3. **Every verdict was unparseable, and the fallback called them passes.** The verifier spent
   its 200-token budget thinking and never wrote a verdict. `stats` flagged it
   (`UNPARSEABLE 2`), which is what that line exists for, but the recorded reason just said
   "unparseable", sending the reader to the regex. It now names the cause: the verifier hit
   `max_verdict_tokens`.

After those fixes, the same three questions: three correct answers, three parseable verdicts.

Then the question the fixes could not answer — **can a real small judge say FAIL?** Ten planted
answers, known truth, the same model judging:

| verifier setup | correct verdicts | tokens per verdict | seconds per verdict |
|---|---|---|---|
| `think: false`, `max_verdict_tokens: 200` | 9 of 10 | 6 | ~2 |
| thinking on, `max_verdict_tokens: 3000` (4 hardest cases) | 4 of 4 | 490–1054 | 21–56 |

The one miss is the instructive part. With thinking off, the judge passed `17 × 23 = 401`:
without reasoning it cannot re-check a computation, and the reviewer prompt tells it to pass
what it cannot check rather than turn every request into two. It caught Sydney, Saturn,
"Gracias" and "91 is prime", which need recall rather than work. Thinking on caught 401 too —
at a hundred times the output tokens per review, which on a paid verifier is a hundred times
the review's output cost.

So the load-bearing assumption held, with conditions attached: a real model does emit a
parseable verdict, *if* the verdict budget fits the way the model answers, and a cheap judge's
verdicts are only as good as the checking it can afford to do.

## The first remote strong tier

Then a real remote judge: `deepseek/deepseek-v4-flash-0731` through OpenRouter, on the
`openai_compatible` backend, answering as `strong` and reviewing the local `qwen3.5:4b`. It
was the `:free` route, so nothing was billed; the tier carried the paid route's list price
($0.04 in / $0.08 out per 1M) so the counterfactual columns had something real to say.

The protocol held — a plain completion, a stream, a verified request and an escalation all
went through first time, and streamed usage arrived and was logged. It broke in three smaller
places:

1. **An unpriced cache read was free.** OpenRouter reported every prompt token of a repeated
   question as cached. `cache_read` defaulted to 0, so a tier priced on input charged nothing
   for its input and every counterfactual against it shrank — a saving invented by an omitted
   field. An omitted `cache_read` now bills at the input rate; a discount has to be written
   down to be claimed.
2. **The truncation hint named the wrong backend's knob.** A thinking judge that ran out of
   `max_verdict_tokens` was told to set `think: false` — Ollama's option, which OpenRouter
   never reads. The hint now names the knob of the verifier tier's own backend.
3. **A stream did not say which tier answered it.** `X-Router-Tier` and the score headers
   went out on plain responses only; a streaming client could not learn where its answer came
   from. Both paths send them now, and a stream also says `X-Router-Verdict: skipped`.

And the judge, on the same ten planted answers plus one real answer from `qwen3.5:4b`:

| remote judge setup | planted verdicts right | tokens per verdict | the real answer |
|---|---|---|---|
| reasoning on, `max_verdict_tokens: 200` | 10 of 10 | 33–200 | cut off unjudged, **accepted** |
| reasoning off, `max_verdict_tokens: 200` | 9 of 10 | 6–26 | judged |
| reasoning on, `max_verdict_tokens: 1024` (hardest cases) | 2 of 2 | 54–183 | judged, correctly `fail` |

The first row is the dangerous one. Ten out of ten looks like a finished judge, but the one
real answer — longer, with markdown in it — ran the judge out of budget before it wrote a
verdict, and `on_unparseable: accept` passed it. That run happened to be correct; the next run
of the same question, the cheap model claimed Australia has no single official capital, and
at 200 tokens nothing would have caught it. Reasoning off repeats the local lesson with a
different victim: it caught `401` this time and passed *"strawberry has two r's"*. A judge
that cannot think cannot count.

So `max_verdict_tokens` now defaults to **1024**. For a judge that answers directly it changes
nothing — it stops at 6–26 tokens by itself — and for a thinking one it is the difference
between a verdict and a silent pass. At this model's list price it is under a hundredth of a
cent a review.

### Still not tested

- **A paid route's invoice.** Every figure the provider returned here was $0, because every
  call was the free route. The machinery below is exercised end to end; whether the drift
  line stays inside 5% on a paid route is still a measurement nobody has made.
- **The classifier on real traffic.** Its report above comes from a planted, perfectly
  separable signal. It has never been trained on verdicts from real requests, and the number
  that comes back from real prompts will be lower — that is the number worth having. What
  stopped it here was volume, not code: `train` refuses under 40 labelled rows, and a
  free-tier OpenRouter key gives 50 requests a day across every free model, most of which
  went on the measurements above. The remote judge is now known to work, so the remaining
  step is traffic: `verification.sample_rate: 1.0`, `log.store_prompts: true`, a few hundred
  real requests, then `llm-router train`.
- **Somebody else's transcripts.** The labels here come from this router's own log, so every
  row is a genuine request. Training on other transcripts is a different job with its own trap:
  they are full of tool results, system notices and attachment placeholders that are not
  prompts at all, and a classifier will happily learn to tell those apart and report a score
  that means nothing.

## The estimate and the invoice

`cost_usd` is an estimate: the price table times the tokens the backend reported. It can be
wrong three ways — a stale price, a token class nobody priced (cache reads were free here
until a provider reported them), or tokens that never arrived. So the provider's own figure
is kept beside it, never merged into it:

| provider | what it says per request | how the router uses it |
|---|---|---|
| OpenRouter | `usage.cost`, on every response and on the last frame of a stream | `billed_cost_usd`, `billed_source: inline` |
| OpenRouter, after the fact | `GET /generation?id=` — cost and native tokens | `llm-router reconcile` |
| OpenAI, Anthropic direct | nothing; daily totals from admin-key cost endpoints | not reconcilable per row — counted as such |
| Ollama | there is no invoice | billed `0` |

A request's bill is all or nothing: a routed call, its review and its escalation are three
bills, and if any one is missing the row's bill is NULL rather than a two-thirds figure
beside a three-call estimate. `stats` compares the two only on rows that have both and
prints the drift; past 5% it says the price table is wrong before it prints a savings line.

**An abandoned stream used to leave no row at all.** Usage arrives in a stream's last frame,
so a client that hangs up first leaves nothing to cost — but it did worse than that: on
disconnect Starlette cancels the response's task group, anyio's cancellation is
level-triggered, and the await that writes the row was cancelled with everything else.
Reproduced against a real uvicorn and Ollama; the row was not written even after a clean
shutdown. The write is now shielded, the row says `499` and *client disconnected*, and keeps
the provider's id from the first frame. Then:

```bash
python -m llm_router -c config.yaml reconcile      # --dry-run to look first
```

asks the provider what that call cost and how many tokens it really used, and recomputes the
row's estimate and every counterfactual from them. Run against OpenRouter on 2026-09-19: an
abandoned stream went from zero tokens and no bill to 11 in / 1 out (the provider stopped
generating when the client left) and its reconciled bill.

What this cannot do is make a counterfactual exact. The column for a tier that never answered
is that tier's price times *this* tier's tokens; no invoice will ever exist for it.

### Checking the price table before the money is spent

Drift in `stats` shows up only after the traffic has run. The price table itself can be
checked first, against OpenRouter's public model list (no key, no quota):

```bash
python -m llm_router -c config.yaml check --prices     # --json for a script; exit 1 on any finding
```

Every OpenRouter tier is compared field by field, in USD per 1M tokens. The command reports
a price that differs (with both numbers), a charged model with no prices (costed at zero), a
model id the provider does not list, and, as a note, rates the config cannot express, such as
a higher price above 200k prompt tokens. Run on 2026-09-19 against a config written from
memory, it found a `cache_read` billed at the full input rate where the provider charges half,
a `cache_write` left out (so costed at zero), and a misspelt model id. Other providers
publish no machine-readable list and are reported as `unchecked`, not as passing.

## A baseline worth beating

Before trusting any router, price the config change it is competing with. Two models from the
same family often sit at a fixed ratio across every price axis — if the cheaper one is 60% of
the dearer one on input, output, cache read and cache write alike, then *"always use the
cheaper one"* saves exactly 40%, on any traffic mix whatsoever, for one line of YAML and no
moving parts.

A router has to beat that, not beat `always top`. The counterfactual table prints every tier
for this reason: the baseline that makes the router look worst is the honest one, and it is
already in the report.
