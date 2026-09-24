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

### A judge that is not a model

The loop's cost is a second full generation per checked request, and that is what makes it
expensive enough to be off by default. `backend: jev` is the other shape a judge can have:
TypeSafe's Jev is not an LLM, it answers a typed yes/no question with one calibrated
probability, and it charges $0.042 per 1M input tokens with the output free — roughly a
hundredth of what the `mid` tier costs to read the same review.

It is a backend and not a new code path, which is the point. A verdict is a boolean, the loop
already knows how to pick a judge, gate it, bill it and log it, and all of that is keyed on a
tier — so Jev is a tier whose adapter happens to answer in one number. `verification.py` was
not changed to support it.

Three things fall out of a judge that returns a probability rather than prose:

- **There is no unparseable verdict.** The failure this project measured twice — a thinking
  judge spending its whole budget reasoning, writing nothing, and `on_unparseable: accept`
  passing an answer nobody judged — is unreachable here. A response with no probability in it
  is raised as a backend error, where `on_verifier_error` decides, and never rendered as a pass.
- **The evidence outlives the verdict.** The probability is written into the verdict line, so
  a recorded run can be re-thresholded afterwards. Moving the line on a text judge means
  paying for the whole run again.
- **A judge is never a counterfactual.** A Jev tier is ineligible to serve every request, with
  reason `cannot_generate`. Without that it would sit in the counterfactual table at $0.042
  per 1M as the cheapest way to have answered everything in the log, which it was not — it was
  not a way of answering anything. Naming one as `default_tier`, in `model_map`, or as
  `escalate_to` is refused at startup, including the case where `escalate_to` would have
  silently *defaulted* to a verifier that cannot answer.

**Whether it is any good at the job is a separate question, and it is now measured.** Jev is a
System One model: one pass, 70-500ms, no reasoning first. The table two sections down measures
what happens to a judge that cannot think - 9 of 10 planted verdicts right, and the one it
missed was *"strawberry has two r's"*. That was the prior. `scripts/judge_eval.py` reviews
sixteen answers whose truth is known, through the real prompt builder, the real backend and
the real parser:

```
python scripts/judge_eval.py -c config.yaml --tier judge     # jev
python scripts/judge_eval.py -c config.yaml --tier cheap     # a text judge, for comparison
```

It prints accuracy beside the always-PASS baseline, recall on the wrong answers separately -
a judge that has learned to say PASS scores well on the first and near zero on the second -
the count of unparseable verdicts, the cost, the latency, and, for a judge that reports a
probability, a threshold sweep over the run it just did.

#### What sixteen cases said, 2026-09-21

| | `cheap` (qwen3.5:4b, local) | `jev-latest` | `jev-preview` |
|---|---|---|---|
| accuracy | **0.938** | 0.875 | 0.812 |
| recall on wrong answers | **0.889** | 0.778 | 0.667 |
| precision on FAIL | 1.000 | 1.000 | 1.000 |
| unparseable verdicts | 1 | **0** | **0** |
| median latency | 143,000 ms | **282 ms** | 269 ms |
| cost, 16 verdicts | $0 | $0.000311 | $0.000311 |

Sixteen cases is a demonstration and not a tuning set, and the accuracy column is the least
interesting one in it. Four things there are worth more than the ranking.

**The local judge wins on accuracy and cannot be used.** 143 seconds a verdict is not a
verification loop, it is a batch job. Jev answers in 282 ms, about 500x faster, at $0.0000194
a verdict. The local tier's $0 is real, and its latency is what that $0 costs.

**Jev misses exactly where a System One model was predicted to miss, and misses confidently.**
Both failures are the same shape: an answer that needs a computation to check. Case 11 - right
method, wrong last step - came back at p=0.93 and p=0.94 on two runs. No threshold rescues
that. Catching it needs a line above 0.94, and the sweep prices the attempt: accuracy falls to
0.688 at 0.95, because the *correct* answers also sit at 0.94-0.95. This project already wrote
down that a judge which cannot think cannot count; this is the same sentence with a number
against it.

**The probabilities are stable, and that is a real difference from the classifier.** Two
identical runs moved every probability by at most 0.04, median 0.01, and flipped zero verdicts.
The project's own classifier gave *both* outcomes on 22% of questions across three samples of
the same text. Jev is not sampling, and it shows. The caveat is that same number read the other
way: a case within 0.04 of the threshold is a coin toss, and case 15 is that case - 0.46 and
0.49 on `jev-latest`, 0.50 on `jev-preview`, which is where `preview` lost it.

**`jev-preview` is worse, and its own catalog entry says it "should be better in most ways".**
0.812 against 0.875, one verdict lost. A vendor's description of an unreleased version is not a
measurement, and checking cost one command.

So the trade is latency against recall, not quality against price: Jev is not better than the
incumbent at judging and is three orders of magnitude faster at it. Whether 0.778 recall at
282 ms beats 0.889 at 143 seconds depends on what a missed bad answer costs, which is a number
this repo does not have and real traffic would supply.

One wire note, because it cost an afternoon and a reader will hit it too: `model` is a
**required** field in the request body, and TypeSafe's written guide says the opposite - that
the endpoint names its own model. The live API answers `422 missing: body.model`, and the
service publishes its own schema at `/openapi.json`, which settles it. The accepted names come
from `GET /v1/models` and are `jev-latest` and `jev-preview`; plain `jev` is rejected.

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

### Labels without a judge

A judge costs one call per label. A question set with an answer key costs nothing, and is
the more reliable grader of the two — a judge that cannot afford to reason was measured
missing arithmetic. `label` takes GSM8K's JSONL:

```bash
python -m llm_router -c config.yaml label --gsm8k test.jsonl --db gsm8k.db --limit 400
python -m llm_router -c config.yaml train --db gsm8k.db --out classifier.json
```

Labelling asks at **temperature 0** by default (`--temperature` to change it, `None` in the
API to leave it to the provider). That default was bought the expensive way — see below —
and it is the difference between a label that records the question and one that records the
sample.

Each question goes through the router in process, so the row is the row real traffic would
write; the reply's final `ANSWER:` line is compared with the key (no such line is a fail, not
a guess at the last number in the text), and the verdict is stored with `verifier_tier`
`ground_truth:gsm8k`. It refuses to write to the serving log — `stats` would count those rows
as traffic — and a rerun resumes where the last one stopped.

What it measures is whether the features can tell a hard word problem from an easy one. That
is a test of the classifier, not a model of your traffic: routing real requests on it is a
claim only real verdicts can support.

**A failure that is unfinished is counted apart from one that is wrong.** A reply with no
`ANSWER:` line may be a model that cannot do the arithmetic, or a model that was cut off at
the tier's output budget, and those are not the same defect: the first is difficulty, the
second is a number in the config. When most of a corpus's failures are the second, the label
report says so and tells you to raise the budget and label again before training — because a
classifier fitted there learns to predict *how long an answer will be*, and the router
already fails an unfinished answer without a model. `train` makes the same objection from the
other side: a corpus whose failures were mostly failed without a review draws a warning.

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

### Then the corpus turned out not to be a corpus

All 60 of those unfinished answers stopped at *exactly* 2048 output tokens, and no correct
answer in the run came within five tokens of it. That is a cap being hit, not a coincidence,
so the 60 were asked again with the budget doubled to 4096. 115 minutes, still $0:

| | |
|---|---|
| passed on the second ask | **33** |
| still unfinished at 4096 | 17 |
| finished, and wrong | 10 |

And then the part that matters. **27 of those 33 recovered answers finished in under 2048
tokens** — several in under 300. The old budget was never binding on them: the same model,
the same question, a different sample, and it simply answered. Only 6 of 60 were genuinely
bought by the extra room.

So the dominant failure mode in this corpus is neither difficulty nor the budget. It is
**run-to-run variance in the cheap model** — a question that runs away to 2048 tokens on one
sample and answers in 249 on the next. That bounds what any classifier reading the *prompt*
can ever do here, and no amount of feature engineering escapes it: an outcome that changes
between two samples of the same question is not a property of the question. It is also the
better explanation for the 0.590 AUC than "the features are weak".

It points somewhere, though. If the answer varies, the variance is visible in the answers —
ask twice and see whether they agree. That reads the generation instead of the prompt, costs
two cheap calls instead of one strong one, and was the next thing to measure.

### Asking three times, and what was left to catch

40 questions — 20 the first run passed, 20 it failed — asked three more times each at the
original budget. 100 minutes, still $0.

**The label is not a property of the question.** 9 of the 40 questions (22%) produced *both*
outcomes across three samples of the same text. Of the 20 the first run failed, 47% of the
resamples passed; of the 20 it passed, 12% of the resamples failed. Nearly half of what that
corpus called a failure was the dice.

Then "the two samples state different answers" as a failure predictor, over all 240 ordered
pairs: **recall 0.974, precision 0.809, lift 2.49x, accuracy 0.917 against a 0.675 baseline**.
The first mechanism in this project to beat "always cheap" — and it is not what it looks like.

Decomposed, 74 of the 78 wrong answers were ones where the served sample **never finished**.
That is the free rule from two sections up: `finish_reason: length`, no second call, no
comparison. Among the 166 pairs whose served sample actually stated an answer, only **4 were
wrong — 2.4%** — and comparing two samples caught 2 of those 4 while escalating 12% of the
traffic, at a precision of 0.100.

So self-consistency is not what is working here. What is working is a rule that was already
free, and once it runs there is almost nothing left: a 4-billion-parameter model that
finishes a GSM8K question is right about 97.6% of the time (roughly 1%–6% wrong, on 4 errors
out of 166, and the questions are clustered so even that is optimistic). No classifier and no
second sample can beat a ceiling of 2.4%, so on this workload the answer is **do not ship
one**. Fix the generation, catch the runaways for nothing, retry them at the same tier, and
keep the money.

That is a negative result about two mechanisms and a positive one about a third. It cost
about four hours of a laptop GPU and nothing else, which is the argument for having the
labelling path at all.

#### `retry_unfinished`: ask again before paying more

The first half of that is already worth shipping, because the check is free and so is knowing
whether it worked. `verification.retry_unfinished: true` re-asks the **same** tier once when
an answer was failed without a review, and escalates only if the second answer runs away too:

```yaml
verification:
  enabled: true
  verifier_tier: strong
  retry_unfinished: true
```

It is off by default because whether it pays is arithmetic rather than a preference. A retry
is worth it when

> cheap tier price < (retry success rate) × (strong tier price)

At the 45% success rate measured here, a local cheap tier costing a fiftieth of the strong one
is free money; a cheap tier at half the strong one's price is a loss. A retry that fails is
still billed — its tokens are added to the escalation's rather than overwritten by them, so
the row shows both calls.

### What the corpus then taught the trainer

Four of those numbers were the trainer's fault rather than the data's, and each one is now
part of what `train` prints:

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
  set is called out as such. `--target-recall 0.8` is the same knob priced in quality rather
  than traffic — *catch 80% of the failures* — which is the number an operator usually has,
  and on a weak model the exchange rate between the two is worth seeing: on this corpus,
  catching 70.6% of the held-out failures costs escalating **69.8%** of the traffic, at which
  point the lift is 1.01x and the model is a coin.
- **The cost table stopped one step short of a decision.** It prints money and bad answers in
  separate columns and refuses to invent an exchange rate, which is right — but the operator
  has one. `--bad-answer-cost 0.05` adds a total to both the policy table and the sweep and
  marks the cheapest row, over a price the report names as *yours* every time it prints it.
- **The AUC was printed with no error bar.** 0.590 on 96 held-out requests containing 17
  failures is compatible with anything from 0.436 to 0.745, so the honest reading of the
  whole exercise is *this corpus cannot tell the model from chance*. Every AUC now carries a
  95% interval (Hanley–McNeil, checked against a 2000-resample bootstrap that gave
  [0.424, 0.743] where the formula gives [0.436, 0.745]), and an interval spanning 0.500 is a
  warning. It is also the cheapest demonstration that the split matters: the same fit scored
  against the *wrong*, row-wise split reads 0.737 with an interval of [0.600, 0.874] — clear
  of chance, and entirely an artefact.

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
`GET /healthz`, `POST /v1/route` and `POST /v1/route/{id}/outcome` (below), and the live
routing view at `GET /routing`.

### Route only: for a caller that runs the model itself

An agent runner cannot hand its work to a proxy: the work is many turns of tools in a
worktree, and the runner spawns the CLI that does it. It asks which model and effort
instead, runs it, and reports what its gate said. Nothing is run here, no quota is spent;
with `router.kind: capabilities` a decision is one Jev call.

```http
POST /v1/route
{
  "task": "Fix the race in core/src/scheduler.rs: a job runs twice after a restart",
  "stage": "implement",                      // optional, shown to Jev and logged
  "files": ["core/src/scheduler.rs"],        // optional
  "attempt": 2,                              // optional
  "gate_output": "...",                      // optional: why the last attempt failed (tail kept)
  "failed": ["claude-sonnet-5@low"],         // optional: tier names or model[@effort] that failed it
  "runners": ["claude", "codex"],            // optional: only tiers these CLIs run
  "packet": {"verifiable": true}             // optional: any other context, as-is
}
-> 200 {"decision_id": "rt_...", "runner": "claude", "model": "claude-opus-5-5",
        "effort": "medium", "tier": "claude:claude-opus-5-5@medium", "success": 0.83,
        "estimated_cost_usd": 0.018, "rule": "...", "router": "capabilities:...",
        "unknown_failed": []}
-> 400 a malformed body; 422 no eligible tier runs on the runners given
```

A retry names what failed, and the router never offers a tier it rates below it. The
decision is written to `route_decisions`, a table of its own: nothing was called, and a row
of zeros among `requests` would read as a free request in every cost report.

```http
POST /v1/route/{decision_id}/outcome
{"status": "pass" | "fail" | "rate_limited" | "error",
 "detail": "optional text", "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
-> 200; 404 unknown decision; 409 an outcome was already reported (the first one is kept)
```

`pass`/`fail` is the label `llm-router calibrate --from-log` learns each model family's
scale from; `error` means the run broke for a reason that says nothing about the model, and
is not learnt from. `rate_limited` takes that subscription off the table until its window
turns -- the router cannot see a 429 on a call it did not make.

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

### 1024 judges arithmetic, and does not always judge prose

The first run of `scripts/judge_eval.py` found the same failure again, in a place the earlier
measurements could not have looked. The local `qwen3.5:4b`, reasoning on, at the 1024 this
section had just settled on:

| | |
|---|---|
| accuracy | 0.938 (15/16) |
| always-PASS baseline | 0.438 |
| recall on the wrong answers | 0.889 (8 of 9 caught) |
| precision on FAIL | 1.000 (8 of 8 flags real) |
| unparseable | **1** |
| cost | $0 |
| latency | 143s median |

The single miss was not a wrong verdict. Asked to review a confident and wrong one-sentence
description of a mutex, the judge spent the entire 1024 reasoning and emitted **no content at
all** — empty answer, `finish_reason: length` — and `on_unparseable: accept` would have shipped
the wrong answer as a pass. It reproduced on a re-run, and the same case answered correctly, as
`FAIL`, at 4096.

What made that case different is the thing the earlier numbers could not vary: every planted
answer behind the 490–1054 figure had arithmetic or a checkable fact in it, and that is what
lets the reasoning stop. An open question gives it nothing to stop on. So `1024` was not
measured too loosely — it was measured on a corpus that could not contain this, which is the
same shape of mistake as fitting a classifier on GSM8K and calling the result general.

The default stays 1024: one case on one model is not a curve, and the honest report of it is
this paragraph rather than a changed number. Raise it if your judge thinks and your traffic is
open-ended.

It also sharpens what a judge like Jev is actually for. The failure here is not ignorance —
the model knew the answer was wrong and said so with more room. It is **reasoning that does not
terminate**, and a System One judge cannot have it: one pass, no chain, always a probability.
That is not a risk Jev reduces. It is a risk that does not exist in it.

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
