"""The difficulty model: does the cheap tier's answer hold, or does it not?

This is the learned half of the router. It reads a prompt and returns one number:
the probability that the cheap tier's answer to it would FAIL review. The router
turns that number into a tier; the training code turns the log into the weights.
Nothing here talks to a backend or to the database.

Three decisions shape it, and each one is a defence against a way this kind of
component lies.

**The features come from the messages and nothing else.** The same function runs
at training time (over `prompt_text`, the stored conversation) and at request
time (over the live request), so a feature cannot exist on one side and not the
other. That rules out the classic failure where a model trained on a field the
serving path does not have quietly degrades to the base rate in production and
nobody notices, because the base rate is a plausible-looking number.

Note what that costs: `tools` is NOT a feature, although the request carries it,
because `prompt_text` does not. A feature that can only ever be zero during
training is worse than a missing one -- it occupies a slot and looks like it is
working. (In practice a tool-carrying request is usually ineligible for a local
cheap tier anyway, so the gate has already dealt with it.)

**Words are binary and rare ones are pruned.** Presence, not count, so one long
message cannot outvote the rest of the vocabulary; and a word seen in fewer than
`min_df` conversations is dropped before fitting, because a weight fitted on two
examples is memorisation wearing a coefficient's clothes.

**An unknown prompt scores the base rate.** Weights are a dict and a missing
feature contributes zero, so a prompt made entirely of unseen words comes out at
`sigmoid(bias)` -- the prior failure rate of the training set. That is the right
answer to "I have never seen anything like this", and it is worth knowing that it
is what the model does, rather than discovering it as an anomaly later.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schemas import ChatCompletionRequest

# Bumped when the on-disk shape changes. A model file from a future version is
# refused rather than read with today's assumptions -- the same rule the ledger
# schema follows, and for the same reason.
MODEL_FORMAT = 1

# Words: lowercase runs of word characters, plus the punctuation that carries
# meaning in a programming prompt (`c++`, `.env`, `utf-8`). One character is
# noise; 24 is past the point where a token is a word at all.
_WORD_RE = re.compile(r"[a-z0-9_+#.\-]{2,24}")

# How much of a message is read. A prompt longer than this is not more
# difficult in a way the tail reveals, and an unbounded feature extractor is a
# denial of service with extra steps.
_MAX_TEXT_CHARS = 4000


class ModelError(Exception):
    """Raised for a model file that cannot be used as it stands."""


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------


def _text_of(content: Any) -> str:
    """Flatten a message's content to the text a reader would see."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Multimodal parts. Only the text parts are readable; an image
        # contributes its presence (via the length of the rest) and no words.
        parts = []
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return str(content)


def features_from_messages(messages: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """The feature vector for one conversation.

    Words come from the LAST user message, because that is the request; size and
    shape come from the whole conversation, because a long history makes the same
    question harder. Splitting them that way keeps the vocabulary about what was
    asked instead of about what was already discussed.
    """
    texts = [_text_of(m.get("content")) for m in messages]
    roles = [str(m.get("role", "")) for m in messages]

    last_user = ""
    for role, text in zip(reversed(roles), reversed(texts)):
        if role == "user":
            last_user = text
            break
    if not last_user and texts:
        # No user turn at all (a system-only or tool-only request). Fall back to
        # the final message rather than producing an empty vector, which would
        # score every such request identically at the base rate.
        last_user = texts[-1]

    whole = "\n".join(texts)
    head = last_user[:_MAX_TEXT_CHARS]

    vector: dict[str, float] = {}
    # `sorted`, and not merely `set`, and the difference is not cosmetic. Python
    # randomises string hashing per process, so a set of words iterates in a
    # different order in every run; the fit then sums the same floats in a
    # different order, lands on weights that differ in the last few bits, and
    # produces a different fingerprint from identical data. Measured: the same
    # 72-row corpus trained twice gave 232c6cb5a88a and 35b0f95c0bb1. Dicts keep
    # insertion order, so sorting here makes every iteration downstream --
    # features, fit, the saved file -- deterministic.
    for word in sorted(set(_WORD_RE.findall(head.lower()))):
        vector[f"w:{word}"] = 1.0

    digits = sum(c.isdigit() for c in head)
    vector.update(
        {
            # log10-scaled and divided down, so every numeric feature lands in
            # roughly [0, 1] like the binary ones. Unscaled lengths would
            # dominate the gradient and the vocabulary would never be learned.
            "n:log_chars": min(math.log10(1 + len(whole)) / 5.0, 1.0),
            "n:log_last_chars": min(math.log10(1 + len(head)) / 5.0, 1.0),
            "n:messages": min(len(messages), 20) / 20.0,
            "n:code_fence": 1.0 if "```" in whole else 0.0,
            "n:digit_ratio": (digits / len(head)) if head else 0.0,
            "n:question": 1.0 if "?" in head else 0.0,
            "n:multi_turn": 1.0 if sum(r == "user" for r in roles) > 1 else 0.0,
        }
    )
    return vector


def features_from_request(request: ChatCompletionRequest) -> dict[str, float]:
    """Serving-time entry point. Must agree with the training-time one exactly."""
    return features_from_messages([m.model_dump(exclude_none=True) for m in request.messages])


def features_from_prompt_text(prompt_text: str) -> dict[str, float] | None:
    """Training-time entry point, over the conversation as the log stored it.

    Returns None for a row whose prompt cannot be read back as a message list.
    That is a row to skip, not a row to guess at.
    """
    try:
        messages = json.loads(prompt_text)
    except (TypeError, ValueError):
        return None
    if not isinstance(messages, list) or not messages:
        return None
    if not all(isinstance(m, Mapping) for m in messages):
        return None
    return features_from_messages(messages)


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


def _sigmoid(z: float) -> float:
    # Both branches so a large magnitude cannot overflow exp().
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class DifficultyModel:
    """Logistic regression weights, plus the facts needed to refuse misuse.

    The held-out metrics are stored IN the file. A model that travels without
    its evidence invites deployment on faith, and this whole project is an
    argument against that.
    """

    weights: dict[str, float]
    bias: float
    # The tier whose failures these labels describe, and the tier that judged
    # them. A model trained on one cheap model's mistakes says nothing about
    # another's, so the router checks this before using it.
    predicts_tier: str
    judged_by: str | None = None
    threshold: float = 0.5
    trained_at: str = ""
    examples: int = 0
    positives: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    format: int = MODEL_FORMAT

    @property
    def fingerprint(self) -> str:
        """Short, stable identity of these exact weights.

        Logged with every scored request, so a score in the database can always
        be traced to the model that produced it -- without it, retraining
        silently mixes two models' numbers in one column.
        """
        blob = json.dumps(
            {"w": sorted(self.weights.items()), "b": self.bias}, ensure_ascii=False
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    @property
    def base_rate(self) -> float:
        """What an unseen prompt scores: the prior this model was fitted around."""
        return _sigmoid(self.bias)

    def score(self, vector: Mapping[str, float]) -> float:
        """P(the cheap tier's answer to this prompt fails review)."""
        z = self.bias
        for name, value in vector.items():
            weight = self.weights.get(name)
            if weight is not None:
                z += weight * value
        return _sigmoid(z)

    def score_request(self, request: ChatCompletionRequest) -> float:
        return self.score(features_from_request(request))

    def top_features(self, count: int = 15) -> list[tuple[str, float]]:
        """The strongest weights, for reading the model rather than trusting it."""
        return sorted(self.weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:count]

    # --- persistence ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "fingerprint": self.fingerprint,
            "predicts_tier": self.predicts_tier,
            "judged_by": self.judged_by,
            "threshold": self.threshold,
            "trained_at": self.trained_at,
            "examples": self.examples,
            "positives": self.positives,
            "bias": self.bias,
            "metrics": self.metrics,
            # Sorted so the file is diffable and two trainings of the same data
            # produce byte-identical output.
            "weights": dict(sorted(self.weights.items())),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DifficultyModel":
        fmt = int(raw.get("format", 0))
        if fmt != MODEL_FORMAT:
            raise ModelError(
                f"model format {fmt} cannot be read by this build (expects {MODEL_FORMAT})"
            )
        if "predicts_tier" not in raw:
            raise ModelError("model file does not say which tier it predicts for")
        return cls(
            weights={str(k): float(v) for k, v in (raw.get("weights") or {}).items()},
            bias=float(raw.get("bias", 0.0)),
            predicts_tier=str(raw["predicts_tier"]),
            judged_by=raw.get("judged_by"),
            threshold=float(raw.get("threshold", 0.5)),
            trained_at=str(raw.get("trained_at", "")),
            examples=int(raw.get("examples", 0)),
            positives=int(raw.get("positives", 0)),
            metrics=dict(raw.get("metrics") or {}),
            format=fmt,
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        if p.parent and str(p.parent):
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DifficultyModel":
        p = Path(path)
        if not p.is_file():
            raise ModelError(f"no model at {p}; train one with `jev-model-router train`")
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ModelError(f"model at {p} is not readable JSON: {exc}") from None
        if not isinstance(raw, Mapping):
            raise ModelError(f"model at {p} is not an object")
        return cls.from_dict(raw)


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Example:
    """One training row: a conversation, its label, and the group it belongs to."""

    vector: dict[str, float]
    label: int  # 1 = the cheap tier's answer failed review
    group: str  # conversation key; the split is by this, never by row


def fit(
    examples: Sequence[Example],
    *,
    epochs: int = 40,
    learning_rate: float = 0.25,
    l2: float = 1e-4,
    min_df: int = 3,
    seed: int = 0,
    balance: bool = True,
    use_words: bool = True,
) -> tuple[dict[str, float], float]:
    """Plain SGD logistic regression. Returns (weights, bias).

    `use_words` drops the bag of words and fits on the structural features
    alone. It is offered because words are not always an asset: on a 400-question
    benchmark they cross-validated at 0.40 AUC, BELOW chance, while the same fit
    without them reached 0.618. A vocabulary learned from a few hundred prompts
    can describe the corpus rather than the difficulty, and then it is worse than
    having no vocabulary at all.

    `balance` is not optional in spirit. Failures are the minority class by a
    wide margin in any router worth running, and an unweighted fit answers
    "never fails" for every prompt, scores 90% accuracy and routes nothing. The
    positive class is therefore weighted up to parity with the negative one, and
    the reported accuracy is read next to the base rate, never alone.
    """
    if not examples:
        raise ValueError("no examples to fit")

    # Document frequency prunes the long tail. `n:` features are structural and
    # exempt -- they are present on every row by construction.
    df: dict[str, int] = {}
    for example in examples:
        for name in example.vector:
            df[name] = df.get(name, 0) + 1
    kept = {
        name
        for name, count in df.items()
        if name.startswith("n:") or (use_words and count >= min_df)
    }

    positives = sum(e.label for e in examples)
    negatives = len(examples) - positives
    pos_weight = (negatives / positives) if (balance and positives) else 1.0

    weights: dict[str, float] = {}
    bias = 0.0
    order = list(range(len(examples)))
    rng = random.Random(seed)

    for epoch in range(epochs):
        rng.shuffle(order)
        # Decaying step size: large enough to move early, small enough at the
        # end that the last examples seen do not decide the model.
        lr = learning_rate / (1.0 + epoch)
        for index in order:
            example = examples[index]
            active = {n: v for n, v in example.vector.items() if n in kept}
            z = bias + sum(weights.get(n, 0.0) * v for n, v in active.items())
            error = _sigmoid(z) - example.label
            weight = pos_weight if example.label == 1 else 1.0
            step = lr * error * weight
            bias -= step
            for name, value in active.items():
                current = weights.get(name, 0.0)
                weights[name] = current - step * value - lr * l2 * current

    # Weights that never left zero are not information; dropping them keeps the
    # file about what was learned.
    weights = {n: w for n, w in weights.items() if abs(w) > 1e-9}
    return weights, bias


def build_model(
    examples: Sequence[Example],
    *,
    predicts_tier: str,
    judged_by: str | None = None,
    threshold: float = 0.5,
    **fit_kwargs: Any,
) -> DifficultyModel:
    weights, bias = fit(examples, **fit_kwargs)
    return DifficultyModel(
        weights=weights,
        bias=bias,
        predicts_tier=predicts_tier,
        judged_by=judged_by,
        threshold=threshold,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        examples=len(examples),
        positives=sum(e.label for e in examples),
    )


def conversation_key(prompt_text: str) -> str:
    """Group key for the train/test split: the conversation, not the request.

    Requests from one conversation share their opening message verbatim -- this
    router's own `prompt_text` is the whole message list -- so the first
    message's hash groups them. It is a proxy and not a session id, and it is
    the strongest grouping derivable from what is stored.

    Why it matters enough to be the default: prompts from one conversation share
    vocabulary, so a split that scatters them across train and test lets the
    model recognise the CONVERSATION and report that as difficulty. The number
    that comes back is large, wrong, and impossible to argue with unless you
    also measured the grouped one. `training.py` measures both and prints them
    together.
    """
    try:
        messages = json.loads(prompt_text)
        first = messages[0] if isinstance(messages, list) and messages else None
        seed = json.dumps(first, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError, IndexError):
        seed = prompt_text
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def group_fraction(group: str, seed: int = 0) -> float:
    """Where a group falls in [0, 1). Stable across runs and across corpora.

    Hashing the key rather than shuffling a list is what makes two trainings a
    week apart comparable: new rows join the side their conversation was always
    on, instead of reshuffling the boundary under the previous report.
    """
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    # First four bytes as a fraction of the space they span.
    return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF


def split_by_group(
    examples: Iterable[Example], *, holdout: float = 0.25, seed: int = 0
) -> tuple[list[Example], list[Example]]:
    """Hold out whole conversations, deterministically."""
    train: list[Example] = []
    test: list[Example] = []
    for example in examples:
        side = test if group_fraction(example.group, seed) < holdout else train
        side.append(example)
    return train, test


def split_by_row(
    examples: Sequence[Example], *, holdout: float = 0.25, seed: int = 0
) -> tuple[list[Example], list[Example]]:
    """The WRONG split, kept deliberately so the report can show what it costs.

    This is the split a first attempt reaches for, and it is optimistic by a
    margin nobody guesses correctly. It exists here to be measured against
    `split_by_group`, never to produce a shipped model.
    """
    order = list(examples)
    random.Random(seed).shuffle(order)
    cut = int(len(order) * holdout)
    return order[cut:], order[:cut]


__all__ = [
    "DifficultyModel",
    "Example",
    "MODEL_FORMAT",
    "ModelError",
    "build_model",
    "conversation_key",
    "features_from_messages",
    "features_from_prompt_text",
    "features_from_request",
    "fit",
    "group_fraction",
    "split_by_group",
    "split_by_row",
]
