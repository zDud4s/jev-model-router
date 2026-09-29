"""What the subscription CLIs share, for the backends that answer through them and the delegates that hand them work.

Kept in one place so the proxy and `jev-model-router delegate` cannot drift on either list.
"""

from __future__ import annotations

from typing import Mapping

# Removed from a CLI's environment, per runner. With an API key present the CLI bills per token
# instead of using the subscription login, and nothing in its output says so. CLAUDECODE is set
# inside a Claude Code session, and `claude` refuses to start nested while it is.
DROP_ENV: dict[str, tuple[str, ...]] = {
    "claude": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE"),
    "codex": ("OPENAI_API_KEY", "CODEX_API_KEY"),
}

# Words either CLI uses when the subscription window is spent. Both can say so in an error text
# with no HTTP status, so the text is all there is to go on.
# Not a bare "limit reached": "context length limit reached" is no spent window, and calling it one
# would lock the subscription for a whole window.
LIMIT_WORDS = ("rate limit", "usage limit", "hour limit", "weekly limit", "quota", "too many requests")


def scrubbed(base: Mapping[str, str], runner: str) -> dict[str, str]:
    """A copy of `base` without the variables `runner`'s CLI must not see."""
    env = dict(base)
    for key in DROP_ENV.get(runner, ()):
        env.pop(key, None)
    return env


def says_limited(text: str | None) -> bool:
    low = (text or "").lower()
    return any(word in low for word in LIMIT_WORDS)


__all__ = ["DROP_ENV", "LIMIT_WORDS", "says_limited", "scrubbed"]
