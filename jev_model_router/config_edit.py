"""Edit the config file from the routing page: validate, preview, write.

The page sends either the whole file (the YAML view) or a list of changes
(the form): `{"path": ["verification", "sample_rate"], "value": 0.5}`, or
`"delete": true`. Changes are applied to the TEXT, line by line, so every
comment the file carries survives a form save. A change that cannot be made
that way (a key written inline, `{a: 1}`) is refused with a pointer to the YAML
view, never approximated: after editing, the new text is parsed again and must
equal the old data with exactly those changes applied.

Nothing here reloads the proxy. The config is read once at startup, so a saved
file is served from the next `serve`; the page says so.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from .calibration import _replace
from .config import _FALLBACK_POLICIES, _JEV_FAILURE_MODES, _RULES, _SUBSCRIPTION_BACKENDS, EFFORTS, ConfigError, parse_config

BACKENDS = ("ollama", "openai_compatible", "jev", "claude_cli", "codex_cli")
ROUTER_KINDS = ("static", "classifier", "capabilities")


class EditError(ValueError):
    """A change the text editor will not make in place."""


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def options() -> dict[str, Any]:
    """The choices the form offers, read from the parser's own constants."""
    return {
        "backends": list(BACKENDS),
        "subscription_backends": sorted(_SUBSCRIPTION_BACKENDS),
        "efforts": list(EFFORTS),
        "router_kinds": list(ROUTER_KINDS),
        "rules": list(_RULES),
        "fallback_policies": list(_FALLBACK_POLICIES),
        "jev_failure_modes": list(_JEV_FAILURE_MODES),
    }


def sources_on_offer(data: Any) -> list[dict[str, Any]]:
    """The providers `init` offers (data/setup.yaml) that the file does not discover yet, as this machine sees them.

    Each carries the `discover` entry to write: a CLI's executable resolved, the
    init-only keys dropped. Nothing here names a provider; the list is data.
    """
    from . import setup_wizard

    setup = setup_wizard.load_setup()
    caps = ((data or {}).get("router") or {}).get("capabilities") if isinstance(data, dict) else None
    have = set((caps or {}).get("discover") or {}) if isinstance(caps, dict) else set()
    offered = {name: source for name, source in (setup.get("sources") or {}).items() if name not in have}
    out = []
    for found in setup_wizard.detect({"sources": offered}):
        entry = {k: v for k, v in offered[found.name].items() if k not in setup_wizard._INIT_ONLY}
        if entry.get("backend") in _SUBSCRIPTION_BACKENDS and found.where:
            entry["base_url"] = found.where
        out.append({"name": found.name, "label": found.label, "ready": found.found, "note": found.note, "entry": entry})
    return out


def check(text: str) -> dict[str, Any]:
    """Parse and validate `text` as the router would at startup; never raises."""
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        return {"ok": False, "error": f"YAML: {exc}"}
    try:
        config = parse_config(raw)
    except ConfigError as exc:
        return {"ok": False, "error": str(exc)}
    except (TypeError, ValueError, AttributeError) as exc:
        # parse_config expects the shapes it documents; a list where a mapping
        # belongs surfaces here rather than as a ConfigError.
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "error": None, "tiers": sorted(config.tiers), "router": config.router.kind}


def diff(old: str, new: str, name: str = "config") -> str:
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{name}", f"b/{name}"))


def write(path: str | Path, text: str, base: str | None) -> None:
    """Validate and write atomically; refuse when the file moved since `base` (its digest) was read."""
    target = Path(path)
    current = target.read_text(encoding="utf-8")
    if base is not None and digest(current) != base:
        raise EditError(f"{target} changed on disk since it was loaded; reload before saving")
    result = check(text)
    if not result["ok"]:
        raise ConfigError(f"{target}: {result['error']}")
    _replace(target, text)


# ------------------------------------------------------------------ changes

def apply_changes(text: str, changes: list[dict[str, Any]]) -> str:
    """Apply form changes to the text, keeping comments; refuse what cannot be done exactly."""
    expected = yaml.safe_load(text) or {}
    lines = text.splitlines()
    for change in changes:
        path = change.get("path")
        if not isinstance(path, list) or not path or not all(isinstance(k, str) and k for k in path):
            raise EditError(f"a change needs a non-empty path of keys, got {path!r}")
        if change.get("delete"):
            _delete(lines, path)
            _remove(expected, path)
        else:
            if "value" not in change:
                raise EditError(f"change to {'.'.join(path)} has neither value nor delete")
            _set(lines, path, change["value"])
            _put(expected, path, copy.deepcopy(change["value"]))
    out = "\n".join(lines) + ("\n" if lines else "")
    if (yaml.safe_load(out) or {}) != expected:
        raise EditError("the form could not make that change in place; use the YAML view")
    return out


def _put(data: dict[str, Any], path: list[str], value: Any) -> None:
    for key in path[:-1]:
        if not isinstance(data.get(key), dict):
            data[key] = {}
        data = data[key]
    data[path[-1]] = value


def _remove(data: dict[str, Any], path: list[str]) -> None:
    for key in path[:-1]:
        data = data.get(key) if isinstance(data, dict) else None
        if not isinstance(data, dict):
            return
    data.pop(path[-1], None)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _content(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def _key_re(key: str) -> re.Pattern[str]:
    quoted = "|".join(re.escape(q) for q in (key, f'"{key}"', f"'{key}'"))
    return re.compile(rf"^( *)(?:{quoted}) *:(?= |$)")


def _block_end(lines: list[str], start: int, indent: int) -> int:
    """The first line after `start` whose content is at `indent` or less."""
    i = start + 1
    while i < len(lines) and not (_content(lines[i]) and _indent(lines[i]) <= indent):
        i += 1
    return i


def _split_value(line: str) -> tuple[str, str, str]:
    """(head up to and including the colon, the inline value, a trailing comment with its spacing)."""
    colon = re.match(r"^ *(?:\"[^\"]*\"|'[^']*'|[^'\"#][^#]*?) *:(?= |$)", line)
    head = line[: colon.end()] if colon else line
    rest = line[len(head):]
    value, comment, quote = rest, "", None
    for i, ch in enumerate(rest):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'" and not rest[:i].strip():
            quote = ch
        elif ch == "#" and (i == 0 or rest[i - 1] == " "):
            value, comment = rest[:i], rest[i:]
            break
    stripped = value.rstrip()
    return head, stripped.strip(), value[len(stripped):] + comment if comment else ""


def _find(lines: list[str], path: list[str]) -> tuple[int, int] | None:
    """The line and indent of the key at `path`, or None."""
    lo, hi, parent = 0, len(lines), -1
    at = None
    for depth, key in enumerate(path):
        pattern = _key_re(key)
        child = next((_indent(lines[i]) for i in range(lo, hi) if _content(lines[i])), None)
        if child is None or child <= parent:
            return None
        at = next((i for i in range(lo, hi) if _content(lines[i]) and _indent(lines[i]) == child
                   and pattern.match(lines[i])), None)
        if at is None:
            return None
        if depth < len(path) - 1:
            _, inline, _ = _split_value(lines[at])
            if inline and inline not in ("~", "null", "{}"):
                raise EditError(f"{'.'.join(path[:depth + 1])} is written inline; edit it in the YAML view")
        lo, hi, parent = at + 1, _block_end(lines, at, child), child
    return at, parent


def _render_scalar(value: Any) -> str:
    if isinstance(value, (dict, list)) and not value:
        return "{}" if isinstance(value, dict) else "[]"
    text = yaml.safe_dump(value, default_flow_style=True, width=10**6, allow_unicode=True).strip()
    return text[:-4].rstrip() if text.endswith("\n...") or text.endswith(" ...") else text.removesuffix("...").rstrip()


def _render_block(value: dict[str, Any], indent: int) -> list[str]:
    dumped = yaml.safe_dump(value, default_flow_style=False, sort_keys=False, width=10**6, allow_unicode=True)
    return [" " * indent + line for line in dumped.splitlines()]


def _set(lines: list[str], path: list[str], value: Any) -> None:
    found = _find(lines, path)
    if found is None:
        _insert(lines, path, value)
        return
    at, indent = found
    head, inline, comment = _split_value(lines[at])
    end = _block_end(lines, at, indent)
    # A value that was a block (children, or a `|` scalar) is replaced whole.
    del lines[at + 1:end if (not inline or inline[0] in "|>") else at + 1]
    if isinstance(value, dict) and value:
        lines[at] = head + comment.rstrip()
        lines[at + 1:at + 1] = _render_block(value, indent + _step(lines))
    else:
        lines[at] = f"{head} {_render_scalar(value)}{comment}"


def _step(lines: list[str]) -> int:
    """The file's own indentation step, read from the first nested key; two when there is none."""
    content = [line for line in lines if _content(line)]
    for line, nxt in zip(content, content[1:]):
        if line.rstrip().endswith(":") and _indent(nxt) > _indent(line):
            return _indent(nxt) - _indent(line)
    return 2


def _insert(lines: list[str], path: list[str], value: Any) -> None:
    parent = _find(lines, path[:-1]) if len(path) > 1 else None
    if len(path) > 1 and parent is None:
        # Build the missing parent first, as an empty block, then fill it.
        _insert(lines, path[:-1], {})
        parent = _find(lines, path[:-1])
    if parent is None:
        lo, hi, indent = 0, len(lines), 0
    else:
        at, own = parent
        head, inline, comment = _split_value(lines[at])
        if inline:  # `{}`, `~` or `null`: becomes a block
            lines[at] = head + comment.rstrip()
        lo, hi = at + 1, _block_end(lines, at, own)
        existing = next((_indent(lines[i]) for i in range(lo, hi) if _content(lines[i])), None)
        indent = existing if existing is not None else own + _step(lines)
    # After the block's own trailing comments, before the blank lines and the
    # outdented comments that introduce whatever follows it.
    own = parent[1] if parent is not None else -1
    pos = hi
    while pos > lo and not _content(lines[pos - 1]) and (
            not lines[pos - 1].strip() or _indent(lines[pos - 1]) <= own):
        pos -= 1
    key = path[-1]
    token = key if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.\-@/]*", key) else _render_scalar(key)
    if isinstance(value, dict):
        new = [f"{' ' * indent}{token}:" + ("" if value else " {}")]
        if value:
            new += _render_block(value, indent + _step(lines))
    else:
        new = [f"{' ' * indent}{token}: {_render_scalar(value)}"]
    lines[pos:pos] = new


def _delete(lines: list[str], path: list[str]) -> None:
    found = _find(lines, path)
    if found is None:
        return
    at, indent = found
    # The comments and blank lines after it introduce the next key: they stay.
    last = max(i for i in range(at, _block_end(lines, at, indent)) if _content(lines[i]))
    del lines[at:last + 1]
