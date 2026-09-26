"""API keys: the environment first, then one per-user secrets file.

A config names a key (`api_key_env`) and never holds it, so it can be committed
and shared. The environment is where the key was always read from, and still
wins. The file exists for the processes whose environment the user does not
control: an agent starts `jev-model-router mcp` with its own, and setting a variable
for good takes a new login on some systems and a registry write on others.
`jev-model-router keys set` writes it; nothing else does.

The file lives outside every repository, under the user's config directory
(`JEV_MODEL_ROUTER_HOME` overrides it), as `NAME=value` lines. It is read on every
lookup rather than cached, so a key set while the proxy runs is used by the
next call that reads it.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

HOME_ENV = "JEV_MODEL_ROUTER_HOME"
FILE_NAME = "secrets.env"
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def home() -> Path:
    """The per-user directory: $JEV_MODEL_ROUTER_HOME, else the platform's config directory."""
    explicit = os.environ.get(HOME_ENV)
    if explicit:
        return Path(explicit).expanduser()
    if sys.platform == "win32" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "jev-model-router"
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "jev-model-router"


def path() -> Path:
    return home() / FILE_NAME


def read() -> dict[str, str]:
    """Every key in the file; {} when there is none."""
    try:
        text = path().read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        return {}
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        out[name.strip()] = value.strip()
    return out


def lookup(name: str) -> str | None:
    """The key's value, from the environment or else the file."""
    return os.environ.get(name) or read().get(name) or None


def source(name: str) -> str | None:
    """Where `lookup` would find it: "env", "file", or None."""
    if os.environ.get(name):
        return "env"
    return "file" if read().get(name) else None


def store(name: str, value: str) -> Path:
    """Write one key, replacing any previous value. Returns the file written."""
    if not _NAME.match(name):
        raise ValueError(f"{name!r} is not a variable name")
    value = value.strip()
    if not value or any(c in value for c in "\r\n"):
        raise ValueError("the key is empty or spans lines")
    keys = read()
    keys[name] = value
    return _write(keys)


def remove(name: str) -> bool:
    """Drop one key from the file. False when it was not there."""
    keys = read()
    if name not in keys:
        return False
    del keys[name]
    _write(keys)
    return True


def _write(keys: dict[str, str]) -> Path:
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True)
    body = "# Written by `jev-model-router keys`. The environment wins over this file.\n"
    body += "".join(f"{k}={v}\n" for k, v in sorted(keys.items()))
    # Atomic, and readable by the owner only from the moment it exists.
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".secrets-", text=True)
    try:
        os.chmod(tmp, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(body)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


__all__ = ["HOME_ENV", "home", "lookup", "path", "read", "remove", "source", "store"]
