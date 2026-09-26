"""`jev-model-router init`: a working config from what this machine has, in one pass.

A capabilities config is two things. Most of it is shared by everyone: the
requirements Jev reads, the rule, the priors per model family. That part ships
with the package (`data/setup.yaml`). The rest is this machine: which CLIs are
installed and where their executables are, whether Ollama runs, which keys
exist, where the log goes. That part is detected here, asked about where it is a
choice, and written once into the user's config directory, which every command
then reads by default.

Nothing here names a model or a provider: the sources on offer, their labels and
the judge's endpoints are all in `data/setup.yaml`. What this module knows is
how to detect a backend kind -- a CLI on PATH, an Ollama by its tags, an API by
its key -- and one rule for the default tier (see `pick_default`).
"""

from __future__ import annotations

import copy
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from . import keystore

DATA = Path(__file__).parent / "data"
# Files copied beside a new config, never over an existing one: the user may have edited them.
DATA_FILES = ("benchmarks.yaml", "benchmarks.derived.json")
# Keys of a setup source that are for init, not for the `discover` entry.
_INIT_ONLY = {"label", "executable"}
# Mean card level a default tier must reach: 2 is "handles most tasks" on the 0-3 scale.
DEFAULT_MIN_LEVEL = 2.0
_PLACEHOLDER = "__init_placeholder__"


def load_setup(path: Path | None = None) -> dict[str, Any]:
    with (path or DATA / "setup.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def find_executable(name: str, which: Callable[[str], str | None] = shutil.which) -> str | None:
    """The file to run for `name`: on Windows the real .exe behind an npm .cmd shim.

    A shim goes through cmd.exe, which mangles a prompt with newlines or quotes
    in it, so the program it wraps is run instead: the .exe it calls, or, when it
    calls a node script, the .exe of the same name inside that script's package.
    Elsewhere PATH already names something runnable.
    """
    found = which(name)
    if not found:
        return None
    path = Path(found)
    if path.suffix.lower() not in (".cmd", ".bat"):
        return str(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return str(path)
    for rel in re.findall(r'"%dp0%\\([^"]+)"', text):
        target = path.parent / rel.replace("\\", "/")
        if target.suffix.lower() == ".exe" and target.is_file():
            return str(target)
        if target.suffix.lower() in (".js", ".cjs", ".mjs"):
            package = _package_root(target)
            if package is not None:
                exe = next((p for p in sorted(package.rglob(f"{name}.exe")) if p.is_file()), None)
                if exe is not None:
                    return str(exe)
    return str(path)


def _package_root(script: Path) -> Path | None:
    for parent in script.parents:
        if (parent / "package.json").is_file():
            return parent
    return None


@dataclass
class Detected:
    name: str
    label: str
    found: bool
    where: str | None = None  # the executable, the URL, or the key's variable
    note: str = ""


def detect(setup: dict[str, Any], *, which=shutil.which, get: Callable[[str], Any] | None = None) -> list[Detected]:
    """What each offered source looks like on this machine, in setup order."""
    out = []
    for name, source in (setup.get("sources") or {}).items():
        label = source.get("label", name)
        backend = source.get("backend")
        if backend in ("claude_cli", "codex_cli"):
            exe = find_executable(source.get("executable") or name, which)
            out.append(Detected(name, label, bool(exe), exe, "" if exe else f"`{source.get('executable')}` not on PATH"))
        elif backend == "ollama":
            url = source.get("base_url", "http://localhost:11434")
            try:
                (get or _get)(f"{url}/api/tags")
                out.append(Detected(name, label, True, url))
            except Exception as exc:  # noqa: BLE001
                out.append(Detected(name, label, False, url, f"not answering ({type(exc).__name__})"))
        else:
            var = source.get("api_key_env")
            has_key = bool(var and keystore.lookup(var))
            out.append(Detected(name, label, has_key, var, "key set" if has_key else f"needs {var}"))
    return out


def _get(url: str) -> Any:
    import httpx

    response = httpx.get(url, timeout=3.0)
    response.raise_for_status()
    return response.json()


def build_raw(setup: dict[str, Any], chosen: list[Detected], judge_first: str, home: Path) -> dict[str, Any]:
    """The config for these sources, before a default tier is picked. Paths are absolute, under `home`."""
    if not chosen:
        raise ValueError("choose at least one source that can answer requests")
    raw = copy.deepcopy(setup["base"])
    caps = raw["router"]["capabilities"]
    caps["benchmarks"] = {"path": str(home / "benchmarks.yaml")}
    discover = {}
    for d in chosen:
        entry = {k: v for k, v in setup["sources"][d.name].items() if k not in _INIT_ONLY}
        if entry.get("backend") in ("claude_cli", "codex_cli") and d.where:
            entry["base_url"] = d.where
        discover[d.name] = entry
    caps["discover"] = discover
    judge = copy.deepcopy(setup["judge"])
    endpoints = judge.get("endpoints") or []
    judge["endpoints"] = sorted(endpoints, key=lambda e: e.get("label") != judge_first)
    raw["tiers"] = {"judge": judge}
    raw["log"] = {"path": str(home / "jev-model-router.db")}
    raw.setdefault("catalog", {})["path"] = str(home / "jev-model-router.catalog.json")
    return raw


def pick_default(raw: dict[str, Any], check=None) -> tuple[str, dict[str, Any]]:
    """The tier requests go to when Jev cannot be asked: (name, its explicit config).

    Rule: among the tiers discovery builds, the cheapest whose mean card level
    reaches DEFAULT_MIN_LEVEL; if none does, the strongest. Cheapest by the
    tier's own prices, else its card's list prices, input plus output.
    """
    from .catalog import check_catalog
    from .config import parse_config
    from .discovery import expand

    probe = copy.deepcopy(raw)
    probe["tiers"][_PLACEHOLDER] = {"backend": "ollama", "model": "placeholder"}
    probe["router"]["default_tier"] = _PLACEHOLDER
    probe["catalog"] = {**probe.get("catalog", {}), "check_on_start": False}
    config = parse_config(probe)
    config, found, _ = expand(config, (check or check_catalog)(config), None)
    cards = config.router.capabilities.cards
    candidates = [d.tier for d in found if d.tier in cards and config.tiers[d.tier].can_serve]
    if not candidates:
        raise ValueError("no model was found on the chosen sources; is the CLI signed in, the listing reachable?")

    def mean(name: str) -> float:
        levels = cards[name].levels
        return sum(levels.values()) / len(levels) if levels else 0.0

    def price(name: str) -> float:
        tier = config.tiers[name]
        prices = tier.prices if tier.prices.configured else cards[name].list_prices
        return prices.input + prices.output

    able = [n for n in candidates if mean(n) >= DEFAULT_MIN_LEVEL]
    name = min(able, key=lambda n: (price(n), -mean(n), n)) if able else max(candidates, key=lambda n: (mean(n), n))
    return name, explicit_tier(config.tiers[name])


def explicit_tier(tier) -> dict[str, Any]:
    """A discovered tier written out by hand, so the config can name it as the default."""
    out: dict[str, Any] = {"backend": tier.backend, "model": tier.model, "base_url": tier.base_url,
                           "context_window": tier.context_window, "timeout_s": tier.timeout_s}
    if tier.supports_tools:
        out["supports_tools"] = True
    if tier.effort and tier.backend in ("claude_cli", "codex_cli"):
        out["effort"] = tier.effort  # an API takes its effort in extra_body instead
    if tier.api_key_env:
        out["api_key_env"] = tier.api_key_env
    if tier.extra_body:
        out["extra_body"] = tier.extra_body
    if tier.prices.configured:
        out["prices"] = {"input": tier.prices.input, "output": tier.prices.output,
                         "cache_read": tier.prices.cache_read, "cache_write": tier.prices.cache_write}
    return out


def finish(raw: dict[str, Any], default: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    name, tier = default
    raw = copy.deepcopy(raw)
    raw["router"]["default_tier"] = name
    raw["tiers"] = {name: tier, **raw["tiers"]}
    return raw


HEADER = """\
# Written by `jev-model-router init`. Every command reads this file when neither -c
# nor JEV_MODEL_ROUTER_CONFIG names another. Run init again to rebuild it (--force).
#
# discover: the providers every model is found on at startup (no quota spent).
# default_tier: where a request goes when Jev cannot be asked.
# Keys are never written here: `jev-model-router keys` shows and sets them.
"""


def write(raw: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = HEADER + "\n" + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, width=100)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)


def copy_data(home: Path) -> list[Path]:
    """The shipped benchmark files, beside the config; an existing copy is kept."""
    written = []
    for name in DATA_FILES:
        target = home / name
        if not target.exists():
            home.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(DATA / name, target)
            written.append(target)
    return written


def default_config_path() -> Path:
    return keystore.home() / "config.yaml"


__all__ = ["Detected", "build_raw", "copy_data", "default_config_path", "detect", "explicit_tier",
           "find_executable", "finish", "load_setup", "pick_default", "write"]


