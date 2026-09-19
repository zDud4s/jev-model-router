"""Command line: `llm-router serve` and `llm-router stats`."""

from __future__ import annotations

import argparse
import json
import sys

from .config import ConfigError, load_config
from .db import RequestLog
from .stats import collect, format_text

DEFAULT_CONFIG = "config.yaml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-router", description=__doc__)
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG, help="path to the YAML config")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the proxy")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    stats = sub.add_parser("stats", help="read the request log back")
    stats.add_argument("--json", action="store_true", help="emit JSON instead of text")

    sub.add_parser("check", help="validate the config and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.command == "check":
        print(f"config OK: {len(config.tiers)} tier(s): {', '.join(config.tiers)}")
        return 0

    if args.command == "stats":
        # Opened read-write on purpose: the migration must run so `stats` works
        # against a database created by an older version of the binary.
        log = RequestLog(config.log.path, store_prompts=config.log.store_prompts)
        try:
            report = collect(log)
            print(json.dumps(report.to_dict(), indent=2) if args.json else format_text(report))
        finally:
            log.close()
        return 0

    if args.command == "serve":
        import uvicorn

        from .app import create_app

        host = args.host or config.server.host
        port = args.port or config.server.port
        uvicorn.run(create_app(config), host=host, port=port)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
