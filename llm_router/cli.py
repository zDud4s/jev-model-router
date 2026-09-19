"""Command line: `llm-router serve`, `stats`, `train` and `check`."""

from __future__ import annotations

import argparse
import json
import sys

from .config import ConfigError, load_config
from .db import RequestLog
from .stats import collect, format_text
from .training import TrainingError, format_report, train_from_log

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

    train = sub.add_parser(
        "train", help="fit a difficulty model from the verdicts in the request log"
    )
    train.add_argument(
        "--out", default=None, help="where to write the model (default: router.model_path)"
    )
    train.add_argument(
        "--tier",
        default=None,
        help="the tier whose failures to model (default: router.default_tier)",
    )
    train.add_argument(
        "--strong-tier",
        default=None,
        help="the escalation target to price against (default: router.strong_tier)",
    )
    train.add_argument("--threshold", type=float, default=0.5)
    train.add_argument("--holdout", type=float, default=0.25)
    train.add_argument("--min-examples", type=int, default=40)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    train.add_argument("--json", action="store_true", help="emit JSON instead of text")

    sub.add_parser("check", help="validate the config and exit")
    return parser


def _train(config, args) -> int:
    predicts = args.tier or config.router.default_tier
    strong = args.strong_tier or config.router.strong_tier
    if not strong:
        print(
            "no escalation target: pass --strong-tier, or set router.strong_tier. "
            "Without one there is nothing to price the model's decisions against.",
            file=sys.stderr,
        )
        return 2
    if strong == predicts:
        print(f"--tier and --strong-tier are both {strong!r}", file=sys.stderr)
        return 2

    log = RequestLog(config.log.path, store_prompts=config.log.store_prompts)
    try:
        report = train_from_log(
            log,
            predicts_tier=predicts,
            strong_tier=strong,
            judged_by=config.verification.verifier_tier,
            threshold=args.threshold,
            holdout=args.holdout,
            min_examples=args.min_examples,
            seed=args.seed,
        )
    except TrainingError as exc:
        # Not an exception trace. Every one of these says what is missing from
        # the log and how to get it, because that is the whole content of the
        # failure.
        print(f"cannot train: {exc}", file=sys.stderr)
        return 3
    finally:
        log.close()

    print(json.dumps(report.to_dict(), indent=2) if args.json else format_report(report))

    destination = args.out or config.router.model_path
    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    if not destination:
        print(
            "\nnothing written: pass --out, or set router.model_path. The report "
            "above is the product; writing the file is a separate decision.",
            file=sys.stderr,
        )
        return 0
    report.model.save(destination)
    print(f"\nwrote {destination}  ({report.model.fingerprint})")
    # Said plainly because the opposite is the natural assumption: the router
    # loads its model once, at startup, so a running server keeps serving the
    # old weights until it is restarted.
    print("a running server keeps its current model until restarted")
    return 0


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

    if args.command == "train":
        return _train(config, args)

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
