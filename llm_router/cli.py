"""Command line: `llm-router serve`, `stats`, `train`, `label`, `reconcile` and `check`."""

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
    train.add_argument(
        "--db",
        default=None,
        help="train from this database instead of the serving log (e.g. one `label` wrote)",
    )
    train.add_argument(
        "--tune",
        action="store_true",
        help="choose the fit's settings by cross-validation on the training split",
    )
    train.add_argument("--threshold", type=float, default=0.5)
    train.add_argument(
        "--target-escalation",
        type=float,
        default=None,
        help="set the threshold to escalate this share of the training split (e.g. 0.15)",
    )
    train.add_argument(
        "--target-recall",
        type=float,
        default=None,
        help="set the threshold to catch this share of the training split's failures "
        "(e.g. 0.8) -- the same knob priced in quality instead of traffic",
    )
    train.add_argument("--holdout", type=float, default=0.25)
    train.add_argument("--min-examples", type=int, default=40)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    train.add_argument("--json", action="store_true", help="emit JSON instead of text")

    label = sub.add_parser(
        "label", help="label a tier's answers from a question set with an answer key"
    )
    label.add_argument("--gsm8k", required=True, help="path to a GSM8K-format JSONL file")
    label.add_argument("--db", required=True, help="database to write; never the serving log")
    label.add_argument(
        "--tier", default=None, help="the tier to label (default: router.default_tier)"
    )
    label.add_argument("--limit", type=int, default=None, help="ask at most this many")
    label.add_argument("--json", action="store_true", help="emit JSON instead of text")

    reconcile = sub.add_parser(
        "reconcile", help="ask the provider what it charged for rows that carry no bill"
    )
    reconcile.add_argument("--limit", type=int, default=500)
    reconcile.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    reconcile.add_argument("--json", action="store_true", help="emit JSON instead of text")

    check = sub.add_parser("check", help="validate the config and exit")
    check.add_argument(
        "--prices",
        action="store_true",
        help="also compare each tier's prices with the provider's published list",
    )
    check.add_argument("--json", action="store_true", help="emit JSON (with --prices)")
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

    log = RequestLog(args.db or config.log.path, store_prompts=config.log.store_prompts)
    judged_by = config.verification.verifier_tier
    if args.db:
        # A labelled database names its own source; the config's verifier
        # judged none of it.
        sources = [r[0] for r in log.query("SELECT DISTINCT verifier_tier FROM verifications")]
        judged_by = ",".join(sorted(s for s in sources if s)) or None
    try:
        report = train_from_log(
            log,
            predicts_tier=predicts,
            strong_tier=strong,
            judged_by=judged_by,
            threshold=args.threshold,
            holdout=args.holdout,
            min_examples=args.min_examples,
            seed=args.seed,
            tune_settings=args.tune,
            target_escalation=args.target_escalation,
            target_recall=args.target_recall,
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
        if not (args.prices and args.json):
            print(f"config OK: {len(config.tiers)} tier(s): {', '.join(config.tiers)}")
        if not args.prices:
            return 0
        from .price_check import check_prices, format_report

        prices = check_prices(config)
        print(json.dumps(prices.to_dict(), indent=2) if args.json else format_report(prices))
        return 0 if prices.ok else 1

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

    if args.command == "label":
        from .benchmark import format_label_report, label, load_gsm8k

        tier = args.tier or config.router.default_tier
        if not tier:
            print("no tier: pass --tier, or set router.default_tier", file=sys.stderr)
            return 2

        def progress(n: int, ok: bool, reason: str) -> None:
            print(f"  {n:>5}  {'pass' if ok else 'FAIL'}  {reason}", file=sys.stderr, flush=True)

        try:
            result = label(
                config,
                load_gsm8k(args.gsm8k),
                db=args.db,
                tier=tier,
                source="gsm8k",
                limit=args.limit,
                progress=progress,
            )
        except ValueError as exc:
            print(f"cannot label: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(result.to_dict(), indent=2)
            if args.json
            else format_label_report(result, db=args.db)
        )
        return 1 if result.errors else 0

    if args.command == "reconcile":
        from .reconcile import format_report, reconcile

        log = RequestLog(config.log.path, store_prompts=config.log.store_prompts)
        try:
            result = reconcile(config, log, limit=args.limit, dry_run=args.dry_run)
        finally:
            log.close()
        print(
            json.dumps(result.to_dict(), indent=2)
            if args.json
            else format_report(result, dry_run=args.dry_run)
        )
        return 1 if result.errors else 0

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
