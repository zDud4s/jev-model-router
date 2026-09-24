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
    train.add_argument(
        "--bad-answer-cost",
        type=float,
        default=None,
        help="what one bad answer costs you, in dollars. Given it, the policy table and "
        "the sweep add a total and name the cheapest; without it they stay separate columns",
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
    label.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="temperature to label at (default 0.0). A label drawn hot records what the "
        "sample did, not what the question is worth",
    )
    label.add_argument("--json", action="store_true", help="emit JSON instead of text")

    reconcile = sub.add_parser(
        "reconcile", help="ask the provider what it charged for rows that carry no bill"
    )
    reconcile.add_argument("--limit", type=int, default=500)
    reconcile.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    reconcile.add_argument("--json", action="store_true", help="emit JSON instead of text")

    calibrate = sub.add_parser(
        "calibrate",
        help="set miss_scale from anchors: tasks with a tier known to be (or not be) enough; asks Jev only",
    )
    calibrate.add_argument("--anchors", help="YAML list of {task, packet?, sufficient?, insufficient?}")
    calibrate.add_argument(
        "--from-log", action="store_true",
        help="fit one scale per model family from judged outcomes in the request log (no calls at all)",
    )
    calibrate.add_argument("--min-samples", type=int, default=30, help="trials a family needs before it gets a scale")
    calibrate.add_argument("--write", action="store_true", help="write the scale into the config file")
    calibrate.add_argument(
        "--margin", type=float, default=0.03,
        help="how far above the target a sufficient tier must land (Jev's readings vary between calls)",
    )

    check = sub.add_parser("check", help="validate the config and exit")
    check.add_argument(
        "--prices",
        action="store_true",
        help="also compare each tier's prices with the provider's published list",
    )
    check.add_argument(
        "--catalog",
        action="store_true",
        help="also check every tier against the models its provider serves now, and update the catalog file",
    )
    check.add_argument("--json", action="store_true", help="emit JSON (with --prices or --catalog)")
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
            bad_answer_cost=args.bad_answer_cost,
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


def _calibrate(config, args) -> int:
    import asyncio

    from .calibration import calibrate, load_anchors, with_scale, write_scale
    from .capabilities import CapabilityRouter
    from .catalog import check_catalog
    from .discovery import expand
    from .schemas import ChatCompletionRequest

    if config.router.kind != "capabilities":
        print("calibrate needs router.kind: capabilities", file=sys.stderr)
        return 2
    if args.from_log:
        return _calibrate_from_log(config, args)
    if not args.anchors:
        print("calibrate needs --anchors or --from-log", file=sys.stderr)
        return 2
    try:
        anchors = load_anchors(args.anchors)
    except ConfigError as exc:
        print(f"anchors: {exc}", file=sys.stderr)
        return 2
    config, _, _ = expand(config, check_catalog(config))

    async def run():
        router = CapabilityRouter(config)
        try:
            scale, results, conflicts = await calibrate(config, anchors, router, margin=args.margin)
            after = CapabilityRouter(with_scale(config, scale), ask=router._ask)
            candidates = [n for n, t in config.tiers.items() if t.can_serve]
            for anchor, result in zip(anchors, results):
                request = ChatCompletionRequest.model_validate({
                    "model": "auto", "messages": [{"role": "user", "content": anchor["task"]}],
                    **({"packet": anchor["packet"]} if anchor.get("packet") else {}),
                })
                async def fixed(packet, questions, needs=result.needs):
                    return needs
                after._ask = fixed
                result.picked_after = (await after.decide(request, candidates)).tier
            return scale, results, conflicts
        finally:
            await router.aclose()

    try:
        scale, results, conflicts = asyncio.run(run())
    except ConfigError as exc:
        print(f"anchors: {exc}", file=sys.stderr)
        return 2
    for result in results:
        print(f"\n{result.task[:90]}")
        print("  need: " + " ".join(f"{k}={v:.2f}" for k, v in result.needs.items()))
        for tier, (op, bound) in result.bounds.items():
            print(f"  {'enough' if op == '<=' else 'not enough'}: {tier}  -> scale {op} {bound:.3f}")
        print(f"  routed after calibration: {result.picked_after}")
    print(f"\nmiss_scale: {scale:.3f} (was {config.router.capabilities.miss_scale:.3f})")
    for conflict in conflicts:
        print(f"  conflict: {conflict}")
    if args.write:
        write_scale(args.config, scale)
        print(f"written to {args.config}")
    return 1 if conflicts else 0


def _calibrate_from_log(config, args) -> int:
    from .calibration import fit_family_scales, log_outcomes, write_family_scales
    from .capabilities import CapabilityRouter
    from .catalog import check_catalog
    from .discovery import expand

    config, _, _ = expand(config, check_catalog(config))

    async def never(packet, questions):  # the log already holds what Jev read
        raise RuntimeError("calibrate --from-log asks nobody")

    router = CapabilityRouter(config, ask=never)
    log = RequestLog(config.log.path)
    try:
        outcomes = log_outcomes(log, config)
    finally:
        log.close()
    if not outcomes:
        print(f"no judged capabilities outcomes in {config.log.path}: turn verification on, "
              "or have the client resend failures with failed_tiers")
        return 1
    fits = fit_family_scales(router, outcomes, min_samples=args.min_samples)
    verdicts = sum(o.source == "verdict" for o in outcomes)
    print(f"{len(outcomes)} outcome(s): {verdicts} verdict(s), {len(outcomes) - verdicts} client-reported failure(s)")
    print(f"{'family':24} {'n':>5} {'passed':>7} {'predicted':>9}  scale")
    for fit in fits:
        scale = f"{fit.scale:.3f}" if fit.scale is not None else f"(needs {args.min_samples})"
        print(f"{fit.family:24} {fit.n:5d} {fit.passes / fit.n:7.0%} {fit.predicted:9.0%}  {scale}")
    scales = {f.family: f.scale for f in fits if f.scale is not None}
    if args.write and scales:
        write_family_scales(args.config, {**config.router.capabilities.family_scales, **scales})
        print(f"written to {args.config}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.command == "calibrate":
        return _calibrate(config, args)

    if args.command == "check":
        quiet = args.json and (args.prices or args.catalog)
        if not quiet:
            print(f"config OK: {len(config.tiers)} tier(s): {', '.join(config.tiers)}")
        if args.catalog:
            from .catalog import check_catalog, write_if_changed

            report = check_catalog(config)
            if config.catalog.path and write_if_changed(report, config.catalog.path) and not quiet:
                print(f"catalog updated: {config.catalog.path}")
            from .discovery import expand

            _, found, unprofiled = expand(config, report)
            if args.json:
                print(json.dumps({**report.to_dict(), "discovered_tiers": [f.tier for f in found],
                                  "unprofiled": unprofiled}, indent=2))
            else:
                print(report.summary())
                if found:
                    print(f"catalog: {len(found)} tier(s) discovered")
                if unprofiled:
                    print(f"  on the fallback profile (write a profile): {', '.join(unprofiled)}")
            if not args.prices:
                return 1 if report.unavailable else 0
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
                temperature=args.temperature,
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
