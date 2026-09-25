"""Command line: `llm-router serve`, `stats`, `train`, `label`, `reconcile`, `calibrate`, `benchmarks` and `check`."""

from __future__ import annotations

import argparse
import json
import os
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
    calibrate.add_argument("--anchors", help="YAML list of {task, packet?, sufficient?, insufficient?, because?}")
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

    bench = sub.add_parser("benchmarks", help="benchmark evidence -> card levels")
    bench_sub = bench.add_subparsers(dest="action", required=True)
    imp = bench_sub.add_parser("import", help="fetch the configured sources and rewrite the imported points")
    imp.add_argument("--source", default=None, help="import only this source")
    bench_sub.add_parser("read", help="ask Jev what each new or changed benchmark measures; writes the sidecar")
    bench_sub.add_parser("check", help="validate the files and show the evidence and levels per model")
    fit = bench_sub.add_parser("fit", help="fit each benchmark's worth and the level line from judged outcomes")
    fit.add_argument("--db", default=None, help="read outcomes from this database instead of the serving log")
    fit.add_argument("--write", action="store_true", help="store the fit in the sidecar")

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


def _expanded(config):
    """The config with discovered tiers, cards derived from benchmark evidence when configured."""
    from .catalog import check_catalog
    from .discovery import expand
    from .scores import load_scores

    report = check_catalog(config)
    scores = load_scores(config)  # a ConfigError here is the caller's to report
    expanded, found, _ = expand(config, report, scores)
    return expanded, found, report, scores


async def _never_ask(packet, questions):
    """For a router built to be read, never to decide: nothing here asks Jev."""
    raise RuntimeError("this command asks nobody")


def _calibrate(config, args) -> int:
    import asyncio

    from .calibration import calibrate, load_anchors, with_caps, with_scale, write_calibration
    from .capabilities import CapabilityRouter
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
        anchors = load_anchors(args.anchors, config.router.capabilities.requirements)
    except ConfigError as exc:
        print(f"anchors: {exc}", file=sys.stderr)
        return 2
    try:
        config, _, _, _ = _expanded(config)
    except ConfigError as exc:
        print(f"benchmarks: {exc}", file=sys.stderr)
        return 2

    async def run():
        router = CapabilityRouter(config)
        try:
            cal = await calibrate(config, anchors, router, margin=args.margin)
            after = CapabilityRouter(with_caps(with_scale(config, cal.scale), cal.caps), ask=router._ask)
            candidates = [n for n, t in config.tiers.items() if t.can_serve]
            for anchor, result in zip(anchors, cal.results):
                request = ChatCompletionRequest.model_validate({
                    "model": "auto", "messages": [{"role": "user", "content": anchor["task"]}],
                    **({"packet": anchor["packet"]} if anchor.get("packet") else {}),
                })
                async def fixed(packet, questions, needs=result.needs):
                    return needs
                after._ask = fixed
                result.picked_after = (await after.decide(request, candidates)).tier
            return cal
        finally:
            await router.aclose()

    try:
        cal = asyncio.run(run())
    except ConfigError as exc:
        print(f"anchors: {exc}", file=sys.stderr)
        return 2
    for result in cal.results:
        print(f"\n{result.task[:90]}")
        print("  need: " + " ".join(f"{k}={v:.2f}" for k, v in result.needs.items()))
        for tier, (op, bound) in result.bounds.items():
            print(f"  {'enough' if op == '<=' else 'not enough'}: {tier}  -> scale {op} {bound:.3f}")
        for tier, own in result.family_scaled.items():
            kind = "enough" if tier in result.sufficient else "not enough"
            print(f"  {kind}: {tier}  -> its family scale {own:.3f} decides it; miss_scale is not bounded by it")
        for _, family, req, old, new, how in result.capped:
            print(f"  capped: {family} {req} {old:.2f} -> {new:.2f} ({how})")
        for tier, req, before, after in result.lowered:
            print(f"    also lowers: {tier} {req} {before:.2f} -> {after:.2f}")
        print(f"  routed after calibration: {result.picked_after}")
    print(f"\nmiss_scale: {cal.scale:.3f} (was {config.router.capabilities.miss_scale:.3f})")
    if cal.caps:
        print("level_caps: " + json.dumps({f: {r: round(v, 3) for r, v in row.items()} for f, row in cal.caps.items()}))
    for conflict in cal.conflicts:
        print(f"  conflict: {conflict}")
    if args.write:
        try:
            # One write: a level_caps the file cannot take leaves miss_scale unwritten too.
            write_calibration(args.config, cal.scale, cal.caps)  # a cap refused for breaking an anchor is not here
        except (ConfigError, OSError) as exc:
            print(f"not written: {exc}", file=sys.stderr)
            return 2
        print(f"written to {args.config}")
    return 1 if cal.conflicts else 0


def _calibrate_from_log(config, args) -> int:
    from .calibration import fit_family_scales, log_outcomes, write_family_scales
    from .capabilities import CapabilityRouter

    try:
        config, _, _, _ = _expanded(config)
    except ConfigError as exc:
        print(f"benchmarks: {exc}", file=sys.stderr)
        return 2

    router = CapabilityRouter(config, ask=_never_ask)  # the log already holds what Jev read
    log = RequestLog(config.log.path)
    try:
        outcomes = log_outcomes(log, config)
    finally:
        log.close()
    if not outcomes:
        print(f"no judged capabilities outcomes in {config.log.path}: report outcomes to "
              "/v1/route/<id>/outcome, turn verification on, or resend failures with failed_tiers")
        return 1
    fits = fit_family_scales(router, outcomes, min_samples=args.min_samples)
    count = {s: sum(o.source == s for o in outcomes) for s in ("outcome", "verdict", "client")}
    print(f"{len(outcomes)} outcome(s): {count['outcome']} reported by a runner, {count['verdict']} verifier "
          f"verdict(s), {count['client']} failure(s) named on a retry")
    print(f"{'family':24} {'n':>5} {'passed':>7} {'predicted':>9}  scale")
    for fit in fits:
        scale = f"{fit.scale:.3f}" if fit.scale is not None else f"(needs {args.min_samples})"
        print(f"{fit.family:24} {fit.n:5d} {fit.passes / fit.n:7.0%} {fit.predicted:9.0%}  {scale}")
    scales = {f.family: f.scale for f in fits if f.scale is not None}
    if args.write and scales:
        try:
            write_family_scales(args.config, {**config.router.capabilities.family_scales, **scales})
        except (ConfigError, OSError) as exc:
            print(f"not written: {exc}", file=sys.stderr)
            return 2
        print(f"written to {args.config}")
    return 0


def _benchmarks(config, args) -> int:
    from .scores import load_scores

    caps = config.router.capabilities
    if config.router.kind != "capabilities" or caps is None or caps.benchmarks is None:
        print("benchmarks needs router.kind: capabilities and a router.capabilities.benchmarks block",
              file=sys.stderr)
        return 2
    try:
        scores = load_scores(config)
    except ConfigError as exc:
        print(f"benchmarks: {exc}", file=sys.stderr)
        return 2
    if args.action == "import":
        return _benchmarks_import(config, scores, args)
    if args.action == "read":
        return _benchmarks_read(config, scores)
    if args.action == "check":
        return _benchmarks_check(config, scores)
    return _benchmarks_fit(config, scores, args)


def _benchmarks_import(config, scores, args) -> int:
    from .scores_import import http_fetch, import_sources

    if args.source and args.source not in scores.sources:
        print(f"no source {args.source!r}; the file has: {', '.join(scores.sources) or 'none'}", file=sys.stderr)
        return 2
    report = import_sources(config, scores, http_fetch(config.router.capabilities.benchmarks.refresh_timeout_s),
                            only=args.source)
    for name, summary in sorted(report.sources.items()):
        print(f"{name}: {summary.rows} row(s) read, {summary.skipped} skipped, {summary.points} point(s) "
              f"in {len(summary.benchmarks)} benchmark(s)")
    for name, why in sorted(report.skipped.items()):
        print(f"skipped: {name}: {why} (its previous points are kept)", file=sys.stderr)
    for name, error in sorted(report.errors.items()):
        print(f"failed: {name}: {error} (its previous points are kept)", file=sys.stderr)
    print(f"written to {scores.imported_path}")
    return 1 if report.errors else 0


def _benchmarks_read(config, scores) -> int:
    import asyncio

    from .capabilities import jev_asker
    from .scores_read import read

    ask = jev_asker(config.tier(config.router.capabilities.jev_tier))

    async def run():
        try:
            return await read(config, scores, ask)
        finally:
            await ask.client.aclose()

    report = asyncio.run(run())
    for key in report.asked:
        print(f"read: {key}")
    print(f"{len(report.asked)} read, {len(report.unchanged)} unchanged, {len(report.manual)} manual")
    for key, error in report.errors.items():
        print(f"jev error on {key}, its old reading is kept: {error}", file=sys.stderr)
    if report.asked:
        print(f"written to {scores.sidecar_path}")
    return 1 if report.errors else 0


def _benchmarks_check(config, scores) -> int:
    from .catalog import check_catalog
    from .discovery import derived_tiers, expand, model_ids, served_ids
    from .scores import base_weights, benchmark_weights, fitted_scales
    from .scores_derive import derive, evidence_summary, line_for, served_keys, startup_lines

    caps = config.router.capabilities
    report = check_catalog(config)
    expanded, found, _ = expand(config, report, scores)
    weights, scales = base_weights(scores, caps), fitted_scales(scores, caps)
    print(f"{len(scores.benchmarks)} benchmark(s), {len(scores.points)} point(s), "
          f"imported {scores.imported_at or 'never'}")
    for name, source in scores.sources.items():
        print(f"  source {name} ({source.origin}): {scores.imported_status.get(name, 'not imported')}")
    for key in scores.benchmarks:
        how, row = weights.get(key, ("unread", {}))
        links = scores.links.get(key, 0)
        if key in scores.scale.detached:  # off the main scale: counts for nothing, however well linked
            marker = " [detached]"
        else:
            marker = "" if links >= 3 else (" [unlinked]" if links == 0 else " [thin]")
        readings = [r for by_effort in scores.readings.get(key, {}).values() for r in by_effort.values()]
        models = len(scores.readings.get(key, {}))
        spread = (f"difficulty {scores.scale.beta[key] / scores.scale.alpha[key]:+.2f}, "
                  f"spread {scores.scale.alpha[key]:.2f}, ") if key in scores.scale.alpha else ""
        info = f", mean information {sum(r.u for r in readings) / len(readings):.2f}" if readings else ""
        print(f"\n{key}{marker}: {spread}{len(readings)} point(s), {models} model(s), {links} linking{info}")
        scale = f" x{scales[key]:.2f} fitted" if key in scales else ""
        print(f"  weights ({how}{scale}): " + (" ".join(f"{r}={w:.2f}" for r, w in row.items() if w > 0) or "none"))
    served = served_ids(found, report, config)
    keys, _ = served_keys(scores, served)
    print("\nevidence per served model:")
    for name in sorted(served):
        summary = evidence_summary(scores, keys[name])
        if not summary:
            print(f"  {name}: no evidence")
            continue
        origins = set().union(*(o for _, o in summary.values()))
        tag = " [vendor-only]" if origins == {"vendor"} else ""
        print(f"  {name}{tag}: " + "; ".join(
            f"{b} ({', '.join(sorted(str(e) for e in efforts))})" for b, (efforts, _) in sorted(summary.items())))
    # Every card `expand` derived, explicit tiers included: the line printed is the one serving uses.
    tiers = derived_tiers(expanded, scores, model_ids(report))
    bench_weights = benchmark_weights(scores, caps)
    line = line_for(caps, scores, list(tiers.values()), bench_weights)
    print(f"\nline over the derived tiers: a={line[0]:.2f} k={line[1]:.2f} profile_weight={scores.c0:.2f}\n")
    cards = expanded.router.capabilities.cards
    for tier, (k, effort, profile) in tiers.items():
        derived = derive(caps, scores, k, effort, profile, line, bench_weights)
        # A tier no benchmark covers is its profile, and is named by the no-evidence line below.
        if not any(derived.coverage[r] > 0 for r in caps.requirements):
            continue
        # Levels and output as served (the card `expand` built); source and C from the evidence.
        print(f"{tier}: output_tokens {cards[tier].output_tokens}")
        for r in caps.requirements:
            level = cards[tier].levels.get(r, 0.0)
            print(f"  {r:14} {level:.2f}  {derived.source[r]:9} C={derived.coverage[r]:.2f}")
    for line_text in startup_lines(scores, caps, served):
        print(line_text)
    for p in scores.superseded:
        print(f"superseded (an independent point replaces it; it can be deleted): {p.benchmark} {p.model} {p.effort}")
    served_keys_all = sorted({k for ks in keys.values() for k in ks})
    unmatched = sorted({p.model for p in scores.points if scores.key(p.model) not in served_keys_all})
    if unmatched:
        import difflib

        print(f"{len(unmatched)} unmatched model string(s) (no served model has the key; they still inform "
              f"the scale when they link benchmarks)")
        # The ones that look like a served model written differently: an `aliases` entry fixes each.
        for text in unmatched:
            near = difflib.get_close_matches(scores.key(text), served_keys_all, n=1, cutoff=0.8)
            if near:
                print(f"  {text!r} is near served key {near[0]!r}: add an alias if it is the same model")
    return 1 if scores.errors else 0


def _benchmarks_fit(config, scores, args) -> int:
    from .calibration import log_outcomes
    from .catalog import check_catalog
    from .discovery import expand, model_ids
    from .scores_fit import fit, write_fit

    if args.db is not None and not os.path.isfile(args.db):
        print(f"no database at {args.db}", file=sys.stderr)  # opening it would create an empty one
        return 2
    report = check_catalog(config)
    expanded, _, _ = expand(config, report, scores)
    source = args.db or config.log.path
    log = RequestLog(source)
    try:
        outcomes = log_outcomes(log, expanded)
    finally:
        log.close()
    if not outcomes:
        print(f"no judged capabilities outcomes in {source}: nothing to fit")
        return 0
    result = fit(expanded, scores, outcomes, model_ids=model_ids(report))
    if result.outcomes == 0:  # offsets are written only when there are outcomes
        print(f"{len(outcomes)} judged outcome(s) in {source}, none on a derived tier: nothing to fit")
        return 0
    print(f"{result.outcomes} outcome(s) on {len(result.tiers)} derived tier(s)")
    print(f"log L: {result.loglik_prior:.2f} at the prior, {result.loglik:.2f} fitted")
    settings = scores.settings
    if settings.a is not None:
        print(f"line: a {settings.a:.2f}, set in the config (not fitted)")
    else:
        print(f"line: a {result.a_before:.2f} served now -> {result.a:.2f} "
              f"(delta_a {scores.fit.delta_a:+.2f} -> {result.delta_a:+.2f})")
    if settings.k is not None:
        print(f"line: k {settings.k:.2f}, set in the config (not fitted)")
    else:
        print(f"line: k {result.k_before:.2f} served now -> {result.k:.2f} "
              f"(k_ratio x{scores.fit.k_ratio:.2f} -> x{result.k_ratio:.2f})")
    for bench, value in result.moved:
        print(f"  {bench}: worth 1.00 -> {value:.2f}")
    for name, edge in result.at_edge:
        print(f"  {name} ended on its box edge {edge}: the outcomes push it further than the fit allows")
    for bench in result.unlinked:
        print(f"  {bench}: unlinked, never enters a level; its worth stays at the prior")
    if args.write:
        write_fit(scores.sidecar_path, result, scores, config)
        print(f"written to {scores.sidecar_path}")
        print("next: `calibrate --from-log`, then `calibrate --anchors`; both fit on these levels")
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

    if args.command == "benchmarks":
        return _benchmarks(config, args)

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
            from .scores import load_scores

            try:
                scores = load_scores(config)
            except ConfigError as exc:
                scores = None
                print(f"benchmarks failed to load: {exc}", file=sys.stderr)
            expanded, found, unprofiled = expand(config, report, scores)
            extra: dict = {}
            lines: list[str] = []
            if config.router.kind == "capabilities":
                from .capabilities import CapabilityRouter
                from .discovery import unused_caps
                from .dominance import summary

                routed = CapabilityRouter(expanded, ask=_never_ask)
                routed.note_unavailable(report.unavailable)
                unused = unused_caps(expanded)
                extra = {"dominated": routed.dominated, "unused_level_caps": unused}
                if routed.dominance_error:
                    lines.append(f"dominance check failed: {routed.dominance_error}")
                lines += summary(routed.dominated, len(expanded.router.capabilities.cards))
                if unused:
                    lines.append(f"  level_caps for no card (a model left the catalog?): {', '.join(unused)}")
            if args.json:
                print(json.dumps({**report.to_dict(), "discovered_tiers": [f.tier for f in found],
                                  "unprofiled": unprofiled, **extra}, indent=2))
            else:
                print(report.summary())
                if found:
                    print(f"catalog: {len(found)} tier(s) discovered")
                if unprofiled:
                    print(f"  on the fallback profile (write a profile): {', '.join(unprofiled)}")
                for line in lines:
                    print(line)
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
