"""Command-line entry point: ``leaksense <command>``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from leaksense import __version__
from leaksense.config import DATA_DIR_VAR, ConfigError, get_data_dir, load_dotenv_file

EXIT_OK, EXIT_FAILED, EXIT_CONFIG = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leaksense",
        description="Leak localization and sensor placement on the Hanoi network (LeakDB).",
    )
    parser.add_argument("--version", action="version", version=f"leaksense {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    v = sub.add_parser("validate-data", help="check the dataset against the data contract (D1-D10)")
    v.add_argument("--data-dir", help=f"dataset HanoiOK folder (default: ${DATA_DIR_VAR})")
    v.add_argument("--deep", action="store_true", help="also parse every pressure and leak file")
    v.add_argument("--out", default="reports/data_validation.json", help="JSON report path")
    v.add_argument("--contract", help="JSON file overriding the default data contract")
    v.add_argument("--quiet", action="store_true", help="no progress output")
    v.set_defaults(func=cmd_validate_data)

    n = sub.add_parser("network-info", help="summary of the nominal network graph (Hanoi.inp)")
    n.add_argument("--data-dir", help=f"dataset HanoiOK folder (default: ${DATA_DIR_VAR})")
    n.add_argument("--json", action="store_true", help="print the summary as JSON")
    n.set_defaults(func=cmd_network_info)

    b = sub.add_parser("build-index", help="build the case index (artifacts/cases.parquet)")
    b.add_argument("--data-dir", help=f"dataset HanoiOK folder (default: ${DATA_DIR_VAR})")
    b.add_argument("--out", help="parquet path (default: $LEAKSENSE_ARTIFACTS_DIR/cases.parquet)")
    b.add_argument("--report", default="reports/case_index.json", help="JSON summary path")
    b.add_argument(
        "--validation",
        default="reports/data_validation.json",
        help="M1 report to cross-check per-pipe pure counts against (skipped if missing)",
    )
    b.add_argument("--contract", help="JSON file overriding the default data contract")
    b.add_argument("--quiet", action="store_true", help="no progress output")
    b.set_defaults(func=cmd_build_index)

    m = sub.add_parser("make-splits", help="locked test fold + grouped 5-fold CV (splits/)")
    m.add_argument("--cases", help="case index (default: $LEAKSENSE_ARTIFACTS_DIR/cases.parquet)")
    m.add_argument("--out", default="splits/splits_v1.json", help="splits JSON path")
    m.add_argument(
        "--index-report",
        default="reports/case_index.json",
        help="M3 report, for the data-manifest hash",
    )
    m.add_argument("--trials", type=int, default=50, help="candidate seeds for type/kind balance")
    m.set_defaults(func=cmd_make_splits)

    from leaksense.feature_cmds import add_parsers

    add_parsers(sub)
    return parser


def _data_dir(args: argparse.Namespace) -> Path:
    return get_data_dir({DATA_DIR_VAR: args.data_dir} if args.data_dir else None)


def cmd_network_info(args: argparse.Namespace) -> int:
    from leaksense.network import load_nominal_network

    net = load_nominal_network(_data_dir(args))
    s = net.summary()
    if args.json:
        print(json.dumps(s, indent=2))
        return EXIT_OK
    print(f"nodes: {s['n_nodes']} ({s['n_junctions']} junctions, reservoir {s['reservoirs']})")
    print(
        f"pipes: {s['n_pipes']} ({s['n_leak_eligible_pipes']} leak-eligible, "
        f"never leaking {s['never_leaking_pipes']})"
    )
    print(f"sensor candidates: {s['n_sensor_candidates']}")
    print(
        f"connected: {s['connected']}  loops: {s['n_loops']}  degree counts: {s['degree_counts']}"
    )
    print(
        f"length: total {s['total_length_m']:,.0f} m, per pipe "
        f"{s['length_m'][0]:,.0f}-{s['length_m'][1]:,.0f} m"
    )
    print(f"diameters (mm): {s['diameters_mm']}  roughness: {s['roughness']}")
    print(f"H-W resistance (SI, Q in m^3/s): {s['resistance'][0]:.4g}-{s['resistance'][1]:.4g}")
    print(
        f"max pipe hops: {s['max_pipe_hops']}  max network distance: {s['max_node_dist_m']:,.0f} m"
    )
    print(f"total nominal base demand: {s['total_base_demand_lps']:,.1f} L/s")
    return EXIT_OK


def cmd_validate_data(args: argparse.Namespace) -> int:
    from leaksense.data.contract import DataContract
    from leaksense.data.validate import validate_dataset

    data_dir = _data_dir(args)
    contract = DataContract.from_json(args.contract) if args.contract else DataContract()
    t0 = time.perf_counter()
    report = validate_dataset(data_dir, contract, deep=args.deep, progress=not args.quiet)
    elapsed = time.perf_counter() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["elapsed_seconds"] = round(elapsed, 1)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    m = report.manifest
    print(
        f"dataset: {m['folder']}  files={m['n_files']}  bytes={m['total_bytes']:,}  "
        f"manifest={m['sha256'][:12]}  mode={'deep' if args.deep else 'fast'}"
    )
    for check in report.checks.values():
        print(f"  {check.id:<4} {check.status.upper():<5} {check.title}")
        for err in check.errors[:3]:
            print(f"         - {err}")
        if check.n_errors > 3:
            print(f"         ... {check.n_errors - 3} more (see {out})")
    print(f"{'OK' if report.ok else 'FAILED'} in {elapsed:.1f}s -> {out}")
    return EXIT_OK if report.ok else EXIT_FAILED


def cmd_build_index(args: argparse.Namespace) -> int:
    from leaksense.config import get_artifacts_dir
    from leaksense.data.index import build_index
    from leaksense.provenance import provenance

    data_dir = _data_dir(args)
    t0 = time.perf_counter()
    from leaksense.data.contract import DataContract

    contract = DataContract.from_json(args.contract) if args.contract else DataContract()
    result = build_index(data_dir, contract, progress=not args.quiet)
    elapsed = time.perf_counter() - t0
    s = result.summary()

    out = Path(args.out) if args.out else get_artifacts_dir() / "cases.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.cases.to_parquet(out, index=False)
    report = {**provenance(data_dir, config=s["config"]), "elapsed_seconds": round(elapsed, 1)}
    report.update(s)

    cand, cases = s["candidates"], s["cases"]
    print(f"candidates: {cand['pure']} pure + {cand['solo']} solo = {cand['total']}")
    print(f"excluded (no baseline): {s['n_excluded']}")
    for e in s["excluded"]:
        print(f"  {e['case_id']}  pipe {e['pipe']}: {e['reason']}")
    print(f"cases: {cases['pure']} pure + {cases['solo']} solo = {cases['total']}")
    print(f"two-leak scenarios without a solo case: {len(s['two_leak_without_solo'])}")
    for e in s["two_leak_without_solo"]:
        print(f"  Scenario-{e['scenario_id']}: solo segment {e['solo_steps']} steps")
    print(f"baseline kind: {s['baseline_kind']}  by case kind: {s['baseline_kind_by_kind']}")
    print(
        f"solo cases clipped by the second leak: {s['solo_clipped']}; "
        f"solo cases whose first leak is leak_node1: {s['solo_first_leak_is_leak_node1']}"
    )
    print(f"n_steps: {s['n_steps']}  short (<24 h): {s['flag_short_case']}")
    print(f"leak type: {s['leak_type']}  leak flow L/s: {s['leak_flow_lps']}")

    pipes = s["per_pipe"]
    bad_pipes = [p for p in pipes if not (p.isdigit() and 2 <= int(p) <= 34)]
    print(f"pipes: {len(pipes)} distinct, all in 2-34: {not bad_pipes}")
    print("pipe  pure  solo  total")
    for p, r in pipes.items():
        print(f"{p:>4}  {r['pure']:>4}  {r['solo']:>4}  {r['total']:>5}")
    totals = {p: r["total"] for p, r in pipes.items()}
    if totals:
        lo, hi = min(totals, key=totals.get), max(totals, key=totals.get)
        print(f"totals range: {totals[lo]} (pipe {lo}) to {totals[hi]} (pipe {hi})")

    ok = not bad_pipes
    vpath = Path(args.validation)
    if vpath.is_file():
        v = json.loads(vpath.read_text())
        d8 = next(c for c in v["checks"] if c["id"] == "D8")["details"]
        expected = d8["single_leak_scenarios_per_pipe"]
        # Compare against candidate pure cases: exclusions are reported separately above.
        got = {p: r["pure"] for p, r in pipes.items() if r["pure"]}
        for e in s["excluded"]:
            if e["kind"] == "pure":
                got[e["pipe"]] = got.get(e["pipe"], 0) + 1
        same = got == expected
        ok &= same
        print(f"per-pipe pure counts (before exclusion) match {vpath} D8: {same}")
        report["d8_pure_counts_match"] = same

    rpath = Path(args.report)
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'OK' if ok else 'FAILED'} in {elapsed:.1f}s -> {out}, {rpath}")
    return EXIT_OK if ok else EXIT_FAILED


def cmd_make_splits(args: argparse.Namespace) -> int:
    import pandas as pd

    from leaksense.config import get_artifacts_dir
    from leaksense.data.splits import make_splits
    from leaksense.provenance import file_sha256, provenance

    cases_path = Path(args.cases) if args.cases else get_artifacts_dir() / "cases.parquet"
    if not cases_path.is_file():
        raise ConfigError(f"{cases_path} not found; run `leaksense build-index` first")
    cases = pd.read_parquet(cases_path)
    from leaksense.data.splits import SplitConfig

    splits = make_splits(cases, SplitConfig(n_seed_trials=args.trials))
    rpt = Path(args.index_report)
    sha = json.loads(rpt.read_text()).get("data_manifest_sha256") if rpt.is_file() else None
    prov = provenance(
        None, config=splits["config"], seed=splits["config"]["seed"], manifest_sha256=sha
    )
    prov["cases_sha256"] = file_sha256(cases_path)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"provenance": prov, **splits}, indent=2) + "\n")

    print(f"cases: {len(cases)}  scenarios: {cases['scenario_id'].nunique()}")
    print(f"test: {len(splits['test'])} scenarios; folds: {[len(f) for f in splits['folds']]}")
    print(
        f"chosen seed {splits['chosen_seed']}, worst-fold imbalance "
        f"(small / solo share) {splits['worst_fold_imbalance']:.3f}"
    )
    print("fold   cases pipes small  big pure solo")
    for r in splits["balance"]:
        print(
            f"{r['fold']:>4}  {r['n_cases']:>6} {r['n_pipes']:>5} {r['small']:>5} {r['big']:>4} "
            f"{r['pure']:>4} {r['solo']:>4}"
        )
    print(f"rare pipes (< {splits['config']['rare_threshold']} cases): {splits['rare_pipes']}")
    print("no scenario overlap: True (asserted)")
    print(f"-> {out}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_OK
    load_dotenv_file()
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
