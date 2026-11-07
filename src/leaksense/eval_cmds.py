"""CLI command for M6: ``leaksense eval``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from leaksense.config import DATA_DIR_VAR, ConfigError, get_artifacts_dir, get_data_dir

EXIT_OK = 0


def add_parsers(sub) -> None:
    from leaksense.evaluate import DEFAULT_KS
    from leaksense.models.baselines import MODELS

    e = sub.add_parser("eval", help="score a model on the CV cases (reports/<name>.json, .md)")
    e.add_argument("--model", required=True, choices=sorted(MODELS))
    e.add_argument(
        "--k",
        default=",".join(map(str, DEFAULT_KS)),
        help="sensor budgets, comma separated (31 = all junctions)",
    )
    e.add_argument("--config", default="configs/full.yaml", help="run config (selects features)")
    e.add_argument("--name", help="report name (default: the model name)")
    e.add_argument("--draws", type=int, default=20, help="random sensor sets per k < 31")
    e.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--splits", default="splits/splits_v1.json")
    e.add_argument("--reports-dir", default="reports")
    e.add_argument("--data-dir", help=f"dataset HanoiOK folder (default: ${DATA_DIR_VAR})")
    e.add_argument("--contract", help="JSON file overriding the default data contract")
    e.set_defaults(func=cmd_eval)


def cmd_eval(args: argparse.Namespace) -> int:
    from leaksense.data.contract import DataContract
    from leaksense.evaluate import build_eval_set, evaluate, write_report
    from leaksense.models.baselines import MODELS
    from leaksense.network import load_nominal_network
    from leaksense.provenance import file_sha256, provenance
    from leaksense.runconfig import load_run_config

    t0 = time.perf_counter()
    data_dir = get_data_dir({DATA_DIR_VAR: args.data_dir} if args.data_dir else None)
    contract = DataContract.from_json(args.contract) if args.contract else DataContract()
    cfg = load_run_config(args.config)
    fdir = get_artifacts_dir() / "features" / cfg.hash()
    if not (fdir / "features.npz").is_file():
        raise ConfigError(
            f"no features at {fdir}; run `leaksense build-features --config {args.config}` first"
        )
    meta = json.loads((fdir / "meta.json").read_text())
    with np.load(fdir / "features.npz") as z:
        feats = {k: z[k] for k in z.files}
    case_stats = pd.read_parquet(fdir / "case_stats.parquet")
    splits = json.loads(Path(args.splits).read_text())
    net = load_nominal_network(data_dir, contract)
    es = build_eval_set(case_stats, feats, splits, net)
    ks = [int(k) for k in str(args.k).split(",")]
    n_sensors = len(net.sensor_ids)
    if any(k < 1 or k > n_sensors for k in ks):
        raise ConfigError(f"--k values must be in 1..{n_sensors}")
    ks = [min(k, n_sensors) for k in ks]

    from leaksense import sensitivity as sens
    from leaksense.models.baselines import NEEDS_SENSITIVITY

    resources = {}
    if args.model in NEEDS_SENSITIVITY:
        manifest_sha = meta.get("data_manifest_sha256") or ""
        spath = sens.sensitivity_path(get_artifacts_dir(), manifest_sha)
        if not spath.is_file():
            raise ConfigError(
                f"no sensitivity matrix at {spath}; run `leaksense build-sensitivity`"
            )
        resources["sensitivity"] = sens.load(spath)
        if str(resources["sensitivity"]["manifest"]) != manifest_sha:
            raise ConfigError("sensitivity matrix and features come from different datasets")
    model = MODELS[args.model](net, **resources)
    results, preds = evaluate(
        model, es, ks, draws=args.draws, n_boot=args.bootstrap, seed=args.seed
    )
    name = args.name or args.model
    report = {
        "name": name,
        "model": args.model,
        "provenance": {
            **provenance(
                None,
                config=cfg.to_dict(),
                seed=args.seed,
                manifest_sha256=meta.get("data_manifest_sha256"),
            ),
            "splits_sha256": file_sha256(args.splits),
        },
        "features_dir": fdir.name,
        "features_sha256_12": meta.get("features_sha256_12"),
        "n_cases": len(es.cases),
        "ks": ks,
        "draws": args.draws,
        "bootstrap": args.bootstrap,
        "tercile_edges": es.tercile_edges,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "results": results,
    }
    out = write_report(report, preds, Path(args.reports_dir), get_artifacts_dir())

    print(f"{name}: {len(es.cases)} CV cases, features {fdir.name}, ks {ks}")
    print("   k   top-1   top-3   top-5   small top-3   median dist (m)")
    for k, r in results.items():
        o, s = r["groups"]["overall"], r["groups"].get("leak_type=small", {})
        st3 = f"{100 * s['top3']['value']:9.1f}" if s else "        -"
        print(
            f"{k:>4} {100 * o['top1']['value']:7.1f} {100 * o['top3']['value']:7.1f} "
            f"{100 * o['top5']['value']:7.1f}   {st3}     {o['dist_m_median']['value']:9.0f}"
        )
    print(f"done in {report['elapsed_seconds']:.0f}s -> {out}, {out.with_suffix('.md')}")
    return EXIT_OK
