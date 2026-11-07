"""CLI command for M6: ``leaksense eval``."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from leaksense.config import DATA_DIR_VAR, ConfigError, get_artifacts_dir, get_data_dir

EXIT_OK = 0


def add_common(e, default_config: str = "configs/full.yaml") -> None:
    """Arguments shared by ``eval`` and ``train``."""
    from leaksense.evaluate import DEFAULT_KS

    e.add_argument(
        "--k",
        default=",".join(map(str, DEFAULT_KS)),
        help="sensor budgets, comma separated (31 = all junctions)",
    )
    e.add_argument("--config", default=default_config, help="run config (selects features)")
    e.add_argument("--name", help="report name (default: the model name)")
    e.add_argument("--draws", type=int, default=20, help="random sensor sets per k < 31")
    e.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--splits", default="splits/splits_v1.json")
    e.add_argument("--reports-dir", default="reports")
    e.add_argument("--data-dir", help=f"dataset HanoiOK folder (default: ${DATA_DIR_VAR})")
    e.add_argument("--contract", help="JSON file overriding the default data contract")


def add_parsers(sub) -> None:
    from leaksense.models.baselines import MODELS

    e = sub.add_parser("eval", help="score a model on the CV cases (reports/<name>.json, .md)")
    e.add_argument("--model", required=True, choices=sorted(MODELS))
    add_common(e)
    e.set_defaults(func=cmd_eval)


@dataclass
class EvalInputs:
    cfg: object  # RunConfig
    fdir: Path
    meta: dict
    es: object  # EvalSet
    net: object  # Network
    ks: list[int]


def load_inputs(args: argparse.Namespace) -> EvalInputs:
    """Features, splits, network and k list for ``eval`` / ``train``."""
    from leaksense.data.contract import DataContract
    from leaksense.evaluate import build_eval_set
    from leaksense.network import load_nominal_network
    from leaksense.runconfig import load_run_config

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
    return EvalInputs(cfg, fdir, meta, es, net, ks)


def load_sensitivity(inp: EvalInputs) -> dict:
    from leaksense import sensitivity as sens

    manifest_sha = inp.meta.get("data_manifest_sha256") or ""
    spath = sens.sensitivity_path(get_artifacts_dir(), manifest_sha)
    if not spath.is_file():
        raise ConfigError(f"no sensitivity matrix at {spath}; run `leaksense build-sensitivity`")
    S = sens.load(spath)
    if str(S["manifest"]) != manifest_sha:
        raise ConfigError("sensitivity matrix and features come from different datasets")
    return S


def run_and_report(
    args: argparse.Namespace,
    inp: EvalInputs,
    model,
    model_name: str,
    t0: float,
    folds=None,
    extra=None,
) -> dict:
    """Evaluate ``model``, write reports/<name>.json and .md, print the k table.

    ``extra`` (a dict, or a function returning one, called after evaluation) adds report keys.
    """
    from leaksense.evaluate import evaluate, write_report
    from leaksense.provenance import file_sha256, provenance

    results, preds = evaluate(
        model, inp.es, inp.ks, draws=args.draws, n_boot=args.bootstrap, seed=args.seed, folds=folds
    )
    n_cases = results[str(inp.ks[0])]["groups"]["overall"]["n"]
    name = args.name or model_name
    report = {
        "name": name,
        "model": model_name,
        "provenance": {
            **provenance(
                None,
                config=inp.cfg.to_dict(),
                seed=args.seed,
                manifest_sha256=inp.meta.get("data_manifest_sha256"),
            ),
            "splits_sha256": file_sha256(args.splits),
        },
        "features_dir": inp.fdir.name,
        "features_sha256_12": inp.meta.get("features_sha256_12"),
        "n_cases": n_cases,
        "ks": inp.ks,
        "draws": args.draws,
        "bootstrap": args.bootstrap,
        "tercile_edges": inp.es.tercile_edges,
        **((extra() if callable(extra) else extra) or {}),
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "results": results,
    }
    out = write_report(report, preds, Path(args.reports_dir), get_artifacts_dir())

    print(f"{name}: {n_cases} CV cases, features {inp.fdir.name}, ks {inp.ks}")
    print("   k   top-1   top-3   top-5   small top-3   median dist (m)")
    for k, r in results.items():
        o, s = r["groups"]["overall"], r["groups"].get("leak_type=small", {})
        st3 = f"{100 * s['top3']['value']:9.1f}" if s else "        -"
        print(
            f"{k:>4} {100 * o['top1']['value']:7.1f} {100 * o['top3']['value']:7.1f} "
            f"{100 * o['top5']['value']:7.1f}   {st3}     {o['dist_m_median']['value']:9.0f}"
        )
    print(f"done in {report['elapsed_seconds']:.0f}s -> {out}, {out.with_suffix('.md')}")
    return report


def cmd_eval(args: argparse.Namespace) -> int:
    from leaksense.models.baselines import MODELS, NEEDS_SENSITIVITY

    t0 = time.perf_counter()
    inp = load_inputs(args)
    resources = {}
    if args.model in NEEDS_SENSITIVITY:
        resources["sensitivity"] = load_sensitivity(inp)
    model = MODELS[args.model](inp.net, **resources)
    run_and_report(args, inp, model, args.model, t0)
    return EXIT_OK
