"""CLI commands for M14: ``leaksense export-serve`` and ``leaksense serve``.

``export-serve`` writes one bundle (``$LEAKSENSE_ARTIFACTS_DIR/serve/bundle.pt``) with everything
the API needs, so serving never reads the raw dataset:

- the nominal network (node and pipe tables, for the map and pipe lengths),
- the headline model refit exactly as the M13 unit ``ml__<headline>_s0`` (all CV cases, seed 0),
  and ``sigmatch_all`` (no training),
- the locked-test cases (the held-out cases) with their cached ΔH windows and the distance /
  hop tables used to score a prediction.

The refit is checked against the saved M13 unit: on the same sensor sets it must give the same
per-case top-3, so the served model is the reported one. The check only re-scores predictions
already in ``reports/raw/final``; no reported number changes.

``serve`` runs the FastAPI app (``leaksense.api``) and the page ``web/index.html``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from leaksense.config import ConfigError, get_artifacts_dir
from leaksense.eval_cmds import EXIT_OK, add_common

BUNDLE_VERSION = 1
CASE_COLS = (
    "case_id",
    "scenario_id",
    "kind",
    "leak_type",
    "pipe",
    "y",
    "leak_flow_lps",
    "baseline_kind",
)


def default_bundle() -> Path:
    return get_artifacts_dir() / "serve" / "bundle.pt"


def add_parsers(sub) -> None:
    e = sub.add_parser(
        "export-serve", help="M14: save the headline model and held-out cases for `serve`"
    )
    add_common(e, default_config="configs/final.yaml")
    e.set_defaults(func=cmd_export, name="final", k=None)
    e.add_argument("--runs-dir", help="M13 unit results to check against (reports/raw/final)")
    e.add_argument("--out", help="bundle path (default $LEAKSENSE_ARTIFACTS_DIR/serve/bundle.pt)")

    s = sub.add_parser("serve", help="M14: run the API and the page on http://HOST:PORT")
    s.add_argument("--bundle", help="bundle from export-serve (default: the artifacts one)")
    s.add_argument("--reports-dir", default="reports", help="folder with final.json")
    s.add_argument("--web-dir", default="web", help="folder with index.html")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)


def _check(model, unit: str, ctx, runs_dir: Path) -> dict:
    """Re-score the refit on the M13 unit's sets at the headline k; must match the saved rows."""
    import pandas as pd

    from leaksense.evaluate import _score
    from leaksense.final_cmds import unit_masks
    from leaksense.placement import COLS, per_case_metrics

    pq = runs_dir / f"{unit}.parquet"
    if not pq.is_file():
        raise ConfigError(f"no M13 result {pq}; run `leaksense final-report` first")
    saved = pd.read_parquet(pq)
    te = ctx.test
    u = next(u for u in ctx.units if u["name"] == unit)
    i3 = list(COLS).index("top3")
    n_sets, worst = 0, 0.0
    for method, k, d, m in unit_masks(u, ctx):
        if k != ctx.hk or method not in ("ladder", "greedy"):
            continue
        got = per_case_metrics(
            _score(model, te.windows, m), te.cases["y"].to_numpy(), te.dist_to_leak, te.class_hops
        )[:, i3]
        rows = saved[(saved["method"] == method) & (saved["k"] == k) & (saved["draw"] == d)]
        want = rows.set_index("case_id").loc[te.cases["case_id"], "top3"].to_numpy()
        worst = max(worst, float(np.abs(got - want).max()))
        n_sets += 1
    if n_sets == 0:
        raise ConfigError(f"{unit} has no rows at k = {ctx.hk} to check against")
    if worst > 1e-6:
        raise ConfigError(
            f"the refit headline model differs from the M13 unit {unit} "
            f"(max top-3 difference {worst:.4f} over {n_sets} sensor sets)"
        )
    return {"unit": unit, "k": ctx.hk, "sensor_sets": n_sets, "max_top3_diff": worst}


def cmd_export(args: argparse.Namespace) -> int:
    import torch

    from leaksense.final_cmds import make_unit_model, setup
    from leaksense.provenance import provenance

    t0 = time.perf_counter()
    args.stand_in_fold = None
    ctx = setup(args)
    head = ctx.fc["headline"]
    unit = f"ml__{head}_s0"
    u = next((u for u in ctx.units if u["name"] == unit), None)
    if u is None:
        raise ConfigError(f"the final config has no unit {unit}")
    runs_dir = Path(args.runs_dir or "reports/raw/final")
    tr, te = ctx.train, ctx.test
    print(f"export-serve: fitting {head} (seed 0) on {len(tr.cases)} CV cases", flush=True)
    model = make_unit_model(u, ctx.inp)
    model.fit(tr.cases, tr.windows)
    model.inner_net, model.inner_val_idx = None, np.zeros(0, dtype=int)  # not needed to serve
    check = _check(model, unit, ctx, runs_dir)
    print(f"export-serve: matches {unit} on {check['sensor_sets']} sensor sets", flush=True)
    sm = make_unit_model({"kind": "nonlearned", "rung": "sigmatch_all"}, ctx.inp)
    sm.fit(tr.cases, tr.windows)

    cols = [c for c in CASE_COLS if c in te.cases.columns]
    bundle = {
        "version": BUNDLE_VERSION,
        "network": ctx.inp.net,
        "models": {head: model, "sigmatch_all": sm},
        "headline": head,
        "cases": te.cases[cols].reset_index(drop=True),
        "windows": [np.asarray(w, dtype=np.float32) for w in te.windows],
        "classes": list(te.classes),
        "dist_to_leak": te.dist_to_leak,
        "class_hops": te.class_hops,
        "check": check,
        "provenance": provenance(
            None,
            config=ctx.cfg.to_dict(),
            seed=0,
            manifest_sha256=ctx.inp.meta.get("data_manifest_sha256"),
        ),
        "features": ctx.inp.fdir.name,
    }
    out = Path(args.out) if args.out else default_bundle()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    torch.save(bundle, tmp)
    tmp.replace(out)
    secs = time.perf_counter() - t0
    print(f"export-serve: {len(te.cases)} held-out cases -> {out} ({secs:.0f}s)")
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from leaksense.api import create_app

    app = create_app(args.bundle, args.reports_dir, args.web_dir)
    print(f"serving on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return EXIT_OK
