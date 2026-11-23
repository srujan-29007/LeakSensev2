"""CLI command for M11: ``leaksense place``.

Per outer fold: train the model once (``fit``), run the learned greedy on the fold's
inner-validation cases with the pre-refit net, then score the greedy, coverage and random sensor
sets on the outer cases with the refit net and with ``sigmatch_all``. Each fold is saved to
``<runs-dir>/<name>__f<fold>.{json,parquet}`` as it finishes (the ``.json`` last), so an
interrupted run resumes at the next fold. A "final" placement is then chosen the same way from
all CV cases (for M13 and M14). The locked test fold is never read.

Writes ``reports/<name>.{json,md}`` and ``reports/<name>_curve.png``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from leaksense.config import ConfigError
from leaksense.eval_cmds import EXIT_OK, add_common, load_inputs, load_sensitivity
from leaksense.placement import (
    COLS,
    as_mask,
    greedy_coverage,
    greedy_learned,
    mean_pairwise_jaccard,
    per_case_metrics,
)

REPORT_GROUPS = (
    "overall",
    "leak_type=small",
    "leak_type=big",
    "leak_flow=T1",
    "leak_flow=T2",
    "leak_flow=T3",
)
METHODS = ("greedy", "coverage", "random")
SCORERS = ("model", "sigmatch_all")


def add_parsers(sub) -> None:
    p = sub.add_parser(
        "place", help="greedy sensor placement vs random and coverage (reports/placement.*)"
    )
    add_common(p, default_config="configs/place_cpu.yaml")
    p.set_defaults(k=None, name="placement", func=cmd_place)
    p.add_argument("--random", type=int, help="random placements per k (default: config, 100)")
    p.add_argument("--fold", type=int, help="run only this outer fold (a smoke run, no report)")
    p.add_argument("--no-final", action="store_true", help="skip the all-CV final placement")
    p.add_argument("--runs-dir", default="reports/raw/placement", help="per-fold results")
    p.add_argument("--aggregate-only", action="store_true", help="only aggregate saved folds")


# ----------------------------------------------------------------------------- runs


def _key(fp: dict) -> str:
    return hashlib.sha256(json.dumps(fp, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _done(runs_dir: Path, name: str, key: str, need_parquet: bool = True) -> bool:
    js = runs_dir / f"{name}.json"
    if not js.is_file() or (need_parquet and not (runs_dir / f"{name}.parquet").is_file()):
        return False
    try:
        return json.loads(js.read_text()).get("key") == key
    except json.JSONDecodeError:
        return False


def _save(runs_dir: Path, name: str, meta: dict, df: pd.DataFrame | None) -> None:
    stage = runs_dir / ".partial"
    stage.mkdir(parents=True, exist_ok=True)
    if df is not None:
        df.to_parquet(stage / f"{name}.parquet", index=False)
        os.replace(stage / f"{name}.parquet", runs_dir / f"{name}.parquet")
    (stage / f"{name}.json").write_text(json.dumps(meta, indent=1, default=str))
    os.replace(stage / f"{name}.json", runs_dir / f"{name}.json")


def _fit_and_greedy(inp, model_cfg: dict, train_idx: np.ndarray, k_max: int):
    """Fit on ``train_idx`` cases; greedy on the inner-validation cases with the pre-refit net."""
    import torch

    from leaksense.models.mlp import net_case_scores
    from leaksense.train_cmds import make_model

    es = inp.es
    model = make_model(inp, model_cfg)
    train = es.cases.iloc[train_idx].reset_index(drop=True)
    windows = [es.windows[i] for i in train_idx]
    model.fit(train, windows)
    n_sensors = len(inp.net.sensor_ids)
    inner_net = getattr(model, "inner_net", None)
    vidx = np.asarray(getattr(model, "inner_val_idx", []), dtype=int)
    if hasattr(model.cfg, "num_threads"):
        torch.set_num_threads(model.cfg.num_threads)
    if inner_net is not None and len(vidx):
        xs = [windows[i] for i in vidx]
        ys = train["y"].to_numpy()[vidx]

        def score_fn(m):
            return net_case_scores(inner_net, model.cfg.transform, [x * m for x in xs], m)

        in_sample = False
    else:  # no inner split (tiny data): greedy on the training cases with the refit net
        xs, ys = windows, train["y"].to_numpy()

        def score_fn(m):
            return model.batch_scores([x * m for x in xs], m)

        in_sample = True
    order, trace = greedy_learned(score_fn, ys, n_sensors, k_max)
    info = {
        "greedy_order": [inp.net.sensor_ids[j] for j in order],
        "greedy_order_idx": order,
        "greedy_trace": trace,
        "n_greedy_cases": int(len(ys)),
        "greedy_in_sample": in_sample,
        "training": model.describe()["folds"][-1] if hasattr(model, "describe") else {},
    }
    return model, info


def run_fold(inp, model_cfg, sig, f: int, ks, rand_masks, cov_order) -> tuple[dict, pd.DataFrame]:
    from leaksense.evaluate import _score

    es = inp.es
    folds = es.cases["fold"].to_numpy()
    n_sensors = len(inp.net.sensor_ids)
    model, info = _fit_and_greedy(inp, model_cfg, np.flatnonzero(folds != f), max(ks))
    order = info["greedy_order_idx"]

    idx = np.flatnonzero(folds == f)
    xs = [es.windows[i] for i in idx]
    ys = es.cases["y"].to_numpy()[idx]
    dist = es.dist_to_leak[idx]
    case_ids = es.cases["case_id"].to_numpy()[idx]
    frames = []
    for k in ks:
        sets = {
            "greedy": [as_mask(order[:k], n_sensors)],
            "coverage": [as_mask(cov_order[:k], n_sensors)],
            "random": list(rand_masks[k]),
        }
        for method, masks in sets.items():
            for d, m in enumerate(masks):
                for scorer, mdl in (("model", model), ("sigmatch_all", sig)):
                    vals = per_case_metrics(_score(mdl, xs, m), ys, dist, es.class_hops)
                    df = pd.DataFrame(vals, columns=list(COLS))
                    df.insert(0, "case_id", case_ids)
                    df.insert(0, "draw", d)
                    df.insert(0, "k", k)
                    df.insert(0, "scorer", scorer)
                    df.insert(0, "method", method)
                    frames.append(df)
    return info, pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------- aggregate


def _summ(values: np.ndarray, scen: np.ndarray, n_boot: int, seed: int) -> dict:
    from leaksense.evaluate import bootstrap, summarise

    point, ci = summarise(values), bootstrap(values, scen, n_boot, seed)
    return {m: {"value": point[m], "lo": ci[m][0], "hi": ci[m][1]} for m in ("top1", "top3")} | {
        "dist_m_median": point["dist_m_median"],
        "n": int(len(values)),
    }


def aggregate(inp, df: pd.DataFrame, ks, n_rand: int, n_boot: int, seed: int) -> dict:
    from leaksense.ablate import _paired
    from leaksense.evaluate import groups

    es = inp.es
    pos = {c: i for i, c in enumerate(es.cases["case_id"])}
    n = len(es.cases)
    scen = es.cases["scenario_id"].to_numpy()
    grp = {g: s for g, s in groups(es).items() if g in REPORT_GROUPS}
    out: dict = {}
    for scorer in SCORERS:
        sub = df[df["scorer"] == scorer]
        out[scorer] = {}
        for k in ks:
            arr = {}
            for method in METHODS:
                part = sub[(sub["method"] == method) & (sub["k"] == k)]
                nd = int(part["draw"].max()) + 1
                a = np.full((nd, n, len(COLS)), np.nan)
                a[part["draw"].to_numpy(), part["case_id"].map(pos).to_numpy()] = part[
                    list(COLS)
                ].to_numpy()
                if np.isnan(a).any():
                    raise ValueError(f"missing cases for {scorer}/{method}/k={k}")
                arr[method] = a
            g, c, r = arr["greedy"][0], arr["coverage"][0], arr["random"]
            rmean = r.mean(axis=0)
            res = {}
            for gname, sel in grp.items():
                if not sel.any():
                    continue
                draws_top3 = r[:, sel, 1].mean(axis=1)
                gt3 = g[sel, 1].mean()
                res[gname] = {
                    "greedy": _summ(g[sel], scen[sel], n_boot, seed),
                    "coverage": _summ(c[sel], scen[sel], n_boot, seed),
                    "random_mean": _summ(rmean[sel], scen[sel], n_boot, seed),
                    "random_top3_draws": {
                        q: float(np.percentile(draws_top3, p))
                        for q, p in (("p5", 5), ("p50", 50), ("p95", 95))
                    },
                    "greedy_percentile": float(
                        ((draws_top3 < gt3).sum() + 0.5 * (draws_top3 == gt3).sum())
                        / len(draws_top3)
                    ),
                    "coverage_percentile": float(
                        (
                            (draws_top3 < c[sel, 1].mean()).sum()
                            + 0.5 * (draws_top3 == c[sel, 1].mean()).sum()
                        )
                        / len(draws_top3)
                    ),
                    "greedy_minus_random": _paired(g[sel] - rmean[sel], scen[sel], n_boot, seed),
                    "greedy_minus_coverage": _paired(g[sel] - c[sel], scen[sel], n_boot, seed),
                }
            out[scorer][str(k)] = res
    return out


def stability(fold_infos: dict, ks, rand_masks, sensor_ids) -> dict:
    out = {}
    for k in ks:
        sets = [info["greedy_order"][:k] for _, info in sorted(fold_infos.items())]
        rsets = [[sensor_ids[j] for j in np.flatnonzero(m)] for m in rand_masks[k]]
        freq = Counter(j for s in sets for j in s)
        out[str(k)] = {
            "sets": sets,
            "mean_pairwise_jaccard": mean_pairwise_jaccard(sets),
            "random_mean_pairwise_jaccard": mean_pairwise_jaccard(rsets),
            "pick_counts": dict(sorted(freq.items(), key=lambda t: (-t[1], int(t[0])))),
        }
    return out


# ----------------------------------------------------------------------------- report


def _pct(d: dict) -> str:
    return f"{100 * d['value']:.1f} [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}]"


def _diff(d: dict) -> str:
    return f"{100 * d['value']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]"


def render_markdown(rep: dict) -> str:
    prov = rep["provenance"]
    git = prov.get("git") or {}
    commit = (git.get("commit") or "?")[:7] + (" (dirty)" if git.get("dirty") else "")
    ks = rep["ks"]
    L = [
        f"# Sensor placement: {rep['name']} (PLAN.md M11)",
        "",
        f"- **Generated by:** `leaksense place --config` (config `{prov['config'].get('name')}`)"
        f" at commit `{commit}`.",
        f"- **Data:** manifest `{(prov.get('data_manifest_sha256') or '')[:12]}…`, features "
        f"`{rep['features_dir']}`, splits sha `{rep['splits_sha256'][:12]}`.",
        f"- **Model:** `{rep['model_label']}` (model seed {rep['model_seed']}), held fixed. "
        f"Cases: {rep['n_cases']} CV cases, scored on their outer fold. The locked test fold is "
        "not used.",
        "- **greedy:** forward selection maximising inner-validation top-3 with the fold's "
        "pre-refit net (cases the net was not trained on); sets are nested.",
        "- **coverage:** physics-only greedy on S (mean over S columns of the largest scaled |S| "
        "among chosen junctions); one set for all folds.",
        f"- **random:** {rep['n_random']} seeded random sets per k, the same in every fold. "
        '"random mean" averages each case over them; p5–p95 is the spread of pooled top-3 '
        "over the sets.",
        "- **CIs:** 95% bootstrap over scenarios. Paired differences are per case on the same "
        "cases.",
        "",
    ]
    for scorer, title in (("model", rep["model_label"]), ("sigmatch_all", "sigmatch_all")):
        for g, gt in (("overall", "all cases"), ("leak_type=small", "small leaks")):
            L += [
                f"## Top-3 (%), {title}, {gt}",
                "",
                "| k | greedy | coverage | random mean | random p5–p95 | greedy pct | "
                "greedy − random | greedy − coverage |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for k in ks:
                r = rep["results"][scorer][str(k)].get(g)
                if not r:
                    continue
                dr = r["random_top3_draws"]
                L.append(
                    f"| {k} | {_pct(r['greedy']['top3'])} | {_pct(r['coverage']['top3'])} | "
                    f"{_pct(r['random_mean']['top3'])} | {100 * dr['p5']:.1f}–"
                    f"{100 * dr['p95']:.1f} | {100 * r['greedy_percentile']:.0f} | "
                    f"{_diff(r['greedy_minus_random']['top3'])} | "
                    f"{_diff(r['greedy_minus_coverage']['top3'])} |"
                )
            L.append("")
    L += [
        f"## Greedy by leak-flow tercile, top-3 (%), {rep['model_label']}",
        "",
        "| k | T1 greedy | T1 random | T2 greedy | T2 random | T3 greedy | T3 random |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in ks:
        r = rep["results"]["model"][str(k)]
        cells = []
        for t in ("T1", "T2", "T3"):
            x = r.get(f"leak_flow={t}")
            cells += (
                [f"{100 * x['greedy']['top3']['value']:.1f}"]
                + [f"{100 * x['random_mean']['top3']['value']:.1f}"]
                if x
                else ["-", "-"]
            )
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L += [
        "",
        "## Stability of greedy sets across folds",
        "",
        "| k | mean pairwise Jaccard | random sets (reference) | most picked (folds) |",
        "|---|---|---|---|",
    ]
    for k in ks:
        s = rep["stability"][str(k)]
        top = ", ".join(f"{j} ({c})" for j, c in list(s["pick_counts"].items())[:8])
        L.append(
            f"| {k} | {s['mean_pairwise_jaccard']:.2f} | "
            f"{s['random_mean_pairwise_jaccard']:.2f} | {top} |"
        )
    L += ["", "Greedy order per fold (junction ids):", ""]
    for f, info in sorted(rep["folds"].items(), key=lambda t: int(t[0])):
        tag = " (in-sample)" if info["greedy_in_sample"] else ""
        L.append(
            f"- fold {f}{tag}, {info['n_greedy_cases']} inner cases: "
            + ", ".join(info["greedy_order"])
        )
    L += ["", f"Coverage order: {', '.join(rep['coverage_order'])}", ""]
    if rep.get("final"):
        fin = rep["final"]
        L += [
            "## Final placement (all CV cases; for M13 and M14)",
            "",
            f"Greedy on {fin['n_greedy_cases']} inner-validation cases of a model fit on all "
            f"CV cases. Order: {', '.join(fin['greedy_order'])}.",
            "",
            "| k | junctions | inner top-3 |",
            "|---|---|---|",
        ]
        for k in ks:
            L.append(
                f"| {k} | {', '.join(fin['greedy_order'][:k])} | "
                f"{100 * fin['greedy_trace'][k - 1]['top3']:.1f} |"
            )
        L.append("")
    L.append(f"Elapsed: {rep['elapsed_seconds']:.0f} s (fold runs: {rep['fold_seconds']}).")
    return "\n".join(L) + "\n"


def plot_curve(rep: dict, path: Path, title: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = rep["ks"]
    panels = [
        (g, t)
        for g, t in (("overall", "All cases"), ("leak_type=small", "Small leaks"))
        if g in rep["results"]["model"][str(ks[0])]
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (g, panel) in zip(axes, panels, strict=True):
        res = rep["results"]["model"]

        def get(m, g=g, res=res):
            return [100 * res[str(k)][g][m]["top3"]["value"] for k in ks]

        lo = [100 * res[str(k)][g]["random_top3_draws"]["p5"] for k in ks]
        hi = [100 * res[str(k)][g]["random_top3_draws"]["p95"] for k in ks]
        ax.fill_between(ks, lo, hi, color="0.85", label="random p5–p95")
        ax.plot(ks, get("random_mean"), color="0.45", marker=".", label="random mean")
        ax.plot(ks, get("coverage"), color="tab:orange", marker="s", label="coverage greedy")
        ax.plot(ks, get("greedy"), color="tab:blue", marker="o", label="learned greedy")
        sres = rep["results"]["sigmatch_all"]
        ax.plot(
            ks,
            [100 * sres[str(k)][g]["random_mean"]["top3"]["value"] for k in ks],
            color="0.45",
            ls="--",
            label="sigmatch_all, random mean",
        )
        ax.set_title(panel)
        ax.set_xlabel("sensors (k)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("top-3 accuracy (%)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(title or f"Sensor placement, {rep['model_label']} (CV, outer folds)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- command


def cmd_place(args: argparse.Namespace) -> int:
    from leaksense.evaluate import random_masks
    from leaksense.models.baselines import SigMatchAll, signatures
    from leaksense.provenance import file_sha256, provenance
    from leaksense.runconfig import load_run_config

    t0 = time.perf_counter()
    cfg = load_run_config(args.config)
    pcfg = dict(cfg.placement)
    if args.k is None:
        args.k = ",".join(str(k) for k in pcfg.get("k", [2, 3, 4, 6, 8, 12, 16]))
    n_rand = int(args.random or pcfg.get("random", 100))
    inp = load_inputs(args)
    ks = sorted(inp.ks)
    if not cfg.model:
        raise ConfigError("the placement config needs a model: section")
    S = load_sensitivity(inp)
    sig = SigMatchAll(inp.net, S)
    cov_order = greedy_coverage(signatures(inp.net, S, "all"), max(ks))
    n_sensors = len(inp.net.sensor_ids)
    rand_masks = {k: random_masks(n_sensors, k, n_rand, args.seed) for k in ks}

    fp = {
        "features_dir": inp.fdir.name,
        "features_sha256_12": inp.meta.get("features_sha256_12"),
        "splits_sha256": file_sha256(args.splits),
        "ks": ks,
        "n_random": n_rand,
        "seed": int(args.seed),
        "model": cfg.model,
    }
    key = _key(fp)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    all_folds = sorted(int(f) for f in inp.es.cases["fold"].unique())
    if args.fold is not None and args.fold not in all_folds:
        raise ConfigError(f"--fold must be one of {all_folds}")
    todo = [args.fold] if args.fold is not None else all_folds

    if not args.aggregate_only:
        for f in todo:
            name = f"{args.name}__f{f}"
            if _done(runs_dir, name, key):
                print(f"{name}: done, skipped")
                continue
            t1 = time.perf_counter()
            print(f"{name}: training and greedy on fold {f}'s training cases", flush=True)
            info, df = run_fold(inp, cfg.model, sig, f, ks, rand_masks, cov_order)
            secs = round(time.perf_counter() - t1, 1)
            _save(runs_dir, name, {"key": key, "fp": fp, "fold": f, "seconds": secs, **info}, df)
            g = df[(df["method"] == "greedy") & (df["scorer"] == "model")]
            r = df[(df["method"] == "random") & (df["scorer"] == "model")]
            print(f"{name}: greedy order {', '.join(info['greedy_order'])}")
            print("   k   greedy top-3   random mean top-3")
            for k in ks:
                print(
                    f"{k:>4}   {100 * g[g['k'] == k]['top3'].mean():12.1f}   "
                    f"{100 * r[r['k'] == k]['top3'].mean():17.1f}"
                )
            print(f"done in {secs:.0f}s", flush=True)
        if args.fold is not None:
            print("single fold: no report written")
            return EXIT_OK

        fname = f"{args.name}__final"
        if not args.no_final and not _done(runs_dir, fname, key, need_parquet=False):
            t1 = time.perf_counter()
            print(f"{fname}: training on all {len(inp.es.cases)} CV cases", flush=True)
            _, info = _fit_and_greedy(inp, cfg.model, np.arange(len(inp.es.cases)), max(ks))
            secs = round(time.perf_counter() - t1, 1)
            _save(runs_dir, fname, {"key": key, "fp": fp, "seconds": secs, **info}, None)
            print(f"{fname}: order {', '.join(info['greedy_order'])} ({secs:.0f}s)", flush=True)

    infos, frames, missing = {}, [], []
    for f in all_folds:
        name = f"{args.name}__f{f}"
        if not _done(runs_dir, name, key):
            missing.append(f)
            continue
        infos[f] = json.loads((runs_dir / f"{name}.json").read_text())
        frames.append(pd.read_parquet(runs_dir / f"{name}.parquet"))
    if missing:
        raise ConfigError(f"folds {missing} have no saved run with this config; run them first")
    final = None
    if _done(runs_dir, f"{args.name}__final", key, need_parquet=False):
        final = json.loads((runs_dir / f"{args.name}__final.json").read_text())

    df = pd.concat(frames, ignore_index=True)
    print(f"aggregating {len(all_folds)} folds", flush=True)
    results = aggregate(inp, df, ks, n_rand, args.bootstrap, args.seed)
    label = pcfg.get("model_label") or cfg.model.get("kind")
    rep = {
        "name": args.name,
        "command": "leaksense place",
        "provenance": provenance(
            None,
            config=cfg.to_dict(),
            seed=args.seed,
            manifest_sha256=inp.meta.get("data_manifest_sha256"),
        ),
        "splits_sha256": file_sha256(args.splits),
        "features_dir": inp.fdir.name,
        "features_sha256_12": inp.meta.get("features_sha256_12"),
        "model_label": label,
        "model_seed": cfg.model.get("seed", 0),
        "n_cases": len(inp.es.cases),
        "ks": ks,
        "n_random": n_rand,
        "bootstrap": args.bootstrap,
        "tercile_edges": inp.es.tercile_edges,
        "coverage_order": [inp.net.sensor_ids[j] for j in cov_order],
        "folds": {str(f): {k: v for k, v in i.items() if k != "fp"} for f, i in infos.items()},
        "fold_seconds": [infos[f]["seconds"] for f in all_folds],
        "final": {k: v for k, v in final.items() if k != "fp"} if final else None,
        "stability": stability(infos, ks, rand_masks, inp.net.sensor_ids),
        "results": results,
    }
    rep["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    out_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / f"{args.name}.json"
    js.write_text(json.dumps(rep, indent=1, default=str))
    js.with_suffix(".md").write_text(render_markdown(rep), encoding="utf-8")
    png = out_dir / f"{args.name}_curve.png"
    plot_curve(rep, png)
    print("   k   greedy   coverage   random mean   (top-3, all cases)")
    for k in ks:
        r = results["model"][str(k)]["overall"]
        print(
            f"{k:>4}   {100 * r['greedy']['top3']['value']:6.1f}   "
            f"{100 * r['coverage']['top3']['value']:8.1f}   "
            f"{100 * r['random_mean']['top3']['value']:11.1f}"
        )
    print(f"done in {rep['elapsed_seconds']:.0f}s -> {js}, {js.with_suffix('.md')}, {png}")
    return EXIT_OK
