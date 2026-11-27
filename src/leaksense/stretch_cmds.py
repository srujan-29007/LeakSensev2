"""CLI for M15: ``leaksense eval --task detect`` and ``leaksense eval --stress overlap``.

Both use the CV scenarios only (never the locked test). See :mod:`leaksense.stretch` and PLAN.md
M15 for the design, which was fixed before any result.

- detect: writes ``reports/detection.{json,md}``.
- overlap: per-fold units in ``reports/raw/overlap/`` (resumable; ``--fold`` runs one), then
  ``reports/overlap.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from leaksense.config import DATA_DIR_VAR, ConfigError, get_artifacts_dir, get_data_dir
from leaksense.eval_cmds import EXIT_OK, load_inputs, load_sensitivity

ALPHAS = (0.01, 0.05)
DETECT_KS = (2, 4, 8, 16, 31)
OVERLAP_KS = (4, 8, 16, 31)


# ----------------------------------------------------------------------------- shared


def _setup(args):
    from leaksense.data.contract import DataContract
    from leaksense.data.features import PressureCache, slot_index

    args.k = "1"  # M15 picks its own sensor budgets (DETECT_KS / OVERLAP_KS)
    inp = load_inputs(args)
    data_dir = get_data_dir({DATA_DIR_VAR: args.data_dir} if args.data_dir else None)
    contract = DataContract.from_json(args.contract) if args.contract else DataContract()
    sha = inp.meta.get("data_manifest_sha256") or ""
    cache = PressureCache(data_dir, get_artifacts_dir(), sha, contract)
    c = inp.es.cases
    folds = dict(zip(c["scenario_id"].astype(int), c["fold"].astype(int), strict=True))
    slots = (slot_index(contract, "tod"), slot_index(contract, "tod_weekday"))
    return inp, data_dir, contract, cache, sha, folds, slots


def _leaks(data_dir, contract, sha: str, sids) -> pd.DataFrame:
    from leaksense.stretch import scan_leaks

    path = get_artifacts_dir() / "stretch" / f"leaks_{sha[:12]}.parquet"
    if path.is_file():
        df = pd.read_parquet(path)
        if set(df["scenario_id"]) >= set(sids):
            return df[df["scenario_id"].isin(sids)].reset_index(drop=True)
    print(f"scanning leaks of {len(sids)} CV scenarios", flush=True)
    df = scan_leaks(Path(data_dir), contract, sids, progress=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def _greedy_order(placement_runs: str) -> list[int]:
    path = Path(placement_runs) / "placement__final.json"
    if not path.is_file():
        raise ConfigError(f"no M11 final placement at {path}")
    return [int(j) for j in json.loads(path.read_text())["greedy_order_idx"]]


def _mask_sets(n: int, ks, draws: int, seed: int, order: list[int]) -> list[tuple]:
    """(method, k, draw, mask): the harness random sets, plus the greedy prefix where it exists."""
    from leaksense.evaluate import random_masks
    from leaksense.placement import as_mask

    out = []
    for k in ks:
        for d, m in enumerate(random_masks(n, k, draws, seed)):
            out.append(("random", k, d, m))
        if k <= len(order):
            out.append(("greedy", k, 0, as_mask(order[:k], n)))
        elif k == n:
            out.append(("greedy", k, 0, np.ones(n, dtype=bool)))
    return out


def _boot_mean(v: np.ndarray, scen: np.ndarray, n_boot: int, seed: int) -> list[float]:
    """Percentile 95% CI of the mean of ``v``, resampling scenarios."""
    if len(v) == 0:
        return [float("nan"), float("nan")]
    uniq, inv = np.unique(scen, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _stat(v, scen, n_boot, seed) -> dict:
    v = np.asarray(v, dtype=float)
    lo, hi = _boot_mean(v, np.asarray(scen), n_boot, seed)
    return {"value": float(v.mean()) if len(v) else float("nan"), "lo": lo, "hi": hi, "n": len(v)}


def _provenance(inp, args, extra: dict) -> dict:
    from leaksense.provenance import file_sha256, provenance

    return {
        **provenance(
            None,
            config=inp.cfg.to_dict(),
            seed=args.seed,
            manifest_sha256=inp.meta.get("data_manifest_sha256"),
        ),
        "splits_sha256": file_sha256(args.splits),
        "features_dir": inp.fdir.name,
        **extra,
    }


def _write(report: dict, md: str, reports_dir: str, name: str) -> Path:
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    js = out / f"{name}.json"
    js.write_bytes(json.dumps(report, indent=1, default=str).encode("utf-8"))
    (out / f"{name}.md").write_bytes(md.encode("utf-8"))
    return js


def _pct(d: dict | None) -> str:
    if not d or d.get("value") is None or not np.isfinite(d["value"]):
        return "-"
    return f"{100 * d['value']:.1f} [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}]"


# ----------------------------------------------------------------------------- detection


def build_detection_samples(inp, data_dir, contract, cache, sha, folds, slots, cfg):
    from leaksense.stretch import detection_samples

    key = json.dumps({"cfg": asdict(cfg), "sids": sorted(folds), "sha": sha}, sort_keys=True)
    import hashlib

    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    path = get_artifacts_dir() / "stretch" / f"detect_{h}.npz"
    leaks = _leaks(data_dir, contract, sha, sorted(folds))
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            out = {k: z[k] for k in z.files}
        return leaks, out
    neg_x, neg_s, pos_x, pos_i, ttd_x, ttd_i = [], [], [], [], [], []
    sids = sorted(folds)
    t0 = time.perf_counter()
    for n, sid in enumerate(sids, 1):
        if n % 50 == 0:
            print(
                f"  detection samples {n}/{len(sids)} ({time.perf_counter() - t0:.0f}s)", flush=True
            )
        p, labels = cache.get(sid), cache.labels(sid)
        lk = leaks[leaks["scenario_id"] == sid]
        nx, _, pos = detection_samples(p, labels, lk, cfg, *slots)
        neg_x.append(nx)
        neg_s.append(np.full(len(nx), sid))
        for i, x, ttd in pos:
            pos_x.append(x)
            pos_i.append(i)
            if ttd is not None:
                ttd_x.append(ttd)
                ttd_i.append(i)
    out = {
        "neg_x": np.concatenate(neg_x).astype(np.float32),
        "neg_sid": np.concatenate(neg_s).astype(np.int64),
        "pos_x": np.array(pos_x, dtype=np.float32),
        "pos_leak": np.array(pos_i, dtype=np.int64),  # row of ``leaks``
        "ttd_x": np.array(ttd_x, dtype=np.float32),
        "ttd_leak": np.array(ttd_i, dtype=np.int64),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return leaks, out


def detection_eval(leaks, s, folds, masks, cfg, edges, n_boot, seed) -> dict:
    from leaksense.stretch import DETECT_STATS, auc, detect_scores

    neg_f = np.array([folds[int(x)] for x in s["neg_sid"]])
    pos_rows = leaks.loc[s["pos_leak"]].reset_index(drop=True)
    pos_sid = pos_rows["scenario_id"].to_numpy()
    pos_f = np.array([folds[int(x)] for x in pos_sid])
    ttd_rows = leaks.loc[s["ttd_leak"]].reset_index(drop=True)
    ttd_f = np.array([folds[int(x)] for x in ttd_rows["scenario_id"]])
    hours = np.arange(1, s["ttd_x"].shape[1] + 1) * cfg.ttd_step / 2 if len(s["ttd_x"]) else []

    fl = pos_rows["flow_median_lps"].to_numpy()
    e = edges.get("leak_flow_lps") or [np.inf, np.inf]
    pos_groups = {
        "overall": np.ones(len(pos_rows), bool),
        "leak_type=small": (pos_rows["leak_type"] == "small").to_numpy(),
        "leak_type=big": (pos_rows["leak_type"] == "big").to_numpy(),
        "leak_flow=T1": fl <= e[0],
        "leak_flow=T2": (fl > e[0]) & (fl <= e[1]),
        "leak_flow=T3": fl > e[1],
    }
    sigma = {}
    for f in sorted(set(folds.values())):
        tr = s["neg_x"][neg_f != f]
        sigma[f] = np.sqrt(np.mean(np.square(tr.astype(np.float64)), axis=0))

    # per (method, k): per-sample detection averaged over draws, per stat and alpha
    acc: dict = {}
    for method, k, _d, m in masks:
        a = acc.setdefault((method, k), {"n": 0})
        a["n"] += 1
        for st in DETECT_STATS:
            neg_sc, pos_sc = np.zeros(len(neg_f)), np.zeros(len(pos_f))
            ttd_sc = np.zeros(s["ttd_x"].shape[:2]) if len(s["ttd_x"]) else None
            thr = {al: np.zeros(len(neg_f)) for al in ALPHAS}
            thr_pos = {al: np.zeros(len(pos_f)) for al in ALPHAS}
            thr_ttd = np.zeros(len(ttd_f))
            for f, sg in sigma.items():
                tr_sc = detect_scores(s["neg_x"][neg_f != f], m, sg)[st]
                te = neg_f == f
                neg_sc[te] = detect_scores(s["neg_x"][te], m, sg)[st]
                pte = pos_f == f
                pos_sc[pte] = detect_scores(s["pos_x"][pte], m, sg)[st]
                for al in ALPHAS:
                    q = float(np.quantile(tr_sc, 1 - al))
                    thr[al][te] = q
                    thr_pos[al][pte] = q
                    if al == ALPHAS[0] and ttd_sc is not None:
                        tte = ttd_f == f
                        ttd_sc[tte] = detect_scores(s["ttd_x"][tte], m, sg)[st]
                        thr_ttd[tte] = q
            for al in ALPHAS:
                kk = (st, al)
                a.setdefault(("fp", kk), np.zeros(len(neg_f)))
                a.setdefault(("tp", kk), np.zeros(len(pos_f)))
                a[("fp", kk)] += neg_sc > thr[al]
                a[("tp", kk)] += pos_sc > thr_pos[al]
            a.setdefault(("auc", st), []).append(auc(pos_sc, neg_sc))
            if ttd_sc is not None:
                over = ttd_sc > thr_ttd[:, None]
                first = np.where(over.any(axis=1), over.argmax(axis=1), -1)
                hrs = np.where(first >= 0, np.asarray(hours)[np.maximum(first, 0)], np.inf)
                a.setdefault(("ttd", st), []).append(hrs)

    res = {}
    for (method, k), a in sorted(acc.items(), key=lambda x: (x[0][0], x[0][1])):
        nd = a["n"]
        r = {"n_sets": nd}
        for st in DETECT_STATS:
            r[st] = {"auc": float(np.nanmean(a[("auc", st)]))}
            for al in ALPHAS:
                fp = a[("fp", (st, al))] / nd
                tp = a[("tp", (st, al))] / nd
                r[st][f"alpha={al}"] = {
                    "fpr": _stat(fp, s["neg_sid"], n_boot, seed),
                    "detect": {
                        g: _stat(tp[sel], pos_sid[sel], n_boot, seed)
                        for g, sel in pos_groups.items()
                    },
                }
            if ("ttd", st) in a:
                hrs = np.stack(a[("ttd", st)])  # (draws, n_ttd)
                det = np.isfinite(hrs)
                r[st]["time_to_detect"] = {
                    "alpha": ALPHAS[0],
                    "n_leaks": int(hrs.shape[1]),
                    "within_6h": float((hrs <= 6).mean()),
                    "within_12h": float((hrs <= 12).mean()),
                    "within_24h": float(det.mean()),
                    "median_hours_if_detected": float(np.median(hrs[det])) if det.any() else None,
                    "by_type_within_24h": {
                        t: float(det[:, (ttd_rows["leak_type"] == t).to_numpy()].mean())
                        for t in ("small", "big")
                        if (ttd_rows["leak_type"] == t).any()
                    },
                }
        res.setdefault(method, {})[str(k)] = r
    return res


def render_detection_md(rep: dict) -> str:
    p = rep["provenance"]
    L = [
        "# M15: leak detection (stretch goal)",
        "",
        f"- **Generated by:** `leaksense eval --task detect`, config `{p['config'].get('name')}`, "
        f"commit `{p['git']['commit'][:10]}`{' (dirty)' if p['git'].get('dirty') else ''}, "
        f"data manifest `{(p.get('data_manifest_sha256') or '')[:12]}`, seed {p['seed']}.",
        f"- **Scenarios:** {rep['n_scenarios']} CV scenarios (5 folds, each scored out of fold); "
        "the locked test is not used.",
        f"- **Samples:** {rep['n_negatives']} leak-free daily windows (negatives; up to "
        f"{rep['config']['neg_per_scenario']} per scenario, ending at least "
        f"{rep['config']['buffer_steps'] // 48} day before any leak) and {rep['n_positives']} "
        f"leaks whose first 24 h is the positive window ({rep['n_leaks']} leaks in these "
        f"scenarios; {rep['n_leaks'] - rep['n_positives']} lack 14 leak-free days before them "
        "or last under 24 h).",
        "- **Statistic:** daily window-mean ΔH against the 14 days right before it (rolling "
        "`tod_weekday` profile), scaled per junction by the training folds' leak-free spread; "
        "`max` = max over sensed junctions of −ΔH/σ, `mean` = their mean. Threshold = the "
        "(1 − α) quantile of the training folds' negative scores, per sensor set.",
        "- **Sensor sets:** random = the harness's 20 seeded sets per k (averaged); greedy = the "
        "M11 final order. 95% CIs: 1000 bootstrap resamples over scenarios.",
        "",
        "## Leak flow ramp check",
        "",
    ]
    r = rep["ramp"]
    z = r["steps_to_95pct"]["n_zero"]
    share = z / r["n_leaks"] if r["n_leaks"] else float("nan")
    L.append(
        f"Over {r['n_leaks']} leaks: flow at the first step / median flow has median "
        f"{r['first_step_ratio']['median']:.3f} (min {r['first_step_ratio']['min']:.3f}); "
        f"steps until 95% of the median: median {r['steps_to_95pct']['median']:.0f}, "
        f"p95 {r['steps_to_95pct']['p95']:.0f}, max {r['steps_to_95pct']['max']}. "
        f"{z} leaks ({100 * share:.1f}%) are at 95% of their median flow from the first step; "
        f"the median coefficient of variation of a leak's flow is {r['flow_cv_median']:.3f}."
    )
    L.append("")
    if r["starts_at_full_size"]:
        L.append(
            "**Leaks start at full size** (the fixed rule: at least 95% of leaks at 95% from "
            "the first step). So the time to detect below mostly measures how fast the window "
            "fills with leak steps, not a growing leak; it is a secondary number."
        )
    else:
        L.append(
            f"**Not every leak starts at full size** by the fixed rule (at least 95% of leaks at "
            f"95% from the first step; here {100 * share:.1f}%), so time to detect is reported. "
            "For the leaks that do start at full size it mostly measures how fast the window "
            "fills with leak steps."
        )
    for st in rep["stats"]:
        L += ["", f"## `{st}` statistic", ""]
        L.append(
            "| sets | k | AUC | FPR (α = 1%) | detected (α = 1%) | small | big | "
            "FPR (α = 5%) | detected (α = 5%) |"
        )
        L.append("|---|---|---|---|---|---|---|---|---|")
        for method in ("greedy", "random"):
            for k, r in rep["results"].get(method, {}).items():
                a1, a5 = r[st]["alpha=0.01"], r[st]["alpha=0.05"]
                L.append(
                    f"| {method} | {k} | {r[st]['auc']:.3f} | {_pct(a1['fpr'])} | "
                    f"{_pct(a1['detect']['overall'])} | {_pct(a1['detect']['leak_type=small'])} | "
                    f"{_pct(a1['detect']['leak_type=big'])} | {_pct(a5['fpr'])} | "
                    f"{_pct(a5['detect']['overall'])} |"
                )
        L += ["", f"Detection at α = 1% by leak-flow tercile (CV edges), `{st}`:", ""]
        L.append("| sets | k | T1 | T2 | T3 |")
        L.append("|---|---|---|---|---|")
        for method in ("greedy", "random"):
            for k, r in rep["results"].get(method, {}).items():
                d = r[st]["alpha=0.01"]["detect"]
                L.append(
                    f"| {method} | {k} | {_pct(d['leak_flow=T1'])} | {_pct(d['leak_flow=T2'])} "
                    f"| {_pct(d['leak_flow=T3'])} |"
                )
        L += ["", f"Time to detect (α = 1%, a window ending every hour after the start), `{st}`:"]
        L += ["", "| sets | k | leaks | ≤ 6 h | ≤ 12 h | ≤ 24 h | median h (detected) |"]
        L.append("|---|---|---|---|---|---|---|")
        for method in ("greedy", "random"):
            for k, r in rep["results"].get(method, {}).items():
                t = r[st].get("time_to_detect")
                if not t:
                    continue
                med = t["median_hours_if_detected"]
                L.append(
                    f"| {method} | {k} | {t['n_leaks']} | {100 * t['within_6h']:.1f} | "
                    f"{100 * t['within_12h']:.1f} | {100 * t['within_24h']:.1f} | "
                    f"{'-' if med is None else f'{med:.0f}'} |"
                )
    L += [
        "",
        "## Notes",
        "",
        "- FPR is per daily window: α = 1% is about 3.7 false alarms per network-year, if days "
        "were independent.",
        "- Leak flow (from `Leaks/`) is used only to report by tercile; it is never an input.",
        "- One non-learned statistic only; a learned detector was out of scope for this "
        "stretch milestone.",
        "",
    ]
    return "\n".join(L)


def cmd_detect(args: argparse.Namespace) -> int:
    from leaksense.stretch import DETECT_STATS, DetectConfig, ramp_summary

    t0 = time.perf_counter()
    inp, data_dir, contract, cache, sha, folds, slots = _setup(args)
    cfg = DetectConfig(seed=int(args.seed))
    leaks, s = build_detection_samples(inp, data_dir, contract, cache, sha, folds, slots, cfg)
    n = len(inp.net.sensor_ids)
    ks = [k for k in DETECT_KS if k <= n] or [n]
    masks = _mask_sets(n, ks, args.draws, args.seed, _greedy_order(args.placement_runs))
    res = detection_eval(leaks, s, folds, masks, cfg, inp.es.tercile_edges, args.bootstrap, 0)
    rep = {
        "name": "detection",
        "command": "leaksense eval --task detect",
        "provenance": _provenance(inp, args, {"placement_runs": args.placement_runs}),
        "config": asdict(cfg),
        "stats": list(DETECT_STATS),
        "alphas": list(ALPHAS),
        "ks": ks,
        "draws": args.draws,
        "bootstrap": args.bootstrap,
        "n_scenarios": len(folds),
        "n_leaks": len(leaks),
        "n_negatives": int(len(s["neg_x"])),
        "n_positives": int(len(s["pos_x"])),
        "n_ttd": int(len(s["ttd_x"])),
        "tercile_edges": inp.es.tercile_edges,
        "ramp": ramp_summary(leaks),
        "results": res,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    out = _write(rep, render_detection_md(rep), args.reports_dir, "detection")
    print(f"detection: {rep['n_negatives']} negatives, {rep['n_positives']} positives")
    for method in ("greedy", "random"):
        for k, r in res.get(method, {}).items():
            a = r["max"]["alpha=0.01"]
            fpr, det = a["fpr"]["value"], a["detect"]["overall"]["value"]
            print(
                f"  {method:6} k={k:>2}  max: AUC {r['max']['auc']:.3f}  FPR "
                f"{100 * fpr:.1f}%  detected {100 * det:.1f}%"
            )
    print(f"done in {rep['elapsed_seconds']:.0f}s -> {out}, {out.with_suffix('.md')}")
    return EXIT_OK


# ----------------------------------------------------------------------------- overlap


def overlap_cases(inp, data_dir, contract, cache, sha, folds, slots, case_index: pd.DataFrame):
    """(periods table for CV two-leak scenarios, scorable overlap cases, their windows)."""
    from leaksense.stretch import overlap_periods, overlap_windows

    leaks = _leaks(data_dir, contract, sha, sorted(folds))
    per = overlap_periods(leaks)
    per["fold"] = per["scenario_id"].map(folds).astype(int)
    solo = case_index[case_index["kind"] == "solo"].set_index("scenario_id")
    cls = {p: i for i, p in enumerate(inp.net.leak_pipe_ids)}
    W = inp.cfg.features.window
    rows, wins = [], []
    for ov in per.itertuples(index=False):
        if ov.ov_steps < W or ov.scenario_id not in solo.index:
            continue
        sid = int(ov.scenario_id)
        p, labels = cache.get(sid), cache.labels(sid)
        x, rows_b = overlap_windows(
            p, labels, solo.loc[sid].to_dict() | {"scenario_id": sid}, ov._asdict(),
            inp.cfg.features, *slots,
        )  # fmt: skip
        if x is None:
            continue
        bmax = int(rows_b.max())
        gap = int(ov.ov_start) - bmax if bmax < ov.ov_start else int(rows_b.min()) - int(ov.ov_end)
        rows.append(
            {
                **ov._asdict(),
                "case_id": f"S{sid:03d}-solo",
                "y1": cls[ov.first_pipe],
                "y2": cls[ov.second_pipe],
                "n_windows": len(x),
                "baseline_kind": solo.loc[sid, "baseline_kind"],
                "baseline_gap_days": gap / 48,
            }
        )
        wins.append(x)
    return per, pd.DataFrame(rows), wins


def overlap_unit(model, name, fold, cases, wins, es, masks) -> pd.DataFrame:
    """Hits for the overlap windows and the same scenarios' solo cases, one fold's cases."""
    from leaksense.evaluate import _score
    from leaksense.placement import topk_hits

    sel = np.flatnonzero(cases["fold"].to_numpy() == fold)
    if len(sel) == 0:
        return pd.DataFrame()
    xs = [wins[i] for i in sel]
    y1 = cases["y1"].to_numpy()[sel]
    y2 = cases["y2"].to_numpy()[sel]
    pos = {c: i for i, c in enumerate(es.cases["case_id"])}
    solo_i = [pos[c] for c in cases["case_id"].to_numpy()[sel]]
    solo_x = [es.windows[i] for i in solo_i]
    ys = es.cases["y"].to_numpy()[solo_i]
    frames = []
    for method, k, d, m in masks:
        sc = _score(model, xs, m)
        ss = _score(model, solo_x, m)
        frames.append(
            pd.DataFrame(
                {
                    "model": name,
                    "method": method,
                    "k": k,
                    "draw": d,
                    "scenario_id": cases["scenario_id"].to_numpy()[sel],
                    "hit_first": topk_hits(sc, y1, 3),
                    "hit_second": topk_hits(sc, y2, 3),
                    "solo_top3": topk_hits(ss, ys, 3),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def overlap_aggregate(df: pd.DataFrame, cases: pd.DataFrame, n_boot: int) -> dict:
    df = df.assign(
        hit_either=np.maximum(df["hit_first"], df["hit_second"]),
        hit_both=np.minimum(df["hit_first"], df["hit_second"]),
    )
    same = cases.set_index("scenario_id")["same_pipe"]
    out = {}
    cols = ["hit_first", "hit_second", "hit_either", "hit_both", "solo_top3"]
    for (model, method, k), g in df.groupby(["model", "method", "k"]):
        per = g.groupby("scenario_id")[cols].mean()
        per["same_pipe"] = same.loc[per.index].to_numpy()
        r = {}
        for grp, sel in (
            ("different_pipes", ~per["same_pipe"]),
            ("same_pipe", per["same_pipe"]),
            ("all", np.ones(len(per), bool)),
        ):
            sub = per[np.asarray(sel)]
            sids = sub.index.to_numpy()
            r[grp] = {c: _stat(sub[c].to_numpy(), sids, n_boot, 0) for c in cols}
            r[grp]["first_minus_solo"] = _stat(
                (sub["hit_first"] - sub["solo_top3"]).to_numpy(), sids, n_boot, 0
            )
        out.setdefault(model, {}).setdefault(method, {})[str(int(k))] = r
    return out


def render_overlap_md(rep: dict) -> str:
    p, c = rep["provenance"], rep["counts"]
    L = [
        "# M15: overlapping-leak stress test (stretch goal)",
        "",
        f"- **Generated by:** `leaksense eval --stress overlap`, commit "
        f"`{p['git']['commit'][:10]}`{' (dirty)' if p['git'].get('dirty') else ''}, data manifest "
        f"`{(p.get('data_manifest_sha256') or '')[:12]}`, seed {p['seed']}.",
        "- **Scope:** CV two-leak scenarios only; each is scored by a model that did not see its "
        "scenario (the headline model refit per fold, and `sigmatch_all`). Evaluation only.",
        "- **Windows:** ΔH windows from the first 7 days where both leaks are active, against the "
        "solo case's M3/M4 baseline (same baseline rows and profile).",
        "- **Hits:** tie-aware top-3 for the first and the second leak; either = the larger, "
        "both = the smaller of the two per sensor set. Solo = the same scenario's solo case, "
        "same model and sets. Random sets: the harness's 20 per k, averaged per case. 95% CIs: "
        "1000 bootstrap resamples over scenarios.",
        "",
        "## Overlap periods (CV two-leak scenarios)",
        "",
        f"- {c['n_two_leak']} two-leak scenarios; {c['n_zero']} have no step with both leaks "
        f"active; {c['n_under_window']} overlap for 1-47 steps (under one window); "
        f"{c['n_scorable']} overlap for at least one day and are scored "
        f"({c['n_same_pipe_scored']} of them have both leaks on the same pipe; "
        f"{c['n_same_pipe']} same-pipe scenarios in all).",
        f"- Overlap length (days), scored scenarios: median {c['ov_days']['median']:.1f}, "
        f"p10 {c['ov_days']['p10']:.1f}, p90 {c['ov_days']['p90']:.1f}, max "
        f"{c['ov_days']['max']:.1f}. Gap from the baseline to the overlap (days): median "
        f"{c['gap_days']['median']:.1f}, p90 {c['gap_days']['p90']:.1f}.",
        "",
    ]
    if rep.get("check"):
        L.append(
            f"- **Fold models:** the refit headline model's solo-case top-3 matches the M10 "
            f"per-fold runs to within {rep['check']['max_diff']:.4f} (k = "
            f"{rep['check']['k']}, {rep['check']['n_cases']} cases)."
        )
        L.append("")
    for model, by in rep["results"].items():
        for method in ("random", "greedy"):
            if method not in by:
                continue
            L += [f"## `{model}`, {method} sensor sets, top-3 (%)", ""]
            L.append(
                "| k | group | n | first leak | second leak | either | both | solo case | "
                "first − solo (points) |"
            )
            L.append("|---|---|---|---|---|---|---|---|---|")
            for k, r in by[method].items():
                for grp in ("different_pipes", "same_pipe"):
                    x = r[grp]
                    n = x["hit_first"]["n"]
                    if n == 0:
                        continue
                    d = x["first_minus_solo"]
                    L.append(
                        f"| {k} | {grp.replace('_', ' ')} | {n} | {_pct(x['hit_first'])} | "
                        f"{_pct(x['hit_second'])} | {_pct(x['hit_either'])} | "
                        f"{_pct(x['hit_both'])} | {_pct(x['solo_top3'])} | "
                        f"{100 * d['value']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}] |"
                    )
            L.append("")
    L += [
        "## Notes",
        "",
        '- Same-pipe scenarios are listed apart: there "either" and "both" equal the '
        "first-leak hit by construction.",
        "- The baseline lies further from the overlap than from the solo case, so baseline drift "
        "(M4) adds error that the solo cases do not have.",
        "",
    ]
    return "\n".join(L)


def cmd_overlap(args: argparse.Namespace) -> int:
    from leaksense.final_cmds import learned_cfg
    from leaksense.models.baselines import MODELS
    from leaksense.place_cmds import _done, _key, _save
    from leaksense.runconfig import load_run_config
    from leaksense.train_cmds import make_model

    t0 = time.perf_counter()
    inp, data_dir, contract, cache, sha, folds, slots = _setup(args)
    cases_path = Path(args.cases) if args.cases else get_artifacts_dir() / "cases.parquet"
    per, cases, wins = overlap_cases(
        inp, data_dir, contract, cache, sha, folds, slots, pd.read_parquet(cases_path)
    )
    mcfg = load_run_config(args.model_config)
    fc = dict(mcfg.final)
    if not fc.get("headline"):
        raise ConfigError(f"{args.model_config} has no final.headline")
    head = fc["headline"]
    head_cfg = learned_cfg(mcfg, fc["learned"][head], 0)
    n = len(inp.net.sensor_ids)
    ks = [k for k in OVERLAP_KS if k <= n] or [n]
    masks = _mask_sets(n, ks, args.draws, args.seed, _greedy_order(args.placement_runs))
    runs = Path(args.runs_dir or "reports/raw/overlap")
    runs.mkdir(parents=True, exist_ok=True)
    es = inp.es
    fold_ids = sorted(set(folds.values()))
    todo = fold_ids if args.fold is None else [int(args.fold)]
    fp = {"cases": cases["scenario_id"].tolist(), "ks": ks, "draws": args.draws,
          "seed": args.seed, "features": inp.fdir.name}  # fmt: skip
    sm = MODELS["sigmatch_all"](inp.net, sensitivity=load_sensitivity(inp))
    for name, cfg in ((head, head_cfg), ("sigmatch_all", None)):
        for f in todo:
            unit = f"overlap__{name}_f{f}"
            key = _key({**fp, "model": cfg or name, "fold": f})
            if _done(runs, unit, key):
                continue
            t1 = time.perf_counter()
            in_fold = len(cases) > 0 and bool((cases["fold"] == f).any())
            if not in_fold:
                model = None  # no overlap case in this fold: nothing to fit or score
            elif cfg is None:
                model = sm
            else:
                print(f"{unit}: fitting on {(es.cases['fold'] != f).sum()} cases", flush=True)
                tr = np.flatnonzero(es.cases["fold"].to_numpy() != f)
                model = make_model(inp, cfg)
                model.fit(es.cases.iloc[tr].reset_index(drop=True), [es.windows[i] for i in tr])
            df = (
                pd.DataFrame()
                if model is None
                else overlap_unit(model, name, f, cases, wins, es, masks)
            )
            secs = round(time.perf_counter() - t1, 1)
            _save(runs, unit, {"key": key, "fold": f, "model": name, "seconds": secs}, df)
            print(f"{unit}: {len(df)} rows ({secs:.0f}s)", flush=True)
    if args.fold is not None:
        print("--fold: no report written")
        return EXIT_OK

    frames = []
    for name, cfg in ((head, head_cfg), ("sigmatch_all", None)):
        for f in fold_ids:
            unit = f"overlap__{name}_f{f}"
            if not _done(runs, unit, _key({**fp, "model": cfg or name, "fold": f})):
                raise ConfigError(f"{unit} not done; run without --fold")
            frames.append(pd.read_parquet(runs / f"{unit}.parquet"))
    df = pd.concat(frames, ignore_index=True)
    check = _fold_check(df, head, cases, es) if len(df) else None
    scored = cases
    c = {
        "n_two_leak": len(per),
        "n_zero": int((per["ov_steps"] == 0).sum()),
        "n_under_window": int(((per["ov_steps"] > 0) & (per["ov_steps"] < 48)).sum()),
        "n_scorable": len(scored),
        "n_same_pipe": int(per["same_pipe"].sum()),
        "n_same_pipe_scored": int(scored["same_pipe"].sum()) if len(scored) else 0,
        "ov_days": _q(scored["ov_steps"] / 48),
        "gap_days": _q(scored["baseline_gap_days"]),
    }
    rep = {
        "name": "overlap",
        "command": "leaksense eval --stress overlap",
        "provenance": _provenance(inp, args, {"model_config": args.model_config}),
        "headline": head,
        "ks": ks,
        "draws": args.draws,
        "bootstrap": args.bootstrap,
        "counts": c,
        "check": check,
        "periods": per.to_dict("records"),
        "results": overlap_aggregate(df, scored, args.bootstrap) if len(df) else {},
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    out = _write(rep, render_overlap_md(rep), args.reports_dir, "overlap")
    r8 = rep["results"].get(head, {}).get("random", {}).get("8", {}).get("different_pipes")
    if r8:
        print(
            f"overlap: {c['n_scorable']} scored; {head} k=8 first {_pct(r8['hit_first'])}, "
            f"either {_pct(r8['hit_either'])}, solo {_pct(r8['solo_top3'])}"
        )
    print(f"done in {rep['elapsed_seconds']:.0f}s -> {out}, {out.with_suffix('.md')}")
    return EXIT_OK


def _q(v) -> dict:
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return {
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
        }
    return {
        "median": float(np.median(v)),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
        "max": float(v.max()),
    }


def _fold_check(df: pd.DataFrame, head: str, cases: pd.DataFrame, es) -> dict | None:
    """Solo-case top-3 of the refit fold models vs the M10 per-fold runs (same sets)."""
    k = 8 if 8 in set(df["k"]) else int(df["k"].max())
    g = df[(df["model"] == head) & (df["method"] == "random") & (df["k"] == k)]
    if g.empty:
        return None
    mine = g.groupby("scenario_id")["solo_top3"].mean()
    diffs = []
    raw = Path("reports/raw/ablation")
    folds = cases.set_index("scenario_id")["fold"]
    for f in sorted(set(folds)):
        pq = raw / f"ablation__{head}_s0_f{f}.parquet"
        if not pq.is_file():
            return None
        a = pd.read_parquet(pq)
        a = a[a["k"] == k].set_index("case_id")["top3"]
        for sid in folds[folds == f].index:
            cid = f"S{int(sid):03d}-solo"
            if cid in a.index and sid in mine.index:
                diffs.append(abs(float(a[cid]) - float(mine[sid])))
    if not diffs:
        return None
    return {"k": k, "n_cases": len(diffs), "max_diff": float(max(diffs))}
