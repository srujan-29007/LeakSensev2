"""Evaluation harness shared by every rung (PLAN.md M6, SPEC.md section 6).

- Cases: the 5 CV folds pooled. Locked-test scenarios are dropped and asserted absent.
- Each case is scored by a model fitted without its own fold (``Model.fit`` sees the other
  CV folds only). The model is fitted once per fold and then scored at every k and mask draw.
- Models return one score per leak-eligible pipe (``Network.leak_pipe_ids``); higher is better.
  The harness zeroes unsensed junctions before calling the model, so masked ΔH cannot leak in.
- Ties are scored exactly: the expected value under random tie-breaking.
- k < 31: no placement exists yet, so each k averages over seeded random sensor sets.
- CIs: percentile bootstrap over scenario ids (cases of a scenario move together).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from leaksense.network import Network

DEFAULT_KS = (2, 3, 4, 6, 8, 12, 16, 31)
TOPK = (1, 3, 5)
METRICS = ("top1", "top3", "top5", "hop_mean", "hop_median", "dist_m_mean", "dist_m_median")


class Model(Protocol):
    """``fit`` gets the training cases and, aligned with them, each case's (n_windows x 31) ΔH.

    A model may also define ``batch_scores(xs, mask) -> (n_cases, 33)`` to score many cases
    under one mask in a single call; the harness uses it when present.
    """

    name: str

    def fit(self, train: pd.DataFrame, windows: list[np.ndarray]) -> None: ...

    def case_scores(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray: ...


class LockedFoldError(AssertionError):
    pass


# ----------------------------------------------------------------------------- distances


def _pipe_ends(net: Network) -> dict[str, tuple[int, int, float]]:
    """pipe id -> (u, v, nominal length m), cached on the network object."""
    cached = getattr(net, "_pipe_ends_cache", None)
    if cached is None:
        t = net.pipes
        cached = {
            str(i): (int(u), int(v), float(ln))
            for i, u, v, ln in zip(t["id"], t["u"], t["v"], t["length_m"], strict=True)
        }
        object.__setattr__(net, "_pipe_ends_cache", cached)  # Network is a frozen dataclass
    return cached


def point_distance(net: Network, pipe_a: str, frac_a: float, pipe_b: str, frac_b: float) -> float:
    """Shortest distance in metres along pipes between two points on pipes.

    A point is (pipe id, fraction from the pipe's start node). Nominal lengths only (D10).
    """
    p = _pipe_ends(net)
    D = net.node_dist_m
    ua, va, la = p[pipe_a]
    ub, vb, lb = p[pipe_b]
    ends_a = ((ua, frac_a * la), (va, (1 - frac_a) * la))
    ends_b = ((ub, frac_b * lb), (vb, (1 - frac_b) * lb))
    best = min(oa + D[ea, eb] + ob for ea, oa in ends_a for eb, ob in ends_b)
    if pipe_a == pipe_b:
        best = min(best, abs(frac_a - frac_b) * la)
    return float(best)


# ----------------------------------------------------------------------------- eval set


@dataclass
class EvalSet:
    cases: pd.DataFrame  # one row per CV case: case_id, scenario_id, fold, pipe, y, ...
    windows: list[np.ndarray]  # per case: (n_windows x 31) window-mean ΔH
    classes: list[str]  # leak-eligible pipe ids, class order
    dist_to_leak: np.ndarray  # (n_cases x 33): midpoint of class pipe -> true leak point (m)
    class_hops: np.ndarray  # (33 x 33) pipe line-graph hops between classes
    tercile_edges: dict = field(default_factory=dict)


def build_eval_set(case_stats: pd.DataFrame, feats: dict, splits: dict, net: Network) -> EvalSet:
    test = set(int(s) for s in splits["test"])
    fold_of = {int(s): k for k, ids in enumerate(splits["folds"]) for s in ids}
    classes = net.leak_pipe_ids
    cls_index = {p: i for i, p in enumerate(classes)}

    cs = case_stats.reset_index(drop=True)
    cs = cs.assign(_row=np.arange(len(cs)))
    keep = cs[cs["scenario_id"].isin(fold_of)].copy()
    if keep["scenario_id"].isin(test).any():
        raise LockedFoldError("a locked-test scenario reached the evaluation set")
    keep["fold"] = keep["scenario_id"].map(fold_of).astype(int)
    keep["y"] = keep["pipe"].map(cls_index)
    if keep["y"].isna().any():
        raise ValueError("case labels outside the leak-eligible pipes")
    keep["y"] = keep["y"].astype(int)

    cow = feats["case_of_window"]
    order = np.argsort(cow, kind="stable")
    bounds = np.searchsorted(cow[order], np.arange(len(cs) + 1))
    windows = [feats["x"][order[bounds[r] : bounds[r + 1]]] for r in keep["_row"]]
    if any(len(w) == 0 for w in windows):
        raise ValueError("a case has no feature windows")

    pi = net.pipe_index
    cls_pipe = np.array([pi[p] for p in classes])
    dist = np.array(
        [
            [point_distance(net, c, 0.5, row.pipe, float(row.split_fraction)) for c in classes]
            for row in keep.itertuples()
        ]
    )
    hops = net.pipe_hops[np.ix_(cls_pipe, cls_pipe)].astype(float)

    keep = keep.drop(columns="_row").reset_index(drop=True)
    edges = {}
    for col in ("leak_flow_lps", "snr"):
        v = keep[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        edges[col] = [float(x) for x in np.quantile(v, [1 / 3, 2 / 3])] if len(v) else []
    return EvalSet(keep, windows, classes, dist, hops, edges)


# ----------------------------------------------------------------------------- metrics


def case_metrics(scores: np.ndarray, y: int, dist_row: np.ndarray, hops: np.ndarray) -> dict:
    """Tie-aware metrics for one case's scores over the classes."""
    s = np.asarray(scores, dtype=float)
    sy = s[y]
    n_better = int((s > sy).sum())
    n_tied = int((s == sy).sum())
    out = {f"top{k}": float(np.clip((k - n_better) / n_tied, 0.0, 1.0)) for k in TOPK}
    top = np.flatnonzero(s == s.max())
    out["hop"] = float(hops[top, y].mean())
    out["dist_m"] = float(dist_row[top].mean())
    return out


def random_masks(n_sensors: int, k: int, draws: int, seed: int) -> np.ndarray:
    if k >= n_sensors:
        return np.ones((1, n_sensors), dtype=bool)
    rng = np.random.default_rng([seed, k])
    masks = np.zeros((draws, n_sensors), dtype=bool)
    for d in range(draws):
        masks[d, rng.choice(n_sensors, size=k, replace=False)] = True
    return masks


def _score(model: Model, xs: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
    """Scores (n_cases, 33) for cases ``xs`` under one mask; unsensed junctions are zeroed."""
    xs = [x * mask for x in xs]  # the harness zeroes unsensed junctions, so masked ΔH can't leak in
    batch = getattr(model, "batch_scores", None)
    if batch is not None:
        return np.asarray(batch(xs, mask), dtype=float)
    return np.stack([np.asarray(model.case_scores(x, mask), dtype=float) for x in xs])


def predict_all(model: Model, es: EvalSet, masks_by_k: dict, folds=None) -> dict:
    """Fit once per fold, then score every k's masks -> {k: (n_draws, n_cases, 5)}.

    The 5 columns are top1, top3, top5, hop, dist (per case and mask draw). ``folds`` limits
    which folds are scored (rows of other folds stay zero); training still uses all others.
    """
    n = len(es.cases)
    out = {k: np.zeros((len(m), n, 5)) for k, m in masks_by_k.items()}
    only = None if folds is None else {int(f) for f in folds}
    folds = es.cases["fold"].to_numpy()
    ys = es.cases["y"].to_numpy()
    for f in np.unique(folds) if only is None else sorted(only):
        tr = np.flatnonzero(folds != f)
        model.fit(es.cases.iloc[tr].reset_index(drop=True), [es.windows[i] for i in tr])
        idx = np.flatnonzero(folds == f)
        xs = [es.windows[i] for i in idx]
        for k, masks in masks_by_k.items():
            for d, m in enumerate(masks):
                sc = _score(model, xs, m)
                for row, i in enumerate(idx):
                    cm = case_metrics(sc[row], int(ys[i]), es.dist_to_leak[i], es.class_hops)
                    out[k][d, i] = [cm["top1"], cm["top3"], cm["top5"], cm["hop"], cm["dist_m"]]
    return out


def predict(model: Model, es: EvalSet, masks: np.ndarray) -> np.ndarray:
    """One mask set -> (n_draws, n_cases, 5); see ``predict_all``."""
    return predict_all(model, es, {0: masks})[0]


def summarise(values: np.ndarray) -> dict:
    """values (n, 5) -> the METRICS dict."""
    return {
        "top1": float(values[:, 0].mean()),
        "top3": float(values[:, 1].mean()),
        "top5": float(values[:, 2].mean()),
        "hop_mean": float(values[:, 3].mean()),
        "hop_median": float(np.median(values[:, 3])),
        "dist_m_mean": float(values[:, 4].mean()),
        "dist_m_median": float(np.median(values[:, 4])),
    }


def bootstrap_samples(
    values: np.ndarray, scenario: np.ndarray, n_boot: int, seed: int
) -> dict[str, np.ndarray]:
    """Bootstrap distribution of each metric, resampling scenario ids with replacement."""
    uniq, inv = np.unique(scenario, return_inverse=True)
    members = [np.flatnonzero(inv == j) for j in range(len(uniq))]
    rng = np.random.default_rng(seed)
    stats = {m: np.empty(n_boot) for m in METRICS}
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[j] for j in pick])
        for m, v in summarise(values[idx]).items():
            stats[m][b] = v
    return stats


def bootstrap(values: np.ndarray, scenario: np.ndarray, n_boot: int, seed: int) -> dict:
    """Percentile 95% CI per metric."""
    samples = bootstrap_samples(values, scenario, n_boot, seed)
    return {
        m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for m, v in samples.items()
    }


def groups(es: EvalSet) -> dict[str, np.ndarray]:
    c = es.cases
    out = {
        "overall": np.ones(len(c), bool),
        "pre_baseline_only": (c["baseline_kind"] == "pre").to_numpy(),
        "leak_type=small": (c["leak_type"] == "small").to_numpy(),
        "leak_type=big": (c["leak_type"] == "big").to_numpy(),
        "kind=pure": (c["kind"] == "pure").to_numpy(),
        "kind=solo": (c["kind"] == "solo").to_numpy(),
        "baseline=pre": (c["baseline_kind"] == "pre").to_numpy(),
        "baseline=post+combined": (c["baseline_kind"] != "pre").to_numpy(),
    }
    for col, name in (("leak_flow_lps", "leak_flow"), ("snr", "snr")):
        v = c[col].to_numpy(dtype=float)
        e = es.tercile_edges.get(col, [])
        if len(e) == 2:
            out[f"{name}=T1"] = np.isfinite(v) & (v <= e[0])
            out[f"{name}=T2"] = (v > e[0]) & (v <= e[1])
            out[f"{name}=T3"] = v > e[1]
    return out


def evaluate(
    model: Model,
    es: EvalSet,
    ks=DEFAULT_KS,
    draws: int = 20,
    n_boot: int = 1000,
    seed: int = 0,
    folds=None,
) -> tuple[dict, pd.DataFrame]:
    """Score ``model`` at each k. ``folds`` (e.g. ``[0]``) scores only those folds' cases."""
    n_sensors = es.windows[0].shape[1]
    keep = np.ones(len(es.cases), bool)
    if folds is not None:
        keep = es.cases["fold"].isin(list(folds)).to_numpy()
        if not keep.any():
            raise ValueError(f"no cases in folds {list(folds)}")
    scen = es.cases["scenario_id"].to_numpy()[keep]
    grp = {g: sel[keep] for g, sel in groups(es).items()}
    cases = es.cases[keep]
    results, pred_rows = {}, []
    masks_by_k = {k: random_masks(n_sensors, k, draws, seed) for k in ks}
    per_by_k = predict_all(model, es, masks_by_k, folds)
    for k in ks:
        masks, per = masks_by_k[k], per_by_k[k][:, keep]
        avg = per.mean(axis=0)
        res = {"n_draws": len(masks), "groups": {}}
        for g, sel in grp.items():
            if not sel.any():
                continue
            point = summarise(avg[sel])
            ci = bootstrap(avg[sel], scen[sel], n_boot, seed)
            res["groups"][g] = {
                "n": int(sel.sum()),
                **{m: {"value": point[m], "lo": ci[m][0], "hi": ci[m][1]} for m in METRICS},
            }
        if len(masks) > 1:
            draw_vals = {m: [summarise(per[d])[m] for d in range(len(masks))] for m in METRICS}
            small = grp["leak_type=small"]
            draw_vals["small_top3"] = [float(per[d][small, 1].mean()) for d in range(len(masks))]
            res["draw_spread"] = {
                m: {q: float(np.percentile(v, p)) for q, p in (("p5", 5), ("p50", 50), ("p95", 95))}
                for m, v in draw_vals.items()
            }
        results[str(k)] = res
        for i, row in enumerate(cases.itertuples()):
            pred_rows.append(
                {
                    "case_id": row.case_id,
                    "k": k,
                    "top1": avg[i, 0],
                    "top3": avg[i, 1],
                    "top5": avg[i, 2],
                    "hop": avg[i, 3],
                    "dist_m": avg[i, 4],
                }
            )
    return results, pd.DataFrame(pred_rows)


# ----------------------------------------------------------------------------- report


def render_markdown(report: dict) -> str:
    res, prov = report["results"], report["provenance"]
    g = prov["git"]
    lines = [
        f"# Evaluation: {report['name']} (model `{report['model']}`)",
        "",
        f"- **Generated by:** `{report.get('command', 'leaksense eval')}`, "
        f"config `{prov['config'].get('name')}`, commit "
        f"`{(g['commit'] or '?')[:10]}`{' (dirty)' if g['dirty'] else ''}, data manifest "
        f"`{(prov['data_manifest_sha256'] or '?')[:12]}`, features `{report['features_dir']}` "
        f"(`{report['features_sha256_12']}`), seed {prov['seed']}.",
        f"- **Cases:** {report['n_cases']} CV cases ({report.get('folds_note', '5 folds pooled')});"
        " the locked test fold is "
        "excluded. k < 31 averages over "
        f"{report['draws']} seeded random sensor sets per k (no placement yet).",
        "- **Metrics:** tie-aware top-k (expected value under random tie-breaking); hop and network"
        " distance (m) from the top-1 pipe's midpoint to the true leak point, nominal lengths. "
        f"95% CIs: {report['bootstrap']} bootstrap resamples over scenarios.",
        "- **Tercile edges:** leak flow "
        + ", ".join(f"{e:.1f}" for e in report["tercile_edges"]["leak_flow_lps"])
        + " L/s; SNR "
        + ", ".join(f"{e:.1f}" for e in report["tercile_edges"]["snr"])
        + ".",
        "",
        "## Accuracy vs sensor budget",
        "",
        "| k | top-1 | top-3 | top-5 | small top-3 | median dist (m) |",
        "|---|---|---|---|---|---|",
    ]

    def ci(d, pct=True):
        f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.0f}")
        return f"{f(d['value'])} [{f(d['lo'])}, {f(d['hi'])}]"

    for k, r in res.items():
        o, s = r["groups"]["overall"], r["groups"].get("leak_type=small")
        lines.append(
            f"| {k} | {ci(o['top1'])} | {ci(o['top3'])} | {ci(o['top5'])} | "
            f"{ci(s['top3']) if s else '-'} | {ci(o['dist_m_median'], False)} |"
        )
    kmax = list(res)[-1]
    lines += [
        "",
        "Accuracy in %, [95% CI].",
        "",
        f"## Breakdown at k = {kmax}",
        "",
        "| group | n | top-1 | top-3 | top-5 | hop mean | dist median (m) |",
        "|---|---|---|---|---|---|---|",
    ]
    for gname, d in res[kmax]["groups"].items():
        lines.append(
            f"| {gname} | {d['n']} | {ci(d['top1'])} | {ci(d['top3'])} | {ci(d['top5'])} "
            f"| {d['hop_mean']['value']:.2f} | {ci(d['dist_m_median'], False)} |"
        )
    spread = [(k, r["draw_spread"]) for k, r in res.items() if "draw_spread" in r]
    if spread:
        lines += [
            "",
            "## Spread across random sensor sets (overall top-3, small-leak top-3, %)",
            "",
            "| k | top-3 p5 / p50 / p95 | small top-3 p5 / p50 / p95 |",
            "|---|---|---|",
        ]
        for k, sp in spread:
            a, b = sp["top3"], sp["small_top3"]
            lines.append(
                f"| {k} | {100 * a['p5']:.1f} / {100 * a['p50']:.1f} / {100 * a['p95']:.1f} | "
                f"{100 * b['p5']:.1f} / {100 * b['p50']:.1f} / {100 * b['p95']:.1f} |"
            )
    tr = report.get("training")
    if tr:
        lines += ["", "## Training (per fold)", ""]
        if tr.get("grid"):
            lines += [
                "Chosen on an inner validation split of each fold's training scenarios: "
                + ", ".join(f"`{k}` from {v}" for k, v in tr["grid"].items())
                + ".",
                "",
            ]
        lines += [
            "| fold | train cases | inner-val cases | chosen | epochs / rounds | inner-val loss |",
            "|---|---|---|---|---|---|",
        ]
        for f, h in zip(tr["eval_folds"], tr["folds"], strict=True):
            ch = ", ".join(f"{k}={v}" for k, v in h["chosen"].items()) or "-"
            vl = "-" if h["val_loss"] is None else f"{h['val_loss']:.3f}"
            lines.append(
                f"| {f} | {h['n_train_cases']} | {h['n_inner_val_cases']} | {ch} | "
                f"{h['epochs']} | {vl} |"
            )
        lines += ["", f"Base settings: `{json.dumps(tr['base'])}`."]
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict, preds: pd.DataFrame, reports_dir: Path, artifacts_dir: Path) -> Path:
    name = report["name"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"{name}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    (reports_dir / f"{name}.md").write_text(render_markdown(report), encoding="utf-8")
    pdir = artifacts_dir / "eval" / name
    pdir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(pdir / "predictions.parquet", index=False)
    return out
