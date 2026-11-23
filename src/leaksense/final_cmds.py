"""CLI command for M13: ``leaksense final-report``.

Scores every rung once on the locked test fold:

- Each **unit** (a non-learned rung, a learned variant at one seed, or an H4 fixed-set model) is
  fit on all CV cases and scored on the test cases under the same seeded random sensor sets as
  the CV reports (``evaluate.random_masks``, 20 per k).
- The headline model (seed 0) and ``sigmatch_all`` are also scored on the M11 final placement:
  the learned greedy order, the coverage order and the same 100 random sets as M11.
- H4: the headline settings trained with ``fixed_sensors`` = the final greedy set at k, scored
  with that set, paired with the masked headline model on the same set.

Each unit is saved to ``<runs-dir>/<unit>.{json,parquet}`` as it finishes (resumable, and
splittable over processes with ``--only``). Scoring the test fold needs ``--confirm-locked-test``;
a saved test unit whose config has changed is never re-scored (the test fold is used once).
``--stand-in-fold F`` runs the same pipeline with CV fold F as a stand-in test set (training on
the other folds) to check the code and timing; it never reads the test fold.

Writes ``reports/final.json``, ``reports/FINAL.md`` and two plots.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from leaksense.config import ConfigError
from leaksense.eval_cmds import EXIT_OK, add_common, load_inputs, load_sensitivity
from leaksense.place_cmds import _done, _key, _save
from leaksense.placement import COLS, as_mask, greedy_coverage, per_case_metrics

H2_KMAX = 6  # H2: k <= 6
H3_P95_KMAX = 8  # H3: above the random p95 for k <= 8
SHOW_GROUPS = (
    "overall",
    "pre_baseline_only",
    "leak_type=small",
    "leak_type=big",
    "kind=pure",
    "kind=solo",
)


def add_parsers(sub) -> None:
    p = sub.add_parser(
        "final-report", help="M13: score every rung once on the locked test fold (FINAL.md)"
    )
    add_common(p, default_config="configs/final.yaml")
    p.set_defaults(k=None, name="final", func=cmd_final)
    p.add_argument(
        "--stand-in-fold",
        type=int,
        help="dry run: use this CV fold as the test set and train on the others",
    )
    p.add_argument("--only", help="comma-separated units to run (see --status); no report")
    p.add_argument("--runs-dir", help="per-unit results (default reports/raw/final[_standin_fF])")
    p.add_argument("--status", action="store_true", help="list the units and which are done")
    p.add_argument("--report-only", action="store_true", help="only build the report")
    p.add_argument(
        "--confirm-locked-test", action="store_true", help="required to score the locked test"
    )


# ----------------------------------------------------------------------------- setup


def subset(es, sel: np.ndarray, edges: dict | None = None):
    from leaksense.evaluate import EvalSet

    idx = np.flatnonzero(sel)
    return EvalSet(
        es.cases.iloc[idx].reset_index(drop=True),
        [es.windows[i] for i in idx],
        es.classes,
        es.dist_to_leak[idx],
        es.class_hops,
        dict(es.tercile_edges if edges is None else edges),
    )


def learned_cfg(cfg, spec: dict, seed: int) -> dict:
    """A learned unit's model block: a full ``model`` block, or ``overrides`` on the config's."""
    if spec.get("model"):
        return {**spec["model"], "seed": int(seed)}
    return {**cfg.model, **(spec.get("overrides") or {}), "seed": int(seed)}


def load_placement(fc: dict, inp, headline: dict) -> dict:
    from leaksense.evaluate import random_masks
    from leaksense.models.baselines import signatures

    pc = fc["placement"]
    path = Path(pc["runs"]) / f"{pc['name']}__final.json"
    if not path.is_file():
        raise ConfigError(f"no M11 final placement at {path}; run `leaksense place` first")
    fin = json.loads(path.read_text())
    fp = fin["fp"]
    if {**fp["model"], "seed": fp["model"].get("seed", 0)} != headline:
        raise ConfigError("the M11 final placement used a different model than the headline")
    if fp["features_dir"] != inp.fdir.name:
        raise ConfigError("the M11 final placement used different features")
    ks = [int(k) for k in fp["ks"]]
    n = len(inp.net.sensor_ids)
    cov = greedy_coverage(signatures(inp.net, load_sensitivity(inp), "all"), max(ks))
    return {
        "order": [int(j) for j in fin["greedy_order_idx"]],
        "order_ids": fin["greedy_order"],
        "coverage": cov,
        "random": {k: random_masks(n, k, int(fp["n_random"]), int(fp["seed"])) for k in ks},
        "ks": ks,
        "n_random": int(fp["n_random"]),
        "key": fin["key"],
        "n_greedy_cases": fin["n_greedy_cases"],
        "greedy_trace": fin["greedy_trace"],
    }


def plan_units(cfg, fc: dict, place: dict) -> list[dict]:
    from leaksense.fixed_cmds import fixed_model_cfg

    units = []
    for r in fc.get("nonlearned", []):
        units.append(
            {"name": f"nl__{r}", "kind": "nonlearned", "rung": r, "placement": r == "sigmatch_all"}
        )
    for v, spec in fc["learned"].items():
        for s in spec.get("seeds", [0]):
            units.append(
                {
                    "name": f"ml__{v}_s{s}",
                    "kind": "learned",
                    "rung": v,
                    "seed": int(s),
                    "model": learned_cfg(cfg, spec, s),
                    "placement": v == fc["headline"] and int(s) == 0,
                }
            )
    head = learned_cfg(cfg, fc["learned"][fc["headline"]], 0)
    for k in fc.get("fixed_ks", []):
        units.append(
            {
                "name": f"fx__k{k}",
                "kind": "fixed",
                "rung": f"fixed_k{k}",
                "k": int(k),
                "model": fixed_model_cfg(head, place["order"][: int(k)]),
                "placement": False,
            }
        )
    return units


def setup(args):
    from leaksense.evaluate import build_test_set
    from leaksense.provenance import file_sha256
    from leaksense.runconfig import load_run_config

    cfg = load_run_config(args.config)
    fc = dict(cfg.final)
    if not fc.get("learned") or fc.get("headline") not in fc["learned"]:
        raise ConfigError("the config needs final.learned with the final.headline variant")
    if "sigmatch_all" not in fc.get("nonlearned", []):
        raise ConfigError("final.nonlearned must include sigmatch_all (placement reference)")
    args.k = ",".join(str(k) for k in fc.get("ks", [2, 3, 4, 6, 8, 12, 16, 31]))
    args.draws = int(fc.get("draws", args.draws))
    inp = load_inputs(args)
    cv = inp.es
    if args.stand_in_fold is None:
        mode = "test"
        with np.load(inp.fdir / "features.npz") as z:
            feats = {k: z[k] for k in z.files}
        cs = pd.read_parquet(inp.fdir / "case_stats.parquet")
        splits = json.loads(Path(args.splits).read_text())
        test = build_test_set(cs, feats, splits, inp.net, cv.tercile_edges)
        train = cv
    else:
        f = int(args.stand_in_fold)
        folds = cv.cases["fold"].to_numpy()
        if f not in set(folds.tolist()):
            raise ConfigError(f"--stand-in-fold must be one of {sorted(set(folds.tolist()))}")
        mode = f"standin_f{f}"
        test, train = subset(cv, folds == f), subset(cv, folds != f)
    head = learned_cfg(cfg, fc["learned"][fc["headline"]], 0)
    place = load_placement(fc, inp, head)
    hk = int(fc.get("headline_k", 8))
    if hk not in inp.ks:
        raise ConfigError(f"final.headline_k {hk} must be one of final.ks")
    runs_dir = Path(
        args.runs_dir or f"reports/raw/{'final' if mode == 'test' else 'final_' + mode}"
    )
    base_fp = {
        "features_dir": inp.fdir.name,
        "features_sha256_12": inp.meta.get("features_sha256_12"),
        "splits_sha256": file_sha256(args.splits),
        "ks": inp.ks,
        "draws": args.draws,
        "seed": int(args.seed),
        "mode": mode,
    }
    return SimpleNamespace(
        cfg=cfg,
        fc=fc,
        inp=inp,
        train=train,
        test=test,
        mode=mode,
        hk=hk,
        place=place,
        units=plan_units(cfg, fc, place),
        runs_dir=runs_dir,
        base_fp=base_fp,
    )


def unit_key(u: dict, ctx) -> str:
    fp = {**ctx.base_fp, "unit": u}
    if u["placement"] or u["kind"] == "fixed":
        fp["placement_key"] = ctx.place["key"]
    return _key(fp)


# ----------------------------------------------------------------------------- units


def unit_masks(u: dict, ctx) -> list[tuple[str, int, int, np.ndarray]]:
    from leaksense.evaluate import random_masks

    n = len(ctx.inp.net.sensor_ids)
    p = ctx.place
    if u["kind"] == "fixed":
        return [("fixed", u["k"], 0, as_mask(p["order"][: u["k"]], n))]
    out = []
    for k in ctx.inp.ks:
        for d, m in enumerate(random_masks(n, k, ctx.base_fp["draws"], ctx.base_fp["seed"])):
            out.append(("ladder", k, d, m))
    if u["placement"]:
        for k in p["ks"]:
            out.append(("greedy", k, 0, as_mask(p["order"][:k], n)))
            out.append(("coverage", k, 0, as_mask(p["coverage"][:k], n)))
            out += [("random", k, d, m) for d, m in enumerate(p["random"][k])]
    return out


def make_unit_model(u: dict, inp):
    if u["kind"] == "nonlearned":
        from leaksense.models.baselines import MODELS, NEEDS_SENSITIVITY

        res = {"sensitivity": load_sensitivity(inp)} if u["rung"] in NEEDS_SENSITIVITY else {}
        return MODELS[u["rung"]](inp.net, **res)
    from leaksense.train_cmds import make_model

    return make_model(inp, u["model"])


def run_unit(u: dict, ctx) -> tuple[dict, pd.DataFrame]:
    from leaksense.evaluate import _score

    model = make_unit_model(u, ctx.inp)
    tr, te = ctx.train, ctx.test
    model.fit(tr.cases, tr.windows)
    ys = te.cases["y"].to_numpy()
    ids = te.cases["case_id"].to_numpy()
    frames = []
    for method, k, d, m in unit_masks(u, ctx):
        vals = per_case_metrics(_score(model, te.windows, m), ys, te.dist_to_leak, te.class_hops)
        df = pd.DataFrame(vals, columns=list(COLS))
        df.insert(0, "case_id", ids)
        df.insert(0, "draw", d)
        df.insert(0, "k", k)
        df.insert(0, "method", method)
        frames.append(df)
    info = {"n_train_cases": len(tr.cases), "n_test_cases": len(te.cases)}
    if hasattr(model, "describe"):
        info["training"] = model.describe()["folds"][-1]
    return info, pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------- aggregate


def _per_case(df: pd.DataFrame, method: str, k: int, pos: dict, n: int) -> np.ndarray:
    """(n, 5): per-case metrics averaged over the draws of ``method`` at ``k``."""
    part = df[(df["method"] == method) & (df["k"] == k)]
    nd = int(part["draw"].max()) + 1
    a = np.full((nd, n, len(COLS)), np.nan)
    a[part["draw"].to_numpy(), part["case_id"].map(pos).to_numpy()] = part[list(COLS)].to_numpy()
    if np.isnan(a).any():
        raise ValueError(f"missing cases for {method}/k={k}")
    return a.mean(axis=0)


def group_stats(es, values: np.ndarray, n_boot: int, seed: int) -> dict:
    from leaksense.evaluate import METRICS, bootstrap, groups, summarise

    scen = es.cases["scenario_id"].to_numpy()
    out = {}
    for g, sel in groups(es).items():
        if not sel.any():
            continue
        point, ci = summarise(values[sel]), bootstrap(values[sel], scen[sel], n_boot, seed)
        out[g] = {
            "n": int(sel.sum()),
            **{m: {"value": point[m], "lo": ci[m][0], "hi": ci[m][1]} for m in METRICS},
        }
    return out


def paired(es, a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict:
    from leaksense.ablate import _paired
    from leaksense.evaluate import groups

    scen = es.cases["scenario_id"].to_numpy()
    out = {}
    for g, sel in groups(es).items():
        if g in ("overall", "leak_type=small") and sel.any():
            out[g] = _paired(a[sel] - b[sel], scen[sel], n_boot, seed)
    return out


def ladder(ctx, frames: dict, n_boot: int, seed: int) -> tuple[dict, dict]:
    """Per rung and k: group stats of per-case metrics averaged over draws, then seeds."""
    te = ctx.test
    pos = {c: i for i, c in enumerate(te.cases["case_id"])}
    n = len(te.cases)
    results, per_case = {}, {}
    for rung in rung_order(ctx):
        units = [u for u in ctx.units if u["rung"] == rung and u["name"] in frames]
        if not units:
            continue
        results[rung], per_case[rung] = {"seeds": [u.get("seed") for u in units]}, {}
        for k in ctx.inp.ks:
            per_seed = np.stack([_per_case(frames[u["name"]], "ladder", k, pos, n) for u in units])
            avg = per_seed.mean(axis=0)
            per_case[rung][k] = avg
            r = group_stats(te, avg, n_boot, seed)
            t3 = per_seed[:, :, 1].mean(axis=1)
            r["overall"]["top3"] |= {"seed_min": float(t3.min()), "seed_max": float(t3.max())}
            results[rung][str(k)] = r
    return results, per_case


def rung_order(ctx) -> list[str]:
    fc = ctx.fc
    return list(fc.get("nonlearned", [])) + list(fc["learned"])


def _verdict(d: dict) -> str:
    return "better" if d["lo"] > 0 else "worse" if d["hi"] < 0 else "no clear difference"


def hypotheses(ctx, results: dict, per_case: dict, n_boot: int, seed: int) -> dict:
    te, fc = ctx.test, ctx.fc
    nonlearned = [r for r in fc.get("nonlearned", []) if r in results]
    learned = [r for r in fc["learned"] if r in results]
    plain = fc.get("plain", "gat")

    h1 = {}
    for v in learned:
        per_k = {}
        for k in [k for k in ctx.inp.ks if k >= ctx.hk]:  # H1: k >= 8
            if str(k) not in results[v] or not nonlearned:
                continue
            best = max(nonlearned, key=lambda r: results[r][str(k)]["overall"]["top3"]["value"])
            ours = results[v][str(k)]["overall"]["top3"]
            theirs = results[best][str(k)]["overall"]["top3"]
            per_k[str(k)] = {
                "vs": best,
                "holds": bool(ours["lo"] > theirs["hi"]),
                "worse": bool(ours["hi"] < theirs["lo"]),
                "paired": paired(te, per_case[v][k], per_case[best][k], n_boot, seed)["overall"],
            }
        h1[v] = per_k

    h2 = {}
    for v in learned:
        if plain not in per_case or v == plain:
            continue
        model = learned_cfg(ctx.cfg, fc["learned"][v], 0)
        if model.get("kind") != "gat" or model.get("physics", "none") == "none":
            continue
        per_k = {}
        for k in [k for k in ctx.inp.ks if k <= H2_KMAX]:
            if k in per_case[v]:
                d = paired(te, per_case[v][k], per_case[plain][k], n_boot, seed)["overall"]
                per_k[str(k)] = {"diff": d["top3"], "verdict": _verdict(d["top3"])}
        n_better = sum(x["verdict"] == "better" for x in per_k.values())
        h2[v] = {
            "per_k": per_k,
            "holds": "yes" if per_k and n_better == len(per_k) else "partly" if n_better else "no",
        }
    return {"H1": h1, "H2": h2}


def placement_results(ctx, frames: dict, n_boot: int, seed: int) -> dict | None:
    from leaksense.place_cmds import aggregate

    head = f"ml__{ctx.fc['headline']}_s0"
    if head not in frames or "nl__sigmatch_all" not in frames:
        return None
    parts = []
    for name, scorer in ((head, "model"), ("nl__sigmatch_all", "sigmatch_all")):
        df = frames[name]
        df = df[df["method"].isin(["greedy", "coverage", "random"])].copy()
        df.insert(1, "scorer", scorer)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    p = ctx.place
    res = aggregate(SimpleNamespace(es=ctx.test), df, p["ks"], p["n_random"], n_boot, seed)
    return res


def h3_verdict(model_res: dict, ks) -> dict:
    per_k = {}
    for k in ks:
        r = model_res[str(k)]["overall"]
        g = r["greedy"]["top3"]["value"]
        dr = r["random_top3_draws"]
        per_k[str(k)] = {
            "beats_median": bool(g > dr["p50"]),
            "above_p95": bool(g > dr["p95"]) if k <= H3_P95_KMAX else None,
        }
    ok = [v["beats_median"] for v in per_k.values()]
    ok += [v["above_p95"] for v in per_k.values() if v["above_p95"] is not None]
    return {
        "per_k": per_k,
        "holds": "yes" if all(ok) else "partly" if any(ok) else "no",
    }


def fixed_results(ctx, frames: dict, n_boot: int, seed: int) -> dict:
    from leaksense.fixed_cmds import compare, h4_verdict

    te = ctx.test
    pos = {c: i for i, c in enumerate(te.cases["case_id"])}
    n = len(te.cases)
    head = f"ml__{ctx.fc['headline']}_s0"
    out = {}
    for u in ctx.units:
        if u["kind"] != "fixed" or u["name"] not in frames or head not in frames:
            continue
        k = u["k"]
        fixed = _per_case(frames[u["name"]], "fixed", k, pos, n)
        masked = _per_case(frames[head], "greedy", k, pos, n)
        out[str(k)] = compare(te, fixed, masked, n_boot, seed)
    return {"per_k": out, "holds": h4_verdict(out) if out else None}


# ----------------------------------------------------------------------------- CV side


def cv_numbers(fc: dict, rungs: list[str], hk: int, full_k: int) -> dict:
    """CV top-3 per rung at the headline k (all, small) and full k, from the CV reports."""
    rep_dir = Path(fc.get("cv_reports", "reports"))
    abl_path = rep_dir / "ablation.json"
    abl = json.loads(abl_path.read_text()) if abl_path.is_file() else {"variants": {}}
    out = {}
    for r in rungs:
        if r in abl["variants"]:
            res = abl["variants"][r]["results"]
            src = "ablation.json"

            def get(k, g, res=res):
                return res[str(k)][g]["top3"] if str(k) in res and g in res[str(k)] else None

        elif (rep_dir / f"{r}.json").is_file():
            res = json.loads((rep_dir / f"{r}.json").read_text())["results"]
            src = f"{r}.json"

            def get(k, g, res=res):
                d = res.get(str(k), {}).get("groups", {})
                return d[g]["top3"] if g in d else None

        else:
            continue
        out[r] = {
            "source": src,
            "k_head": get(hk, "overall"),
            "k_head_small": get(hk, "leak_type=small"),
            "k_full": get(full_k, "overall"),
        }
    return out


def cv_hypotheses(fc: dict) -> dict:
    rep_dir = Path(fc.get("cv_reports", "reports"))
    out = {}
    p = rep_dir / "placement.json"
    if p.is_file():
        pl = json.loads(p.read_text())
        out["H3"] = h3_verdict(pl["results"]["model"], pl["ks"])
    p = rep_dir / "fixed_budget.json"
    if p.is_file():
        fb = json.loads(p.read_text())
        out["H4"] = {"holds": fb["h4"], "per_k": fb["results"]}
    p = rep_dir / "ablation.json"
    if p.is_file():
        v = json.loads(p.read_text())["verdicts"]
        out["H1"] = v.get("H1_vs_best_nonlearned_top3_k_ge_8", {})
        out["H2"] = v.get("H2_physics_vs_plain_top3_k_le_6", {})
    return out


def cv_placement(fc: dict) -> dict:
    """M11 facts quoted in FINAL.md: set stability, inner-case counts, final-greedy saturation."""
    p = Path(fc.get("cv_reports", "reports")) / "placement.json"
    if not p.is_file():
        return {}
    pl = json.loads(p.read_text())
    out = {
        "fold_greedy_cases": [pl["folds"][f]["n_greedy_cases"] for f in sorted(pl["folds"])],
        "stability": {
            k: {
                "jaccard": s["mean_pairwise_jaccard"],
                "random_jaccard": s["random_mean_pairwise_jaccard"],
            }
            for k, s in pl["stability"].items()
        },
    }
    fin = pl.get("final")
    if fin:
        t3 = [t["top3"] for t in fin["greedy_trace"]]
        out["final_greedy_cases"] = fin["n_greedy_cases"]
        out["final_saturates_at_k"] = int(np.argmax(np.isclose(t3, max(t3)))) + 1
        out["final_max_inner_top3"] = float(max(t3))
    return out


# ----------------------------------------------------------------------------- leak flow (H5)


def flow_curve(ctx, per_case: dict, path: Path | None) -> dict:
    """Top-3 at the headline k (random sets) vs leak flow: test cases and CV out-of-fold."""
    fc = ctx.fc
    head = fc["headline"]
    k = ctx.hk
    cv = ctx.inp.es
    cv_flow = cv.cases["leak_flow_lps"].to_numpy(dtype=float)
    edges = np.quantile(cv_flow[np.isfinite(cv_flow)], np.linspace(0, 1, 7))
    edges[0], edges[-1] = 0.0, np.inf

    def bins(flow, top3):
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            sel = (flow > lo) & (flow <= hi) if lo > 0 else (flow <= hi)
            rows.append(
                {
                    "lo": float(lo),
                    "hi": float(hi),
                    "n": int(sel.sum()),
                    "flow_median": float(np.median(flow[sel])) if sel.any() else None,
                    "top3": float(top3[sel].mean()) if sel.any() else None,
                }
            )
        return rows

    te_flow = ctx.test.cases["leak_flow_lps"].to_numpy(dtype=float)
    series = {}
    for rung in (head, "sigmatch_all"):
        if rung in per_case and k in per_case[rung]:
            series[f"test {rung}"] = bins(te_flow, per_case[rung][k][:, 1])
    raw = Path("reports/raw/ablation")
    seeds = fc["learned"][head].get("seeds", [0])
    cv_top3 = np.zeros(len(cv.cases))
    cnt = np.zeros(len(cv.cases))
    pos = {c: i for i, c in enumerate(cv.cases["case_id"])}
    for s in seeds:
        for f in sorted(cv.cases["fold"].unique()):
            pq = raw / f"ablation__{head}_s{s}_f{f}.parquet"
            if not pq.is_file():
                continue
            d = pd.read_parquet(pq)
            d = d[d["k"] == k]
            i = d["case_id"].map(pos)
            ok = i.notna().to_numpy()
            cv_top3[i[ok].astype(int).to_numpy()] += d["top3"].to_numpy()[ok]
            cnt[i[ok].astype(int).to_numpy()] += 1
    if (cnt > 0).all():
        series[f"CV {head}"] = bins(cv_flow, cv_top3 / cnt)
    out = {"k": k, "edges": [float(e) for e in edges], "series": series}
    if path is not None and series:
        plot_flow(out, path)
    return out


def plot_flow(curve: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    styles = {"CV": ("0.45", "-", "s"), "test": ("tab:blue", "-", "o")}
    for name, rows in curve["series"].items():
        rows = [r for r in rows if r["n"] > 0]
        color, ls, mk = styles[name.split()[0]]
        if "sigmatch_all" in name:
            color, ls = "tab:orange", "--"
        ax.plot(
            [r["flow_median"] for r in rows],
            [100 * r["top3"] for r in rows],
            color=color,
            ls=ls,
            marker=mk,
            label=f"{name} (n per bin: {', '.join(str(r['n']) for r in rows)})",
        )
    ax.set_xscale("log")
    ax.set_xlabel("leak flow (L/s, bin median; bins = CV sextiles)")
    ax.set_ylabel(f"top-3 accuracy (%), k = {curve['k']}, random sets")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Accuracy vs leak flow (SPEC H5; reporting only, never a model input)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- report


def test_summary(es) -> dict:
    c = es.cases
    return {
        "n_cases": len(c),
        "n_scenarios": int(c["scenario_id"].nunique()),
        "leak_type": c["leak_type"].value_counts().to_dict(),
        "kind": c["kind"].value_counts().to_dict(),
        "baseline_kind": c["baseline_kind"].value_counts().to_dict(),
        "leak_flow_lps_median": {
            t: float(v) for t, v in c.groupby("leak_type")["leak_flow_lps"].median().items()
        },
    }


def data_notes(rep_dir: Path) -> dict:
    out = {}
    p = rep_dir / "case_index.json"
    if p.is_file():
        ci = json.loads(p.read_text())
        two = ci.get("two_leak_without_solo")
        out["cases"] = {
            "config": ci.get("config"),
            "candidates": ci.get("candidates"),
            "cases": ci.get("cases"),
            "n_excluded": ci.get("n_excluded"),
            "n_two_leak_without_solo": len(two) if isinstance(two, list) else None,
        }
    p = rep_dir / "baseline_choice.json"
    if p.is_file():
        out["baseline"] = json.loads(p.read_text()).get("decision")
    return out


def build_report(args, ctx, frames: dict, t0: float) -> dict:
    from leaksense.provenance import provenance

    nb, sd = args.bootstrap, args.seed
    results, per_case = ladder(ctx, frames, nb, sd)
    head = ctx.fc["headline"]
    others = [r for r in results if r != head]
    vs_head = {
        r: {str(k): paired(ctx.test, per_case[head][k], per_case[r][k], nb, sd) for k in ctx.inp.ks}
        for r in others
        if head in per_case
    }
    hyp = hypotheses(ctx, results, per_case, nb, sd)
    place = placement_results(ctx, frames, nb, sd)
    if place is not None:
        hyp["H3"] = h3_verdict(place["model"], ctx.place["ks"])
    hyp["H4"] = fixed_results(ctx, frames, nb, sd)
    if head in results:
        hyp["H5"] = {
            str(k): {
                g: results[head][str(k)][g]["top3"]
                for g in ("overall", "leak_type=small", "leak_type=big")
                if g in results[head][str(k)]
            }
            for k in ctx.inp.ks
        }
    rep_dir = Path(args.reports_dir)
    out_dir = rep_dir if ctx.mode == "test" else ctx.runs_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    flow = flow_curve(ctx, per_case, out_dir / "final_leak_flow.png")
    done = [u["name"] for u in ctx.units if u["name"] in frames]
    rep = {
        "name": args.name,
        "command": "leaksense final-report",
        "mode": ctx.mode,
        "provenance": provenance(
            None,
            config=ctx.cfg.to_dict(),
            seed=sd,
            manifest_sha256=ctx.inp.meta.get("data_manifest_sha256"),
        ),
        "fingerprint": ctx.base_fp,
        "bootstrap": nb,
        "ks": ctx.inp.ks,
        "draws": ctx.base_fp["draws"],
        "headline": head,
        "headline_k": ctx.hk,
        "full_k": max(ctx.inp.ks),
        "plain": ctx.fc.get("plain"),
        "rule_pick": ctx.fc.get("rule_pick"),
        "n_train_cases": len(ctx.train.cases),
        "test_set": test_summary(ctx.test),
        "tercile_edges": ctx.test.tercile_edges,
        "data_notes": data_notes(rep_dir),
        "units_done": done,
        "units_missing": [u["name"] for u in ctx.units if u["name"] not in frames],
        "unit_seconds": {n: frames[n].attrs.get("seconds") for n in done},
        "placement": {
            "final_order": ctx.place["order_ids"],
            "coverage_order": [ctx.inp.net.sensor_ids[j] for j in ctx.place["coverage"]],
            "ks": ctx.place["ks"],
            "n_random": ctx.place["n_random"],
            "results": place,
        },
        "results": results,
        "headline_minus": vs_head,
        "hypotheses": hyp,
        "cv": {
            "top3": cv_numbers(ctx.fc, list(results), ctx.hk, max(ctx.inp.ks)),
            "hypotheses": cv_hypotheses(ctx.fc),
            "placement": cv_placement(ctx.fc),
        },
        "leak_flow_curve": flow,
    }
    if place is not None:
        from leaksense.place_cmds import plot_curve

        plot_curve(
            {"ks": ctx.place["ks"], "results": place, "model_label": head},
            out_dir / "final_placement_curve.png",
            title=f"Sensor placement, {head} ({'locked test' if ctx.mode == 'test' else ctx.mode})",
        )
    rep["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    js = out_dir / "final.json"
    js.write_text(json.dumps(rep, indent=1, default=str))
    from leaksense.final_report import render_markdown

    (out_dir / "FINAL.md").write_text(render_markdown(rep), encoding="utf-8")
    return rep


# ----------------------------------------------------------------------------- command


def cmd_final(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    ctx = setup(args)
    runs_dir = ctx.runs_dir
    keys = {u["name"]: unit_key(u, ctx) for u in ctx.units}
    done = {n: _done(runs_dir, n, keys[n]) for n in keys}
    print(
        f"final-report ({ctx.mode}): train {len(ctx.train.cases)} CV cases, "
        f"test {len(ctx.test.cases)} cases; {sum(done.values())}/{len(done)} units done"
    )
    if args.status:
        for n, ok in done.items():
            print(f"  {'done' if ok else '----'}  {n}")
        return EXIT_OK
    only = None
    if args.only:
        only = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = set(only) - set(keys)
        if unknown:
            raise ConfigError(f"unknown units {sorted(unknown)}; see --status")
    todo = [u for u in ctx.units if not done[u["name"]] and (only is None or u["name"] in only)]
    if todo and not args.report_only:
        if ctx.mode == "test":
            for u in todo:
                js = runs_dir / f"{u['name']}.json"
                if js.is_file():
                    raise ConfigError(
                        f"{u['name']} was already scored on the locked test with another config; "
                        "the test fold is scored once. Restore the config, or remove the file "
                        "by hand if you really mean to re-score."
                    )
            if not args.confirm_locked_test:
                raise ConfigError(
                    f"{len(todo)} units would score the locked test fold (once); "
                    "add --confirm-locked-test"
                )
        runs_dir.mkdir(parents=True, exist_ok=True)
        for u in todo:
            t1 = time.perf_counter()
            print(f"{u['name']}: fitting on {len(ctx.train.cases)} cases", flush=True)
            info, df = run_unit(u, ctx)
            secs = round(time.perf_counter() - t1, 1)
            meta = {"key": keys[u["name"]], "fp": ctx.base_fp, "unit": u, "seconds": secs}
            _save(runs_dir, u["name"], {**meta, **info}, df)
            lad = df[(df["method"] == "ladder") & (df["k"] == ctx.hk)]
            msg = f"k={ctx.hk} top-3 {100 * lad['top3'].mean():.1f}" if len(lad) else ""
            if u["kind"] == "fixed":
                msg = f"fixed k={u['k']} top-3 {100 * df['top3'].mean():.1f}"
            print(f"{u['name']}: {msg} ({secs:.0f}s)", flush=True)
            done[u["name"]] = True
    if only is not None and not args.report_only:
        print("--only: no report written")
        return EXIT_OK

    frames = {}
    for u in ctx.units:
        n = u["name"]
        if _done(runs_dir, n, keys[n]):
            df = pd.read_parquet(runs_dir / f"{n}.parquet")
            df.attrs["seconds"] = json.loads((runs_dir / f"{n}.json").read_text())["seconds"]
            frames[n] = df
    missing = [u["name"] for u in ctx.units if u["name"] not in frames]
    if missing and ctx.mode == "test":
        raise ConfigError(f"units not scored yet: {missing}")
    rep = build_report(args, ctx, frames, t0)
    out = Path(args.reports_dir) if ctx.mode == "test" else ctx.runs_dir / "report"
    head = rep["headline"]
    if head in rep["results"]:
        print("   k   top-3 (headline)   small top-3")
        for k in ctx.inp.ks:
            r = rep["results"][head][str(k)]
            s = r.get("leak_type=small", {}).get("top3", {}).get("value", float("nan"))
            print(f"{k:>4}   {100 * r['overall']['top3']['value']:16.1f}   {100 * s:11.1f}")
    print(f"done in {rep['elapsed_seconds']:.0f}s -> {out / 'FINAL.md'}, {out / 'final.json'}")
    return EXIT_OK
