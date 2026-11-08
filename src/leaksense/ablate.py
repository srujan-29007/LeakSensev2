"""CLI command for M10: ``leaksense ablate``.

Runs every (variant, model seed) of a GAT ablation through the same harness as ``train``, plus
the non-learned reference rungs, and aggregates them into ``reports/<config name>.{json,md}``.

- **Same everything.** Every run uses the same splits, features, k list, evaluation seed and
  draws, so ``evaluate.random_masks`` gives every run the same sensor sets per k. Only the
  model settings and the model seed differ. The aggregator refuses runs that disagree.
- **Resumable.** Each run writes ``<runs-dir>/<config>__<variant>_s<seed>.json`` and its
  per-case predictions (``.parquet``) next to it. A run whose files exist with the same key is
  skipped, so runs can be split over processes (``--variants``, ``--seeds``, ``--no-aggregate``)
  or machines (copy the runs dir back, then ``--aggregate-only``).
- **Paired comparisons.** Variants are compared case by case, on seed-averaged per-case metrics,
  with a bootstrap over scenarios: the same cases and sensor sets on both sides.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from leaksense.config import ConfigError, get_artifacts_dir
from leaksense.eval_cmds import EXIT_OK, add_common, load_inputs, load_sensitivity, run_and_report

COLS = ("top1", "top3", "top5", "hop", "dist_m")
DIFF_METRICS = ("top1", "top3", "top5", "dist_m_mean")
REPORT_GROUPS = (
    "overall",
    "pre_baseline_only",
    "leak_type=small",
    "leak_type=big",
    "kind=pure",
    "kind=solo",
    "leak_flow=T1",
    "leak_flow=T2",
    "leak_flow=T3",
)
H2_KS = (2, 3, 4, 6)
H1_KS = (8, 12, 16, 31)
SELECT_K = 8  # stage-2 rule: best seed-0 small-leak top-3 at k = 8 (SPEC H5 headline)
N_FINAL_PHYSICS = 2


@dataclass(frozen=True)
class AblationSpec:
    variants: dict[str, dict]  # name -> model overrides on the config's model block
    seeds: list[int]
    folds: list[int] | None  # None = all CV folds
    reference: list[str]  # non-learned rungs scored alongside, e.g. sigmatch_all
    baseline: str  # the variant the others are compared with


def parse_spec(cfg) -> AblationSpec:
    a = dict(cfg.ablation or {})
    unknown = set(a) - {"variants", "seeds", "folds", "reference", "baseline"}
    if unknown:
        raise ConfigError(f"unknown ablation keys: {sorted(unknown)}")
    variants = {str(k): dict(v or {}) for k, v in (a.get("variants") or {}).items()}
    if not variants:
        raise ConfigError("the config needs an ablation.variants block")
    for v, over in variants.items():
        if not re.fullmatch(r"[a-z0-9_]+", v):
            raise ConfigError(f"variant name {v!r}: use lowercase letters, digits and _")
        if "seed" in over or "kind" in over:
            raise ConfigError(f"variant {v}: seed and kind are set by the ablation, not a variant")
    baseline = str(a.get("baseline", "gat"))
    if baseline not in variants:
        raise ConfigError(f"ablation.baseline {baseline!r} is not a variant")
    folds = a.get("folds")
    return AblationSpec(
        variants=variants,
        seeds=[int(s) for s in a.get("seeds", [0])],
        folds=None if folds is None else [int(f) for f in folds],
        reference=[str(r) for r in a.get("reference", ["sigmatch_all"])],
        baseline=baseline,
    )


def add_parsers(sub) -> None:
    a = sub.add_parser(
        "ablate",
        help="M10: train every GAT variant x seed on the same folds and sensor sets, "
        "then write reports/<config name>.json and .md",
    )
    add_common(a, default_config="configs/ablation_cpu.yaml")
    a.add_argument("--variants", help="comma-separated variants to run (default: all)")
    a.add_argument("--seeds", help="comma-separated model seeds to run (default: config seeds)")
    a.add_argument("--runs-dir", default="reports/raw/ablation", help="per-run reports")
    g = a.add_mutually_exclusive_group()
    g.add_argument("--no-aggregate", action="store_true", help="only run (parallel workers)")
    g.add_argument("--aggregate-only", action="store_true", help="only aggregate existing runs")
    a.set_defaults(func=cmd_ablate)


# ----------------------------------------------------------------------------- runs


def _fingerprint(args, inp, spec: AblationSpec) -> dict:
    from leaksense.provenance import file_sha256

    return {
        "features_dir": inp.fdir.name,
        "features_sha256_12": inp.meta.get("features_sha256_12"),
        "splits_sha256": file_sha256(args.splits),
        "ks": list(inp.ks),
        "draws": int(args.draws),
        "eval_seed": int(args.seed),
        "folds": spec.folds,
    }


def _key(fp: dict, what: dict) -> str:
    blob = json.dumps({"fp": fp, "what": what}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _run_name(cfg_name: str, variant: str, seed: int | None) -> str:
    return f"{cfg_name}__{variant}" + ("" if seed is None else f"_s{seed}")


def _is_done(runs_dir: Path, name: str, key: str) -> bool:
    js, pq = runs_dir / f"{name}.json", runs_dir / f"{name}.parquet"
    if not (js.is_file() and pq.is_file()):
        return False
    return json.loads(js.read_text()).get("ablation", {}).get("key") == key


def _run(args, inp, name, model, model_name, folds, meta, runs_dir: Path, training=None) -> None:
    t0 = time.perf_counter()
    ns = copy.copy(args)
    ns.name, ns.reports_dir = name, str(runs_dir)

    def extra() -> dict:
        out = {"command": "leaksense ablate", "ablation": meta}
        if training is not None:
            out["training"] = training()
        return out

    run_and_report(ns, inp, model, model_name, t0, folds=folds, extra=extra)
    # keep the per-case predictions next to the run, so a runs dir is self-contained
    src = get_artifacts_dir() / "eval" / name / "predictions.parquet"
    shutil.copyfile(src, runs_dir / f"{name}.parquet")


def run_all(args, inp, spec: AblationSpec, variants: list[str], seeds: list[int]) -> None:
    from leaksense.models.baselines import MODELS, NEEDS_SENSITIVITY
    from leaksense.train_cmds import make_model

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    fp = _fingerprint(args, inp, spec)
    cfg = inp.cfg
    if cfg.model.get("kind") != "gat":
        raise ConfigError("the ablation's model block must be kind: gat")

    eval_folds = spec.folds or sorted(int(f) for f in inp.es.cases["fold"].unique())
    for ref in spec.reference:
        if ref not in MODELS:
            raise ConfigError(f"unknown reference rung {ref!r}; one of {sorted(MODELS)}")
        name = _run_name(cfg.name, f"ref_{ref}", None)
        key = _key(fp, {"reference": ref})
        if _is_done(runs_dir, name, key):
            print(f"{name}: done, skipped")
            continue
        res = {"sensitivity": load_sensitivity(inp)} if ref in NEEDS_SENSITIVITY else {}
        meta = {"key": key, "kind": "reference", "variant": ref, "seed": None, "fp": fp}
        _run(args, inp, name, MODELS[ref](inp.net, **res), ref, spec.folds, meta, runs_dir)

    for v in variants:
        for s in seeds:
            model_cfg = {**cfg.model, **spec.variants[v], "seed": s}
            name = _run_name(cfg.name, v, s)
            key = _key(fp, {"model": model_cfg})
            if _is_done(runs_dir, name, key):
                print(f"{name}: done, skipped")
                continue
            vinp = dataclasses.replace(inp, cfg=dataclasses.replace(cfg, model=model_cfg))
            model = make_model(vinp, model_cfg)
            meta = {"key": key, "kind": "model", "variant": v, "seed": s, "fp": fp}
            print(f"{name}: training {v} (seed {s}) on folds {spec.folds or 'all'}")
            _run(
                args,
                vinp,
                name,
                model,
                "gat",
                spec.folds,
                meta,
                runs_dir,
                training=lambda m=model: {**m.describe(), "eval_folds": eval_folds},
            )


# ----------------------------------------------------------------------------- aggregate


@dataclass
class Run:
    variant: str
    seed: int | None
    kind: str
    report: dict
    values: np.ndarray  # (n_ks, n_cases, 5), cases in the aggregator's case order


def load_runs(runs_dir: Path, cfg_name: str, fp: dict, case_ids: list[str], ks) -> list[Run]:
    runs = []
    for js in sorted(Path(runs_dir).glob(f"{cfg_name}__*.json")):
        rep = json.loads(js.read_text())
        meta = rep.get("ablation")
        if not meta:
            continue
        if meta["fp"] != fp:
            diff = sorted(k for k in fp if meta["fp"].get(k) != fp[k])
            raise ConfigError(
                f"{js.name} was run with different {diff}; every run must share the splits, "
                "features, k list, evaluation seed, draws and folds"
            )
        pq = js.with_suffix(".parquet")
        if not pq.is_file():
            raise ConfigError(f"{js.name} has no predictions file {pq.name}")
        p = pd.read_parquet(pq)
        vals = np.zeros((len(ks), len(case_ids), len(COLS)))
        for i, k in enumerate(ks):
            pk = p[p["k"] == k].set_index("case_id")
            missing = set(case_ids) - set(pk.index)
            if missing:
                raise ConfigError(f"{js.name}: {len(missing)} cases missing at k = {k}")
            vals[i] = pk.loc[case_ids, list(COLS)].to_numpy(dtype=float)
        runs.append(Run(meta["variant"], meta["seed"], meta["kind"], rep, vals))
    return runs


def _ci(samples: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _paired(d: np.ndarray, scen: np.ndarray, n_boot: int, seed: int) -> dict:
    """Mean per-case difference (n, 5) with a bootstrap CI over scenarios, per metric."""
    from leaksense.evaluate import bootstrap_samples

    samples = bootstrap_samples(d, scen, n_boot, seed)
    point = {"top1": d[:, 0].mean(), "top3": d[:, 1].mean(), "top5": d[:, 2].mean()}
    point["dist_m_mean"] = d[:, 4].mean()
    out = {}
    for m in DIFF_METRICS:
        lo, hi = _ci(samples[m])
        out[m] = {"value": float(point[m]), "lo": lo, "hi": hi}
    return out


def _verdict(d: dict) -> str:
    return "better" if d["lo"] > 0 else "worse" if d["hi"] < 0 else "no clear difference"


def aggregate(args, inp, spec: AblationSpec) -> dict:
    from leaksense.evaluate import bootstrap, groups, summarise
    from leaksense.provenance import provenance

    fp = _fingerprint(args, inp, spec)
    cases = inp.es.cases
    keep = np.ones(len(cases), bool) if spec.folds is None else cases["fold"].isin(spec.folds)
    keep = np.asarray(keep)
    case_ids = cases.loc[keep, "case_id"].tolist()
    scen = cases.loc[keep, "scenario_id"].to_numpy()
    grp = {g: sel[keep] for g, sel in groups(inp.es).items() if g in REPORT_GROUPS}
    ks = list(inp.ks)
    runs = load_runs(Path(args.runs_dir), inp.cfg.name, fp, case_ids, ks)
    if not runs:
        raise ConfigError(f"no runs for {inp.cfg.name} in {args.runs_dir}")

    by_variant: dict[str, list[Run]] = {}
    for r in runs:
        by_variant.setdefault(r.variant, []).append(r)
    for rs in by_variant.values():
        rs.sort(key=lambda r: -1 if r.seed is None else r.seed)
    order = [v for v in spec.variants if v in by_variant]
    order += [r for r in spec.reference if r in by_variant]
    mean = {v: np.mean([r.values for r in by_variant[v]], axis=0) for v in order}
    n_boot, seed = int(args.bootstrap), int(args.seed)

    variants = {}
    for v in order:
        rs = by_variant[v]
        res = {}
        for i, k in enumerate(ks):
            gk = {}
            for g, sel in grp.items():
                if not sel.any():
                    continue
                point = summarise(mean[v][i][sel])
                ci = bootstrap(mean[v][i][sel], scen[sel], n_boot, seed)
                per_seed = [summarise(r.values[i][sel]) for r in rs]
                gk[g] = {
                    "n": int(sel.sum()),
                    **{
                        m: {
                            "value": point[m],
                            "lo": ci[m][0],
                            "hi": ci[m][1],
                            "seed_min": min(p[m] for p in per_seed),
                            "seed_max": max(p[m] for p in per_seed),
                        }
                        for m in ("top1", "top3", "top5", "dist_m_median")
                    },
                }
            res[str(k)] = gk
        variants[v] = {
            "kind": rs[0].kind,
            "seeds": [r.seed for r in rs],
            "model": None
            if rs[0].kind == "reference"
            else rs[0].report["provenance"]["config"]["model"],
            "elapsed_seconds": [r.report["elapsed_seconds"] for r in rs],
            "git": sorted({r.report["provenance"]["git"]["commit"] or "?" for r in rs}),
            "dirty": any(r.report["provenance"]["git"]["dirty"] for r in rs),
            "training": [
                {
                    "seed": r.seed,
                    "epochs": [f["epochs"] for f in r.report.get("training", {}).get("folds", [])],
                    "physics_params": [
                        f.get("physics_params")
                        for f in r.report.get("training", {}).get("folds", [])
                    ],
                }
                for r in rs
                if r.kind == "model"
            ],
            "results": res,
        }

    comparisons = {}
    against = [spec.baseline] + list(spec.reference)
    for ref in against:
        if ref not in mean:
            continue
        comp = {}
        for v in order:
            if v == ref or variants[v]["kind"] == "reference":
                continue
            ck = {}
            for i, k in enumerate(ks):
                d = mean[v][i] - mean[ref][i]
                ck[str(k)] = {
                    g: _paired(d[sel], scen[sel], n_boot, seed)
                    for g, sel in grp.items()
                    if g in ("overall", "leak_type=small", "leak_type=big") and sel.any()
                }
            comp[v] = ck
        comparisons[ref] = comp

    verdicts = _verdicts(spec, variants, comparisons, by_variant, ks, grp)
    report = {
        "name": inp.cfg.name,
        "command": "leaksense ablate",
        "provenance": provenance(
            None,
            config=inp.cfg.to_dict(),
            seed=seed,
            manifest_sha256=inp.meta.get("data_manifest_sha256"),
        ),
        "fingerprint": fp,
        "bootstrap": n_boot,
        "n_cases": len(case_ids),
        "tercile_edges": inp.es.tercile_edges,
        "baseline": spec.baseline,
        "reference": spec.reference,
        "variants": variants,
        "comparisons": comparisons,
        "verdicts": verdicts,
    }
    return report


def _verdicts(spec, variants, comparisons, by_variant, ks, grp) -> dict:
    base = comparisons.get(spec.baseline, {})
    physics = [
        v
        for v, d in variants.items()
        if d["kind"] == "model" and (d["model"] or {}).get("physics", "none") != "none"
    ]
    h2 = {}
    for v in physics:
        per_k = {
            str(k): _verdict(base[v][str(k)]["overall"]["top3"])
            for k in H2_KS
            if v in base and str(k) in base[v]
        }
        n_better = sum(x == "better" for x in per_k.values())
        h2[v] = {
            "per_k": per_k,
            "holds": "yes" if per_k and n_better == len(per_k) else "partly" if n_better else "no",
        }

    h1 = {}
    refs = [r for r in spec.reference if r in variants]
    for v, d in variants.items():
        if d["kind"] != "model" or not refs:
            continue
        per_k = {}
        for k in H1_KS:
            if str(k) not in d["results"]:
                continue
            ours = d["results"][str(k)]["overall"]["top3"]
            best = max(
                refs, key=lambda r: variants[r]["results"][str(k)]["overall"]["top3"]["value"]
            )
            theirs = variants[best]["results"][str(k)]["overall"]["top3"]
            per_k[str(k)] = {
                "vs": best,
                "holds": bool(ours["lo"] > theirs["hi"]),
                "worse": bool(ours["hi"] < theirs["lo"]),
            }
        h1[v] = per_k

    # stage-2 rule (fixed in PLAN M10): baseline + the two physics variants with the best
    # seed-0 small-leak top-3 at k = 8
    i8 = ks.index(SELECT_K) if SELECT_K in ks else None
    small = grp.get("leak_type=small")
    score = {}
    if i8 is not None and small is not None:
        for v in physics:
            r0 = [r for r in by_variant[v] if r.seed == 0]
            if r0:
                score[v] = float(r0[0].values[i8][small, 1].mean())
    best = sorted(score, key=lambda v: -score[v])[:N_FINAL_PHYSICS]
    return {
        "H2_physics_vs_plain_top3_k_le_6": h2,
        "H1_vs_best_nonlearned_top3_k_ge_8": h1,
        "stage2": {
            "rule": f"{spec.baseline} + the {N_FINAL_PHYSICS} physics variants with the best "
            f"seed-0 small-leak top-3 at k = {SELECT_K}",
            "seed0_small_top3_k8": score,
            "final_variants": [spec.baseline, *best] if score else [],
        },
    }


# ----------------------------------------------------------------------------- markdown


def _pct(d: dict, seeds: bool = False) -> str:
    s = f"{100 * d['value']:.1f} [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}]"
    if seeds and d.get("seed_max", d["value"]) - d.get("seed_min", d["value"]) > 1e-12:
        s += f" ({100 * d['seed_min']:.1f}–{100 * d['seed_max']:.1f})"
    return s


def _diff(d: dict) -> str:
    return f"{100 * d['value']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]"


def render_markdown(rep: dict) -> str:
    prov, fp, V = rep["provenance"], rep["fingerprint"], rep["variants"]
    ks = fp["ks"]
    names = list(V)
    commits = sorted({c[:7] for v in V.values() for c in v["git"]})
    dirty = any(v["dirty"] for v in V.values())
    lines = [
        f"# Ablation: {rep['name']} (PLAN.md M10)",
        "",
        f"- **Generated by:** `leaksense ablate --config` (config `{rep['name']}`), "
        f"aggregated at commit `{(prov['git']['commit'] or '?')[:7]}`; runs at "
        f"{', '.join(f'`{c}`' for c in commits)}{' (some dirty)' if dirty else ''}.",
        f"- **Data:** manifest `{(prov['data_manifest_sha256'] or '?')[:12]}…`, features "
        f"`{fp['features_dir']}`, splits sha `{fp['splits_sha256'][:12]}`.",
        f"- **Cases:** {rep['n_cases']} CV cases, folds "
        f"{'all' if fp['folds'] is None else fp['folds']}. The locked test fold is not used.",
        f"- **Sensor sets:** {fp['draws']} seeded random sets per k < 31 (evaluation seed "
        f"{fp['eval_seed']}), identical for every run.",
        f"- **CIs:** 95% bootstrap over scenarios ({rep['bootstrap']} resamples) on per-case "
        "metrics averaged over model seeds. In brackets after a CI: min–max over seeds.",
        "- **Paired differences:** per case, variant − comparator, on the same cases and sensor "
        "sets.",
        "",
        "## Runs",
        "",
        "| variant | model change | seeds | epochs per fold | learned physics | time (s) |",
        "|---|---|---|---|---|---|",
    ]
    base_model = V[rep["baseline"]]["model"] or {}
    for v in names:
        d = V[v]
        if d["kind"] == "reference":
            lines.append(
                f"| `{v}` | non-learned reference | - | - | - | {d['elapsed_seconds'][0]:.0f} |"
            )
            continue
        change = {k: x for k, x in d["model"].items() if k != "seed" and base_model.get(k) != x}
        ep = "; ".join(",".join(map(str, t["epochs"])) for t in d["training"])
        phys = "; ".join(
            ",".join("/".join(f"{k}={x:g}" for k, x in p.items()) for p in t["physics_params"] if p)
            for t in d["training"]
        ).strip("; ")
        secs = ", ".join(f"{s:.0f}" for s in d["elapsed_seconds"])
        lines.append(
            f"| `{v}` | {json.dumps(change) if change else 'plain'} | "
            f"{', '.join(map(str, d['seeds']))} | {ep} | {phys or '-'} | {secs} |"
        )

    def table(title, group, metric):
        out = ["", f"## {title}", "", "| variant | " + " | ".join(f"k = {k}" for k in ks) + " |"]
        out.append("|---|" + "---|" * len(ks))
        for v in names:
            cells = []
            for k in ks:
                g = V[v]["results"][str(k)].get(group)
                cells.append(_pct(g[metric], seeds=True) if g else "-")
            out.append(f"| `{v}` | " + " | ".join(cells) + " |")
        return out

    lines += table("Top-3 accuracy, all cases (%)", "overall", "top3")
    lines += table(
        "Top-3 accuracy, small leaks (%) — the SPEC H5 headline", "leak_type=small", "top3"
    )
    lines += table("Top-1 accuracy, all cases (%)", "overall", "top1")

    for ref, comp in rep["comparisons"].items():
        for group, label in (("overall", "all cases"), ("leak_type=small", "small leaks")):
            lines += [
                "",
                f"## Paired top-3 difference vs `{ref}`, {label} (points)",
                "",
                "| variant | " + " | ".join(f"k = {k}" for k in ks) + " |",
                "|---|" + "---|" * len(ks),
            ]
            for v, ck in comp.items():
                lines.append(
                    f"| `{v}` | " + " | ".join(_diff(ck[str(k)][group]["top3"]) for k in ks) + " |"
                )

    k8 = "8" if "8" in V[names[0]]["results"] else str(ks[0])
    tg = ["leak_type=small", "leak_type=big", "kind=pure", "kind=solo"]
    tg += ["leak_flow=T1", "leak_flow=T2", "leak_flow=T3"]
    edges = rep["tercile_edges"].get("leak_flow_lps", [])
    lines += [
        "",
        f"## Breakdown at k = {k8}: top-3 (%)",
        "",
        f"Leak-flow tercile edges: {', '.join(f'{e:.0f}' for e in edges)} L/s.",
        "",
        "| variant | " + " | ".join(tg) + " |",
        "|---|" + "---|" * len(tg),
    ]
    for v in names:
        r = V[v]["results"][k8]
        lines.append(
            f"| `{v}` | "
            + " | ".join(f"{100 * r[g]['top3']['value']:.1f}" if g in r else "-" for g in tg)
            + " |"
        )

    vd = rep["verdicts"]
    lines += ["", "## Hypotheses (CV, not the locked test)", ""]
    lines.append(
        "**H2** (physics inputs improve on the plain GAT at k ≤ 6): paired top-3 difference vs "
        f"`{rep['baseline']}`, CI above 0 = better."
    )
    lines.append("")
    for v, h in vd["H2_physics_vs_plain_top3_k_le_6"].items():
        per = ", ".join(f"k={k}: {x}" for k, x in h["per_k"].items())
        lines.append(f"- `{v}`: **{h['holds']}** ({per})")
    lines += [
        "",
        "**H1** (beats the best non-learned rung on top-3 at k ≥ 8, non-overlapping CIs):",
        "",
    ]
    for v, per in vd["H1_vs_best_nonlearned_top3_k_ge_8"].items():
        cells = ", ".join(
            f"k={k}: {'holds' if x['holds'] else 'worse' if x['worse'] else 'overlap'} vs {x['vs']}"
            for k, x in per.items()
        )
        lines.append(f"- `{v}`: {cells}")
    s2 = vd["stage2"]
    lines += ["", f"**Stage-2 rule:** {s2['rule']}.", ""]
    if s2["seed0_small_top3_k8"]:
        sc = ", ".join(f"`{v}` {100 * x:.1f}" for v, x in s2["seed0_small_top3_k8"].items())
        lines.append(f"Seed-0 small-leak top-3 at k = 8: {sc}.")
        lines.append(f"Final variants: {', '.join(f'`{v}`' for v in s2['final_variants'])}.")
    lines.append("")
    return "\n".join(lines)


def write_report(rep: dict, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"{rep['name']}.json"
    out.write_text(json.dumps(rep, indent=2) + "\n")
    out.with_suffix(".md").write_text(render_markdown(rep), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------- command


def cmd_ablate(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    inp = load_inputs(args)
    spec = parse_spec(inp.cfg)
    variants = list(spec.variants)
    if args.variants:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        bad = [v for v in variants if v not in spec.variants]
        if bad:
            raise ConfigError(f"unknown variants {bad}; the config has {list(spec.variants)}")
    seeds = spec.seeds if not args.seeds else [int(s) for s in args.seeds.split(",")]
    if not args.aggregate_only:
        run_all(args, inp, spec, variants, seeds)
    if args.no_aggregate:
        print(f"runs done in {time.perf_counter() - t0:.0f}s")
        return EXIT_OK
    rep = aggregate(args, inp, spec)
    out = write_report(rep, Path(args.reports_dir))
    s2 = rep["verdicts"]["stage2"]
    print(f"aggregated {len(rep['variants'])} variants -> {out}, {out.with_suffix('.md')}")
    if s2["final_variants"]:
        print(f"stage-2 final variants: {', '.join(s2['final_variants'])}")
    print(f"done in {time.perf_counter() - t0:.0f}s")
    return EXIT_OK
