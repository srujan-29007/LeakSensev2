"""CLI command for M10: ``leaksense ablate``.

Runs every (variant, model seed) of a GAT ablation through the same harness as ``train``, plus
the non-learned reference rungs, and aggregates them into ``reports/<config name>.{json,md}``.

- **Same everything.** Every run uses the same splits, features, k list, evaluation seed and
  draws, so ``evaluate.random_masks`` gives every run the same sensor sets per k. Only the
  model settings and the model seed differ. The aggregator refuses runs that disagree.
- **Resumable.** Each (variant, seed, fold) writes
  ``<runs-dir>/<config>__<variant>_s<seed>_f<fold>.json`` and its per-case predictions
  (``.parquet``) as soon as it finishes. Files are staged in ``<runs-dir>/.partial`` and moved in
  with the ``.json`` last, so a killed run leaves no half-written result. A run whose files exist
  with the same key is skipped, so runs can be split over processes (``--variants``, ``--seeds``,
  ``--no-aggregate``) or machines (copy the runs dir back, then ``--aggregate-only``).
  ``--status`` lists done and missing runs without training. Saved runs of the same config with a
  different fingerprint (features, splits, k list, draws, seed, folds) stop the command before
  any training, because they could never be aggregated with the new runs.
- **Paired comparisons.** Variants are compared case by case, on seed-averaged per-case metrics,
  with a bootstrap over scenarios: the same cases and sensor sets on both sides.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
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
SELECT_K = 8
SELECT_KS = (2, 3, 4, 6, 8)  # stage-2 selection budgets (inner validation)
N_FINAL_PHYSICS = 2
FIXED_FINAL = ("gat", "gat_noembed")  # always in stage 2
PHYSICS_ORDER = ("none", "prior", "node", "edge")  # tie-break: fewer added inputs first
ADOPT_SEEDS = (0, 1, 2)
STAGE2_RULE = (
    "gat and gat_noembed, plus the 2 physics variants with the highest inner-validation score: "
    "seed-0 runs, each fold's chosen inner-validation net (before the refit) scored on that "
    "fold's inner-validation cases with the harness's 20 seeded sensor sets; score = mean over "
    "k in {2, 3, 4, 6, 8} of the case-weighted mean over the 5 folds of case-level top-3. "
    "Ties (equal to 0.1 point) are broken by inner small-leak top-3 at k = 8 (to 0.1 point), "
    "then by fewer added inputs (prior < node < edge; mid < all levels), then by name. Outer-fold "
    "results are not used."
)
NOEMBED_RULE = (
    "adopt node_embed: 0 as the default for M11 on if, with seeds 0, 1 and 2 of both gat and "
    "gat_noembed over all 5 folds, the paired top-3 difference (gat_noembed - gat, all cases, "
    "seed-averaged per case) has its 95% CI lower bound above 0 at k = 8 and a point estimate "
    ">= 0 at every k; otherwise keep the embedding"
)


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
    g.add_argument(
        "--status", action="store_true", help="list done and missing runs, train nothing"
    )
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


def _run_name(cfg_name: str, variant: str, seed: int | None, fold: int | None = None) -> str:
    name = f"{cfg_name}__{variant}" + ("" if seed is None else f"_s{seed}")
    return name + ("" if fold is None else f"_f{fold}")


def _is_done(runs_dir: Path, name: str, key: str) -> bool:
    js, pq = runs_dir / f"{name}.json", runs_dir / f"{name}.parquet"
    if not (js.is_file() and pq.is_file()):
        return False
    try:
        meta = json.loads(js.read_text()).get("ablation", {})
    except json.JSONDecodeError:  # cut off mid-write (before staging existed): run it again
        return False
    return meta.get("key") == key


def check_runs_dir(runs_dir: Path, cfg_name: str, fp: dict) -> None:
    """Refuse to add runs next to saved runs of the same config with a different fingerprint."""
    for js in sorted(Path(runs_dir).glob(f"{cfg_name}__*.json")):
        try:
            meta = json.loads(js.read_text()).get("ablation")
        except json.JSONDecodeError:
            continue
        if meta and meta.get("fp") != fp:
            diff = sorted(k for k in fp if meta["fp"].get(k) != fp[k])
            raise ConfigError(
                f"{js.name} in {runs_dir} was run with different {diff} "
                f"(saved {[meta['fp'].get(k) for k in diff]}, now {[fp[k] for k in diff]}); "
                "new runs could not be aggregated with it. Rebuild the same features and "
                "splits, or use another --runs-dir"
            )


def _run(args, inp, name, model, model_name, folds, meta, runs_dir: Path, training=None) -> None:
    t0 = time.perf_counter()
    stage = runs_dir / ".partial"
    ns = copy.copy(args)
    ns.name, ns.reports_dir = name, str(stage)

    def extra() -> dict:
        out = {"command": "leaksense ablate", "ablation": meta}
        if training is not None:
            out["training"] = training()
        return out

    run_and_report(ns, inp, model, model_name, t0, folds=folds, extra=extra)
    # keep the per-case predictions next to the run, so a runs dir is self-contained. The .json
    # is moved in last: once it is there, the run is complete
    src = get_artifacts_dir() / "eval" / name / "predictions.parquet"
    shutil.copyfile(src, stage / f"{name}.parquet")
    for ext in ("parquet", "md", "json"):
        os.replace(stage / f"{name}.{ext}", runs_dir / f"{name}.{ext}")


def plan_runs(args, inp, spec: AblationSpec, variants: list[str], seeds: list[int]) -> list[dict]:
    """Every run, in order: the reference rungs, then each (variant, seed, fold)."""
    from leaksense.models.baselines import MODELS

    fp = _fingerprint(args, inp, spec)
    cfg = inp.cfg
    if cfg.model.get("kind") != "gat":
        raise ConfigError("the ablation's model block must be kind: gat")
    plan = []
    for ref in spec.reference:
        if ref not in MODELS:
            raise ConfigError(f"unknown reference rung {ref!r}; one of {sorted(MODELS)}")
        name = _run_name(cfg.name, f"ref_{ref}", None)
        plan.append({"name": name, "key": _key(fp, {"reference": ref}), "reference": ref})
    eval_folds = spec.folds or sorted(int(f) for f in inp.es.cases["fold"].unique())
    for v in variants:
        for s in seeds:
            model_cfg = {**cfg.model, **spec.variants[v], "seed": s}
            for f in eval_folds:
                key = _key(fp, {"model": model_cfg, "fold": f})
                name = _run_name(cfg.name, v, s, f)
                plan.append({"name": name, "key": key, "variant": v, "seed": s, "fold": f})
    return plan


def print_status(args, inp, spec: AblationSpec, variants: list[str], seeds: list[int]) -> None:
    runs_dir = Path(args.runs_dir)
    check_runs_dir(runs_dir, inp.cfg.name, _fingerprint(args, inp, spec))
    plan = plan_runs(args, inp, spec, variants, seeds)
    n_done = 0
    for r in plan:
        done = _is_done(runs_dir, r["name"], r["key"])
        n_done += done
        print(f"{'done' if done else 'todo'}  {r['name']}")
    print(f"{n_done} of {len(plan)} runs done in {runs_dir}")


def run_all(args, inp, spec: AblationSpec, variants: list[str], seeds: list[int]) -> None:
    from leaksense.models.baselines import MODELS, NEEDS_SENSITIVITY
    from leaksense.train_cmds import make_model

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    fp = _fingerprint(args, inp, spec)
    check_runs_dir(runs_dir, inp.cfg.name, fp)
    cfg = inp.cfg
    plan = plan_runs(args, inp, spec, variants, seeds)

    for r in plan:
        if "reference" not in r:
            continue
        ref, name, key = r["reference"], r["name"], r["key"]
        if _is_done(runs_dir, name, key):
            print(f"{name}: done, skipped")
            continue
        res = {"sensitivity": load_sensitivity(inp)} if ref in NEEDS_SENSITIVITY else {}
        meta = {
            "key": key,
            "kind": "reference",
            "variant": ref,
            "seed": None,
            "fold": None,
            "fp": fp,
        }
        _run(args, inp, name, MODELS[ref](inp.net, **res), ref, spec.folds, meta, runs_dir)

    # one run per (variant, seed, fold): each fold is saved as soon as it finishes, so an
    # interrupted run resumes from the next fold
    for r in plan:
        if "reference" in r:
            continue
        v, s, f, name, key = r["variant"], r["seed"], r["fold"], r["name"], r["key"]
        if _is_done(runs_dir, name, key):
            print(f"{name}: done, skipped")
            continue
        model_cfg = {**cfg.model, **spec.variants[v], "seed": s}
        vinp = dataclasses.replace(inp, cfg=dataclasses.replace(cfg, model=model_cfg))
        model = make_model(vinp, model_cfg)
        meta = {"key": key, "kind": "model", "variant": v, "seed": s, "fold": f, "fp": fp}
        print(f"{name}: training {v} (seed {s}), scoring fold {f}", flush=True)
        _run(
            args,
            vinp,
            name,
            model,
            "gat",
            [f],
            meta,
            runs_dir,
            training=lambda m=model, f=f: {**m.describe(), "eval_folds": [f]},
        )


# ----------------------------------------------------------------------------- aggregate


@dataclass
class Run:
    """One (variant, seed): its per-fold reports merged; a reference rung is one report."""

    variant: str
    seed: int | None
    kind: str
    reports: list[dict]  # one per fold, in fold order (one for a reference)
    values: np.ndarray  # (n_ks, n_cases, 5), cases in the aggregator's case order

    @property
    def model(self) -> dict | None:
        return (
            None if self.kind == "reference" else self.reports[0]["provenance"]["config"]["model"]
        )

    @property
    def fold_history(self) -> list[dict]:
        return [h for r in self.reports for h in r.get("training", {}).get("folds", [])]


def load_runs(
    runs_dir: Path, cfg_name: str, fp: dict, case_ids: list[str], ks, folds: list[int]
) -> tuple[list[Run], list[str]]:
    """Runs for ``cfg_name`` -> (complete runs, names of incomplete (variant, seed) pairs)."""
    parts: dict[tuple, list] = {}
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
        key = (meta["variant"], meta["seed"], meta["kind"])
        parts.setdefault(key, []).append((meta.get("fold"), rep, pd.read_parquet(pq)))

    runs, incomplete = [], []
    for (variant, seed, kind), items in parts.items():
        items.sort(key=lambda t: -1 if t[0] is None else t[0])
        got = [f for f, _, _ in items]
        if kind == "model" and got != folds:
            incomplete.append(f"{variant} seed {seed}: folds {got} of {folds}")
            continue
        p = pd.concat([t[2] for t in items])
        vals = np.zeros((len(ks), len(case_ids), len(COLS)))
        for i, k in enumerate(ks):
            pk = p[p["k"] == k].set_index("case_id")
            missing = set(case_ids) - set(pk.index)
            if missing:
                raise ConfigError(f"{variant} seed {seed}: {len(missing)} cases missing at k = {k}")
            vals[i] = pk.loc[case_ids, list(COLS)].to_numpy(dtype=float)
        runs.append(Run(variant, seed, kind, [t[1] for t in items], vals))
    return runs, incomplete


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
    folds = spec.folds or sorted(int(f) for f in cases["fold"].unique())
    runs, incomplete = load_runs(Path(args.runs_dir), inp.cfg.name, fp, case_ids, ks, folds)
    if not any(r.variant == spec.baseline for r in runs):
        raise ConfigError(
            f"no complete {spec.baseline} run for {inp.cfg.name} in {args.runs_dir}"
            + (f"; incomplete: {incomplete}" if incomplete else "")
        )

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
            "model": rs[0].model,
            "elapsed_seconds": [round(sum(x["elapsed_seconds"] for x in r.reports), 1) for r in rs],
            "git": sorted({x["provenance"]["git"]["commit"] or "?" for r in rs for x in r.reports}),
            "dirty": any(x["provenance"]["git"]["dirty"] for r in rs for x in r.reports),
            "training": [
                {
                    "seed": r.seed,
                    "epochs": [h["epochs"] for h in r.fold_history],
                    "physics_params": [h.get("physics_params") for h in r.fold_history],
                    "inner_eval": [h.get("inner_eval") for h in r.fold_history],
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

    verdicts = _verdicts(spec, variants, comparisons, by_variant, ks)
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
        "incomplete_runs": incomplete,
        "variants": variants,
        "comparisons": comparisons,
        "verdicts": verdicts,
    }
    return report


def _verdicts(spec, variants, comparisons, by_variant, ks) -> dict:
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

    return {
        "H2_physics_vs_plain_top3_k_le_6": h2,
        "H1_vs_best_nonlearned_top3_k_ge_8": h1,
        "stage2": stage2_selection(spec, variants, by_variant),
        "noembed_adoption": noembed_adoption(spec, variants, comparisons, ks),
    }


def inner_score(run: Run) -> dict | None:
    """Inner-validation selection score of one run, pooled over folds (None if not recorded).

    score = mean over k in SELECT_KS of the case-weighted mean over folds of inner top-3;
    small_k8 = the same for small leaks at k = 8 (first tie-break).
    """
    hist = [h.get("inner_eval") for h in run.fold_history]
    if not hist or any(not h or not all(str(k) in h for k in SELECT_KS) for h in hist):
        return None
    per_k = []
    for k in SELECT_KS:
        n = sum(h[str(k)]["n"] for h in hist)
        per_k.append(sum(h[str(k)]["top3"] * h[str(k)]["n"] for h in hist) / n)
    ns = sum(h["8"].get("n_small", 0) for h in hist)
    small = sum(h["8"].get("small_top3", 0) * h["8"].get("n_small", 0) for h in hist)
    return {"score": float(np.mean(per_k)), "small_k8": small / ns if ns else 0.0}


def stage2_selection(spec, variants, by_variant) -> dict:
    """PLAN M10 stage-2 rule: inner-validation scores of the seed-0 runs only."""
    physics = [
        v
        for v, d in variants.items()
        if d["kind"] == "model" and (d["model"] or {}).get("physics", "none") != "none"
    ]
    scores, ineligible = {}, []
    for v in physics:
        r0 = [r for r in by_variant[v] if r.seed == 0]
        sc = inner_score(r0[0]) if r0 else None
        if sc is None:
            ineligible.append(v)
        else:
            scores[v] = sc

    def key(v):
        d = variants[v]["model"]
        return (
            -round(100 * scores[v]["score"], 1),
            -round(100 * scores[v]["small_k8"], 1),
            PHYSICS_ORDER.index(d.get("physics")),
            d.get("s_levels", "all") != "mid",  # fewer signatures first
            v,
        )

    ranked = sorted(scores, key=key)
    fixed = [x for x in FIXED_FINAL if x in variants]
    return {
        "rule": STAGE2_RULE,
        "inner_scores": scores,
        "ranked": ranked,
        "ineligible_no_inner_eval": ineligible,
        "final_variants": fixed + ranked[:N_FINAL_PHYSICS] if scores else [],
    }


def noembed_adoption(spec, variants, comparisons, ks) -> dict:
    """PLAN M10 criterion for making node_embed: 0 the default (needs seeds 0, 1, 2 of both)."""
    out = {"rule": NOEMBED_RULE}
    have = {v: set(variants.get(v, {}).get("seeds", [])) for v in ("gat", "gat_noembed")}
    if not all(set(ADOPT_SEEDS) <= s for s in have.values()):
        return {**out, "decision": "pending", "why": f"needs seeds {ADOPT_SEEDS} of both"}
    if set(variants["gat"]["seeds"]) != set(variants["gat_noembed"]["seeds"]):
        return {**out, "decision": "pending", "why": "gat and gat_noembed have different seeds"}
    comp = comparisons.get("gat", {}).get("gat_noembed")
    if comp is None or str(SELECT_K) not in comp:
        return {**out, "decision": "pending", "why": "no paired comparison"}
    d8 = comp[str(SELECT_K)]["overall"]["top3"]
    points = {str(k): comp[str(k)]["overall"]["top3"]["value"] for k in ks}
    ok = d8["lo"] > 0 and all(x >= 0 for x in points.values())
    return {
        **out,
        "decision": "adopt node_embed: 0" if ok else "keep the embedding",
        "k8_paired_top3": d8,
        "points": points,
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
    lines += ["", "## Stage-2 selection (inner validation only)", "", f"Rule: {s2['rule']}", ""]
    if s2["inner_scores"]:
        lines += [
            "| rank | variant | inner score (top-3, k = 2–8) | inner small-leak top-3, k = 8 |"
        ]
        lines += ["|---|---|---|---|"]
        for n, v in enumerate(s2["ranked"], 1):
            sc = s2["inner_scores"][v]
            lines.append(f"| {n} | `{v}` | {100 * sc['score']:.1f} | {100 * sc['small_k8']:.1f} |")
        lines += ["", f"**Final variants:** {', '.join(f'`{v}`' for v in s2['final_variants'])}."]
    if s2["ineligible_no_inner_eval"]:
        lines.append(f"No inner-validation scores (not ranked): {s2['ineligible_no_inner_eval']}.")
    na = vd["noembed_adoption"]
    lines += ["", "## Learned node embedding", "", f"Rule: {na['rule']}.", ""]
    if na["decision"] == "pending":
        lines.append(f"**Decision: pending** ({na['why']}).")
    else:
        d = na["k8_paired_top3"]
        lines.append(
            f"**Decision: {na['decision']}.** Paired top-3 at k = 8: {_diff(d)}; points by k: "
            + ", ".join(f"{k}: {100 * x:+.1f}" for k, x in na["points"].items())
            + "."
        )
    if rep.get("incomplete_runs"):
        lines += ["", f"**Incomplete runs (left out):** {'; '.join(rep['incomplete_runs'])}."]
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
    if args.status:
        print_status(args, inp, spec, variants, seeds)
        return EXIT_OK
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
