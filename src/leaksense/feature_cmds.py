"""CLI commands for M4: ``baseline-study``, ``build-features``, ``show-case``."""

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
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/full.yaml", help="run config YAML")
    common.add_argument("--data-dir", help=f"dataset HanoiOK folder (default: ${DATA_DIR_VAR})")
    common.add_argument(
        "--cases", help="case index (default: $LEAKSENSE_ARTIFACTS_DIR/cases.parquet)"
    )
    common.add_argument("--quiet", action="store_true", help="no progress output")

    s = sub.add_parser(
        "baseline-study",
        parents=[common],
        help="measure baseline methods, noise floor and drift (reports/)",
    )
    s.add_argument("--splits", default="splits/splits_v1.json", help="splits JSON (test excluded)")
    s.add_argument("--out", default="reports/baseline_choice", help="report path without suffix")
    s.set_defaults(func=cmd_baseline_study)

    b = sub.add_parser(
        "build-features", parents=[common], help="cache pressures, build windowed dH"
    )
    b.set_defaults(func=cmd_build_features)

    c = sub.add_parser("show-case", parents=[common], help="per-junction mean dH of one case")
    c.add_argument("case", help="case id, e.g. S034-pure, or a scenario number")
    c.set_defaults(func=cmd_show_case)


def _setup(args):
    from leaksense.data.features import PressureCache
    from leaksense.data.validate import manifest
    from leaksense.runconfig import load_run_config

    data_dir = get_data_dir({DATA_DIR_VAR: args.data_dir} if args.data_dir else None)
    cfg = load_run_config(args.config)
    cases_path = Path(args.cases) if args.cases else get_artifacts_dir() / "cases.parquet"
    if not cases_path.is_file():
        raise ConfigError(f"{cases_path} not found; run `leaksense build-index` first")
    cases = pd.read_parquet(cases_path)
    sha = manifest(data_dir)["sha256"]
    cache = PressureCache(data_dir, get_artifacts_dir(), sha)
    return data_dir, cfg, cases, cache, sha


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cmd_baseline_study(args: argparse.Namespace) -> int:
    from leaksense.data import baseline_study as bs
    from leaksense.data.splits import non_test_scenarios
    from leaksense.provenance import provenance

    t0 = time.perf_counter()
    data_dir, cfg, cases, cache, sha = _setup(args)
    splits = json.loads(Path(args.splits).read_text())
    test_ids = set(splits["test"])
    allowed = sorted(non_test_scenarios(splits) & set(cases["scenario_id"]))
    chosen = [s for s in cfg.select_scenarios(allowed) if s not in test_ids]
    sources = [1, *chosen]
    work = cases[cases["scenario_id"].isin(chosen)]
    say = (lambda *a: None) if args.quiet else print

    say(f"sources: Scenario-1 + {len(chosen)} non-test scenarios (test fold excluded)")
    stretches = bs.leak_free_stretches(cases, sources, cache, test_ids)
    comp = bs.method_comparison(stretches, cache, progress=not args.quiet)
    decision = bs.choose(comp)
    long_label = decision["method_by_stratum"][">28d"]
    by_label = {m.label: m for m in bs.METHODS}
    lag_methods = sorted(
        {
            by_label[long_label],
            *(m for m in bs.METHODS if m.estimator == "mean" and m.baseline_days in (14, 28)),
        },
        key=lambda m: m.label,
    )
    say("  lag study ...")
    lag = bs.lag_study(stretches, cache, lag_methods, decision["window"])
    max_lag = bs.choose_lag(lag, long_label)
    feat = bs.to_feature_config(cfg.features, decision, long_label, max_lag)
    noise = comp[
        (comp["stratum"] == ">28d") & (comp["method"] == long_label) & (comp["W"] == feat.window)
    ]["median_rms_m"].iloc[0]
    step_noise = comp[
        (comp["stratum"] == ">28d") & (comp["method"] == long_label) & (comp["W"] == 1)
    ]["median_rms_m"].iloc[0]
    say("  start-1 and signal checks ...")
    start = bs.start_check(work, cache, feat)
    signal = bs.signal_check(work, cache, feat, float(noise))
    elapsed = time.perf_counter() - t0

    report = {
        **provenance(None, config=cfg.to_dict(), seed=cfg.seed, manifest_sha256=sha),
        "elapsed_seconds": round(elapsed, 1),
        "n_source_scenarios": len(sources),
        "excluded_test_scenarios": len(test_ids),
        "rules": {
            "method": f"lowest median window RMS at W=48; within {bs.TIE_TOLERANCE:.0%}, "
            "the simpler slot/estimator, then the lower noise",
            "window": f"smallest W with median RMS <= {bs.NOISE_TARGET_M} m",
            "stride": "W/2",
            "max_lag_days": "largest lag up to which median |bias| < the lag 0-7 d RMS",
        },
        "decision": {**decision, "max_lag_days": max_lag},
        "chosen_features": feat.__dict__,
        "leak_free_noise_m": {"per_step": float(step_noise), f"window_{feat.window}": float(noise)},
        "start_check": start,
        "methods": comp.to_dict(orient="records"),
        "lag": lag.to_dict(orient="records"),
        "signal": signal.to_dict(orient="records"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    out.with_suffix(".md").write_text(
        _render_md(report, comp, lag, signal, feat, cfg.name), encoding="utf-8"
    )

    say(f"method by history: {decision['method_by_stratum']}")
    say(f"window {feat.window} steps, stride {feat.stride}, max lag {max_lag} days")
    say(f"leak-free noise: per step {step_noise:.3f} m, window {noise:.3f} m")
    say(f"start check (median mean |dH|, m): {start}")
    say(f"done in {elapsed:.0f}s -> {out.with_suffix('.md')}, {out.with_suffix('.json')}")
    return EXIT_OK


def _render_md(report, comp, lag, signal, feat, cfg_name) -> str:
    d = report["decision"]
    long_label = d["method_by_stratum"][">28d"]
    w48 = comp[comp["W"] == 48].pivot(index="method", columns="stratum", values="median_rms_m")
    w48 = w48.sort_values(">28d").reset_index()
    vs_w = comp[(comp["stratum"] == ">28d") & (comp["method"] == long_label)][
        ["W", "n_pseudo", "median_rms_m", "p95_rms_m"]
    ]
    g = report["git"]
    lines = [
        "# Baseline choice (M4)",
        "",
        f"- **Generated by:** `leaksense baseline-study --config` ({cfg_name}), commit "
        f"`{(g['commit'] or '?')[:10]}`{' (dirty)' if g['dirty'] else ''}, "
        f"data manifest `{report['data_manifest_sha256'][:12]}`, seed {report['seed']}.",
        f"- **Sources:** leak-free rows only — Scenario-1 plus the pre-leak stretch of "
        f"{report['n_source_scenarios'] - 1} non-test scenarios. The "
        f"{report['excluded_test_scenarios']}"
        " locked-test scenarios were not loaded.",
        "- **Method:** pseudo-cases split a leak-free stretch at a pseudo-start s. The baseline is "
        "the N days before s; ΔH is evaluated over the 7 days after s. The true ΔH is 0, so the "
        "RMS of window-mean ΔH is the baseline error (noise + bias).",
        "",
        "## Decision",
        "",
        f"- Slot method / baseline days / estimator: **{long_label}** "
        f"(history 14-28 d: {d['method_by_stratum'].get('history 14-28d')}).",
        f"- Window **{d['window']}** steps, stride **{d['stride']}**; max lag "
        f"**{d['max_lag_days']} days**.",
        "- Leak-free noise at that method: "
        f"{report['leak_free_noise_m']['per_step']:.3f} m per step, "
        f"{report['leak_free_noise_m'][f'window_{feat.window}']:.3f} m per window.",
        "- Rules: " + "; ".join(f"{k}: {v}" for k, v in report["rules"].items()) + ".",
        "",
        "## Methods: median window RMS (m) at W = 48",
        "",
        _md_table(w48),
        "",
        f"## Noise vs window length ({long_label}, history > 28 d)",
        "",
        _md_table(vs_w.reset_index(drop=True)),
        "",
        f"## Drift vs lag (W = {d['window']})",
        "",
        _md_table(lag[["method", "lag_days", "n", "median_abs_bias_m", "median_rms_m"]]),
        "",
        "## Check: no leak signal before Leak Start",
        "",
        "Median over big-leak pure cases (pre baseline) of mean |ΔH| over junctions, in m. The "
        "baseline here ends at start-5, so every column is out of sample:",
        "",
        "| " + " | ".join(report["start_check"]) + " |",
        "|" + "---|" * len(report["start_check"]),
        "| "
        + " | ".join(
            f"{v:.3f}" if isinstance(v, float) else str(v) for v in report["start_check"].values()
        )
        + " |",
        "",
        "## Signal: largest |case-mean ΔH| over junctions (non-test cases)",
        "",
        _md_table(signal),
        "",
    ]
    return "\n".join(lines)


def cmd_build_features(args: argparse.Namespace) -> int:
    from leaksense.data.features import build_features, features_hash, save_features
    from leaksense.provenance import provenance

    t0 = time.perf_counter()
    data_dir, cfg, cases, cache, sha = _setup(args)
    keep = cfg.select_scenarios(sorted(cases["scenario_id"]))
    sub = cases[cases["scenario_id"].isin(keep)]
    fs = build_features(sub, cache, cfg.features, progress=not args.quiet)
    elapsed = time.perf_counter() - t0
    out_dir = get_artifacts_dir() / "features" / cfg.hash()
    meta = {
        **provenance(None, config=cfg.to_dict(), seed=cfg.seed, manifest_sha256=sha),
        "elapsed_seconds": round(elapsed, 1),
        "features_sha256_12": features_hash(fs),
        "n_cases": len(fs.cases),
        "n_windows": int(len(fs.x)),
    }
    save_features(fs, out_dir, meta)
    cs = fs.cases
    print(
        f"config {cfg.name} ({cfg.hash()}): {len(cs)} cases from {len(keep)} scenarios, "
        f"{len(fs.x)} windows"
    )
    print(
        f"features: slot={cfg.features.slot} baseline_days={cfg.features.baseline_days} "
        f"estimator={cfg.features.estimator} window={cfg.features.window} "
        f"stride={cfg.features.stride} max_lag_days={cfg.features.max_lag_days}"
    )
    print(
        f"windows per case: min {cs['n_windows'].min()} median {cs['n_windows'].median():.0f} "
        f"max {cs['n_windows'].max()}; short-window cases: "
        f"{cs.loc[cs['short_window'], 'case_id'].tolist()}"
    )
    print(f"weekday slots filled from time of day: {int(cs['n_fallback_slots'].sum())}")
    q = cs["snr"].quantile([0.1, 0.5, 0.9]).round(1).tolist()
    print(
        f"snr p10/p50/p90: {q}  (by type: "
        f"{cs.groupby('leak_type')['snr'].median().round(1).to_dict()})"
    )
    print(f"done in {elapsed:.0f}s -> {out_dir}")
    return EXIT_OK


def cmd_show_case(args: argparse.Namespace) -> int:
    from leaksense.data.features import case_delta_h, case_windows, slot_index
    from leaksense.network import load_nominal_network

    data_dir, cfg, cases, cache, _ = _setup(args)
    key = args.case
    if key.isdigit():
        row = cases[cases["scenario_id"] == int(key)]
    else:
        row = cases[cases["case_id"] == key]
    if row.empty:
        raise ConfigError(f"no case {key!r} in the index")
    case = row.iloc[0]
    sid = int(case["scenario_id"])
    c = cache.contract
    cd = case_delta_h(
        cache.get(sid),
        case,
        cfg.features,
        cache.labels(sid),
        slot_index(c, "tod"),
        slot_index(c, "tod_weekday"),
    )
    win = case_windows(cd.dh, case, cfg.features)
    mean = win.x.mean(axis=0)
    net = load_nominal_network(data_dir)
    pipe = net.pipes.set_index("id").loc[case["pipe"]]
    ends = {pipe["start"], pipe["end"]}
    print(
        f"{case['case_id']}: pipe {case['pipe']} ({pipe['start']}->{pipe['end']}), "
        f"{case['leak_type']}, {case['leak_flow_lps']:.0f} L/s, {case['n_steps']} steps, "
        f"baseline {case['baseline_kind']} ({len(cd.rows_b)} rows), {len(win.x)} windows"
    )
    print("node  mean dH (m)")
    for j, v in zip(c.junctions, mean, strict=True):
        mark = "  <- pipe end" if j in ends else ""
        print(f"{j:>4}  {v:8.2f}{mark}")
    others = [abs(v) for j, v in zip(c.junctions, mean, strict=True) if j not in ends]
    print(f"largest |dH| away from the pipe ends: {max(others):.2f} m")
    print(
        f"mean dH over windows; min {np.min(mean):.2f} m "
        f"at node {c.junctions[int(np.argmin(mean))]}"
    )
    return EXIT_OK
