"""CLI command for M7: ``leaksense build-sensitivity``."""

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
    s = sub.add_parser(
        "build-sensitivity", help="simulate the sensitivity matrix S on the nominal network"
    )
    s.add_argument("--config", default="configs/full.yaml", help="run config (scenario subset)")
    s.add_argument("--splits", default="splits/splits_v1.json", help="test fold is excluded")
    s.add_argument("--cases", help="case index (default: $LEAKSENSE_ARTIFACTS_DIR/cases.parquet)")
    s.add_argument("--out", default="reports/sensitivity", help="report path without suffix")
    s.add_argument("--data-dir", help=f"dataset HanoiOK folder (default: ${DATA_DIR_VAR})")
    s.add_argument("--contract", help="JSON file overriding the default data contract")
    s.add_argument("--quiet", action="store_true", help="no progress output")
    s.set_defaults(func=cmd_build_sensitivity)


def cmd_build_sensitivity(args: argparse.Namespace) -> int:
    from leaksense import sensitivity as sens
    from leaksense.data.contract import DataContract
    from leaksense.data.splits import non_test_scenarios
    from leaksense.data.validate import manifest
    from leaksense.network import load_nominal_network
    from leaksense.provenance import provenance
    from leaksense.runconfig import load_run_config

    t0 = time.perf_counter()
    say = (lambda *a: None) if args.quiet else print
    data_dir = get_data_dir({DATA_DIR_VAR: args.data_dir} if args.data_dir else None)
    contract = DataContract.from_json(args.contract) if args.contract else DataContract()
    cfg = load_run_config(args.config)
    splits = json.loads(Path(args.splits).read_text())
    test_ids = set(int(s) for s in splits["test"])
    cases_path = Path(args.cases) if args.cases else get_artifacts_dir() / "cases.parquet"
    if not cases_path.is_file():
        raise ConfigError(f"{cases_path} not found; run `leaksense build-index` first")
    cases = pd.read_parquet(cases_path)
    net = load_nominal_network(data_dir, contract)
    pipes = net.leak_pipe_ids
    nominal_total = float(net.nodes["base_demand_lps"].sum())
    sha = manifest(data_dir)["sha256"]

    allowed = sorted(non_test_scenarios(splits) & set(cases["scenario_id"]))
    sources = [1, *[s for s in cfg.select_scenarios(allowed) if s not in test_ids]]
    if test_ids & set(sources):
        raise AssertionError("demand levels must not use locked-test scenarios")
    say(f"demand levels from {len(sources)} scenarios (Scenario-1 + non-test), junction columns")
    levels = sens.demand_levels(data_dir, contract, sources, nominal_total)
    sizes = sens.leak_sizes(cases, test_ids)
    say(f"alpha: {levels['alpha']}  sizes (L/s): {sizes}")

    res = sens.simulate(
        data_dir / "Hanoi.inp", contract, pipes, levels["alpha"], sizes, progress=not args.quiet
    )
    S = res["S"]
    stab = sens.level_stability(S, pipes)
    out_s = sens.sensitivity_path(get_artifacts_dir(), sha)
    sens.save(out_s, S, contract, pipes, levels["alpha"], sizes, sha)
    elapsed = time.perf_counter() - t0

    J = list(contract.junctions)
    mid = sens.LEVELS.index("mid")
    largest = {}
    for pi, pid in enumerate(pipes):
        row = net.pipes.set_index("id").loc[pid]
        col = S[:, pi, sens.SIZES.index("big"), mid]
        j = int(np.argmin(col))
        largest[pid] = {
            "ends": [row["start"], row["end"]],
            "junction": J[j],
            "drop_m": float(col[j]),
            "small_drop_m": float(S[:, pi, 0, mid].min()),
        }
    report = {
        **provenance(None, config=cfg.to_dict(), seed=None, manifest_sha256=sha),
        "elapsed_seconds": round(elapsed, 1),
        "n_runs": len(sens.LEVELS) * (1 + len(pipes) * len(sens.SIZES)),
        "demand_levels": levels,
        "leak_sizes_lps": sizes,
        "shape": {
            "junctions": len(J),
            "pipes": len(pipes),
            "sizes": list(sens.SIZES),
            "levels": list(sens.LEVELS),
        },
        "negative_pressure_warnings": res["negatives"],
        "level_stability": stab,
        "largest_drop_mid_level": largest,
        "artifact": str(out_s.relative_to(get_artifacts_dir()).as_posix()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    out.with_suffix(".md").write_text(_render(report), encoding="utf-8")

    neg = res["negatives"]
    say(f"S: {S.shape} (junctions x pipes x sizes x levels), all <= 0: {bool(S.max() <= 1e-6)}")
    say(
        f"negative-pressure warnings: {len(neg)} runs "
        f"{sorted({(n['level'], n['size']) for n in neg})}"
    )
    say(
        f"cosine across levels: min {stab['min_cosine']:.4f} -> "
        f"{'mid level only' if stab['use_mid_only'] else 'levels differ; M10 may ablate them'}"
    )
    say(f"done in {elapsed:.1f}s -> {out_s}, {out.with_suffix('.md')}")
    return EXIT_OK


def _render(r: dict) -> str:
    g, lv, st = r["git"], r["demand_levels"], r["level_stability"]
    lines = [
        "# Sensitivity matrix S (M7)",
        "",
        f"- **Generated by:** `leaksense build-sensitivity`, commit `{(g['commit'] or '?')[:10]}`"
        f"{' (dirty)' if g['dirty'] else ''}, data manifest `{r['data_manifest_sha256'][:12]}`.",
        "- **What:** pressure change (m) at the 31 junctions for a fixed leak demand at the "
        "midpoint of each of the 33 eligible pipes, on the nominal `Hanoi.inp`, EpanetSimulator, "
        f"{r['n_runs']} snapshot runs in {r['elapsed_seconds']} s.",
        f"- **Demand levels:** every nominal base demand × α. α = quantile of the total junction "
        f"demand per timestep over {lv['n_scenarios']} scenarios (Scenario-1 + non-test; junction "
        f"columns only) ÷ the nominal total {lv['nominal_total_lps']:.1f} L/s.",
        "",
        "| level | quantile | total demand (L/s) | α |",
        "|---|---|---|---|",
    ]
    for name, q in zip(("low", "mid", "high"), ("p5", "p50", "p95"), strict=True):
        lines.append(
            f"| {name} | {q} | {lv['total_demand_lps'][q]:.1f} | {lv['alpha'][name]:.3f} |"
        )
    sz = r["leak_sizes_lps"]
    lines += [
        "",
        f"- **Leak sizes:** median leak flow of non-test cases — small {sz['small']:.1f} L/s, "
        f"big {sz['big']:.1f} L/s.",
        f"- **Stability across levels:** smallest cosine between the low/mid/high columns of any "
        f"pipe = {st['min_cosine']:.4f} -> "
        + (
            "**mid level only** for `sigmatch` and the M10 prior (all ≥ 0.99)."
            if st["use_mid_only"]
            else "levels differ (< 0.99); M10 may ablate them as inputs."
        ),
        f"- **Negative-pressure warnings:** {len(r['negative_pressure_warnings'])} runs.",
        "",
    ]
    if r["negative_pressure_warnings"]:
        lines += [
            "| level | size | pipe | negative nodes | min pressure (m) |",
            "|---|---|---|---|---|",
        ]
        for n in r["negative_pressure_warnings"]:
            lines.append(
                f"| {n['level']} | {n['size']} | {n['pipe']} | {n['n_negative_nodes']} | "
                f"{n['min_pressure_m']:.1f} |"
            )
        lines.append("")
    lines += [
        "## Largest drop per pipe (mid level)",
        "",
        "| pipe | ends | junction of largest drop (big) | big drop (m) | small drop (m) | "
        "min cosine low/mid/high (small, big) |",
        "|---|---|---|---|---|---|",
    ]
    for pid, d in r["largest_drop_mid_level"].items():
        c = st["per_pipe_min_cosine"][pid]
        lines.append(
            f"| {pid} | {d['ends'][0]}–{d['ends'][1]} | {d['junction']} | "
            f"{d['drop_m']:.1f} | {d['small_drop_m']:.2f} | "
            f"{c['small']:.4f}, {c['big']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)
