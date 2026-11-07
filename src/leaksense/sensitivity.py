"""Sensitivity matrix S on the nominal network (PLAN.md M7).

``S[j, p, size, level]`` is the pressure change (m) at junction j when a fixed leak demand of
``size`` is placed at the midpoint of pipe p, with every nominal base demand scaled by the
demand level α. Only the nominal ``Hanoi.inp`` is simulated (no scenario .inp, D9).

Inputs and their no-oracle status:
- α: quantiles of the total junction demand per timestep (junction columns ``2``..``32`` of
  ``Node_demands.csv`` only; never the reservoir column ``1``, which carries the leak flow, and
  never ``leak_node*``), over non-test scenarios, divided by the nominal total. One global scalar
  per level, never a per-case input.
- Leak sizes: median leak flow per leak type over non-test cases (aggregates only).

Leaks are plain junction demands, run with ``EpanetSimulator``. WNTR's ``add_leak`` is not used:
EPANET ignores WNTR leak models. WNTR works in SI, so L/s are converted to m^3/s.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from leaksense.data.contract import DataContract

LEVELS = ("low", "mid", "high")
LEVEL_QUANTILES = (5, 50, 95)
SIZES = ("small", "big")
STABLE_COSINE = 0.99
LEAK_NODE = "SENS_LEAK"


# ----------------------------------------------------------------------------- inputs


def load_junction_demands(scenario_dir: Path, contract: DataContract) -> np.ndarray:
    """Junction demands (rows x 31, L/s) from ``Node_demands.csv``: junction columns only."""
    wanted = set(contract.junctions)
    path = Path(scenario_dir) / "Node_demands.csv"
    df = pd.read_csv(path, usecols=lambda c: c in wanted, dtype=np.float64, engine="c")
    cols = list(df.columns)
    if cols != list(contract.junctions) or contract.reservoir in cols:
        raise ValueError(f"{path}: expected exactly the junction columns, got {cols[:3]}...")
    return df.to_numpy()


def demand_levels(
    data_dir: Path, contract: DataContract, scenario_ids: list[int], nominal_total_lps: float
) -> dict:
    """α per level = quantile of total junction demand per timestep / nominal total."""
    totals = [
        load_junction_demands(Path(data_dir) / f"Scenario-{s}", contract).sum(axis=1)
        for s in scenario_ids
    ]
    t = np.concatenate(totals)
    q = {f"p{p}": float(np.percentile(t, p)) for p in LEVEL_QUANTILES}
    alpha = {
        lv: q[f"p{p}"] / nominal_total_lps for lv, p in zip(LEVELS, LEVEL_QUANTILES, strict=True)
    }
    return {
        "alpha": alpha,
        "total_demand_lps": q,
        "nominal_total_lps": nominal_total_lps,
        "n_scenarios": len(scenario_ids),
        "n_steps": int(len(t)),
    }


def leak_sizes(cases: pd.DataFrame, test_ids: set[int]) -> dict[str, float]:
    """Median leak flow (L/s) per leak type over non-test cases."""
    c = cases[~cases["scenario_id"].isin(test_ids)]
    med = c.groupby("leak_type")["leak_flow_lps"].median()
    missing = [s for s in SIZES if s not in med]
    if missing:
        raise ValueError(f"no non-test cases of leak type {missing}")
    return {s: float(med[s]) for s in SIZES}


# ----------------------------------------------------------------------------- simulation


def _run(wn) -> pd.Series:
    import wntr

    with tempfile.TemporaryDirectory() as d:
        res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=str(Path(d) / "run"))
    return res.node["pressure"].iloc[0]


def _scaled_network(inp_path: Path, alpha: float):
    import wntr

    wn = wntr.network.WaterNetworkModel(str(inp_path))
    for _, j in wn.junctions():
        for d in j.demand_timeseries_list:
            d.base_value = d.base_value * alpha
    return wn


def simulate(
    inp_path: Path,
    contract: DataContract,
    pipes: list[str],
    alphas: dict[str, float],
    sizes_lps: dict[str, float],
    progress: bool = False,
) -> dict:
    """Run 1 + len(pipes) * len(sizes) snapshots per level -> S and checks."""
    import wntr

    J = list(contract.junctions)
    S = np.zeros((len(J), len(pipes), len(SIZES), len(LEVELS)))
    p0_all, negatives, split_errors = {}, [], []
    for li, lv in enumerate(LEVELS):
        wn = _scaled_network(inp_path, alphas[lv])
        p0 = _run(wn)
        p0_all[lv] = p0[J].to_numpy(dtype=float)
        for pi, pid in enumerate(pipes):
            length = wn.get_link(pid).length
            w2 = wntr.morph.split_pipe(
                wn, pid, f"{pid}_SENS_B", LEAK_NODE, split_at_point=0.5, return_copy=True
            )
            pieces = w2.get_link(pid).length + w2.get_link(f"{pid}_SENS_B").length
            if abs(pieces - length) > 1e-6 * max(length, 1.0):
                split_errors.append(f"pipe {pid}: pieces {pieces} != {length}")
            leak = w2.get_node(LEAK_NODE)
            for si, sz in enumerate(SIZES):
                leak.demand_timeseries_list[0].base_value = sizes_lps[sz] / 1000.0  # L/s -> m^3/s
                p = _run(w2)
                S[:, pi, si, li] = p[J].to_numpy(dtype=float) - p0_all[lv]
                neg = [n for n in [*J, LEAK_NODE] if p[n] < 0]
                if neg:
                    negatives.append(
                        {
                            "level": lv,
                            "size": sz,
                            "pipe": pid,
                            "n_negative_nodes": len(neg),
                            "min_pressure_m": float(p[[*J, LEAK_NODE]].min()),
                        }
                    )
        if progress:
            print(
                f"  level {lv} (alpha {alphas[lv]:.3f}): {len(pipes) * len(SIZES) + 1} runs",
                flush=True,
            )
    if split_errors:
        raise AssertionError("; ".join(split_errors))
    if S.max() > 1e-6:
        j, p, s, lv = np.unravel_index(int(S.argmax()), S.shape)
        raise AssertionError(
            f"a leak raised pressure: +{S.max():.3g} m at junction {J[j]} for pipe {pipes[p]}"
        )
    return {"S": S, "p0": p0_all, "negatives": negatives}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def level_stability(S: np.ndarray, pipes: list[str]) -> dict:
    """Per pipe and size, the smallest cosine between the low/mid/high columns."""
    per = {}
    for pi, pid in enumerate(pipes):
        per[pid] = {
            sz: min(cosine(S[:, pi, si, a], S[:, pi, si, b]) for a, b in ((0, 1), (1, 2), (0, 2)))
            for si, sz in enumerate(SIZES)
        }
    worst = min(v for d in per.values() for v in d.values())
    return {
        "per_pipe_min_cosine": per,
        "min_cosine": worst,
        "use_mid_only": bool(worst >= STABLE_COSINE),
    }


def save(
    path: Path,
    S: np.ndarray,
    contract: DataContract,
    pipes: list[str],
    alphas: dict,
    sizes: dict,
    manifest_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        S=S,
        junctions=np.array(contract.junctions),
        pipes=np.array(pipes),
        levels=np.array(LEVELS),
        alpha=np.array([alphas[lv] for lv in LEVELS]),
        sizes=np.array(SIZES),
        sizes_lps=np.array([sizes[s] for s in SIZES]),
        manifest=np.array(manifest_sha256),
    )


def load(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def sensitivity_path(artifacts_dir: Path, manifest_sha256: str) -> Path:
    return Path(artifacts_dir) / "sensitivity" / manifest_sha256[:12] / "S.npz"
