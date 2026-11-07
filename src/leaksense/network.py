"""The nominal network graph, built once from ``$LEAKSENSE_DATA_DIR/Hanoi.inp``.

Topology is identical in every scenario (D9), so one graph serves every case. Parameters are
the *nominal* ones: an operator only knows the nominal network, and each scenario's own
perturbed values are ground truth that never reaches a model.

Units (D5, ``UNITS LPS``): length m, diameter mm, Hazen-Williams roughness, flow L/s. The
Hazen-Williams resistance is computed in SI, so it takes diameter in m and flow in m^3/s.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from leaksense.data.contract import DataContract
from leaksense.data.inp import InpNetwork, merge_split_pipes, read_network

# SI Hazen-Williams: h [m] = 10.667 * L / (C^1.852 * D^4.871) * Q^1.852, D in m, Q in m^3/s.
HW_COEFF_SI = 10.667
HW_FLOW_EXP = 1.852
HW_DIAM_EXP = 4.871

EDGE_FEATURES = ("length_m", "diameter_m", "roughness", "resistance")


def hazen_williams_resistance(length_m: float, diameter_mm: float, roughness: float) -> float:
    """Resistance r so that headloss = r * |Q|^1.852 with Q in m^3/s (diameter given in mm)."""
    d_m = diameter_mm / 1000.0
    return HW_COEFF_SI * length_m / (roughness**HW_FLOW_EXP * d_m**HW_DIAM_EXP)


def headloss_m(resistance: float, flow_lps: float) -> float:
    """Headloss in m for a flow in L/s (converted to m^3/s)."""
    return resistance * (abs(flow_lps) / 1000.0) ** HW_FLOW_EXP


def _id_key(x: str) -> tuple[int, str]:
    m = re.match(r"\d+", x)
    return (int(m.group()) if m else 10**9, x)


def _all_pairs_shortest(weights: np.ndarray) -> np.ndarray:
    """Floyd-Warshall on a dense matrix (inf = no edge). Fine for a few dozen nodes."""
    d = weights.astype(float).copy()
    np.fill_diagonal(d, 0.0)
    for k in range(len(d)):
        d = np.minimum(d, d[:, k : k + 1] + d[k : k + 1, :])
    return d


@dataclass(frozen=True)
class Network:
    """Node table, pipe table and the derived graphs, all indexed by position.

    - ``nodes``: id, kind (junction/reservoir), elevation, base_demand_lps, x, y, sensor.
    - ``pipes``: id, start, end, u, v (node positions), length_m, diameter_mm, roughness,
      resistance, leak_eligible.
    - ``edge_index`` (2, 2P): each pipe in both directions; column e and e + P are pipe e.
    - ``edge_attr`` (2P, 4): ``EDGE_FEATURES`` for each directed edge.
    - ``pipe_hops`` (P, P): hop distance on the pipe line-graph (adjacent pipes = 1).
    - ``node_dist_m`` (N, N): shortest path along pipes, weighted by nominal length.
    - ``node_hops`` (N, N): unweighted hop count between nodes (inf if disconnected).
    """

    nodes: pd.DataFrame
    pipes: pd.DataFrame
    edge_index: np.ndarray
    edge_attr: np.ndarray
    pipe_hops: np.ndarray
    node_dist_m: np.ndarray
    node_hops: np.ndarray

    @property
    def node_ids(self) -> list[str]:
        return list(self.nodes["id"])

    @property
    def pipe_ids(self) -> list[str]:
        return list(self.pipes["id"])

    @property
    def node_index(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.nodes["id"])}

    @property
    def pipe_index(self) -> dict[str, int]:
        return {p: i for i, p in enumerate(self.pipes["id"])}

    @property
    def sensor_ids(self) -> list[str]:
        return list(self.nodes.loc[self.nodes["sensor"], "id"])

    @property
    def leak_pipe_ids(self) -> list[str]:
        """The classes: leak-eligible pipes in id order (pipes 2-34 on Hanoi)."""
        return list(self.pipes.loc[self.pipes["leak_eligible"], "id"])

    @property
    def is_connected(self) -> bool:
        return bool(np.isfinite(self.node_dist_m).all())

    @property
    def n_loops(self) -> int:
        """Independent cycles of a connected graph: P - N + 1."""
        return len(self.pipes) - len(self.nodes) + 1

    def summary(self) -> dict:
        p, n = self.pipes, self.nodes
        degree = np.bincount(np.concatenate([p["u"], p["v"]]), minlength=len(n))
        finite = self.node_dist_m[np.isfinite(self.node_dist_m)]
        return {
            "n_nodes": len(n),
            "n_junctions": int((n["kind"] == "junction").sum()),
            "reservoirs": list(n.loc[n["kind"] == "reservoir", "id"]),
            "n_pipes": len(p),
            "n_leak_eligible_pipes": int(p["leak_eligible"].sum()),
            "never_leaking_pipes": list(p.loc[~p["leak_eligible"], "id"]),
            "n_sensor_candidates": int(n["sensor"].sum()),
            "connected": self.is_connected,
            "n_loops": self.n_loops,
            "degree_counts": {
                int(k): int(v) for k, v in zip(*np.unique(degree, return_counts=True), strict=True)
            },
            "total_length_m": float(p["length_m"].sum()),
            "length_m": [float(p["length_m"].min()), float(p["length_m"].max())],
            "diameters_mm": sorted({float(d) for d in p["diameter_mm"]}),
            "roughness": sorted({float(c) for c in p["roughness"]}),
            "resistance": [float(p["resistance"].min()), float(p["resistance"].max())],
            "max_pipe_hops": int(self.pipe_hops.max()),
            "max_node_dist_m": float(finite.max()) if finite.size else 0.0,
            "total_base_demand_lps": float(n["base_demand_lps"].sum()),
        }


def build_network(net: InpNetwork, contract: DataContract | None = None) -> Network:
    """Build the graph from a parsed .inp (split pipes, if any, are merged back first)."""
    c = contract or DataContract()
    if net.tanks or net.pumps or net.valves:
        raise ValueError("network has tanks, pumps or valves; only pipes are supported")
    merged = merge_split_pipes(net)

    junctions = {j: v for j, v in net.junctions.items() if not j.startswith("leak")}
    node_ids = sorted([*junctions, *net.reservoirs], key=_id_key)
    if set(net.reservoirs) != {c.reservoir}:
        raise ValueError(f"expected reservoir {c.reservoir}, found {sorted(net.reservoirs)}")
    missing = [i for i in node_ids if i not in net.coords]
    if missing:
        raise ValueError(f"nodes without coordinates: {missing}")
    sensors = set(c.junctions)
    nodes = pd.DataFrame(
        {
            "id": node_ids,
            "kind": ["reservoir" if i in net.reservoirs else "junction" for i in node_ids],
            "elevation": [
                junctions[i][0] if i in junctions else net.reservoirs[i] for i in node_ids
            ],
            "base_demand_lps": [junctions[i][1] if i in junctions else 0.0 for i in node_ids],
            "x": [net.coords[i][0] for i in node_ids],
            "y": [net.coords[i][1] for i in node_ids],
            "sensor": [i in sensors for i in node_ids],
        }
    )
    index = {n: i for i, n in enumerate(node_ids)}

    rows = []
    for pid in sorted(merged, key=_id_key):
        m = merged[pid]
        if m.start not in index or m.end not in index:
            raise ValueError(f"pipe {pid} connects unknown node(s) {m.start}->{m.end}")
        length, diam, rough = m.length, m.diameters[0], m.roughnesses[0]
        rows.append(
            {
                "id": pid,
                "start": m.start,
                "end": m.end,
                "u": index[m.start],
                "v": index[m.end],
                "length_m": length,
                "diameter_mm": diam,
                "roughness": rough,
                "resistance": hazen_williams_resistance(length, diam, rough),
                "leak_eligible": pid not in c.never_leaking_pipes,
            }
        )
    pipes = pd.DataFrame(rows)

    u, v = pipes["u"].to_numpy(), pipes["v"].to_numpy()
    edge_index = np.stack([np.concatenate([u, v]), np.concatenate([v, u])]).astype(np.int64)
    feats = np.column_stack(
        [pipes["length_m"], pipes["diameter_mm"] / 1000.0, pipes["roughness"], pipes["resistance"]]
    )
    edge_attr = np.concatenate([feats, feats]).astype(np.float64)

    n = len(nodes)
    w = np.full((n, n), np.inf)
    for a, b, length in zip(u, v, pipes["length_m"], strict=True):
        w[a, b] = w[b, a] = min(w[a, b], length)
    node_dist = _all_pairs_shortest(w)
    node_hops = _all_pairs_shortest(np.where(np.isfinite(w), 1.0, np.inf))

    ends = np.stack([u, v], axis=1)
    share = (ends[:, None, :, None] == ends[None, :, None, :]).any(axis=(2, 3))
    lw = np.where(share, 1.0, np.inf)
    hops = _all_pairs_shortest(lw)
    if np.isfinite(hops).all():
        hops = hops.astype(np.int64)

    return Network(nodes, pipes, edge_index, edge_attr, hops, node_dist, node_hops)


def load_nominal_network(
    data_dir: str | os.PathLike[str], contract: DataContract | None = None
) -> Network:
    path = Path(data_dir) / "Hanoi.inp"
    return build_network(read_network(path), contract)
