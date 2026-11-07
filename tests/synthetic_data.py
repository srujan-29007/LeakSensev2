"""Generate a tiny synthetic dataset in the exact LeakDB Hanoi file format.

Mirrors the verified data contract (SPEC.md section 8): same file names, headers, column
order, leak-split pipes (X, Xl, Xll), key/value info files and timestamp format, but with a
5-pipe network, 4 scenarios and 2 days at 30-minute steps. No real data is copied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from leaksense.data.contract import DataContract

RESERVOIR = "1"
JUNCTIONS = ("2", "3", "4", "5")
# id: (start, end, length m, diameter mm, roughness)
NOMINAL_PIPES = {
    "1": ("1", "2", 100.0, 1016.0, 130.0),
    "2": ("2", "3", 1000.0, 800.0, 130.0),
    "3": ("3", "4", 800.0, 600.0, 130.0),
    "4": ("4", "5", 600.0, 500.0, 130.0),
    "5": ("2", "5", 1200.0, 700.0, 130.0),
}
COORDS = {
    "1": (0.0, 0.0),
    "2": (100.0, 0.0),
    "3": (100.0, 300.0),
    "4": (300.0, 300.0),
    "5": (300.0, 0.0),
}
N_ROWS = 96
FIRST = "2024-01-01 00:00:00"


@dataclass
class LeakSpec:
    pipe: str
    start_row: int
    end_row: int  # inclusive
    kind: str = "small"
    fraction: float = 0.5


@dataclass
class ScenarioSpec:
    leaks: list[LeakSpec] = field(default_factory=list)


DEFAULT_SPECS: dict[int, ScenarioSpec] = {
    1: ScenarioSpec(),
    2: ScenarioSpec([LeakSpec("3", 40, 70, "small", 0.25)]),
    3: ScenarioSpec([LeakSpec("2", 20, 80, "big", 0.6), LeakSpec("4", 50, 90, "small", 0.3)]),
    4: ScenarioSpec([LeakSpec("4", 10, 30, "big", 0.2), LeakSpec("4", 60, 95, "small", 0.7)]),
}


def contract_for(specs: dict[int, ScenarioSpec]) -> DataContract:
    counts: dict[int, int] = {}
    for spec in specs.values():
        counts[len(spec.leaks)] = counts.get(len(spec.leaks), 0) + 1
    ts = timestamps()
    return DataContract(
        n_scenarios=len(specs),
        label_counts=tuple(sorted(counts.items())),
        n_rows=N_ROWS,
        first_timestamp=ts[0],
        last_timestamp=ts[-1],
        step_minutes=30,
        duration_text="48 hours",
        time_step_text="30.0 min",
        reservoir=RESERVOIR,
        junctions=JUNCTIONS,
        pipes=tuple(NOMINAL_PIPES),
        never_leaking_pipes=("1",),
        units="LPS",
        param_tolerance=0.20,
    )


def timestamps() -> list[str]:
    idx = pd.date_range(FIRST, periods=N_ROWS, freq="30min")
    return list(idx.strftime("%Y-%m-%d %H:%M:%S"))


def _fmt(x: float) -> str:
    return f"{x:.5f}"


def _write_inp(path: Path, pipes: dict, junctions: dict, coords: dict, patterns: bool) -> None:
    lines = ["[TITLE]", "synthetic", "", "[JUNCTIONS]", ";ID Elev Demand Pattern"]
    for jid, (elev, dem) in junctions.items():
        pat = f" P_{jid}" if patterns and not jid.startswith("leak") else ""
        lines.append(f" {jid} {elev} {dem}{pat} ;")
    lines += ["", "[RESERVOIRS]", ";ID Head Pattern", f" {RESERVOIR} 100 ;", "", "[TANKS]", ""]
    lines += ["[PIPES]", ";ID Node1 Node2 Length Diameter Roughness MinorLoss Status"]
    for pid, (a, b, length, diam, rough) in pipes.items():
        lines.append(f" {pid} {a} {b} {length} {diam} {rough} 0 Open ;")
    lines += ["", "[PUMPS]", "", "[VALVES]", "", "[DEMANDS]", "", "[STATUS]", "", "[PATTERNS]"]
    if patterns:
        # Long enough that read_inp_sections takes the head/tail path with a small tail_bytes.
        for jid in junctions:
            if not jid.startswith("leak"):
                for _ in range(40):
                    lines.append(f" P_{jid} " + " ".join(["1.0"] * 6))
    lines += ["", "[CURVES]", "", "[CONTROLS]", "", "[TIMES]"]
    if patterns:
        lines += [
            " DURATION 48:00:00",
            " HYDRAULIC TIMESTEP 00:30:00",
            " PATTERN TIMESTEP 00:30:00",
        ]
    else:
        lines += [" DURATION 00:00:00", " HYDRAULIC TIMESTEP 01:00:00"]
    lines += [
        "",
        "[OPTIONS]",
        " UNITS LPS",
        " HEADLOSS H-W",
        "",
        "[COORDINATES]",
        ";Node X-Coord Y-Coord",
    ]
    for nid, (x, y) in coords.items():
        lines.append(f" {nid} {x:.9f} {y:.9f}")
    lines += ["", "[VERTICES]", "", "[END]", ""]
    path.write_text("\n".join(lines))


def _kv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("\n".join(["Description , Value"] + [f"{k} , {v}" for k, v in rows]) + "\n")


def _wide_csv(path: Path, ts: list[str], columns: list[str], values: np.ndarray) -> None:
    lines = ["," + ",".join(columns)]
    for t, row in zip(ts, values, strict=True):
        lines.append(t + "," + ",".join(_fmt(v) for v in row))
    path.write_text("\n".join(lines) + "\n")


def make_dataset(root: Path, specs: dict[int, ScenarioSpec] | None = None, seed: int = 0) -> Path:
    """Write a synthetic HanoiOK folder under ``root`` and return its path."""
    specs = DEFAULT_SPECS if specs is None else specs
    rng = np.random.default_rng(seed)
    ts = timestamps()
    data = root / "HanoiOK"
    data.mkdir(parents=True)

    nominal_junctions = {j: (30.0, 50.0) for j in JUNCTIONS}
    _write_inp(data / "Hanoi.inp", NOMINAL_PIPES, nominal_junctions, COORDS, patterns=False)
    (data / "Labels.csv").write_text(
        "Scenario,Label\n" + "".join(f"{s},{len(sp.leaks)}\n" for s, sp in sorted(specs.items()))
    )

    for sid, spec in sorted(specs.items()):
        d = data / f"Scenario-{sid}"
        d.mkdir()
        # Per-scenario perturbed parameters (within +-10%).
        pipes = {}
        for pid, (a, b, length, diam, rough) in NOMINAL_PIPES.items():
            f = 1 + rng.uniform(-0.1, 0.1, size=3)
            pipes[pid] = (a, b, length * f[0], diam * f[1], rough * f[2])
        junctions = {j: (30.0, 50.0 * (1 + rng.uniform(-0.1, 0.1))) for j in JUNCTIONS}
        coords = dict(COORDS)
        extra_links: list[str] = []
        leak_ids = [f"leak_node{k}" for k in range(len(spec.leaks))]
        # Split pipes; a second leak on the same pipe extends the chain (X, Xl, Xll).
        for lid, leak in zip(leak_ids, spec.leaks, strict=True):
            chain = [leak.pipe]
            while chain[-1] + "l" in pipes:
                chain.append(chain[-1] + "l")
            last = chain[-1]
            a, b, length, diam, rough = pipes[last]
            piece = last + "l"
            pipes[last] = (a, lid, length * leak.fraction, diam, rough)
            pipes[piece] = (lid, b, length * (1 - leak.fraction), diam, rough)
            extra_links.append(piece)
            junctions[lid] = (30.0, 0.0)
            (xa, ya), (xb, yb) = coords[a], coords[b]
            coords[lid] = (xa + leak.fraction * (xb - xa), ya + leak.fraction * (yb - ya))
        _write_inp(d / f"Hanoi_Scenario-{sid}.inp", pipes, junctions, coords, patterns=True)
        _kv(
            d / f"Scenario-{sid}_info.csv",
            [
                ("Network_Name", "Hanoi"),
                ("Duration", "48 hours"),
                ("Time_Step", "30.0 min"),
                ("Uncertainty_Topology_(%)", "NO"),
                ("Uncertainty_Length_(%)", "10.0"),
                ("Uncertainty_Diameter_(%)", "10.0"),
                ("Uncertainty_Roughness_(%)", "10.0"),
            ],
        )

        node_cols = list(JUNCTIONS) + leak_ids + [RESERVOIR]
        active = np.zeros((N_ROWS, len(spec.leaks)), dtype=bool)
        flows = np.zeros((N_ROWS, len(spec.leaks)))
        for k, leak in enumerate(spec.leaks):
            active[leak.start_row : leak.end_row + 1, k] = True
            level = 650.0 if leak.kind == "big" else 110.0
            flows[:, k] = np.where(active[:, k], level + rng.normal(0, 5, N_ROWS), 0.0)

        pressure = 60 + rng.normal(0, 0.3, (N_ROWS, len(node_cols)))
        pressure -= flows.sum(axis=1, keepdims=True) / 100.0
        pressure[:, -1] = 0.0  # reservoir
        _wide_csv(d / "Node_pressures.csv", ts, node_cols, pressure)

        demand = np.abs(rng.normal(40, 10, (N_ROWS, len(node_cols))))
        for k in range(len(spec.leaks)):
            demand[:, len(JUNCTIONS) + k] = flows[:, k]
        demand[:, -1] = -demand[:, :-1].sum(axis=1)
        _wide_csv(d / "Node_demands.csv", ts, node_cols, demand)

        link_cols = list(NOMINAL_PIPES) + extra_links
        _wide_csv(
            d / "Link_flows.csv", ts, link_cols, rng.normal(100, 30, (N_ROWS, len(link_cols)))
        )

        label = active.any(axis=1).astype(float)
        (d / "Labels.csv").write_text(
            "Timestamp,Label\n" + "".join(f"{t},{v:.1f}\n" for t, v in zip(ts, label, strict=True))
        )

        if spec.leaks:
            (d / "Leaks").mkdir()
            for k, (lid, leak) in enumerate(zip(leak_ids, spec.leaks, strict=True)):
                _kv(
                    d / "Leaks" / f"Leak_{lid}_info.csv",
                    [
                        ("Leak Node", lid),
                        ("Leak Area", "0.07" if leak.kind == "big" else "0.003"),
                        ("Leak Diameter", "0.3" if leak.kind == "big" else "0.06"),
                        ("Leak Type", leak.kind),
                        ("Leak Start", ts[leak.start_row]),
                        ("Leak End", ts[leak.end_row]),
                    ],
                )
                (d / "Leaks" / f"Leak_{lid}_demand.csv").write_text(
                    f",{lid}\n"
                    + "".join(f"{t},{_fmt(v)}\n" for t, v in zip(ts, flows[:, k], strict=True))
                )
    return data
