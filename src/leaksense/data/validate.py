"""``leaksense validate-data``: assert the data contract D1-D10 (SPEC.md section 8).

Read-only. The default (fast) mode reads directory listings, CSV header/first/last lines, the
small per-scenario Labels.csv and info files, and the .inp header and tail sections.
``deep=True`` also parses every Node_pressures.csv and leak demand file in full, and counts the
rows of Node_demands.csv / Link_flows.csv.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from leaksense import __version__
from leaksense.data.contract import DataContract
from leaksense.data.inp import (
    InpNetwork,
    MergedPipe,
    SplitPipeError,
    merge_split_pipes,
    read_network,
)

MAX_STORED_ERRORS = 50
SCENARIO_FILES = ("Node_pressures.csv", "Node_demands.csv", "Link_flows.csv", "Labels.csv")

CHECK_TITLES = {
    "D1": "Top-level layout: Hanoi.inp, Labels.csv, Scenario-1..N with expected label counts",
    "D2": "Per-scenario files present; Leaks/ present iff the scenario has leaks",
    "D3": "Row count, timestamps and time step identical everywhere",
    "D4": "CSV column names and order",
    "D5": "Units, reservoir pressure, no NaN / negative pressures",
    "D6": "Leak info files: header, keys, type, ISO start/end",
    "D7": "Leak demand non-zero exactly over [start, end]; Labels.csv = union of leaks",
    "D8": "Each leak splits pipe X into X, Xl(, Xll); label X in the leak-eligible pipes",
    "D9": "Topology and coordinates equal nominal; parameters within tolerance",
    "D10": "Coordinates vs pipe lengths (informational)",
}


@dataclass
class Check:
    id: str
    title: str
    informational: bool = False
    errors: list[str] = field(default_factory=list)
    n_errors: int = 0
    details: dict = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.n_errors += 1
        if len(self.errors) < MAX_STORED_ERRORS:
            self.errors.append(msg)

    @property
    def status(self) -> str:
        if self.informational:
            return "info"
        return "fail" if self.n_errors else "pass"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "n_errors": self.n_errors,
            "errors": self.errors,
            "details": self.details,
        }


@dataclass
class ValidationReport:
    checks: dict[str, Check]
    deep: bool
    contract: DataContract
    manifest: dict
    stats: dict
    leaks: list[dict]

    @property
    def ok(self) -> bool:
        return all(c.status != "fail" for c in self.checks.values())

    def to_dict(self) -> dict:
        return {
            "leaksense_version": __version__,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "deep": self.deep,
            "ok": self.ok,
            "manifest": self.manifest,
            "contract": self.contract.to_dict(),
            "checks": [c.to_dict() for c in self.checks.values()],
            "stats": self.stats,
        }


# ----------------------------------------------------------------------------- helpers


def manifest(data_dir: Path) -> dict:
    """Hash of every (relative path, size) pair: identifies the dataset version cheaply."""
    entries = []
    total = 0
    for root, _dirs, files in os.walk(data_dir):
        for name in files:
            p = Path(root) / name
            size = p.stat().st_size
            total += size
            entries.append(f"{p.relative_to(data_dir).as_posix()}\t{size}")
    entries.sort()
    digest = hashlib.sha256("\n".join(entries).encode()).hexdigest()
    return {"folder": data_dir.name, "n_files": len(entries), "total_bytes": total, "sha256": digest}


def read_kv(path: Path) -> tuple[list[str], list[tuple[str, str]]]:
    """Read a 'Description , Value' file -> (header tokens, [(key, value)]) with spaces stripped."""
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\r\n") for ln in f if ln.strip()]
    header = [t.strip() for t in lines[0].split(",")] if lines else []
    rows = []
    for ln in lines[1:]:
        key, _, value = ln.partition(",")
        rows.append((key.strip(), value.strip()))
    return header, rows


def first_lines(path: Path, n: int = 2) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(line.rstrip("\r\n"))
            if len(out) == n:
                break
    return out


def last_line(path: Path) -> str:
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 8192))
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    return lines[-1].decode("utf-8") if lines else ""


def count_rows(path: Path) -> int:
    """Number of non-empty lines minus the header, reading in binary chunks."""
    n = 0
    last = b"\n"
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":
        n += 1
    return n - 1


def expected_timestamps(c: DataContract) -> list[str]:
    idx = pd.date_range(c.first_timestamp, periods=c.n_rows, freq=f"{c.step_minutes}min")
    return list(idx.strftime("%Y-%m-%d %H:%M:%S"))


def rel_dev(value: float, nominal: float) -> float:
    return value / nominal - 1.0 if nominal else math.inf


def scenario_ids(data_dir: Path) -> tuple[list[int], list[str]]:
    ids, other = [], []
    for p in data_dir.iterdir():
        m = re.fullmatch(r"Scenario-(\d+)", p.name)
        if m and p.is_dir():
            ids.append(int(m.group(1)))
        elif p.name not in ("Hanoi.inp", "Labels.csv"):
            other.append(p.name)
    return sorted(ids), sorted(other)


# ----------------------------------------------------------------------------- main entry


def validate_dataset(
    data_dir: str | Path,
    contract: DataContract | None = None,
    deep: bool = False,
    progress: bool = False,
    inp_tail_bytes: int = 65_536,
) -> ValidationReport:
    data_dir = Path(data_dir)
    c = contract or DataContract()
    checks = {cid: Check(cid, title, informational=(cid == "D10")) for cid, title in CHECK_TITLES.items()}
    stats: dict = {}
    leaks: list[dict] = []

    ts_expected = expected_timestamps(c)
    ts_index = {t: i for i, t in enumerate(ts_expected)}
    if ts_expected[-1] != c.last_timestamp:
        checks["D3"].fail(
            f"contract inconsistent: {c.n_rows} rows from {c.first_timestamp} end at "
            f"{ts_expected[-1]}, not {c.last_timestamp}"
        )

    # ---- D1: top level
    d1 = checks["D1"]
    labels_top: dict[int, int] = {}
    for name in ("Hanoi.inp", "Labels.csv"):
        if not (data_dir / name).is_file():
            d1.fail(f"missing {name}")
    if (data_dir / "Labels.csv").is_file():
        with open(data_dir / "Labels.csv", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            if header != ["Scenario", "Label"]:
                d1.fail(f"Labels.csv header {header} != ['Scenario', 'Label']")
            for row in reader:
                if row:
                    labels_top[int(row[0])] = int(row[1])
    counts = Counter(labels_top.values())
    d1.details["label_counts"] = {str(k): v for k, v in sorted(counts.items())}
    if dict(counts) != c.label_count_map:
        d1.fail(f"label counts {dict(sorted(counts.items()))} != {c.label_count_map}")
    ids, other = scenario_ids(data_dir)
    d1.details.update(n_scenarios=len(ids), other_entries=other)
    if ids != list(range(1, c.n_scenarios + 1)):
        d1.fail(f"scenario folders are not exactly 1..{c.n_scenarios} (found {len(ids)})")
    if set(ids) != set(labels_top):
        d1.fail("scenario folders and Labels.csv scenario ids differ")

    # ---- nominal network (D5 units, D9 structure, D10 scale)
    nominal: InpNetwork | None = None
    if (data_dir / "Hanoi.inp").is_file():
        nominal = read_network(data_dir / "Hanoi.inp")
        _check_nominal(nominal, c, checks)

    # ---- per scenario
    exp_node_base = list(c.junctions)
    exp_link_base = list(c.pipes)
    ref_times: dict | None = None
    uncertainty = Counter()
    max_dev = defaultdict(float)
    base_demand_dev = 0.0
    zero_stats = {"scenarios_with_zero_junction_pressure": 0, "zero_cells": 0, "zero_rows": 0,
                  "zero_rows_outside_leaks": 0, "by_scenario": {}}
    leak_flow_medians: dict[str, list[float]] = defaultdict(list)

    for n, sid in enumerate(ids, 1):
        if progress and (n % 50 == 0 or n == len(ids)):
            print(f"  validated {n}/{len(ids)} scenarios", file=sys.stderr)
        d = data_dir / f"Scenario-{sid}"
        n_leaks = labels_top.get(sid, -1)
        tag = f"Scenario-{sid}"

        # D2 files
        d2 = checks["D2"]
        missing = [f for f in (*SCENARIO_FILES, f"Hanoi_Scenario-{sid}.inp", f"Scenario-{sid}_info.csv")
                   if not (d / f).is_file()]
        if missing:
            d2.fail(f"{tag}: missing {missing}")
            continue
        leaks_dir = d / "Leaks"
        info_files = sorted(leaks_dir.glob("Leak_*_info.csv")) if leaks_dir.is_dir() else []
        if n_leaks == 0 and leaks_dir.exists():
            d2.fail(f"{tag}: has Leaks/ but Labels.csv says 0 leaks")
        if n_leaks > 0 and not leaks_dir.is_dir():
            d2.fail(f"{tag}: Leaks/ missing")
        leak_ids = [f.name[len("Leak_"):-len("_info.csv")] for f in info_files]
        if leak_ids != [f"leak_node{k}" for k in range(max(n_leaks, 0))]:
            d2.fail(f"{tag}: leak files {leak_ids} do not match {n_leaks} leak(s)")
        for lid in leak_ids:
            if not (leaks_dir / f"Leak_{lid}_demand.csv").is_file():
                d2.fail(f"{tag}: missing Leak_{lid}_demand.csv")

        # Scenario info (D3 duration/step, D9 uncertainty)
        hdr, rows = read_kv(d / f"Scenario-{sid}_info.csv")
        info = dict(rows)
        if hdr != ["Description", "Value"]:
            checks["D3"].fail(f"{tag}: info header {hdr}")
        if info.get("Duration") != c.duration_text or info.get("Time_Step") != c.time_step_text:
            checks["D3"].fail(f"{tag}: Duration/Time_Step {info.get('Duration')}/{info.get('Time_Step')}")
        if info.get("Uncertainty_Topology_(%)") != "NO":
            checks["D9"].fail(f"{tag}: topology uncertainty {info.get('Uncertainty_Topology_(%)')}")
        unc = tuple(
            round(float(info.get(f"Uncertainty_{p}_(%)", "nan")), 6)
            for p in ("Length", "Diameter", "Roughness")
        )
        uncertainty[unc] += 1

        # D4 headers
        d4 = checks["D4"]
        exp_nodes = ["", *exp_node_base, *leak_ids, c.reservoir]
        heads = {f: first_lines(d / f, 2) for f in SCENARIO_FILES}
        for f in ("Node_pressures.csv", "Node_demands.csv"):
            cols = heads[f][0].split(",")
            if cols != exp_nodes:
                d4.fail(f"{tag}/{f}: columns {cols[:3]}...{cols[-3:]} != expected order")
        link_cols = heads["Link_flows.csv"][0].split(",")
        link_extra = link_cols[1 + len(exp_link_base):]
        if link_cols[: 1 + len(exp_link_base)] != ["", *exp_link_base]:
            d4.fail(f"{tag}/Link_flows.csv: base link columns differ")
        if len(link_extra) != len(leak_ids) or not all(re.fullmatch(r"\d+l+", x) for x in link_extra):
            d4.fail(f"{tag}/Link_flows.csv: extra columns {link_extra} for {len(leak_ids)} leak(s)")
        if heads["Labels.csv"][0] != "Timestamp,Label":
            d4.fail(f"{tag}/Labels.csv header {heads['Labels.csv'][0]!r}")

        # D3 timestamps: first/last row of the wide CSVs, full Labels.csv
        d3 = checks["D3"]
        for f in ("Node_pressures.csv", "Node_demands.csv", "Link_flows.csv"):
            first_ts = heads[f][1].split(",", 1)[0] if len(heads[f]) > 1 else None
            last_ts = last_line(d / f).split(",", 1)[0]
            if first_ts != c.first_timestamp or last_ts != c.last_timestamp:
                d3.fail(f"{tag}/{f}: spans {first_ts} .. {last_ts}")
        with open(d / "Labels.csv", encoding="utf-8") as f:
            next(f)
            pairs = [ln.rstrip("\r\n").split(",") for ln in f if ln.strip()]
        lab_ts = [p[0] for p in pairs]
        lab_val = np.array([float(p[1]) for p in pairs])
        if lab_ts != ts_expected:
            d3.fail(f"{tag}/Labels.csv: {len(lab_ts)} timestamps differ from the expected grid")
        if not set(np.unique(lab_val)) <= {0.0, 1.0}:
            checks["D7"].fail(f"{tag}/Labels.csv: values {sorted(set(np.unique(lab_val)))}")

        # D6 leak info
        d6 = checks["D6"]
        leak_windows: dict[str, tuple[int, int]] = {}
        leak_meta: dict[str, dict] = {}
        for lid in leak_ids:
            hdr, rows = read_kv(leaks_dir / f"Leak_{lid}_info.csv")
            keys = [k for k, _ in rows]
            info = dict(rows)
            if hdr != ["Description", "Value"]:
                d6.fail(f"{tag}/{lid}: header {hdr}")
            if tuple(keys) != c.leak_info_keys:
                d6.fail(f"{tag}/{lid}: keys {keys}")
                continue
            if info["Leak Node"] != lid:
                d6.fail(f"{tag}/{lid}: Leak Node {info['Leak Node']!r}")
            if info["Leak Type"] not in c.leak_types:
                d6.fail(f"{tag}/{lid}: Leak Type {info['Leak Type']!r}")
            try:
                area, diam = float(info["Leak Area"]), float(info["Leak Diameter"])
                start = datetime.fromisoformat(info["Leak Start"]).strftime("%Y-%m-%d %H:%M:%S")
                end = datetime.fromisoformat(info["Leak End"]).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError as e:
                d6.fail(f"{tag}/{lid}: unparsable value ({e})")
                continue
            if not (area > 0 and diam > 0):
                d6.fail(f"{tag}/{lid}: non-positive area/diameter")
            if start not in ts_index or end not in ts_index or ts_index[start] > ts_index[end]:
                d6.fail(f"{tag}/{lid}: start/end {start}..{end} not on the time grid or reversed")
                continue
            leak_windows[lid] = (ts_index[start], ts_index[end])
            leak_meta[lid] = {"type": info["Leak Type"], "area": area, "start": start, "end": end}

        # D7 (fast part): Labels.csv == union of [start, end]
        if len(leak_windows) == len(leak_ids) and len(lab_val) == c.n_rows:
            union = np.zeros(c.n_rows)
            for a, b in leak_windows.values():
                union[a : b + 1] = 1.0
            if not np.array_equal(union, lab_val):
                bad = int((union != lab_val).sum())
                checks["D7"].fail(f"{tag}: Labels.csv differs from the leak windows on {bad} rows")
        for lid in leak_ids:
            hdr = first_lines(leaks_dir / f"Leak_{lid}_demand.csv", 1)
            if not hdr or hdr[0] != f",{lid}":
                checks["D7"].fail(f"{tag}/{lid}: demand header {hdr[:1]}")

        # D8 / D9 from the scenario .inp
        try:
            net = read_network(d / f"Hanoi_Scenario-{sid}.inp", tail_bytes=inp_tail_bytes)
            merged = merge_split_pipes(net)
        except (SplitPipeError, ValueError, IndexError) as e:
            checks["D8"].fail(f"{tag}: cannot parse/merge .inp ({e})")
            continue
        times_key = tuple(sorted(net.times.items()))
        if ref_times is None:
            ref_times = {"key": times_key, "scenario": sid}
        elif times_key != ref_times["key"]:
            checks["D9"].fail(f"{tag}: [TIMES] differs from Scenario-{ref_times['scenario']}")
        if net.options.get("UNITS") != c.units:
            checks["D5"].fail(f"{tag}: UNITS {net.options.get('UNITS')}")
        _check_split_pipes(tag, sid, merged, link_extra, leak_ids, leak_meta, c, checks, leaks)
        if nominal is not None:
            dev, bdev = _check_against_nominal(tag, net, merged, nominal, c, checks)
            for k, v in dev.items():
                max_dev[k] = max(max_dev[k], v)
            base_demand_dev = max(base_demand_dev, bdev)

        # deep checks
        if deep:
            _deep_scenario(tag, sid, d, leak_ids, leak_windows, leak_meta, c, ts_expected,
                           checks, zero_stats, leak_flow_medians)

    # ---- details and stats
    checks["D3"].details.update(n_rows=c.n_rows, first=c.first_timestamp, last=c.last_timestamp,
                                step_minutes=c.step_minutes, deep_row_counts=deep)
    checks["D9"].details.update(
        uncertainty_levels_pct=sorted({v for key in uncertainty for v in key}),
        n_uncertainty_combinations=len(uncertainty),
        max_abs_rel_deviation={k: round(v, 6) for k, v in sorted(max_dev.items())},
        max_abs_rel_base_demand_deviation=round(base_demand_dev, 6),
    )
    pure = Counter(lk["pipe"] for lk in leaks if lk["n_leaks"] == 1)
    checks["D8"].details.update(
        n_leaks=len(leaks),
        pipes_in_single_leak_scenarios=len(pure),
        single_leak_scenarios_per_pipe={p: pure[p] for p in sorted(pure, key=_pipe_key)},
        never_leaking_observed=sorted(set(c.pipes) - {lk["pipe"] for lk in leaks}, key=_pipe_key),
        double_split_scenarios=sorted({lk["scenario"] for lk in leaks if lk["piece"].endswith("ll")}),
    )
    types = Counter(lk["type"] for lk in leaks)
    stats["leak_types"] = dict(types)
    if deep:
        checks["D5"].details["zero_pressure"] = zero_stats
        stats["leak_flow_lps_median_per_leak"] = {
            k: {"min": round(min(v), 2), "median": round(statistics.median(v), 2), "max": round(max(v), 2)}
            for k, v in leak_flow_medians.items() if v
        }
    else:
        checks["D5"].details["note"] = "NaN/negative/reservoir/zero-pressure checks need --deep"
        checks["D7"].details["note"] = "leak demand window check needs --deep"

    return ValidationReport(checks=checks, deep=deep, contract=c, manifest=manifest(data_dir),
                            stats=stats, leaks=leaks)


def _pipe_key(p: str) -> tuple[int, str]:
    digits = re.match(r"\d+", p)
    return (int(digits.group()) if digits else 0, p)


def _check_nominal(nominal: InpNetwork, c: DataContract, checks: dict[str, Check]) -> None:
    d9 = checks["D9"]
    if set(nominal.junctions) != set(c.junctions):
        d9.fail(f"nominal junctions {sorted(nominal.junctions, key=_pipe_key)} != contract")
    if set(nominal.reservoirs) != {c.reservoir}:
        d9.fail(f"nominal reservoirs {sorted(nominal.reservoirs)}")
    if set(nominal.pipes) != set(c.pipes):
        d9.fail("nominal pipe ids differ from contract")
    if nominal.tanks or nominal.pumps or nominal.valves:
        d9.fail("nominal network has tanks/pumps/valves")
    if set(nominal.coords) != set(c.junctions) | {c.reservoir}:
        d9.fail("nominal coordinates do not cover exactly the nodes")
    if nominal.options.get("UNITS") != c.units:
        checks["D5"].fail(f"nominal UNITS {nominal.options.get('UNITS')} != {c.units}")
    checks["D5"].details["units"] = nominal.options.get("UNITS")
    checks["D5"].details["headloss"] = nominal.options.get("HEADLOSS")

    # D10: how far coordinate distance is from pipe length.
    ratios = []
    for p in nominal.pipes.values():
        if p.start in nominal.coords and p.end in nominal.coords:
            (xa, ya), (xb, yb) = nominal.coords[p.start], nominal.coords[p.end]
            dist = math.hypot(xb - xa, yb - ya)
            if dist > 0:
                ratios.append(p.length / dist)
    if ratios:
        checks["D10"].details["pipe_length_over_coordinate_distance"] = {
            "min": round(min(ratios), 3),
            "median": round(statistics.median(ratios), 3),
            "max": round(max(ratios), 3),
        }
        checks["D10"].details["conclusion"] = (
            "coordinates are not to scale; use pipe lengths" if max(ratios) / min(ratios) > 1.5
            else "coordinates roughly proportional to pipe lengths"
        )


def _check_split_pipes(tag, sid, merged: dict[str, MergedPipe], link_extra, leak_ids, leak_meta,
                       c: DataContract, checks, leaks: list[dict]) -> None:
    d8 = checks["D8"]
    inp_extra = sorted((p for m in merged.values() for p in m.pieces[1:]), key=_pipe_key)
    if sorted(link_extra, key=_pipe_key) != inp_extra:
        d8.fail(f"{tag}: Link_flows extra columns {link_extra} != .inp pieces {inp_extra}")
    found = {ln: m for m in merged.values() for ln in m.leak_nodes}
    if sorted(found) != sorted(leak_ids):
        d8.fail(f"{tag}: leak nodes in .inp {sorted(found)} != leak files {leak_ids}")
    for lid in leak_ids:
        if lid not in found:
            continue
        m = found[lid]
        if m.id in c.never_leaking_pipes or m.id not in c.pipes:
            d8.fail(f"{tag}/{lid}: leak on pipe {m.id}, which is not leak-eligible")
        if len(set(m.diameters)) != 1 or len(set(m.roughnesses)) != 1:
            d8.fail(f"{tag}/{lid}: pieces of pipe {m.id} differ in diameter/roughness")
        piece = m.pieces[m.leak_nodes.index(lid) + 1]
        meta = leak_meta.get(lid, {})
        leaks.append({
            "scenario": sid, "leak": lid, "n_leaks": len(leak_ids), "pipe": m.id, "piece": piece,
            "fraction": round(m.leak_fraction(lid), 6), "type": meta.get("type"),
            "start": meta.get("start"), "end": meta.get("end"),
        })


def _check_against_nominal(tag, net: InpNetwork, merged, nominal: InpNetwork, c: DataContract,
                           checks) -> tuple[dict[str, float], float]:
    d9 = checks["D9"]
    dev = {"length": 0.0, "diameter": 0.0, "roughness": 0.0}
    if set(merged) != set(nominal.pipes):
        d9.fail(f"{tag}: merged pipe ids differ from nominal")
        return dev, 0.0
    tol = c.param_tolerance + 1e-6
    for pid, m in merged.items():
        nom = nominal.pipes[pid]
        if (m.start, m.end) != (nom.start, nom.end):
            d9.fail(f"{tag}: pipe {pid} connects {m.start}->{m.end}, nominal {nom.start}->{nom.end}")
        for key, value, nominal_value in (
            ("length", m.length, nom.length),
            ("diameter", m.diameters[0], nom.diameter),
            ("roughness", m.roughnesses[0], nom.roughness),
        ):
            r = abs(rel_dev(value, nominal_value))
            dev[key] = max(dev[key], r)
            if r > tol:
                d9.fail(f"{tag}: pipe {pid} {key} deviates {r:.3f} > {c.param_tolerance}")
    for nid, xy in nominal.coords.items():
        got = net.coords.get(nid)
        if got is None or abs(got[0] - xy[0]) > 1e-6 or abs(got[1] - xy[1]) > 1e-6:
            d9.fail(f"{tag}: coordinates of node {nid} differ from nominal")
    if set(net.reservoirs) != set(nominal.reservoirs) or net.tanks or net.pumps or net.valves:
        d9.fail(f"{tag}: reservoirs/tanks/pumps/valves differ from nominal")
    if set(j for j in net.junctions if not j.startswith("leak")) != set(nominal.junctions):
        d9.fail(f"{tag}: junction ids differ from nominal")
    bdev = 0.0
    for jid, (_elev, dem) in nominal.junctions.items():
        if jid in net.junctions and dem:
            bdev = max(bdev, abs(rel_dev(net.junctions[jid][1], dem)))
    return dev, bdev


def _deep_scenario(tag, sid, d: Path, leak_ids, leak_windows, leak_meta, c: DataContract,
                   ts_expected, checks, zero_stats, leak_flow_medians) -> None:
    d3, d5, d7 = checks["D3"], checks["D5"], checks["D7"]
    for f in ("Node_demands.csv", "Link_flows.csv"):
        rows = count_rows(d / f)
        if rows != c.n_rows:
            d3.fail(f"{tag}/{f}: {rows} rows != {c.n_rows}")

    df = pd.read_csv(d / "Node_pressures.csv", index_col=0, dtype={0: str})
    if len(df) != c.n_rows or list(df.index) != ts_expected:
        d3.fail(f"{tag}/Node_pressures.csv: {len(df)} rows / timestamps differ")
    values = df.to_numpy(dtype=float)
    if np.isnan(values).any():
        d5.fail(f"{tag}: {int(np.isnan(values).sum())} NaN pressures")
    if (values < 0).any():
        d5.fail(f"{tag}: {int((values < 0).sum())} negative pressures")
    if c.reservoir in df.columns and not (df[c.reservoir] == 0).all():
        d5.fail(f"{tag}: reservoir pressure is not always 0")
    junction = df[[j for j in c.junctions if j in df.columns]].to_numpy(dtype=float)
    zero = junction == 0.0
    if zero.any():
        rows_zero = zero.any(axis=1)
        active = np.zeros(len(rows_zero), dtype=bool)
        for a, b in leak_windows.values():
            active[a : b + 1] = True
        outside = int((rows_zero & ~active[: len(rows_zero)]).sum())
        zero_stats["scenarios_with_zero_junction_pressure"] += 1
        zero_stats["zero_cells"] += int(zero.sum())
        zero_stats["zero_rows"] += int(rows_zero.sum())
        zero_stats["zero_rows_outside_leaks"] += outside
        if len(zero_stats["by_scenario"]) < 200:
            zero_stats["by_scenario"][str(sid)] = {
                "zero_cells": int(zero.sum()),
                "zero_rows": int(rows_zero.sum()),
                "zero_rows_outside_leaks": outside,
                "leak_types": sorted({leak_meta[lid]["type"] for lid in leak_meta}),
            }

    for lid in leak_ids:
        f = d / "Leaks" / f"Leak_{lid}_demand.csv"
        ld = pd.read_csv(f, index_col=0, dtype={0: str})
        if len(ld) != c.n_rows or list(ld.index) != ts_expected:
            d3.fail(f"{tag}/{f.name}: {len(ld)} rows / timestamps differ")
            continue
        flow = ld.iloc[:, 0].to_numpy(dtype=float)
        nz = np.flatnonzero(flow != 0)
        if lid in leak_windows:
            a, b = leak_windows[lid]
            if len(nz) != b - a + 1 or (len(nz) and (nz[0] != a or nz[-1] != b)):
                d7.fail(f"{tag}/{lid}: non-zero flow rows do not match [start, end]")
            if len(nz):
                leak_flow_medians[leak_meta[lid]["type"]].append(float(np.median(flow[nz])))
