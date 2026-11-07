"""``leaksense build-index``: one row per case in ``artifacts/cases.parquet`` (SPEC.md section 5).

A case is a pure single-leak scenario, or the solo segment of a two-leak scenario: from the
start of the leak that starts first until the other leak starts or the first leak ends,
whichever comes first, if that lasts at least ``min_solo_steps``. The first leak is chosen by
start time, not by its id: in many two-leak scenarios ``leak_node1`` starts before
``leak_node0``.

Labels come from the scenario ``.inp`` (the split pipe X, via ``MergedPipe``, looked up by leak
id) and are cross-checked against the ``Xl``/``Xll`` column names in ``Link_flows.csv``
(header only). Leak-active timesteps come from the leak demand files and must equal the
per-scenario ``Labels.csv``. Leak flow, type and area are ground truth, kept for reporting only.

Only small files and headers are read: info files, leak demand files, Labels.csv, the .inp head
and tail, and the first line of Link_flows.csv. Nothing here reads pressures.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from leaksense.data.contract import DataContract
from leaksense.data.inp import merge_split_pipes, read_network
from leaksense.data.validate import expected_timestamps, first_lines, read_kv

STEPS_PER_DAY = 48  # 30-minute steps (D3)


@dataclass(frozen=True)
class IndexConfig:
    min_solo_steps: int = 3 * STEPS_PER_DAY  # a solo segment needs at least 3 days
    min_baseline_steps: int = 14 * STEPS_PER_DAY  # a baseline needs at least 14 leak-free days
    short_case_steps: int = STEPS_PER_DAY  # flag cases lasting under 24 h (end - start)


@dataclass
class Leak:
    id: str
    pipe: str
    piece: str  # the split piece after the leak node: Xl or Xll
    fraction: float
    type: str
    area: float
    diameter: float
    start: int  # row index, inclusive
    end: int  # row index, inclusive
    flow: np.ndarray = field(repr=False)


class IndexBuildError(RuntimeError):
    """The dataset disagrees with itself (labels, link columns, leak windows)."""


def segments(mask: np.ndarray) -> list[list[int]]:
    """Runs of True in ``mask`` as inclusive [start, end] pairs."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return [[int(a), int(b) - 1] for a, b in zip(d[::2], d[1::2], strict=True)]


def choose_baseline(
    active: np.ndarray, case_start: int, case_end: int, min_steps: int
) -> tuple[str | None, list[list[int]], int, int]:
    """SPEC section 5 baseline rule -> (kind, segments, pre_steps, post_steps).

    ``active`` marks every timestep where any leak of the scenario is active. Pre = leak-free
    steps before the case, post = leak-free steps after it. Try pre, then post, then both.
    """
    n = len(active)
    idx = np.arange(n)
    free = ~active
    pre = free & (idx < case_start)
    post = free & (idx > case_end)
    n_pre, n_post = int(pre.sum()), int(post.sum())
    if n_pre >= min_steps:
        kind, mask = "pre", pre
    elif n_post >= min_steps:
        kind, mask = "post", post
    elif n_pre + n_post >= min_steps:
        kind, mask = "combined", pre | post
    else:
        return None, [], n_pre, n_post
    if (mask & active).any():  # cannot happen by construction; keep the invariant loud
        raise AssertionError("baseline overlaps a leak-active timestep")
    return kind, segments(mask), n_pre, n_post


def _read_series(path: Path, n_rows: int) -> np.ndarray:
    df = pd.read_csv(path, index_col=0)
    if len(df) != n_rows:
        raise IndexBuildError(f"{path}: {len(df)} rows, expected {n_rows}")
    return df.iloc[:, 0].to_numpy(dtype=float)


def _read_leaks(d: Path, sid: int, ts_index: dict[str, int], c: DataContract) -> list[Leak]:
    leaks_dir = d / "Leaks"
    info_files = sorted(leaks_dir.glob("Leak_*_info.csv")) if leaks_dir.is_dir() else []
    if not info_files:
        return []
    merged = merge_split_pipes(read_network(d / f"Hanoi_Scenario-{sid}.inp"))
    by_leak = {ln: m for m in merged.values() for ln in m.leak_nodes}

    link_header = first_lines(d / "Link_flows.csv", 1)[0].split(",")
    link_extra = [col for col in link_header[1:] if col not in c.pipes]
    inp_pieces = sorted(p for m in merged.values() for p in m.pieces[1:])
    if sorted(link_extra) != inp_pieces:
        raise IndexBuildError(
            f"Scenario-{sid}: Link_flows pieces {link_extra} != .inp {inp_pieces}"
        )

    leaks = []
    for f in info_files:
        lid = f.name[len("Leak_") : -len("_info.csv")]
        info = dict(read_kv(f)[1])
        if lid not in by_leak:
            raise IndexBuildError(f"Scenario-{sid}: {lid} is not on any split pipe in the .inp")
        m = by_leak[lid]
        piece = m.pieces[m.leak_nodes.index(lid) + 1]
        if piece not in link_extra or piece.rstrip("l") != m.id:
            raise IndexBuildError(f"Scenario-{sid}/{lid}: piece {piece} does not match Link_flows")
        if m.id not in c.pipes or m.id in c.never_leaking_pipes:
            raise IndexBuildError(f"Scenario-{sid}/{lid}: pipe {m.id} is not leak-eligible")
        start = datetime.fromisoformat(info["Leak Start"]).strftime("%Y-%m-%d %H:%M:%S")
        end = datetime.fromisoformat(info["Leak End"]).strftime("%Y-%m-%d %H:%M:%S")
        flow = _read_series(leaks_dir / f"Leak_{lid}_demand.csv", c.n_rows)
        leaks.append(
            Leak(
                id=lid,
                pipe=m.id,
                piece=piece,
                fraction=m.leak_fraction(lid),
                type=info["Leak Type"],
                area=float(info["Leak Area"]),
                diameter=float(info["Leak Diameter"]),
                start=ts_index[start],
                end=ts_index[end],
                flow=flow,
            )
        )
    return leaks


CASE_COLUMNS = [
    "case_id", "scenario_id", "kind", "n_leaks_in_scenario", "leak_id", "pipe", "piece",
    "split_fraction", "leak_type", "area", "leak_diameter", "start", "end", "start_time",
    "end_time", "n_steps", "clipped", "second_leak_id", "second_pipe", "second_start",
    "leak_flow_lps", "pre_steps", "post_steps", "baseline_kind", "baseline_segments",
    "baseline_steps", "flag_short_case", "flag_pre_short",
]  # fmt: skip


@dataclass
class IndexResult:
    cases: pd.DataFrame
    excluded: list[dict]  # candidate cases with no valid baseline
    no_solo: list[dict]  # two-leak scenarios without a long enough solo segment
    config: IndexConfig
    n_scenarios: int

    def summary(self) -> dict:
        cases = self.cases
        n_pure_cand = int((cases["kind"] == "pure").sum()) + sum(
            e["kind"] == "pure" for e in self.excluded
        )
        n_solo_cand = int((cases["kind"] == "solo").sum()) + sum(
            e["kind"] == "solo" for e in self.excluded
        )
        per_pipe = (
            cases.groupby(["pipe", "kind"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=["pure", "solo"], fill_value=0)
        )
        per_pipe = per_pipe.loc[sorted(per_pipe.index, key=int)]
        return {
            "config": asdict(self.config),
            "n_scenarios": self.n_scenarios,
            "candidates": {
                "pure": n_pure_cand,
                "solo": n_solo_cand,
                "total": n_pure_cand + n_solo_cand,
            },
            "cases": {
                "pure": int((cases["kind"] == "pure").sum()),
                "solo": int((cases["kind"] == "solo").sum()),
                "total": len(cases),
            },
            "n_excluded": len(self.excluded),
            "excluded": self.excluded,
            "two_leak_without_solo": self.no_solo,
            "baseline_kind": {k: int(v) for k, v in cases["baseline_kind"].value_counts().items()},
            "baseline_kind_by_kind": {
                f"{k}/{b}": int(v)
                for (k, b), v in cases.groupby(["kind", "baseline_kind"]).size().items()
            },
            "per_pipe": {
                p: {"pure": int(r["pure"]), "solo": int(r["solo"]), "total": int(r.sum())}
                for p, r in per_pipe.iterrows()
            },
            "leak_type": {k: int(v) for k, v in cases["leak_type"].value_counts().items()},
            "solo_clipped": int(cases["clipped"].sum()),
            "solo_first_leak_is_leak_node1": int(
                ((cases["kind"] == "solo") & (cases["leak_id"] == "leak_node1")).sum()
            ),
            "n_steps": {
                "min": int(cases["n_steps"].min()),
                "median": float(cases["n_steps"].median()),
                "max": int(cases["n_steps"].max()),
            }
            if len(cases)
            else {},
            "flag_short_case": cases.loc[cases["flag_short_case"], "case_id"].tolist(),
            "leak_flow_lps": {
                k: {
                    "min": round(float(g.min()), 2),
                    "median": round(float(g.median()), 2),
                    "max": round(float(g.max()), 2),
                }
                for k, g in cases.groupby("leak_type")["leak_flow_lps"]
            },
        }


def build_index(
    data_dir: str | Path,
    contract: DataContract | None = None,
    config: IndexConfig | None = None,
    progress: bool = False,
) -> IndexResult:
    data_dir = Path(data_dir)
    c = contract or DataContract()
    cfg = config or IndexConfig()
    ts = expected_timestamps(c)
    ts_index = {t: i for i, t in enumerate(ts)}

    with open(data_dir / "Labels.csv", encoding="utf-8") as f:
        top = {int(r[0]): int(r[1]) for r in list(csv.reader(f))[1:] if r}

    rows: list[dict] = []
    excluded: list[dict] = []
    no_solo: list[dict] = []
    for n, sid in enumerate(sorted(top), 1):
        if progress and (n % 50 == 0 or n == len(top)):
            print(f"  indexed {n}/{len(top)} scenarios", flush=True)
        if top[sid] == 0:
            continue
        d = data_dir / f"Scenario-{sid}"
        leaks = _read_leaks(d, sid, ts_index, c)
        if len(leaks) != top[sid]:
            raise IndexBuildError(
                f"Scenario-{sid}: {len(leaks)} leak files, Labels.csv says {top[sid]}"
            )
        if len(leaks) > 2:
            raise IndexBuildError(f"Scenario-{sid}: {len(leaks)} leaks; only 1 or 2 are supported")

        active = np.zeros(c.n_rows, dtype=bool)
        for lk in leaks:
            nz = np.flatnonzero(lk.flow != 0)
            if len(nz) != lk.end - lk.start + 1 or nz[0] != lk.start or nz[-1] != lk.end:
                raise IndexBuildError(
                    f"Scenario-{sid}/{lk.id}: demand is not non-zero on [start, end]"
                )
            active[lk.start : lk.end + 1] = True
        labels = _read_series(d / "Labels.csv", c.n_rows) != 0
        if not np.array_equal(labels, active):
            raise IndexBuildError(f"Scenario-{sid}: Labels.csv differs from the leak windows")

        leaks.sort(key=lambda lk: (lk.start, lk.id))
        first = leaks[0]
        second = leaks[1] if len(leaks) == 2 else None
        start, end, clipped = first.start, first.end, False
        if second is not None:
            if second.start <= end:
                end, clipped = second.start - 1, True
            n_steps = end - start + 1
            if n_steps < cfg.min_solo_steps:
                no_solo.append(
                    {
                        "scenario_id": sid,
                        "solo_steps": max(n_steps, 0),
                        "first": first.id,
                        "pipes": [first.pipe, second.pipe],
                    }
                )
                continue
        kind = "pure" if second is None else "solo"
        n_steps = end - start + 1
        bkind, bsegs, n_pre, n_post = choose_baseline(active, start, end, cfg.min_baseline_steps)
        case_id = f"S{sid:03d}-{kind}"
        if bkind is None:
            excluded.append(
                {
                    "case_id": case_id,
                    "scenario_id": sid,
                    "kind": kind,
                    "pipe": first.pipe,
                    "pre_steps": n_pre,
                    "post_steps": n_post,
                    "reason": f"only {n_pre} pre + {n_post} post leak-free steps "
                    f"(< {cfg.min_baseline_steps})",
                }
            )
            continue
        rows.append(
            {
                "case_id": case_id,
                "scenario_id": sid,
                "kind": kind,
                "n_leaks_in_scenario": len(leaks),
                "leak_id": first.id,
                "pipe": first.pipe,
                "piece": first.piece,
                "split_fraction": first.fraction,
                "leak_type": first.type,
                "area": first.area,
                "leak_diameter": first.diameter,
                "start": start,
                "end": end,
                "start_time": ts[start],
                "end_time": ts[end],
                "n_steps": n_steps,
                "clipped": clipped,
                "second_leak_id": second.id if second else None,
                "second_pipe": second.pipe if second else None,
                "second_start": second.start if second else None,
                "leak_flow_lps": float(np.median(first.flow[start : end + 1])),
                "pre_steps": n_pre,
                "post_steps": n_post,
                "baseline_kind": bkind,
                "baseline_segments": bsegs,
                "baseline_steps": int(sum(b - a + 1 for a, b in bsegs)),
                "flag_short_case": n_steps <= cfg.short_case_steps,
                "flag_pre_short": n_pre < cfg.min_baseline_steps,
            }
        )

    cases = pd.DataFrame(rows, columns=CASE_COLUMNS)
    cases["second_start"] = cases["second_start"].astype("Int64")
    return IndexResult(cases, excluded, no_solo, cfg, len(top))
