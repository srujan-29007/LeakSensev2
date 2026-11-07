"""Minimal, read-only EPANET .inp parsing for validation and labels.

Scenario .inp files are ~5 MB, almost all of it in [PATTERNS]. ``read_inp_sections`` reads
the head of the file up to [PATTERNS] and the sections after it from the file's tail, so the
pattern block is never parsed. (WNTR is used for hydraulics later; it is not needed here.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

Sections = dict[str, list[list[str]]]


def parse_inp_lines(lines, skip: frozenset[str] = frozenset()) -> Sections:
    """Parse .inp text lines into {"[SECTION]": [token rows]}, dropping comments."""
    sections: Sections = {}
    current: str | None = None
    for raw in lines:
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            current = line.upper()
            sections.setdefault(current, [])
            continue
        if current is not None and current not in skip:
            sections[current].append(line.split())
    return sections


def read_inp_sections(path: str | os.PathLike[str], tail_bytes: int = 65_536) -> Sections:
    """Read all sections except the bodies of [PATTERNS] (and [CURVES])."""
    path = Path(path)
    size = path.stat().st_size
    skip = frozenset({"[PATTERNS]", "[CURVES]"})
    if size <= 2 * tail_bytes:
        with open(path, encoding="utf-8", errors="replace") as f:
            return parse_inp_lines(f, skip)

    head: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip().upper().startswith("[PATTERNS]"):
                break
            head.append(line)
    with open(path, "rb") as f:
        f.seek(size - tail_bytes)
        tail = f.read().decode("utf-8", errors="replace").splitlines()[1:]  # drop partial line
    first_header = next((i for i, ln in enumerate(tail) if ln.strip().startswith("[")), None)
    if first_header is None or not any(
        ln.strip().upper().startswith("[COORDINATES]") for ln in tail[first_header:]
    ):
        with open(path, encoding="utf-8", errors="replace") as f:  # unusual layout: full parse
            return parse_inp_lines(f, skip)
    sections = parse_inp_lines(head, skip)
    sections.update(parse_inp_lines(tail[first_header:], skip))
    return sections


@dataclass(frozen=True)
class Pipe:
    id: str
    start: str
    end: str
    length: float
    diameter: float
    roughness: float


@dataclass
class InpNetwork:
    junctions: dict[str, tuple[float, float]]  # id -> (elevation, base demand)
    reservoirs: dict[str, float]  # id -> head
    tanks: list[str]
    pumps: list[str]
    valves: list[str]
    pipes: dict[str, Pipe]
    coords: dict[str, tuple[float, float]]
    options: dict[str, str]
    times: dict[str, str]


def _keyed(rows: list[list[str]]) -> dict[str, str]:
    """[OPTIONS]/[TIMES] rows: last token is the value, the rest is the (multi-word) key."""
    out = {}
    for row in rows:
        if len(row) >= 2:
            out[" ".join(row[:-1]).upper()] = row[-1]
    return out


def network_from_sections(s: Sections) -> InpNetwork:
    junctions = {
        r[0]: (float(r[1]), float(r[2]) if len(r) > 2 else 0.0) for r in s.get("[JUNCTIONS]", [])
    }
    reservoirs = {r[0]: float(r[1]) for r in s.get("[RESERVOIRS]", [])}
    pipes = {
        r[0]: Pipe(r[0], r[1], r[2], float(r[3]), float(r[4]), float(r[5]))
        for r in s.get("[PIPES]", [])
    }
    coords = {r[0]: (float(r[1]), float(r[2])) for r in s.get("[COORDINATES]", [])}
    return InpNetwork(
        junctions=junctions,
        reservoirs=reservoirs,
        tanks=[r[0] for r in s.get("[TANKS]", [])],
        pumps=[r[0] for r in s.get("[PUMPS]", [])],
        valves=[r[0] for r in s.get("[VALVES]", [])],
        pipes=pipes,
        coords=coords,
        options=_keyed(s.get("[OPTIONS]", [])),
        times=_keyed(s.get("[TIMES]", [])),
    )


def read_network(path: str | os.PathLike[str], tail_bytes: int = 65_536) -> InpNetwork:
    return network_from_sections(read_inp_sections(path, tail_bytes=tail_bytes))


@dataclass
class MergedPipe:
    """A nominal pipe reassembled from its leak-split pieces X, Xl, Xll, ..."""

    id: str
    start: str
    end: str
    pieces: list[str]
    piece_lengths: list[float]
    diameters: list[float]
    roughnesses: list[float]
    leak_nodes: list[str] = field(default_factory=list)  # in order from start to end

    @property
    def length(self) -> float:
        return sum(self.piece_lengths)

    def leak_fraction(self, leak_node: str) -> float:
        """Position of ``leak_node`` along the pipe, 0 at start, 1 at end."""
        i = self.leak_nodes.index(leak_node)
        return sum(self.piece_lengths[: i + 1]) / self.length


class SplitPipeError(ValueError):
    pass


def merge_split_pipes(net: InpNetwork) -> dict[str, MergedPipe]:
    """Undo leak insertion: chain X: a->leak, Xl: leak->(leak|b), ... back into pipe X."""
    merged: dict[str, MergedPipe] = {}
    used: set[str] = set()
    for pid in net.pipes:
        if pid.endswith("l") and pid.rstrip("l") in net.pipes:
            continue  # a piece, handled with its base
        chain = [pid]
        while chain[-1] + "l" in net.pipes:
            chain.append(chain[-1] + "l")
        pieces = [net.pipes[c] for c in chain]
        leak_nodes = []
        for a, b in zip(pieces, pieces[1:], strict=False):
            if a.end != b.start:
                raise SplitPipeError(
                    f"pipe {pid}: piece {a.id} ends at {a.end}, {b.id} starts at {b.start}"
                )
            if not a.end.startswith("leak"):
                raise SplitPipeError(f"pipe {pid}: split node {a.end} is not a leak node")
            leak_nodes.append(a.end)
        if pieces[-1].end.startswith("leak"):
            raise SplitPipeError(f"pipe {pid}: chain ends at leak node {pieces[-1].end}")
        merged[pid] = MergedPipe(
            id=pid,
            start=pieces[0].start,
            end=pieces[-1].end,
            pieces=chain,
            piece_lengths=[p.length for p in pieces],
            diameters=[p.diameter for p in pieces],
            roughnesses=[p.roughness for p in pieces],
            leak_nodes=leak_nodes,
        )
        used.update(chain)
    orphans = set(net.pipes) - used
    if orphans:
        raise SplitPipeError(f"pipe pieces without a base pipe: {sorted(orphans)}")
    return merged
