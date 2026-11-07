"""The data contract (SPEC.md section 8) as explicit, overridable expectations.

The defaults describe the real LeakDB Hanoi dataset. Tests build a tiny synthetic dataset
and pass a matching contract, so the same validation code runs on both.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataContract:
    n_scenarios: int = 500
    # (number of leaks, number of scenarios) pairs, from the top-level Labels.csv.
    label_counts: tuple[tuple[int, int], ...] = ((0, 1), (1, 244), (2, 255))
    n_rows: int = 17_520
    first_timestamp: str = "2024-01-01 00:00:00"
    last_timestamp: str = "2024-12-30 23:30:00"
    step_minutes: int = 30
    duration_text: str = "8760 hours"
    time_step_text: str = "30.0 min"
    reservoir: str = "1"
    junctions: tuple[str, ...] = tuple(str(i) for i in range(2, 33))
    pipes: tuple[str, ...] = tuple(str(i) for i in range(1, 35))
    never_leaking_pipes: tuple[str, ...] = ("1",)
    units: str = "LPS"
    # Max relative deviation of per-scenario pipe length/diameter/roughness from nominal.
    param_tolerance: float = 0.20
    leak_types: tuple[str, ...] = ("small", "big")
    leak_info_keys: tuple[str, ...] = (
        "Leak Node",
        "Leak Area",
        "Leak Diameter",
        "Leak Type",
        "Leak Start",
        "Leak End",
    )

    @property
    def label_count_map(self) -> dict[int, int]:
        return dict(self.label_counts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DataContract:
        kwargs = {}
        for key, value in d.items():
            if key == "label_counts":
                value = tuple((int(a), int(b)) for a, b in value)
            elif isinstance(value, list):
                value = tuple(value)
            kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str | Path) -> DataContract:
        return cls.from_dict(json.loads(Path(path).read_text()))
