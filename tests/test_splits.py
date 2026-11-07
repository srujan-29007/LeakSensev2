import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leaksense.cli import main
from leaksense.config import DATA_DIR_VAR
from leaksense.data.splits import SplitConfig, SplitError, check_splits, make_splits


def fake_cases(seed: int = 0) -> pd.DataFrame:
    """~200 cases over 20 pipes; pipe 2 is rare (4 cases), pipe 3 very rare (2 cases)."""
    rng = np.random.default_rng(seed)
    counts = {"2": 4, "3": 2, **{str(p): int(rng.integers(6, 20)) for p in range(4, 22)}}
    rows, sid = [], 1
    for pipe, n in counts.items():
        for _ in range(n):
            sid += 1
            rows.append(
                {
                    "scenario_id": sid,
                    "pipe": pipe,
                    "leak_type": "small" if rng.random() < 0.5 else "big",
                    "kind": "pure" if rng.random() < 0.5 else "solo",
                }
            )
    return pd.DataFrame(rows)


CFG = SplitConfig(n_seed_trials=5)


def test_no_overlap_and_all_assigned():
    cases = fake_cases()
    s = make_splits(cases, CFG)
    parts = [s["test"], *s["folds"]]
    flat = [x for p in parts for x in p]
    assert len(flat) == len(set(flat)) == len(cases)
    assert 0.10 < len(s["test"]) / len(cases) < 0.20
    assert all(len(f) > 0 for f in s["folds"]) and len(s["folds"]) == 5


def test_deterministic_for_seed():
    cases = fake_cases()
    a, b = make_splits(cases, CFG), make_splits(cases.sample(frac=1, random_state=3), CFG)
    assert a["test"] == b["test"] and a["folds"] == b["folds"]
    c = make_splits(cases, SplitConfig(seed=1, n_seed_trials=5))
    assert c["test"] != a["test"]


def test_rare_pipe_rule():
    cases = fake_cases()
    s = make_splits(cases, CFG)
    test = set(s["test"])
    for _pipe, g in cases.groupby("pipe"):
        n_test = int(g["scenario_id"].isin(test).sum())
        if n_test:
            assert len(g) - n_test >= CFG.min_train_per_pipe
    assert set(s["rare_pipes"]) == {"2", "3"}
    assert s["rare_pipes"]["3"]["n_test"] == 0  # only 2 cases: both stay in training


def test_check_splits_catches_overlap():
    cases = fake_cases()
    s = make_splits(cases, CFG)
    s["folds"][0] = s["folds"][0] + s["test"][:1]
    with pytest.raises(SplitError, match="more than one"):
        check_splits(s, cases)


def test_cli_make_splits(tmp_path, capsys):
    cases_path = tmp_path / "cases.parquet"
    fake_cases().to_parquet(cases_path)
    out = tmp_path / "splits.json"
    rc = main(["make-splits", "--cases", str(cases_path), "--out", str(out),
               "--index-report", str(tmp_path / "none.json"), "--trials", "2"])  # fmt: skip
    assert rc == 0
    s = json.loads(out.read_text())
    assert s["provenance"]["cases_sha256"] and s["provenance"]["git"]
    assert "fold   cases" in capsys.readouterr().out


@pytest.mark.data
def test_committed_splits_match_index():
    """splits/splits_v1.json covers exactly the indexed scenarios with no overlap."""
    from leaksense.config import get_artifacts_dir

    assert os.environ.get(DATA_DIR_VAR)
    cases = pd.read_parquet(get_artifacts_dir() / "cases.parquet")
    s = json.loads(Path("splits/splits_v1.json").read_text())
    check_splits(s, cases)
