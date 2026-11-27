"""M15: leak scan, detection and the overlap stress test on a small synthetic dataset."""

import json

import numpy as np
import pandas as pd
import pytest

from leaksense.data.index import build_index
from leaksense.stretch import auc, detect_scores, overlap_periods

from conftest import FINAL
from synthetic_data import LeakSpec, ScenarioSpec, contract_for, make_dataset

ROWS = 40 * 48
SPECS = {
    1: ScenarioSpec(),
    # first leak alone for 200 steps (a solo case), then both active for 101 steps
    2: ScenarioSpec([LeakSpec("3", 1500, 1800, "big", 0.5), LeakSpec("5", 1700, 1850, "small")]),
    3: ScenarioSpec([LeakSpec("2", 1750, 1900, "small", 0.4)]),
    4: ScenarioSpec([LeakSpec("5", 1690, 1850, "big", 0.6)]),  # locked test
}


@pytest.fixture
def stretch_env(tmp_path, monkeypatch):
    from leaksense.cli import main

    data_dir = make_dataset(tmp_path, SPECS, n_rows=ROWS, daily_amp=2.0)
    contract = contract_for(SPECS, n_rows=ROWS)
    (tmp_path / "contract.json").write_text(json.dumps(contract.to_dict()))
    build_index(data_dir, contract).cases.to_parquet(tmp_path / "cases.parquet")
    (tmp_path / "splits.json").write_text(json.dumps({"test": [4], "folds": [[2], [3]]}))
    base = "name: t\nfeatures:\n  slot: tod_weekday\n  max_lag_days: 7\n"
    (tmp_path / "cfg.yaml").write_text(base)
    (tmp_path / "final.yaml").write_text(base + FINAL)
    monkeypatch.setenv("LEAKSENSE_ARTIFACTS_DIR", str(tmp_path / "art"))
    common = ["--data-dir", str(data_dir), "--contract", str(tmp_path / "contract.json")]
    common += ["--cases", str(tmp_path / "cases.parquet"), "--config", str(tmp_path / "cfg.yaml")]
    assert main(["build-features", *common, "--quiet"]) == 0
    sens = ["build-sensitivity", *common, "--splits", str(tmp_path / "splits.json")]
    assert main([*sens, "--out", str(tmp_path / "sens"), "--quiet"]) == 0
    (tmp_path / "place").mkdir()
    (tmp_path / "place" / "placement__final.json").write_text(
        json.dumps({"greedy_order_idx": [2, 0, 3, 1]})
    )
    ev = ["eval", *common, "--splits", str(tmp_path / "splits.json"), "--draws", "3"]
    ev += ["--bootstrap", "20", "--reports-dir", str(tmp_path / "rep")]
    ev += ["--placement-runs", str(tmp_path / "place")]
    return tmp_path, ev


def test_detect_scores_are_one_sided_and_auc():
    x = np.array([[-1.0, 0.0, 2.0], [0.5, 0.5, 0.5]])
    m = np.array([True, True, False])
    s = detect_scores(x, m, np.ones(3))
    assert s["max"].tolist() == [1.0, -0.5]  # a pressure rise never raises the score
    assert s["mean"].tolist() == [0.5, -0.5]
    assert auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert auc(np.array([1.0]), np.array([1.0])) == 0.5


def test_overlap_periods_counts_both_active_steps():
    leaks = pd.DataFrame(
        [
            (2, 2, 0, "3", "big", 10, 50, 5.0),
            (2, 2, 1, "5", "small", 40, 60, 1.0),
            (3, 2, 0, "2", "big", 10, 20, 5.0),
            (3, 2, 1, "2", "small", 30, 40, 1.0),
        ],
        columns=[
            "scenario_id", "n_leaks", "order", "pipe", "leak_type", "start", "end",
            "flow_median_lps",
        ],
    )  # fmt: skip
    per = overlap_periods(leaks).set_index("scenario_id")
    assert per.loc[2, "ov_steps"] == 11 and per.loc[2, "ov_start"] == 40
    assert per.loc[3, "ov_steps"] == 0 and per.loc[3, "same_pipe"]


def test_cli_detect_uses_cv_scenarios_only(stretch_env):
    from leaksense.cli import main

    tmp_path, ev = stretch_env
    assert main([*ev, "--task", "detect"]) == 0
    rep = json.loads((tmp_path / "rep" / "detection.json").read_text())
    assert rep["n_scenarios"] == 2  # scenarios 2 and 3; the locked test (4) is never read
    leaks = pd.read_parquet(next((tmp_path / "art" / "stretch").glob("leaks_*.parquet")))
    assert 4 not in set(leaks["scenario_id"]) and len(leaks) == 3
    assert rep["ramp"]["n_leaks"] == 3 and rep["n_negatives"] > 0
    r = rep["results"]["greedy"]["4"]["max"]
    assert 0 <= r["alpha=0.01"]["fpr"]["value"] <= 1 and 0 <= r["auc"] <= 1
    assert set(rep["results"]) == {"greedy", "random"}
    md = (tmp_path / "rep" / "detection.md").read_text(encoding="utf-8")
    assert "ramp check" in md


def test_cli_overlap_scores_the_both_active_period(stretch_env):
    from leaksense.cli import main

    tmp_path, ev = stretch_env
    ov = [*ev, "--stress", "overlap", "--model-config", str(tmp_path / "final.yaml")]
    ov += ["--runs-dir", str(tmp_path / "ovruns")]
    assert main([*ov, "--fold", "0"]) == 0
    assert not (tmp_path / "rep" / "overlap.json").exists()
    assert main(ov) == 0
    rep = json.loads((tmp_path / "rep" / "overlap.json").read_text())
    c = rep["counts"]
    assert c["n_two_leak"] == 1 and c["n_scorable"] == 1
    assert c["ov_days"]["median"] == pytest.approx(101 / 48)
    r = rep["results"]["gat_snode"]["random"]["4"]["different_pipes"]
    assert r["hit_first"]["n"] == 1 and 0 <= r["hit_either"]["value"] <= 1
    assert r["hit_either"]["value"] >= r["hit_first"]["value"] >= r["hit_both"]["value"]
    assert "sigmatch_all" in rep["results"]
    df = pd.read_parquet(tmp_path / "ovruns" / "overlap__gat_snode_f0.parquet")
    assert set(df["scenario_id"]) == {2}
