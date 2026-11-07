import json
import os

import numpy as np
import pandas as pd
import pytest

from leaksense.cli import main
from leaksense.config import DATA_DIR_VAR
from leaksense.data.index import (
    IndexBuildError,
    IndexConfig,
    build_index,
    choose_baseline,
    segments,
)

from synthetic_data import LeakSpec, ScenarioSpec, contract_for, make_dataset

CFG = IndexConfig(min_solo_steps=10, min_baseline_steps=20, short_case_steps=48)

SPECS = {
    1: ScenarioSpec(),
    2: ScenarioSpec([LeakSpec("3", 40, 70, "small", 0.25)]),  # pure, pre baseline
    3: ScenarioSpec([LeakSpec("2", 5, 90, "big", 0.5)]),  # 5 pre + 5 post: excluded
    4: ScenarioSpec([LeakSpec("5", 12, 85, "small", 0.4)]),  # 12 pre + 10 post: combined
    # leak_node1 starts first; the gap before leak_node0 counts as post-leak baseline
    5: ScenarioSpec([LeakSpec("2", 50, 80, "big", 0.6), LeakSpec("5", 10, 40, "small", 0.3)]),
    6: ScenarioSpec([LeakSpec("2", 30, 60), LeakSpec("3", 35, 70)]),  # solo of 5 steps: none
    # same pipe split twice; leak_node1 (on piece 4ll) starts first
    7: ScenarioSpec([LeakSpec("4", 60, 95, "big", 0.2), LeakSpec("4", 10, 30, "small", 0.5)]),
    # second leak starts while the first is active: solo clipped at step 44
    8: ScenarioSpec([LeakSpec("3", 25, 80, "big", 0.5), LeakSpec("5", 45, 60, "small", 0.5)]),
}


@pytest.fixture
def custom(tmp_path):
    return make_dataset(tmp_path, SPECS), contract_for(SPECS)


def test_segments():
    assert segments(np.array([0, 1, 1, 0, 1], dtype=bool)) == [[1, 2], [4, 4]]
    assert segments(np.zeros(3, dtype=bool)) == []


def test_choose_baseline_order():
    active = np.zeros(100, dtype=bool)
    active[30:60] = True
    assert choose_baseline(active, 30, 59, 30)[:2] == ("pre", [[0, 29]])
    assert choose_baseline(active, 30, 59, 35)[:2] == ("post", [[60, 99]])
    assert choose_baseline(active, 30, 59, 50)[:2] == ("combined", [[0, 29], [60, 99]])
    assert choose_baseline(active, 30, 59, 71) == (None, [], 30, 40)


def test_index_cases(custom):
    data_dir, contract = custom
    res = build_index(data_dir, contract, CFG)
    cases = res.cases.set_index("scenario_id")
    assert sorted(cases.index) == [2, 4, 5, 7, 8]
    assert [e["scenario_id"] for e in res.excluded] == [3]
    assert res.excluded[0]["pre_steps"] == 5 and res.excluded[0]["post_steps"] == 5
    assert [e["scenario_id"] for e in res.no_solo] == [6]

    c2 = cases.loc[2]
    assert (c2["kind"], c2["pipe"], c2["piece"]) == ("pure", "3", "3l")
    assert c2["split_fraction"] == pytest.approx(0.25)
    assert (c2["start"], c2["end"], c2["n_steps"]) == (40, 70, 31)
    assert c2["baseline_kind"] == "pre" and list(map(list, c2["baseline_segments"])) == [[0, 39]]
    assert c2["flag_short_case"]  # 31 steps < 24 h
    assert 100 < c2["leak_flow_lps"] < 120  # synthetic small leak ~110 L/s

    c4 = cases.loc[4]
    assert c4["baseline_kind"] == "combined" and c4["baseline_steps"] == 22

    c5 = cases.loc[5]  # first leak by start time is leak_node1
    assert (c5["kind"], c5["leak_id"], c5["pipe"], c5["second_pipe"]) == (
        "solo",
        "leak_node1",
        "5",
        "2",
    )
    assert (c5["start"], c5["end"], c5["clipped"]) == (10, 40, False)
    assert c5["baseline_kind"] == "post"
    assert list(map(list, c5["baseline_segments"])) == [[41, 49], [81, 95]]

    c7 = cases.loc[7]  # Xll: leak mapped by id, not by chain position
    assert (c7["leak_id"], c7["pipe"], c7["piece"]) == ("leak_node1", "4", "4ll")
    assert c7["split_fraction"] == pytest.approx(0.2 + 0.8 * 0.5)
    assert (c7["start"], c7["end"]) == (10, 30) and c7["baseline_kind"] == "post"

    c8 = cases.loc[8]
    assert (c8["start"], c8["end"], c8["clipped"], c8["second_start"]) == (25, 44, True, 45)


def test_baseline_never_overlaps_a_leak(custom):
    data_dir, contract = custom
    res = build_index(data_dir, contract, CFG)
    for _, row in res.cases.iterrows():
        sid = row["scenario_id"]
        labels = pd.read_csv(data_dir / f"Scenario-{sid}" / "Labels.csv")["Label"].to_numpy()
        for a, b in row["baseline_segments"]:
            assert labels[a : b + 1].sum() == 0
        assert labels[row["start"] : row["end"] + 1].all()
        assert row["baseline_steps"] >= CFG.min_baseline_steps


def test_link_flows_mismatch_is_caught(custom):
    data_dir, contract = custom
    f = data_dir / "Scenario-2" / "Link_flows.csv"
    lines = f.read_text().splitlines()
    lines[0] = lines[0].replace(",3l", ",2l")
    f.write_text("\n".join(lines) + "\n")
    with pytest.raises(IndexBuildError, match="Link_flows"):
        build_index(data_dir, contract, CFG)


def test_labels_mismatch_is_caught(custom):
    data_dir, contract = custom
    f = data_dir / "Scenario-2" / "Labels.csv"
    df = pd.read_csv(f)
    df.loc[5, "Label"] = 1.0
    df.to_csv(f, index=False, float_format="%.1f")
    with pytest.raises(IndexBuildError, match="Labels.csv"):
        build_index(data_dir, contract, CFG)


def test_index_does_not_write_to_dataset(custom):
    data_dir, contract = custom
    before = {p: p.stat().st_mtime_ns for p in data_dir.rglob("*")}
    build_index(data_dir, contract, CFG)
    after = {p: p.stat().st_mtime_ns for p in data_dir.rglob("*")}
    assert before == after


def test_cli_build_index(synthetic, tmp_path, monkeypatch, capsys):
    data_dir, contract = synthetic
    (tmp_path / "contract.json").write_text(json.dumps(contract.to_dict()))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LEAKSENSE_ARTIFACTS_DIR", str(tmp_path / "art"))
    rc = main(
        [
            "build-index",
            "--data-dir",
            str(data_dir),
            "--quiet",
            "--contract",
            "contract.json",
            "--report",
            str(tmp_path / "idx.json"),
            "--validation",
            "missing.json",
        ]
    )
    out = capsys.readouterr().out
    # With the real thresholds (14-day baseline), the 2-day synthetic data has no usable case.
    assert rc == 0 and "excluded (no baseline): 1" in out
    report = json.loads((tmp_path / "idx.json").read_text())
    assert report["candidates"]["pure"] == 1 and report["cases"]["total"] == 0
    assert report["data_manifest_sha256"] and "git" in report
    assert len(pd.read_parquet(tmp_path / "art" / "cases.parquet")) == 0


@pytest.mark.data
def test_real_index():
    res = build_index(os.environ[DATA_DIR_VAR])
    s = res.summary()
    assert s["candidates"] == {"pure": 244, "solo": 245, "total": 489}
    assert s["cases"]["total"] + s["n_excluded"] == 489
    assert set(res.cases["pipe"]) == {str(p) for p in range(2, 35)}
    assert len(res.no_solo) == 10
    assert sorted(res.cases.loc[res.cases["piece"].str.endswith("ll"), "scenario_id"]) == [
        46,
        269,
        441,
    ]
