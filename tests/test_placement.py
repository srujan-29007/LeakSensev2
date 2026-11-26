import json

import numpy as np
import pandas as pd
import pytest

from leaksense.evaluate import case_metrics
from leaksense.placement import (
    greedy_coverage,
    greedy_learned,
    jaccard,
    mean_pairwise_jaccard,
    topk_hits,
)

from conftest import FINAL


def _toy_score_fn(n_sensors=6, n_cls=4, n_cases=40, seed=0):
    """Cases whose labels are readable only from junction 3 (plus a weaker hint at 1)."""
    rng = np.random.default_rng(seed)
    ys = rng.integers(0, n_cls, n_cases)

    def score_fn(m):
        sc = np.zeros((n_cases, n_cls))
        if m[3]:
            sc[np.arange(n_cases), ys] += 2.0
        if m[1]:
            sc[np.arange(n_cases), (ys + 1) % n_cls] += 1.0
            sc[np.arange(n_cases), ys] += 0.5
        return sc

    return score_fn, ys, n_sensors


def test_topk_hits_matches_case_metrics():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 4, (50, 7)).astype(float)  # many ties
    ys = rng.integers(0, 7, 50)
    want = [
        case_metrics(s[i], int(ys[i]), np.zeros(7), np.zeros((7, 7)))["top3"] for i in range(50)
    ]
    assert np.allclose(topk_hits(s, ys, 3), want)


def test_greedy_picks_the_informative_junction_first_and_is_nested():
    fn, ys, n = _toy_score_fn()
    order, trace = greedy_learned(fn, ys, n, 4)
    assert order[0] == 3
    assert len(set(order)) == 4 and trace[0]["top3"] == 1.0
    order2, _ = greedy_learned(fn, ys, n, 2)
    assert order2 == order[:2]  # nested: a smaller budget is a prefix


def test_greedy_is_deterministic_and_breaks_ties_by_lowest_junction():
    ys = np.array([0, 1, 2])

    def flat(m):
        return np.zeros((3, 3))  # every set scores the same

    a, _ = greedy_learned(flat, ys, 5, 3)
    b, _ = greedy_learned(flat, ys, 5, 3)
    assert a == b == [0, 1, 2]


def test_coverage_greedy_order():
    # 3 junctions x 2 pipes: 0 sees pipe 0, 2 sees pipe 1, 1 sees some of both
    S = np.array([[-1.0, 0.0], [-0.6, -0.6], [0.0, -1.0]])
    assert greedy_coverage(S, 3) == [1, 0, 2]  # 1 covers 0.6 of both (mean 0.6 > 0.5)
    S2 = np.array([[-1.0, 0.0], [-0.2, -0.2], [0.0, -1.0]])
    assert greedy_coverage(S2, 2) == [0, 2]
    assert greedy_coverage(S2[:, :, None, None] * np.ones((1, 1, 2, 3)), 2) == [0, 2]


def test_jaccard():
    assert jaccard([1, 2], [2, 3]) == pytest.approx(1 / 3)
    assert mean_pairwise_jaccard([[1, 2], [1, 2], [1, 2]]) == 1.0


def test_fit_keeps_the_inner_net_and_its_validation_rows(synthetic):
    from leaksense.models.mlp import MaskedMLP, inner_split
    from leaksense.network import load_nominal_network

    data_dir, contract = synthetic
    net = load_nominal_network(data_dir, contract)
    c, j = len(net.leak_pipe_ids), len(net.sensor_ids)
    ys = np.repeat(np.arange(c), 10)
    cases = pd.DataFrame({"case_id": [f"c{i}" for i in range(len(ys))]})
    cases["scenario_id"] = np.arange(len(ys)) + 2
    cases["y"] = ys
    cases["pipe"] = [net.leak_pipe_ids[y] for y in ys]
    rng = np.random.default_rng(0)
    windows = [rng.normal(-y, 0.1, (3, j)) for y in ys]
    cfg = {"hidden": 8, "max_epochs": 3, "patience": 2, "val_frac": 0.3, "seed": 0}
    m = MaskedMLP(net, cfg)
    m.fit(cases, windows)
    want = np.flatnonzero(inner_split(cases, 0.3, _seed(cases)))
    assert m.inner_net is not None and m.inner_net is not m.net  # refit makes a new net
    assert len(m.inner_val_idx) == m.history[-1]["n_inner_val_cases"] > 0
    assert list(m.inner_val_idx) == list(want)


def _seed(cases):
    from leaksense.models.mlp import _seed_for

    return _seed_for(cases, 0)


def test_cli_place_runs_resumes_and_reports(place_env):
    from leaksense.cli import main

    tmp_path, args = place_env
    assert main([*args, "--fold", "0"]) == 0
    runs = tmp_path / "runs"
    assert (runs / "placement__f0.json").is_file() and not (tmp_path / "rep").exists()
    stamp = (runs / "placement__f0.json").stat().st_mtime_ns

    assert main(args) == 0
    assert (runs / "placement__f0.json").stat().st_mtime_ns == stamp  # done fold skipped
    rep = json.loads((tmp_path / "rep" / "placement.json").read_text())
    assert rep["ks"] == [1, 2] and rep["n_random"] == 3
    assert set(rep["folds"]) == {"0", "1"}
    assert rep["final"]["greedy_order"]
    r = rep["results"]["model"]["2"]["overall"]
    assert 0 <= r["greedy"]["top3"]["value"] <= 1 and 0 <= r["greedy_percentile"] <= 1
    assert "sigmatch_all" in rep["results"]
    # the locked test scenario (4) never reaches the placement cases
    df = pd.concat(pd.read_parquet(p) for p in runs.glob("placement__f*.parquet"))
    assert not df["case_id"].str.startswith("S004").any()
    assert (tmp_path / "rep" / "placement_curve.png").stat().st_size > 0
    assert "Sensor placement" in (tmp_path / "rep" / "placement.md").read_text(encoding="utf-8")


def test_train_masks_fixed_set_and_unchanged_random_stream():
    from dataclasses import replace

    from leaksense.models.mlp import MLPConfig, random_k_masks, train_masks

    cfg = MLPConfig()
    a = train_masks(50, 31, cfg, np.random.default_rng(3), 1)
    b = random_k_masks(50, 31, 1, np.random.default_rng(3))
    assert (a == b).all()  # no fixed set: the same random stream as before H4
    m = train_masks(5, 31, replace(cfg, fixed_sensors="4,26"), np.random.default_rng(3), 1)
    assert (m.sum(axis=1) == 2).all() and m[:, [4, 26]].all()
    with pytest.raises(ValueError):
        MLPConfig(fixed_sensors="3,3")


def test_cli_fixed_budget_pairs_with_the_placement_run(place_env):
    from leaksense.cli import main

    tmp_path, args = place_env
    assert main(args) == 0
    fixed = [a if a != str(tmp_path / "runs") else str(tmp_path / "fixed") for a in args]
    fixed = ["fixed-budget", *fixed[1:], "--placement-runs", str(tmp_path / "runs")]
    fixed += ["--k", "1,2"]
    assert main(fixed) == 0
    rep = json.loads((tmp_path / "rep" / "fixed_budget.json").read_text())
    assert rep["ks"] == [1, 2] and rep["h4"] in {"yes", "partly", "no"}
    # the masked column is the M11 greedy-set score on the same cases
    pl = json.loads((tmp_path / "rep" / "placement.json").read_text())
    want = pl["results"]["model"]["2"]["overall"]["greedy"]["top3"]["value"]
    assert rep["results"]["2"]["overall"]["masked"]["top3"]["value"] == pytest.approx(want)
    run = json.loads((tmp_path / "fixed" / "fixed__k2_f0.json").read_text())
    assert run["sensors_idx"] == pl["folds"]["0"]["greedy_order_idx"][:2]
    assert "H4" in (tmp_path / "rep" / "fixed_budget.md").read_text(encoding="utf-8")


def test_cli_final_report_scores_the_locked_test_once(place_env):
    from leaksense.cli import main

    tmp_path, args = place_env
    assert main(args) == 0  # M11 placement, with the final set
    base = (tmp_path / "cfg.yaml").read_text().split("model:")[0]
    cfg = tmp_path / "final.yaml"
    final = FINAL.replace("RUNS", str(tmp_path / "runs")).replace("REP", str(tmp_path / "rep"))
    cfg.write_text(base + final)
    i = args.index("--config")
    common = args[1:i] + ["--config", str(cfg)] + args[i + 2 :]
    j = common.index("--runs-dir")
    common[j + 1] = str(tmp_path / "final_runs")
    k = common.index("--reports-dir")
    common[k + 1] = str(tmp_path / "rep_final")
    run = ["final-report", *common]

    assert main(run) != 0  # scoring the locked test needs an explicit flag
    assert not (tmp_path / "final_runs").exists()
    assert main([*run, "--confirm-locked-test"]) == 0
    rep = json.loads((tmp_path / "rep_final" / "final.json").read_text())
    assert rep["mode"] == "test" and not rep["units_missing"]
    assert rep["test_set"]["n_cases"] == 1
    df = pd.read_parquet(tmp_path / "final_runs" / "ml__gat_snode_s0.parquet")
    assert set(df["case_id"]) == {"S004-pure"} or df["case_id"].str.startswith("S004").all()
    assert set(df["method"]) == {"ladder", "greedy", "coverage", "random"}
    assert rep["results"]["gat_snode"]["seeds"] == [0, 1]
    assert "H4" in rep["hypotheses"] and rep["hypotheses"]["H4"]["per_k"]
    md = (tmp_path / "rep_final" / "FINAL.md").read_text(encoding="utf-8")
    assert "locked test" in md and "Limitations" in md
    stamp = (tmp_path / "final_runs" / "nl__chance.json").stat().st_mtime_ns

    assert main(run) == 0  # all units done: only the report is rebuilt, no flag needed
    assert (tmp_path / "final_runs" / "nl__chance.json").stat().st_mtime_ns == stamp
    cfg.write_text(base + final.replace("draws: 3", "draws: 4"))
    assert main([*run, "--confirm-locked-test"]) != 0  # a changed config never re-scores test


def test_cli_final_report_stand_in_never_reads_the_test_fold(place_env):
    from leaksense.cli import main

    tmp_path, args = place_env
    assert main(args) == 0
    base = (tmp_path / "cfg.yaml").read_text().split("model:")[0]
    cfg = tmp_path / "final.yaml"
    final = FINAL.replace("RUNS", str(tmp_path / "runs")).replace("REP", str(tmp_path / "rep"))
    cfg.write_text(base + final)
    i = args.index("--config")
    common = args[1:i] + ["--config", str(cfg)] + args[i + 2 :]
    j = common.index("--runs-dir")
    common[j + 1] = str(tmp_path / "standin")
    assert main(["final-report", *common, "--stand-in-fold", "0"]) == 0
    rep = json.loads((tmp_path / "standin" / "report" / "final.json").read_text())
    assert rep["mode"] == "standin_f0" and rep["n_train_cases"] == 1
    for p in (tmp_path / "standin").glob("*.parquet"):
        assert not pd.read_parquet(p)["case_id"].str.startswith("S004").any()
    md = (tmp_path / "standin" / "report" / "FINAL.md").read_text(encoding="utf-8")
    assert "Stand-in run" in md
