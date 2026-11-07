import json
import os

import numpy as np
import pandas as pd
import pytest

from leaksense.config import DATA_DIR_VAR
from leaksense.evaluate import (
    EvalSet,
    LockedFoldError,
    bootstrap,
    bootstrap_samples,
    build_eval_set,
    case_metrics,
    evaluate,
    groups,
    point_distance,
    predict,
    random_masks,
)
from leaksense.models.baselines import Chance, MaxDH, Prior
from leaksense.network import load_nominal_network


@pytest.fixture
def net(synthetic):
    """Synthetic graph: stub 1-2 (100 m), square 2-3 (1000), 3-4 (800), 4-5 (600), 2-5 (1200)."""
    data_dir, contract = synthetic
    return load_nominal_network(data_dir, contract)


# ----------------------------------------------------------------------------- tie-aware top-k


def test_chance_is_exact():
    s = np.zeros(33)
    m = case_metrics(s, 7, np.arange(33.0), np.ones((33, 33)))
    assert m["top1"] == pytest.approx(1 / 33) and m["top3"] == pytest.approx(3 / 33)
    assert m["top5"] == pytest.approx(5 / 33)
    assert m["dist_m"] == pytest.approx(16.0)  # mean over all 33 tied classes


def test_three_way_tie():
    s = np.zeros(33)
    s[:3] = 1.0
    hops = np.zeros((33, 33))
    hops[0, 1], hops[2, 1] = 2.0, 4.0
    m = case_metrics(s, 1, np.zeros(33), hops)
    assert m["top1"] == pytest.approx(1 / 3) and m["top3"] == 1.0
    assert m["hop"] == pytest.approx((2 + 0 + 4) / 3)
    m = case_metrics(s, 5, np.zeros(33), hops)  # 3 better, 30 tied at 0
    assert m["top1"] == 0 and m["top3"] == 0 and m["top5"] == pytest.approx(2 / 30)


# ----------------------------------------------------------------------------- distances


def test_network_distance_hand_computed(net):
    # Leak on pipe 3 (3->4, 800 m) at 0.1: 80 m from node 3. Predicted pipe 5 (2-5) midpoint:
    # via node 2: 600 + 1000 + 80 = 1680; via node 5: 600 + 600 + 720 = 1920.
    assert point_distance(net, "5", 0.5, "3", 0.1) == pytest.approx(1680)
    # Same pipe: |0.5 - 0.25| * 800.
    assert point_distance(net, "3", 0.5, "3", 0.25) == pytest.approx(200)
    # Pipe 2 midpoint to pipe 4 (4->5, 600 m) at 0.9 (60 m from node 5): the shortest route
    # leaves pipe 2 by its far end, node 2: 500 + 1200 (2-5) + 60 = 1760 (via node 3: 1840).
    assert point_distance(net, "2", 0.5, "4", 0.9) == pytest.approx(1760)
    # Symmetric.
    assert point_distance(net, "4", 0.9, "2", 0.5) == pytest.approx(1760)


def test_hops_between_classes(net):
    pi = net.pipe_index
    assert net.pipe_hops[pi["2"], pi["3"]] == 1  # share node 3
    assert net.pipe_hops[pi["2"], pi["4"]] == 2


# ----------------------------------------------------------------------------- bootstrap


def test_bootstrap_resamples_scenarios_together():
    # Two scenarios with two cases each: scenario 10 all hits, scenario 20 all misses.
    values = np.zeros((4, 5))
    values[:2, :3] = 1.0
    scen = np.array([10, 10, 20, 20])
    s = bootstrap_samples(values, scen, 300, seed=1)
    assert set(np.round(s["top1"], 6)) <= {0.0, 0.5, 1.0}  # never 0.25 / 0.75
    a, b = bootstrap(values, scen, 300, 1), bootstrap(values, scen, 300, 1)
    assert a == b
    lo, hi = a["top1"]
    assert lo <= 0.5 <= hi


# ----------------------------------------------------------------------------- breakdowns


def _toy_set(n=9, folds=None, y=None) -> EvalSet:
    cases = pd.DataFrame(
        {
            "case_id": [f"c{i}" for i in range(n)],
            "scenario_id": np.arange(n) + 2,
            "fold": folds if folds is not None else np.arange(n) % 3,
            "y": y if y is not None else np.zeros(n, int),
            "leak_type": ["small", "big", "small"] * (n // 3),
            "kind": ["pure"] * n,
            "baseline_kind": ["pre"] * (n - 1) + ["post"],
            "leak_flow_lps": np.arange(1.0, n + 1),
            "snr": [np.nan] + list(np.arange(1.0, n)),
        }
    )
    return EvalSet(
        cases,
        [np.zeros((2, 4)) for _ in range(n)],
        ["2", "3", "4", "5"],
        np.zeros((n, 4)),
        np.zeros((4, 4)),
        {"leak_flow_lps": [3.0, 6.0], "snr": [3.0, 6.0]},
    )


def test_groups():
    g = groups(_toy_set())
    assert g["overall"].sum() == 9 and g["leak_type=small"].sum() == 6
    assert [g[f"leak_flow=T{i}"].sum() for i in (1, 2, 3)] == [3, 3, 3]
    assert g["baseline=post+combined"].sum() == 1 and g["pre_baseline_only"].sum() == 8
    assert sum(g[f"snr=T{i}"].sum() for i in (1, 2, 3)) == 8  # the NaN SNR case is left out


# ----------------------------------------------------------------------------- rungs


def test_maxdh_ranking(net):
    m = MaxDH(net)
    assert m.d.shape == (4, 4) and "1" not in net.leak_pipe_ids
    # Junctions 2, 3, 4, 5: largest drop at 3, then 4.
    x = np.array([[-0.1, -5.0, -2.0, 0.0]])
    sc = m.case_scores(x, np.ones(4, bool))
    classes = net.leak_pipe_ids
    order = [classes[i] for i in np.argsort(-sc, kind="stable")]
    # Keys (d3, d4, ...): pipe 3 (3-4) = (0, 0); pipe 2 (2-3) = (0, 1); pipe 4 (4-5) = (1, 0).
    assert order[:3] == ["3", "2", "4"]
    # Only junction 3 sensed: pipes 2 and 3 tie for first.
    only3 = np.array([False, True, False, False])
    sc = m.case_scores(x * only3, only3)
    y = classes.index("3")
    assert case_metrics(sc, y, np.zeros(4), np.zeros((4, 4)))["top1"] == pytest.approx(0.5)


def test_maxdh_ignores_masked_junctions(net):
    m = MaxDH(net)
    mask = np.array([True, True, False, True])
    x = np.array([[-0.1, -1.0, -50.0, -0.3]])
    a = m.case_scores(x, mask)
    x2 = x.copy()
    x2[0, 2] = 99.0
    assert np.array_equal(a, m.case_scores(x2, mask))


def test_prior_never_sees_own_fold():
    # Fold 0 cases are all class 0, fold 1 cases all class 1: a prior fitted on the other
    # fold always predicts the wrong class.
    es = _toy_set(n=6, folds=np.array([0, 0, 0, 1, 1, 1]), y=np.array([0, 0, 0, 1, 1, 1]))

    class Net4:
        leak_pipe_ids = ["2", "3", "4", "5"]

    per = predict(Prior(Net4()), es, np.ones((1, 4), bool))
    assert (per[0, :, 0] == 0).all()  # top-1 always wrong


def test_chance_through_harness():
    class Net4:
        leak_pipe_ids = ["2", "3", "4", "5"]

    res, preds = evaluate(Chance(Net4()), _toy_set(), ks=(2, 4), draws=3, n_boot=50)
    assert res["4"]["groups"]["overall"]["top1"]["value"] == pytest.approx(1 / 4)
    assert res["2"]["n_draws"] == 3 and "draw_spread" in res["2"]
    assert len(preds) == 2 * 9


def test_random_masks():
    m = random_masks(31, 4, 20, seed=0)
    assert m.shape == (20, 31) and (m.sum(axis=1) == 4).all()
    assert np.array_equal(m, random_masks(31, 4, 20, seed=0))
    assert random_masks(31, 31, 20, 0).shape == (1, 31)


# ----------------------------------------------------------------------------- harness guards


def test_test_fold_never_evaluated(net):
    cs = pd.DataFrame(
        {
            "case_id": ["a", "b"],
            "scenario_id": [2, 3],
            "pipe": ["3", "4"],
            "split_fraction": [0.5, 0.5],
            "leak_flow_lps": [1.0, 2.0],
            "snr": [1.0, 2.0],
        }
    )
    feats = {"x": np.zeros((2, 4), np.float32), "case_of_window": np.array([0, 1])}
    es = build_eval_set(cs, feats, {"test": [3], "folds": [[2]]}, net)
    assert list(es.cases["scenario_id"]) == [2]
    with pytest.raises(LockedFoldError):
        build_eval_set(cs, feats, {"test": [3], "folds": [[2, 3]]}, net)


def test_cli_eval(cli_env, capsys):
    from leaksense.cli import main

    tmp_path, common = cli_env
    assert main(["build-features", *common, "--quiet"]) == 0
    eval_args = common[:4] + common[6:]  # eval takes no --cases
    rep = tmp_path / "rep"
    for model in ("chance", "maxdh"):
        rc = main(
            [
                "eval",
                "--model",
                model,
                "--k",
                "2,4",
                "--draws",
                "3",
                "--bootstrap",
                "50",
                "--splits",
                str(tmp_path / "splits.json"),
                "--reports-dir",
                str(rep),
                *eval_args,
            ]
        )
        assert rc == 0
    r = json.loads((rep / "chance.json").read_text())
    assert r["results"]["4"]["groups"]["overall"]["top1"]["value"] == pytest.approx(1 / 4)
    assert r["n_cases"] == 2  # scenario 4 is the test fold
    assert r["provenance"]["git"]["commit"] and r["provenance"]["splits_sha256"]
    g = json.loads((rep / "maxdh.json").read_text())["results"]["4"]["groups"]
    assert {"overall", "leak_type=small", "kind=pure", "baseline=pre"} <= set(g)
    assert "## Breakdown at k = 4" in (rep / "maxdh.md").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- real data


@pytest.mark.data
def test_real_chance_and_breakdowns():
    from leaksense.config import get_artifacts_dir
    from leaksense.runconfig import load_run_config

    assert os.environ.get(DATA_DIR_VAR)
    fdir = get_artifacts_dir() / "features" / load_run_config("configs/full.yaml").hash()
    with np.load(fdir / "features.npz") as z:
        feats = {k: z[k] for k in z.files}
    cs = pd.read_parquet(fdir / "case_stats.parquet")
    splits = json.loads(open("splits/splits_v1.json").read())
    net = load_nominal_network(os.environ[DATA_DIR_VAR])
    es = build_eval_set(cs, feats, splits, net)
    assert len(es.cases) == 419 and not es.cases["scenario_id"].isin(splits["test"]).any()
    res, _ = evaluate(Chance(net), es, ks=(31,), n_boot=20)
    assert res["31"]["groups"]["overall"]["top1"]["value"] == pytest.approx(1 / 33)
    res, _ = evaluate(MaxDH(net), es, ks=(31,), n_boot=20)
    g = set(res["31"]["groups"])
    for family in ("overall", "leak_type=", "kind=", "baseline=", "leak_flow=T", "snr=T"):
        assert any(name.startswith(family) for name in g), family
