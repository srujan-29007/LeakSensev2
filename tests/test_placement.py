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


PLACE = """model:
  kind: gat
  max_epochs: 2
  hidden: 8
  heads: 2
  layers: 2
  node_embed: 0
  physics: node
  val_frac: 0.0
  num_threads: 1
placement:
  model_label: toy
  k: [1, 2]
  random: 3
"""


@pytest.fixture
def place_env(cli_env):
    from leaksense.cli import main

    tmp_path, common = cli_env
    assert main(["build-features", *common, "--quiet"]) == 0
    sens = ["build-sensitivity", *common, "--splits", str(tmp_path / "splits.json")]
    assert main([*sens, "--out", str(tmp_path / "sens"), "--quiet"]) == 0
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(cfg.read_text() + PLACE)
    args = ["place", "--bootstrap", "20", "--splits", str(tmp_path / "splits.json")]
    args += ["--reports-dir", str(tmp_path / "rep"), "--runs-dir", str(tmp_path / "runs")]
    args += [*(common[:4] + common[6:])]
    return tmp_path, args


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
