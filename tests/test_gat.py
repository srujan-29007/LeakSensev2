import json

import numpy as np
import pandas as pd
import pytest
import torch

from leaksense.models.gat import GATConfig, MaskedGAT, build_graph, standardised_pipe_features
from leaksense.models.mlp import encode, parse_config, random_k_masks
from leaksense.network import load_nominal_network


@pytest.fixture
def net(synthetic):
    """Synthetic graph: reservoir 1, junctions 2-5, pipes 1-5 (pipe 1 never leaks)."""
    data_dir, contract = synthetic
    return load_nominal_network(data_dir, contract)


def _cases(net, ys, n_windows=6, noise=0.05, seed=0):
    """One case per label: each class has its own ΔH pattern over the junctions plus noise."""
    j, c = len(net.sensor_ids), len(net.leak_pipe_ids)
    proto = -np.abs(np.random.default_rng(123).normal(0, 3, size=(c, j)))
    rng = np.random.default_rng(seed)
    cases = pd.DataFrame(
        {"case_id": [f"c{i}" for i in range(len(ys))], "scenario_id": np.arange(len(ys)) + 2}
    )
    cases["y"] = np.asarray(ys)
    windows = [proto[y] + rng.normal(0, noise, size=(n_windows, j)) for y in ys]
    return cases, windows


def _gat(net, **kw):
    small = {"hidden": 16, "heads": 2, "layers": 2, "node_embed": 4, "val_frac": 0.0}
    return MaskedGAT(net, {**small, "num_threads": 1, **kw})


def test_graph_tensors(net):
    g = build_graph(net)
    assert g.n_nodes == 5 and g.edge_index.shape == (2, 10) and g.edge_attr.shape == (10, 4)
    assert len(g.pipe_u) == len(net.leak_pipe_ids) == 4  # pipe 1 excluded
    assert 0 not in g.sensor_pos.tolist()  # reservoir (position 0) is never a sensor
    f = standardised_pipe_features(net)
    assert np.isfinite(f).all() and np.allclose(f[:, 2], 0)  # constant roughness -> 0
    assert np.allclose(f[:, 0].mean(), 0) and np.allclose(f[:, 0].std(), 1)


def test_output_shapes(net):
    cases, windows = _cases(net, [0, 1, 2, 3])
    m = _gat(net, max_epochs=2)
    m.fit(cases, windows)
    s = m.batch_scores(windows, np.ones(4, bool))
    assert s.shape == (4, 4) and np.isfinite(s).all()
    assert np.allclose(np.logaddexp.reduce(s, axis=1) <= 0, True)  # mean of log-probs
    single = np.stack([m.case_scores(w, np.ones(4, bool)) for w in windows])
    assert np.allclose(s, single, atol=1e-5)


@pytest.mark.parametrize("transform", ["scale_magnitude", "signed_log1p", "none"])
def test_masked_junction_cannot_change_output(net, transform):
    cases, windows = _cases(net, [0, 1, 2, 3])
    m = _gat(net, max_epochs=3, transform=transform)
    m.fit(cases, windows)
    mask = np.array([True, False, True, False])
    x = windows[0].copy()
    a = m.case_scores(x, mask)
    x[:, ~mask] = 1e3  # garbage in every unsensed junction
    assert np.allclose(a, m.case_scores(x, mask))


def test_reservoir_input_is_zero(net):
    m = _gat(net)
    g = m.graph
    gnet = m._make_net(m.base, 2 * 4 + 1)
    seen = {}
    gnet.inp.register_forward_pre_hook(lambda mod, inp: seen.update(x=inp[0]))
    x = encode(np.full((3, 4), -5.0), np.ones((3, 4), bool), "scale_magnitude")
    gnet(torch.from_numpy(x.astype(np.float32)))
    res = [i for i in range(g.n_nodes) if i not in g.sensor_pos.tolist()]
    assert res == [0] and (seen["x"][:, 0, :2] == 0).all()


def test_overfits_five_cases(net):
    cases, windows = _cases(net, [0, 1, 2, 3, 1])
    m = _gat(net, k_min=4, dropout=0.0, max_epochs=150, lr=0.005)
    m.fit(cases, windows)
    pred = m.batch_scores(windows, np.ones(4, bool)).argmax(axis=1)
    assert (pred == cases["y"].to_numpy()).all()


def test_fit_is_deterministic(net):
    cases, windows = _cases(net, [0, 1, 2, 3])
    a, b = _gat(net, max_epochs=4), _gat(net, max_epochs=4)
    a.fit(cases, windows)
    b.fit(cases, windows)
    mask = random_k_masks(1, 4, 2, np.random.default_rng(0))[0]
    assert np.array_equal(a.batch_scores(windows, mask), b.batch_scores(windows, mask))


def test_config():
    base, grid = parse_config({"kind": "gat", "hidden": [16, 32], "heads": 4}, GATConfig)
    assert base.hidden == 16 and grid == {"hidden": [16, 32]} and base.layers == 4
    with pytest.raises(ValueError):
        GATConfig(hidden=30, heads=4)
    with pytest.raises(ValueError, match="unknown"):
        parse_config({"hedas": 2}, GATConfig)


def test_cli_train_gat(cli_env):
    from leaksense.cli import main

    tmp_path, common = cli_env
    assert main(["build-features", *common, "--quiet"]) == 0
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        cfg.read_text() + "model:\n  kind: gat\n  max_epochs: 3\n  hidden: 8\n  heads: 2\n"
        "  layers: 2\n  val_frac: 0.0\n  num_threads: 1\n"
    )
    args = common[:4] + common[6:]
    rep = tmp_path / "rep"
    base = ["train", "--k", "2,4", "--draws", "2", "--bootstrap", "20"]
    base += ["--splits", str(tmp_path / "splits.json"), "--reports-dir", str(rep), *args]
    assert main([*base, "--fold", "1"]) == 0
    r = json.loads((rep / "gat_fold1.json").read_text())
    assert r["model"] == "gat" and r["training"]["eval_folds"] == [1]
    assert r["training"]["base"]["heads"] == 2
