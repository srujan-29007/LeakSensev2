import json

import numpy as np
import pandas as pd
import pytest

from leaksense.models.mlp import (
    MaskedMLP,
    class_weights,
    encode,
    inner_split,
    parse_config,
    random_k_masks,
)

J, C = 31, 33


class FakeNet:
    leak_pipe_ids = [str(p) for p in range(2, 2 + C)]


def _cases(ys, n_windows=6, noise=0.05, seed=0):
    """One case per label in ``ys``: each class has its own fixed ΔH pattern plus noise."""
    rng = np.random.default_rng(seed)
    proto = -np.abs(np.random.default_rng(123).normal(0, 3, size=(C, J)))
    cases = pd.DataFrame(
        {"case_id": [f"c{i}" for i in range(len(ys))], "scenario_id": np.arange(len(ys)) + 2}
    )
    cases["y"] = np.asarray(ys)
    windows = [proto[y] + rng.normal(0, noise, size=(n_windows, J)) for y in ys]
    return cases, windows


def test_overfits_five_cases():
    cases, windows = _cases([0, 7, 13, 21, 32])
    m = MaskedMLP(
        FakeNet(),
        {"k_min": 31, "dropout": 0.0, "max_epochs": 200, "val_frac": 0.0, "lr": 0.003},
    )
    m.fit(cases, windows)
    full = np.ones(J, bool)
    pred = m.batch_scores(windows, full).argmax(axis=1)
    assert (pred == cases["y"].to_numpy()).all()
    assert m.history[0]["n_inner_val_cases"] == 0


def test_masked_junction_cannot_change_output():
    cases, windows = _cases([0, 1, 2, 3, 4])
    m = MaskedMLP(FakeNet(), {"max_epochs": 3, "val_frac": 0.0})
    m.fit(cases, windows)
    mask = np.zeros(J, bool)
    mask[[0, 5, 9]] = True
    x = windows[0].copy()
    a = m.case_scores(x, mask)
    x[:, ~mask] = 1e3  # garbage in every unsensed junction
    assert np.allclose(a, m.case_scores(x, mask))
    assert a.shape == (C,)


def test_batch_scores_mean_log_probs():
    cases, windows = _cases([0, 1, 2], n_windows=4)
    m = MaskedMLP(FakeNet(), {"max_epochs": 2, "val_frac": 0.0})
    m.fit(cases, windows)
    mask = np.ones(J, bool)
    b = m.batch_scores(windows, mask)
    single = np.stack([m.case_scores(w, mask) for w in windows])
    assert np.allclose(b, single, atol=1e-5)
    assert np.allclose(np.logaddexp.reduce(b, axis=1) <= 0, True)  # mean of log-probs


def test_fit_is_deterministic():
    cases, windows = _cases([0, 1, 2, 3])
    a, b = (MaskedMLP(FakeNet(), {"max_epochs": 5, "val_frac": 0.0}) for _ in range(2))
    a.fit(cases, windows)
    b.fit(cases, windows)
    mask = random_k_masks(1, J, 4, np.random.default_rng(0))[0]
    assert np.array_equal(a.batch_scores(windows, mask), b.batch_scores(windows, mask))


def test_encode_and_masks():
    x = -np.arange(1, J + 1, dtype=float)[None]
    m = np.zeros((1, J), bool)
    m[0, :3] = True
    e = encode(x, m, "none")
    assert e.shape == (1, 2 * J) and (e[0, 3:J] == 0).all() and e[0, J:].sum() == 3
    assert encode(x, m, "scale_magnitude").shape == (1, 2 * J + 1)
    masks = random_k_masks(5000, J, 1, np.random.default_rng(0))
    k = masks.sum(axis=1)
    assert k.min() == 1 and k.max() == J and abs(k.mean() - 16) < 0.5
    assert (random_k_masks(10, J, 31, np.random.default_rng(0)).sum(axis=1) == J).all()


def test_class_weights():
    y = np.array([0, 0, 0, 1])
    w = class_weights(y, 3, "inverse")
    assert w[2] == 0 and w[1] == pytest.approx(3 * w[0])
    assert w[y].mean() == pytest.approx(1.0)  # normalised over the training cases
    assert np.allclose(class_weights(y, 3, "none")[:2], 1.0)
    s = class_weights(y, 3, "sqrt_inverse")
    assert s[1] == pytest.approx(np.sqrt(3) * s[0])


def test_inner_split_grouped_and_spares_rare_pipes():
    ys = [0] * 20 + [1] * 3
    cases, _ = _cases(ys)
    cases.loc[1, "scenario_id"] = cases.loc[0, "scenario_id"]  # two cases of one scenario
    v = inner_split(cases, 0.15, seed=0)
    assert not v[20:].any()  # pipe with 3 cases stays in training
    assert 2 <= v[:20].sum() <= 4
    assert v[0] == v[1]  # scenario kept together
    assert np.array_equal(v, inner_split(cases, 0.15, seed=0))


def test_grid_choice_recorded():
    cases, windows = _cases(list(range(10)) * 8, n_windows=3, noise=0.3)
    m = MaskedMLP(
        FakeNet(),
        {"transform": ["none", "signed_log1p"], "max_epochs": 15, "patience": 5, "val_frac": 0.2},
    )
    m.fit(cases, windows)
    h = m.history[0]
    assert h["n_inner_val_cases"] == 10  # 1 of 8 per pipe (floor(0.2 * 8))
    assert len(h["trials"]) == 2 and h["chosen"]["transform"] in ("none", "signed_log1p")
    assert 1 <= h["epochs"] <= 15
    assert m.describe()["grid"] == {"transform": ["none", "signed_log1p"]}


def test_parse_config_rejects_unknown_and_bad_values():
    with pytest.raises(ValueError, match="unknown"):
        parse_config({"kind": "mlp", "hiden": 3})
    with pytest.raises(ValueError):
        parse_config({"transform": ["none", "log"]})


def test_cli_train(cli_env, capsys):
    from leaksense.cli import main

    tmp_path, common = cli_env
    assert main(["build-features", *common, "--quiet"]) == 0
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        cfg.read_text() + "model:\n  kind: mlp\n  max_epochs: 5\n  hidden: 16\n  val_frac: 0.0\n"
    )
    args = common[:4] + common[6:]  # train takes no --cases
    rep = tmp_path / "rep"
    base = ["train", "--k", "2,4", "--draws", "2", "--bootstrap", "20"]
    base += ["--splits", str(tmp_path / "splits.json"), "--reports-dir", str(rep), *args]
    assert main([*base, "--fold", "1"]) == 0
    r = json.loads((rep / "mlp_fold1.json").read_text())
    assert r["n_cases"] == 1 and r["training"]["eval_folds"] == [1]
    assert main(base) == 0
    r = json.loads((rep / "mlp.json").read_text())
    assert r["n_cases"] == 2 and len(r["training"]["folds"]) == 2
    assert r["command"] == "leaksense train" and r["provenance"]["config"]["model"]["kind"] == "mlp"
    md = (rep / "mlp.md").read_text(encoding="utf-8")
    assert "## Training (per fold)" in md and "`leaksense train`" in md
    capsys.readouterr()
    assert main([*base, "--fold", "7"]) != 0
    assert "--fold must be one of" in capsys.readouterr().err


# ----------------------------------------------------------------------------- GBM


def _gbm(**kw):
    from leaksense.models.gbm import MaskedGBM

    return MaskedGBM(FakeNet(), {"val_frac": 0.0, "num_threads": 1, **kw})


def test_gbm_overfits_five_cases():
    cases, windows = _cases([0, 7, 13, 21, 32])
    m = _gbm(k_min=31, n_masks=1, min_child_samples=1, max_rounds=60, bagging_fraction=1.0)
    m.fit(cases, windows)
    pred = m.batch_scores(windows, np.ones(J, bool)).argmax(axis=1)
    assert (pred == cases["y"].to_numpy()).all()


def test_gbm_masking_determinism_and_batch():
    cases, windows = _cases([0, 1, 2, 3, 4])
    a, b = _gbm(max_rounds=10, min_child_samples=2), _gbm(max_rounds=10, min_child_samples=2)
    a.fit(cases, windows)
    b.fit(cases, windows)
    mask = np.zeros(J, bool)
    mask[[1, 4, 30]] = True
    s = a.batch_scores(windows, mask)
    assert s.shape == (5, C) and np.array_equal(s, b.batch_scores(windows, mask))
    x = windows[0].copy()
    x[:, ~mask] = -1e3
    assert np.allclose(a.case_scores(x, mask), s[0])


def test_gbm_early_stopping_and_grid():
    cases, windows = _cases(list(range(10)) * 8, n_windows=3, noise=0.3)
    m = _gbm(
        transform=["none", "scale_magnitude"],
        val_frac=0.2,
        max_rounds=40,
        early_stopping=5,
        n_masks=2,
        min_child_samples=5,
    )
    m.fit(cases, windows)
    h = m.history[0]
    assert len(h["trials"]) == 2 and 1 <= h["epochs"] <= 40 and h["n_inner_val_cases"] == 10


def test_cli_train_gbm(cli_env):
    from leaksense.cli import main

    tmp_path, common = cli_env
    assert main(["build-features", *common, "--quiet"]) == 0
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        cfg.read_text() + "model:\n  kind: gbm\n  max_rounds: 5\n  val_frac: 0.0\n"
        "  min_child_samples: 1\n  num_threads: 1\n"
    )
    args = common[:4] + common[6:]
    rep = tmp_path / "rep"
    rc = main(
        [
            "train",
            "--k",
            "2,4",
            "--draws",
            "2",
            "--bootstrap",
            "20",
            "--splits",
            str(tmp_path / "splits.json"),
            "--reports-dir",
            str(rep),
            *args,
        ]
    )
    assert rc == 0
    r = json.loads((rep / "gbm.json").read_text())
    assert r["model"] == "gbm" and len(r["training"]["folds"]) == 2
