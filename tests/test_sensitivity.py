import json
import os

import numpy as np
import pandas as pd
import pytest

from leaksense import sensitivity as sens
from leaksense.config import DATA_DIR_VAR
from leaksense.evaluate import case_metrics
from leaksense.models.baselines import SigMatch
from leaksense.network import hazen_williams_resistance

ALPHAS = {"low": 0.3, "mid": 0.5, "high": 0.8}
SIZES = {"small": 100.0, "big": 500.0}  # L/s; synthetic total base demand is 4 x 50 = 200 L/s
PIPES = ["2", "3", "4", "5"]


@pytest.fixture
def narrow(synthetic, tmp_path):
    """Synthetic nominal network with pipe 1 (reservoir -> node 2) narrowed to 300 mm."""
    data_dir, contract = synthetic
    text = (data_dir / "Hanoi.inp").read_text()
    assert " 1 1 2 100.0 1016.0 130.0" in text
    inp = tmp_path / "narrow.inp"
    inp.write_text(text.replace(" 1 1 2 100.0 1016.0 130.0", " 1 1 2 100.0 300.0 130.0"))
    return inp, contract


@pytest.fixture
def sim(narrow):
    inp, contract = narrow
    return sens.simulate(inp, contract, PIPES, ALPHAS, SIZES), contract


def test_units_against_hand_hazen_williams(sim):
    """Node 2 hangs off the reservoir through pipe 1 only: its drop is r1 [(Q0+q)^n - Q0^n]."""
    res, _ = sim
    S = res["S"]
    r1 = hazen_williams_resistance(100.0, 300.0, 130.0)
    for li, lv in enumerate(sens.LEVELS):
        q0 = ALPHAS[lv] * 200.0 / 1000.0  # m^3/s
        for si, sz in enumerate(sens.SIZES):
            q = SIZES[sz] / 1000.0
            expected = -r1 * ((q0 + q) ** 1.852 - q0**1.852)
            for pi in range(len(PIPES)):
                assert S[0, pi, si, li] == pytest.approx(expected, rel=0.01)


def test_values(sim):
    res, _ = sim
    S = res["S"]
    assert S.shape == (4, 4, 2, 3)
    assert (S <= 1e-6).all()
    assert (S[:, :, 1] <= S[:, :, 0] + 1e-6).all()  # big drops at least as much as small
    flat = S.reshape(4, 4, -1)
    for a in range(4):
        for b in range(a + 1, 4):
            assert np.abs(flat[:, a] - flat[:, b]).max() > 1e-3


def test_stability_and_save(sim, tmp_path):
    res, contract = sim
    st = sens.level_stability(res["S"], PIPES)
    assert set(st["per_pipe_min_cosine"]) == set(PIPES) and 0 < st["min_cosine"] <= 1
    path = tmp_path / "S.npz"
    sens.save(path, res["S"], contract, PIPES, ALPHAS, SIZES, "f" * 64)
    z = sens.load(path)
    assert np.array_equal(z["S"], res["S"]) and list(z["pipes"]) == PIPES
    assert list(z["levels"]) == ["low", "mid", "high"] and str(z["manifest"]) == "f" * 64


def test_demand_levels_read_junction_columns_only(synthetic, monkeypatch):
    data_dir, contract = synthetic
    seen = []
    real = pd.read_csv

    def spy(path, *a, **kw):
        seen.append((os.path.basename(str(path)), kw.get("usecols")))
        return real(path, *a, **kw)

    monkeypatch.setattr(pd, "read_csv", spy)
    lv = sens.demand_levels(data_dir, contract, [1, 2], nominal_total_lps=200.0)
    assert {name for name, _ in seen} == {"Node_demands.csv"}
    for _, uc in seen:
        assert callable(uc) and uc("2") and not uc("1") and not uc("leak_node0")
    monkeypatch.setattr(pd, "read_csv", real)
    tot = np.concatenate(
        [
            pd.read_csv(data_dir / f"Scenario-{s}" / "Node_demands.csv")[list(contract.junctions)]
            .sum(axis=1)
            .to_numpy()
            for s in (1, 2)
        ]
    )
    assert lv["alpha"]["mid"] == pytest.approx(np.percentile(tot, 50) / 200.0)
    assert lv["alpha"]["low"] < lv["alpha"]["mid"] < lv["alpha"]["high"]


def test_leak_sizes_ignore_test_fold():
    cases = pd.DataFrame(
        {
            "scenario_id": [2, 3, 4, 5],
            "leak_type": ["small", "small", "big", "big"],
            "leak_flow_lps": [10.0, 20.0, 900.0, 5000.0],
        }
    )
    assert sens.leak_sizes(cases, {5}) == {"small": 15.0, "big": 900.0}


# ----------------------------------------------------------------------------- sigmatch


class Net4:
    leak_pipe_ids = PIPES
    sensor_ids = ["2", "3", "4", "5"]


def _toy_sensitivity(seed=0):
    rng = np.random.default_rng(seed)
    S = -np.abs(rng.normal(1, 0.5, (4, 4, 2, 3)))
    return {
        "S": S,
        "pipes": np.array(PIPES),
        "junctions": np.array(Net4.sensor_ids),
        "levels": np.array(sens.LEVELS),
    }


def test_sigmatch_matches_its_own_column():
    sd = _toy_sensitivity()
    m = SigMatch(Net4(), sd)
    mid = sens.LEVELS.index("mid")
    full = np.ones(4, bool)
    for p in range(4):
        x = sd["S"][:, p, 1, mid][None, :]
        assert int(np.argmax(m.case_scores(x, full))) == p
        assert np.array_equal(m.case_scores(3.7 * x, full), m.case_scores(x, full))


def test_sigmatch_masking_and_single_sensor():
    sd = _toy_sensitivity(1)
    m = SigMatch(Net4(), sd)
    mask = np.array([True, False, True, True])
    x = sd["S"][:, 2, 0, 1][None, :].copy()
    a = m.case_scores(x * mask, mask)
    x[0, 1] = 999.0
    assert np.array_equal(a, m.case_scores(x * mask, mask))
    one = np.array([False, False, True, False])
    sc = m.case_scores(x * one, one)  # all negative columns: cosine is 1 everywhere
    assert np.allclose(sc, 1.0)
    assert case_metrics(sc, 0, np.zeros(4), np.zeros((4, 4)))["top1"] == pytest.approx(0.25)


def test_sigmatch_rejects_mismatched_sensitivity():
    sd = _toy_sensitivity()
    sd["pipes"] = np.array(["2", "3", "4", "9"])
    with pytest.raises(ValueError, match="pipes"):
        SigMatch(Net4(), sd)


# ----------------------------------------------------------------------------- CLI


def test_cli_build_sensitivity_and_eval(cli_env):
    from leaksense.cli import main

    tmp_path, common = cli_env
    out = tmp_path / "rep" / "sens"
    args = [
        "build-sensitivity",
        *common,
        "--splits",
        str(tmp_path / "splits.json"),
        "--out",
        str(out),
        "--quiet",
    ]
    assert main(args) == 0
    r = json.loads(out.with_suffix(".json").read_text())
    assert r["shape"]["pipes"] == 4 and r["demand_levels"]["n_scenarios"] == 3  # 1 + folds 2, 3
    assert r["git"]["commit"] and r["data_manifest_sha256"]
    assert (tmp_path / "art" / r["artifact"]).is_file()
    assert main(["build-features", *common, "--quiet"]) == 0
    eval_args = common[:4] + common[6:]
    assert (
        main(
            [
                "eval",
                "--model",
                "sigmatch",
                "--k",
                "4",
                "--bootstrap",
                "20",
                "--splits",
                str(tmp_path / "splits.json"),
                "--reports-dir",
                str(tmp_path / "rep"),
                *eval_args,
            ]
        )
        == 0
    )
    assert (tmp_path / "rep" / "sigmatch.json").is_file()


# ----------------------------------------------------------------------------- real data


@pytest.mark.data
def test_real_sensitivity_scenario_34():
    from leaksense.config import get_artifacts_dir
    from leaksense.data.validate import manifest
    from leaksense.runconfig import load_run_config

    data_dir = os.environ[DATA_DIR_VAR]
    from pathlib import Path

    sha = manifest(Path(data_dir))["sha256"]
    z = sens.load(sens.sensitivity_path(get_artifacts_dir(), sha))
    S, pipes, J = z["S"], list(z["pipes"]), list(z["junctions"])
    assert S.shape == (31, 33, 2, 3) and (S <= 1e-6).all()
    big_mid = S[:, :, sens.SIZES.index("big"), sens.LEVELS.index("mid")]
    p22 = pipes.index("22")
    assert J[int(np.argmin(big_mid[:, p22]))] in ("21", "22")

    fdir = get_artifacts_dir() / "features" / load_run_config("configs/full.yaml").hash()
    cs = pd.read_parquet(fdir / "case_stats.parquet")
    with np.load(fdir / "features.npz") as f:
        x, cow = f["x"], f["case_of_window"]
    row = int(np.flatnonzero(cs["case_id"].to_numpy() == "S034-pure")[0])
    dh = x[cow == row].mean(axis=0)
    cos = [sens.cosine(dh, big_mid[:, p]) for p in range(len(pipes))]
    assert pipes[int(np.argmax(cos))] == "22"
