import builtins
import os
import re
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from leaksense.config import DATA_DIR_VAR
from leaksense.data import features as F
from leaksense.data.baseline_study import choose, choose_lag
from leaksense.data.index import IndexConfig, build_index
from leaksense.runconfig import FeatureConfig, RunConfig, load_run_config
from leaksense.transforms import apply, scale_magnitude, signed_log1p

from synthetic_data import LeakSpec, ScenarioSpec, contract_for, make_dataset

N_ROWS = 10 * 48
SPECS = {
    1: ScenarioSpec(),
    2: ScenarioSpec([LeakSpec("3", 300, 400, "big", 0.5)]),  # pre baseline (6.25 days)
    3: ScenarioSpec([LeakSpec("2", 20, 200, "small", 0.5)]),  # 20 pre steps -> post baseline
    4: ScenarioSpec([LeakSpec("4", 260, 470), LeakSpec("5", 330, 400)]),  # solo 260..329
    5: ScenarioSpec([LeakSpec("5", 150, 160, "small", 0.5)]),  # 11 steps: shorter than a window
}
ICFG = IndexConfig(min_solo_steps=48, min_baseline_steps=96, short_case_steps=48)
FCFG = FeatureConfig(
    slot="tod",
    baseline_days=4,
    estimator="mean",
    window=12,
    stride=6,
    max_windows_per_case=8,
    snr_holdout_days=1.0,
)


@pytest.fixture
def ds(tmp_path):
    data_dir = make_dataset(tmp_path, SPECS, n_rows=N_ROWS, daily_amp=2.0)
    contract = contract_for(SPECS, n_rows=N_ROWS)
    cases = build_index(data_dir, contract, ICFG).cases
    cache = F.PressureCache(data_dir, tmp_path / "art", "f" * 64, contract)
    slots = F.slot_index(contract, "tod"), F.slot_index(contract, "tod_weekday")
    return data_dir, contract, cases.set_index("scenario_id", drop=False), cache, slots


def test_index_kinds(ds):
    _, _, cases, _, _ = ds
    assert cases.loc[2, "baseline_kind"] == "pre"
    assert cases.loc[3, "baseline_kind"] == "post"
    assert cases.loc[4, "kind"] == "solo" and cases.loc[4, "end"] == 329


def test_load_pressures_ignores_leak_columns(ds):
    data_dir, contract, _, _, _ = ds
    f = data_dir / "Scenario-2" / "Node_pressures.csv"
    lines = f.read_text().splitlines()
    header = lines[0].split(",")
    k = header.index("leak_node0")
    rows = [ln.split(",") for ln in lines[1:]]
    for r in rows:
        r[k] = "garbage"  # would fail float parsing if the column were read
    f.write_text("\n".join([lines[0]] + [",".join(r) for r in rows]) + "\n")
    p = F.load_pressures(data_dir / "Scenario-2", contract)
    assert p.shape == (N_ROWS, 4) and p.dtype == np.float32 and np.isfinite(p).all()


def test_only_pressures_and_labels_are_read(ds, monkeypatch):
    _, _, cases, cache, _ = ds
    seen, usecols = [], []
    real_read_csv, real_open = pd.read_csv, builtins.open

    def spy_read_csv(path, *a, **kw):
        seen.append(os.path.basename(str(path)))
        usecols.append(kw.get("usecols"))
        return real_read_csv(path, *a, **kw)

    def guard_open(file, *a, **kw):
        if re.search(r"(Node_demands|Link_flows)\.csv$", str(file)):
            raise AssertionError(f"forbidden file opened: {file}")
        return real_open(file, *a, **kw)

    monkeypatch.setattr(pd, "read_csv", spy_read_csv)
    monkeypatch.setattr(builtins, "open", guard_open)
    F.build_features(cases, cache, FCFG)
    assert set(seen) == {"Node_pressures.csv", "Labels.csv"}
    for name, uc in zip(seen, usecols, strict=True):
        if name == "Node_pressures.csv":
            assert callable(uc) and not uc("leak_node0") and not uc("1") and uc("2")


def _leak_free_case(start, end, n_pre):
    return pd.Series(
        {
            "case_id": "pseudo",
            "start": start,
            "end": end,
            "baseline_kind": "pre",
            "baseline_segments": [[start - n_pre, start - 1]],
        }
    )


def test_dh_is_zero_on_leak_free_window(ds):
    _, _, _, cache, (st, sw) = ds
    p = cache.get(1)  # Scenario-1 has no leak; daily cycle of 2 m plus 0.3 m noise
    case = _leak_free_case(300, 400, 300)
    cd = F.case_delta_h(p, case, FCFG, np.zeros(N_ROWS, bool), st, sw)
    n = cd.dh.shape[0]
    assert abs(cd.dh.mean()) < 3 * 0.3 / np.sqrt(n * 4)
    assert cd.dh.std() < 0.5  # the 2 m daily cycle is removed by slot matching
    raw = p[300:401] - p[cd.rows_b].mean(axis=0)
    assert raw.std() > 1.0  # ...and would not be without it


def test_poison_rows_outside_baseline(ds):
    """The profile depends only on baseline rows: poison everything else, nothing changes."""
    _, _, cases, cache, (st, sw) = ds
    for sid in (2, 3, 4):  # pre, post, solo
        case = cases.loc[sid]
        p, labels = cache.get(sid), cache.labels(sid)
        rows_b = F.baseline_rows(case, FCFG.baseline_days, 0, labels)
        prof, _ = F.slot_profile(p[rows_b], rows_b, "tod", "mean", st, sw)
        poisoned = np.full_like(p, np.nan)
        poisoned[rows_b] = p[rows_b]
        prof2, _ = F.slot_profile(poisoned[rows_b], rows_b, "tod", "mean", st, sw)
        assert np.isfinite(prof2).all() and np.array_equal(prof, prof2)
        # ΔH on the case window is unchanged when every non-baseline, non-case row is poisoned
        start, end = int(case["start"]), int(case["end"])
        keep = np.zeros(N_ROWS, bool)
        keep[rows_b] = True
        keep[start : end + 1] = True
        p3 = np.where(keep[:, None], p, np.float32(1e6))
        a = F.case_delta_h(p, case, FCFG, labels, st, sw).dh
        b = F.case_delta_h(p3, case, FCFG, labels, st, sw).dh
        assert np.array_equal(a, b)
        if case["baseline_kind"] == "pre":
            assert rows_b.max() < start
        else:
            assert rows_b.min() > end
        assert not labels[rows_b].any()


def test_baseline_rows_nearest_days(ds):
    _, _, cases, cache, _ = ds
    pre, post = cases.loc[2], cases.loc[3]
    rb = F.baseline_rows(pre, 4, 0, cache.labels(2))
    assert (rb.min(), rb.max()) == (300 - 4 * 48, 299)
    rb = F.baseline_rows(post, 2, 0, cache.labels(3))
    assert (rb.min(), rb.max()) == (201, 201 + 2 * 48 - 1)
    rb = F.baseline_rows(pre, 4, 5, cache.labels(2))
    assert rb.max() == 294  # pre_guard_steps
    combined = pd.Series(
        {
            "case_id": "c",
            "start": 100,
            "end": 200,
            "baseline_kind": "combined",
            "baseline_segments": [[0, 99], [201, 479]],
        }
    )
    rb = F.baseline_rows(combined, 1, 0, np.zeros(N_ROWS, bool))
    assert set(rb) == set(range(76, 100)) | set(range(201, 225))


def test_baseline_rows_asserts(ds):
    _, _, cases, cache, _ = ds
    case = cases.loc[2].copy()
    labels = cache.labels(2)
    bad = case.copy()
    bad["baseline_segments"] = [[0, 310]]  # reaches into the case
    with pytest.raises(F.LeakageError):
        F.baseline_rows(bad, None, 0, labels)
    lab = labels.copy()
    lab[290] = True  # a leak-active row inside the baseline
    with pytest.raises(F.LeakageError, match="leak-active"):
        F.baseline_rows(case, None, 0, lab)


def test_weekday_fallback(ds):
    _, _, _, cache, (st, sw) = ds
    p = cache.get(1)
    case = _leak_free_case(300, 400, 300)
    cfg = replace(FCFG, slot="tod_weekday", baseline_days=4)
    cd = F.case_delta_h(p, case, cfg, np.zeros(N_ROWS, bool), st, sw)
    assert cd.n_fallback_slots == 3 * 48  # 4 days cover 4 of 7 weekdays
    assert np.isfinite(cd.dh).all()


def test_window_selection():
    cfg = replace(FCFG, window=12, stride=6, max_lag_days=1.0, max_windows_per_case=100)
    pre = F.window_starts(200, "pre", cfg)
    post = F.window_starts(200, "post", cfg)
    assert pre.min() == 0 and (pre + 12 <= 48).all()
    assert post.max() == 200 - 12 and (post >= 200 - 48).all()
    both = F.window_starts(200, "combined", cfg)
    assert set(both) == set(pre) | set(post)
    thin = F.window_starts(500, "pre", replace(cfg, max_lag_days=None, max_windows_per_case=5))
    assert len(thin) == 5 and thin[0] == 0 and thin[-1] == (500 - 12) // 6 * 6


def test_build_features(ds):
    _, _, cases, cache, _ = ds
    fs = F.build_features(cases, cache, FCFG)
    cs = fs.cases
    assert len(cs) == len(cases) == 4
    w = pd.Series(fs.weight).groupby(fs.case_of_window).sum()
    assert np.allclose(w, 1.0)
    short = cs.set_index("scenario_id").loc[5]
    assert short["short_window"] and short["n_windows"] == 1
    assert fs.x.shape == (len(fs.case_of_window), 4) and fs.x.dtype == np.float32
    big = cs.set_index("scenario_id").loc[2]
    assert big["max_abs_mean_dh"] > 4  # synthetic big leak: -650 / 100 m
    assert (fs.window_start >= cs["start"].to_numpy()[fs.case_of_window]).all()


def test_run_config(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text(
        "name: t\nseed: 1\nn_scenarios: 3\ninclude_scenarios: [7]\n"
        "features:\n  slot: tod_weekday\n  window: 24\n"
    )
    cfg = load_run_config(f)
    assert cfg.features.slot == "tod_weekday" and cfg.features.window == 24
    sel = cfg.select_scenarios(list(range(2, 20)))
    assert 7 in sel and len(sel) == 3 and sel == cfg.select_scenarios(list(range(2, 20)))
    assert cfg.hash() != RunConfig().hash()
    f.write_text("features:\n  windwo: 3\n")
    with pytest.raises(ValueError, match="unknown"):
        load_run_config(f)
    for path in ("configs/cpu_small.yaml", "configs/full.yaml"):
        load_run_config(path)


def test_choose_rules():
    rows = []
    for m, v in (
        ("tod/14d/mean", 0.20),
        ("tod_weekday/14d/mean", 0.15),
        ("tod_weekday/14d/median", 0.15),
        ("tod_weekday/7d/mean", 0.152),
    ):
        for W, f in ((24, 1.4), (48, 1.0), (96, 0.7)):
            rows.append({"stratum": ">28d", "method": m, "W": W, "median_rms_m": v * f})
    d = choose(pd.DataFrame(rows))
    assert d["method_by_stratum"][">28d"] == "tod_weekday/14d/mean"
    assert d["window"] == 48 and d["stride"] == 24
    lag = pd.DataFrame(
        {
            "method": "m",
            "lag_lo": [0, 7, 14],
            "lag_hi": [7, 14, 30],
            "median_abs_bias_m": [0.05, 0.1, 0.3],
            "median_rms_m": [0.15, 0.2, 0.4],
        }
    )
    assert choose_lag(lag, "m") == 14.0


def test_transforms():
    x = np.array([[0.5, -2.0, 30.0], [0.0, 0.0, 0.0]])
    y = signed_log1p(x)
    assert np.allclose(y[0], [np.log1p(0.5), -np.log1p(2.0), np.log1p(30.0)])
    mask = np.array([[True, True, False], [True, True, True]])
    pat, mag = scale_magnitude(x, mask)
    assert np.allclose(pat[0], [0.25, -1.0, 0.0]) and np.isclose(mag[0, 0], np.log1p(2.0))
    assert np.allclose(pat[1], 0) and mag[1, 0] == 0
    assert apply("scale_magnitude", x, mask).shape == (2, 4)
    with pytest.raises(ValueError):
        apply("nope", x)


@pytest.mark.data
def test_show_case_scenario_34(capsys):
    """Inventory reference: Scenario 34, pipe 22 (21->22), ~665 L/s."""
    from leaksense.cli import main

    assert os.environ.get(DATA_DIR_VAR)
    assert main(["show-case", "S034-pure", "--config", "configs/full.yaml"]) == 0
    out = capsys.readouterr().out
    vals = dict(re.findall(r"^\s*(\d+)\s+(-?\d+\.\d+)", out, flags=re.M))
    vals = {k: float(v) for k, v in vals.items()}
    assert vals["22"] == pytest.approx(-61, abs=3) and vals["21"] == pytest.approx(-33, abs=3)
    assert all(abs(v) <= 4.5 for k, v in vals.items() if k not in ("21", "22"))


# ----------------------------------------------------------------------------- CLI (synthetic)


def test_cli_build_features_and_show_case(cli_env, capsys):
    import json

    from leaksense.cli import main

    tmp_path, common = cli_env
    assert main(["build-features", *common, "--quiet"]) == 0
    out_dirs = list((tmp_path / "art" / "features").iterdir())
    assert len(out_dirs) == 1
    d = out_dirs[0]
    z = np.load(d / "features.npz")
    assert z["x"].shape[1] == 4 and len(z["x"]) == len(z["weight"]) == len(z["case_of_window"])
    stats = pd.read_parquet(d / "case_stats.parquet")
    assert len(stats) == 3 and stats["snr"].notna().all()
    meta = json.loads((d / "meta.json").read_text())
    assert meta["git"]["commit"] and meta["data_manifest_sha256"] and "config" in meta
    capsys.readouterr()
    assert main(["show-case", "S002-pure", *common]) == 0
    out = capsys.readouterr().out
    assert len(re.findall(r"^\s*[2-5]\s+-?\d+\.\d+", out, flags=re.M)) == 4
    assert "<- pipe end" in out


def test_cli_baseline_study_skips_test_fold(cli_env, monkeypatch):
    import json

    from leaksense.cli import main

    tmp_path, common = cli_env
    loaded = []
    real_get = F.PressureCache.get

    def spy(self, sid):
        loaded.append(sid)
        return real_get(self, sid)

    monkeypatch.setattr(F.PressureCache, "get", spy)
    out = tmp_path / "rep" / "bc"
    args = [
        "baseline-study",
        *common,
        "--splits",
        str(tmp_path / "splits.json"),
        "--out",
        str(out),
        "--quiet",
    ]
    assert main(args) == 0
    assert set(loaded) <= {1, 2, 3} and 4 not in loaded
    rep = json.loads(out.with_suffix(".json").read_text())
    assert rep["excluded_test_scenarios"] == 1 and rep["decision"]["window"] in (
        1,
        6,
        12,
        24,
        48,
        96,
        336,
    )
    assert rep["git"]["commit"] and rep["data_manifest_sha256"]
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "## Decision" in md and "tod_weekday" in md


def test_choose_needs_long_history():
    with pytest.raises(ValueError, match="more than 28 days"):
        choose(pd.DataFrame())
