import json
import os

import pytest

from leaksense.config import DATA_DIR_VAR
from leaksense.data.index import build_index

from synthetic_data import DEFAULT_SPECS, LeakSpec, ScenarioSpec, contract_for, make_dataset


@pytest.fixture
def synthetic(tmp_path):
    """(data_dir, contract) for a fresh synthetic dataset in the real file format."""
    return make_dataset(tmp_path), contract_for(DEFAULT_SPECS)


def pytest_collection_modifyitems(config, items):
    if os.environ.get(DATA_DIR_VAR):
        return
    skip = pytest.mark.skip(reason=f"{DATA_DIR_VAR} not set")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)


LONG_ROWS = 40 * 48  # long enough for the >28-day pseudo-cases of the baseline study
LONG_SPECS = {
    1: ScenarioSpec(),
    2: ScenarioSpec([LeakSpec("3", 1700, 1800, "big", 0.5)]),
    3: ScenarioSpec([LeakSpec("2", 1750, 1900, "small", 0.4)]),
    4: ScenarioSpec([LeakSpec("5", 1690, 1850, "big", 0.6)]),  # listed as test below
}


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """A 40-day synthetic dataset, case index, contract, splits and config for CLI tests.

    Returns (tmp_path, common CLI args). Scenario 4 is the locked test fold.
    """
    data_dir = make_dataset(tmp_path, LONG_SPECS, n_rows=LONG_ROWS, daily_amp=2.0)
    contract = contract_for(LONG_SPECS, n_rows=LONG_ROWS)
    (tmp_path / "contract.json").write_text(json.dumps(contract.to_dict()))
    build_index(data_dir, contract).cases.to_parquet(tmp_path / "cases.parquet")
    (tmp_path / "splits.json").write_text(json.dumps({"test": [4], "folds": [[2], [3]]}))
    (tmp_path / "cfg.yaml").write_text(
        "name: t\nfeatures:\n  slot: tod_weekday\n  max_lag_days: 7\n"
    )
    monkeypatch.setenv("LEAKSENSE_ARTIFACTS_DIR", str(tmp_path / "art"))
    common = [
        "--data-dir",
        str(data_dir),
        "--contract",
        str(tmp_path / "contract.json"),
        "--cases",
        str(tmp_path / "cases.parquet"),
        "--config",
        str(tmp_path / "cfg.yaml"),
    ]
    return tmp_path, common
