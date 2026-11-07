from pathlib import Path

import pytest

from leaksense.config import (
    ARTIFACTS_DIR_VAR,
    DATA_DIR_VAR,
    ConfigError,
    get_artifacts_dir,
    get_data_dir,
)


def make_minimal_data_dir(root: Path) -> Path:
    d = root / "HanoiOK"
    (d / "Scenario-1").mkdir(parents=True)
    (d / "Hanoi.inp").write_text("[END]\n")
    (d / "Labels.csv").write_text("Scenario,Label\n1,0\n")
    return d


def test_unset_data_dir_raises_clear_error():
    with pytest.raises(ConfigError, match=DATA_DIR_VAR):
        get_data_dir({})


def test_blank_data_dir_raises():
    with pytest.raises(ConfigError, match="not set"):
        get_data_dir({DATA_DIR_VAR: "   "})


def test_nonexistent_data_dir_raises(tmp_path):
    with pytest.raises(ConfigError, match="not an existing directory"):
        get_data_dir({DATA_DIR_VAR: str(tmp_path / "nope")})


def test_data_dir_missing_entries_lists_them(tmp_path):
    d = make_minimal_data_dir(tmp_path)
    (d / "Labels.csv").unlink()
    with pytest.raises(ConfigError, match="Labels.csv"):
        get_data_dir({DATA_DIR_VAR: str(d)})


def test_valid_data_dir(tmp_path):
    d = make_minimal_data_dir(tmp_path)
    assert get_data_dir({DATA_DIR_VAR: str(d)}) == d.resolve()


def test_artifacts_dir_default_and_override(tmp_path):
    assert get_artifacts_dir({}).name == "artifacts"
    assert get_artifacts_dir({ARTIFACTS_DIR_VAR: str(tmp_path)}) == tmp_path.resolve()
