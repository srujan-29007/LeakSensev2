import os

import pytest

from leaksense.config import DATA_DIR_VAR

from synthetic_data import DEFAULT_SPECS, contract_for, make_dataset


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
