from functools import lru_cache

import pytest
from batread import Dataset

from batcamp import Octree
from batcamp.builder import DEFAULT_AXIS_RHO_TOL
from sample_data_helper import data_file


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-pooch",
        action="store_true",
        default=False,
        help="run tests that fetch files via pooch/network",
    )
    parser.addoption(
        "--run-perf",
        action="store_true",
        default=False,
        help="run performance-sensitive tests",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_pooch = pytest.mark.skip(reason="need --run-pooch option to run")
    skip_perf = pytest.mark.skip(reason="need --run-perf option to run")
    for item in items:
        if "pooch" in item.keywords and not config.getoption("--run-pooch"):
            item.add_marker(skip_pooch)
        if "perf" in item.keywords and not config.getoption("--run-perf"):
            item.add_marker(skip_perf)


@lru_cache(maxsize=1)
def _build_difflevels_rpa_case() -> tuple[Dataset, Octree]:
    """Build and cache one representative spherical dataset/tree pair."""
    input_file = data_file("3d__var_4_n00005000.plt")
    assert input_file.exists(), f"Missing sample file: {input_file}"
    ds = Dataset.from_file(str(input_file))
    assert ds.corners is not None

    tree = Octree.from_ds(
        ds,
        tree_coord="rpa",
        axis_rho_tol=DEFAULT_AXIS_RHO_TOL,
        level_rtol=1e-4,
        level_atol=1e-9,
    )
    return ds, tree


@pytest.fixture(scope="session")
def difflevels_rpa_case() -> tuple[Dataset, Octree]:
    return _build_difflevels_rpa_case()
