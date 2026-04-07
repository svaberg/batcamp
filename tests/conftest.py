from functools import lru_cache

import pytest
from batread import Dataset

from batcamp import Octree
from batcamp.builder import DEFAULT_AXIS_TOL
from sample_data_helper import data_file


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-pooch",
        action="store_true",
        default=False,
        help="include pooch/network tests in an unqualified test run",
    )
    parser.addoption(
        "--run-perf",
        action="store_true",
        default=False,
        help="include performance-sensitive tests in an unqualified test run",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # Explicit nodeids or file paths should run as requested; marker gating only
    # applies to broad unqualified runs like plain `pytest`.
    explicit_selection = bool(config.invocation_params.args)
    skip_pooch = pytest.mark.skip(reason="need --run-pooch option to run")
    skip_perf = pytest.mark.skip(reason="need --run-perf option to run")
    for item in items:
        if "pooch" in item.keywords and not config.getoption("--run-pooch") and not explicit_selection:
            item.add_marker(skip_pooch)
        if "perf" in item.keywords and not config.getoption("--run-perf") and not explicit_selection:
            item.add_marker(skip_perf)


@lru_cache(maxsize=1)
def _build_difflevels_rpa_case() -> tuple[Dataset, Octree]:
    """Build and cache one representative spherical dataset/tree pair."""
    input_file = data_file("3d__var_1_n00000000.plt")
    assert input_file.exists(), f"Missing sample file: {input_file}"
    ds = Dataset.from_file(str(input_file))
    assert ds.corners is not None

    tree = Octree.from_ds(
        ds,
        tree_coord="rpa",
        build_axis_tol=DEFAULT_AXIS_TOL,
        build_level_rtol=1e-4,
        build_level_atol=1e-9,
    )
    return ds, tree


@pytest.fixture(scope="session")
def difflevels_rpa_case() -> tuple[Dataset, Octree]:
    return _build_difflevels_rpa_case()
