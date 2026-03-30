from functools import lru_cache

import numpy as np
import pytest
from batread import Dataset

from batcamp import Octree
from batcamp import build_octree_from_ds
from batcamp.builder import DEFAULT_AXIS_RHO_TOL
from batcamp.builder import point_refinement_levels
import batcamp.builder_spherical as spherical_builder
from sample_data_helper import data_file


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-pooch",
        action="store_true",
        default=False,
        help="run tests that fetch files via pooch/network",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-pooch"):
        return
    skip_pooch = pytest.mark.skip(reason="need --run-pooch option to run")
    for item in items:
        if "pooch" in item.keywords:
            item.add_marker(skip_pooch)


@lru_cache(maxsize=1)
def _build_difflevels_rpa_context() -> dict[str, object]:
    """Private test helper: build and cache shared spherical test context."""
    input_file = data_file("3d__var_4_n00005000.plt")
    assert input_file.exists(), f"Missing sample file: {input_file}"
    ds = Dataset.from_file(str(input_file))
    assert ds.corners is not None

    corners = np.asarray(ds.corners, dtype=np.int64)
    tree = build_octree_from_ds(
        ds,
        tree_coord="rpa",
        axis_rho_tol=DEFAULT_AXIS_RHO_TOL,
        level_rtol=1e-4,
        level_atol=1e-9,
    )
    azimuth_span, azimuth_center, _levels, expected, coarse = spherical_builder.compute_azimuth_spans_and_levels(
        np.column_stack((np.asarray(ds["X [R]"]), np.asarray(ds["Y [R]"]), np.asarray(ds["Z [R]"]))),
        corners=corners,
        rtol=1e-4,
        atol=1e-9,
        axis_rho_tol=DEFAULT_AXIS_RHO_TOL,
    )
    assert tree.cell_levels is not None
    cell_levels = tree.cell_levels
    point_levels = point_refinement_levels(
        n_points=ds.points.shape[0],
        corners=corners,
        cell_levels=cell_levels,
    )

    return {
        "ds": ds,
        "corners": corners,
        "azimuth_span": azimuth_span,
        "azimuth_center": azimuth_center,
        "cell_levels": cell_levels,
        "expected": expected,
        "coarse": coarse,
        "point_levels": point_levels,
        "tree": tree,
    }


@pytest.fixture(scope="session")
def difflevels_rpa_context() -> dict[str, object]:
    return _build_difflevels_rpa_context()
