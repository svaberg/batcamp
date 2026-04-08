from functools import lru_cache

import pytest
from batread import Dataset

from batcamp import Octree
from batcamp.builder import DEFAULT_AXIS_TOL
from sample_data_helper import data_file


@lru_cache(maxsize=1)
def _build_difflevels_rpa_case() -> tuple[Dataset, Octree]:
    """Build and cache one representative spherical dataset/tree pair."""
    input_file = data_file("3d__var_2_n00060005.plt")
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
