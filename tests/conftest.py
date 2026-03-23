from functools import lru_cache

import numpy as np
import pytest
from batread import Dataset

from batcamp import Octree
from batcamp.builder import point_refinement_levels
from batcamp.builder_spherical import SphericalOctreeBuilder
from batcamp.octree import DEFAULT_AXIS_RHO_TOL
from sample_data_helper import data_file


@lru_cache(maxsize=1)
def _build_difflevels_rpa_context() -> dict[str, object]:
    """Private test helper: build and cache shared spherical test context."""
    input_file = data_file("3d__var_2_n00060005.plt")
    assert input_file.exists(), f"Missing sample file: {input_file}"
    ds = Dataset.from_file(str(input_file))
    assert ds.corners is not None

    corners = np.asarray(ds.corners, dtype=np.int64)
    tree = Octree.from_dataset(
        ds,
        tree_coord="rpa",
        axis_rho_tol=DEFAULT_AXIS_RHO_TOL,
        level_rtol=1e-4,
        level_atol=1e-9,
    )
    delta_phi, center_phi, _levels, expected, coarse = SphericalOctreeBuilder.compute_delta_phi_and_levels(
        ds,
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
        "delta_phi": delta_phi,
        "center_phi": center_phi,
        "cell_levels": cell_levels,
        "expected": expected,
        "coarse": coarse,
        "point_levels": point_levels,
        "tree": tree,
        "lookup": tree.lookup,
    }


@pytest.fixture(scope="session")
def difflevels_rpa_context() -> dict[str, object]:
    return _build_difflevels_rpa_context()
