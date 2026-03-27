from __future__ import annotations

import numpy as np
import pytest
from batread import Dataset

from batcamp import Octree
from batcamp import OctreeBuilder
from batcamp import OctreeInterpolator
from batcamp.constants import XYZ_VARS
from batcamp.spherical import _xyz_to_rpa_components


@pytest.fixture(scope="module")
def regression_context(difflevels_rpa_context: dict[str, object]) -> tuple[Dataset, Octree]:
    """Reuse session-cached difflevels dataset/tree pair."""
    return difflevels_rpa_context["ds"], difflevels_rpa_context["tree"]


def test_xyz_to_rpa_components_stable_and_finite() -> None:
    """Regression: xyz->rpa conversion should be finite and non-recursive."""
    q = np.array([1.0, 0.0, 0.0], dtype=float)
    r, polar, azimuth = _xyz_to_rpa_components(float(q[0]), float(q[1]), float(q[2]))
    assert np.isfinite(r)
    assert np.isfinite(polar)
    assert np.isfinite(azimuth)
    assert np.isclose(r, 1.0, rtol=0.0, atol=1e-15)
    assert np.isclose(polar, np.pi / 2.0, rtol=0.0, atol=1e-15)
    assert np.isclose(azimuth, 0.0, rtol=0.0, atol=1e-15)


@pytest.mark.slow
def test_default_tree_inference_selects_rpa_for_regression_dataset(regression_context) -> None:
    """Default tree inference should stay on spherical geometry for this dataset."""
    ds, _tree = regression_context
    tree = OctreeBuilder().build(ds)
    interp = OctreeInterpolator(tree, np.asarray(ds["Rho [g/cm^3]"]))
    assert tree.tree_coord == "rpa"

    q = np.array([1.0, 0.0, 0.0], dtype=float)
    vals, cell_ids = interp(q, return_cell_ids=True)
    assert int(np.asarray(cell_ids).reshape(-1)[0]) >= 0
    assert np.isfinite(float(np.asarray(vals).reshape(-1)[0]))


@pytest.mark.slow
def test_lookup_outside_domain_returns_none(regression_context) -> None:
    """Regression: lookup outside radial domain should not snap to nearest cell."""
    _ds, tree = regression_context
    _r_lo, r_hi = tree.domain_bounds(coord="rpa")
    r_max = float(r_hi[0])
    q = np.array([r_max + 50.0, 0.0, 0.0], dtype=float)
    assert int(tree.lookup_points(q, coord="xyz")[0]) < 0


@pytest.mark.slow
def test_load_uses_dataset_corners(tmp_path, regression_context) -> None:
    """Regression: loaded trees should resolve lookups from explicit point/corner geometry."""
    ds, tree = regression_context
    path = tmp_path / "tree_regression.npz"
    tree.save(path)

    loaded = Octree.load(
        path,
        points=np.column_stack(tuple(np.asarray(ds[name], dtype=float) for name in XYZ_VARS)),
        corners=np.asarray(ds.corners, dtype=np.int64),
    )

    # Ensure lookups are functional via explicit geometry arrays.
    q = np.array([1.0, 0.0, 0.0], dtype=float)
    assert int(loaded.lookup_points(q, coord="xyz")[0]) >= 0
