from __future__ import annotations

import numpy as np
import pytest
from batread import Dataset

from batcamp import Octree
from batcamp import OctreeBuilder
from batcamp import OctreeInterpolator
from batcamp.octree import _xyz_to_rpa_components


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
    interp = OctreeInterpolator(tree, ["Rho [g/cm^3]"])
    assert tree.tree_coord == "rpa"

    q = np.array([1.0, 0.0, 0.0], dtype=float)
    vals, cids = interp(q, return_cell_ids=True)
    assert int(np.asarray(cids).reshape(-1)[0]) >= 0
    assert np.isfinite(float(np.asarray(vals).reshape(-1)[0]))


@pytest.mark.slow
def test_lookup_outside_domain_returns_none(regression_context) -> None:
    """Regression: lookup outside radial domain should not snap to nearest cell."""
    _ds, tree = regression_context
    _r_lo, r_hi = tree.domain_bounds(coord="rpa")
    r_max = float(r_hi[0])
    q = np.array([r_max + 50.0, 0.0, 0.0], dtype=float)
    hit = tree.lookup_point(q, coord="xyz")
    assert hit is None


@pytest.mark.slow
def test_load_uses_dataset_corners(tmp_path, regression_context) -> None:
    """Regression: loaded trees should resolve lookups from bound dataset corners."""
    ds, tree = regression_context
    path = tmp_path / "tree_regression.npz"
    tree.save(path)

    loaded = Octree.load(path, ds=ds)
    assert loaded.ds is ds

    # Ensure lookups are functional via ds.corners.
    q = np.array([1.0, 0.0, 0.0], dtype=float)
    hit = loaded.lookup_point(q, coord="xyz")
    assert hit is not None
