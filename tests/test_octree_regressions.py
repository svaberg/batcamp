from __future__ import annotations

import numpy as np
import pytest
from starwinds_readplt.dataset import Dataset

from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp import OctreeRayTracer
from batcamp import SphericalOctree


@pytest.fixture(scope="module")
def regression_context(difflevels_rpa_context: dict[str, object]) -> tuple[Dataset, Octree]:
    """Reuse session-cached difflevels dataset/tree pair."""
    return difflevels_rpa_context["ds"], difflevels_rpa_context["tree"]


def test_regression_xyz_to_rpa_is_stable_and_finite() -> None:
    """Regression: xyz->rpa conversion should be finite and non-recursive."""
    q = np.array([1.0, 0.0, 0.0], dtype=float)
    r, polar, azimuth = SphericalOctree.xyz_to_rpa(q)
    assert np.isfinite(r)
    assert np.isfinite(polar)
    assert np.isfinite(azimuth)
    assert np.isclose(r, 1.0, rtol=0.0, atol=1e-15)
    assert np.isclose(polar, np.pi / 2.0, rtol=0.0, atol=1e-15)
    assert np.isclose(azimuth, 0.0, rtol=0.0, atol=1e-15)


@pytest.mark.slow
def test_regression_interpolator_without_tree_falls_back_to_rpa(regression_context) -> None:
    """No-tree interpolator should recover from invalid xyz reconstruction on spherical data."""
    ds, _tree = regression_context
    interp = OctreeInterpolator(ds, ["Rho [g/cm^3]"], tree=None)
    assert interp.tree.tree_coord == "rpa"

    q = np.array([1.0, 0.0, 0.0], dtype=float)
    vals, cids = interp(q, return_cell_ids=True)
    assert int(np.asarray(cids).reshape(-1)[0]) >= 0
    assert np.isfinite(float(np.asarray(vals).reshape(-1)[0]))


def test_regression_quickstart_explicit_tree_equals_auto_tree() -> None:
    """Quickstart contract: explicit prebuilt tree and auto-tree must agree."""
    from sample_data_helper import data_file

    ds = Dataset.from_file(str(data_file("3d__var_4_n00000000.plt")))
    tree = Octree.from_dataset(ds, tree_coord="rpa")
    queries = np.asarray(tree.lookup._cell_centers[:16], dtype=float)

    interp_explicit = OctreeInterpolator(ds, ["Rho [g/cm^3]"], tree=tree)
    interp_auto = OctreeInterpolator(ds, ["Rho [g/cm^3]"])

    vals_explicit, cids_explicit = interp_explicit(queries, return_cell_ids=True)
    vals_auto, cids_auto = interp_auto(queries, return_cell_ids=True)

    np.testing.assert_array_equal(np.asarray(cids_explicit), np.asarray(cids_auto))
    np.testing.assert_allclose(np.asarray(vals_explicit), np.asarray(vals_auto), rtol=0.0, atol=1e-12)


@pytest.mark.slow
def test_regression_lookup_outside_domain_returns_none(regression_context) -> None:
    """Regression: lookup outside radial domain should not snap to nearest cell."""
    _ds, tree = regression_context
    r_max = float(tree.lookup._r_max)
    q = np.array([r_max + 50.0, 0.0, 0.0], dtype=float)
    hit = tree.lookup_point(q, space="xyz")
    assert hit is None


@pytest.mark.slow
def test_regression_trace_ray_from_outside_returns_empty(regression_context) -> None:
    """Regression: ray trace started outside the domain should return no segments."""
    _ds, tree = regression_context
    r_max = float(tree.lookup._r_max)
    origin = np.array([r_max + 25.0, 0.0, 0.0], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    segments = OctreeRayTracer(tree).trace(origin, direction, 0.0, 10.0)
    assert segments == []


@pytest.mark.slow
def test_regression_load_uses_dataset_corners(tmp_path, regression_context) -> None:
    """Regression: loaded trees should resolve lookups from bound dataset corners."""
    ds, tree = regression_context
    path = tmp_path / "tree_regression.npz"
    tree.save(path)

    loaded = Octree.load(path, ds=ds)
    assert loaded.ds is ds

    # Ensure lookups are functional via ds.corners.
    q = np.array([1.0, 0.0, 0.0], dtype=float)
    hit = loaded.lookup_point(q, space="xyz")
    assert hit is not None
