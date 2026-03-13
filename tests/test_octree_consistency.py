from __future__ import annotations

import numpy as np
import pytest

from batcamp import Octree
from batcamp import OctreeInterpolator


@pytest.fixture(scope="module")
def advanced_context(difflevels_rpa_context: dict[str, object]) -> tuple[object, Octree]:
    """Reuse session-cached difflevels dataset/tree pair."""
    return difflevels_rpa_context["ds"], difflevels_rpa_context["tree"]


def _select_center_queries(tree: Octree, *, n_query: int, seed: int) -> np.ndarray:
    """Private test helper: pick deterministic random interior query points."""
    rng = np.random.default_rng(seed)
    centers = np.asarray(tree.cell_centers, dtype=float)
    n = min(int(n_query), int(centers.shape[0]))
    idx = rng.choice(centers.shape[0], size=n, replace=False)
    return centers[idx]


def _xyz_to_rpa_numpy(q_xyz: np.ndarray) -> np.ndarray:
    """Private test helper: convert one xyz point to one rpa point."""
    q = np.asarray(q_xyz, dtype=float).reshape(3)
    r = float(np.linalg.norm(q))
    zr = float(q[2] / max(r, float(np.finfo(float).tiny)))
    polar = float(np.arccos(np.clip(zr, -1.0, 1.0)))
    azimuth = float(np.mod(np.arctan2(q[1], q[0]), 2.0 * np.pi))
    return np.array([r, polar, azimuth], dtype=float)


@pytest.mark.slow
def test_lookup_xyz_rpa_consistency(advanced_context) -> None:
    """Many interior points should map to the same cell in xyz and rpa lookup coords."""
    _ds, tree = advanced_context
    queries = _select_center_queries(tree, n_query=64, seed=1)

    for q in queries:
        hit_xyz = tree.lookup_point(q, coord="xyz")
        assert hit_xyz is not None
        hit_rpa = tree.lookup_point(_xyz_to_rpa_numpy(q), coord="rpa")
        assert hit_rpa is not None
        assert int(hit_xyz.cell_id) == int(hit_rpa.cell_id)


@pytest.mark.slow
def test_loaded_tree_interpolator_match(advanced_context, tmp_path) -> None:
    """Interpolator outputs should be equal when using original vs loaded tree."""
    ds, tree = advanced_context
    path = tmp_path / "advanced_interp_tree.npz"
    tree.save(path)
    loaded = Octree.load(path, ds=ds)

    interp_a = OctreeInterpolator(ds, ["Rho [g/cm^3]"], tree=tree)
    interp_b = OctreeInterpolator(ds, ["Rho [g/cm^3]"], tree=loaded)

    queries = _select_center_queries(tree, n_query=64, seed=7)
    vals_a, cids_a = interp_a(queries, return_cell_ids=True)
    vals_b, cids_b = interp_b(queries, return_cell_ids=True)

    assert np.array_equal(cids_a, cids_b)
    assert np.allclose(vals_a, vals_b, atol=0.0, rtol=0.0, equal_nan=True)
