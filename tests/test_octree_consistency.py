from __future__ import annotations

import numpy as np
import pytest

from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp.constants import XYZ_VARS
from octree_test_support import cell_bounds


@pytest.fixture(scope="module")
def advanced_context(difflevels_rpa_context: dict[str, object]) -> tuple[object, Octree]:
    """Reuse session-cached difflevels dataset/tree pair."""
    return difflevels_rpa_context["ds"], difflevels_rpa_context["tree"]


def _select_interior_queries(tree: Octree, *, n_query: int, seed: int) -> np.ndarray:
    """Private test helper: pick deterministic random interior query points."""
    rng = np.random.default_rng(seed)
    n_cells = int(tree.cell_count)
    n = min(int(n_query), n_cells)
    idx = rng.choice(n_cells, size=n, replace=False)
    queries = []
    for cell_id in idx.tolist():
        lo, hi = cell_bounds(tree, int(cell_id), coord="xyz")
        queries.append(0.5 * (np.asarray(lo, dtype=float) + np.asarray(hi, dtype=float)))
    return np.asarray(queries, dtype=float)


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
    queries = _select_interior_queries(tree, n_query=64, seed=1)

    for q in queries:
        cell_id_xyz = int(tree.lookup_points(q, coord="xyz")[0])
        assert cell_id_xyz >= 0
        cell_id_rpa = int(tree.lookup_points(_xyz_to_rpa_numpy(q), coord="rpa")[0])
        assert cell_id_rpa >= 0
        assert cell_id_xyz == cell_id_rpa


@pytest.mark.slow
def test_loaded_tree_interpolator_match(advanced_context, tmp_path) -> None:
    """Interpolator outputs should be equal when using original vs loaded tree."""
    ds, tree = advanced_context
    path = tmp_path / "advanced_interp_tree.npz"
    tree.save(path)
    loaded = Octree.load(
        path,
        points=np.column_stack(tuple(np.asarray(ds[name], dtype=float) for name in XYZ_VARS)),
        corners=np.asarray(ds.corners, dtype=np.int64),
    )

    values = np.asarray(ds["Rho [g/cm^3]"])
    interp_a = OctreeInterpolator(tree, values)
    interp_b = OctreeInterpolator(loaded, values)

    queries = _select_interior_queries(tree, n_query=64, seed=7)
    vals_a, cids_a = interp_a(queries, return_cell_ids=True)
    vals_b, cids_b = interp_b(queries, return_cell_ids=True)

    assert np.array_equal(cids_a, cids_b)
    assert np.allclose(vals_a, vals_b, atol=0.0, rtol=0.0, equal_nan=True)
