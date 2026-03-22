from __future__ import annotations

import math

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


def _rpa_to_xyz_numpy(q_rpa: np.ndarray) -> np.ndarray:
    """Private test helper: convert one rpa point to one xyz point."""
    q = np.array(q_rpa, dtype=float).reshape(3)
    sin_polar = math.sin(float(q[1]))
    return np.array(
        [
            float(q[0]) * sin_polar * math.cos(float(q[2])),
            float(q[0]) * sin_polar * math.sin(float(q[2])),
            float(q[0]) * math.cos(float(q[1])),
        ],
        dtype=float,
    )


def _midpoint_rpa(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Private test helper: midpoint of a wrapped `(r, polar, azimuth)` box."""
    phi_width = float((float(hi[2]) - float(lo[2])) % (2.0 * math.pi))
    if phi_width == 0.0 and not np.isclose(float(lo[2]), float(hi[2]), rtol=0.0, atol=0.0):
        phi_width = 2.0 * math.pi
    return np.array(
        [
            0.5 * (float(lo[0]) + float(hi[0])),
            0.5 * (float(lo[1]) + float(hi[1])),
            float((float(lo[2]) + 0.5 * phi_width) % (2.0 * math.pi)),
        ],
        dtype=float,
    )


@pytest.mark.slow
def test_coarse_spherical_cell_has_loose_xyz_bbox(advanced_context) -> None:
    """A coarse spherical cell can contain points in its xyz bbox that are not in the cell."""
    _ds, tree = advanced_context
    assert tree.cell_levels is not None
    coarse_ids = np.flatnonzero(np.array(tree.cell_levels, dtype=np.int64) == int(tree.min_level))
    assert coarse_ids.size > 0

    for cid in coarse_ids.tolist():
        lo_rpa, hi_rpa = tree.cell_bounds(int(cid), coord="rpa")
        q_rpa = _midpoint_rpa(lo_rpa, hi_rpa)
        q_xyz = _rpa_to_xyz_numpy(q_rpa)

        hit_xyz = tree.lookup_point(q_xyz, coord="xyz")
        hit_rpa = tree.lookup_point(q_rpa, coord="rpa")
        if hit_xyz is None or hit_rpa is None:
            continue
        if int(hit_xyz.cell_id) != int(cid) or int(hit_rpa.cell_id) != int(cid):
            continue

        lo_xyz, hi_xyz = tree.cell_bounds(int(cid), coord="xyz")
        for x in (float(lo_xyz[0]), float(hi_xyz[0])):
            for y in (float(lo_xyz[1]), float(hi_xyz[1])):
                for z in (float(lo_xyz[2]), float(hi_xyz[2])):
                    q_bbox = np.array([x, y, z], dtype=float)
                    if tree.contains_cell(int(cid), q_bbox, coord="xyz"):
                        continue
                    hit_bbox = tree.lookup_point(q_bbox, coord="xyz")
                    assert hit_bbox is None or int(hit_bbox.cell_id) != int(cid)
                    return

    raise AssertionError(
        "Expected at least one coarse spherical cell whose Cartesian bounding box extends beyond the true cell."
    )


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
