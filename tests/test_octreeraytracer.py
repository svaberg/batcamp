from __future__ import annotations

import numpy as np
import pytest
from starwinds_readplt.dataset import Dataset

from batcamp import Octree
from batcamp import OctreeRayTracer
from fake_dataset import FakeDataset as _FakeDataset
from fake_dataset import build_spherical_hex_mesh as _build_spherical_hex_mesh


@pytest.fixture(scope="module")
def advanced_context(difflevels_rpa_context: dict[str, object]) -> tuple[object, Octree]:
    """Private test helper: reuse session-cached difflevels dataset/tree pair."""
    return difflevels_rpa_context["ds"], difflevels_rpa_context["tree"]


@pytest.fixture(scope="module")
def regression_context(difflevels_rpa_context: dict[str, object]) -> tuple[Dataset, Octree]:
    """Private test helper: reuse session-cached regression dataset/tree pair."""
    return difflevels_rpa_context["ds"], difflevels_rpa_context["tree"]


def _select_resolvable_center_near_radius(tree: Octree, *, target_r: float) -> np.ndarray:
    """Private test helper: pick one resolvable center near `target_r`."""
    centers = np.asarray(tree.cell_centers, dtype=float)
    center_r = np.linalg.norm(centers, axis=1)
    order = np.argsort(np.abs(center_r - float(target_r)))
    for idx in order.tolist():
        q = np.asarray(centers[int(idx)], dtype=float)
        if tree.lookup_point(q, coord="xyz") is not None:
            return q
    raise AssertionError("No resolvable center found near requested radius.")


def _build_regular_fake_dataset(
    *,
    nr: int = 1,
    ntheta: int = 2,
    nphi: int = 4,
) -> _FakeDataset:
    """Private test helper: build a regular spherical dataset for edge-case tests."""
    points, corners = _build_spherical_hex_mesh(
        nr=nr,
        ntheta=ntheta,
        nphi=nphi,
        r_min=1.0,
        r_max=2.0,
    )
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    scalar = 2.0 * x - 1.0 * y + 0.5 * z + 7.0
    return _FakeDataset(
        points=points,
        corners=corners,
        variables={
            Octree.X_VAR: x,
            Octree.Y_VAR: y,
            Octree.Z_VAR: z,
            "Scalar": scalar,
        },
    )


@pytest.mark.slow
def test_trace_ray_segments_are_ordered_and_inside_cells(advanced_context) -> None:
    """Ray traversal segments must be monotone and contain their midpoint sample."""
    _ds, tree = advanced_context
    origin = _select_resolvable_center_near_radius(tree, target_r=1.0)
    direction = np.array([1.0, 0.32, 0.11], dtype=float)
    t_start = 0.0
    t_end = 6.5

    segments = OctreeRayTracer(tree).trace(origin, direction, t_start, t_end)
    assert segments, "Expected at least one traversed segment."

    ray_dir = direction / np.linalg.norm(direction)
    prev_exit = float(t_start)
    for seg in segments:
        assert float(seg.t_exit) >= float(seg.t_enter)
        assert float(seg.t_enter) >= prev_exit - 1e-6
        prev_exit = float(seg.t_exit)
        mid_t = 0.5 * (float(seg.t_enter) + float(seg.t_exit))
        p_mid = origin + mid_t * ray_dir
        assert tree.contains_cell(int(seg.cell_id), p_mid, coord="xyz", tol=1e-6)
    assert float(segments[0].t_enter) >= float(t_start) - 1e-8
    assert float(segments[-1].t_exit) <= float(t_end) + 1e-6


@pytest.mark.slow
def test_loaded_tree_matches_original_ray_walk(advanced_context, tmp_path) -> None:
    """Persisted/reloaded tree should produce equivalent ray traversal segments."""
    _ds, tree = advanced_context
    path = tmp_path / "advanced_ray_tree.npz"
    tree.save(path)
    loaded = Octree.load(path, ds=tree.ds)

    origin = _select_resolvable_center_near_radius(tree, target_r=1.0)
    direction = np.array([1.0, 0.32, 0.11], dtype=float)
    t_start = 0.0
    t_end = 6.5

    seg_a = OctreeRayTracer(tree).trace(origin, direction, t_start, t_end)
    seg_b = OctreeRayTracer(loaded).trace(origin, direction, t_start, t_end)
    assert len(seg_a) == len(seg_b)
    for a, b in zip(seg_a, seg_b):
        assert int(a.cell_id) == int(b.cell_id)
        assert np.isclose(float(a.t_enter), float(b.t_enter), atol=1e-8, rtol=0.0)
        assert np.isclose(float(a.t_exit), float(b.t_exit), atol=1e-8, rtol=0.0)


@pytest.mark.slow
def test_regression_trace_ray_from_outside_returns_empty(regression_context) -> None:
    """Outside-start outward rays should traverse no segments."""
    _ds, tree = regression_context
    _r_lo, r_hi = tree.domain_bounds(coord="rpa")
    r_max = float(r_hi[0])
    origin = np.array([r_max + 25.0, 0.0, 0.0], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    segments = OctreeRayTracer(tree).trace(origin, direction, 0.0, 10.0)
    assert segments == []


def test_trace_ray_returns_empty_for_non_increasing_interval(regression_context) -> None:
    """Ray trace should return empty when `t_end <= t_start`."""
    _ds, tree = regression_context
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    assert OctreeRayTracer(tree).trace(origin, direction, 1.0, 1.0) == []
    assert OctreeRayTracer(tree).trace(origin, direction, 2.0, 1.0) == []


def test_fake_trace_ray_zero_direction_raises() -> None:
    """Ray trace should reject zero-length direction vectors."""
    ds = _build_regular_fake_dataset()
    tree = Octree.from_dataset(ds, tree_coord="rpa")
    with pytest.raises(ValueError, match="direction_xyz must be finite and non-zero"):
        OctreeRayTracer(tree).trace(
            origin_xyz=np.array([1.0, 0.0, 0.0]),
            direction_xyz=np.array([0.0, 0.0, 0.0]),
            t_start=0.0,
            t_end=1.0,
        )
