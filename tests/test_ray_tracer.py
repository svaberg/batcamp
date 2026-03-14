from __future__ import annotations

import numpy as np
import pytest
from batread.dataset import Dataset

from batcamp import Octree
from batcamp import OctreeRayTracer
from fake_dataset import FakeDataset as _FakeDataset
from fake_dataset import build_spherical_hex_mesh as _build_spherical_hex_mesh
from sample_data_helper import data_file


def _build_regular_fake_dataset(
    *,
    nr: int = 1,
    ntheta: int = 2,
    nphi: int = 4,
) -> _FakeDataset:
    """Private test helper: build a small spherical shell dataset."""
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


def test_trace_arrays_ordered_and_inside_cells() -> None:
    """Ray tracer should return ordered array segments that stay in the domain."""
    ds = _build_regular_fake_dataset()
    tree = Octree.from_dataset(ds, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)

    origin = np.array([2.2, 0.1, 0.0], dtype=float)
    direction = np.array([-1.0, 0.2, 0.1], dtype=float)
    t_start = 0.0
    t_end = 6.0

    cell_ids, t_enter, t_exit = tracer.trace(origin, direction, t_start, t_end)
    assert cell_ids.ndim == 1 and t_enter.ndim == 1 and t_exit.ndim == 1
    assert cell_ids.shape == t_enter.shape == t_exit.shape
    assert cell_ids.size > 0
    assert np.all(cell_ids >= 0)
    assert np.all(t_exit >= t_enter)
    if t_enter.size > 1:
        assert np.all(np.diff(t_enter) >= -1.0e-8)
    assert float(t_enter[0]) >= float(t_start) - 1.0e-8
    assert float(t_exit[-1]) <= float(t_end) + 1.0e-6


def test_loaded_tree_matches_original_walk(tmp_path) -> None:
    """Persisted/reloaded trees should return equal segment arrays."""
    ds = _build_regular_fake_dataset()
    tree = Octree.from_dataset(ds, tree_coord="rpa")
    path = tmp_path / "ray_tree.npz"
    tree.save(path)
    loaded = Octree.load(path, ds=tree.ds)

    origin = np.array([2.2, 0.1, 0.0], dtype=float)
    direction = np.array([-1.0, 0.2, 0.1], dtype=float)
    t_start = 0.0
    t_end = 6.0

    a = OctreeRayTracer(tree).trace(origin, direction, t_start, t_end)
    b = OctreeRayTracer(loaded).trace(origin, direction, t_start, t_end)
    assert len(a) == 3 and len(b) == 3
    for av, bv in zip(a, b):
        np.testing.assert_allclose(np.asarray(av), np.asarray(bv), atol=1.0e-8, rtol=0.0)


def test_outside_outward_ray_returns_empty() -> None:
    """Outside-start outward rays should trace zero segments."""
    ds = _build_regular_fake_dataset()
    tree = Octree.from_dataset(ds, tree_coord="rpa")
    _r_lo, r_hi = tree.domain_bounds(coord="rpa")
    r_max = float(r_hi[0])
    origin = np.array([r_max + 25.0, 0.0, 0.0], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)

    cell_ids, t_enter, t_exit = OctreeRayTracer(tree).trace(origin, direction, 0.0, 10.0)
    assert cell_ids.size == 0
    assert t_enter.size == 0
    assert t_exit.size == 0


def test_non_increasing_interval_returns_empty() -> None:
    """Ray trace should return empty arrays when `t_end <= t_start`."""
    ds = _build_regular_fake_dataset()
    tree = Octree.from_dataset(ds, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)
    origin = np.array([0.0, 0.0, 0.0], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)

    a = tracer.trace(origin, direction, 1.0, 1.0)
    b = tracer.trace(origin, direction, 2.0, 1.0)
    assert all(np.asarray(v).size == 0 for v in a)
    assert all(np.asarray(v).size == 0 for v in b)


def test_zero_direction_raises() -> None:
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


def test_example_specific_failing_ray_trace_midpoints_match_lookup() -> None:
    """Provided example: traced segments should stay on oracle cells for the bad 16x16 ray."""
    ds = Dataset.from_file(str(data_file("3d__var_1_n00000000.plt")))
    tree = Octree.from_dataset(ds)
    tracer = OctreeRayTracer(tree)

    origin = np.array([-48.000096, 3.2, -41.6], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    dmin, dmax = tree.domain_bounds(coord="xyz")
    t_end = float((float(dmax[0]) - origin[0]) * 0.999999)

    cell_ids, enters, exits = tracer.trace(origin, direction, 0.0, t_end)

    assert cell_ids.size > 0
    if cell_ids.size > 1:
        assert np.all(cell_ids[1:] != cell_ids[:-1])
    midpoints = 0.5 * (enters + exits)
    for cell_id, t_mid in zip(cell_ids, midpoints, strict=True):
        hit = tree.lookup_point(origin + float(t_mid) * direction, coord="xyz")
        assert hit is not None
        assert int(hit.cell_id) == int(cell_id)
