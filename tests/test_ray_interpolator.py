from __future__ import annotations

import numpy as np
import pytest

from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp import OctreeRayInterpolator
from fake_dataset import FakeDataset as _FakeDataset
from fake_dataset import build_cartesian_hex_mesh as _build_cartesian_hex_mesh
from fake_dataset import build_spherical_hex_mesh as _build_spherical_hex_mesh


def _build_fake_dataset(
    *,
    nr: int = 1,
    ntheta: int = 2,
    nphi: int = 4,
) -> _FakeDataset:
    """Private test helper: build a small regular spherical hexahedral dataset."""
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
    scalar = 3.0 * x - 2.0 * y + 0.5 * z + 1.0
    scalar2 = 2.0 * scalar + 3.0
    return _FakeDataset(
        points=points,
        corners=corners,
        variables={
            Octree.X_VAR: x,
            Octree.Y_VAR: y,
            Octree.Z_VAR: z,
            "Scalar": scalar,
            "Scalar2": scalar2,
        },
    )


def _build_fake_cartesian_dataset() -> _FakeDataset:
    """Private test helper: build a small regular Cartesian hexahedral dataset."""
    points, corners = _build_cartesian_hex_mesh(
        x_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        y_edges=np.array([-0.5, 0.5], dtype=float),
        z_edges=np.array([-0.25, 0.75], dtype=float),
    )
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    scalar = 2.5 * x - 1.25 * y + 0.75 * z + 3.0
    scalar2 = -0.5 * scalar + 2.0
    return _FakeDataset(
        points=points,
        corners=corners,
        variables={
            Octree.X_VAR: x,
            Octree.Y_VAR: y,
            Octree.Z_VAR: z,
            "Scalar": scalar,
            "Scalar2": scalar2,
        },
    )


def test_sample_rejects_bad_args() -> None:
    """Ray sampling should reject non-positive sample count and zero direction."""
    ds = _build_fake_dataset()
    interp = OctreeInterpolator(ds, ["Scalar"])
    ray = OctreeRayInterpolator(interp)
    with pytest.raises(ValueError, match="n_samples must be positive"):
        ray.sample(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.0, 1.0, 0)
    with pytest.raises(ValueError, match="direction_xyz must be finite and non-zero"):
        ray.sample(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 0.0, 1.0, 10)


def test_integrate_rejects_bad_args() -> None:
    """Bulk ray integration should validate origin shape, chunk size and interval."""
    ds = _build_fake_dataset()
    interp = OctreeInterpolator(ds, ["Scalar"])
    ray = OctreeRayInterpolator(interp)
    with pytest.raises(ValueError, match="origins_xyz must have shape"):
        ray.integrate_field_along_rays(np.array([1.0, 2.0]), np.array([1.0, 0.0, 0.0]), 0.0, 1.0)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        ray.integrate_field_along_rays(
            np.array([[1.0, 0.0, 0.0]]),
            np.array([1.0, 0.0, 0.0]),
            0.0,
            1.0,
            chunk_size=0,
        )
    with pytest.raises(ValueError, match="t_end must be greater than t_start"):
        ray.integrate_field_along_rays(np.array([[1.0, 0.0, 0.0]]), np.array([1.0, 0.0, 0.0]), 1.0, 1.0)


def test_cartesian_outside_inward_traces_and_integrates() -> None:
    """Outside-start inward rays should enter the domain and produce valid arrays."""
    ds = _build_fake_cartesian_dataset()
    tree = Octree.from_dataset(ds, tree_coord="xyz")
    interp = OctreeInterpolator(ds, ["Scalar"], tree=tree)
    ray = OctreeRayInterpolator(interp)

    dmin, dmax = tree.domain_bounds(coord="xyz")
    origin = np.array(
        [float(dmin[0]) - 1.0, 0.5 * float(dmin[1] + dmax[1]), 0.5 * float(dmin[2] + dmax[2])],
        dtype=float,
    )
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    t0 = 0.0
    t1 = float(dmax[0] - dmin[0]) + 2.0

    cell_ids, t_enter, t_exit = ray.ray_tracer.trace(origin, direction, t0, t1)
    assert cell_ids.size > 0
    assert t_enter.size == cell_ids.size == t_exit.size
    assert float(t_enter[0]) > 0.0
    assert np.all(cell_ids >= 0)
    assert np.all(t_exit >= t_enter)

    origins = origin.reshape(1, 3)
    exact = np.asarray(ray.integrate_field_along_rays(origins, direction, t0, t1), dtype=float)
    midpoint = np.asarray(ray.integrate_field_along_rays_midpoint(origins, direction, t0, t1), dtype=float)
    assert np.all(np.isfinite(exact))
    assert np.all(np.isfinite(midpoint))
    assert float(exact[0]) > 0.0
    assert float(midpoint[0]) > 0.0


def test_spherical_outside_inward_traces_and_integrates() -> None:
    """Outside-start inward rays should trace/integrate on spherical trees too."""
    ds = _build_fake_dataset()
    tree = Octree.from_dataset(ds, tree_coord="rpa")
    interp = OctreeInterpolator(ds, ["Scalar"], tree=tree)
    ray = OctreeRayInterpolator(interp)

    dmin, dmax = tree.domain_bounds(coord="xyz")
    origin = np.array([float(dmax[0]) + 1.0, 0.2, -0.15], dtype=float)
    direction = np.array([-1.0, 0.0, 0.0], dtype=float)
    t0 = 0.0
    t1 = float(dmax[0] - dmin[0]) + 3.0

    cell_ids, t_enter, t_exit = ray.ray_tracer.trace(origin, direction, t0, t1)
    assert cell_ids.size > 0
    assert t_enter.size == cell_ids.size == t_exit.size
    assert float(t_enter[0]) > 0.0
    assert np.all(cell_ids >= 0)
    assert np.all(t_exit >= t_enter)

    _t_vals, vals, cell_ids_sample, _segments = ray.sample(origin, direction, t0, t1, 96)
    cids = np.asarray(cell_ids_sample, dtype=np.int64).reshape(-1)
    v = np.asarray(vals, dtype=float).reshape(-1)
    assert np.any(cids >= 0)
    assert np.any(np.isfinite(v[cids >= 0]))


def test_adaptive_midpoint_offsets_consistent() -> None:
    """Adaptive midpoint packing should return monotone offsets and matching lengths."""
    ds = _build_fake_cartesian_dataset()
    tree = Octree.from_dataset(ds, tree_coord="xyz")
    interp = OctreeInterpolator(ds, ["Scalar"], tree=tree)
    ray = OctreeRayInterpolator(interp)

    origins = np.array(
        [[0.0, -0.2, 0.1], [0.0, 0.0, 0.2], [0.0, 0.2, 0.3]],
        dtype=float,
    )
    mids, weights, offsets = ray.adaptive_midpoint_rule(
        origins,
        np.array([1.0, 0.1, -0.05], dtype=float),
        0.0,
        2.0,
        chunk_size=2,
    )

    assert mids.ndim == 2 and mids.shape[1] == 3
    assert weights.ndim == 1
    assert mids.shape[0] == weights.shape[0]
    assert offsets.shape == (origins.shape[0] + 1,)
    assert int(offsets[0]) == 0
    assert int(offsets[-1]) == int(weights.shape[0])
    assert np.all(np.diff(offsets) >= 0)
    assert np.all(weights >= 0.0)


def test_midpoint_matches_exact_for_linear_field() -> None:
    """Midpoint quadrature should match exact integral for globally linear fields."""
    ds = _build_fake_cartesian_dataset()
    tree = Octree.from_dataset(ds, tree_coord="xyz")
    interp = OctreeInterpolator(ds, ["Scalar"], tree=tree)
    ray = OctreeRayInterpolator(interp)

    origins = np.array(
        [[0.0, -0.2, 0.0], [0.0, 0.0, 0.2], [0.0, 0.2, 0.4]],
        dtype=float,
    )
    direction = np.array([1.0, 0.2, -0.1], dtype=float)
    t0 = 0.0
    t1 = 1.0

    exact = np.asarray(ray.integrate_field_along_rays(origins, direction, t0, t1, chunk_size=2), dtype=float)
    midpoint = np.asarray(
        ray.integrate_field_along_rays_midpoint(origins, direction, t0, t1, chunk_size=2),
        dtype=float,
    )
    assert np.allclose(midpoint, exact, atol=1e-8, rtol=1e-9)


def test_vector_integrals_shape_on_all_miss() -> None:
    """Vector ray integration should keep `(n_rays, n_components)` shape on all misses."""
    ds = _build_fake_cartesian_dataset()
    tree = Octree.from_dataset(ds, tree_coord="xyz")
    interp = OctreeInterpolator(ds, ["Scalar", "Scalar2"], tree=tree)
    ray = OctreeRayInterpolator(interp)

    origins = np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)

    exact = np.asarray(ray.integrate_field_along_rays(origins, direction, 0.0, 1.0), dtype=float)
    midpoint = np.asarray(ray.integrate_field_along_rays_midpoint(origins, direction, 0.0, 1.0), dtype=float)

    assert exact.shape == (origins.shape[0], 2)
    assert midpoint.shape == (origins.shape[0], 2)
    assert np.all(np.isnan(exact))
    assert np.all(np.isnan(midpoint))
