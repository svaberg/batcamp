from __future__ import annotations

import numpy as np
import pytest
from batread.dataset import Dataset

from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp import OctreeRayInterpolator
from batcamp.ray import _ray_cell_geometry_for_maxdepth
from fake_dataset import FakeDataset as _FakeDataset
from fake_dataset import build_cartesian_hex_mesh as _build_cartesian_hex_mesh
from fake_dataset import build_spherical_hex_mesh as _build_spherical_hex_mesh
from sample_data_helper import data_file


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


def _build_single_cell_trilinear_dataset() -> _FakeDataset:
    """Private test helper: one Cartesian cell with trilinear field `x*y`."""
    points, corners = _build_cartesian_hex_mesh(
        x_edges=np.array([0.0, 1.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        z_edges=np.array([0.0, 1.0], dtype=float),
    )
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return _FakeDataset(
        points=points,
        corners=corners,
        variables={
            Octree.X_VAR: x,
            Octree.Y_VAR: y,
            Octree.Z_VAR: z,
            "TrilinearXY": x * y,
        },
    )


def _build_depth1_cartesian_dataset() -> _FakeDataset:
    """Private test helper: 2x2x2 Cartesian mesh on `[0, 2]^3`."""
    points, corners = _build_cartesian_hex_mesh(
        x_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        y_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        z_edges=np.array([0.0, 1.0, 2.0], dtype=float),
    )
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return _FakeDataset(
        points=points,
        corners=corners,
        variables={
            Octree.X_VAR: x,
            Octree.Y_VAR: y,
            Octree.Z_VAR: z,
            "Curved": x * x + y + 0.5 * z,
        },
    )


def _build_root_cartesian_dataset() -> _FakeDataset:
    """Private test helper: one Cartesian root cell on `[0, 2]^3`."""
    points, corners = _build_cartesian_hex_mesh(
        x_edges=np.array([0.0, 2.0], dtype=float),
        y_edges=np.array([0.0, 2.0], dtype=float),
        z_edges=np.array([0.0, 2.0], dtype=float),
    )
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return _FakeDataset(
        points=points,
        corners=corners,
        variables={
            Octree.X_VAR: x,
            Octree.Y_VAR: y,
            Octree.Z_VAR: z,
            "Curved": x * x + y + 0.5 * z,
        },
    )


def _corner_point_multiset(points: np.ndarray, corner_ids: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    """Private test helper: canonical corner-point multiset with duplicates preserved."""
    pts = np.asarray(points[np.asarray(corner_ids, dtype=np.int64)], dtype=float)
    return tuple(sorted(tuple(np.round(p, 12)) for p in pts))


def _dense_ray_oracle(
    interp: OctreeInterpolator,
    origin: np.ndarray,
    direction: np.ndarray,
    t_start: float,
    t_end: float,
    *,
    n_samples: int,
) -> float:
    """Private test helper: dense line-sampling oracle for one ray."""
    t = np.linspace(float(t_start), float(t_end), int(n_samples), dtype=float)
    d = np.asarray(direction, dtype=float).reshape(1, 3)
    pts = np.asarray(origin, dtype=float).reshape(1, 3) + t[:, None] * d
    vals = np.asarray(interp(pts, query_coord="xyz", log_outside_domain=False), dtype=float).reshape(-1)
    finite = np.isfinite(vals)
    return float(np.trapezoid(np.where(finite, vals, 0.0), x=t))


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


def test_direct_ray_integral_matches_exact_for_trilinear_field() -> None:
    """Direct ray integral should match the exact value on a trilinear field."""
    ds = _build_single_cell_trilinear_dataset()
    tree = Octree.from_dataset(ds, tree_coord="xyz")
    interp = OctreeInterpolator(ds, ["TrilinearXY"], tree=tree)
    ray = OctreeRayInterpolator(interp)

    origin = np.array([0.0, 0.0, 0.5], dtype=float)
    direction = np.array([1.0, 1.0, 0.0], dtype=float)
    t0 = 0.0
    t1 = float(np.sqrt(2.0))

    direct = float(np.asarray(ray.integrate_field_along_rays(origin[None, :], direction, t0, t1), dtype=float)[0])
    exact = float(np.sqrt(2.0) / 3.0)
    assert np.isclose(direct, exact, rtol=1e-12, atol=1e-12)


def test_cartesian_maxdepth_zero_matches_root_cell_interpolation() -> None:
    """`maxdepth=0` should match one-root-cell interpolation on a depth-1 Cartesian mesh."""
    fine = _build_depth1_cartesian_dataset()
    coarse = _build_root_cartesian_dataset()
    fine_interp = OctreeInterpolator(fine, ["Curved"], tree=Octree.from_dataset(fine, tree_coord="xyz"))
    coarse_interp = OctreeInterpolator(coarse, ["Curved"], tree=Octree.from_dataset(coarse, tree_coord="xyz"))

    cut = OctreeRayInterpolator(fine_interp, maxdepth=0)
    root = OctreeRayInterpolator(coarse_interp)

    origins = np.array(
        [[0.0, 0.25, 0.25], [0.0, 1.25, 0.75], [0.0, 1.5, 1.5]],
        dtype=float,
    )
    direction = np.array([1.0, 0.0, 0.0], dtype=float)

    cut_vals = np.asarray(cut.integrate_field_along_rays(origins, direction, 0.0, 2.0), dtype=float)
    root_vals = np.asarray(root.integrate_field_along_rays(origins, direction, 0.0, 2.0), dtype=float)
    assert np.allclose(cut_vals, root_vals, atol=1e-12, rtol=1e-12, equal_nan=True)


def test_cartesian_maxdepth_full_matches_default() -> None:
    """Full-depth Cartesian `maxdepth` should match the default ray path exactly."""
    fine = _build_depth1_cartesian_dataset()
    interp = OctreeInterpolator(fine, ["Curved"], tree=Octree.from_dataset(fine, tree_coord="xyz"))

    default = OctreeRayInterpolator(interp)
    full = OctreeRayInterpolator(interp, maxdepth=int(interp.tree.depth))

    origins = np.array(
        [[0.0, 0.25, 0.25], [0.0, 1.25, 0.75], [0.0, 1.5, 1.5]],
        dtype=float,
    )
    direction = np.array([1.0, 0.0, 0.0], dtype=float)

    default_counts = np.asarray(default.segment_counts(origins, direction, 0.0, 2.0), dtype=np.int64)
    full_counts = np.asarray(full.segment_counts(origins, direction, 0.0, 2.0), dtype=np.int64)
    default_vals = np.asarray(default.integrate_field_along_rays(origins, direction, 0.0, 2.0), dtype=float)
    full_vals = np.asarray(full.integrate_field_along_rays(origins, direction, 0.0, 2.0), dtype=float)

    assert np.array_equal(default_counts, full_counts)
    assert np.allclose(default_vals, full_vals, atol=1e-12, rtol=1e-12)


def test_spherical_maxdepth_zero_matches_root_cell_interpolation() -> None:
    """`maxdepth=0` should preserve coarse spherical hit coverage."""
    fine = _build_fake_dataset(nr=2, ntheta=4, nphi=8)
    coarse = _build_fake_dataset(nr=1, ntheta=2, nphi=4)
    fine_interp = OctreeInterpolator(fine, ["Scalar"], tree=Octree.from_dataset(fine, tree_coord="rpa"))
    coarse_interp = OctreeInterpolator(coarse, ["Scalar"], tree=Octree.from_dataset(coarse, tree_coord="rpa"))

    cut = OctreeRayInterpolator(fine_interp, maxdepth=0)
    root = OctreeRayInterpolator(coarse_interp)

    origins = np.array(
        [[-2.1, -0.2, 0.2], [-2.1, 0.3, 0.1]],
        dtype=float,
    )
    direction = np.array([1.0, 0.0, 0.0], dtype=float)

    cut_counts = np.asarray(cut.segment_counts(origins, direction, 0.0, 4.2), dtype=np.int64)
    root_counts = np.asarray(root.segment_counts(origins, direction, 0.0, 4.2), dtype=np.int64)
    cut_vals = np.asarray(cut.integrate_field_along_rays(origins, direction, 0.0, 4.2), dtype=float)
    root_vals = np.asarray(root.integrate_field_along_rays(origins, direction, 0.0, 4.2), dtype=float)

    assert np.array_equal(cut_counts, root_counts)
    assert np.array_equal(np.isfinite(cut_vals), np.isfinite(root_vals))


def test_spherical_maxdepth_zero_geometry_matches_root_cells() -> None:
    """Spherical `maxdepth=0` geometry should reproduce the true root-cell corners."""
    fine = _build_fake_dataset(nr=2, ntheta=4, nphi=8)
    coarse = _build_fake_dataset(nr=1, ntheta=2, nphi=4)
    tree = Octree.from_dataset(fine, tree_coord="rpa")
    geometry = _ray_cell_geometry_for_maxdepth(tree, 0)

    grouped = sorted(_corner_point_multiset(fine.points, geometry.corners[node_id]) for node_id in range(8))
    root = sorted(_corner_point_multiset(coarse.points, coarse.corners[cell_id]) for cell_id in range(8))

    assert grouped == root


def test_spherical_maxdepth_full_matches_default() -> None:
    """Full-depth spherical `maxdepth` should match the default ray path exactly."""
    fine = _build_fake_dataset(nr=2, ntheta=4, nphi=8)
    interp = OctreeInterpolator(fine, ["Scalar"], tree=Octree.from_dataset(fine, tree_coord="rpa"))

    default = OctreeRayInterpolator(interp)
    full = OctreeRayInterpolator(interp, maxdepth=int(interp.tree.depth))

    origins = np.array(
        [[-2.1, -0.2, 0.2], [-2.1, 0.3, 0.1]],
        dtype=float,
    )
    direction = np.array([1.0, 0.0, 0.0], dtype=float)

    default_counts = np.asarray(default.segment_counts(origins, direction, 0.0, 4.2), dtype=np.int64)
    full_counts = np.asarray(full.segment_counts(origins, direction, 0.0, 4.2), dtype=np.int64)
    default_vals = np.asarray(default.integrate_field_along_rays(origins, direction, 0.0, 4.2), dtype=float)
    full_vals = np.asarray(full.integrate_field_along_rays(origins, direction, 0.0, 4.2), dtype=float)

    assert np.array_equal(default_counts, full_counts)
    assert np.allclose(default_vals, full_vals, atol=1e-12, rtol=1e-12)


def test_spherical_maxdepth_one_matches_depth_one_mesh() -> None:
    """Spherical `maxdepth=1` should match a true depth-1 mesh on the same rays."""
    fine = _build_fake_dataset(nr=4, ntheta=8, nphi=16)
    coarse = _build_fake_dataset(nr=2, ntheta=4, nphi=8)
    fine_interp = OctreeInterpolator(fine, ["Scalar"], tree=Octree.from_dataset(fine, tree_coord="rpa"))
    coarse_interp = OctreeInterpolator(coarse, ["Scalar"], tree=Octree.from_dataset(coarse, tree_coord="rpa"))

    cut = OctreeRayInterpolator(fine_interp, maxdepth=1)
    ref = OctreeRayInterpolator(coarse_interp)

    origins = np.array(
        [[-2.1, -0.2, 0.2], [-2.1, 0.3, 0.1], [-2.1, 0.1, -0.25]],
        dtype=float,
    )
    direction = np.array([1.0, 0.0, 0.0], dtype=float)

    cut_counts = np.asarray(cut.segment_counts(origins, direction, 0.0, 4.2), dtype=np.int64)
    ref_counts = np.asarray(ref.segment_counts(origins, direction, 0.0, 4.2), dtype=np.int64)
    cut_vals = np.asarray(cut.integrate_field_along_rays(origins, direction, 0.0, 4.2), dtype=float)
    ref_vals = np.asarray(ref.integrate_field_along_rays(origins, direction, 0.0, 4.2), dtype=float)

    assert np.array_equal(cut_counts, ref_counts)
    assert np.allclose(cut_vals, ref_vals, atol=1e-12, rtol=1e-12)


def test_example_specific_failing_ray_matches_dense_oracle() -> None:
    """Provided example: known bad 8x8 ray should match a dense line-sampling oracle."""
    ds = Dataset.from_file(str(data_file("3d__var_1_n00000000.plt")))
    interp = OctreeInterpolator(ds, ["Rho [g/cm^3]"])
    ray = OctreeRayInterpolator(interp)

    dmin, dmax = interp.tree.domain_bounds(coord="xyz")
    y = np.linspace(float(dmin[1]), float(dmax[1]), 8, dtype=float)
    z = np.linspace(float(dmin[2]), float(dmax[2]), 8, dtype=float)
    x_span = float(dmax[0] - dmin[0])
    x0 = float(dmin[0] - 1.0e-6 * max(1.0, x_span))
    origin = np.array([x0, y[1], z[4]], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    t_end = float((float(dmax[0]) - x0) * 0.999999)

    direct = float(np.asarray(ray.integrate_field_along_rays(origin[None, :], direction, 0.0, t_end), dtype=float)[0])
    oracle = _dense_ray_oracle(interp, origin, direction, 0.0, t_end, n_samples=16384)
    assert np.isclose(direct, oracle, rtol=1e-2, atol=0.0)


def test_example_nan_ray_should_not_miss_finite_grid_signal() -> None:
    """Provided example: known `grid_pos_ray_nan` ray should stay finite."""
    ds = Dataset.from_file(str(data_file("3d__var_1_n00000000.plt")))
    interp = OctreeInterpolator(ds, ["Rho [g/cm^3]"])
    ray = OctreeRayInterpolator(interp)

    dmin, dmax = interp.tree.domain_bounds(coord="xyz")
    x_span = float(dmax[0] - dmin[0])
    x0 = float(dmin[0] - 1.0e-6 * max(1.0, x_span))
    origin = np.array(
        [x0, -8.961876832844574, -47.15542521994135],
        dtype=float,
    )
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    t_end = float((float(dmax[0]) - x0) * 0.999999)

    direct = float(np.asarray(ray.integrate_field_along_rays(origin[None, :], direction, 0.0, t_end), dtype=float)[0])
    oracle = _dense_ray_oracle(interp, origin, direction, 0.0, t_end, n_samples=16384)

    assert np.isfinite(oracle)
    assert oracle > 0.0
    assert np.isfinite(direct)
    assert direct > 0.0
    assert np.isclose(direct, oracle, rtol=2e-2, atol=0.0)


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
