from __future__ import annotations

import math
import numpy as np
import pytest
from batread.dataset import Dataset

from batcamp import OctreeInterpolator
from batcamp import OctreeBuilder
from batcamp import OctreeRayInterpolator
from batcamp import OctreeRayTracer
from batcamp.octree import Octree
from batcamp.ray import FlatCamera
from batcamp.ray import FovCamera
from fake_dataset import FakeDataset as _FakeDataset
from fake_dataset import build_cartesian_hex_mesh as _build_cartesian_hex_mesh
from sample_data_helper import data_file


def _diagnostic_ray_setup() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Private test helper: one deterministic SC diagnostic ray."""
    r_shell = 4.0
    p_start = r_shell * np.array([1.0, 0.0, 0.0], dtype=float)
    p_end = r_shell * np.array([-0.25, 0.96, 0.11], dtype=float)
    p_end = r_shell * (p_end / np.linalg.norm(p_end))
    origin = p_start
    direction = p_end - p_start
    t0 = 0.0
    t1 = float(np.linalg.norm(direction))
    return origin, direction, t0, t1


@pytest.fixture(scope="module")
def _sc_interp() -> OctreeInterpolator:
    """Private test helper: shared SC interpolator for ray contract tests."""
    ds = Dataset.from_file(str(data_file("3d__var_4_n00044000.plt")))
    return OctreeInterpolator(OctreeBuilder().build(ds), ["Rho [g/cm^3]"])


def _integrate_rho2_with_rays(
    ray: OctreeRayInterpolator,
    interp: OctreeInterpolator,
    origins_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    t_end: float,
) -> np.ndarray:
    """Private test helper: integrate `rho^2` along rays using midpoint segments."""
    mids, weights, offsets = ray.adaptive_midpoint_rule(
        origins_xyz,
        direction_xyz,
        0.0,
        float(t_end),
        chunk_size=4096,
    )
    out = np.full(origins_xyz.shape[0], np.nan, dtype=float)
    if mids.shape[0] == 0:
        return out
    rho_mid = np.asarray(
        interp(mids, query_coord="xyz", log_outside_domain=False),
        dtype=float,
    ).reshape(-1)
    for i in range(out.shape[0]):
        s = int(offsets[i])
        e = int(offsets[i + 1])
        if e <= s:
            continue
        seg_rho = rho_mid[s:e]
        seg_w = weights[s:e]
        finite = np.isfinite(seg_rho)
        if not np.any(finite):
            continue
        out[i] = float(np.sum((seg_rho[finite] * seg_rho[finite]) * seg_w[finite]))
    return out


def _integrate_rho2_resample_baseline(
    interp: OctreeInterpolator,
    origins_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    t_end: float,
    *,
    n_steps: int = 256,
) -> np.ndarray:
    """Private test helper: baseline `rho^2` line integral from uniform resampling."""
    t = np.linspace(0.0, float(t_end), int(n_steps), dtype=float)
    dt = float(t[1] - t[0]) if t.size > 1 else float(t_end)
    d = np.asarray(direction_xyz, dtype=float).reshape(3)
    d = d / np.linalg.norm(d)
    q = origins_xyz[:, None, :] + t[None, :, None] * d[None, None, :]
    rho = np.asarray(
        interp(q.reshape(-1, 3), query_coord="xyz", log_outside_domain=False),
        dtype=float,
    ).reshape(origins_xyz.shape[0], t.size)
    rho2 = np.where(np.isfinite(rho), rho * rho, 0.0)
    return np.sum(rho2, axis=1) * dt


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


@pytest.mark.pooch
def test_trace_and_sample_sc(_sc_interp: OctreeInterpolator) -> None:
    """Ray contract: traced arrays and sampled values are valid."""
    interp = _sc_interp
    origin, direction, t0, t1 = _diagnostic_ray_setup()

    cell_ids, t_enter, t_exit = OctreeRayTracer(interp.tree).trace(origin, direction, t0, t1)
    assert cell_ids.size > 0
    assert t_enter.shape == cell_ids.shape == t_exit.shape
    assert np.all(cell_ids >= 0)
    assert float(t_enter[0]) >= 0.0
    assert float(t_exit[-1]) <= t1 + 1e-12

    n_samples = 1200
    t_values, ray_values, cell_ids_sample, _ = OctreeRayInterpolator(interp).sample(
        origin,
        direction,
        t0,
        t1,
        n_samples,
    )
    vals = np.asarray(ray_values, dtype=float).reshape(-1)
    cids = np.asarray(cell_ids_sample, dtype=np.int64).reshape(-1)
    assert np.asarray(t_values, dtype=float).shape == (n_samples,)
    assert vals.shape == (n_samples,)
    assert cids.shape == (n_samples,)
    assert np.all(cids >= 0)
    assert np.all(np.isfinite(vals))


@pytest.mark.parametrize(
    "file_name",
    [
        pytest.param("3d__var_4_n00044000.plt", id="sc", marks=pytest.mark.pooch),
        pytest.param("3d__var_4_n00005000.plt", id="ih", marks=pytest.mark.pooch),
    ],
)
@pytest.mark.slow
def test_rho2_matches_resample_baseline(file_name: str) -> None:
    """Ray contract: outside-start rays that hit data must integrate correctly."""
    ds = Dataset.from_file(str(data_file(file_name)))
    interp = OctreeInterpolator(OctreeBuilder().build(ds), ["Rho [g/cm^3]"])
    ray = OctreeRayInterpolator(interp)

    dmin, dmax = interp.tree.domain_bounds(coord="xyz")
    xmin, xmax = float(dmin[0]), float(dmax[0])
    ymin, ymax = float(dmin[1]), float(dmax[1])
    zmin, zmax = float(dmin[2]), float(dmax[2])

    ny, nz = 3, 3
    yg, zg = np.meshgrid(np.linspace(ymin, ymax, ny), np.linspace(zmin, zmax, nz), indexing="xy")
    x0 = xmin - 0.05 * (xmax - xmin)
    origins = np.column_stack((np.full(yg.size, x0, dtype=float), yg.ravel(), zg.ravel()))
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    t_end = (xmax - x0) * 0.999999

    base_vals = _integrate_rho2_resample_baseline(interp, origins, direction, t_end, n_steps=96)
    candidates = np.where(np.isfinite(base_vals) & (base_vals > 0.0))[0]
    assert candidates.size > 0
    idx = int(candidates[0])
    ray_vals = _integrate_rho2_with_rays(ray, interp, origins[idx : idx + 1], direction, t_end)
    ray_val = float(ray_vals[0])
    base_val = float(base_vals[idx])

    assert np.isfinite(ray_val)
    assert ray_val > 0.0
    rel = abs(ray_val - base_val) / max(abs(base_val), 1e-30)
    assert rel < 1e-8


def test_flat_camera_from_domain_x_matches_current_compare_setup() -> None:
    """Camera contract: flat domain-x camera reproduces the existing compare launch plane."""
    bounds = (-2.0, 3.0, -5.0, 7.0, -11.0, 13.0)
    camera = FlatCamera.from_domain_x(bounds)

    origins, direction, t_end, image_shape = camera.rays(ny=3, nz=4)

    x0 = float(bounds[0] - 1.0e-6 * max(1.0, bounds[1] - bounds[0]))
    y = np.linspace(bounds[2], bounds[3], 3, dtype=float)
    z = np.linspace(bounds[4], bounds[5], 4, dtype=float)
    yg, zg = np.meshgrid(y, z, indexing="xy")
    expected_origins = np.column_stack((np.full(yg.size, x0, dtype=float), yg.ravel(), zg.ravel()))
    expected_t_end = float((bounds[1] - x0) * 0.999999)

    assert image_shape == (4, 3)
    np.testing.assert_allclose(origins, expected_origins, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(direction, np.array([1.0, 0.0, 0.0], dtype=float), rtol=0.0, atol=1e-12)
    assert t_end == pytest.approx(expected_t_end, rel=0.0, abs=1e-12)


def test_fov_camera_center_ray_points_at_target() -> None:
    """Camera contract: odd-sized FOV image has a center ray through the target."""
    camera = FovCamera(
        eye_xyz=np.array([-5.0, 0.0, 0.0], dtype=float),
        target_xyz=np.array([0.0, 0.0, 0.0], dtype=float),
        up_hint_xyz=np.array([0.0, 0.0, 1.0], dtype=float),
        vertical_fov_degrees=60.0,
        t_end=10.0,
    )

    origins, directions, t_end, image_shape = camera.rays(ny=3, nz=3)
    direction_img = directions.reshape(image_shape + (3,))
    origin_img = origins.reshape(image_shape + (3,))

    assert image_shape == (3, 3)
    np.testing.assert_allclose(origin_img[1, 1], np.array([-5.0, 0.0, 0.0], dtype=float), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(direction_img[1, 1], np.array([1.0, 0.0, 0.0], dtype=float), rtol=0.0, atol=1e-12)
    assert t_end == pytest.approx(10.0, rel=0.0, abs=0.0)


def test_fov_camera_vertical_extent_matches_requested_fov() -> None:
    """Camera contract: top and bottom center rays span the requested vertical FOV."""
    camera = FovCamera(
        eye_xyz=np.array([-5.0, 0.0, 0.0], dtype=float),
        target_xyz=np.array([0.0, 0.0, 0.0], dtype=float),
        up_hint_xyz=np.array([0.0, 0.0, 1.0], dtype=float),
        vertical_fov_degrees=60.0,
        t_end=10.0,
    )

    _origins, directions, _t_end, image_shape = camera.rays(ny=3, nz=3)
    direction_img = directions.reshape(image_shape + (3,))
    d_top = direction_img[2, 1]
    d_bot = direction_img[0, 1]
    angle = math.degrees(math.acos(np.clip(float(np.dot(d_top, d_bot)), -1.0, 1.0)))

    assert angle == pytest.approx(60.0, rel=0.0, abs=1e-12)


def test_ray_interpolator_accepts_one_direction_per_ray() -> None:
    """Ray contract: per-ray directions from FOV camera go through the ray interpolator."""
    ds = _build_single_cell_trilinear_dataset()
    interp = OctreeInterpolator(OctreeBuilder().build(ds, tree_coord="xyz"), ["TrilinearXY"])
    ray = OctreeRayInterpolator(interp)
    camera = FovCamera(
        eye_xyz=np.array([-1.0, 0.5, 0.5], dtype=float),
        target_xyz=np.array([0.5, 0.5, 0.5], dtype=float),
        up_hint_xyz=np.array([0.0, 0.0, 1.0], dtype=float),
        vertical_fov_degrees=30.0,
        t_end=3.0,
    )

    origins, directions, t_end, image_shape = camera.rays(ny=1, nz=1)
    per_ray = np.asarray(ray.integrate_field_along_rays(origins, directions, 0.0, t_end), dtype=float)
    shared = np.asarray(ray.integrate_field_along_rays(origins, directions[0], 0.0, t_end), dtype=float)
    counts = np.asarray(ray.segment_counts(origins, directions, 0.0, t_end), dtype=np.int64)

    assert image_shape == (1, 1)
    assert counts.tolist() == [1]
    np.testing.assert_allclose(per_ray, shared, rtol=0.0, atol=1e-12, equal_nan=True)
