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
def test_trace_sc(_sc_interp: OctreeInterpolator) -> None:
    """Ray contract: traced arrays are valid."""
    interp = _sc_interp
    origin, direction, t0, t1 = _diagnostic_ray_setup()

    cell_ids, t_enter, t_exit = OctreeRayTracer(interp.tree).trace(origin, direction, t0, t1)
    assert cell_ids.size > 0
    assert t_enter.shape == cell_ids.shape == t_exit.shape
    assert np.all(cell_ids >= 0)
    assert float(t_enter[0]) >= 0.0
    assert float(t_exit[-1]) <= t1 + 1e-12


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
    counts = np.asarray(ray.ray_tracer.segment_counts(origins, directions, 0.0, t_end), dtype=np.int64)

    assert image_shape == (1, 1)
    assert counts.tolist() == [1]
    np.testing.assert_allclose(per_ray, shared, rtol=0.0, atol=1e-12, equal_nan=True)
