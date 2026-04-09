from __future__ import annotations

import math

import numpy as np
import pytest

from batcamp import Octree
from batcamp import OctreeRayTracer
from batcamp import camera_rays
from fake_dataset import build_cartesian_hex_mesh
from fake_dataset import build_spherical_hex_mesh
from sample_data_helper import data_file
from batread import Dataset


def _build_xyz_tree() -> Octree:
    """Return one small Cartesian tree with known box bounds."""
    points, corners = build_cartesian_hex_mesh(
        x_edges=np.array([-1.0, 0.0, 1.0], dtype=float),
        y_edges=np.array([-0.75, 0.0, 0.75], dtype=float),
        z_edges=np.array([-0.5, 0.0, 0.5], dtype=float),
    )
    return Octree(points, corners, tree_coord="xyz")


def _build_rpa_tree() -> Octree:
    """Return one small spherical tree for unsupported-case coverage."""
    points, corners = build_spherical_hex_mesh(
        nr=2,
        npolar=2,
        nazimuth=4,
        r_min=1.0,
        r_max=2.0,
    )
    return Octree(points, corners, tree_coord="rpa")


def test_seed_domain_parallel_camera_returns_midpoints_in_cartesian_box() -> None:
    """Parallel camera rays should seed at the midpoint of the visible box interval."""
    tree = _build_xyz_tree()
    tracer = OctreeRayTracer(tree)
    origins, directions = camera_rays(
        origin=[-5.0, 0.0, 0.0],
        target=[0.0, 0.0, 0.0],
        up=[0.0, 0.0, 1.0],
        nx=5,
        ny=3,
        width=1.0,
        height=0.5,
        projection="parallel",
    )

    seeds = tracer.seed_domain(origins, directions)

    assert seeds.shape == (3, 5, 3)
    np.testing.assert_allclose(seeds[..., 0], 0.0)
    np.testing.assert_allclose(seeds[..., 1], origins[..., 1])
    np.testing.assert_allclose(seeds[..., 2], origins[..., 2])


def test_seed_domain_reports_cartesian_misses_as_nan() -> None:
    """Cartesian rays outside the box should miss cleanly."""
    tree = _build_xyz_tree()
    tracer = OctreeRayTracer(tree)
    origins = np.array(
        [
            [[-5.0, 0.0, 0.0], [-5.0, 2.0, 0.0]],
            [[-5.0, 0.0, 1.0], [-5.0, 0.25, 0.25]],
        ],
        dtype=float,
    )
    directions = np.broadcast_to(np.array([1.0, 0.0, 0.0], dtype=float), origins.shape).copy()

    seeds = tracer.seed_domain(origins, directions)

    np.testing.assert_allclose(seeds[0, 0], np.array([0.0, 0.0, 0.0]))
    assert np.all(np.isnan(seeds[0, 1]))
    assert np.all(np.isnan(seeds[1, 0]))
    np.testing.assert_allclose(seeds[1, 1], np.array([0.0, 0.25, 0.25]))


def test_seed_domain_respects_requested_t_window() -> None:
    """The returned seed should sit inside the requested `t` window."""
    tree = _build_xyz_tree()
    tracer = OctreeRayTracer(tree)
    origin = np.array([-5.0, 0.0, 0.0], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)

    seeds = tracer.seed_domain(origin, direction, t_min=4.5, t_max=5.5)

    np.testing.assert_allclose(seeds, np.array([[0.0, 0.0, 0.0]]))


def test_seed_domain_from_inside_uses_visible_midpoint() -> None:
    """Rays starting inside the box should seed at the visible midpoint."""
    tree = _build_xyz_tree()
    tracer = OctreeRayTracer(tree)
    origin = np.array([0.0, 0.0, 0.0], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)

    seeds = tracer.seed_domain(origin, direction)

    np.testing.assert_allclose(seeds, np.array([[0.5, 0.0, 0.0]]))


def test_seed_domain_rpa_front_shell_uses_front_visible_segment() -> None:
    """Spherical shell seeding should stay on the front visible shell segment."""
    tracer = OctreeRayTracer(_build_rpa_tree())
    origin = np.array([-5.0, 0.0, 0.0], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)

    seeds = tracer.seed_domain(origin, direction)

    np.testing.assert_allclose(seeds, np.array([[-1.5, 0.0, 0.0]]))


def test_seed_domain_rpa_handles_miss_and_shell_only_paths() -> None:
    """Off-axis shell rays should either miss or seed on the front shell."""
    tracer = OctreeRayTracer(_build_rpa_tree())
    origins = np.array(
        [
            [-5.0, 3.0, 0.0],
            [-5.0, 1.5, 0.0],
            [-5.0, 0.5, 0.0],
        ],
        dtype=float,
    )
    directions = np.broadcast_to(np.array([1.0, 0.0, 0.0], dtype=float), origins.shape).copy()

    seeds = tracer.seed_domain(origins, directions)

    assert np.all(np.isnan(seeds[0]))
    np.testing.assert_allclose(seeds[1], np.array([0.0, 1.5, 0.0]))
    np.testing.assert_allclose(seeds[2], np.array([-math.sqrt(2.0), 0.5, 0.0]))


def test_seed_domain_rpa_sample_file_uses_mid_shell_seed() -> None:
    """The local spherical sample file should seed central rays at mid-shell radius."""
    ds = Dataset.from_file(str(data_file("3d__var_1_n00000000.plt")))
    tree = Octree.from_ds(ds, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)
    domain_lo, domain_hi = tree.domain_bounds(coord="rpa")
    r_min = float(domain_lo[0])
    r_max = float(domain_hi[0])
    origin = np.array([-60.0, 0.0, 0.0], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)

    seeds = tracer.seed_domain(origin, direction)

    np.testing.assert_allclose(seeds, np.array([[-0.5 * (r_min + r_max), 0.0, 0.0]]), atol=1e-10)
