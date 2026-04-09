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


def _build_xyz_coarse_fine_tree() -> Octree:
    """Return one Cartesian tree with one depth-1 leaf facing four depth-2 leaves."""
    x_edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    y_edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    z_edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)

    node_index = -np.ones((x_edges.size, y_edges.size, z_edges.size), dtype=np.int64)
    xyz_list: list[tuple[float, float, float]] = []
    node_id = 0
    for ix, x in enumerate(x_edges):
        for iy, y in enumerate(y_edges):
            for iz, z in enumerate(z_edges):
                xyz_list.append((float(x), float(y), float(z)))
                node_index[ix, iy, iz] = node_id
                node_id += 1

    def cell(ix0: int, ix1: int, iy0: int, iy1: int, iz0: int, iz1: int) -> list[int]:
        return [
            int(node_index[ix0, iy0, iz0]),
            int(node_index[ix1, iy0, iz0]),
            int(node_index[ix1, iy1, iz0]),
            int(node_index[ix0, iy1, iz0]),
            int(node_index[ix0, iy0, iz1]),
            int(node_index[ix1, iy0, iz1]),
            int(node_index[ix1, iy1, iz1]),
            int(node_index[ix0, iy1, iz1]),
        ]

    corners = np.array(
        [
            cell(0, 2, 0, 2, 0, 2),
            cell(0, 2, 0, 2, 2, 4),
            cell(0, 2, 2, 4, 0, 2),
            cell(0, 2, 2, 4, 2, 4),
            cell(2, 4, 0, 2, 2, 4),
            cell(2, 4, 2, 4, 0, 2),
            cell(2, 4, 2, 4, 2, 4),
            cell(2, 3, 0, 1, 0, 1),
            cell(2, 3, 0, 1, 1, 2),
            cell(2, 3, 1, 2, 0, 1),
            cell(2, 3, 1, 2, 1, 2),
            cell(3, 4, 0, 1, 0, 1),
            cell(3, 4, 0, 1, 1, 2),
            cell(3, 4, 1, 2, 0, 1),
            cell(3, 4, 1, 2, 1, 2),
        ],
        dtype=np.int64,
    )
    points = np.array(xyz_list, dtype=float)
    return Octree(points, corners, tree_coord="xyz")


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


def test_resolve_transition_same_level_and_boundary_faces() -> None:
    """Same-level neighbors and domain exits should resolve exactly."""
    tracer = OctreeRayTracer(_build_xyz_tree())

    for subface in range(4):
        assert tracer._resolve_transition(0, 1, subface) == 4
        assert tracer._resolve_transition(4, 0, subface) == 0
        assert tracer._resolve_transition(0, 0, subface) == -1
        assert tracer._resolve_transition(4, 1, subface) == -1


def test_resolve_transition_coarse_face_uses_subface_quadrants() -> None:
    """One coarse face should resolve to the correct fine neighbor for each quadrant."""
    tree = _build_xyz_coarse_fine_tree()
    tracer = OctreeRayTracer(tree)

    np.testing.assert_array_equal(
        tree.cell_levels,
        np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int64),
    )
    assert tracer._resolve_transition(0, 1, 0) == 7
    assert tracer._resolve_transition(0, 1, 1) == 8
    assert tracer._resolve_transition(0, 1, 2) == 9
    assert tracer._resolve_transition(0, 1, 3) == 10


def test_resolve_transition_fine_face_caches_one_coarse_neighbor() -> None:
    """Fine-to-coarse transitions should reuse one cached coarse neighbor across subfaces."""
    tracer = OctreeRayTracer(_build_xyz_coarse_fine_tree())

    assert tracer._next_leaf[7, 0, 0] == -2
    for subface in range(4):
        assert tracer._resolve_transition(7, 0, subface) == 0
        assert tracer._resolve_transition(10, 0, subface) == 0
    assert tracer._next_leaf[7, 0, 0] == 0


def test_trace_one_xyz_ray_walks_same_level_cells_exactly() -> None:
    """One simple Cartesian ray should cross two same-level leaves with exact intervals."""
    tracer = OctreeRayTracer(_build_xyz_tree())
    leaf_ids, t_enter, t_exit = tracer._trace_one_ray(
        0,
        np.array([-2.0, -0.3, -0.2], dtype=float),
        np.array([1.0, 0.0, 0.0], dtype=float),
    )

    np.testing.assert_array_equal(leaf_ids, np.array([0, 4], dtype=np.int64))
    np.testing.assert_allclose(t_enter, np.array([1.0, 2.0], dtype=float))
    np.testing.assert_allclose(t_exit, np.array([2.0, 3.0], dtype=float))


def test_trace_one_xyz_ray_walks_coarse_to_fine_cells_exactly() -> None:
    """One Cartesian ray should walk from one coarse leaf into two finer leaves."""
    tracer = OctreeRayTracer(_build_xyz_coarse_fine_tree())
    leaf_ids, t_enter, t_exit = tracer._trace_one_ray(
        0,
        np.array([-0.1, 0.1875, 0.1875], dtype=float),
        np.array([1.0, 0.0, 0.0], dtype=float),
    )

    np.testing.assert_array_equal(leaf_ids, np.array([0, 7, 11], dtype=np.int64))
    np.testing.assert_allclose(t_enter, np.array([0.1, 0.6, 0.85], dtype=float))
    np.testing.assert_allclose(t_exit, np.array([0.6, 0.85, 1.1], dtype=float))


def test_trace_one_xyz_ray_walks_fine_to_coarse_cells_exactly() -> None:
    """One Cartesian ray should walk from finer leaves back into one coarse leaf."""
    tracer = OctreeRayTracer(_build_xyz_coarse_fine_tree())
    leaf_ids, t_enter, t_exit = tracer._trace_one_ray(
        11,
        np.array([1.1, 0.1875, 0.1875], dtype=float),
        np.array([-1.0, 0.0, 0.0], dtype=float),
    )

    np.testing.assert_array_equal(leaf_ids, np.array([11, 7, 0], dtype=np.int64))
    np.testing.assert_allclose(t_enter, np.array([0.1, 0.35, 0.6], dtype=float))
    np.testing.assert_allclose(t_exit, np.array([0.35, 0.6, 1.1], dtype=float))


def test_trace_one_ray_uses_spherical_sample_file_from_seed_leaf() -> None:
    """One forward spherical ray should trace monotonically to the traced planar inner boundary."""
    ds = Dataset.from_file(str(data_file("3d__var_1_n00000000.plt")))
    tree = Octree.from_ds(ds, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)
    origin = np.array([-60.0, 0.5, 0.25], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    seed_xyz = tracer.seed_domain(origin, direction)
    start_leaf = int(tree.lookup_points(seed_xyz, coord="xyz")[0])

    leaf_ids, t_enter, t_exit = tracer._trace_one_ray(start_leaf, origin, direction)
    t_seed = float(seed_xyz[0, 0] - origin[0])

    assert leaf_ids.size > 0
    assert np.all(np.diff(t_enter) >= 0.0)
    assert np.all(np.diff(t_exit) >= 0.0)
    assert np.all(t_exit > t_enter)
    assert t_enter[0] <= t_seed <= t_exit[0]

    last_leaf = int(leaf_ids[-1])
    before_exit = origin + (t_exit[-1] - 1.0e-9) * direction
    after_exit = origin + (t_exit[-1] + 1.0e-9) * direction
    assert tracer._point_inside_cell(last_leaf, before_exit, 1.0e-12)
    assert not tracer._point_inside_cell(last_leaf, after_exit, 1.0e-12)
