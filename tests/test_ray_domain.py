from __future__ import annotations

import math

import numpy as np
import pytest

from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp import OctreeRayTracer
from batcamp import camera_rays
from batcamp import render_midpoint_image
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


def _analytic_visible_shell_length(impact_parameter: np.ndarray, r_min: float, r_max: float) -> np.ndarray:
    """Return the visible path length through one opaque-inner spherical shell."""
    b = np.asarray(impact_parameter, dtype=float)
    out = np.zeros_like(b)
    front_only = b < r_min
    shell_only = (b >= r_min) & (b < r_max)
    out[front_only] = np.sqrt(r_max * r_max - b[front_only] * b[front_only]) - np.sqrt(
        r_min * r_min - b[front_only] * b[front_only]
    )
    out[shell_only] = 2.0 * np.sqrt(r_max * r_max - b[shell_only] * b[shell_only])
    return out


def _analytic_sphere_chord_length(impact_parameter: np.ndarray, radius: float) -> np.ndarray:
    """Return the full chord length through one sphere."""
    b = np.asarray(impact_parameter, dtype=float)
    out = np.zeros_like(b)
    inside = b < radius
    out[inside] = 2.0 * np.sqrt(radius * radius - b[inside] * b[inside])
    return out


def _perpendicular_unit(direction: np.ndarray) -> np.ndarray:
    """Return one deterministic unit vector perpendicular to one 3D direction."""
    d = np.asarray(direction, dtype=float)
    basis = np.array([0.0, 0.0, 1.0], dtype=float)
    if np.isclose(abs(float(np.dot(d, basis))), np.linalg.norm(d), atol=1.0e-12, rtol=0.0):
        basis = np.array([0.0, 1.0, 0.0], dtype=float)
    perp = np.cross(d, basis)
    return perp / np.linalg.norm(perp)


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


def test_seed_interval_candidates_capture_both_sides_of_one_cartesian_interior_seed() -> None:
    """One traced interior seed should expose the local forward and backward intervals in one leaf."""
    tracer = OctreeRayTracer(_build_xyz_tree())
    seed_xyz = np.array([-0.5, -0.3, -0.2], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    seed_leaf = int(tracer.tree.lookup_points(seed_xyz.reshape(1, 3), coord="xyz")[0])

    intervals = tracer._seed_interval_candidates(seed_leaf, seed_xyz, direction)

    assert intervals == [(0.0, 0.5, 0), (-0.5, -0.0, 0)]
    assert tracer._combine_seed_intervals(intervals) == [(-0.5, 0.5, 0)]
    np.testing.assert_allclose(tracer._canonicalize_seed(seed_leaf, seed_xyz, direction)[0], seed_xyz)
    assert tracer._canonicalize_seed(seed_leaf, seed_xyz, direction)[1] == 0


def test_select_seed_branches_split_one_cartesian_face_seed_into_two_leaves() -> None:
    """One seed on an internal face should produce one backward and one forward start leaf."""
    tracer = OctreeRayTracer(_build_xyz_tree())
    origin = np.array([-2.0, -0.3, -0.2], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    seed_xyz = tracer.seed_domain(origin, direction).reshape(3)
    seed_leaf = int(tracer.tree.lookup_points(seed_xyz.reshape(1, 3), coord="xyz")[0])

    intervals = sorted(tracer._usable_seed_intervals(tracer._seed_interval_candidates(seed_leaf, seed_xyz, direction)))
    backward_seed, backward_leaf = tracer._select_seed_branch(seed_leaf, seed_xyz, direction, branch="backward")
    forward_seed, forward_leaf = tracer._select_seed_branch(seed_leaf, seed_xyz, direction, branch="forward")

    assert tracer._select_common_seed(seed_leaf, seed_xyz, direction) is None
    assert intervals == [(-1.0, -0.0, 0), (0.0, 1.0, 4)]
    assert backward_leaf == 0
    assert forward_leaf == 4
    np.testing.assert_allclose(backward_seed, np.array([-0.5, -0.3, -0.2], dtype=float))
    np.testing.assert_allclose(forward_seed, np.array([0.5, -0.3, -0.2], dtype=float))


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


def test_cell_segment_returns_exact_cartesian_exit_from_one_interior_point() -> None:
    """One direct cell-segment query should recover the exact exit interval and crossed face."""
    tracer = OctreeRayTracer(_build_xyz_tree())
    seed_xyz = np.array([-0.5, -0.3, -0.2], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)

    segment_enter, segment_exit, exit_face, subface_id = tracer._cell_segment(
        0,
        seed_xyz,
        direction,
        current_t=0.0,
        t_min=0.0,
    )

    np.testing.assert_allclose([segment_enter, segment_exit], np.array([0.0, 0.5], dtype=float))
    assert exit_face == 1
    assert subface_id == 3


def test_trace_one_ray_uses_spherical_sample_file_from_seed_leaf() -> None:
    """One forward spherical sample-file walk should reproduce the current exact segment contract."""
    ds = Dataset.from_file(str(data_file("3d__var_1_n00000000.plt")))
    tree = Octree.from_ds(ds, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)
    origin = np.array([-60.0, 0.5, 0.25], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    seed_xyz = tracer.seed_domain(origin, direction)
    start_leaf = int(tree.lookup_points(seed_xyz, coord="xyz")[0])

    leaf_ids, t_enter, t_exit = tracer._trace_one_ray(start_leaf, origin, direction)
    t_seed = float(seed_xyz[0, 0] - origin[0])

    np.testing.assert_allclose(seed_xyz, np.array([[-24.49362194, 0.5, 0.25]], dtype=float), atol=1.0e-8, rtol=0.0)
    assert start_leaf == 5685
    assert leaf_ids.size == 25
    np.testing.assert_array_equal(leaf_ids[:5], np.array([5685, 5684, 5683, 5682, 5591], dtype=np.int64))
    np.testing.assert_allclose(
        t_enter[:5],
        np.array([30.44827931, 36.79939909, 41.78896761, 45.70886953, 48.78842853], dtype=float),
        atol=1.0e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        t_exit[:5],
        np.array([36.79939909, 41.78896761, 45.70886953, 48.78842853, 51.2078844], dtype=float),
        atol=1.0e-8,
        rtol=0.0,
    )
    np.testing.assert_array_equal(leaf_ids[-5:], np.array([4834, 4833, 4832, 4831, 4830], dtype=np.int64))
    np.testing.assert_allclose(
        t_enter[-5:],
        np.array([59.15631661, 59.16091436, 59.16551221, 59.17011024, 59.17470791], dtype=float),
        atol=1.0e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        t_exit[-5:],
        np.array([59.16091436, 59.16551221, 59.17011024, 59.17470791, 59.17930579], dtype=float),
        atol=1.0e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(float(np.sum(t_exit - t_enter)), 28.731026478149758, atol=1.0e-10, rtol=0.0)
    assert t_enter[0] <= t_seed <= t_exit[0]

    last_leaf = int(leaf_ids[-1])
    before_exit = origin + (t_exit[-1] - 1.0e-9) * direction
    after_exit = origin + (t_exit[-1] + 1.0e-9) * direction
    assert tracer._point_inside_cell(last_leaf, before_exit, 1.0e-12)
    assert not tracer._point_inside_cell(last_leaf, after_exit, 1.0e-12)


def test_cell_segment_handles_one_real_sample_seam_leaf_exit() -> None:
    """One real seam-aligned sample leaf should still return one forward segment."""
    ds = Dataset.from_file(str(data_file("3d__var_1_n00000000.plt")))
    tree = Octree.from_ds(ds, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)
    point_xyz = np.array([-0.5, 0.0, 5.76], dtype=float)
    direction_xyz = np.array([1.0, 0.0, 0.0], dtype=float)
    leaf_id = int(tree.lookup_points(point_xyz.reshape(1, 3), coord="xyz")[0])

    segment_enter, segment_exit, exit_face, subface_id = tracer._cell_segment(
        leaf_id,
        point_xyz,
        direction_xyz,
        current_t=0.0,
        t_min=0.0,
    )

    assert leaf_id == 6045
    np.testing.assert_allclose(segment_enter, 0.0, atol=1.0e-15, rtol=0.0)
    np.testing.assert_allclose(segment_exit, 0.5000001195410285, atol=1.0e-12, rtol=0.0)
    assert exit_face == 4
    assert subface_id == 0


def test_canonicalize_seed_uses_the_only_visible_local_branch_in_spherical_shell() -> None:
    """One smooth-shell seed outside the traced mesh should canonicalize into the single local traced branch."""
    points, corners = build_spherical_hex_mesh(
        nr=8,
        npolar=24,
        nazimuth=32,
        r_min=1.0,
        r_max=2.0,
    )
    tree = Octree(points, corners, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)
    impact_parameter = 0.3
    z_value = 0.2
    origin = np.array([-5.0, math.sqrt(impact_parameter * impact_parameter - z_value * z_value), z_value], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    seed_xyz = tracer.seed_domain(origin, direction).reshape(3)
    seed_leaf = int(tree.lookup_points(seed_xyz.reshape(1, 3), coord="xyz")[0])

    intervals = tracer._usable_seed_intervals(tracer._seed_interval_candidates(seed_leaf, seed_xyz, direction))
    canonical_seed, canonical_leaf = tracer._canonicalize_seed(seed_leaf, seed_xyz, direction)
    forward_branch = tracer._select_seed_branch(seed_leaf, seed_xyz, direction, branch="forward")

    assert tracer._select_common_seed(seed_leaf, seed_xyz, direction) is None
    assert tracer._select_seed_branch(seed_leaf, seed_xyz, direction, branch="backward") is None
    np.testing.assert_allclose(seed_xyz, np.array([-1.46969385, 0.2236068, 0.2], dtype=float), atol=1.0e-8, rtol=0.0)
    assert seed_leaf == 2639
    assert len(intervals) == 1
    np.testing.assert_allclose(
        np.array(intervals[0][:2], dtype=float),
        np.array([0.005387463546405868, 0.13256348045557478], dtype=float),
        atol=1.0e-15,
        rtol=0.0,
    )
    assert intervals[0][2] == 2639
    assert forward_branch is not None
    np.testing.assert_allclose(forward_branch[0], np.array([-1.40071837, 0.2236068, 0.2], dtype=float), atol=1.0e-8, rtol=0.0)
    assert forward_branch[1] == 2639
    assert canonical_leaf == 2639
    np.testing.assert_allclose(canonical_seed, np.array([-1.40071837, 0.2236068, 0.2], dtype=float), atol=1.0e-8, rtol=0.0)


def test_common_seed_resolves_one_tied_shared_interval_on_the_real_spherical_sample() -> None:
    """A real sample-file seam ray may expose multiple shared leaves but should pick one consistently."""
    ds = Dataset.from_file(str(data_file("3d__var_1_n00000000.plt")))
    tree = Octree.from_ds(ds, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)
    origins, directions = camera_rays(
        origin=[-60.0, 0.0, 0.0],
        target=[0.0, 0.0, 0.0],
        up=[0.0, 0.0, 1.0],
        nx=11,
        ny=9,
        width=48.0,
        height=36.0,
        projection="parallel",
    )
    origin = origins[3, 5]
    direction = directions[3, 5]
    seed_xyz = tracer.seed_domain(origin, direction).reshape(3)
    seed_leaf = int(tree.lookup_points(seed_xyz.reshape(1, 3), coord="xyz")[0])

    intervals = sorted(tracer._usable_seed_intervals(tracer._seed_interval_candidates(seed_leaf, seed_xyz, direction)))
    common_seed = tracer._select_common_seed(seed_leaf, seed_xyz, direction)
    canonical_seed, canonical_leaf = tracer._canonicalize_seed(seed_leaf, seed_xyz, direction)

    np.testing.assert_allclose(seed_xyz, np.array([-24.17126426, 0.0, -4.0], dtype=float), atol=1.0e-8, rtol=0.0)
    assert seed_leaf == 3069
    assert len(intervals) == 2
    assert {leaf_id for _, _, leaf_id in intervals} == {3069, 7467}
    assert all(enter < 0.0 < exit_ for enter, exit_, _ in intervals)
    assert common_seed is not None
    np.testing.assert_allclose(common_seed[0], np.array([-26.0560636, 0.0, -4.0], dtype=float), atol=1.0e-8, rtol=0.0)
    assert common_seed[1] in {3069, 7467}
    assert canonical_leaf == common_seed[1]
    np.testing.assert_allclose(canonical_seed, common_seed[0])


def test_real_sample_face_seed_tie_break_stays_geometric() -> None:
    """A real sample-file face seed may resolve tied shared leaves, but the local geometry should stay consistent."""
    ds = Dataset.from_file(str(data_file("3d__var_1_n00000000.plt")))
    tree = Octree.from_ds(ds, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)
    origins, directions = camera_rays(
        origin=[-60.0, 0.0, 0.0],
        target=[0.0, 0.0, 0.0],
        up=[0.0, 0.0, 1.0],
        nx=11,
        ny=9,
        width=48.0,
        height=36.0,
        projection="parallel",
    )
    origin = origins[4, 8]
    direction = directions[4, 8]
    seed_xyz = tracer.seed_domain(origin, direction).reshape(3)
    seed_leaf = int(tree.lookup_points(seed_xyz.reshape(1, 3), coord="xyz")[0])

    intervals = sorted(tracer._usable_seed_intervals(tracer._seed_interval_candidates(seed_leaf, seed_xyz, direction)))
    common_seed = tracer._select_common_seed(seed_leaf, seed_xyz, direction)
    backward_seed, backward_leaf = tracer._select_seed_branch(seed_leaf, seed_xyz, direction, branch="backward")
    forward_seed, forward_leaf = tracer._select_seed_branch(seed_leaf, seed_xyz, direction, branch="forward")

    np.testing.assert_allclose(seed_xyz, np.array([-20.70937264, -13.09090909, 0.0], dtype=float), atol=1.0e-8, rtol=0.0)
    assert seed_leaf == 10095
    assert len(intervals) == 2
    assert {leaf_id for _, _, leaf_id in intervals} == {7479, 10095}
    assert all(enter < 0.0 < exit_ for enter, exit_, _ in intervals)
    assert common_seed is not None
    assert common_seed[1] in {7479, 10095}
    assert backward_leaf in {7479, 10095}
    assert forward_leaf in {7479, 10095}
    np.testing.assert_allclose(backward_seed, forward_seed, atol=1.0e-8, rtol=0.0)
    np.testing.assert_allclose(common_seed[0], np.array([-23.01252393, -13.09090909, 0.0], dtype=float), atol=1.0e-8, rtol=0.0)


def test_trace_returns_two_way_packed_segments_for_one_seeded_cartesian_ray() -> None:
    """Public trace should merge the backward and forward seed walks into one packed ray result."""
    tracer = OctreeRayTracer(_build_xyz_tree())
    origin = np.array([-2.0, -0.3, -0.2], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    seed_xyz = np.array([[-0.5, -0.3, -0.2]], dtype=float)

    segments = tracer.trace(origin, direction, seed_xyz=seed_xyz)

    np.testing.assert_array_equal(segments.ray_offsets, np.array([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(segments.cell_ids, np.array([0, 4], dtype=np.int64))
    np.testing.assert_allclose(segments.t_enter, np.array([1.0, 2.0], dtype=float))
    np.testing.assert_allclose(segments.t_exit, np.array([2.0, 3.0], dtype=float))


def test_trace_handles_seed_on_one_internal_cartesian_face() -> None:
    """Default Cartesian domain seeding may land on one internal face and should still trace exactly."""
    tracer = OctreeRayTracer(_build_xyz_tree())
    origin = np.array([-2.0, -0.3, -0.2], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)

    segments = tracer.trace(origin, direction)

    np.testing.assert_array_equal(segments.ray_offsets, np.array([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(segments.cell_ids, np.array([0, 4], dtype=np.int64))
    np.testing.assert_allclose(segments.t_enter, np.array([1.0, 2.0], dtype=float))
    np.testing.assert_allclose(segments.t_exit, np.array([2.0, 3.0], dtype=float))


def test_render_midpoint_image_integrates_constant_density_along_seeded_ray() -> None:
    """Midpoint rendering should reduce to segment-length summation for one constant field."""
    tree = _build_xyz_tree()
    tracer = OctreeRayTracer(tree)
    values = np.ones(int(np.max(tree.corners)) + 1, dtype=float)
    interp = OctreeInterpolator(tree, values)
    origins = np.array([[[-2.0, -0.3, -0.2]]], dtype=float)
    directions = np.array([[[1.0, 0.0, 0.0]]], dtype=float)
    seed_xyz = np.array([[[-0.5, -0.3, -0.2]]], dtype=float)

    segments = tracer.trace(origins, directions, seed_xyz=seed_xyz)
    image = render_midpoint_image(interp, origins, directions, segments)

    assert image.shape == (1, 1)
    np.testing.assert_allclose(image, np.array([[2.0]], dtype=float))


def test_render_midpoint_image_matches_cartesian_constant_reference_grid() -> None:
    """Constant-density Cartesian images should reduce exactly to box path lengths."""
    points, corners = build_cartesian_hex_mesh(
        x_edges=np.array([-1.0, 0.0, 1.0], dtype=float),
        y_edges=np.array([-1.0, 1.0], dtype=float),
        z_edges=np.array([-1.0, 1.0], dtype=float),
    )
    tree = Octree(points, corners, tree_coord="xyz")
    tracer = OctreeRayTracer(tree)
    interp = OctreeInterpolator(tree, np.ones(points.shape[0], dtype=float))

    ys = np.array([-0.5, 0.0, 0.5], dtype=float)
    zs = np.array([-0.25, 0.25], dtype=float)
    origins = np.zeros((zs.size, ys.size, 3), dtype=float)
    origins[..., 0] = -2.0
    origins[..., 1] = ys[None, :]
    origins[..., 2] = zs[:, None]
    directions = np.zeros_like(origins)
    directions[..., 0] = 1.0

    image = render_midpoint_image(interp, origins, directions, tracer.trace(origins, directions))

    np.testing.assert_allclose(image, np.full((zs.size, ys.size), 2.0, dtype=float))


def test_render_midpoint_image_matches_cartesian_linear_reference_grid() -> None:
    """One linear Cartesian field should integrate exactly under midpoint sampling."""
    points, corners = build_cartesian_hex_mesh(
        x_edges=np.array([-1.0, 0.0, 1.0], dtype=float),
        y_edges=np.array([-1.0, 1.0], dtype=float),
        z_edges=np.array([-1.0, 1.0], dtype=float),
    )
    tree = Octree(points, corners, tree_coord="xyz")
    tracer = OctreeRayTracer(tree)
    interp = OctreeInterpolator(tree, points[:, 0] + 2.0 * points[:, 1])

    ys = np.array([-0.5, 0.0, 0.5], dtype=float)
    zs = np.array([-0.25, 0.25], dtype=float)
    origins = np.zeros((zs.size, ys.size, 3), dtype=float)
    origins[..., 0] = -2.0
    origins[..., 1] = ys[None, :]
    origins[..., 2] = zs[:, None]
    directions = np.zeros_like(origins)
    directions[..., 0] = 1.0

    image = render_midpoint_image(interp, origins, directions, tracer.trace(origins, directions))

    np.testing.assert_allclose(image, 4.0 * origins[..., 1], atol=1.0e-12, rtol=0.0)


def test_render_midpoint_image_matches_spherical_shell_reference_for_off_degenerate_rays() -> None:
    """Constant-density shell rays should agree with the smooth shell to mesh accuracy."""
    points, corners = build_spherical_hex_mesh(
        nr=8,
        npolar=24,
        nazimuth=32,
        r_min=1.0,
        r_max=2.0,
    )
    tree = Octree(points, corners, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)
    interp = OctreeInterpolator(tree, np.ones(points.shape[0], dtype=float))

    impact_parameter = np.array([0.3, 0.6, 0.9, 1.2, 1.6], dtype=float)
    z_value = 0.2
    y_value = np.sqrt(impact_parameter * impact_parameter - z_value * z_value)
    origins = np.column_stack((np.full(impact_parameter.size, -5.0), y_value, np.full(impact_parameter.size, z_value)))
    directions = np.tile(np.array([1.0, 0.0, 0.0], dtype=float), (impact_parameter.size, 1))

    image = render_midpoint_image(interp, origins, directions, tracer.trace(origins, directions))
    expected = _analytic_visible_shell_length(impact_parameter, 1.0, 2.0)

    np.testing.assert_allclose(image, expected, atol=4.0e-2, rtol=0.0)


def test_render_midpoint_image_matches_sphere_chord_reference_for_constant_field() -> None:
    """Constant-density sphere rays should agree with the analytic chord length.

    The current spherical builder requires `r_min > 0`, so this test uses a tiny
    positive inner radius and only impact parameters larger than that radius.
    For those rays, the visible shell length is exactly the full-sphere chord.
    """
    r_min = 1.0e-3
    r_max = 2.0
    points, corners = build_spherical_hex_mesh(
        nr=8,
        npolar=24,
        nazimuth=32,
        r_min=r_min,
        r_max=r_max,
    )
    tree = Octree(points, corners, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)
    interp = OctreeInterpolator(tree, np.ones(points.shape[0], dtype=float))

    impact_parameter = np.array([0.3, 0.6, 0.9, 1.2, 1.6], dtype=float)
    z_value = 0.2
    y_value = np.sqrt(impact_parameter * impact_parameter - z_value * z_value)
    origins = np.column_stack((np.full(impact_parameter.size, -5.0), y_value, np.full(impact_parameter.size, z_value)))
    directions = np.tile(np.array([1.0, 0.0, 0.0], dtype=float), (impact_parameter.size, 1))

    image = render_midpoint_image(interp, origins, directions, tracer.trace(origins, directions))
    expected = _analytic_sphere_chord_length(impact_parameter, r_max)

    np.testing.assert_allclose(image, expected, atol=4.0e-2, rtol=0.0)


def test_trace_matches_shell_chord_reference_for_special_and_generic_directions() -> None:
    """Paired opposite-view shell traces should recover the full analytic shell chord."""
    r_min = 1.0
    r_max = 2.0
    points, corners = build_spherical_hex_mesh(
        nr=8,
        npolar=24,
        nazimuth=32,
        r_min=r_min,
        r_max=r_max,
    )
    tree = Octree(points, corners, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)
    interp = OctreeInterpolator(tree, np.ones(points.shape[0], dtype=float))

    direction_xyz = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=float,
    )
    direction_xyz /= np.linalg.norm(direction_xyz, axis=1, keepdims=True)
    impact_parameter = np.array([0.3, 0.9, 1.2, 1.6], dtype=float)

    forward_length = np.empty(direction_xyz.shape[0], dtype=float)
    backward_length = np.empty(direction_xyz.shape[0], dtype=float)
    for ray_id, (direction, impact) in enumerate(zip(direction_xyz, impact_parameter, strict=True)):
        offset_xyz = impact * _perpendicular_unit(direction)
        origin_forward = offset_xyz - 5.0 * direction
        origin_backward = offset_xyz + 5.0 * direction

        forward_image = render_midpoint_image(
            interp,
            origin_forward.reshape(1, 3),
            direction.reshape(1, 3),
            tracer.trace(origin_forward.reshape(1, 3), direction.reshape(1, 3)),
        )
        backward_image = render_midpoint_image(
            interp,
            origin_backward.reshape(1, 3),
            (-direction).reshape(1, 3),
            tracer.trace(origin_backward.reshape(1, 3), (-direction).reshape(1, 3)),
        )
        forward_length[ray_id] = float(forward_image[0])
        backward_length[ray_id] = float(backward_image[0])

    expected = _analytic_sphere_chord_length(impact_parameter, r_max) - _analytic_sphere_chord_length(impact_parameter, r_min)
    through_hole = impact_parameter < r_min
    shell_only = ~through_hole
    np.testing.assert_allclose(forward_length[through_hole] + backward_length[through_hole], expected[through_hole], atol=5.0e-2, rtol=0.0)
    np.testing.assert_allclose(forward_length[shell_only], expected[shell_only], atol=5.0e-2, rtol=0.0)
    np.testing.assert_allclose(backward_length[shell_only], expected[shell_only], atol=5.0e-2, rtol=0.0)


def test_trace_handles_one_spherical_symmetry_ray_with_exact_geometric_regression() -> None:
    """One exact symmetry ray should preserve the corrected traced interval geometry."""
    ds = Dataset.from_file(str(data_file("3d__var_1_n00000000.plt")))
    tree = Octree.from_ds(ds, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)
    origin = np.array([-60.0, 0.0, -12.0], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    seed_xyz = tracer.seed_domain(origin, direction)

    segments = tracer.trace(origin, direction, seed_xyz=seed_xyz)

    np.testing.assert_allclose(seed_xyz, np.array([[-21.36000974, 0.0, -12.0]], dtype=float), atol=1.0e-8, rtol=0.0)
    assert segments.cell_ids.size == 38
    assert np.all(np.diff(segments.t_enter) >= 0.0)
    assert np.all(np.diff(segments.t_exit) >= 0.0)
    np.testing.assert_allclose(segments.t_enter[0], 13.72182951, atol=1.0e-8, rtol=0.0)
    np.testing.assert_allclose(segments.t_exit[-1], 106.27817001, atol=1.0e-8, rtol=0.0)
    np.testing.assert_allclose(float(np.sum(segments.segment_length)), 92.5563404981293, atol=1.0e-10, rtol=0.0)


def test_trace_handles_one_spherical_axis_ray_with_exact_geometric_regression() -> None:
    """One spherical axis ray should preserve the corrected traced interval geometry."""
    ds = Dataset.from_file(str(data_file("3d__var_1_n00000000.plt")))
    tree = Octree.from_ds(ds, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)
    origin = np.array([-60.0, 0.0, 8.0], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    seed_xyz = tracer.seed_domain(origin, direction)

    segments = tracer.trace(origin, direction, seed_xyz=seed_xyz)

    np.testing.assert_allclose(seed_xyz, np.array([[-23.1570727, 0.0, 8.0]], dtype=float), atol=1.0e-8, rtol=0.0)
    assert segments.cell_ids.size == 43
    assert np.all(np.diff(segments.t_enter) >= 0.0)
    assert np.all(np.diff(segments.t_exit) >= 0.0)
    np.testing.assert_allclose(segments.t_enter[0], 12.78793421, atol=1.0e-8, rtol=0.0)
    np.testing.assert_allclose(segments.t_exit[-1], 107.21206448, atol=1.0e-8, rtol=0.0)
    np.testing.assert_allclose(float(np.sum(segments.segment_length)), 94.42413027227191, atol=1.0e-10, rtol=0.0)


def test_trace_matches_mirrored_real_sample_y_zero_rays() -> None:
    """Mirrored `y = 0` sample rays should have the same traced total length."""
    ds = Dataset.from_file(str(data_file("3d__var_1_n00000000.plt")))
    tree = Octree.from_ds(ds, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)

    for z_value in np.array([4.8, 6.4, 8.0, 11.2, 14.4, 20.0], dtype=float):
        positive_origin = np.array([-60.0, 0.0, z_value], dtype=float)
        negative_origin = np.array([-60.0, 0.0, -z_value], dtype=float)
        positive_segments = tracer.trace(positive_origin, direction, seed_xyz=tracer.seed_domain(positive_origin, direction))
        negative_segments = tracer.trace(negative_origin, direction, seed_xyz=tracer.seed_domain(negative_origin, direction))
        np.testing.assert_allclose(
            float(np.sum(positive_segments.segment_length)),
            float(np.sum(negative_segments.segment_length)),
            atol=2.0e-5,
            rtol=0.0,
        )


def test_render_midpoint_image_preserves_real_sample_central_symmetry_for_constant_field() -> None:
    """A constant-field sample image should keep symmetric central row and column lengths."""
    ds = Dataset.from_file(str(data_file("3d__var_1_n00000000.plt")))
    tree = Octree.from_ds(ds, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)
    interp = OctreeInterpolator(tree, np.ones(int(np.max(tree.corners)) + 1, dtype=float))
    origins, directions = camera_rays(
        origin=[-60.0, 0.0, 0.0],
        target=[0.0, 0.0, 0.0],
        up=[0.0, 0.0, 1.0],
        nx=31,
        ny=25,
        width=50.0,
        height=40.0,
        projection="parallel",
    )

    image = render_midpoint_image(interp, origins, directions, tracer.trace(origins, directions))
    center_row = image[12, :]
    center_col = image[:, 15]

    np.testing.assert_allclose(center_row, center_row[::-1], atol=2.0e-5, rtol=0.0)
    np.testing.assert_allclose(center_col, center_col[::-1], atol=2.0e-5, rtol=0.0)
