from __future__ import annotations

import math

import numpy as np
import pytest
from numba import njit

from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp import OctreeRayTracer
from batcamp import camera_rays
from batcamp import render_midpoint_image
from batcamp.ray import trace_one_ray_kernel
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


def _build_xyz_regular_tree() -> Octree:
    """Return one one-root uniform Cartesian tree spanning [-4, 4] in each axis."""
    edges = np.arange(-4.0, 5.0, dtype=float)
    points, corners = build_cartesian_hex_mesh(
        x_edges=edges,
        y_edges=edges,
        z_edges=edges,
    )
    return Octree(points, corners, tree_coord="xyz")


def _regular_mesh_points() -> np.ndarray:
    """Return logical mesh points of the regular [-4, 4]^3 grid."""
    return np.asarray(_build_xyz_regular_tree()._points, dtype=float)


def _build_xyz_stretched_tree() -> Octree:
    """Return one one-root uniform Cartesian tree with one separable nonlinear stretch."""
    regular_tree = _build_xyz_regular_tree()
    stretched_tree = object.__new__(Octree)
    stretched_tree._init_from_state(
        root_shape=regular_tree.root_shape,
        tree_coord=regular_tree.tree_coord,
        cell_levels=regular_tree.cell_levels,
        cell_ijk=regular_tree.cell_ijk[: regular_tree.corners.shape[0]],
        points=_stretch_regular_xyz(np.asarray(regular_tree._points, dtype=float)),
        corners=np.asarray(regular_tree.corners, dtype=np.int64),
    )
    return stretched_tree


def _stretch_regular_xyz(logical_xyz: np.ndarray) -> np.ndarray:
    """Return one separable nonlinear xyz stretch of regular-grid coordinates."""
    xyz = np.asarray(logical_xyz, dtype=float)
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    stretched = np.empty_like(xyz)
    stretched[..., 0] = x + 0.02 * x * x * x
    stretched[..., 1] = y + 0.015 * y * y * y
    stretched[..., 2] = z + 0.01 * z * z * z
    return stretched


def _stretched_regular_edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stretched regular-grid edge coordinates for x, y, and z."""
    edges = np.arange(-4.0, 5.0, dtype=float)
    x_edges = edges + 0.02 * edges * edges * edges
    y_edges = edges + 0.015 * edges * edges * edges
    z_edges = edges + 0.01 * edges * edges * edges
    return x_edges, y_edges, z_edges


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


def _assert_segments_match_expected_path(
    origin: np.ndarray,
    direction: np.ndarray,
    t_enter: np.ndarray,
    t_exit: np.ndarray,
    *,
    expected_exit_xyz: np.ndarray,
    expected_length: float,
) -> None:
    """Assert that traced segments are contiguous and match one expected geometric path."""
    assert t_enter.size == t_exit.size
    assert np.all(t_exit > t_enter)
    np.testing.assert_allclose(t_enter[1:], t_exit[:-1], atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(origin + t_exit[-1] * direction, expected_exit_xyz, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(
        float(np.sum((t_exit - t_enter) * np.linalg.norm(direction))),
        expected_length,
        atol=1.0e-12,
        rtol=0.0,
    )


def _expected_box_exit_and_length(
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    box_lo: np.ndarray | None = None,
    box_hi: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Return the exact box exit point and physical path length for one ray."""
    if box_lo is None:
        box_lo = np.array([-4.0, -4.0, -4.0], dtype=float)
    if box_hi is None:
        box_hi = np.array([4.0, 4.0, 4.0], dtype=float)
    t_exit = np.inf
    for axis in range(3):
        if direction[axis] > 0.0:
            t_axis = (box_hi[axis] - origin[axis]) / direction[axis]
        elif direction[axis] < 0.0:
            t_axis = (box_lo[axis] - origin[axis]) / direction[axis]
        else:
            continue
        if t_axis < t_exit:
            t_exit = float(t_axis)
    if not np.isfinite(t_exit):
        raise ValueError("direction must have at least one nonzero component.")
    return origin + t_exit * direction, float(t_exit * np.linalg.norm(direction))


@pytest.fixture(scope="module")
def xyz_regular_tracer() -> OctreeRayTracer:
    """Return one reusable tracer for the uniform [-4, 4]^3 unit grid."""
    return OctreeRayTracer(_build_xyz_regular_tree())


@pytest.fixture(scope="module")
def xyz_stretched_grid_case() -> tuple[OctreeRayTracer, np.ndarray, np.ndarray]:
    """Return one reusable tracer plus outer box bounds for one stretched uniform grid."""
    tree = _build_xyz_stretched_tree()
    box_lo, box_hi = tree.domain_bounds(coord="xyz")
    return OctreeRayTracer(tree), np.asarray(box_lo, dtype=float), np.asarray(box_hi, dtype=float)


def _lookup_regular_start_leaf(tracer: OctreeRayTracer, origin: np.ndarray, direction: np.ndarray) -> int:
    """Return one seam-aware start leaf for one regular-grid ray."""
    start_xyz = np.array(origin, dtype=float, copy=True)
    seam_mask = np.isclose(start_xyz, np.round(start_xyz), atol=1.0e-12, rtol=0.0)
    start_xyz[seam_mask] += 1.0e-12 * np.sign(direction[seam_mask])
    return int(tracer.tree.lookup_points(start_xyz[None, :], coord="xyz")[0])


def _assert_regular_ray_matches_expected_path(
    tracer: OctreeRayTracer,
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    start_leaf: int | None = None,
    expected_first_t: float = 0.0,
    expected_exit_xyz: np.ndarray | None = None,
    expected_length: float | None = None,
    box_lo: np.ndarray | None = None,
    box_hi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace one regular-grid ray and assert that it matches the exact box path."""
    if start_leaf is None:
        start_leaf = _lookup_regular_start_leaf(tracer, origin, direction)

    leaf_ids, t_enter, t_exit = trace_one_ray_kernel(
        start_leaf,
        origin,
        direction,
        0.0,
        np.inf,
        tracer.trace_kernel_state(),
    )

    assert leaf_ids.size > 0, f"no traced segments for origin={origin.tolist()} direction={direction.tolist()}"
    np.testing.assert_allclose(t_enter[0], expected_first_t, atol=1.0e-12, rtol=0.0)
    if expected_exit_xyz is None or expected_length is None:
        expected_exit_xyz, expected_length = _expected_box_exit_and_length(origin, direction, box_lo=box_lo, box_hi=box_hi)
    _assert_segments_match_expected_path(
        origin,
        direction,
        t_enter,
        t_exit,
        expected_exit_xyz=expected_exit_xyz,
        expected_length=expected_length,
    )
    return leaf_ids, t_enter, t_exit


def _assert_regular_corner_sweep(tracer: OctreeRayTracer, origin: np.ndarray) -> None:
    """Assert that one regular-grid start point traces correctly toward every mesh corner."""
    for target in tracer.tree._points:
        direction = np.asarray(target - origin, dtype=float)
        if np.allclose(direction, 0.0):
            continue
        _assert_regular_ray_matches_expected_path(tracer, origin, direction)


def _lookup_stretched_start_leaf(
    tracer: OctreeRayTracer,
    logical_origin: np.ndarray,
    logical_direction: np.ndarray,
) -> int:
    """Return one seam-aware start leaf for one stretched-grid ray."""
    start_logical = np.array(logical_origin, dtype=float, copy=True)
    seam_mask = np.isclose(start_logical, np.round(start_logical), atol=1.0e-12, rtol=0.0)
    start_logical[seam_mask] += 1.0e-12 * np.sign(logical_direction[seam_mask])
    start_xyz = _stretch_regular_xyz(start_logical)
    return int(tracer.tree.lookup_points(start_xyz[None, :], coord="xyz")[0])


def _assert_stretched_corner_sweep(
    tracer: OctreeRayTracer,
    logical_origin: np.ndarray,
    *,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
) -> None:
    """Assert that one stretched-grid start point traces correctly toward every mesh corner."""
    origin = _stretch_regular_xyz(np.asarray(logical_origin, dtype=float))
    for logical_target in _regular_mesh_points():
        logical_direction = np.asarray(logical_target - logical_origin, dtype=float)
        if np.allclose(logical_direction, 0.0):
            continue
        direction = _stretched_local_direction(logical_origin, logical_target)
        _assert_regular_ray_matches_expected_path(
            tracer,
            origin,
            direction,
            start_leaf=_lookup_stretched_start_leaf(tracer, logical_origin, logical_direction),
            box_lo=box_lo,
            box_hi=box_hi,
        )


def _stretched_local_direction(logical_origin: np.ndarray, logical_target: np.ndarray) -> np.ndarray:
    """Return one physical-space direction toward the stretched image of one logical target."""
    physical_origin = _stretch_regular_xyz(np.asarray(logical_origin, dtype=float))
    physical_target = _stretch_regular_xyz(np.asarray(logical_target, dtype=float))
    return np.asarray(physical_target - physical_origin, dtype=float)


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
    """Off-axis shell rays should either miss or seed on the visible front interval."""
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


def test_trace_one_ray_kernel_walks_same_level_cells_exactly() -> None:
    """One simple Cartesian ray should cross two same-level leaves with exact intervals."""
    tracer = OctreeRayTracer(_build_xyz_tree())
    leaf_ids, t_enter, t_exit = trace_one_ray_kernel(
        0,
        np.array([-2.0, -0.3, -0.2], dtype=float),
        np.array([1.0, 0.0, 0.0], dtype=float),
        0.0,
        np.inf,
        tracer.trace_kernel_state(),
    )

    np.testing.assert_array_equal(leaf_ids, np.array([0, 4], dtype=np.int64))
    np.testing.assert_allclose(t_enter, np.array([1.0, 2.0], dtype=float))
    np.testing.assert_allclose(t_exit, np.array([2.0, 3.0], dtype=float))


def test_trace_one_ray_kernel_walks_full_regular_cartesian_line_exactly(xyz_regular_tracer: OctreeRayTracer) -> None:
    """One same-level Cartesian ray should cross all eight x-cells of the one-root [-4, 4]^3 grid."""
    origin = np.array([-5.0, -3.5, -3.5], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    start_leaf = int(
        xyz_regular_tracer.tree.lookup_points(np.array([[-3.5, -3.5, -3.5]], dtype=float), coord="xyz")[0]
    )

    leaf_ids, t_enter, t_exit = _assert_regular_ray_matches_expected_path(
        xyz_regular_tracer,
        origin,
        direction,
        start_leaf=start_leaf,
        expected_first_t=1.0,
        expected_exit_xyz=np.array([4.0, -3.5, -3.5], dtype=float),
        expected_length=8.0,
    )

    assert xyz_regular_tracer.tree.root_shape == (1, 1, 1)
    np.testing.assert_array_equal(np.unique(xyz_regular_tracer.tree.cell_levels), np.array([3], dtype=np.int64))
    np.testing.assert_array_equal(leaf_ids, np.arange(0, 512, 64, dtype=np.int64))
    np.testing.assert_allclose(t_enter, 1.0 + np.arange(8, dtype=float))
    np.testing.assert_allclose(t_exit, 2.0 + np.arange(8, dtype=float))


@pytest.mark.parametrize(
    "direction",
    [
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
    ],
    ids=[
        "---",
        "--+",
        "-+-",
        "-++",
        "+--",
        "+-+",
        "++-",
        "+++",
    ],
)
def test_trace_one_ray_kernel_handles_all_space_diagonal_regular_rays(
    xyz_regular_tracer: OctreeRayTracer, direction: tuple[float, float, float]
) -> None:
    """All unit-grid space diagonals from one interior point should trace contiguously to the expected box exit."""
    origin = np.array([0.5, 0.5, 0.5], dtype=float)
    _assert_regular_ray_matches_expected_path(xyz_regular_tracer, origin, np.array(direction, dtype=float))


@pytest.mark.parametrize(
    "direction",
    [
        (-1.0, -1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, 0.0, -1.0),
        (-1.0, 0.0, 1.0),
        (1.0, 0.0, -1.0),
        (1.0, 0.0, 1.0),
        (0.0, -1.0, -1.0),
        (0.0, -1.0, 1.0),
        (0.0, 1.0, -1.0),
        (0.0, 1.0, 1.0),
    ],
    ids=[
        "xy--",
        "xy-+",
        "xy+-",
        "xy++",
        "xz--",
        "xz-+",
        "xz+-",
        "xz++",
        "yz--",
        "yz-+",
        "yz+-",
        "yz++",
    ],
)
def test_trace_one_ray_kernel_handles_all_plane_diagonal_regular_rays(
    xyz_regular_tracer: OctreeRayTracer, direction: tuple[float, float, float]
) -> None:
    """All unit-grid plane diagonals from one interior point should trace contiguously to the expected box exit."""
    origin = np.array([0.5, 0.5, 0.5], dtype=float)
    _assert_regular_ray_matches_expected_path(xyz_regular_tracer, origin, np.array(direction, dtype=float))


def test_trace_one_ray_kernel_handles_all_regular_corner_directions_from_cell_center(
    xyz_regular_tracer: OctreeRayTracer,
) -> None:
    """All rays from one unit-cell center toward mesh corners should trace contiguously to the expected box exit."""
    origin = np.array([0.5, 0.5, 0.5], dtype=float)
    _assert_regular_corner_sweep(xyz_regular_tracer, origin)


def test_trace_one_ray_kernel_handles_all_regular_corner_directions_from_x_face_seam(
    xyz_regular_tracer: OctreeRayTracer,
) -> None:
    """All rays from one face-seam point toward mesh corners should trace contiguously to the expected box exit."""
    origin = np.array([0.5, 0.0, 0.0], dtype=float)
    _assert_regular_corner_sweep(xyz_regular_tracer, origin)


def test_trace_one_ray_kernel_handles_all_regular_corner_directions_from_xy_edge_seam(
    xyz_regular_tracer: OctreeRayTracer,
) -> None:
    """All rays from one edge-seam point toward mesh corners should trace contiguously to the expected box exit."""
    origin = np.array([0.5, 0.5, 0.0], dtype=float)
    _assert_regular_corner_sweep(xyz_regular_tracer, origin)


@pytest.mark.parametrize(
    "direction",
    [
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
    ],
    ids=[
        "origin---",
        "origin--+",
        "origin-+-",
        "origin-++",
        "origin+--",
        "origin+-+",
        "origin++-",
        "origin+++",
    ],
)
def test_trace_one_ray_kernel_handles_origin_space_diagonal_regular_rays(
    xyz_regular_tracer: OctreeRayTracer, direction: tuple[float, float, float]
) -> None:
    """All space diagonals from the true mesh corner origin should trace contiguously to the expected box exit."""
    origin = np.array([0.0, 0.0, 0.0], dtype=float)
    _assert_regular_ray_matches_expected_path(xyz_regular_tracer, origin, np.array(direction, dtype=float))


@pytest.mark.parametrize(
    "direction",
    [
        (-1.0, -1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, 0.0, -1.0),
        (-1.0, 0.0, 1.0),
        (1.0, 0.0, -1.0),
        (1.0, 0.0, 1.0),
        (0.0, -1.0, -1.0),
        (0.0, -1.0, 1.0),
        (0.0, 1.0, -1.0),
        (0.0, 1.0, 1.0),
    ],
    ids=[
        "origin- -0".replace(" ", ""),
        "origin-+0",
        "origin+-0",
        "origin++0",
        "origin-0-",
        "origin-0+",
        "origin+0-",
        "origin+0+",
        "origin0--",
        "origin0-+",
        "origin0+-",
        "origin0++",
    ],
)
def test_trace_one_ray_kernel_handles_origin_plane_diagonal_regular_rays(
    xyz_regular_tracer: OctreeRayTracer, direction: tuple[float, float, float]
) -> None:
    """All plane diagonals from the true mesh corner origin should trace contiguously to the expected box exit."""
    origin = np.array([0.0, 0.0, 0.0], dtype=float)
    _assert_regular_ray_matches_expected_path(xyz_regular_tracer, origin, np.array(direction, dtype=float))


def test_trace_one_ray_kernel_handles_all_regular_corner_directions_from_origin(
    xyz_regular_tracer: OctreeRayTracer,
) -> None:
    """All rays from the true mesh corner origin toward mesh corners should trace contiguously to the expected box exit."""
    origin = np.array([0.0, 0.0, 0.0], dtype=float)
    _assert_regular_corner_sweep(xyz_regular_tracer, origin)


@pytest.mark.parametrize(
    "origin",
    [
        np.array([4.0, 0.5, 0.5], dtype=float),
        np.array([0.5, 4.0, 0.5], dtype=float),
        np.array([0.5, 0.5, 4.0], dtype=float),
        np.array([4.0, 4.0, 0.5], dtype=float),
        np.array([4.0, 0.5, 4.0], dtype=float),
        np.array([0.5, 4.0, 4.0], dtype=float),
        np.array([4.0, 4.0, 4.0], dtype=float),
    ],
    ids=[
        "boundary-x",
        "boundary-y",
        "boundary-z",
        "boundary-xy",
        "boundary-xz",
        "boundary-yz",
        "boundary-xyz",
    ],
)
def test_trace_one_ray_kernel_handles_all_regular_corner_directions_from_domain_boundary(
    xyz_regular_tracer: OctreeRayTracer,
    origin: np.ndarray,
) -> None:
    """All rays from one nonwarped domain-boundary point toward mesh corners should trace contiguously to the expected box exit."""
    _assert_regular_corner_sweep(xyz_regular_tracer, origin)


def test_trace_one_ray_kernel_handles_random_regular_rays(xyz_regular_tracer: OctreeRayTracer) -> None:
    """One fixed-seed batch of random interior starts and directions should trace to the exact box exit."""
    rng = np.random.default_rng(20260410)
    origins = rng.uniform(-3.75, 3.75, size=(64, 3))
    directions = rng.normal(size=(64, 3))
    for ray_id in range(directions.shape[0]):
        if np.linalg.norm(directions[ray_id]) <= 1.0e-12:
            directions[ray_id] = np.array([1.0, 0.0, 0.0], dtype=float)
        _assert_regular_ray_matches_expected_path(
            xyz_regular_tracer,
            np.asarray(origins[ray_id], dtype=float),
            np.asarray(directions[ray_id], dtype=float),
        )


def test_trace_one_ray_kernel_walks_full_stretched_cartesian_line_exactly(
    xyz_stretched_grid_case: tuple[OctreeRayTracer, np.ndarray, np.ndarray],
) -> None:
    """One same-level Cartesian ray should cross all eight stretched x-cells of the one-root grid."""
    tracer, _, _ = xyz_stretched_grid_case
    x_edges, _, _ = _stretched_regular_edges()
    logical_origin = np.array([-5.0, -3.5, -3.5], dtype=float)
    origin = _stretch_regular_xyz(logical_origin)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    start_xyz = _stretch_regular_xyz(np.array([-3.5, -3.5, -3.5], dtype=float))
    start_leaf = int(tracer.tree.lookup_points(start_xyz[None, :], coord="xyz")[0])

    leaf_ids, t_enter, t_exit = _assert_regular_ray_matches_expected_path(
        tracer,
        origin,
        direction,
        start_leaf=start_leaf,
        expected_first_t=float(x_edges[0] - origin[0]),
        expected_exit_xyz=np.array([x_edges[-1], origin[1], origin[2]], dtype=float),
        expected_length=float(x_edges[-1] - x_edges[0]),
    )

    assert tracer.tree.root_shape == (1, 1, 1)
    np.testing.assert_array_equal(np.unique(tracer.tree.cell_levels), np.array([3], dtype=np.int64))
    np.testing.assert_array_equal(leaf_ids, np.arange(0, 512, 64, dtype=np.int64))
    np.testing.assert_allclose(t_enter, x_edges[:-1] - origin[0], atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(t_exit, x_edges[1:] - origin[0], atol=1.0e-12, rtol=0.0)


@pytest.mark.parametrize(
    "direction",
    [
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
    ],
    ids=[
        "stretched---",
        "stretched--+",
        "stretched-+-",
        "stretched-++",
        "stretched+--",
        "stretched+-+",
        "stretched++-",
        "stretched+++",
    ],
)
def test_trace_one_ray_kernel_handles_all_space_diagonal_stretched_rays(
    xyz_stretched_grid_case: tuple[OctreeRayTracer, np.ndarray, np.ndarray],
    direction: tuple[float, float, float],
) -> None:
    """All stretched local space diagonals from one interior point should trace contiguously to the box exit."""
    tracer, box_lo, box_hi = xyz_stretched_grid_case
    logical_origin = np.array([0.5, 0.5, 0.5], dtype=float)
    logical_direction = np.array(direction, dtype=float)
    origin = _stretch_regular_xyz(logical_origin)
    direction_xyz = _stretched_local_direction(logical_origin, logical_origin + 0.5 * logical_direction)
    _assert_regular_ray_matches_expected_path(
        tracer,
        origin,
        direction_xyz,
        start_leaf=_lookup_stretched_start_leaf(tracer, logical_origin, logical_direction),
        box_lo=box_lo,
        box_hi=box_hi,
    )


@pytest.mark.parametrize(
    "direction",
    [
        (-1.0, -1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, 0.0, -1.0),
        (-1.0, 0.0, 1.0),
        (1.0, 0.0, -1.0),
        (1.0, 0.0, 1.0),
        (0.0, -1.0, -1.0),
        (0.0, -1.0, 1.0),
        (0.0, 1.0, -1.0),
        (0.0, 1.0, 1.0),
    ],
    ids=[
        "stretched-xy--",
        "stretched-xy-+",
        "stretched-xy+-",
        "stretched-xy++",
        "stretched-xz--",
        "stretched-xz-+",
        "stretched-xz+-",
        "stretched-xz++",
        "stretched-yz--",
        "stretched-yz-+",
        "stretched-yz+-",
        "stretched-yz++",
    ],
)
def test_trace_one_ray_kernel_handles_all_plane_diagonal_stretched_rays(
    xyz_stretched_grid_case: tuple[OctreeRayTracer, np.ndarray, np.ndarray],
    direction: tuple[float, float, float],
) -> None:
    """All stretched local plane diagonals from one interior point should trace contiguously to the box exit."""
    tracer, box_lo, box_hi = xyz_stretched_grid_case
    logical_origin = np.array([0.5, 0.5, 0.5], dtype=float)
    logical_direction = np.array(direction, dtype=float)
    origin = _stretch_regular_xyz(logical_origin)
    direction_xyz = _stretched_local_direction(logical_origin, logical_origin + 0.5 * logical_direction)
    _assert_regular_ray_matches_expected_path(
        tracer,
        origin,
        direction_xyz,
        start_leaf=_lookup_stretched_start_leaf(tracer, logical_origin, logical_direction),
        box_lo=box_lo,
        box_hi=box_hi,
    )


def test_trace_one_ray_kernel_handles_all_stretched_corner_directions_from_cell_center(
    xyz_stretched_grid_case: tuple[OctreeRayTracer, np.ndarray, np.ndarray],
) -> None:
    """All rays from one stretched unit-cell center toward stretched mesh corners should trace correctly."""
    tracer, box_lo, box_hi = xyz_stretched_grid_case
    _assert_stretched_corner_sweep(tracer, np.array([0.5, 0.5, 0.5], dtype=float), box_lo=box_lo, box_hi=box_hi)


def test_trace_one_ray_kernel_handles_all_stretched_corner_directions_from_x_face_seam(
    xyz_stretched_grid_case: tuple[OctreeRayTracer, np.ndarray, np.ndarray],
) -> None:
    """All rays from one stretched face-seam point toward stretched mesh corners should trace correctly."""
    tracer, box_lo, box_hi = xyz_stretched_grid_case
    _assert_stretched_corner_sweep(tracer, np.array([0.5, 0.0, 0.0], dtype=float), box_lo=box_lo, box_hi=box_hi)


def test_trace_one_ray_kernel_handles_all_stretched_corner_directions_from_xy_edge_seam(
    xyz_stretched_grid_case: tuple[OctreeRayTracer, np.ndarray, np.ndarray],
) -> None:
    """All rays from one stretched edge-seam point toward stretched mesh corners should trace correctly."""
    tracer, box_lo, box_hi = xyz_stretched_grid_case
    _assert_stretched_corner_sweep(tracer, np.array([0.5, 0.5, 0.0], dtype=float), box_lo=box_lo, box_hi=box_hi)


@pytest.mark.parametrize(
    "direction",
    [
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
    ],
    ids=[
        "stretched-origin---",
        "stretched-origin--+",
        "stretched-origin-+-",
        "stretched-origin-++",
        "stretched-origin+--",
        "stretched-origin+-+",
        "stretched-origin++-",
        "stretched-origin+++",
    ],
)
def test_trace_one_ray_kernel_handles_origin_space_diagonal_stretched_rays(
    xyz_stretched_grid_case: tuple[OctreeRayTracer, np.ndarray, np.ndarray],
    direction: tuple[float, float, float],
) -> None:
    """All stretched space diagonals from the true mesh corner origin should trace contiguously to the box exit."""
    tracer, box_lo, box_hi = xyz_stretched_grid_case
    logical_origin = np.array([0.0, 0.0, 0.0], dtype=float)
    logical_direction = np.array(direction, dtype=float)
    origin = _stretch_regular_xyz(logical_origin)
    direction_xyz = _stretched_local_direction(logical_origin, logical_origin + logical_direction)
    _assert_regular_ray_matches_expected_path(
        tracer,
        origin,
        direction_xyz,
        start_leaf=_lookup_stretched_start_leaf(tracer, logical_origin, logical_direction),
        box_lo=box_lo,
        box_hi=box_hi,
    )


@pytest.mark.parametrize(
    "direction",
    [
        (-1.0, -1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, 0.0, -1.0),
        (-1.0, 0.0, 1.0),
        (1.0, 0.0, -1.0),
        (1.0, 0.0, 1.0),
        (0.0, -1.0, -1.0),
        (0.0, -1.0, 1.0),
        (0.0, 1.0, -1.0),
        (0.0, 1.0, 1.0),
    ],
    ids=[
        "stretched-origin- -0".replace(" ", ""),
        "stretched-origin-+0",
        "stretched-origin+-0",
        "stretched-origin++0",
        "stretched-origin-0-",
        "stretched-origin-0+",
        "stretched-origin+0-",
        "stretched-origin+0+",
        "stretched-origin0--",
        "stretched-origin0-+",
        "stretched-origin0+-",
        "stretched-origin0++",
    ],
)
def test_trace_one_ray_kernel_handles_origin_plane_diagonal_stretched_rays(
    xyz_stretched_grid_case: tuple[OctreeRayTracer, np.ndarray, np.ndarray],
    direction: tuple[float, float, float],
) -> None:
    """All stretched plane diagonals from the true mesh corner origin should trace contiguously to the box exit."""
    tracer, box_lo, box_hi = xyz_stretched_grid_case
    logical_origin = np.array([0.0, 0.0, 0.0], dtype=float)
    logical_direction = np.array(direction, dtype=float)
    origin = _stretch_regular_xyz(logical_origin)
    direction_xyz = _stretched_local_direction(logical_origin, logical_origin + logical_direction)
    _assert_regular_ray_matches_expected_path(
        tracer,
        origin,
        direction_xyz,
        start_leaf=_lookup_stretched_start_leaf(tracer, logical_origin, logical_direction),
        box_lo=box_lo,
        box_hi=box_hi,
    )


def test_trace_one_ray_kernel_handles_all_stretched_corner_directions_from_origin(
    xyz_stretched_grid_case: tuple[OctreeRayTracer, np.ndarray, np.ndarray],
) -> None:
    """All rays from the true stretched mesh corner origin toward stretched mesh corners should trace correctly."""
    tracer, box_lo, box_hi = xyz_stretched_grid_case
    _assert_stretched_corner_sweep(tracer, np.array([0.0, 0.0, 0.0], dtype=float), box_lo=box_lo, box_hi=box_hi)


@pytest.mark.parametrize(
    "logical_origin",
    [
        np.array([4.0, 0.5, 0.5], dtype=float),
        np.array([0.5, 4.0, 0.5], dtype=float),
        np.array([0.5, 0.5, 4.0], dtype=float),
        np.array([4.0, 4.0, 0.5], dtype=float),
        np.array([4.0, 0.5, 4.0], dtype=float),
        np.array([0.5, 4.0, 4.0], dtype=float),
        np.array([4.0, 4.0, 4.0], dtype=float),
    ],
    ids=[
        "stretched-boundary-x",
        "stretched-boundary-y",
        "stretched-boundary-z",
        "stretched-boundary-xy",
        "stretched-boundary-xz",
        "stretched-boundary-yz",
        "stretched-boundary-xyz",
    ],
)
def test_trace_one_ray_kernel_handles_all_stretched_corner_directions_from_domain_boundary(
    xyz_stretched_grid_case: tuple[OctreeRayTracer, np.ndarray, np.ndarray],
    logical_origin: np.ndarray,
) -> None:
    """All rays from one stretched domain-boundary point toward stretched mesh corners should trace correctly."""
    tracer, box_lo, box_hi = xyz_stretched_grid_case
    _assert_stretched_corner_sweep(tracer, logical_origin, box_lo=box_lo, box_hi=box_hi)


def test_trace_one_ray_kernel_walks_coarse_to_fine_cells_exactly() -> None:
    """One Cartesian ray should walk from one coarse leaf into two finer leaves."""
    tracer = OctreeRayTracer(_build_xyz_coarse_fine_tree())
    leaf_ids, t_enter, t_exit = trace_one_ray_kernel(
        0,
        np.array([-0.1, 0.1875, 0.1875], dtype=float),
        np.array([1.0, 0.0, 0.0], dtype=float),
        0.0,
        np.inf,
        tracer.trace_kernel_state(),
    )

    np.testing.assert_array_equal(leaf_ids, np.array([0, 7, 11], dtype=np.int64))
    np.testing.assert_allclose(t_enter, np.array([0.1, 0.6, 0.85], dtype=float))
    np.testing.assert_allclose(t_exit, np.array([0.6, 0.85, 1.1], dtype=float))


def test_trace_one_ray_kernel_walks_fine_to_coarse_cells_exactly() -> None:
    """One Cartesian ray should walk from finer leaves back into one coarse leaf."""
    tracer = OctreeRayTracer(_build_xyz_coarse_fine_tree())
    leaf_ids, t_enter, t_exit = trace_one_ray_kernel(
        11,
        np.array([1.1, 0.1875, 0.1875], dtype=float),
        np.array([-1.0, 0.0, 0.0], dtype=float),
        0.0,
        np.inf,
        tracer.trace_kernel_state(),
    )

    np.testing.assert_array_equal(leaf_ids, np.array([11, 7, 0], dtype=np.int64))
    np.testing.assert_allclose(t_enter, np.array([0.1, 0.35, 0.6], dtype=float))
    np.testing.assert_allclose(t_exit, np.array([0.35, 0.6, 1.1], dtype=float))


@pytest.mark.parametrize(
    ("origin", "direction", "expected_leaf_ids", "expected_t_enter", "expected_t_exit"),
    [
        (
            np.array([-2.0, 0.0, -0.2], dtype=float),
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0, 4], dtype=np.int64),
            np.array([1.0, 2.0], dtype=float),
            np.array([2.0, 3.0], dtype=float),
        ),
        (
            np.array([-2.0, 0.0, 0.0], dtype=float),
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0, 4], dtype=np.int64),
            np.array([1.0, 2.0], dtype=float),
            np.array([2.0, 3.0], dtype=float),
        ),
        (
            np.array([-0.5, -0.375, -0.25], dtype=float),
            np.array([1.0, 0.75, 0.5], dtype=float),
            np.array([0, 7], dtype=np.int64),
            np.array([0.0, 0.5], dtype=float),
            np.array([0.5, 1.5], dtype=float),
        ),
    ],
    ids=["uniform-face-plane", "uniform-edge-line", "uniform-corner-crossing"],
)
def test_trace_one_ray_kernel_handles_exact_uniform_cartesian_degeneracies(
    origin: np.ndarray,
    direction: np.ndarray,
    expected_leaf_ids: np.ndarray,
    expected_t_enter: np.ndarray,
    expected_t_exit: np.ndarray,
) -> None:
    """Exact face-, edge-, and corner-degenerate Cartesian walks should stay deterministic."""
    tracer = OctreeRayTracer(_build_xyz_tree())

    leaf_ids, t_enter, t_exit = trace_one_ray_kernel(
        0,
        origin,
        direction,
        0.0,
        np.inf,
        tracer.trace_kernel_state(),
    )

    np.testing.assert_array_equal(leaf_ids, expected_leaf_ids)
    np.testing.assert_allclose(t_enter, expected_t_enter)
    np.testing.assert_allclose(t_exit, expected_t_exit)


@pytest.mark.parametrize(
    ("origin", "direction"),
    [
        (
            np.array([-0.1, 0.1875, 0.25], dtype=float),
            np.array([1.0, 0.0, 0.0], dtype=float),
        ),
        (
            np.array([-0.1, 0.25, 0.25], dtype=float),
            np.array([1.0, 0.0, 0.0], dtype=float),
        ),
    ],
    ids=["refined-face-plane", "refined-edge-line"],
)
def test_trace_one_ray_kernel_handles_exact_refined_cartesian_degeneracies(
    origin: np.ndarray,
    direction: np.ndarray,
) -> None:
    """Exact coarse-to-fine face and edge crossings should keep the current deterministic path."""
    tracer = OctreeRayTracer(_build_xyz_coarse_fine_tree())

    leaf_ids, t_enter, t_exit = trace_one_ray_kernel(
        0,
        origin,
        direction,
        0.0,
        np.inf,
        tracer.trace_kernel_state(),
    )

    np.testing.assert_array_equal(leaf_ids, np.array([0, 7, 11], dtype=np.int64))
    np.testing.assert_allclose(t_enter, np.array([0.1, 0.6, 0.85], dtype=float))
    np.testing.assert_allclose(t_exit, np.array([0.6, 0.85, 1.1], dtype=float))


def test_trace_one_ray_kernel_uses_spherical_sample_file_from_seed_leaf() -> None:
    """One forward spherical sample-file walk should reproduce the current exact segment contract."""
    ds = Dataset.from_file(str(data_file("3d__var_1_n00000000.plt")))
    tree = Octree.from_ds(ds, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)
    origin = np.array([-60.0, 0.5, 0.25], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    seed_xyz = tracer.seed_domain(origin, direction)
    start_leaf = int(tree.lookup_points(seed_xyz, coord="xyz")[0])

    leaf_ids, t_enter, t_exit = trace_one_ray_kernel(
        start_leaf,
        origin,
        direction,
        0.0,
        np.inf,
        tracer.trace_kernel_state(),
    )
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


def test_public_trace_one_ray_kernel_is_jittable() -> None:
    """The public one-ray tracing kernel should compile against raw tracer state."""
    tracer = OctreeRayTracer(_build_xyz_tree())
    origin = np.array([-2.0, -0.3, -0.2], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    seed_xyz = tracer.seed_domain(origin, direction)
    start_leaf = int(tracer.tree.lookup_points(seed_xyz, coord="xyz")[0])
    trace_state = tracer.trace_kernel_state()
    expected_leaf_ids, expected_t_enter, expected_t_exit = trace_one_ray_kernel(
        start_leaf,
        origin,
        direction,
        0.0,
        np.inf,
        trace_state,
    )

    @njit(cache=True)
    def run_trace(
        start_leaf_id: int,
        origin_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        state: tuple,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return trace_one_ray_kernel(start_leaf_id, origin_xyz, direction_xyz, 0.0, np.inf, state)

    leaf_ids, t_enter, t_exit = run_trace(start_leaf, origin, direction, trace_state)

    np.testing.assert_array_equal(leaf_ids, expected_leaf_ids)
    np.testing.assert_allclose(t_enter, expected_t_enter, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(t_exit, expected_t_exit, atol=0.0, rtol=0.0)
    assert run_trace.signatures
    assert trace_one_ray_kernel.signatures


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


def test_trace_matches_shell_chord_reference_for_exact_center_and_tangent_cases() -> None:
    """Exact center and tangent shell rays should match the analytic shell chord."""
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

    impact_parameter = np.array([0.0, r_min, r_max], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    forward_length = np.empty(impact_parameter.size, dtype=float)
    backward_length = np.empty(impact_parameter.size, dtype=float)
    for ray_id, impact in enumerate(impact_parameter):
        offset_xyz = np.array([0.0, impact, 0.0], dtype=float)
        origin_forward = offset_xyz - 5.0 * direction
        origin_backward = offset_xyz + 5.0 * direction
        forward_segments = tracer.trace(origin_forward.reshape(1, 3), direction.reshape(1, 3))
        backward_segments = tracer.trace(origin_backward.reshape(1, 3), (-direction).reshape(1, 3))
        forward_length[ray_id] = float(
            render_midpoint_image(interp, origin_forward.reshape(1, 3), direction.reshape(1, 3), forward_segments)[0]
        )
        backward_length[ray_id] = float(
            render_midpoint_image(interp, origin_backward.reshape(1, 3), (-direction).reshape(1, 3), backward_segments)[0]
        )

    expected_shell = _analytic_sphere_chord_length(impact_parameter, r_max) - _analytic_sphere_chord_length(impact_parameter, r_min)
    np.testing.assert_allclose(forward_length[0] + backward_length[0], expected_shell[0], atol=5.0e-2, rtol=0.0)
    np.testing.assert_allclose(forward_length[1], _analytic_sphere_chord_length(np.array([r_min]), r_max)[0], atol=5.0e-2, rtol=0.0)
    np.testing.assert_allclose(backward_length[1], _analytic_sphere_chord_length(np.array([r_min]), r_max)[0], atol=5.0e-2, rtol=0.0)
    np.testing.assert_allclose(forward_length[2], 0.0, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(backward_length[2], 0.0, atol=1.0e-12, rtol=0.0)


def test_seed_domain_rpa_reports_inside_hole_start_as_nan() -> None:
    """Rays starting inside the opaque inner hole should expose no visible interval."""
    points, corners = build_spherical_hex_mesh(
        nr=8,
        npolar=24,
        nazimuth=32,
        r_min=1.0,
        r_max=2.0,
    )
    tree = Octree(points, corners, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)

    seed_xyz = tracer.seed_domain(np.array([0.5, 0.0, 0.0], dtype=float), np.array([1.0, 0.0, 0.0], dtype=float))

    assert np.all(np.isnan(seed_xyz))


def test_render_midpoint_image_from_inside_shell_matches_inner_boundary_distance() -> None:
    """A one-sided shell ray starting inside the shell should stop at the inner boundary."""
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
    origin = np.array([-1.5, 0.0, 0.0], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)

    image = render_midpoint_image(
        interp,
        origin.reshape(1, 3),
        direction.reshape(1, 3),
        tracer.trace(origin.reshape(1, 3), direction.reshape(1, 3)),
    )

    np.testing.assert_allclose(image, np.array([0.5], dtype=float), atol=5.0e-4, rtol=0.0)


def test_trace_handles_one_spherical_symmetry_ray_with_exact_geometric_regression() -> None:
    """One exact symmetry ray should preserve the corrected traced interval geometry."""
    ds = Dataset.from_file(str(data_file("3d__var_1_n00000000.plt")))
    tree = Octree.from_ds(ds, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)
    origin = np.array([-60.0, 0.0, -12.0], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    seed_xyz = tracer.seed_domain(origin, direction)

    segments = tracer.trace(origin, direction, seed_xyz=seed_xyz)

    np.testing.assert_allclose(seed_xyz, np.array([[0.0, 0.0, -12.0]], dtype=float), atol=1.0e-12, rtol=0.0)
    assert np.all(np.diff(segments.t_enter) >= 0.0)
    assert np.all(np.diff(segments.t_exit) >= 0.0)
    np.testing.assert_allclose(segments.t_enter[0], 13.72182951, atol=1.0e-6, rtol=0.0)
    np.testing.assert_allclose(segments.t_exit[-1], 106.27817001, atol=1.0e-6, rtol=0.0)
    np.testing.assert_allclose(float(np.sum(segments.segment_length)), 92.5563404981293, atol=1.0e-6, rtol=0.0)


def test_trace_handles_one_spherical_axis_ray_with_exact_geometric_regression() -> None:
    """One spherical axis ray should preserve the corrected traced interval geometry."""
    ds = Dataset.from_file(str(data_file("3d__var_1_n00000000.plt")))
    tree = Octree.from_ds(ds, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)
    origin = np.array([-60.0, 0.0, 8.0], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    seed_xyz = tracer.seed_domain(origin, direction)

    segments = tracer.trace(origin, direction, seed_xyz=seed_xyz)

    np.testing.assert_allclose(seed_xyz, np.array([[0.0, 0.0, 8.0]], dtype=float), atol=1.0e-12, rtol=0.0)
    assert np.all(np.diff(segments.t_enter) >= 0.0)
    assert np.all(np.diff(segments.t_exit) >= 0.0)
    np.testing.assert_allclose(segments.t_enter[0], 12.78793421, atol=2.0e-6, rtol=0.0)
    np.testing.assert_allclose(segments.t_exit[-1], 107.21206448, atol=1.0e-6, rtol=0.0)
    np.testing.assert_allclose(float(np.sum(segments.segment_length)), 94.42413027227191, atol=3.0e-6, rtol=0.0)


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
            atol=5.0e-5,
            rtol=0.0,
        )


def test_trace_handles_batched_real_sample_shell_only_symmetric_rays() -> None:
    """One small batched sample patch should trace shell-only symmetry rays without bootstrap failure."""
    ds = Dataset.from_file(str(data_file("3d__var_1_n00000000.plt")))
    tree = Octree.from_ds(ds, tree_coord="rpa")
    tracer = OctreeRayTracer(tree)

    origins = np.array(
        [
            [[-48.0, 22.4, -28.8], [-48.0, 22.4, -22.4]],
            [[-48.0, 28.8, -28.8], [-48.0, 28.8, -22.4]],
        ],
        dtype=float,
    )
    directions = np.zeros_like(origins)
    directions[..., 0] = 1.0

    seed_xyz = tracer.seed_domain(origins, directions)
    np.testing.assert_allclose(seed_xyz[0, 1], np.array([0.0, 22.4, -22.4], dtype=float), atol=1.0e-12, rtol=0.0)

    segments = tracer.trace(origins, directions, seed_xyz=seed_xyz)
    counts = np.diff(segments.ray_offsets).reshape(2, 2)

    assert counts[0, 1] > 0
    assert counts[0, 0] > 0
    assert np.all(np.isfinite(segments.t_enter))
    assert np.all(np.isfinite(segments.t_exit))
    assert np.all(segments.t_exit > segments.t_enter)


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

    np.testing.assert_allclose(center_row, center_row[::-1], atol=5.0e-5, rtol=0.0)
    np.testing.assert_allclose(center_col, center_col[::-1], atol=5.0e-5, rtol=0.0)
