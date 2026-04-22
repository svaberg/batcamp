from __future__ import annotations

import logging
import math
import numpy as np
import pytest
from batread import Dataset

from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp import OctreeRayTracer
from batcamp import raytracer_cartesian
from batcamp import render_midpoint_image
from fake_dataset import build_cartesian_hex_mesh
from sample_data_helper import data_file


def _build_xyz_tree() -> Octree:
    """Return one small Cartesian tree with known box bounds."""
    points, corners = build_cartesian_hex_mesh(
        x_edges=np.array([-1.0, 0.0, 1.0], dtype=float),
        y_edges=np.array([-0.75, 0.0, 0.75], dtype=float),
        z_edges=np.array([-0.5, 0.0, 0.5], dtype=float),
    )
    return Octree(points, corners, tree_coord="xyz")


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
    return Octree(np.array(xyz_list, dtype=float), corners, tree_coord="xyz")


def _build_long_x_tree(n_x: int = 40) -> Octree:
    """Return one Cartesian strip with many x-cells for chunk-growth tests."""
    points, corners = build_cartesian_hex_mesh(
        x_edges=np.arange(int(n_x) + 1, dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        z_edges=np.array([0.0, 1.0], dtype=float),
    )
    return Octree(points, corners, tree_coord="xyz")


def _build_reference_tree() -> Octree:
    """Return one two-cell Cartesian tree for simple image references."""
    points, corners = build_cartesian_hex_mesh(
        x_edges=np.array([-1.0, 0.0, 1.0], dtype=float),
        y_edges=np.array([-1.0, 1.0], dtype=float),
        z_edges=np.array([-1.0, 1.0], dtype=float),
    )
    return Octree(points, corners, tree_coord="xyz")


def _build_unit_tree() -> Octree:
    """Return one single-cell Cartesian tree on the unit cube."""
    points, corners = build_cartesian_hex_mesh(
        x_edges=np.array([0.0, 1.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        z_edges=np.array([0.0, 1.0], dtype=float),
    )
    return Octree(points, corners, tree_coord="xyz")


def _direction_id(signs: tuple[int, int, int]) -> str:
    """Return one compact pytest id for one `{-1,0,1}^3` direction."""
    labels = []
    for sign, axis in zip(signs, ("x", "y", "z"), strict=True):
        if sign == 0:
            continue
        labels.append(("+" if sign > 0 else "-") + axis)
    return "".join(labels)


_CARDINAL_DIRECTION_PARAMS = [
    pytest.param(
        np.array(signs, dtype=float) / np.linalg.norm(np.array(signs, dtype=float)),
        id=_direction_id(signs),
    )
    for signs in (
        (ix, iy, iz)
        for ix in (-1, 0, 1)
        for iy in (-1, 0, 1)
        for iz in (-1, 0, 1)
        if (ix, iy, iz) != (0, 0, 0)
    )
]


def _orthonormal_plane_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one orthonormal basis for the plane perpendicular to `direction`."""
    axis = int(np.argmin(np.abs(direction)))
    seed = np.zeros(3, dtype=float)
    seed[axis] = 1.0
    plane_u = np.cross(direction, seed)
    plane_u /= np.linalg.norm(plane_u)
    plane_v = np.cross(direction, plane_u)
    return plane_u, plane_v


def _xyz_box_corners(tree: Octree) -> np.ndarray:
    """Return the eight Cartesian box corners for one `xyz` tree domain."""
    domain_lo = np.asarray(tree.packed_domain_bounds[:, 0], dtype=float)
    domain_hi = domain_lo + np.asarray(tree.packed_domain_bounds[:, 1], dtype=float)
    return np.array(
        [
            [x, y, z]
            for x in (float(domain_lo[0]), float(domain_hi[0]))
            for y in (float(domain_lo[1]), float(domain_hi[1]))
            for z in (float(domain_lo[2]), float(domain_hi[2]))
        ],
        dtype=float,
    )


def _pixel_centers(start: float, stop: float, resolution: int) -> np.ndarray:
    """Return one evenly spaced run of pixel-center coordinates."""
    edges = np.linspace(float(start), float(stop), int(resolution) + 1, dtype=float)
    return np.asarray((edges[:-1] + edges[1:]) / 2.0, dtype=float)


def _real_file_chord_rays(
    tree: Octree,
    direction: np.ndarray,
    *,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return one ray plane for one real file and one normalized direction."""
    plane_u, plane_v = _orthonormal_plane_basis(direction)
    if str(tree.tree_coord) == "xyz":
        corners = _xyz_box_corners(tree)
        along = np.asarray(corners @ direction, dtype=float)
        coords_u = np.asarray(corners @ plane_u, dtype=float)
        coords_v = np.asarray(corners @ plane_v, dtype=float)
        pad = 0.1 * float(np.max(along) - np.min(along))
        start_t = float(np.min(along) - pad)
        u_coords = _pixel_centers(float(np.min(coords_u)), float(np.max(coords_u)), int(resolution))
        v_coords = _pixel_centers(float(np.min(coords_v)), float(np.max(coords_v)), int(resolution))
    else:
        domain_rmax = float(np.sum(tree.packed_domain_bounds[0], dtype=float))
        start_t = -1.1 * domain_rmax
        u_coords = _pixel_centers(-domain_rmax, domain_rmax, int(resolution))
        v_coords = _pixel_centers(-domain_rmax, domain_rmax, int(resolution))

    plane_uu, plane_vv = np.meshgrid(u_coords, v_coords, indexing="xy")
    origins = (
        plane_uu[..., None] * plane_u[None, None, :]
        + plane_vv[..., None] * plane_v[None, None, :]
        + start_t * direction[None, None, :]
    )
    return origins, direction, plane_uu, plane_vv


def _expected_constant_density_path_lengths(
    tree: Octree,
    origins: np.ndarray,
    direction: np.ndarray,
    plane_u: np.ndarray,
    plane_v: np.ndarray,
) -> np.ndarray:
    """Return expected traced path lengths for one real-file ray plane."""
    if str(tree.tree_coord) == "xyz":
        domain_lo = np.asarray(tree.packed_domain_bounds[:, 0], dtype=float)
        domain_hi = domain_lo + np.asarray(tree.packed_domain_bounds[:, 1], dtype=float)
        out = np.zeros(origins.shape[:-1], dtype=float)
        flat_origins = origins.reshape(-1, 3)
        for ray_id, origin in enumerate(flat_origins):
            interval = _domain_interval_xyz(origin, direction, domain_lo, domain_hi)
            if interval is None:
                continue
            out.reshape(-1)[ray_id] = float(interval[1] - interval[0])
        return out

    domain_rmin = float(tree.packed_domain_bounds[0, 0])
    domain_rmax = float(np.sum(tree.packed_domain_bounds[0], dtype=float))
    ray_rmin_sq = plane_u * plane_u + plane_v * plane_v
    out = np.zeros_like(plane_u, dtype=float)
    outer_half = np.zeros_like(plane_u, dtype=float)
    outer = ray_rmin_sq < domain_rmax * domain_rmax
    outer_half[outer] = np.sqrt(domain_rmax * domain_rmax - ray_rmin_sq[outer])
    out[outer] = 2.0 * outer_half[outer]
    inner = ray_rmin_sq < domain_rmin * domain_rmin
    out[inner] = outer_half[inner] - np.sqrt(domain_rmin * domain_rmin - ray_rmin_sq[inner])
    return out


def _trace_path_lengths(segments) -> np.ndarray:
    """Return one image of positive traced path lengths."""
    out = np.zeros(segments.ray_shape, dtype=float)
    for ray_id in range(int(np.prod(segments.ray_shape, dtype=int))):
        cell_ids, times = _ray_slice(segments, ray_id)
        _positive_cell_ids, positive_times = _positive_trace(cell_ids, times)
        if positive_times.size:
            out.reshape(-1)[ray_id] = float(np.sum(np.diff(positive_times), dtype=float))
    return out


def _assert_path_lengths_match_expected(
    tree: Octree,
    traced: np.ndarray,
    expected: np.ndarray,
    plane_u: np.ndarray,
    plane_v: np.ndarray,
    direction: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> None:
    """Assert traced path lengths with one detailed first-failure report."""
    close = np.isclose(traced, expected, atol=atol, rtol=rtol)
    if np.all(close):
        return

    diff = np.abs(traced - expected)
    diff[close] = -1.0
    ij = np.unravel_index(int(np.argmax(diff)), diff.shape)
    plane_u_ij = float(plane_u[ij])
    plane_v_ij = float(plane_v[ij])
    impact = float(np.hypot(plane_u_ij, plane_v_ij))
    traced_ij = float(traced[ij])
    expected_ij = float(expected[ij])
    if str(tree.tree_coord) == "rpa":
        domain_rmin = float(tree.packed_domain_bounds[0, 0])
        domain_rmax = float(np.sum(tree.packed_domain_bounds[0], dtype=float))
        outer_chord = 0.0 if impact >= domain_rmax else 2.0 * math.sqrt(domain_rmax * domain_rmax - impact * impact)
        inner_chord = 0.0 if impact >= domain_rmin else 2.0 * math.sqrt(domain_rmin * domain_rmin - impact * impact)
        raise AssertionError(
            "traced path length mismatch "
            f"at ij={ij}: direction={tuple(float(value) for value in direction)}, "
            f"plane_u={plane_u_ij:g}, plane_v={plane_v_ij:g}, "
            f"ray_rmin={impact:g}, domain_rmin={domain_rmin:g}, domain_rmax={domain_rmax:g}, "
            f"outer_chord={outer_chord:g}, inner_chord={inner_chord:g}, "
            f"traced={traced_ij:g}, expected={expected_ij:g}"
        )

    raise AssertionError(
        "traced path length mismatch "
        f"at ij={ij}: direction={tuple(float(value) for value in direction)}, "
        f"plane_u={plane_u_ij:g}, plane_v={plane_v_ij:g}, traced={traced_ij:g}, expected={expected_ij:g}"
    )


def _ray_slice(segments, ray_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return one packed crossing-trace slice."""
    cell_lo = int(segments.ray_offsets[ray_id])
    cell_hi = int(segments.ray_offsets[ray_id + 1])
    time_lo = int(segments.time_offsets[ray_id])
    time_hi = int(segments.time_offsets[ray_id + 1])
    return segments.cell_ids[cell_lo:cell_hi], segments.times[time_lo:time_hi]


def _positive_trace(cell_ids: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop zero-length hops and merge adjacent positive intervals with the same owner."""
    positive_cell_ids: list[int] = []
    positive_times: list[float] = []
    for cell_id, t_start, t_stop in zip(cell_ids, times[:-1], times[1:]):
        if not t_stop > t_start:
            continue
        if positive_cell_ids and positive_cell_ids[-1] == int(cell_id) and positive_times[-1] == float(t_start):
            positive_times[-1] = float(t_stop)
            continue
        if not positive_times:
            positive_times.append(float(t_start))
        positive_cell_ids.append(int(cell_id))
        positive_times.append(float(t_stop))
    return np.array(positive_cell_ids, dtype=np.int64), np.array(positive_times, dtype=float)


_REAL_FILE_PARAMS = [
    pytest.param("3d__var_1_n00000000.plt", id="local_example"),
    pytest.param("3d__var_2_n00006003.plt", id="local_xyz"),
    pytest.param("3d__var_2_n00060005.plt", id="local_rpa"),
    pytest.param("3d__var_4_n00044000.plt", id="pooch_sc", marks=pytest.mark.pooch),
    pytest.param("3d__var_4_n00005000.plt", id="pooch_ih", marks=pytest.mark.pooch),
]


def _domain_interval_xyz(
    origin: np.ndarray,
    direction: np.ndarray,
    domain_lo: np.ndarray,
    domain_hi: np.ndarray,
) -> tuple[float, float] | None:
    """Return the analytical clipped domain interval for one Cartesian ray."""
    t_enter = -np.inf
    t_exit = np.inf
    for axis in range(3):
        direction_value = float(direction[axis])
        origin_value = float(origin[axis])
        lo_value = float(domain_lo[axis])
        hi_value = float(domain_hi[axis])
        if direction_value == 0.0:
            if origin_value < lo_value or origin_value > hi_value:
                return None
            continue
        t0 = (lo_value - origin_value) / direction_value
        t1 = (hi_value - origin_value) / direction_value
        if t0 > t1:
            t0, t1 = t1, t0
        if t0 > t_enter:
            t_enter = t0
        if t1 < t_exit:
            t_exit = t1
        if t_enter > t_exit:
            return None
    return float(t_enter), float(t_exit)


def _assert_positive_trace_forms_one_ray(
    tree: Octree,
    origin: np.ndarray,
    direction: np.ndarray,
    cell_ids: np.ndarray,
    times: np.ndarray,
    *,
    t_min: float = 0.0,
    t_max: float = np.inf,
    atol: float = 1.0e-12,
) -> None:
    """Check one public crossing trace against the clipped analytical ray interval."""
    positive_cell_ids, positive_times = _positive_trace(cell_ids, times)
    interval = _domain_interval_xyz(origin, direction, *tree.domain_bounds(coord="xyz"))
    if interval is None:
        assert positive_cell_ids.size == 0
        assert positive_times.size == 0
        return
    start_t = max(float(t_min), float(interval[0]))
    stop_t = min(float(t_max), float(interval[1]))
    if not start_t < stop_t:
        assert positive_cell_ids.size == 0
        assert positive_times.size == 0
        return

    assert positive_times.size == positive_cell_ids.size + 1
    assert np.all(positive_times[1:] > positive_times[:-1])
    np.testing.assert_allclose(positive_times[0], start_t, atol=atol, rtol=0.0)
    np.testing.assert_allclose(positive_times[-1], stop_t, atol=atol, rtol=0.0)
    np.testing.assert_allclose(np.sum(np.diff(positive_times), dtype=float), stop_t - start_t, atol=atol, rtol=0.0)

    mid_t = (positive_times[:-1] + positive_times[1:]) / 2.0
    mid_xyz = origin[None, :] + mid_t[:, None] * direction[None, :]
    owners = np.asarray(tree.lookup_points(mid_xyz, coord="xyz"), dtype=np.int64)
    np.testing.assert_array_equal(owners, positive_cell_ids)


def test_trace_returns_event_segments_for_one_cartesian_ray() -> None:
    tracer = OctreeRayTracer(_build_xyz_tree())
    segments = tracer.trace(np.array([-2.0, -0.3, -0.2], dtype=float), np.array([1.0, 0.0, 0.0], dtype=float))

    assert segments.ray_shape == (1,)
    np.testing.assert_array_equal(segments.ray_offsets, np.array([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(segments.time_offsets, np.array([0, 3], dtype=np.int64))
    np.testing.assert_array_equal(segments.cell_ids, np.array([0, 4], dtype=np.int64))
    np.testing.assert_allclose(segments.times, np.array([1.0, 2.0, 3.0], dtype=float), atol=0.0, rtol=0.0)


def test_raysegments_1d_indexing_unpacks_one_packed_slice() -> None:
    tracer = OctreeRayTracer(_build_xyz_tree())
    segments = tracer.trace(
        np.array(
            [
                [-2.0, -0.3, -0.2],
                [2.0, 2.0, 2.0],
                [-2.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        np.array([1.0, 0.0, 0.0], dtype=float),
    )

    first_cell_ids, first_times = segments[0]
    np.testing.assert_array_equal(first_cell_ids, np.array([0, 4], dtype=np.int64))
    np.testing.assert_allclose(first_times, np.array([1.0, 2.0, 3.0], dtype=float), atol=0.0, rtol=0.0)

    empty_cell_ids, empty_times = segments[1]
    np.testing.assert_array_equal(empty_cell_ids, np.empty(0, dtype=np.int64))
    np.testing.assert_array_equal(empty_times, np.empty(0, dtype=float))

    last_cell_ids, last_times = segments[-1]
    expected_last_cell_ids, expected_last_times = _ray_slice(segments, 2)
    np.testing.assert_array_equal(last_cell_ids, expected_last_cell_ids)
    np.testing.assert_array_equal(last_times, expected_last_times)

    with pytest.raises(IndexError):
        segments[segments.n_rays]


def test_raysegments_grid_indexing_matches_batched_ray_layout() -> None:
    tracer = OctreeRayTracer(_build_xyz_tree())
    origins = np.array(
        [
            [[-2.0, -0.3, -0.2], [-2.0, 0.3, 0.2]],
            [[2.0, 2.0, 2.0], [-2.0, 0.0, 0.0]],
        ],
        dtype=float,
    )
    segments = tracer.trace(origins, np.array([1.0, 0.0, 0.0], dtype=float))

    ij_cell_ids, ij_times = segments[0, 1]
    expected_ij_cell_ids, expected_ij_times = _ray_slice(segments, 1)
    np.testing.assert_array_equal(ij_cell_ids, expected_ij_cell_ids)
    np.testing.assert_array_equal(ij_times, expected_ij_times)

    miss_cell_ids, miss_times = segments[1, 0]
    np.testing.assert_array_equal(miss_cell_ids, np.empty(0, dtype=np.int64))
    np.testing.assert_array_equal(miss_times, np.empty(0, dtype=float))

    neg_cell_ids, neg_times = segments[-1, -1]
    expected_neg_cell_ids, expected_neg_times = _ray_slice(segments, 3)
    np.testing.assert_array_equal(neg_cell_ids, expected_neg_cell_ids)
    np.testing.assert_array_equal(neg_times, expected_neg_times)

    row_subset = segments[0]
    assert row_subset.shape == (2,)
    np.testing.assert_array_equal(row_subset.origins, origins[0])

    with pytest.raises(IndexError):
        segments[2, 0]


def test_raysegments_slice_indexing_returns_repacked_subset() -> None:
    tracer = OctreeRayTracer(_build_xyz_tree())
    origins = np.array(
        [
            [[-2.0, -0.3, -0.2], [-2.0, 0.3, 0.2]],
            [[2.0, 2.0, 2.0], [-2.0, 0.0, 0.0]],
        ],
        dtype=float,
    )
    directions = np.array([1.0, 0.0, 0.0], dtype=float)
    segments = tracer.trace(origins, directions)

    subset = segments[0, ::-1]

    assert isinstance(subset, type(segments))
    assert subset.shape == (2,)
    np.testing.assert_array_equal(subset.origins, origins[0, ::-1])
    np.testing.assert_array_equal(subset.directions, directions)

    first_cell_ids, first_times = subset[0]
    expected_first_cell_ids, expected_first_times = _ray_slice(segments, 1)
    np.testing.assert_array_equal(first_cell_ids, expected_first_cell_ids)
    np.testing.assert_array_equal(first_times, expected_first_times)
    np.testing.assert_allclose(
        subset.xyz(0),
        origins[0, 1] + expected_first_times[:, None] * directions,
        atol=0.0,
        rtol=0.0,
    )

    second_cell_ids, second_times = subset[1]
    expected_second_cell_ids, expected_second_times = _ray_slice(segments, 0)
    np.testing.assert_array_equal(second_cell_ids, expected_second_cell_ids)
    np.testing.assert_array_equal(second_times, expected_second_times)


def test_raysegments_mask_indexing_returns_row_major_subset() -> None:
    tracer = OctreeRayTracer(_build_xyz_tree())
    origins = np.array(
        [
            [[-2.0, -0.3, -0.2], [-2.0, 0.3, 0.2]],
            [[2.0, 2.0, 2.0], [-2.0, 0.0, 0.0]],
        ],
        dtype=float,
    )
    directions = np.array([1.0, 0.0, 0.0], dtype=float)
    segments = tracer.trace(origins, directions)

    subset = segments[np.array([[False, True], [False, True]])]

    assert subset.shape == (2,)
    np.testing.assert_array_equal(subset.origins, np.array([origins[0, 1], origins[1, 1]], dtype=float))
    np.testing.assert_array_equal(subset.directions, directions)
    np.testing.assert_array_equal(subset[0][0], _ray_slice(segments, 1)[0])
    np.testing.assert_array_equal(subset[1][0], _ray_slice(segments, 3)[0])


def test_raysegments_fancy_indexing_returns_selected_subset() -> None:
    tracer = OctreeRayTracer(_build_xyz_tree())
    origins = np.array(
        [
            [[-2.0, -0.3, -0.2], [-2.0, 0.3, 0.2]],
            [[2.0, 2.0, 2.0], [-2.0, 0.0, 0.0]],
        ],
        dtype=float,
    )
    directions = np.zeros_like(origins)
    directions[..., 0] = 1.0
    segments = tracer.trace(origins, directions)

    subset = segments[np.array([0, 1]), np.array([1, 1])]

    assert subset.shape == (2,)
    np.testing.assert_array_equal(subset.origins, np.array([origins[0, 1], origins[1, 1]], dtype=float))
    np.testing.assert_array_equal(subset.directions, np.array([directions[0, 1], directions[1, 1]], dtype=float))
    np.testing.assert_allclose(
        subset.xyz(1),
        origins[1, 1] + _ray_slice(segments, 3)[1][:, None] * directions[1, 1],
        atol=0.0,
        rtol=0.0,
    )


def test_raysegments_xyz_uses_stored_shared_direction() -> None:
    tracer = OctreeRayTracer(_build_xyz_tree())
    origins = np.array(
        [
            [[-2.0, -0.3, -0.2], [-2.0, 0.3, 0.2]],
            [[2.0, 2.0, 2.0], [-2.0, 0.0, 0.0]],
        ],
        dtype=float,
    )
    directions = np.array([1.0, 0.0, 0.0], dtype=float)
    segments = tracer.trace(origins, directions)

    _, ray_times = segments[0, 1]
    np.testing.assert_allclose(
        segments.xyz(0, 1),
        origins[0, 1] + ray_times[:, None] * directions,
        atol=0.0,
        rtol=0.0,
    )


def test_raysegments_xyz_uses_stored_per_ray_direction() -> None:
    tracer = OctreeRayTracer(_build_xyz_tree())
    origins = np.array(
        [
            [[-2.0, -0.3, -0.2], [-2.0, 0.3, 0.2]],
            [[2.0, 2.0, 2.0], [-2.0, 0.0, 0.0]],
        ],
        dtype=float,
    )
    directions = np.zeros_like(origins)
    directions[..., 0] = 1.0
    segments = tracer.trace(origins, directions)

    _, ray_times = segments[-1, -1]
    np.testing.assert_allclose(
        segments.xyz(-1, -1),
        origins[-1, -1] + ray_times[:, None] * directions[-1, -1],
        atol=0.0,
        rtol=0.0,
    )


def test_trace_clips_one_cartesian_ray() -> None:
    tracer = OctreeRayTracer(_build_xyz_tree())
    segments = tracer.trace(
        np.array([-2.0, -0.3, -0.2], dtype=float),
        np.array([1.0, 0.0, 0.0], dtype=float),
        t_min=1.5,
        t_max=2.5,
    )

    np.testing.assert_array_equal(segments.cell_ids, np.array([0, 4], dtype=np.int64))
    np.testing.assert_allclose(segments.times, np.array([1.5, 2.0, 2.5], dtype=float), atol=0.0, rtol=0.0)


def test_trace_handles_one_miss() -> None:
    tracer = OctreeRayTracer(_build_xyz_tree())
    segments = tracer.trace(np.array([2.0, 2.0, 2.0], dtype=float), np.array([1.0, 0.0, 0.0], dtype=float))

    np.testing.assert_array_equal(segments.ray_offsets, np.array([0, 0], dtype=np.int64))
    np.testing.assert_array_equal(segments.time_offsets, np.array([0, 0], dtype=np.int64))
    assert segments.cell_ids.size == 0
    assert segments.times.size == 0


def test_trace_preserves_batch_shape() -> None:
    tracer = OctreeRayTracer(_build_xyz_tree())
    origins = np.array(
        [
            [[-2.0, -0.3, -0.2], [-2.0, 0.3, 0.2]],
            [[2.0, 2.0, 2.0], [-2.0, 0.0, 0.0]],
        ],
        dtype=float,
    )
    directions = np.zeros_like(origins)
    directions[..., 0] = 1.0
    segments = tracer.trace(origins, directions)

    assert segments.ray_shape == (2, 2)
    assert segments.shape == (2, 2)
    assert segments.ndim == 2
    assert segments.size == 4
    np.testing.assert_array_equal(segments.origins, origins)
    np.testing.assert_array_equal(segments.directions, directions)
    np.testing.assert_array_equal(segments.ray_offsets, np.array([0, 2, 4, 4, 6], dtype=np.int64))
    np.testing.assert_array_equal(segments.time_offsets, np.array([0, 3, 6, 6, 9], dtype=np.int64))
    np.testing.assert_array_equal(_ray_slice(segments, 2)[0], np.empty(0, dtype=np.int64))
    np.testing.assert_array_equal(_ray_slice(segments, 2)[1], np.empty(0, dtype=float))


def test_trace_broadcasts_one_direction_vector_over_batched_origins() -> None:
    tracer = OctreeRayTracer(_build_xyz_tree())
    origins = np.array(
        [
            [[-2.0, -0.3, -0.2], [-2.0, 0.3, 0.2]],
            [[2.0, 2.0, 2.0], [-2.0, 0.0, 0.0]],
        ],
        dtype=float,
    )
    directions = np.array([1.0, 0.0, 0.0], dtype=float)

    segments = tracer.trace(origins, directions)

    assert segments.ray_shape == (2, 2)
    np.testing.assert_array_equal(segments.origins, origins)
    np.testing.assert_array_equal(segments.directions, directions)
    np.testing.assert_array_equal(segments.ray_offsets, np.array([0, 2, 4, 4, 6], dtype=np.int64))
    np.testing.assert_array_equal(segments.time_offsets, np.array([0, 3, 6, 6, 9], dtype=np.int64))
    np.testing.assert_array_equal(_ray_slice(segments, 2)[0], np.empty(0, dtype=np.int64))
    np.testing.assert_array_equal(_ray_slice(segments, 2)[1], np.empty(0, dtype=float))


def test_trace_grows_chunk_event_capacity_when_the_initial_capacity_is_too_small(monkeypatch, caplog) -> None:
    tree = _build_long_x_tree()
    monkeypatch.setattr(raytracer_cartesian, "DEFAULT_CROSSING_BUFFER_SIZE", 32)
    tracer = OctreeRayTracer(tree)

    origin = np.array([-1.0, 0.5, 0.5], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    with caplog.at_level(logging.DEBUG, logger="batcamp.raytracer"):
        segments = tracer.trace(origin, direction)

    positive_cell_ids, positive_times = _positive_trace(*_ray_slice(segments, 0))
    assert positive_cell_ids.size == 40
    np.testing.assert_allclose(positive_times[0], 1.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(positive_times[-1], 41.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.diff(positive_times), np.ones(40, dtype=float), atol=0.0, rtol=0.0)
    _assert_positive_trace_forms_one_ray(tree, origin, direction, *_ray_slice(segments, 0))
    assert tracer._crossing_buffer_size == 64
    assert any("grow crossing buffer" in record.getMessage() for record in caplog.records)
    assert any("_trace_segments" in record.getMessage() for record in caplog.records)

    tracer.trace(origin, direction, t_max=5.5)
    assert tracer._crossing_buffer_size == 64


def test_trace_positive_corner_crossing_matches_expected_path() -> None:
    tree = _build_xyz_tree()
    tracer = OctreeRayTracer(tree)
    origin = np.array([-0.5, -0.375, -0.25], dtype=float)
    direction = np.array([1.0, 0.75, 0.5], dtype=float)
    segments = tracer.trace(origin, direction)

    positive_cell_ids, positive_times = _positive_trace(*_ray_slice(segments, 0))
    np.testing.assert_array_equal(positive_cell_ids, np.array([0, 7], dtype=np.int64))
    np.testing.assert_allclose(positive_times, np.array([0.0, 0.5, 1.5], dtype=float), atol=1.0e-12, rtol=0.0)
    _assert_positive_trace_forms_one_ray(tree, origin, direction, *_ray_slice(segments, 0))


def test_trace_walks_coarse_to_fine_cells_exactly() -> None:
    tree = _build_xyz_coarse_fine_tree()
    tracer = OctreeRayTracer(tree)
    origin = np.array([-0.1, 0.125, 0.125], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    segments = tracer.trace(origin, direction)

    positive_cell_ids, positive_times = _positive_trace(*_ray_slice(segments, 0))
    np.testing.assert_array_equal(positive_cell_ids, np.array([0, 7, 11], dtype=np.int64))
    np.testing.assert_allclose(positive_times, np.array([0.1, 0.6, 0.85, 1.1], dtype=float), atol=1.0e-12, rtol=0.0)
    _assert_positive_trace_forms_one_ray(tree, origin, direction, *_ray_slice(segments, 0))


def test_trace_walks_fine_to_coarse_cells_exactly() -> None:
    tree = _build_xyz_coarse_fine_tree()
    tracer = OctreeRayTracer(tree)
    origin = np.array([1.1, 0.125, 0.125], dtype=float)
    direction = np.array([-1.0, 0.0, 0.0], dtype=float)
    segments = tracer.trace(origin, direction)

    positive_cell_ids, positive_times = _positive_trace(*_ray_slice(segments, 0))
    np.testing.assert_array_equal(positive_cell_ids, np.array([11, 7, 0], dtype=np.int64))
    np.testing.assert_allclose(positive_times, np.array([0.1, 0.35, 0.6, 1.1], dtype=float), atol=1.0e-12, rtol=0.0)
    _assert_positive_trace_forms_one_ray(tree, origin, direction, *_ray_slice(segments, 0))


def test_trace_refined_edge_crossing_keeps_positive_topology() -> None:
    tree = _build_xyz_coarse_fine_tree()
    tracer = OctreeRayTracer(tree)
    origin = np.array([-0.1, 0.25, 0.25], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    segments = tracer.trace(origin, direction)

    positive_cell_ids, positive_times = _positive_trace(*_ray_slice(segments, 0))
    np.testing.assert_array_equal(positive_cell_ids, np.array([0, 7, 11], dtype=np.int64))
    np.testing.assert_allclose(positive_times, np.array([0.1, 0.6, 0.85, 1.1], dtype=float), atol=1.0e-12, rtol=0.0)
    _assert_positive_trace_forms_one_ray(tree, origin, direction, *_ray_slice(segments, 0))


def test_render_midpoint_image_uses_segment_cell_ids(monkeypatch) -> None:
    tree = _build_xyz_tree()
    tracer = OctreeRayTracer(tree)
    interp = OctreeInterpolator(tree, np.ones(int(np.max(tree.corners)) + 1, dtype=float))
    origins = np.array([[[-2.0, -0.3, -0.2]]], dtype=float)
    directions = np.array([[[1.0, 0.0, 0.0]]], dtype=float)
    segments = tracer.trace(origins, directions)

    def _fail_lookup_points(*_args, **_kwargs):
        raise AssertionError("render_midpoint_image should not call lookup_points when segment cell ids are known.")

    monkeypatch.setattr(interp.tree, "lookup_points", _fail_lookup_points)

    image = render_midpoint_image(interp, origins, directions, segments)

    np.testing.assert_allclose(image, np.array([[2.0]], dtype=float), atol=1.0e-12, rtol=0.0)


def test_render_midpoint_image_preserves_vector_components() -> None:
    tree = _build_xyz_tree()
    tracer = OctreeRayTracer(tree)
    n_points = int(np.max(tree.corners)) + 1
    interp = OctreeInterpolator(
        tree,
        np.column_stack((np.ones(n_points, dtype=float), np.full(n_points, 3.0, dtype=float))),
    )
    origins = np.array([[[-2.0, -0.3, -0.2]]], dtype=float)
    directions = np.array([[[1.0, 0.0, 0.0]]], dtype=float)
    segments = tracer.trace(origins, directions)

    image = render_midpoint_image(interp, origins, directions, segments)

    assert image.shape == (1, 1, 2)
    np.testing.assert_allclose(image, np.array([[[2.0, 6.0]]], dtype=float), atol=1.0e-12, rtol=0.0)


def test_trace_skips_failed_rays_with_empty_segments(monkeypatch) -> None:
    tree = _build_xyz_tree()
    tracer = OctreeRayTracer(tree)

    def fake_trace_rays(
        root_cell_ids: np.ndarray,
        cell_child: np.ndarray,
        cell_bounds: np.ndarray,
        domain_bounds: np.ndarray,
        cell_neighbor: np.ndarray,
        cell_depth: np.ndarray,
        origins: np.ndarray,
        directions: np.ndarray,
        t_min: float,
        t_max: float,
        cell_counts: np.ndarray,
        time_counts: np.ndarray,
        cell_ids_out: np.ndarray,
        times_out: np.ndarray,
    ) -> None:
        del root_cell_ids, cell_child, cell_bounds, domain_bounds, cell_neighbor, cell_depth
        del origins, directions, t_min, t_max
        cell_counts[:] = 0
        time_counts[:] = 0
        cell_counts[0] = 2
        time_counts[0] = 3
        cell_ids_out[0, 0] = 0
        cell_ids_out[0, 1] = 1
        times_out[0, 0] = 1.0
        times_out[0, 1] = 2.0
        times_out[0, 2] = 3.0
        cell_counts[1] = -2
        time_counts[1] = -2

    monkeypatch.setattr(tracer._raytracer_module, "trace_rays", fake_trace_rays)

    origins = np.array([[-2.0, -0.3, -0.2], [-2.0, -0.3, -0.2]], dtype=float)
    directions = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)

    segments = tracer.trace(origins, directions)

    np.testing.assert_array_equal(segments.ray_offsets, np.array([0, 2, 2], dtype=np.int64))
    np.testing.assert_array_equal(segments.time_offsets, np.array([0, 3, 3], dtype=np.int64))
    np.testing.assert_array_equal(segments.cell_ids, np.array([0, 1], dtype=np.int64))
    np.testing.assert_allclose(segments.times, np.array([1.0, 2.0, 3.0], dtype=float), atol=1.0e-12, rtol=0.0)


def test_midpoint_image_preserves_vector_components() -> None:
    tree = _build_xyz_tree()
    tracer = OctreeRayTracer(tree)
    n_points = int(np.max(tree.corners)) + 1
    interp = OctreeInterpolator(
        tree,
        np.column_stack((np.ones(n_points, dtype=float), np.full(n_points, 3.0, dtype=float))),
    )
    origins = np.array([[[-2.0, -0.3, -0.2]]], dtype=float)
    directions = np.array([[[1.0, 0.0, 0.0]]], dtype=float)

    image, counts = tracer.midpoint_image(interp, origins, directions)

    assert image.shape == (1, 1, 2)
    np.testing.assert_array_equal(counts, np.array([[2]], dtype=np.int64))
    np.testing.assert_allclose(image, np.array([[[2.0, 6.0]]], dtype=float), atol=1.0e-12, rtol=0.0)


def test_trilinear_image_integrates_bilinear_slanted_ray() -> None:
    tree = _build_unit_tree()
    tracer = OctreeRayTracer(tree)
    points = np.asarray(tree._points, dtype=float)
    interp = OctreeInterpolator(tree, points[:, 0] * points[:, 1])

    origins = np.array([[[-0.25, 0.125, 0.5]]], dtype=float)
    directions = np.array([[[1.0, 0.5, 0.0]]], dtype=float)

    image_trilinear, counts_trilinear = tracer.trilinear_image(interp, origins, directions)

    np.testing.assert_array_equal(counts_trilinear, np.array([[1]], dtype=np.int64))
    np.testing.assert_allclose(image_trilinear, np.array([[7.0 / 24.0]], dtype=float), atol=1.0e-12, rtol=0.0)


def test_trilinear_image_preserves_vector_components() -> None:
    tree = _build_xyz_tree()
    tracer = OctreeRayTracer(tree)
    n_points = int(np.max(tree.corners)) + 1
    interp = OctreeInterpolator(
        tree,
        np.column_stack((np.ones(n_points, dtype=float), np.full(n_points, 3.0, dtype=float))),
    )
    origins = np.array([[[-2.0, -0.3, -0.2]]], dtype=float)
    directions = np.array([[[1.0, 0.0, 0.0]]], dtype=float)

    image, counts = tracer.trilinear_image(interp, origins, directions)

    assert image.shape == (1, 1, 2)
    np.testing.assert_array_equal(counts, np.array([[2]], dtype=np.int64))
    np.testing.assert_allclose(image, np.array([[[2.0, 6.0]]], dtype=float), atol=1.0e-12, rtol=0.0)


@pytest.mark.parametrize("direction", _CARDINAL_DIRECTION_PARAMS)
@pytest.mark.parametrize("file_name", _REAL_FILE_PARAMS)
def test_trace_path_lengths_match_constant_density_path_length_on_real_files(
    file_name: str,
    direction: np.ndarray,
) -> None:
    ds = Dataset.from_file(str(data_file(file_name)))
    tree = Octree.from_ds(ds)
    tracer = OctreeRayTracer(tree)
    origins, directions, plane_u, plane_v = _real_file_chord_rays(tree, direction, resolution=65)

    segments = tracer.trace(origins, directions)
    traced_lengths = _trace_path_lengths(segments)
    expected = _expected_constant_density_path_lengths(tree, origins, directions, plane_u, plane_v)

    _assert_path_lengths_match_expected(
        tree,
        traced_lengths,
        expected,
        plane_u,
        plane_v,
        directions,
        atol=1.0e-9,
        rtol=1.0e-9,
    )
