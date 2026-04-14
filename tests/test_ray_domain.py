from __future__ import annotations

import numpy as np

import batcamp.ray as ray_module
from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp import OctreeRayTracer
from batcamp import render_midpoint_image
from fake_dataset import build_cartesian_hex_mesh


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


def _ray_slice(segments, ray_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return one packed event trace slice."""
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
    """Check one public event trace against the clipped analytical ray interval."""
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
    np.testing.assert_array_equal(segments.ray_offsets, np.array([0, 2, 4, 4, 6], dtype=np.int64))
    np.testing.assert_array_equal(segments.time_offsets, np.array([0, 3, 6, 6, 9], dtype=np.int64))
    np.testing.assert_array_equal(_ray_slice(segments, 2)[0], np.empty(0, dtype=np.int64))
    np.testing.assert_array_equal(_ray_slice(segments, 2)[1], np.empty(0, dtype=float))


def test_trace_grows_chunk_event_capacity_when_the_initial_capacity_is_too_small(monkeypatch) -> None:
    tree = _build_long_x_tree()
    tracer = OctreeRayTracer(tree)
    monkeypatch.setattr(ray_module, "_TRACE_RAY_INITIAL_EVENTS", 32)

    origin = np.array([-1.0, 0.5, 0.5], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    segments = tracer.trace(origin, direction)

    positive_cell_ids, positive_times = _positive_trace(*_ray_slice(segments, 0))
    assert positive_cell_ids.size == 40
    np.testing.assert_allclose(positive_times[0], 1.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(positive_times[-1], 41.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.diff(positive_times), np.ones(40, dtype=float), atol=0.0, rtol=0.0)
    _assert_positive_trace_forms_one_ray(tree, origin, direction, *_ray_slice(segments, 0))


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


def test_render_midpoint_image_integrates_constant_density_along_one_ray() -> None:
    tree = _build_xyz_tree()
    tracer = OctreeRayTracer(tree)
    interp = OctreeInterpolator(tree, np.ones(int(np.max(tree.corners)) + 1, dtype=float))
    origins = np.array([[[-2.0, -0.3, -0.2]]], dtype=float)
    directions = np.array([[[1.0, 0.0, 0.0]]], dtype=float)

    image = render_midpoint_image(interp, origins, directions, tracer.trace(origins, directions))

    assert image.shape == (1, 1)
    np.testing.assert_allclose(image, np.array([[2.0]], dtype=float), atol=1.0e-12, rtol=0.0)


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


def test_render_midpoint_image_matches_cartesian_linear_reference_grid() -> None:
    tree = _build_reference_tree()
    tracer = OctreeRayTracer(tree)
    interp = OctreeInterpolator(tree, np.asarray(tree._points, dtype=float)[:, 0] + 2.0 * np.asarray(tree._points, dtype=float)[:, 1])

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
