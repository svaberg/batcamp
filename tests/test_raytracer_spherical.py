from __future__ import annotations

import itertools
from functools import lru_cache
import math

import numpy as np
import pytest

from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp import OctreeRayTracer
from batcamp import render_midpoint_image
from batcamp import raytracer_spherical
from fake_dataset import build_spherical_hex_mesh

_PI = math.pi
_TWO_PI = 2.0 * math.pi
_TIME_ATOL = 1.0e-12
_TIME_RTOL = 64.0 * np.finfo(np.float64).eps

_RAY_DIRECTIONS = (
    (1.0, 0.0, 0.0),
    (1.0, 0.08, 0.0),
    (1.0, 0.0, 0.06),
    (1.0, 0.05, -0.04),
)
_INTERIOR_ORIGINS = (
    (1.5, 0.95, 0.55),
    (1.5, 1.35, 1.35),
    (2.5, 1.85, 1.95),
    (2.5, 2.25, 2.45),
)
_SEAM_ORIGINS = tuple(
    (radius, polar, azimuth)
    for radius, polar in (
        (1.5, 0.95),
        (1.5, 1.35),
        (2.5, 1.85),
        (2.5, 2.25),
    )
    for azimuth in (
        0.0,
        1.0e-8,
        1.0e-6,
        _TWO_PI - 1.0e-8,
        _TWO_PI - 1.0e-6,
    )
)
_SEAM_DIRECTIONS = (
    (-0.2, -1.0, 0.0),
    (0.1, -1.0, 0.03),
    (0.2, 1.0, -0.02),
    (-0.3, 1.0, 0.01),
)
_SEAM_STRESS_ORIGINS = tuple(
    (radius, polar, azimuth)
    for radius in (1.25, 1.5, 1.75, 2.25, 2.5, 2.75)
    for polar in (0.4, 0.75, 0.95, 1.35, 1.85, 2.25, 2.6)
    for azimuth in (
        0.0,
        1.0e-12,
        1.0e-10,
        1.0e-8,
        1.0e-6,
        1.0e-4,
        _TWO_PI - 1.0e-12,
        _TWO_PI - 1.0e-10,
        _TWO_PI - 1.0e-8,
        _TWO_PI - 1.0e-6,
        _TWO_PI - 1.0e-4,
    )
)
_SEAM_STRESS_DIRECTIONS = (
    (-1.0, -4.0, 0.0),
    (-0.3, -1.0, 0.01),
    (0.1, -1.0, 0.03),
    (0.2, 1.0, -0.02),
    (0.6, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 0.2),
    (1.0, 0.0, -0.2),
)
_FIRST_EVENT_SINGLE_FACE_CASES = (
    ((2.5, 0.39269908169872414, 0.39269908169872414), (2.0, 0.39269908169872414, 0.39269908169872414), (0,)),
    ((1.5, 0.39269908169872414, 0.39269908169872414), (2.0, 0.39269908169872414, 0.39269908169872414), (1,)),
    ((1.5, 1.1780972450961724, 0.39269908169872414), (1.5, 0.7853981633974483, 0.39269908169872414), (2,)),
    ((1.5, 0.39269908169872414, 0.39269908169872414), (1.5, 0.7853981633974483, 0.39269908169872414), (3,)),
    ((1.5, 0.39269908169872414, 0.39269908169872414), (1.5, 0.39269908169872414, 0.0), (4,)),
    ((1.5, 0.39269908169872414, 0.39269908169872414), (1.5, 0.39269908169872414, 0.7853981633974483), (5,)),
)
_FIRST_EVENT_MULTIFACE_CASES = (
    ((2.5, 1.1780972450961724, 0.39269908169872414), (2.0, 0.7853981633974483, 0.39269908169872414), (0, 2)),
    ((2.5, 1.1780972450961724, 0.39269908169872414), (2.0, 0.7853981633974483, 0.0), (0, 2, 4)),
    ((2.5, 1.1780972450961724, 0.39269908169872414), (2.0, 0.7853981633974483, 0.7853981633974483), (0, 2, 5)),
    ((2.5, 0.39269908169872414, 0.39269908169872414), (2.0, 0.7853981633974483, 0.39269908169872414), (0, 3)),
    ((2.5, 0.39269908169872414, 0.39269908169872414), (2.0, 0.7853981633974483, 0.0), (0, 3, 4)),
    ((2.5, 0.39269908169872414, 0.39269908169872414), (2.0, 0.7853981633974483, 0.7853981633974483), (0, 3, 5)),
    ((2.5, 0.39269908169872414, 0.39269908169872414), (2.0, 0.39269908169872414, 0.0), (0, 4)),
    ((2.5, 0.39269908169872414, 0.39269908169872414), (2.0, 0.39269908169872414, 0.7853981633974483), (0, 5)),
    ((1.5, 1.1780972450961724, 0.39269908169872414), (2.0, 0.7853981633974483, 0.39269908169872414), (1, 2)),
    ((1.5, 1.1780972450961724, 0.39269908169872414), (2.0, 0.7853981633974483, 0.0), (1, 2, 4)),
    ((1.5, 1.1780972450961724, 0.39269908169872414), (2.0, 0.7853981633974483, 0.7853981633974483), (1, 2, 5)),
    ((1.5, 0.39269908169872414, 0.39269908169872414), (2.0, 0.7853981633974483, 0.39269908169872414), (1, 3)),
    ((1.5, 0.39269908169872414, 0.39269908169872414), (2.0, 0.7853981633974483, 0.0), (1, 3, 4)),
    ((1.5, 0.39269908169872414, 0.39269908169872414), (2.0, 0.7853981633974483, 0.7853981633974483), (1, 3, 5)),
    ((1.5, 0.39269908169872414, 0.39269908169872414), (2.0, 0.39269908169872414, 0.0), (1, 4)),
    ((1.5, 0.39269908169872414, 0.39269908169872414), (2.0, 0.39269908169872414, 0.7853981633974483), (1, 5)),
    ((1.5, 1.1780972450961724, 0.39269908169872414), (1.5, 0.7853981633974483, 0.0), (2, 4)),
    ((1.5, 1.1780972450961724, 0.39269908169872414), (1.5, 0.7853981633974483, 0.7853981633974483), (2, 5)),
    ((1.5, 0.39269908169872414, 0.39269908169872414), (1.5, 0.7853981633974483, 0.0), (3, 4)),
    ((1.5, 0.39269908169872414, 0.39269908169872414), (1.5, 0.7853981633974483, 0.7853981633974483), (3, 5)),
)


def _build_uniform_rpa_tree() -> Octree:
    """Return one small uniform spherical shell tree."""
    points, corners = build_spherical_hex_mesh(
        nr=2,
        npolar=4,
        nazimuth=8,
        r_min=1.0,
        r_max=3.0,
    )
    return Octree(points, corners, tree_coord="rpa")


def _normalize_direction(direction_xyz: tuple[float, float, float]) -> np.ndarray:
    """Return one unit-length Cartesian direction vector."""
    direction = np.asarray(direction_xyz, dtype=float)
    return direction / np.linalg.norm(direction)


def _cell_exit_event_rpa(
    cell_bounds: np.ndarray,
    cell_id: int,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    t_current: float,
) -> tuple[float, tuple[int, ...], bool]:
    """Test wrapper around the production spherical exit finder."""
    active_faces = np.empty(3, dtype=np.int64)
    candidate_times = np.empty(6, dtype=np.float64)
    roots = np.empty(2, dtype=np.float64)
    candidate_xyz = np.empty(3, dtype=np.float64)
    t_exit, n_active_face, axis_transfer = raytracer_spherical.find_exit(
        cell_bounds,
        cell_id,
        origin_xyz,
        direction_xyz,
        t_current,
        active_faces,
        candidate_times,
        roots,
        candidate_xyz,
    )
    return float(t_exit), tuple(int(active_faces[i]) for i in range(n_active_face)), bool(axis_transfer)


def _fill_active_face_state_rpa(active_faces: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Test wrapper around the production active-face state helper."""
    active_faces_array = np.asarray(active_faces, dtype=np.int64)
    active_face_by_axis = np.full(3, -1, dtype=np.int64)
    active_face_order = np.full(6, -1, dtype=np.int64)
    current_face_id = int(active_faces_array[0]) if active_faces_array.size else -1
    raytracer_spherical._fill_active_face_state(
        active_faces_array,
        int(active_faces_array.size),
        current_face_id,
        active_face_by_axis,
        active_face_order,
    )
    return active_face_by_axis, active_face_order


def _event_subface_id_rpa(
    cell_neighbor: np.ndarray,
    domain_bounds: np.ndarray,
    cell_bounds: np.ndarray,
    cell_id: int,
    face_id: int,
    active_face_by_axis: np.ndarray,
    active_face_order: np.ndarray,
    current_face_order: int,
    scratch_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    scratch_rpa: np.ndarray,
) -> int:
    """Test wrapper around the production subface selector."""
    return int(
        raytracer_spherical.find_subface(
            cell_neighbor,
            domain_bounds,
            cell_bounds,
            cell_id,
            face_id,
            active_face_by_axis,
            active_face_order,
            current_face_order,
            scratch_xyz,
            direction_xyz,
            scratch_rpa,
        )
    )


def _find_cell_rpa_single(tree: Octree, query_xyz: np.ndarray) -> int:
    """Test wrapper around the production spherical cell lookup."""
    query_xyz = np.asarray(query_xyz, dtype=float)
    query_rpa = np.asarray(
        raytracer_spherical.xyz_to_rpa_components(
            float(query_xyz[0]),
            float(query_xyz[1]),
            float(query_xyz[2]),
        ),
        dtype=float,
    )
    return int(
        raytracer_spherical.find_cell(
            query_rpa,
            tree.cell_child,
            tree._root_cell_ids,
            tree.cell_bounds,
            tree._domain_bounds,
        )
    )


def trace_rpa_test_path(
    tree: Octree,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    *,
    t_min: float = 0.0,
    t_max: float = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    """Test wrapper around the production batched spherical trace for one ray."""
    tracer = OctreeRayTracer(tree)
    segments = tracer.trace(
        origin_xyz,
        direction_xyz,
        t_min=t_min,
        t_max=t_max,
    )
    if segments.n_rays != 1:
        raise AssertionError("trace_rpa_test_path expects exactly one traced ray.")
    return segments.cell_ids.copy(), segments.times.copy()


def trace_rpa_test_rays(
    tree: Octree,
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    t_min: float = 0.0,
    t_max: float = np.inf,
):
    """Test wrapper around the production batched spherical trace."""
    tracer = OctreeRayTracer(tree)
    return tracer.trace(
        origins,
        directions,
        t_min=t_min,
        t_max=t_max,
    )


def walk_faces_rpa(
    tree: Octree,
    start_cell_id: int,
    active_faces: tuple[int, ...],
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    t_event: float,
) -> tuple[int, ...]:
    """Test wrapper around the production spherical face walk."""
    active_faces_array = np.asarray(active_faces, dtype=np.int64)
    crossing = np.empty(3, dtype=np.float64)
    crossing_rpa = np.empty(3, dtype=np.float64)
    path = np.empty(3, dtype=np.int64)
    active_face_by_axis = np.empty(3, dtype=np.int64)
    active_face_order = np.empty(6, dtype=np.int64)
    path_count = raytracer_spherical.walk_faces(
        tree.cell_neighbor,
        tree._domain_bounds,
        tree.cell_bounds,
        start_cell_id,
        active_faces_array,
        int(active_faces_array.size),
        direction_xyz,
        origin_xyz,
        t_event,
        crossing,
        crossing_rpa,
        path,
        active_face_by_axis,
        active_face_order,
    )
    if path_count < 0:
        raise ValueError("Spherical event face patch is ambiguous under destination-side ownership.")
    return tuple(int(path[i]) for i in range(path_count))


def _rpa_to_xyz(rpa: tuple[float, float, float]) -> np.ndarray:
    """Convert one spherical point to Cartesian coordinates."""
    radius, polar, azimuth = rpa
    sin_polar = math.sin(float(polar))
    return np.array(
        (
            float(radius) * sin_polar * math.cos(float(azimuth)),
            float(radius) * sin_polar * math.sin(float(azimuth)),
            float(radius) * math.cos(float(polar)),
        ),
        dtype=float,
    )


def _times_close(left: float, right: float) -> bool:
    """Return whether two ray parameters should be treated as one event."""
    scale = max(abs(float(left)), abs(float(right)), 1.0)
    return abs(float(left) - float(right)) <= (_TIME_ATOL + _TIME_RTOL * scale)


def _quadratic_roots(a: float, b: float, c: float) -> np.ndarray:
    """Return sorted real roots of one quadratic or linear polynomial."""
    qa = float(a)
    qb = float(b)
    qc = float(c)
    scale = max(abs(qa), abs(qb), abs(qc), 1.0)
    tol = _TIME_ATOL + _TIME_RTOL * scale
    if abs(qa) <= tol:
        if abs(qb) <= tol:
            return np.empty(0, dtype=np.float64)
        return np.array((-qc / qb,), dtype=np.float64)
    disc = qb * qb - 4.0 * qa * qc
    if disc < -tol:
        return np.empty(0, dtype=np.float64)
    if disc < 0.0:
        disc = 0.0
    sqrt_disc = math.sqrt(disc)
    denom = 2.0 * qa
    root0 = (-qb - sqrt_disc) / denom
    root1 = (-qb + sqrt_disc) / denom
    if _times_close(root0, root1):
        return np.array((0.5 * (root0 + root1),), dtype=np.float64)
    return np.array(tuple(sorted((root0, root1))), dtype=np.float64)


def _sphere_roots(origin_xyz: np.ndarray, direction_xyz: np.ndarray, radius: float) -> np.ndarray:
    """Return ray parameters where one line meets one sphere centered at the origin."""
    radius_value = float(radius)
    a = float(np.dot(direction_xyz, direction_xyz))
    b = 2.0 * float(np.dot(origin_xyz, direction_xyz))
    c = float(np.dot(origin_xyz, origin_xyz)) - radius_value * radius_value
    return _quadratic_roots(a, b, c)


def _polar_roots(origin_xyz: np.ndarray, direction_xyz: np.ndarray, polar: float) -> np.ndarray:
    """Return ray parameters where one line meets one constant-polar cone."""
    polar_value = float(polar)
    if polar_value <= 0.0 or polar_value >= _PI:
        return np.empty(0, dtype=np.float64)
    if _times_close(polar_value, 0.5 * _PI):
        dz = float(direction_xyz[2])
        if abs(dz) <= (_TIME_ATOL + _TIME_RTOL):
            return np.empty(0, dtype=np.float64)
        return np.array((-float(origin_xyz[2]) / dz,), dtype=np.float64)

    cos_polar = math.cos(polar_value)
    sin_polar = math.sin(polar_value)
    xy0 = float(origin_xyz[0] * origin_xyz[0] + origin_xyz[1] * origin_xyz[1])
    xyd = float(direction_xyz[0] * direction_xyz[0] + direction_xyz[1] * direction_xyz[1])
    xy_cross = float(origin_xyz[0] * direction_xyz[0] + origin_xyz[1] * direction_xyz[1])
    z0 = float(origin_xyz[2])
    zd = float(direction_xyz[2])
    cos_sq = cos_polar * cos_polar
    sin_sq = sin_polar * sin_polar
    roots = _quadratic_roots(
        cos_sq * xyd - sin_sq * zd * zd,
        2.0 * (cos_sq * xy_cross - sin_sq * z0 * zd),
        cos_sq * xy0 - sin_sq * z0 * z0,
    )
    if roots.size == 0:
        return roots

    valid: list[float] = []
    for root in roots.tolist():
        point_xyz = np.asarray(origin_xyz, dtype=float) + float(root) * np.asarray(direction_xyz, dtype=float)
        radius = float(np.linalg.norm(point_xyz))
        if radius <= (_TIME_ATOL + _TIME_RTOL):
            continue
        if float(point_xyz[2]) * cos_polar < -(_TIME_ATOL + _TIME_RTOL):
            continue
        polar_here = math.acos(np.clip(float(point_xyz[2]) / radius, -1.0, 1.0))
        if _times_close(polar_here, polar_value):
            valid.append(float(root))
    return np.asarray(valid, dtype=np.float64)


def _azimuth_plane_roots(origin_xyz: np.ndarray, direction_xyz: np.ndarray, azimuth: float) -> np.ndarray:
    """Return the ray parameter where one line meets one constant-azimuth plane."""
    azimuth_value = float(azimuth)
    normal_x = math.sin(azimuth_value)
    normal_y = -math.cos(azimuth_value)
    numerator = normal_x * float(origin_xyz[0]) + normal_y * float(origin_xyz[1])
    denominator = normal_x * float(direction_xyz[0]) + normal_y * float(direction_xyz[1])
    tol = _TIME_ATOL + _TIME_RTOL * max(abs(numerator), abs(denominator), 1.0)
    if abs(denominator) <= tol:
        return np.empty(0, dtype=np.float64)
    return np.array((-numerator / denominator,), dtype=np.float64)


def _append_candidate_time(times: list[float], value: float, t_lo: float, t_hi: float) -> None:
    """Append one clipped candidate event time."""
    t = float(value)
    if t < t_lo and not _times_close(t, t_lo):
        return
    if t > t_hi and not _times_close(t, t_hi):
        return
    if _times_close(t, t_lo):
        t = float(t_lo)
    elif _times_close(t, t_hi):
        t = float(t_hi)
    times.append(t)


def _unique_sorted_times(times: list[float], t_lo: float, t_hi: float) -> np.ndarray:
    """Return sorted unique event times including the interval endpoints."""
    ordered = np.sort(np.asarray(times, dtype=np.float64))
    unique: list[float] = [float(ordered[0])]
    for value in ordered[1:]:
        if _times_close(float(value), unique[-1]):
            unique[-1] = float(value)
            continue
        unique.append(float(value))
    out = np.asarray(unique, dtype=np.float64)
    out[0] = float(t_lo)
    out[-1] = float(t_hi)
    return out


def _domain_interval_rpa(origin_xyz: np.ndarray, direction_xyz: np.ndarray, r_max: float) -> tuple[float, float] | None:
    """Return the clipped outer-sphere interval for one spherical ray walk."""
    roots = _sphere_roots(origin_xyz, direction_xyz, float(r_max))
    if roots.size < 2:
        return None
    t_enter = float(roots[0])
    t_exit = float(roots[-1])
    if not t_enter < t_exit:
        return None
    return t_enter, t_exit


def _oracle_event_times(
    tree: Octree,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    t_lo: float,
    t_hi: float,
) -> np.ndarray:
    """Return exact fine-grid boundary times for one spherical ray segment."""
    candidate_times: list[float] = [float(t_lo), float(t_hi)]

    radial_edges = np.asarray(tree.radial_edges, dtype=np.float64)
    radial_edges = radial_edges[np.isfinite(radial_edges)]
    for radius in np.unique(radial_edges):
        for root in _sphere_roots(origin_xyz, direction_xyz, float(radius)).tolist():
            _append_candidate_time(candidate_times, float(root), float(t_lo), float(t_hi))

    n_polar = int(tree.leaf_shape[1])
    for polar_id in range(1, n_polar):
        polar = polar_id * (_PI / float(n_polar))
        for root in _polar_roots(origin_xyz, direction_xyz, polar).tolist():
            _append_candidate_time(candidate_times, float(root), float(t_lo), float(t_hi))

    n_azimuth = int(tree.leaf_shape[2])
    for azimuth_id in range(n_azimuth + 1):
        azimuth = azimuth_id * (_TWO_PI / float(n_azimuth))
        for root in _azimuth_plane_roots(origin_xyz, direction_xyz, azimuth).tolist():
            _append_candidate_time(candidate_times, float(root), float(t_lo), float(t_hi))
    return _unique_sorted_times(candidate_times, float(t_lo), float(t_hi))


def _lookup_oracle_trace(
    tree: Octree,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    *,
    t_min: float = 0.0,
    t_max: float = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the independent midpoint-ownership oracle trace until first exit."""
    _domain_lo, domain_hi = tree.domain_bounds(coord="rpa")
    interval = _domain_interval_rpa(origin_xyz, direction_xyz, float(domain_hi[0]))
    if interval is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    t_lo = max(float(t_min), float(interval[0]))
    t_hi = min(float(t_max), float(interval[1]))
    if not t_lo < t_hi:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    event_times = _oracle_event_times(tree, origin_xyz, direction_xyz, t_lo, t_hi)
    out_cell_ids: list[int] = []
    out_times: list[float] = [float(event_times[0])]
    started = False
    for t_start, t_stop in zip(event_times[:-1], event_times[1:]):
        if not float(t_stop) > float(t_start):
            continue
        t_mid = 0.5 * (float(t_start) + float(t_stop))
        mid_xyz = np.asarray(origin_xyz, dtype=float) + t_mid * np.asarray(direction_xyz, dtype=float)
        owner = int(tree.lookup_points(mid_xyz[None, :], coord="xyz")[0])
        if not started:
            if owner < 0:
                raise ValueError("The oracle trace must start inside one occupied RPA leaf.")
            started = True
        elif owner < 0:
            break
        if out_cell_ids and owner == out_cell_ids[-1]:
            out_times[-1] = float(t_stop)
            continue
        out_cell_ids.append(owner)
        out_times.append(float(t_stop))
    return np.asarray(out_cell_ids, dtype=np.int64), np.asarray(out_times, dtype=np.float64)


def _assert_trace_matches_lookup_oracle(
    tree: Octree,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    *,
    t_min: float = 0.0,
    t_max: float = np.inf,
) -> None:
    """Check one standalone `rpa` trace against the independent lookup oracle."""
    traced_cell_ids, traced_times = trace_rpa_test_path(tree, origin_xyz, direction_xyz, t_min=t_min, t_max=t_max)
    oracle_cell_ids, oracle_times = _lookup_oracle_trace(tree, origin_xyz, direction_xyz, t_min=t_min, t_max=t_max)

    np.testing.assert_array_equal(traced_cell_ids, oracle_cell_ids)
    np.testing.assert_allclose(traced_times, oracle_times, atol=1.0e-12, rtol=0.0)
    if traced_cell_ids.size == 0:
        return

    mid_t = 0.5 * (traced_times[:-1] + traced_times[1:])
    mid_xyz = np.asarray(origin_xyz, dtype=float)[None, :] + mid_t[:, None] * np.asarray(direction_xyz, dtype=float)[None, :]
    owners = np.asarray(tree.lookup_points(mid_xyz, coord="xyz"), dtype=np.int64)
    np.testing.assert_array_equal(owners, traced_cell_ids)


def _assert_trace_sweep_matches_lookup_oracle_rpa(
    tree: Octree,
    origins_rpa: tuple[tuple[float, float, float], ...],
    direction_xyz: tuple[float, float, float],
    *,
    clipped: bool = False,
) -> None:
    """Check one whole origin sweep for one direction against the independent oracle."""
    direction = _normalize_direction(direction_xyz)
    for origin_rpa in origins_rpa:
        origin = _rpa_to_xyz(origin_rpa)
        try:
            if clipped:
                _full_cell_ids, full_times = _lookup_oracle_trace(tree, origin, direction)
                assert full_times.size >= 2
                clip_lo = float(full_times[0] + 0.173 * (full_times[-1] - full_times[0]))
                clip_hi = float(full_times[0] + 0.781 * (full_times[-1] - full_times[0]))
                _assert_trace_matches_lookup_oracle(tree, origin, direction, t_min=clip_lo, t_max=clip_hi)
            else:
                _assert_trace_matches_lookup_oracle(tree, origin, direction)
        except AssertionError as exc:
            raise AssertionError(
                f"origin_rpa={origin_rpa} direction_xyz={direction_xyz} clipped={bool(clipped)}"
            ) from exc


def _assert_direction_sweep_matches_lookup_oracle_rpa(
    tree: Octree,
    origins_rpa: tuple[tuple[float, float, float], ...],
    directions_xyz: tuple[tuple[float, float, float], ...],
    *,
    clipped: bool = False,
) -> None:
    """Check one whole direction/origin sweep against the independent oracle."""
    for direction_xyz in directions_xyz:
        try:
            _assert_trace_sweep_matches_lookup_oracle_rpa(tree, origins_rpa, direction_xyz, clipped=clipped)
        except AssertionError as exc:
            raise AssertionError(f"direction_xyz={direction_xyz} clipped={bool(clipped)}") from exc


def _render_standalone_midpoint_image(
    interpolator: OctreeInterpolator,
    origins: np.ndarray,
    directions: np.ndarray,
    segments,
) -> np.ndarray:
    """Integrate one standalone packed segment bundle by midpoint sampling."""
    o = np.asarray(origins, dtype=float)
    d = np.asarray(directions, dtype=float)
    assert o.shape == d.shape
    ray_shape = tuple(o.shape[:-1]) if o.ndim > 1 else (1,)
    o_flat = o.reshape((-1, 3))
    d_flat = d.reshape((-1, 3))
    accum = np.zeros((o_flat.shape[0],), dtype=float)
    for ray_id in range(o_flat.shape[0]):
        cell_lo = int(segments.ray_offsets[ray_id])
        cell_hi = int(segments.ray_offsets[ray_id + 1])
        if cell_lo == cell_hi:
            continue
        time_lo = int(segments.time_offsets[ray_id])
        time_hi = int(segments.time_offsets[ray_id + 1])
        ray_times = np.asarray(segments.times[time_lo:time_hi], dtype=float)
        segment_lengths = np.diff(ray_times)
        mid_t = 0.5 * (ray_times[:-1] + ray_times[1:])
        mid_xyz = o_flat[ray_id] + mid_t[:, None] * d_flat[ray_id]
        samples = np.asarray(interpolator(mid_xyz, query_coord="xyz", log_outside_domain=False), dtype=float)
        accum[ray_id] = float(np.sum(samples * segment_lengths, dtype=float))
    return accum.reshape(ray_shape)


def _ray_slice(segments, ray_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return one packed standalone ray slice."""
    cell_lo = int(segments.ray_offsets[ray_id])
    cell_hi = int(segments.ray_offsets[ray_id + 1])
    time_lo = int(segments.time_offsets[ray_id])
    time_hi = int(segments.time_offsets[ray_id + 1])
    return segments.cell_ids[cell_lo:cell_hi], segments.times[time_lo:time_hi]


def _direction_to_target(origin_rpa: tuple[float, float, float], target_rpa: tuple[float, float, float]) -> np.ndarray:
    """Return one unit Cartesian direction from one RPA point to another."""
    origin_xyz = _rpa_to_xyz(origin_rpa)
    target_xyz = _rpa_to_xyz(target_rpa)
    return _normalize_direction(tuple((target_xyz - origin_xyz).tolist()))


def _lookup_after_first_event_rpa(tree: Octree, origin_xyz: np.ndarray, direction_xyz: np.ndarray) -> tuple[int, float]:
    """Return the owner and time of the first open interval after one event."""
    oracle_cell_ids, oracle_times = _lookup_oracle_trace(tree, origin_xyz, direction_xyz)
    assert oracle_cell_ids.size >= 2
    return int(oracle_cell_ids[1]), float(oracle_times[1])


def _assert_first_event_matches_lookup_rpa(
    origin_rpa: tuple[float, float, float],
    target_rpa: tuple[float, float, float],
    expected_active_faces: tuple[int, ...],
) -> None:
    """Check one first face, edge, or corner event against independent post-event ownership."""
    tree = _build_uniform_rpa_tree()
    origin = _rpa_to_xyz(origin_rpa)
    direction = _direction_to_target(origin_rpa, target_rpa)
    start_cell_id = _find_cell_rpa_single(tree, origin)
    assert start_cell_id >= 0

    t_exit, active_faces, axis_transfer = _cell_exit_event_rpa(tree.cell_bounds, start_cell_id, origin, direction, 0.0)
    assert not axis_transfer
    path = walk_faces_rpa(tree, start_cell_id, active_faces, origin, direction, t_exit)
    traced_cell_ids, traced_times = trace_rpa_test_path(tree, origin, direction)
    expected_owner, expected_t_exit = _lookup_after_first_event_rpa(tree, origin, direction)

    assert active_faces == expected_active_faces
    assert _times_close(t_exit, expected_t_exit)
    assert traced_cell_ids[0] == start_cell_id
    assert _times_close(traced_times[1], t_exit)
    assert 1 <= len(path) <= len(active_faces)
    assert path[-1] == expected_owner


def _assert_multiface_event_is_order_invariant_rpa(
    origin_rpa: tuple[float, float, float],
    target_rpa: tuple[float, float, float],
    expected_active_faces: tuple[int, ...],
) -> None:
    """Check that one tied RPA event reaches the same final owner for every face order."""
    tree = _build_uniform_rpa_tree()
    origin = _rpa_to_xyz(origin_rpa)
    direction = _direction_to_target(origin_rpa, target_rpa)
    start_cell_id = _find_cell_rpa_single(tree, origin)
    assert start_cell_id >= 0

    t_exit, active_faces, axis_transfer = _cell_exit_event_rpa(tree.cell_bounds, start_cell_id, origin, direction, 0.0)
    assert not axis_transfer
    assert active_faces == expected_active_faces

    expected_owner, _expected_t_exit = _lookup_after_first_event_rpa(tree, origin, direction)
    finals = set()
    for permutation in itertools.permutations(active_faces):
        path = walk_faces_rpa(tree, start_cell_id, permutation, origin, direction, t_exit)
        finals.add(int(path[-1]))
    assert finals == {expected_owner}


def _benchmark_plane_axis_points(lo: float, hi: float, n: int) -> np.ndarray:
    step = (float(hi) - float(lo)) / float(n)
    return float(lo) + (np.arange(int(n), dtype=float) + 0.5) * step


def _benchmark_ray_setup(
    *,
    n_plane: int,
    bounds: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    y = _benchmark_plane_axis_points(ymin, ymax, int(n_plane))
    z = _benchmark_plane_axis_points(zmin, zmax, int(n_plane))
    yg, zg = np.meshgrid(y, z, indexing="xy")
    xg = np.full_like(yg, float(xmin), dtype=float)
    origins = np.stack((xg, yg, zg), axis=-1)
    directions = np.zeros_like(origins)
    directions[..., 0] = 1.0
    return origins, directions, float(xmax - xmin)


@lru_cache(maxsize=1)
def _build_sc_benchmark_trace_case():
    """Return the cached SC 64x64 benchmark tracing setup."""
    from batread import Dataset

    from sample_data_helper import data_file

    ds = Dataset.from_file(str(data_file("3d__var_4_n00044000.plt")))
    tree = Octree.from_ds(ds)
    xyz = np.column_stack(tuple(np.asarray(ds[name], dtype=float) for name in ("X [R]", "Y [R]", "Z [R]")))
    dmin = np.min(xyz, axis=0)
    dmax = np.max(xyz, axis=0)
    bounds = (
        float(dmin[0]),
        float(dmax[0]),
        float(dmin[1]),
        float(dmax[1]),
        float(dmin[2]),
        float(dmax[2]),
    )
    origins, directions, t_end = _benchmark_ray_setup(n_plane=64, bounds=bounds)
    return ds, tree, origins, directions, float(t_end)


def _sc_benchmark_ray(
    iz: int,
    iy: int,
) -> tuple[Octree, np.ndarray, np.ndarray, float]:
    """Return one traced SC benchmark ray by image indices."""
    _ds, tree, origins, directions, t_end = _build_sc_benchmark_trace_case()
    return tree, np.asarray(origins[iz, iy], dtype=float), np.asarray(directions[iz, iy], dtype=float), float(t_end)


def test_trace_rpa_test_path_matches_lookup_oracle_for_many_interior_rays() -> None:
    tree = _build_uniform_rpa_tree()
    _assert_direction_sweep_matches_lookup_oracle_rpa(tree, _INTERIOR_ORIGINS, _RAY_DIRECTIONS)


def test_trace_rpa_test_path_matches_lookup_oracle_for_clipped_interior_rays() -> None:
    tree = _build_uniform_rpa_tree()
    _assert_direction_sweep_matches_lookup_oracle_rpa(tree, _INTERIOR_ORIGINS, _RAY_DIRECTIONS, clipped=True)


def test_trace_rpa_test_path_matches_lookup_oracle_for_many_seam_rays() -> None:
    tree = _build_uniform_rpa_tree()
    _assert_direction_sweep_matches_lookup_oracle_rpa(tree, _SEAM_ORIGINS, _SEAM_DIRECTIONS)


def test_trace_rpa_test_path_matches_lookup_oracle_for_dense_seam_starts() -> None:
    tree = _build_uniform_rpa_tree()
    _assert_direction_sweep_matches_lookup_oracle_rpa(tree, _SEAM_STRESS_ORIGINS, _SEAM_STRESS_DIRECTIONS)


def test_trace_rpa_test_path_matches_lookup_oracle_for_clipped_seam_rays() -> None:
    tree = _build_uniform_rpa_tree()
    _assert_direction_sweep_matches_lookup_oracle_rpa(tree, _SEAM_ORIGINS, _SEAM_DIRECTIONS, clipped=True)


@pytest.mark.parametrize(
    ("origin_xyz", "direction_xyz"),
    (
        ((-0.4, 0.0, 2.0), (1.0, 0.0, 0.0)),
        ((-0.4, 0.0, -2.0), (1.0, 0.0, 0.0)),
    ),
    ids=("north_pole_crossing", "south_pole_crossing"),
)
def test_trace_rpa_test_path_matches_lookup_oracle_when_crossing_the_polar_axis_over_one_pole(
    origin_xyz: tuple[float, float, float],
    direction_xyz: tuple[float, float, float],
) -> None:
    tree = _build_uniform_rpa_tree()
    origin = np.asarray(origin_xyz, dtype=float)
    direction = _normalize_direction(direction_xyz)

    cell_ids, times = trace_rpa_test_path(tree, origin, direction)

    assert np.all(np.diff(times) > 0.0)
    _assert_trace_matches_lookup_oracle(tree, origin, direction)
    assert np.all(cell_ids >= 0)


def test_trace_rpa_test_path_matches_lookup_oracle_for_one_exact_seam_start_into_the_last_wedge() -> None:
    tree = _build_uniform_rpa_tree()
    origin = _rpa_to_xyz((1.5, 0.95, 0.0))
    direction = _normalize_direction((-0.2, -1.0, 0.0))

    _assert_trace_matches_lookup_oracle(tree, origin, direction)


def test_trace_rpa_test_path_matches_lookup_oracle_for_one_exact_seam_start_into_the_first_wedge() -> None:
    tree = _build_uniform_rpa_tree()
    origin = _rpa_to_xyz((1.5, 0.95, 0.0))
    direction = _normalize_direction((0.2, 1.0, -0.02))

    _assert_trace_matches_lookup_oracle(tree, origin, direction)


def test_find_cell_rpa_single_uses_one_exact_half_open_owner_at_the_seam() -> None:
    tree = _build_uniform_rpa_tree()

    first_wedge = _rpa_to_xyz((1.5, 0.95, 0.0))
    last_wedge = _rpa_to_xyz((1.5, 0.95, _TWO_PI - 1.0e-12))

    assert _find_cell_rpa_single(tree, first_wedge) == 8
    assert _find_cell_rpa_single(tree, last_wedge) == 15


def test_event_subface_id_rpa_uses_destination_side_rpa_intervals() -> None:
    domain_bounds = np.array(
        (
            (1.0, 2.0),
            (0.0, 1.0),
            (0.0, 0.5),
        ),
        dtype=float,
    )
    cell_bounds = np.array(
        (
            (
                (1.0, 1.0),
                (0.0, 1.0),
                (0.0, 0.5),
            ),
            (
                (2.0, 0.25),
                (0.0, 0.49),
                (0.0, 0.25),
            ),
            (
                (2.0, 0.25),
                (0.0, 0.49),
                (0.25, 0.25),
            ),
            (
                (2.0, 0.25),
                (0.49, 0.51),
                (0.0, 0.25),
            ),
            (
                (2.0, 0.25),
                (0.49, 0.51),
                (0.25, 0.25),
            ),
        ),
        dtype=float,
    )
    cell_neighbor = -np.ones((5, 6, 4), dtype=np.int64)
    cell_neighbor[0, 1] = np.array((1, 2, 3, 4), dtype=np.int64)
    scratch_rpa = np.array((2.0, 0.5, 0.25), dtype=float)
    scratch_xyz = _rpa_to_xyz((2.0, 0.5, 0.25))
    direction_xyz = np.asarray(scratch_xyz, dtype=float)

    active_face_by_axis, active_face_order = _fill_active_face_state_rpa((1,))
    subface_id = _event_subface_id_rpa(
        cell_neighbor,
        domain_bounds,
        cell_bounds,
        0,
        1,
        active_face_by_axis,
        active_face_order,
        int(active_face_order[1]),
        np.asarray(scratch_xyz, dtype=float),
        direction_xyz,
        scratch_rpa,
    )

    assert subface_id == 2


def test_rpa_first_event_matches_lookup_after_the_event() -> None:
    for origin_rpa, target_rpa, expected_active_faces in _FIRST_EVENT_SINGLE_FACE_CASES + _FIRST_EVENT_MULTIFACE_CASES:
        try:
            _assert_first_event_matches_lookup_rpa(origin_rpa, target_rpa, expected_active_faces)
        except AssertionError as exc:
            raise AssertionError(
                f"origin_rpa={origin_rpa} target_rpa={target_rpa} expected_active_faces={expected_active_faces}"
            ) from exc


def test_rpa_multiface_first_events_are_order_invariant() -> None:
    for origin_rpa, target_rpa, expected_active_faces in _FIRST_EVENT_MULTIFACE_CASES:
        try:
            _assert_multiface_event_is_order_invariant_rpa(origin_rpa, target_rpa, expected_active_faces)
        except AssertionError as exc:
            raise AssertionError(
                f"origin_rpa={origin_rpa} target_rpa={target_rpa} expected_active_faces={expected_active_faces}"
            ) from exc


def test_trace_rpa_test_path_handles_one_outside_entry_start() -> None:
    tree = _build_uniform_rpa_tree()
    origin = np.array((-4.5, 0.25, -0.55), dtype=float)
    direction = _normalize_direction((1.0, 0.0, 0.0))

    _assert_trace_matches_lookup_oracle(tree, origin, direction)


def test_trace_rpa_test_path_handles_one_miss() -> None:
    tree = _build_uniform_rpa_tree()
    origin = np.array((4.0, 4.0, 4.0), dtype=float)
    direction = _normalize_direction((1.0, 0.0, 0.0))

    cell_ids, times = trace_rpa_test_path(tree, origin, direction)

    assert cell_ids.size == 0
    assert times.size == 0


def test_trace_rpa_test_path_stops_at_the_inner_cavity_exit() -> None:
    tree = _build_uniform_rpa_tree()
    origin = np.array((2.0, 0.5, 0.25), dtype=float)
    direction = _normalize_direction((-1.0, 0.0, 0.0))

    cell_ids, times = trace_rpa_test_path(tree, origin, direction)

    assert np.all(cell_ids >= 0)
    assert np.all(np.diff(times) > 0.0)
    interval = _domain_interval_rpa(origin, direction, 3.0)
    assert interval is not None
    assert times[-1] < float(interval[1])
    _assert_trace_matches_lookup_oracle(tree, origin, direction)


def test_trace_rpa_test_rays_preserve_batch_shape_and_match_single_ray_paths() -> None:
    tree = _build_uniform_rpa_tree()
    origins = np.array(
        [
            [_rpa_to_xyz(_INTERIOR_ORIGINS[0]), _rpa_to_xyz(_INTERIOR_ORIGINS[1])],
            [_rpa_to_xyz(_INTERIOR_ORIGINS[2]), _rpa_to_xyz(_INTERIOR_ORIGINS[3])],
        ],
        dtype=float,
    )
    directions = np.array(
        [
            [[1.0, 0.0, 0.0], [1.0, 0.05, -0.04]],
            [[1.0, 0.08, 0.0], [1.0, 0.0, 0.06]],
        ],
        dtype=float,
    )
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    segments = trace_rpa_test_rays(tree, origins, directions)

    assert segments.ray_shape == (2, 2)
    for ray_id, (origin, direction) in enumerate(
        zip(origins.reshape((-1, 3)), directions.reshape((-1, 3)), strict=True)
    ):
        expected_cell_ids, expected_times = trace_rpa_test_path(tree, origin, direction)
        ray_cell_ids, ray_times = _ray_slice(segments, ray_id)
        np.testing.assert_array_equal(ray_cell_ids, expected_cell_ids)
        np.testing.assert_array_equal(ray_times, expected_times)


def test_trace_rpa_test_rays_preserve_batch_shape_and_match_single_seam_paths() -> None:
    tree = _build_uniform_rpa_tree()
    seam_origins = np.array([_rpa_to_xyz(origin_rpa) for origin_rpa in _SEAM_ORIGINS], dtype=float).reshape((4, 5, 3))
    seam_directions = np.array(
        [
            [_normalize_direction(_SEAM_DIRECTIONS[(row + col) % len(_SEAM_DIRECTIONS)]) for col in range(5)]
            for row in range(4)
        ],
        dtype=float,
    )

    segments = trace_rpa_test_rays(tree, seam_origins, seam_directions)

    assert segments.ray_shape == (4, 5)
    for ray_id, (origin, direction) in enumerate(
        zip(seam_origins.reshape((-1, 3)), seam_directions.reshape((-1, 3)), strict=True)
    ):
        expected_cell_ids, expected_times = trace_rpa_test_path(tree, origin, direction)
        ray_cell_ids, ray_times = _ray_slice(segments, ray_id)
        np.testing.assert_array_equal(ray_cell_ids, expected_cell_ids)
        np.testing.assert_array_equal(ray_times, expected_times)


def test_trace_rpa_test_rays_drive_constant_shell_integral() -> None:
    tree = _build_uniform_rpa_tree()
    interp = OctreeInterpolator(tree, np.ones(int(np.max(tree.corners)) + 1, dtype=float))
    origins = np.array([[[2.0, 0.5, 0.25]]], dtype=float)
    directions = np.array([[[-1.0, 0.0, 0.0]]], dtype=float)

    segments = trace_rpa_test_rays(tree, origins, directions)
    image = _render_standalone_midpoint_image(interp, origins, directions, segments)

    x0 = float(origins[0, 0, 0])
    impact_sq = float(origins[0, 0, 1] * origins[0, 0, 1] + origins[0, 0, 2] * origins[0, 0, 2])
    expected = x0 - math.sqrt(1.0 - impact_sq)

    assert segments.ray_shape == (1, 1)
    assert image.shape == (1, 1)
    np.testing.assert_allclose(image, np.array([[expected]], dtype=float), atol=1.0e-10, rtol=0.0)


def test_rpa_midpoint_image_matches_standalone_midpoint_render() -> None:
    tree = _build_uniform_rpa_tree()
    tracer = OctreeRayTracer(tree)
    interpolator = OctreeInterpolator(tree, np.ones(int(np.max(tree.corners)) + 1, dtype=float))
    origins = np.array([[[2.0, 0.5, 0.25]]], dtype=float)
    directions = np.array([[[-1.0, 0.0, 0.0]]], dtype=float)

    image, counts = tracer.midpoint_image(interpolator, origins, directions)
    segments = tracer.trace(origins, directions)
    expected = _render_standalone_midpoint_image(interpolator, origins, directions, segments)

    np.testing.assert_array_equal(counts, np.array([[2]], dtype=np.int64))
    np.testing.assert_allclose(image, expected, atol=1.0e-10, rtol=0.0)


def test_rpa_render_midpoint_image_matches_standalone_midpoint_render() -> None:
    tree = _build_uniform_rpa_tree()
    tracer = OctreeRayTracer(tree)
    interpolator = OctreeInterpolator(tree, np.ones(int(np.max(tree.corners)) + 1, dtype=float))
    origins = np.array([[[2.0, 0.5, 0.25]]], dtype=float)
    directions = np.array([[[-1.0, 0.0, 0.0]]], dtype=float)
    segments = tracer.trace(origins, directions)

    image = render_midpoint_image(interpolator, origins, directions, segments)
    expected = _render_standalone_midpoint_image(interpolator, origins, directions, segments)

    np.testing.assert_allclose(image, expected, atol=1.0e-10, rtol=0.0)


def test_rpa_render_midpoint_image_preserves_vector_components() -> None:
    tree = _build_uniform_rpa_tree()
    tracer = OctreeRayTracer(tree)
    n_points = int(np.max(tree.corners)) + 1
    interpolator = OctreeInterpolator(
        tree,
        np.column_stack((np.ones(n_points, dtype=float), np.full(n_points, 3.0, dtype=float))),
    )
    origins = np.array([[[2.0, 0.5, 0.25]]], dtype=float)
    directions = np.array([[[-1.0, 0.0, 0.0]]], dtype=float)
    segments = tracer.trace(origins, directions)
    _cell_ids, times = _ray_slice(segments, 0)
    total_length = float(np.sum(np.diff(np.asarray(times, dtype=float)), dtype=float))

    image = render_midpoint_image(interpolator, origins, directions, segments)

    assert image.shape == (1, 1, 2)
    np.testing.assert_allclose(
        image,
        np.array([[[total_length, 3.0 * total_length]]], dtype=float),
        atol=1.0e-10,
        rtol=0.0,
    )


def test_rpa_midpoint_image_preserves_vector_components() -> None:
    tree = _build_uniform_rpa_tree()
    tracer = OctreeRayTracer(tree)
    n_points = int(np.max(tree.corners)) + 1
    interpolator = OctreeInterpolator(
        tree,
        np.column_stack((np.ones(n_points, dtype=float), np.full(n_points, 3.0, dtype=float))),
    )
    origins = np.array([[[2.0, 0.5, 0.25]]], dtype=float)
    directions = np.array([[[-1.0, 0.0, 0.0]]], dtype=float)
    segments = tracer.trace(origins, directions)
    _cell_ids, times = _ray_slice(segments, 0)
    total_length = float(np.sum(np.diff(np.asarray(times, dtype=float)), dtype=float))

    image, counts = tracer.midpoint_image(interpolator, origins, directions)

    assert image.shape == (1, 1, 2)
    np.testing.assert_array_equal(counts, np.array([[2]], dtype=np.int64))
    np.testing.assert_allclose(
        image,
        np.array([[[total_length, 3.0 * total_length]]], dtype=float),
        atol=1.0e-10,
        rtol=0.0,
    )


def test_rpa_trace_chunk_uses_spherical_initial_crossing_capacity(monkeypatch) -> None:
    tree = _build_uniform_rpa_tree()
    monkeypatch.setattr(raytracer_spherical, "DEFAULT_CROSSING_BUFFER_SIZE", 1)
    tracer = OctreeRayTracer(tree)

    segments = tracer.trace(
        np.array([[[2.0, 0.5, 0.25]]], dtype=float),
        np.array([[[-1.0, 0.0, 0.0]]], dtype=float),
    )

    np.testing.assert_array_equal(np.diff(segments.ray_offsets), np.array([2], dtype=np.int64))
    np.testing.assert_array_equal(np.diff(segments.time_offsets), np.array([3], dtype=np.int64))
    assert tracer._crossing_buffer_size == 2


@pytest.mark.pooch
def test_sc_benchmark_artifact_rays_render_match_standalone_midpoint_image() -> None:
    ds, tree, origins, directions, t_end = _build_sc_benchmark_trace_case()
    tracer = OctreeRayTracer(tree)
    interpolator = OctreeInterpolator(tree, np.asarray(ds["Rho [g/cm^3]"], dtype=float))
    sample_origins = np.asarray(
        (
            origins[32, 33],
            origins[41, 22],
        ),
        dtype=float,
    )
    sample_directions = np.asarray(
        (
            directions[32, 33],
            directions[41, 22],
        ),
        dtype=float,
    )

    segments = tracer.trace(sample_origins, sample_directions, t_min=0.0, t_max=float(t_end))
    image = render_midpoint_image(interpolator, sample_origins, sample_directions, segments)
    expected = _render_standalone_midpoint_image(interpolator, sample_origins, sample_directions, segments)

    np.testing.assert_allclose(image, expected, atol=1.0e-12, rtol=0.0)


@pytest.mark.pooch
def test_trace_sc_benchmark_vertical_artifact_ray_matches_lookup_oracle() -> None:
    tree, origin, direction, t_end = _sc_benchmark_ray(32, 33)
    _assert_trace_matches_lookup_oracle(tree, origin, direction, t_max=float(t_end))


@pytest.mark.pooch
def test_trace_sc_benchmark_corner_artifact_ray_matches_lookup_oracle() -> None:
    tree, origin, direction, t_end = _sc_benchmark_ray(41, 22)
    _assert_trace_matches_lookup_oracle(tree, origin, direction, t_max=float(t_end))


def test_trace_rpa_camera_entry_ray_matches_lookup_oracle_at_outer_boundary() -> None:
    tree = _build_uniform_rpa_tree()
    origin = np.array(
        (
            9.391828094135867,
            -0.420667604443791,
            -0.20731583241380047,
        ),
        dtype=float,
    )
    direction = np.array(
        (
            -0.9654086575047712,
            -0.03160420711004458,
            -0.25881904510252074,
        ),
        dtype=float,
    )

    _assert_trace_matches_lookup_oracle(tree, origin, direction)


def test_rpa_trilinear_image_matches_midpoint_for_constant_field() -> None:
    tree = _build_uniform_rpa_tree()
    tracer = OctreeRayTracer(tree)
    interpolator = OctreeInterpolator(tree, np.ones(int(np.max(tree.corners)) + 1, dtype=float))
    origins = np.array([[[2.0, 0.5, 0.25]]], dtype=float)
    directions = np.array([[[-1.0, 0.0, 0.0]]], dtype=float)

    image_trilinear, counts_trilinear = tracer.trilinear_image(interpolator, origins, directions)
    image_midpoint, counts_midpoint = tracer.midpoint_image(interpolator, origins, directions)

    np.testing.assert_array_equal(counts_trilinear, counts_midpoint)
    np.testing.assert_allclose(image_trilinear, image_midpoint, atol=1.0e-10, rtol=0.0)


def test_rpa_trilinear_image_integrates_radial_field_on_radial_ray() -> None:
    tree = _build_uniform_rpa_tree()
    tracer = OctreeRayTracer(tree)
    point_radii = np.linalg.norm(np.asarray(tree._points, dtype=float), axis=1)
    interpolator = OctreeInterpolator(tree, point_radii)
    radius_start = 1.25
    segment_span = 1.5
    polar = 0.95
    azimuth = 0.55
    origins = _rpa_to_xyz((radius_start, polar, azimuth)).reshape((1, 1, 3))
    directions = _rpa_to_xyz((1.0, polar, azimuth)).reshape((1, 1, 3))

    image, counts = tracer.trilinear_image(
        interpolator,
        origins,
        directions,
        t_min=0.0,
        t_max=segment_span,
    )

    expected = radius_start * segment_span + 0.5 * segment_span * segment_span

    np.testing.assert_array_equal(counts, np.array([[2]], dtype=np.int64))
    np.testing.assert_allclose(image, np.array([[expected]], dtype=float), atol=1.0e-10, rtol=0.0)


def test_rpa_trilinear_image_preserves_vector_components() -> None:
    tree = _build_uniform_rpa_tree()
    tracer = OctreeRayTracer(tree)
    n_points = int(np.max(tree.corners)) + 1
    interpolator = OctreeInterpolator(
        tree,
        np.column_stack((np.ones(n_points, dtype=float), np.full(n_points, 3.0, dtype=float))),
    )
    origins = np.array([[[2.0, 0.5, 0.25]]], dtype=float)
    directions = np.array([[[-1.0, 0.0, 0.0]]], dtype=float)
    segments = tracer.trace(origins, directions)
    _cell_ids, times = _ray_slice(segments, 0)
    total_length = float(np.sum(np.diff(np.asarray(times, dtype=float)), dtype=float))

    image, counts = tracer.trilinear_image(interpolator, origins, directions)

    assert image.shape == (1, 1, 2)
    np.testing.assert_array_equal(counts, np.array([[2]], dtype=np.int64))
    np.testing.assert_allclose(
        image,
        np.array([[[total_length, 3.0 * total_length]]], dtype=float),
        atol=1.0e-10,
        rtol=0.0,
    )
