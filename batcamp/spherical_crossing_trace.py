#!/usr/bin/env python3
"""Spherical crossing-trace kernels."""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from numba import prange

from .octree import START
from .octree import WIDTH
from .octree import _FACE_AXIS
from .octree import _FACE_SIDE
from .octree import _FACE_TANGENTIAL_AXES
from .spherical import xyz_to_rpa_components

DEFAULT_CROSSING_BUFFER_SIZE = 32
TRACE_BUFFER_OVERFLOW = -0x0001
TRACE_START_CELL_NOT_FOUND = -0x0002
TRACE_START_WALK_FAILED = -0x0004
TRACE_INVALID_CROSSING = -0x0008
TRACE_START_QUERY_NOT_FINITE = -0x0010
TRACE_START_NO_ROOT_OWNER = -0x0040
TRACE_START_OUTSIDE_DOMAIN_AZIMUTH_INTERVAL = -0x0080
TRACE_START_OUTSIDE_DOMAIN_AZIMUTH_OPEN_LOWER = -0x0100
TRACE_START_OUTSIDE_DOMAIN_RADIUS_INTERVAL = -0x0200
TRACE_START_OUTSIDE_DOMAIN_POLAR_INTERVAL = -0x0400
TRACE_START_OUTSIDE_DOMAIN_RADIUS_OPEN_LOWER = -0x0800
TRACE_START_OUTSIDE_DOMAIN_POLAR_OPEN_LOWER = -0x1000
CONTAINS_BOX_OK = 0x0001
DOMAIN_CONTAINS_ATOL = 1.0e-10
_PI = math.pi
_TWO_PI = 2.0 * math.pi
_TIME_ATOL = 1.0e-12
_TIME_RTOL = 64.0 * np.finfo(np.float64).eps


@njit(cache=True)
def _times_close(left: float, right: float) -> bool:
    """Return whether two ray parameters should be treated as one event."""
    scale = max(abs(float(left)), abs(float(right)), 1.0)
    return abs(float(left) - float(right)) <= (_TIME_ATOL + _TIME_RTOL * scale)


@njit(cache=True)
def _quadratic_roots(
    a: float,
    b: float,
    c: float,
    roots_out: np.ndarray,  # out
) -> int:
    """Write sorted real roots of one quadratic or linear polynomial."""
    qa = float(a)
    qb = float(b)
    qc = float(c)
    scale = max(abs(qa), abs(qb), abs(qc), 1.0)
    tol = _TIME_ATOL + _TIME_RTOL * scale
    if abs(qa) <= tol:
        if abs(qb) <= tol:
            return 0
        roots_out[0] = -qc / qb
        return 1
    disc = qb * qb - 4.0 * qa * qc
    if disc < -tol:
        return 0
    if disc < 0.0:
        disc = 0.0
    sqrt_disc = math.sqrt(disc)
    denom = 2.0 * qa
    root0 = (-qb - sqrt_disc) / denom
    root1 = (-qb + sqrt_disc) / denom
    if _times_close(root0, root1):
        roots_out[0] = 0.5 * (root0 + root1)
        return 1
    if root0 < root1:
        roots_out[0] = root0
        roots_out[1] = root1
    else:
        roots_out[0] = root1
        roots_out[1] = root0
    return 2


@njit(cache=True)
def _fill_xyz_at_time(
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    time: float,
    out: np.ndarray,  # out
) -> None:
    """Fill one Cartesian point on one straight ray at one parameter time."""
    t = float(time)
    out[0] = float(origin_xyz[0]) + t * float(direction_xyz[0])
    out[1] = float(origin_xyz[1]) + t * float(direction_xyz[1])
    out[2] = float(origin_xyz[2]) + t * float(direction_xyz[2])


@njit(cache=True)
def _fill_rpa_at_time(
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    time: float,
    out: np.ndarray,  # out
) -> None:
    """Fill spherical coordinates for one straight ray at one parameter time."""
    point_x = float(origin_xyz[0]) + float(time) * float(direction_xyz[0])
    point_y = float(origin_xyz[1]) + float(time) * float(direction_xyz[1])
    point_z = float(origin_xyz[2]) + float(time) * float(direction_xyz[2])
    radius, polar, azimuth = xyz_to_rpa_components(point_x, point_y, point_z)
    out[0] = radius
    out[1] = polar
    out[2] = azimuth


@njit(cache=True)
def _sphere_roots(
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    radius: float,
    roots_out: np.ndarray,  # out
) -> int:
    """Write ray parameters where one line meets one sphere centered at the origin."""
    radius_value = float(radius)
    a = float(np.dot(direction_xyz, direction_xyz))
    b = 2.0 * float(np.dot(origin_xyz, direction_xyz))
    c = float(np.dot(origin_xyz, origin_xyz)) - radius_value * radius_value
    return _quadratic_roots(a, b, c, roots_out)


@njit(cache=True)
def _polar_roots(
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    polar: float,
    roots_out: np.ndarray,  # out
) -> int:
    """Write ray parameters where one line meets one constant-polar cone."""
    polar_value = float(polar)
    if polar_value <= 0.0 or polar_value >= _PI:
        return 0
    if _times_close(polar_value, 0.5 * _PI):
        dz = float(direction_xyz[2])
        if abs(dz) <= (_TIME_ATOL + _TIME_RTOL):
            return 0
        roots_out[0] = -float(origin_xyz[2]) / dz
        return 1

    cos_polar = math.cos(polar_value)
    sin_polar = math.sin(polar_value)
    xy0 = float(origin_xyz[0] * origin_xyz[0] + origin_xyz[1] * origin_xyz[1])
    xyd = float(direction_xyz[0] * direction_xyz[0] + direction_xyz[1] * direction_xyz[1])
    xy_cross = float(origin_xyz[0] * direction_xyz[0] + origin_xyz[1] * direction_xyz[1])
    z0 = float(origin_xyz[2])
    zd = float(direction_xyz[2])
    cos_sq = cos_polar * cos_polar
    sin_sq = sin_polar * sin_polar
    n_root = _quadratic_roots(
        cos_sq * xyd - sin_sq * zd * zd,
        2.0 * (cos_sq * xy_cross - sin_sq * z0 * zd),
        cos_sq * xy0 - sin_sq * z0 * z0,
        roots_out,
    )
    if n_root == 0:
        return 0

    n_valid = 0
    for root_pos in range(n_root):
        root = float(roots_out[root_pos])
        point_x = float(origin_xyz[0]) + root * float(direction_xyz[0])
        point_y = float(origin_xyz[1]) + root * float(direction_xyz[1])
        point_z = float(origin_xyz[2]) + root * float(direction_xyz[2])
        radius = math.sqrt(point_x * point_x + point_y * point_y + point_z * point_z)
        if radius <= (_TIME_ATOL + _TIME_RTOL):
            continue
        if point_z * cos_polar < -(_TIME_ATOL + _TIME_RTOL):
            continue
        roots_out[n_valid] = root
        n_valid += 1
    return int(n_valid)


@njit(cache=True)
def _azimuth_plane_roots(
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    azimuth: float,
    roots_out: np.ndarray,  # out
) -> int:
    """Write the ray parameter where one line meets one constant-azimuth plane."""
    azimuth_value = float(azimuth)
    normal_x = math.sin(azimuth_value)
    normal_y = -math.cos(azimuth_value)
    numerator = normal_x * float(origin_xyz[0]) + normal_y * float(origin_xyz[1])
    denominator = normal_x * float(direction_xyz[0]) + normal_y * float(direction_xyz[1])
    tol = _TIME_ATOL + _TIME_RTOL * max(abs(numerator), abs(denominator), 1.0)
    if abs(denominator) <= tol:
        return 0
    roots_out[0] = -numerator / denominator
    return 1


@njit(cache=True)
def _rpa_coordinate_roots(
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    axis: int,
    value: float,
    roots_out: np.ndarray,  # out
) -> int:
    """Write ray times where one RPA coordinate equals one value."""
    if int(axis) == 0:
        return _sphere_roots(origin_xyz, direction_xyz, float(value), roots_out)
    if int(axis) == 1:
        return _polar_roots(origin_xyz, direction_xyz, float(value), roots_out)
    if int(axis) == 2:
        return _azimuth_plane_roots(origin_xyz, direction_xyz, float(value), roots_out)
    raise ValueError("RPA axis id must be 0, 1, or 2.")


@njit(cache=True)
def _face_roots(
    cell_bounds: np.ndarray,
    cell_id: int,
    face_id: int,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    roots_out: np.ndarray,  # out
) -> int:
    """Write ray times where one RPA trajectory meets one cell face."""
    bounds = cell_bounds[int(cell_id)]
    axis = int(_FACE_AXIS[int(face_id)])
    side = int(_FACE_SIDE[int(face_id)])
    face_value = float(bounds[axis, START]) + float(side) * float(bounds[axis, WIDTH])
    return _rpa_coordinate_roots(origin_xyz, direction_xyz, axis, face_value, roots_out)


@njit(cache=True)
def _next_time_after(roots: np.ndarray, n_root: int, time: float) -> float:
    """Return the nearest root strictly after one ray time, or infinity when absent."""
    best = math.inf
    time_value = float(time)
    for root_pos in range(n_root):
        root_value = float(roots[root_pos])
        if root_value <= time_value or _times_close(root_value, time_value):
            continue
        if root_value < best:
            best = root_value
    return best


@njit(cache=True)
def _axis_interval_unwrapped(axis: int, start: float, width: float, reference: float) -> tuple[float, float]:
    """Return one coordinate interval unwrapped around one reference value."""
    start_value = float(start)
    width_value = float(width)
    if int(axis) != 2:
        return start_value, start_value + width_value
    if width_value >= (_TWO_PI - (_TIME_ATOL + _TIME_RTOL * max(abs(width_value), 1.0))):
        return float(reference), float(reference) + width_value
    start_unwrapped = float(reference) + ((start_value - float(reference) + _PI) % _TWO_PI - _PI)
    stop_unwrapped = start_unwrapped + width_value
    tol = _TIME_ATOL + _TIME_RTOL * max(abs(float(reference)), abs(start_unwrapped), abs(stop_unwrapped), 1.0)
    if float(reference) < start_unwrapped - tol:
        start_unwrapped -= _TWO_PI
        stop_unwrapped -= _TWO_PI
    elif float(reference) > stop_unwrapped + tol:
        start_unwrapped += _TWO_PI
        stop_unwrapped += _TWO_PI
    return float(start_unwrapped), float(stop_unwrapped)


@njit(cache=True)
def _axis_value_unwrapped(axis: int, value: float, reference: float) -> float:
    """Return one coordinate value unwrapped around one reference."""
    if int(axis) != 2:
        return float(value)
    return float(reference) + ((float(value) - float(reference) + _PI) % _TWO_PI - _PI)


@njit(cache=True)
def _contains_box(query_rpa: np.ndarray, bounds: np.ndarray, domain_bounds: np.ndarray, atol: float = 0.0) -> int:
    """Return whether one RPA query lies in one exact half-open cell box plus one miss code."""
    for axis in range(3):
        value = float(query_rpa[axis])
        start = float(bounds[axis, START])
        width = float(bounds[axis, WIDTH])
        stop = start + width
        domain_start = float(domain_bounds[axis, START])
        domain_width = float(domain_bounds[axis, WIDTH])
        if int(axis) == 2:
            start_u, stop_u = _axis_interval_unwrapped(axis, start, width, value)
            value_u = _axis_value_unwrapped(axis, value, value)
            domain_start_u, _domain_stop_u = _axis_interval_unwrapped(axis, domain_start, domain_width, value)
            scale = max(abs(start_u), abs(stop_u), abs(value_u), 1.0)
            tol = max(8.0 * np.finfo(np.float64).eps * scale, float(atol))
            if value_u < start_u - tol or value_u > stop_u + tol:
                return TRACE_START_OUTSIDE_DOMAIN_AZIMUTH_INTERVAL
            if not _times_close(start_u, domain_start_u) and abs(value_u - start_u) <= tol:
                return TRACE_START_OUTSIDE_DOMAIN_AZIMUTH_OPEN_LOWER
            continue
        axis_id = int(axis)
        scale = max(abs(start), abs(stop), abs(value), 1.0)
        tol = max(8.0 * np.finfo(np.float64).eps * scale, float(atol))
        if value < start - tol or value > stop + tol:
            if axis_id == 0:
                return TRACE_START_OUTSIDE_DOMAIN_RADIUS_INTERVAL
            return TRACE_START_OUTSIDE_DOMAIN_POLAR_INTERVAL
        if not _times_close(start, domain_start) and abs(value - start) <= tol:
            if axis_id == 0:
                return TRACE_START_OUTSIDE_DOMAIN_RADIUS_OPEN_LOWER
            return TRACE_START_OUTSIDE_DOMAIN_POLAR_OPEN_LOWER
    return CONTAINS_BOX_OK


@njit(cache=True)
def _axis_transfer_time(
    cell_bounds: np.ndarray,
    cell_id: int,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    t_current: float,
) -> float:
    """Return one future pole-axis crossing time for a polar-cap cell, or infinity when absent."""
    bounds = cell_bounds[int(cell_id)]
    polar_start = float(bounds[1, START])
    polar_stop = float(bounds[1, START] + bounds[1, WIDTH])
    touches_north_pole = _times_close(polar_start, 0.0)
    touches_south_pole = _times_close(polar_stop, _PI)
    if not touches_north_pole and not touches_south_pole:
        return math.inf

    x0 = float(origin_xyz[0])
    y0 = float(origin_xyz[1])
    z0 = float(origin_xyz[2])
    dx = float(direction_xyz[0])
    dy = float(direction_xyz[1])
    dz = float(direction_xyz[2])
    tol_x = _TIME_ATOL + _TIME_RTOL * max(abs(x0), abs(dx), 1.0)
    tol_y = _TIME_ATOL + _TIME_RTOL * max(abs(y0), abs(dy), 1.0)
    if abs(dx) <= tol_x and abs(dy) <= tol_y:
        return math.inf
    if abs(dx) <= tol_x:
        if abs(x0) > tol_x or abs(dy) <= tol_y:
            return math.inf
        axis_time = -y0 / dy
    elif abs(dy) <= tol_y:
        if abs(y0) > tol_y:
            return math.inf
        axis_time = -x0 / dx
    else:
        t_x = -x0 / dx
        t_y = -y0 / dy
        if not _times_close(float(t_x), float(t_y)):
            return math.inf
        axis_time = 0.5 * (float(t_x) + float(t_y))
    if float(axis_time) <= float(t_current) or _times_close(float(axis_time), float(t_current)):
        return math.inf

    z_axis = z0 + float(axis_time) * dz
    tol_z = _TIME_ATOL + _TIME_RTOL * max(abs(z0), abs(dz), abs(z_axis), 1.0)
    if touches_north_pole and z_axis <= tol_z:
        return math.inf
    if touches_south_pole and z_axis >= -tol_z:
        return math.inf
    return float(axis_time)


@njit(cache=True)
def _axis_transfer_destination_cell(
    cell_depth: np.ndarray,
    cell_child: np.ndarray,
    cell_bounds: np.ndarray,
    crossing_xyz: np.ndarray,
    direction_xyz: np.ndarray,
) -> int:
    """Return the analytic post-pole destination leaf at one axis-crossing time."""
    x_crossing = float(crossing_xyz[0])
    y_crossing = float(crossing_xyz[1])
    z_crossing = float(crossing_xyz[2])
    r_crossing = math.sqrt(
        x_crossing * x_crossing
        + y_crossing * y_crossing
        + z_crossing * z_crossing
    )
    if not (_times_close(x_crossing, 0.0) and _times_close(y_crossing, 0.0)):
        return -1
    if _times_close(z_crossing, 0.0):
        return -1

    dx = float(direction_xyz[0])
    dy = float(direction_xyz[1])
    dz = float(direction_xyz[2])
    if _times_close(dx, 0.0) and _times_close(dy, 0.0):
        return -1

    north_pole = z_crossing > 0.0
    azimuth_after = math.atan2(dy, dx) % _TWO_PI
    radial_tendency = z_crossing * dz
    radial_outward = radial_tendency > 0.0 or _times_close(radial_tendency, 0.0)
    matched_cell = -1
    matched_p_width = math.inf
    for cell_id in range(int(cell_depth.shape[0])):
        if int(cell_depth[cell_id]) < 0:
            continue
        is_leaf = True
        for child_ord in range(8):
            if int(cell_child[cell_id, child_ord]) >= 0:
                is_leaf = False
                break
        if not is_leaf:
            continue
        bounds = cell_bounds[int(cell_id)]
        r_start = float(bounds[0, START])
        r_stop = r_start + float(bounds[0, WIDTH])
        if radial_outward:
            if r_crossing < r_start and not _times_close(r_crossing, r_start):
                continue
            if r_crossing > r_stop or _times_close(r_crossing, r_stop):
                continue
        else:
            if r_crossing < r_start or _times_close(r_crossing, r_start):
                continue
            if r_crossing > r_stop and not _times_close(r_crossing, r_stop):
                continue

        p_start = float(bounds[1, START])
        p_stop = p_start + float(bounds[1, WIDTH])
        if north_pole:
            if not _times_close(p_start, 0.0):
                continue
        else:
            if not _times_close(p_stop, _PI):
                continue

        a_start = float(bounds[2, START])
        a_width = float(bounds[2, WIDTH])
        a_start_u, a_stop_u = _axis_interval_unwrapped(2, a_start, a_width, azimuth_after)
        a_value_u = _axis_value_unwrapped(2, azimuth_after, azimuth_after)
        tol = _TIME_ATOL + _TIME_RTOL * max(abs(a_start_u), abs(a_stop_u), abs(a_value_u), 1.0)
        if a_value_u < a_start_u - tol:
            continue
        if a_value_u > a_stop_u + tol or _times_close(a_value_u, a_stop_u):
            continue

        p_width = float(bounds[1, WIDTH])
        if matched_cell < 0 or p_width < matched_p_width:
            matched_cell = int(cell_id)
            matched_p_width = p_width
            continue
        if not _times_close(p_width, matched_p_width) and p_width > matched_p_width:
            continue
        return -2
    return int(matched_cell)


@njit(cache=True)
def _coordinate_velocity_sign(point_xyz: np.ndarray, direction_xyz: np.ndarray, axis: int) -> int:
    """Return the local sign of one RPA coordinate velocity at one point."""
    x = float(point_xyz[0])
    y = float(point_xyz[1])
    z = float(point_xyz[2])
    dx = float(direction_xyz[0])
    dy = float(direction_xyz[1])
    dz = float(direction_xyz[2])
    if int(axis) == 0:
        numerator = x * dx + y * dy + z * dz
        scale = max(abs(x * dx), abs(y * dy), abs(z * dz), 1.0)
    elif int(axis) == 1:
        rho_sq = x * x + y * y
        if _times_close(rho_sq, 0.0):
            return 0
        xy_dot = x * dx + y * dy
        numerator = z * xy_dot - rho_sq * dz
        scale = max(abs(z * xy_dot), abs(rho_sq * dz), abs(rho_sq), 1.0)
    elif int(axis) == 2:
        rho_sq = x * x + y * y
        if _times_close(rho_sq, 0.0):
            return 0
        numerator = x * dy - y * dx
        scale = max(abs(x * dy), abs(y * dx), abs(rho_sq), 1.0)
    else:
        raise ValueError("RPA axis id must be 0, 1, or 2.")
    tol = _TIME_ATOL + _TIME_RTOL * scale
    if numerator > tol:
        return 1
    if numerator < -tol:
        return -1
    return 0


@njit(cache=True)
def find_domain_interval(
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    domain_bounds: np.ndarray,
) -> tuple[bool, float, float]:
    """Return whether one ray intersects the spherical domain plus the clipped parameter interval."""
    r_max = float(domain_bounds[0, START] + domain_bounds[0, WIDTH])
    roots = np.empty(2, dtype=np.float64)
    n_root = _sphere_roots(origin_xyz, direction_xyz, r_max, roots)
    if n_root < 2:
        # The line does not cross the outer spherical boundary twice.
        return False, 0.0, 0.0
    t_enter = float(roots[0])
    t_exit = float(roots[n_root - 1])
    if not t_enter < t_exit:
        # Degenerate or reversed roots do not define a positive entry interval.
        return False, 0.0, 0.0
    return True, t_enter, t_exit


@njit(cache=True)
def find_cell(
    query_rpa: np.ndarray,
    cell_child: np.ndarray,
    root_cell_ids: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
) -> int:
    """Resolve one RPA query to its exact half-open RPA owner."""
    if not (
        math.isfinite(float(query_rpa[0]))
        and math.isfinite(float(query_rpa[1]))
        and math.isfinite(float(query_rpa[2]))
    ):
        return TRACE_START_QUERY_NOT_FINITE
    domain_code = _contains_box(query_rpa, domain_bounds, domain_bounds, DOMAIN_CONTAINS_ATOL)
    if domain_code < 0:
        return int(domain_code)
    current = -1
    for root_pos in range(int(root_cell_ids.shape[0])):
        root_cell_id = int(root_cell_ids[root_pos])
        if _contains_box(query_rpa, cell_bounds[root_cell_id], domain_bounds) > 0:
            current = root_cell_id
            break
    if current < 0:
        return TRACE_START_NO_ROOT_OWNER
    while True:
        next_cell_id = -1
        for child_ord in range(8):
            child_id = int(cell_child[current, child_ord])
            if child_id < 0:
                continue
            if _contains_box(query_rpa, cell_bounds[child_id], domain_bounds) > 0:
                next_cell_id = child_id
                break
        if next_cell_id < 0:
            return int(current)
        current = int(next_cell_id)


@njit(cache=True)
def find_exit(
    cell_bounds: np.ndarray,
    cell_id: int,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    t_current: float,
    active_faces: np.ndarray,  # out
    candidate_times: np.ndarray,  # out
    roots: np.ndarray,  # out
    candidate_xyz: np.ndarray,  # out
) -> tuple[float, int, bool]:
    """Return one leaf exit event as `t_exit`, active face ids, and axis-transfer state."""
    candidate_times[:] = math.inf
    n_candidate = 0

    for face_id in range(6):
        n_root = _face_roots(cell_bounds, int(cell_id), int(face_id), origin_xyz, direction_xyz, roots)
        candidate_time = _next_time_after(roots, n_root, float(t_current))
        if math.isinf(candidate_time):
            continue
        _fill_xyz_at_time(origin_xyz, direction_xyz, float(candidate_time), candidate_xyz)
        axis = int(_FACE_AXIS[int(face_id)])
        side = int(_FACE_SIDE[int(face_id)])
        direction_sign = _coordinate_velocity_sign(candidate_xyz, direction_xyz, axis)
        if side == 0:
            if direction_sign >= 0:
                continue
        else:
            if direction_sign <= 0:
                continue
        candidate_times[face_id] = float(candidate_time)
        n_candidate += 1

    axis_transfer_time = _axis_transfer_time(cell_bounds, int(cell_id), origin_xyz, direction_xyz, float(t_current))
    has_axis_transfer = not math.isinf(axis_transfer_time)
    if n_candidate == 0 and not has_axis_transfer:
        return np.nan, -1, False
    t_exit = math.inf
    for face_id in range(6):
        candidate_time = float(candidate_times[face_id])
        if candidate_time < t_exit:
            t_exit = candidate_time
    if has_axis_transfer:
        if float(axis_transfer_time) < t_exit:
            t_exit = float(axis_transfer_time)
    n_active_face = 0
    for face_id in range(6):
        candidate_time = float(candidate_times[face_id])
        if math.isinf(candidate_time):
            continue
        if _times_close(candidate_time, float(t_exit)):
            active_faces[n_active_face] = int(face_id)
            n_active_face += 1
    axis_transfer = has_axis_transfer and _times_close(float(axis_transfer_time), float(t_exit))
    return float(t_exit), int(n_active_face), bool(axis_transfer)


@njit(cache=True)
def _fill_active_face_state(
    active_faces: np.ndarray,
    n_active_face: int,
    current_face_id: int,
    active_face_by_axis: np.ndarray,  # out
    active_face_order: np.ndarray,  # out
) -> int:
    """Fill reusable active-face lookup arrays and return the current face order."""
    active_face_by_axis[:] = -1
    active_face_order[:] = -1
    current_face_order = -1
    for order in range(n_active_face):
        face_id = int(active_faces[order])
        active_face_by_axis[int(_FACE_AXIS[face_id])] = face_id
        active_face_order[face_id] = int(order)
        if face_id == int(current_face_id):
            current_face_order = int(order)
    return int(current_face_order)


@njit(cache=True)
def is_on_face(
    cell_bounds: np.ndarray,
    cell_id: int,
    face_id: int,
    crossing_xyz: np.ndarray,
) -> bool:
    """Return whether one snapped spherical event lies on one cell face."""
    bounds = cell_bounds[int(cell_id)]
    axis = int(_FACE_AXIS[int(face_id)])
    width = float(bounds[axis, WIDTH])
    start_u, stop_u = _axis_interval_unwrapped(axis, float(bounds[axis, START]), width, float(crossing_xyz[axis]))
    value_u = _axis_value_unwrapped(axis, float(crossing_xyz[axis]), float(crossing_xyz[axis]))
    face_value = start_u if int(_FACE_SIDE[int(face_id)]) == 0 else stop_u
    scale = max(abs(face_value), abs(value_u), 1.0)
    if abs(value_u - face_value) > (_TIME_ATOL + _TIME_RTOL * scale):
        return False
    for tangential_axis in _FACE_TANGENTIAL_AXES[int(face_id)]:
        tangential_axis = int(tangential_axis)
        tangential_start = float(bounds[tangential_axis, START])
        tangential_width = float(bounds[tangential_axis, WIDTH])
        tangential_start_u, tangential_stop_u = _axis_interval_unwrapped(
            tangential_axis,
            tangential_start,
            tangential_width,
            float(crossing_xyz[tangential_axis]),
        )
        tangential_value_u = _axis_value_unwrapped(
            tangential_axis,
            float(crossing_xyz[tangential_axis]),
            float(crossing_xyz[tangential_axis]),
        )
        scale = max(abs(tangential_start_u), abs(tangential_stop_u), abs(tangential_value_u), 1.0)
        tol = _TIME_ATOL + _TIME_RTOL * scale
        if tangential_value_u < tangential_start_u - tol or tangential_value_u > tangential_stop_u + tol:
            return False
    return True


@njit(cache=True)
def find_subface(
    cell_neighbor: np.ndarray,
    domain_bounds: np.ndarray,
    cell_bounds: np.ndarray,
    cell_id: int,
    face_id: int,
    active_face_by_axis: np.ndarray,
    active_face_order: np.ndarray,
    current_face_order: int,
    crossing_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    crossing_rpa: np.ndarray,
) -> int:
    """Return the destination-side owning face patch for one spherical face crossing."""
    row = cell_neighbor[int(cell_id), int(face_id)]
    first_neighbor = -1
    first_slot = -1
    has_multiple_neighbors = False
    for subface_id in range(4):
        neighbor_id = int(row[subface_id])
        if neighbor_id < 0:
            continue
        if first_neighbor < 0:
            first_neighbor = int(neighbor_id)
            first_slot = int(subface_id)
            continue
        if int(neighbor_id) != int(first_neighbor):
            has_multiple_neighbors = True
            break
    if first_neighbor < 0:
        return 0
    if not has_multiple_neighbors:
        return int(first_slot)

    current_bounds = cell_bounds[int(cell_id)]
    matched_subface = -1
    matched_neighbor = -1
    tangential_axes = _FACE_TANGENTIAL_AXES[int(face_id)]
    for subface_id in range(4):
        neighbor_id = int(cell_neighbor[int(cell_id), int(face_id), subface_id])
        if neighbor_id < 0:
            continue
        neighbor_bounds = cell_bounds[int(neighbor_id)]
        contains = True
        for tangential_axis in tangential_axes:
            axis = int(tangential_axis)
            value = float(crossing_rpa[axis])
            current_start_u, current_stop_u = _axis_interval_unwrapped(
                axis,
                float(current_bounds[axis, START]),
                float(current_bounds[axis, WIDTH]),
                value,
            )
            neighbor_start_u, neighbor_stop_u = _axis_interval_unwrapped(
                axis,
                float(neighbor_bounds[axis, START]),
                float(neighbor_bounds[axis, WIDTH]),
                value,
            )
            domain_start_u, _domain_stop_u = _axis_interval_unwrapped(
                axis,
                float(domain_bounds[axis, START]),
                float(domain_bounds[axis, WIDTH]),
                value,
            )
            value_u = _axis_value_unwrapped(axis, value, value)
            scale = max(
                abs(current_start_u),
                abs(current_stop_u),
                abs(neighbor_start_u),
                abs(neighbor_stop_u),
                abs(value_u),
                1.0,
            )
            tol = _TIME_ATOL + _TIME_RTOL * scale
            active_face_id = int(active_face_by_axis[axis])
            direction_sign = _coordinate_velocity_sign(crossing_xyz, direction_xyz, axis)
            implicit_active_side = -1
            if abs(value_u - current_start_u) <= tol and direction_sign < 0:
                implicit_active_side = 0
            elif abs(value_u - current_stop_u) <= tol and direction_sign > 0:
                implicit_active_side = 1
            if active_face_id >= 0 or implicit_active_side >= 0:
                if active_face_id >= 0:
                    active_side = int(_FACE_SIDE[int(active_face_id)])
                    if int(active_face_order[int(active_face_id)]) < int(current_face_order):
                        active_side = 1 - active_side
                else:
                    active_side = int(implicit_active_side)
                contains_current_interval = (
                    neighbor_start_u <= current_start_u + tol
                    and current_stop_u <= neighbor_stop_u + tol
                )
                if not contains_current_interval:
                    if active_side == 0:
                        if abs(neighbor_start_u - current_start_u) > tol:
                            contains = False
                            break
                    else:
                        if abs(neighbor_stop_u - current_stop_u) > tol:
                            contains = False
                            break
                if value_u < current_start_u - tol or value_u > current_stop_u + tol:
                    contains = False
                    break
                continue

            if direction_sign > 0:
                if value_u < neighbor_start_u - tol or value_u > neighbor_stop_u + tol:
                    contains = False
                    break
                if abs(value_u - neighbor_stop_u) <= tol:
                    contains = False
                    break
                continue
            if direction_sign < 0:
                if value_u < neighbor_start_u - tol or value_u > neighbor_stop_u + tol:
                    contains = False
                    break
                if abs(value_u - neighbor_start_u) <= tol:
                    contains = False
                    break
                continue

            if abs(neighbor_start_u - domain_start_u) <= tol:
                if value_u < neighbor_start_u - tol or value_u > neighbor_stop_u + tol:
                    contains = False
                    break
            else:
                if value_u < neighbor_start_u - tol or value_u > neighbor_stop_u + tol:
                    contains = False
                    break
                if abs(value_u - neighbor_start_u) <= tol:
                    contains = False
                    break
        if not contains:
            continue
        if matched_neighbor < 0:
            matched_subface = int(subface_id)
            matched_neighbor = int(neighbor_id)
            continue
        if int(neighbor_id) != int(matched_neighbor):
            return -2
    if matched_subface < 0:
        return -1
    return int(matched_subface)


@njit(cache=True)
def _write_crossing(
    cell_bounds: np.ndarray,
    cell_id: int,
    active_faces: np.ndarray,
    n_active_face: int,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    t_event: float,
    crossing_rpa: np.ndarray,  # out
) -> None:
    """Fill scratch coordinates derived from one crossing time."""
    _fill_rpa_at_time(origin_xyz, direction_xyz, float(t_event), crossing_rpa)
    bounds = cell_bounds[int(cell_id)]
    for face_pos in range(n_active_face):
        face_id = int(active_faces[face_pos])
        axis = int(_FACE_AXIS[int(face_id)])
        side = int(_FACE_SIDE[int(face_id)])
        if side == 0:
            crossing_rpa[axis] = float(bounds[axis, START])
        else:
            crossing_rpa[axis] = float(bounds[axis, START] + bounds[axis, WIDTH])


@njit(cache=True)
def walk_faces(
    cell_neighbor: np.ndarray,
    domain_bounds: np.ndarray,
    cell_bounds: np.ndarray,
    start_cell_id: int,
    active_faces: np.ndarray,
    n_active_face: int,
    direction_xyz: np.ndarray,
    origin_xyz: np.ndarray,
    t_event: float,
    crossing_xyz: np.ndarray,  # out
    crossing_rpa: np.ndarray,  # out
    path: np.ndarray,  # out
    active_face_by_axis: np.ndarray,  # out
    active_face_order: np.ndarray,  # out
) -> int:
    """Walk one time-defined spherical crossing through the face/subface neighbor graph."""
    _fill_xyz_at_time(origin_xyz, direction_xyz, float(t_event), crossing_xyz)
    _write_crossing(
        cell_bounds,
        start_cell_id,
        active_faces,
        n_active_face,
        origin_xyz,
        direction_xyz,
        t_event,
        crossing_rpa,
    )
    current_cell = int(start_cell_id)
    path_count = 0
    for face_pos in range(n_active_face):
        face_id = int(active_faces[face_pos])
        if current_cell < 0:
            break
        if not is_on_face(cell_bounds, current_cell, face_id, crossing_rpa):
            continue
        current_face_order = _fill_active_face_state(
            active_faces,
            n_active_face,
            face_id,
            active_face_by_axis,
            active_face_order,
        )
        subface_id = find_subface(
            cell_neighbor,
            domain_bounds,
            cell_bounds,
            current_cell,
            face_id,
            active_face_by_axis,
            active_face_order,
            current_face_order,
            crossing_xyz,
            direction_xyz,
            crossing_rpa,
        )
        if subface_id < 0:
            return -1
        current_cell = int(cell_neighbor[current_cell, face_id, subface_id])
        path[path_count] = current_cell
        path_count += 1
    return int(path_count)


@njit(cache=True)
def _start_active_faces(
    cell_bounds: np.ndarray,
    cell_id: int,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    t_event: float,
    start_xyz: np.ndarray,
    active_faces_out: np.ndarray,  # out
    roots: np.ndarray,  # out
) -> int:
    """Return faces whose crossing time is snapped to the start time."""
    n_active_face = 0
    for face_id in range(6):
        n_root = _face_roots(cell_bounds, int(cell_id), int(face_id), origin_xyz, direction_xyz, roots)
        face_at_start = False
        for root_pos in range(n_root):
            if _times_close(float(roots[root_pos]), float(t_event)):
                face_at_start = True
                break
        if not face_at_start:
            continue
        axis = int(_FACE_AXIS[int(face_id)])
        side = int(_FACE_SIDE[int(face_id)])
        direction_sign = _coordinate_velocity_sign(start_xyz, direction_xyz, axis)
        if side == 0:
            if direction_sign >= 0:
                continue
        else:
            if direction_sign <= 0:
                continue
        active_faces_out[n_active_face] = int(face_id)
        n_active_face += 1
    return int(n_active_face)


@njit(cache=True)
def _start_cell_owner(
    root_cell_ids: np.ndarray,
    cell_child: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
    cell_neighbor: np.ndarray,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    t_event: float,
    active_faces: np.ndarray,  # out
    path: np.ndarray,  # out
    active_face_by_axis: np.ndarray,  # out
    active_face_order: np.ndarray,  # out
    crossing_xyz: np.ndarray,  # out
    crossing_rpa: np.ndarray,  # out
    roots: np.ndarray,  # out
) -> int:
    """Return the leaf that owns the immediate open interval after one start point."""
    start_rpa = np.empty(3, dtype=np.float64)
    _fill_rpa_at_time(origin_xyz, direction_xyz, float(t_event), start_rpa)
    # Domain-entry roots can land a few ulps outside the closed radial/polar bounds.
    # Snap those nonperiodic axes back onto the exact domain face before the half-open lookup.
    for axis in range(2):
        domain_start = float(domain_bounds[axis, START])
        domain_stop = domain_start + float(domain_bounds[axis, WIDTH])
        value = float(start_rpa[axis])
        scale = max(abs(value), abs(domain_start), abs(domain_stop), 1.0)
        tol = max(8.0 * np.finfo(np.float64).eps * scale, DOMAIN_CONTAINS_ATOL)
        if abs(value - domain_start) <= tol:
            start_rpa[axis] = domain_start
        elif abs(value - domain_stop) <= tol:
            start_rpa[axis] = domain_stop
    current_cell = find_cell(start_rpa, cell_child, root_cell_ids, cell_bounds, domain_bounds)
    if current_cell < 0:
        return int(current_cell)
    _fill_xyz_at_time(origin_xyz, direction_xyz, float(t_event), crossing_xyz)
    n_active_face = _start_active_faces(
        cell_bounds,
        current_cell,
        origin_xyz,
        direction_xyz,
        float(t_event),
        crossing_xyz,
        active_faces,
        roots,
    )
    if n_active_face == 0:
        return int(current_cell)
    path_count = walk_faces(
        cell_neighbor,
        domain_bounds,
        cell_bounds,
        current_cell,
        active_faces,
        n_active_face,
        direction_xyz,
        origin_xyz,
        float(t_event),
        crossing_xyz,
        crossing_rpa,
        path,
        active_face_by_axis,
        active_face_order,
    )
    if path_count <= 0:
        return TRACE_START_WALK_FAILED
    return int(path[path_count - 1])


@njit(cache=True)
def _trace_ray(
    root_cell_ids: np.ndarray,
    cell_child: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
    cell_neighbor: np.ndarray,
    cell_depth: np.ndarray,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    t_min: float,
    t_max: float,
    cell_ids_out: np.ndarray,  # out
    times_out: np.ndarray,  # out
) -> tuple[int, int]:
    """Trace one spherical ray into fixed scratch buffers."""
    max_cells = int(cell_ids_out.shape[0])
    has_interval, domain_enter, domain_exit = find_domain_interval(origin_xyz, direction_xyz, domain_bounds)
    if not has_interval:
        return 0, 0
    start_t = max(float(t_min), float(domain_enter))
    stop_t = min(float(t_max), float(domain_exit))
    if not (start_t < stop_t):
        return 0, 0

    crossing_xyz = np.empty(3, dtype=np.float64)
    crossing_rpa = np.empty(3, dtype=np.float64)
    active_faces = np.empty(3, dtype=np.int64)
    path = np.empty(3, dtype=np.int64)
    active_face_by_axis = np.empty(3, dtype=np.int64)
    active_face_order = np.empty(6, dtype=np.int64)
    candidate_times = np.empty(6, dtype=np.float64)
    roots = np.empty(2, dtype=np.float64)
    candidate_xyz = np.empty(3, dtype=np.float64)

    current_cell = _start_cell_owner(
        root_cell_ids,
        cell_child,
        cell_bounds,
        domain_bounds,
        cell_neighbor,
        origin_xyz,
        direction_xyz,
        float(start_t),
        active_faces,
        path,
        active_face_by_axis,
        active_face_order,
        crossing_xyz,
        crossing_rpa,
        roots,
    )
    if current_cell < 0:
        return current_cell, current_cell

    n_cell = 0
    n_time = 1
    times_out[0] = float(start_t)
    t_current = float(start_t)
    while current_cell >= 0 and t_current < stop_t and not _times_close(float(t_current), float(stop_t)):
        t_exit, n_active_face, axis_transfer = find_exit(
            cell_bounds,
            current_cell,
            origin_xyz,
            direction_xyz,
            t_current,
            active_faces,
            candidate_times,
            roots,
            candidate_xyz,
        )
        if n_active_face < 0:
            return TRACE_INVALID_CROSSING, TRACE_INVALID_CROSSING
        if t_exit > stop_t and not _times_close(float(t_exit), float(stop_t)):
            if n_cell >= max_cells:
                return TRACE_BUFFER_OVERFLOW, TRACE_BUFFER_OVERFLOW
            cell_ids_out[n_cell] = int(current_cell)
            times_out[n_time] = float(stop_t)
            n_cell += 1
            n_time += 1
            break
        if axis_transfer:
            _fill_xyz_at_time(origin_xyz, direction_xyz, float(t_exit), crossing_xyz)
            next_cell = _axis_transfer_destination_cell(
                cell_depth,
                cell_child,
                cell_bounds,
                crossing_xyz,
                direction_xyz,
            )
            if next_cell == TRACE_START_CELL_NOT_FOUND:
                return TRACE_INVALID_CROSSING, TRACE_INVALID_CROSSING
            if next_cell != int(current_cell) or next_cell < 0:
                if n_cell >= max_cells:
                    return TRACE_BUFFER_OVERFLOW, TRACE_BUFFER_OVERFLOW
                cell_ids_out[n_cell] = int(current_cell)
                times_out[n_time] = float(t_exit)
                n_cell += 1
                n_time += 1
            current_cell = int(next_cell)
            t_current = float(t_exit)
            continue

        if n_cell >= max_cells:
            return TRACE_BUFFER_OVERFLOW, TRACE_BUFFER_OVERFLOW
        cell_ids_out[n_cell] = int(current_cell)
        times_out[n_time] = float(t_exit)
        n_cell += 1
        n_time += 1
        path_count = walk_faces(
            cell_neighbor,
            domain_bounds,
            cell_bounds,
            current_cell,
            active_faces,
            n_active_face,
            direction_xyz,
            origin_xyz,
            float(t_exit),
            crossing_xyz,
            crossing_rpa,
            path,
            active_face_by_axis,
            active_face_order,
        )
        if path_count < 0:
            return TRACE_INVALID_CROSSING, TRACE_INVALID_CROSSING
        for path_pos in range(path_count - 1):
            intermediate_cell = int(path[path_pos])
            if intermediate_cell < 0:
                break
            if n_cell >= max_cells:
                return TRACE_BUFFER_OVERFLOW, TRACE_BUFFER_OVERFLOW
            cell_ids_out[n_cell] = int(intermediate_cell)
            times_out[n_time] = float(t_exit)
            n_cell += 1
            n_time += 1
        if path_count > 0:
            current_cell = int(path[path_count - 1])
        else:
            current_cell = -1
        t_current = float(t_exit)
    return int(n_cell), int(n_time)


@njit(cache=True, parallel=True)
def trace_rays(
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
    cell_counts: np.ndarray,  # out
    time_counts: np.ndarray,  # out
    cell_ids_out: np.ndarray,  # out
    times_out: np.ndarray,  # out
) -> None:
    """Trace flat spherical rays into fixed per-ray scratch buffers."""
    n_rays = int(origins.shape[0])
    for ray_id in prange(n_rays):
        n_cell, n_time = _trace_ray(
            root_cell_ids,
            cell_child,
            cell_bounds,
            domain_bounds,
            cell_neighbor,
            cell_depth,
            origins[ray_id],
            directions[ray_id],
            t_min,
            t_max,
            cell_ids_out[ray_id],
            times_out[ray_id],
        )
        cell_counts[ray_id] = int(n_cell)
        time_counts[ray_id] = int(n_time)
