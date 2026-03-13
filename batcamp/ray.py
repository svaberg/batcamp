#!/usr/bin/env python3
"""Ray traversal and ray-based interpolation helpers for octrees."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from numba import njit
import numpy as np

from .cartesian import CartesianLookupKernelState
from .cartesian import _lookup_xyz_cell_id_kernel
from .octree import Octree
from .spherical import SphericalLookupKernelState
from .spherical import _lookup_rpa_cell_id_kernel
from .spherical import _xyz_to_rpa_components

if TYPE_CHECKING:
    from .interpolator import OctreeInterpolator


HEX_TETS_INDEX = np.array(
    (
        (0, 1, 2, 6),
        (0, 2, 3, 6),
        (0, 3, 7, 6),
        (0, 7, 4, 6),
        (0, 4, 5, 6),
        (0, 5, 1, 6),
    ),
    dtype=np.int64,
)
TET_FACES_INDEX = np.array(
    (
        (1, 2, 3),
        (0, 3, 2),
        (0, 1, 3),
        (0, 2, 1),
    ),
    dtype=np.int64,
)

_TRACE_CONTAIN_TOL = 1e-8
_DEFAULT_TRACE_BOUNDARY_TOL = 1e-9
_AXIS_ALIGNED_DIR_TOL = 1e-12
_TET_EPS_ABS = 1e-12
_TET_EPS_REL = 1e-9


@njit(cache=True)
def _contains_xyz_from_state(
    cid: int,
    x: float,
    y: float,
    z: float,
    lookup_state: CartesianLookupKernelState,
    tol: float = _TRACE_CONTAIN_TOL,
) -> bool:
    """Return whether a Cartesian point lies inside one cell."""
    if x < (lookup_state.cell_x_min[cid] - tol) or x > (lookup_state.cell_x_max[cid] + tol):
        return False
    if y < (lookup_state.cell_y_min[cid] - tol) or y > (lookup_state.cell_y_max[cid] + tol):
        return False
    if z < (lookup_state.cell_z_min[cid] - tol) or z > (lookup_state.cell_z_max[cid] + tol):
        return False
    return True


@njit(cache=True)
def _contains_rpa_from_components(
    cid: int,
    r: float,
    polar: float,
    azimuth: float,
    lookup_state: SphericalLookupKernelState,
    tol: float = _TRACE_CONTAIN_TOL,
) -> bool:
    """Return whether a spherical point lies inside one cell."""
    if r < (lookup_state.cell_r_min[cid] - tol) or r > (lookup_state.cell_r_max[cid] + tol):
        return False
    if polar < (lookup_state.cell_theta_min[cid] - tol) or polar > (lookup_state.cell_theta_max[cid] + tol):
        return False
    width = lookup_state.cell_phi_width[cid]
    dphi = (azimuth - lookup_state.cell_phi_start[cid]) % (2.0 * math.pi)
    if width >= ((2.0 * math.pi) - tol):
        return True
    return dphi <= (width + tol)


@njit(cache=True)
def _forward_face_exit_dt(
    coord: float,
    direction_component: float,
    coord_min: float,
    coord_max: float,
    abs_eps: float,
    dir_eps: float,
) -> float:
    """Return forward time to hit one Cartesian face along one axis."""
    if abs(direction_component) <= dir_eps:
        return np.inf
    if direction_component > 0.0:
        dt = (coord_max - coord) / direction_component
    else:
        dt = (coord_min - coord) / direction_component
    if dt > abs_eps:
        return dt
    return 0.0


@njit(cache=True)
def _clip_ray_interval_to_slab(
    origin_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    slab_min_xyz: np.ndarray,
    slab_max_xyz: np.ndarray,
    tol: float,
) -> tuple[bool, float, float]:
    """Clip ray interval `[t_start, t_end]` to one slab box."""
    t_near = t_start
    t_far = t_end
    dir_eps = 1e-15
    for axis in range(3):
        o = origin_xyz[axis]
        d = direction_xyz_unit[axis]
        bmin = slab_min_xyz[axis] - tol
        bmax = slab_max_xyz[axis] + tol
        if abs(d) <= dir_eps:
            if o < bmin or o > bmax:
                return False, t_start, t_start
            continue
        t0 = (bmin - o) / d
        t1 = (bmax - o) / d
        if t0 > t1:
            tmp = t0
            t0 = t1
            t1 = tmp
        if t0 > t_near:
            t_near = t0
        if t1 < t_far:
            t_far = t1
        if t_far <= t_near:
            return False, t_near, t_far
    return True, t_near, t_far


@njit(cache=True)
def _clip_ray_interval_to_sphere(
    origin_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    radius: float,
    tol: float,
) -> tuple[bool, float, float]:
    """Clip ray interval `[t_start, t_end]` to one sphere centered at origin."""
    if radius <= 0.0:
        return False, t_start, t_start

    ox = origin_xyz[0]
    oy = origin_xyz[1]
    oz = origin_xyz[2]
    dx = direction_xyz_unit[0]
    dy = direction_xyz_unit[1]
    dz = direction_xyz_unit[2]

    b = ox * dx + oy * dy + oz * dz
    c = (ox * ox + oy * oy + oz * oz) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return False, t_start, t_start

    s = math.sqrt(max(0.0, disc))
    ta = -b - s
    tb = -b + s
    if ta > tb:
        tmp = ta
        ta = tb
        tb = tmp

    t_near = t_start
    t_far = t_end
    if (ta - tol) > t_near:
        t_near = ta - tol
    if (tb + tol) < t_far:
        t_far = tb + tol
    if t_far <= t_near:
        return False, t_near, t_far
    return True, t_near, t_far


@njit(cache=True)
def _seek_first_cell_xyz(
    origin_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    abs_eps: float,
    lookup_state: CartesianLookupKernelState,
) -> tuple[bool, float, int, float, float, float]:
    """Seek the first `t` where the ray point resolves to a Cartesian cell."""
    t = t_start
    x = origin_xyz[0] + t * direction_xyz_unit[0]
    y = origin_xyz[1] + t * direction_xyz_unit[1]
    z = origin_xyz[2] + t * direction_xyz_unit[2]
    cid = _lookup_xyz_cell_id_kernel(x, y, z, lookup_state, -1)
    if cid >= 0:
        return True, t, cid, x, y, z

    lo = t
    hi = t
    dt = max(abs_eps, 1e-6 * (1.0 + abs(t_end - t_start)))
    cid_hi = -1
    while hi < t_end:
        hi = hi + dt
        if hi > t_end:
            hi = t_end
        xh = origin_xyz[0] + hi * direction_xyz_unit[0]
        yh = origin_xyz[1] + hi * direction_xyz_unit[1]
        zh = origin_xyz[2] + hi * direction_xyz_unit[2]
        cid_hi = _lookup_xyz_cell_id_kernel(xh, yh, zh, lookup_state, -1)
        if cid_hi >= 0:
            break
        if hi >= t_end:
            return False, t_start, -1, x, y, z
        dt = dt * 2.0
    if cid_hi < 0:
        return False, t_start, -1, x, y, z

    lo_out = lo
    hi_in = hi
    cid_in = cid_hi
    for _ in range(48):
        mid = 0.5 * (lo_out + hi_in)
        xm = origin_xyz[0] + mid * direction_xyz_unit[0]
        ym = origin_xyz[1] + mid * direction_xyz_unit[1]
        zm = origin_xyz[2] + mid * direction_xyz_unit[2]
        cid_mid = _lookup_xyz_cell_id_kernel(xm, ym, zm, lookup_state, cid_in)
        if cid_mid >= 0:
            hi_in = mid
            cid_in = cid_mid
        else:
            lo_out = mid
        if (hi_in - lo_out) <= abs_eps:
            break

    t = hi_in
    if t < t_end:
        t = t + abs_eps
        if t > t_end:
            t = t_end
    x = origin_xyz[0] + t * direction_xyz_unit[0]
    y = origin_xyz[1] + t * direction_xyz_unit[1]
    z = origin_xyz[2] + t * direction_xyz_unit[2]
    cid = _lookup_xyz_cell_id_kernel(x, y, z, lookup_state, cid_in)
    if cid < 0:
        t = hi_in
        x = origin_xyz[0] + t * direction_xyz_unit[0]
        y = origin_xyz[1] + t * direction_xyz_unit[1]
        z = origin_xyz[2] + t * direction_xyz_unit[2]
        cid = _lookup_xyz_cell_id_kernel(x, y, z, lookup_state, cid_in)
    if cid < 0:
        return False, t_start, -1, x, y, z
    return True, t, cid, x, y, z


@njit(cache=True)
def _seek_first_cell_rpa(
    origin_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    abs_eps: float,
    lookup_state: SphericalLookupKernelState,
) -> tuple[bool, float, int, float, float, float, float, float, float]:
    """Seek the first `t` where the ray point resolves to a spherical cell."""
    t = t_start
    x = origin_xyz[0] + t * direction_xyz_unit[0]
    y = origin_xyz[1] + t * direction_xyz_unit[1]
    z = origin_xyz[2] + t * direction_xyz_unit[2]
    r, polar, azimuth = _xyz_to_rpa_components(x, y, z)
    cid = _lookup_rpa_cell_id_kernel(r, polar, azimuth, lookup_state, -1)
    if cid >= 0:
        return True, t, cid, x, y, z, r, polar, azimuth

    lo = t
    hi = t
    dt = max(abs_eps, 1e-6 * (1.0 + abs(t_end - t_start)))
    cid_hi = -1
    while hi < t_end:
        hi = hi + dt
        if hi > t_end:
            hi = t_end
        xh = origin_xyz[0] + hi * direction_xyz_unit[0]
        yh = origin_xyz[1] + hi * direction_xyz_unit[1]
        zh = origin_xyz[2] + hi * direction_xyz_unit[2]
        r_hi, polar_hi, azimuth_hi = _xyz_to_rpa_components(xh, yh, zh)
        cid_hi = _lookup_rpa_cell_id_kernel(r_hi, polar_hi, azimuth_hi, lookup_state, -1)
        if cid_hi >= 0:
            break
        if hi >= t_end:
            return False, t_start, -1, x, y, z, r, polar, azimuth
        dt = dt * 2.0
    if cid_hi < 0:
        return False, t_start, -1, x, y, z, r, polar, azimuth

    lo_out = lo
    hi_in = hi
    cid_in = cid_hi
    for _ in range(48):
        mid = 0.5 * (lo_out + hi_in)
        xm = origin_xyz[0] + mid * direction_xyz_unit[0]
        ym = origin_xyz[1] + mid * direction_xyz_unit[1]
        zm = origin_xyz[2] + mid * direction_xyz_unit[2]
        r_mid, polar_mid, azimuth_mid = _xyz_to_rpa_components(xm, ym, zm)
        cid_mid = _lookup_rpa_cell_id_kernel(r_mid, polar_mid, azimuth_mid, lookup_state, cid_in)
        if cid_mid >= 0:
            hi_in = mid
            cid_in = cid_mid
        else:
            lo_out = mid
        if (hi_in - lo_out) <= abs_eps:
            break

    t = hi_in
    if t < t_end:
        t = t + abs_eps
        if t > t_end:
            t = t_end
    x = origin_xyz[0] + t * direction_xyz_unit[0]
    y = origin_xyz[1] + t * direction_xyz_unit[1]
    z = origin_xyz[2] + t * direction_xyz_unit[2]
    r, polar, azimuth = _xyz_to_rpa_components(x, y, z)
    cid = _lookup_rpa_cell_id_kernel(r, polar, azimuth, lookup_state, cid_in)
    if cid < 0:
        t = hi_in
        x = origin_xyz[0] + t * direction_xyz_unit[0]
        y = origin_xyz[1] + t * direction_xyz_unit[1]
        z = origin_xyz[2] + t * direction_xyz_unit[2]
        r, polar, azimuth = _xyz_to_rpa_components(x, y, z)
        cid = _lookup_rpa_cell_id_kernel(r, polar, azimuth, lookup_state, cid_in)
    if cid < 0:
        return False, t_start, -1, x, y, z, r, polar, azimuth
    return True, t, cid, x, y, z, r, polar, azimuth


@njit(cache=True)
def _trace_segments_xyz_kernel(
    origin_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    boundary_tol: float,
    lookup_state: CartesianLookupKernelState,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Trace Cartesian ray segments with exact per-cell boundary walk."""
    cell_ids = np.empty(max_steps, dtype=np.int64)
    enters = np.empty(max_steps, dtype=np.float64)
    exits = np.empty(max_steps, dtype=np.float64)

    if t_end <= t_start or max_steps <= 0:
        return 0, cell_ids, enters, exits

    clipped, t0_clip, t1_clip = _clip_ray_interval_to_slab(
        origin_xyz,
        direction_xyz_unit,
        t_start,
        t_end,
        lookup_state.xyz_min,
        lookup_state.xyz_max,
        boundary_tol,
    )
    if not clipped or t1_clip <= t0_clip:
        return 0, cell_ids, enters, exits

    abs_eps = max(boundary_tol * (1.0 + abs(t1_clip - t0_clip)), 1e-12)
    t = t0_clip
    if t0_clip > t_start:
        t = min(t1_clip, t0_clip + abs_eps)
    if t >= t1_clip:
        return 0, cell_ids, enters, exits

    t_end = t1_clip
    found, t, cid, x, y, z = _seek_first_cell_xyz(
        origin_xyz,
        direction_xyz_unit,
        t,
        t_end,
        abs_eps,
        lookup_state,
    )
    if not found or cid < 0:
        return 0, cell_ids, enters, exits

    d0 = direction_xyz_unit[0]
    d1 = direction_xyz_unit[1]
    d2 = direction_xyz_unit[2]
    dir_eps = 1e-15
    n_seg = 0
    for _ in range(max_steps):
        if t >= (t_end - abs_eps):
            break

        if not _contains_xyz_from_state(cid, x, y, z, lookup_state):
            cid_near = _lookup_xyz_cell_id_kernel(x, y, z, lookup_state, cid)
            if cid_near < 0:
                break
            cid = cid_near

        tx = _forward_face_exit_dt(
            x,
            d0,
            lookup_state.cell_x_min[cid],
            lookup_state.cell_x_max[cid],
            abs_eps,
            dir_eps,
        )
        ty = _forward_face_exit_dt(
            y,
            d1,
            lookup_state.cell_y_min[cid],
            lookup_state.cell_y_max[cid],
            abs_eps,
            dir_eps,
        )
        tz = _forward_face_exit_dt(
            z,
            d2,
            lookup_state.cell_z_min[cid],
            lookup_state.cell_z_max[cid],
            abs_eps,
            dir_eps,
        )

        dt_exit = tx
        if ty < dt_exit:
            dt_exit = ty
        if tz < dt_exit:
            dt_exit = tz
        if not math.isfinite(dt_exit):
            dt_exit = t_end - t
        if dt_exit <= abs_eps:
            dt_exit = abs_eps
        t_exit = t + dt_exit
        if t_exit > t_end:
            t_exit = t_end
        if t_exit < t:
            t_exit = t

        cell_ids[n_seg] = cid
        enters[n_seg] = t
        exits[n_seg] = t_exit
        n_seg += 1
        if n_seg >= max_steps:
            break
        if t_exit >= (t_end - abs_eps):
            break

        t_next = t_exit + abs_eps
        if t_next > t_end:
            t_next = t_end
        if t_next <= t + abs_eps * 0.25:
            t_next = t + abs_eps
            if t_next > t_end:
                t_next = t_end

        x_next = origin_xyz[0] + t_next * d0
        y_next = origin_xyz[1] + t_next * d1
        z_next = origin_xyz[2] + t_next * d2
        cid_next = _lookup_xyz_cell_id_kernel(x_next, y_next, z_next, lookup_state, cid)
        if cid_next < 0:
            break
        if cid_next == cid and t_next < t_end:
            t_next = t_next + 4.0 * abs_eps
            if t_next > t_end:
                t_next = t_end
            x_next = origin_xyz[0] + t_next * d0
            y_next = origin_xyz[1] + t_next * d1
            z_next = origin_xyz[2] + t_next * d2
            cid_next = _lookup_xyz_cell_id_kernel(x_next, y_next, z_next, lookup_state, cid)
            if cid_next < 0:
                break

        t = t_next
        x = x_next
        y = y_next
        z = z_next
        cid = cid_next

    return n_seg, cell_ids, enters, exits


@njit(cache=True)
def _trace_segments_rpa_kernel(
    origin_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    bisect_iters: int,
    boundary_tol: float,
    lookup_state: SphericalLookupKernelState,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Trace straight Cartesian rays on spherical trees using compiled kernels."""
    cell_ids = np.empty(max_steps, dtype=np.int64)
    enters = np.empty(max_steps, dtype=np.float64)
    exits = np.empty(max_steps, dtype=np.float64)

    if t_end <= t_start or max_steps <= 0:
        return 0, cell_ids, enters, exits

    abs_eps = max(boundary_tol * (1.0 + abs(t_end - t_start)), 1e-12)
    found, t, cid, x, y, z, r, polar, azimuth = _seek_first_cell_rpa(
        origin_xyz,
        direction_xyz_unit,
        t_start,
        t_end,
        abs_eps,
        lookup_state,
    )
    if not found or cid < 0:
        return 0, cell_ids, enters, exits

    n_seg = 0
    for _ in range(max_steps):
        if t >= (t_end - abs_eps):
            break

        if not _contains_rpa_from_components(cid, r, polar, azimuth, lookup_state):
            cid_near = _lookup_rpa_cell_id_kernel(r, polar, azimuth, lookup_state, cid)
            if cid_near < 0:
                break
            cid = cid_near

        r_span = lookup_state.cell_r_max[cid] - lookup_state.cell_r_min[cid]
        theta_span = lookup_state.cell_theta_max[cid] - lookup_state.cell_theta_min[cid]
        phi_span = lookup_state.cell_phi_width[cid]
        two_pi = 2.0 * math.pi
        if phi_span > two_pi:
            phi_span = two_pi

        length_scale = lookup_state.cell_r_max[cid]
        if length_scale < 1.0:
            length_scale = 1.0

        dt = r_span
        t_theta = length_scale * theta_span
        if t_theta > dt:
            dt = t_theta
        t_phi = length_scale * phi_span
        if t_phi > dt:
            dt = t_phi
        if dt < 1e-6:
            dt = 1e-6

        t_hi = t + dt
        if t_hi > t_end:
            t_hi = t_end

        x_hi = origin_xyz[0] + t_hi * direction_xyz_unit[0]
        y_hi = origin_xyz[1] + t_hi * direction_xyz_unit[1]
        z_hi = origin_xyz[2] + t_hi * direction_xyz_unit[2]
        r_hi, polar_hi, azimuth_hi = _xyz_to_rpa_components(x_hi, y_hi, z_hi)
        while t_hi < t_end and _contains_rpa_from_components(cid, r_hi, polar_hi, azimuth_hi, lookup_state):
            dt *= 2.0
            t_hi = t + dt
            if t_hi > t_end:
                t_hi = t_end
            x_hi = origin_xyz[0] + t_hi * direction_xyz_unit[0]
            y_hi = origin_xyz[1] + t_hi * direction_xyz_unit[1]
            z_hi = origin_xyz[2] + t_hi * direction_xyz_unit[2]
            r_hi, polar_hi, azimuth_hi = _xyz_to_rpa_components(x_hi, y_hi, z_hi)

        if _contains_rpa_from_components(cid, r_hi, polar_hi, azimuth_hi, lookup_state):
            cell_ids[n_seg] = cid
            enters[n_seg] = t
            exits[n_seg] = t_end
            n_seg += 1
            break

        lo = t
        hi = t_hi
        for _ in range(bisect_iters):
            mid = 0.5 * (lo + hi)
            x_mid = origin_xyz[0] + mid * direction_xyz_unit[0]
            y_mid = origin_xyz[1] + mid * direction_xyz_unit[1]
            z_mid = origin_xyz[2] + mid * direction_xyz_unit[2]
            r_mid, polar_mid, azimuth_mid = _xyz_to_rpa_components(x_mid, y_mid, z_mid)
            if _contains_rpa_from_components(cid, r_mid, polar_mid, azimuth_mid, lookup_state):
                lo = mid
            else:
                hi = mid
            if (hi - lo) <= abs_eps:
                break

        t_exit = lo
        if t_exit < t:
            t_exit = t
        cell_ids[n_seg] = cid
        enters[n_seg] = t
        exits[n_seg] = t_exit
        n_seg += 1
        if n_seg >= max_steps:
            break

        t_next = hi + abs_eps
        if t_next > t_end:
            t_next = t_end
        if t_next <= t + abs_eps * 0.25:
            t_next = t + abs_eps
            if t_next > t_end:
                t_next = t_end

        x_next = origin_xyz[0] + t_next * direction_xyz_unit[0]
        y_next = origin_xyz[1] + t_next * direction_xyz_unit[1]
        z_next = origin_xyz[2] + t_next * direction_xyz_unit[2]
        r_next, polar_next, azimuth_next = _xyz_to_rpa_components(x_next, y_next, z_next)
        cid_next = _lookup_rpa_cell_id_kernel(r_next, polar_next, azimuth_next, lookup_state, cid)
        if cid_next < 0:
            break
        if cid_next == cid and t_next < t_end:
            t_next = t_next + 4.0 * abs_eps
            if t_next > t_end:
                t_next = t_end
            x_next = origin_xyz[0] + t_next * direction_xyz_unit[0]
            y_next = origin_xyz[1] + t_next * direction_xyz_unit[1]
            z_next = origin_xyz[2] + t_next * direction_xyz_unit[2]
            r_next, polar_next, azimuth_next = _xyz_to_rpa_components(x_next, y_next, z_next)
            cid_next = _lookup_rpa_cell_id_kernel(r_next, polar_next, azimuth_next, lookup_state, cid)
            if cid_next < 0:
                break

        t = t_next
        x = x_next
        y = y_next
        z = z_next
        r = r_next
        polar = polar_next
        azimuth = azimuth_next
        cid = cid_next

    return n_seg, cell_ids, enters, exits


def _normalize_direction(direction_xyz: np.ndarray) -> np.ndarray:
    """Normalize one Cartesian ray direction."""
    d = np.asarray(direction_xyz, dtype=float).reshape(3)
    norm = float(np.linalg.norm(d))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("direction_xyz must be finite and non-zero.")
    return d / norm


def _as_xyz(point_xyz: np.ndarray) -> np.ndarray:
    """Coerce one Cartesian point to shape `(3,)` float array."""
    return np.asarray(point_xyz, dtype=float).reshape(3)


def _coerce_origins_xyz(origins_xyz: np.ndarray) -> np.ndarray:
    """Validate and normalize ray origins to shape `(n_rays, 3)`."""
    origins = np.asarray(origins_xyz, dtype=float)
    if origins.ndim == 1:
        if origins.size != 3:
            raise ValueError("origins_xyz must have shape (n_rays, 3) or (3,).")
        origins = origins.reshape(1, 3)
    if origins.ndim != 2 or origins.shape[1] != 3:
        raise ValueError("origins_xyz must have shape (n_rays, 3) or (3,).")
    return origins


def _coerce_ray_interval(t_start: float, t_end: float) -> tuple[float, float]:
    """Validate a ray interval and return `(t0, t1)` as floats."""
    t0 = float(t_start)
    t1 = float(t_end)
    if t1 <= t0:
        raise ValueError("t_end must be greater than t_start.")
    return t0, t1


def _coerce_positive_chunk_size(chunk_size: int) -> int:
    """Validate chunk size for batched ray workflows."""
    chunk = int(chunk_size)
    if chunk <= 0:
        raise ValueError("chunk_size must be positive.")
    return chunk


@dataclass(frozen=True)
class RaySegment:
    """Ray interval `[t_enter, t_exit]` that remains inside one leaf cell."""

    cell_id: int
    t_enter: float
    t_exit: float


@dataclass(frozen=True)
class RayLinearPiece:
    """Linear function f(t) = slope*t + intercept over [t_start, t_end]."""

    t_start: float
    t_end: float
    cell_id: int
    tet_id: int
    slope: np.ndarray
    intercept: np.ndarray


class OctreeRayTracer:
    """Ray tracer operating on an already-built and bound `Octree`."""

    def __init__(self, tree: Octree) -> None:
        """Store the tree used for cell-segment tracing."""
        self.tree = tree
        dmin, dmax = self.tree.domain_bounds(coord="xyz")
        self._domain_xyz_min = np.asarray(dmin, dtype=float).reshape(3)
        self._domain_xyz_max = np.asarray(dmax, dtype=float).reshape(3)
        _r_lo, r_hi = self.tree.domain_bounds(coord="rpa")
        self._domain_r_max = float(np.asarray(r_hi, dtype=float).reshape(3)[0])

    def trace(
        self,
        origin_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
        *,
        max_steps: int = 100000,
        bisect_iters: int = 48,
        boundary_tol: float = _DEFAULT_TRACE_BOUNDARY_TOL,
    ) -> list[RaySegment]:
        """Trace a Cartesian ray into contiguous per-cell segments."""
        o = _as_xyz(origin_xyz)
        d = _normalize_direction(direction_xyz)
        return self.trace_prepared(
            o,
            d,
            float(t_start),
            float(t_end),
            max_steps=max_steps,
            bisect_iters=bisect_iters,
            boundary_tol=boundary_tol,
        )

    def trace_prepared(
        self,
        origin_xyz: np.ndarray,
        direction_xyz_unit: np.ndarray,
        t_start: float,
        t_end: float,
        *,
        max_steps: int = 100000,
        bisect_iters: int = 48,
        boundary_tol: float = _DEFAULT_TRACE_BOUNDARY_TOL,
    ) -> list[RaySegment]:
        """Trace ray segments for pre-normalized inputs using compiled kernels."""
        o = origin_xyz
        d = direction_xyz_unit
        t0 = float(t_start)
        t1 = float(t_end)
        if t1 <= t0:
            return []

        tree_coord = str(self.tree.tree_coord)
        clipped, t0_clip, t1_clip = _clip_ray_interval_to_slab(
            o,
            d,
            t0,
            t1,
            self._domain_xyz_min,
            self._domain_xyz_max,
            float(boundary_tol),
        )
        if not clipped or t1_clip <= t0_clip:
            return []

        if tree_coord == "rpa":
            clipped_sphere, t0_sphere, t1_sphere = _clip_ray_interval_to_sphere(
                o,
                d,
                t0_clip,
                t1_clip,
                float(self._domain_r_max),
                float(boundary_tol),
            )
            if not clipped_sphere or t1_sphere <= t0_sphere:
                return []
            t0_clip = t0_sphere
            t1_clip = t1_sphere

        abs_eps = max(float(boundary_tol) * (1.0 + abs(t1_clip - t0_clip)), 1e-12)
        t0 = t0_clip
        if t0_clip > float(t_start):
            t0 = min(t1_clip, t0_clip + abs_eps)
        t1 = t1_clip
        if t1 <= t0:
            return []

        lookup_state = self.tree.lookup.lookup_state
        if tree_coord == "xyz":
            if not isinstance(lookup_state, CartesianLookupKernelState):
                raise TypeError(
                    "Cartesian ray tracing requires CartesianLookupKernelState; "
                    f"got {type(lookup_state).__name__}."
                )
            n_seg, cids, enters, exits = _trace_segments_xyz_kernel(
                o,
                d,
                t0,
                t1,
                int(max_steps),
                float(boundary_tol),
                lookup_state,
            )
        elif tree_coord == "rpa":
            if not isinstance(lookup_state, SphericalLookupKernelState):
                raise TypeError(
                    "Spherical ray tracing requires SphericalLookupKernelState; "
                    f"got {type(lookup_state).__name__}."
                )
            n_seg, cids, enters, exits = _trace_segments_rpa_kernel(
                o,
                d,
                t0,
                t1,
                int(max_steps),
                int(bisect_iters),
                float(boundary_tol),
                lookup_state,
            )
        else:
            raise ValueError(f"Unsupported tree_coord '{tree_coord}'.")

        return [
            RaySegment(
                cell_id=int(cids[i]),
                t_enter=float(enters[i]),
                t_exit=float(exits[i]),
            )
            for i in range(int(n_seg))
        ]


class OctreeRayInterpolator:
    """Ray sampling and piecewise-linear extraction on `OctreeInterpolator`."""

    def __init__(self, interpolator: "OctreeInterpolator") -> None:
        """Store the interpolator and its bound tree for ray operations."""
        self.interpolator = interpolator
        self.tree = interpolator.tree
        self.ray_tracer = OctreeRayTracer(self.tree)

    @staticmethod
    def point_in_tet(point_xyz: np.ndarray, tet_xyz: np.ndarray, *, tol: float = 1e-10) -> bool:
        """Test whether a point is inside/on one tetrahedron."""
        a = tet_xyz[0]
        mat = np.column_stack((tet_xyz[1] - a, tet_xyz[2] - a, tet_xyz[3] - a))
        try:
            uvw = np.linalg.solve(mat, point_xyz - a)
        except np.linalg.LinAlgError:
            return False
        b0 = 1.0 - float(np.sum(uvw))
        bary = np.array([b0, float(uvw[0]), float(uvw[1]), float(uvw[2])], dtype=float)
        return bool(np.all(bary >= -tol) and np.all(bary <= 1.0 + tol))

    def find_tet_in_hex(self, cell_xyz: np.ndarray, point_xyz: np.ndarray, *, tol: float = 1e-10) -> int | None:
        """Find which tet in the local 6-tet split contains a point."""
        p = np.array(point_xyz, dtype=float)
        for tet_idx, tet in enumerate(HEX_TETS_INDEX):
            tet_xyz = cell_xyz[tet]
            if self.point_in_tet(p, tet_xyz, tol=tol):
                return int(tet_idx)
        return None

    @staticmethod
    def ray_triangle_hit_t(
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        tri_xyz: np.ndarray,
        *,
        t_min: float,
        t_max: float,
        tol: float,
    ) -> float | None:
        """Intersect one triangle with a ray and return hit `t` when valid."""
        p0 = tri_xyz[0]
        p1 = tri_xyz[1]
        p2 = tri_xyz[2]
        e1 = p1 - p0
        e2 = p2 - p0
        h = np.cross(ray_dir, e2)
        a = float(np.dot(e1, h))
        if abs(a) <= tol:
            return None
        f = 1.0 / a
        s = ray_origin - p0
        u = f * float(np.dot(s, h))
        if u < -tol or u > 1.0 + tol:
            return None
        q = np.cross(s, e1)
        v = f * float(np.dot(ray_dir, q))
        if v < -tol or (u + v) > 1.0 + tol:
            return None
        t = f * float(np.dot(e2, q))
        if t <= t_min + tol or t > t_max + tol:
            return None
        return float(t)

    @staticmethod
    def tet_ray_linear_coefficients(
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        tet_xyz: np.ndarray,
        tet_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute linear-in-`t` coefficients on one linear tetrahedron."""
        mat = np.column_stack((tet_xyz, np.ones(4, dtype=float)))
        rhs = tet_values.reshape(4, -1)
        coef = np.linalg.solve(mat, rhs)
        slope = coef[0] * ray_dir[0] + coef[1] * ray_dir[1] + coef[2] * ray_dir[2]
        intercept = (
            coef[0] * ray_origin[0]
            + coef[1] * ray_origin[1]
            + coef[2] * ray_origin[2]
            + coef[3]
        )
        trailing = tet_values.shape[1:]
        return slope.reshape(trailing), intercept.reshape(trailing)

    def sample(
        self,
        origin_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
        n_samples: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[RaySegment]]:
        """Sample interpolated values at uniform `t` points on one ray."""
        if int(n_samples) <= 0:
            raise ValueError("n_samples must be positive.")

        o = _as_xyz(origin_xyz)
        d = _normalize_direction(direction_xyz)
        t_values = np.linspace(float(t_start), float(t_end), int(n_samples))
        query_xyz = o[None, :] + t_values[:, None] * d[None, :]
        values, cell_ids = self.interpolator(
            query_xyz,
            query_coord="xyz",
            return_cell_ids=True,
            log_outside_domain=False,
        )
        segments = self.ray_tracer.trace_prepared(o, d, float(t_start), float(t_end))
        return t_values, values, np.array(cell_ids, dtype=np.int64), segments

    def integrate_field_along_rays(
        self,
        origins_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
        *,
        chunk_size: int = 2048,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Integrate interpolated field(s) along many rays using linear pieces."""
        origins = _coerce_origins_xyz(origins_xyz)
        _coerce_positive_chunk_size(chunk_size)
        d = _normalize_direction(direction_xyz)
        t0, t1 = _coerce_ray_interval(t_start, t_end)

        n_rays = int(origins.shape[0])
        ncomp = int(self.interpolator.n_value_components)
        out = np.full((n_rays, ncomp), np.nan, dtype=float)

        for i, o in enumerate(origins):
            segs = self.ray_tracer.trace_prepared(o, d, t0, t1)
            if not segs:
                if ncomp == 1:
                    out[i, 0] = 0.0
                continue
            pieces = self.linear_pieces(o, d, t0, t1, segments=segs)
            if not pieces:
                if ncomp == 1:
                    out[i, 0] = 0.0
                continue
            col = np.zeros(ncomp, dtype=float)
            for piece in pieces:
                a = float(piece.t_start)
                b = float(piece.t_end)
                if b <= a:
                    continue
                slope = np.asarray(piece.slope, dtype=float).reshape(-1)
                intercept = np.asarray(piece.intercept, dtype=float).reshape(-1)
                col += 0.5 * slope * (b * b - a * a) + intercept * (b - a)
            out[i] = float(scale) * col

        if ncomp == 1:
            return out[:, 0]
        return out

    def adaptive_midpoint_rule(
        self,
        origins_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
        *,
        chunk_size: int = 2048,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build flattened midpoint quadrature data for many rays."""
        origins = _coerce_origins_xyz(origins_xyz)
        _coerce_positive_chunk_size(chunk_size)
        d = _normalize_direction(direction_xyz)
        t0, t1 = _coerce_ray_interval(t_start, t_end)

        n_rays = int(origins.shape[0])
        ray_offsets = np.zeros(n_rays + 1, dtype=np.int64)
        midpoints: list[np.ndarray] = []
        weights: list[float] = []

        k = 0
        for i, o in enumerate(origins):
            ray_offsets[i] = k
            segs = self.ray_tracer.trace_prepared(o, d, t0, t1)
            for seg in segs:
                ta = float(seg.t_enter)
                tb = float(seg.t_exit)
                dt = tb - ta
                if dt <= 0.0:
                    continue
                tm = 0.5 * (ta + tb)
                midpoints.append(o + tm * d)
                weights.append(float(dt))
                k += 1
        ray_offsets[n_rays] = k

        if k == 0:
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0,), dtype=float),
                ray_offsets,
            )
        return np.asarray(midpoints, dtype=float), np.asarray(weights, dtype=float), ray_offsets

    def integrate_midpoint_rule(
        self,
        midpoints_xyz: np.ndarray,
        weights: np.ndarray,
        ray_offsets: np.ndarray,
        *,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Integrate the interpolator field using flattened midpoint quadrature."""
        mids = np.asarray(midpoints_xyz, dtype=float)
        w = np.asarray(weights, dtype=float).reshape(-1)
        offsets = np.asarray(ray_offsets, dtype=np.int64).reshape(-1)

        if mids.ndim != 2 or mids.shape[1] != 3:
            raise ValueError("midpoints_xyz must have shape (n_samples, 3).")
        if mids.shape[0] != w.shape[0]:
            raise ValueError("weights length must equal number of midpoint samples.")
        if offsets.size < 1:
            raise ValueError("ray_offsets must have length >= 1.")
        if offsets[0] != 0:
            raise ValueError("ray_offsets must start at 0.")
        if offsets[-1] != mids.shape[0]:
            raise ValueError("ray_offsets must end at n_samples.")
        if np.any(np.diff(offsets) < 0):
            raise ValueError("ray_offsets must be non-decreasing.")

        n_rays = int(offsets.size - 1)
        ncomp = int(self.interpolator.n_value_components)
        out = np.full((n_rays, ncomp), np.nan, dtype=float)

        if mids.shape[0] == 0:
            if ncomp == 1:
                out[:, 0] = 0.0
                return out[:, 0]
            return out

        values, cell_ids = self.interpolator(
            mids,
            query_coord="xyz",
            return_cell_ids=True,
            log_outside_domain=False,
        )
        value_arr = np.asarray(values, dtype=float)
        if value_arr.ndim == 1:
            value_arr = value_arr.reshape(-1, 1)
        cids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)

        for i in range(n_rays):
            s = int(offsets[i])
            e = int(offsets[i + 1])
            if e <= s:
                if ncomp == 1:
                    out[i, 0] = 0.0
                continue
            valid = cids[s:e] >= 0
            if not np.any(valid):
                if ncomp == 1:
                    out[i, 0] = 0.0
                continue
            seg_vals = value_arr[s:e][valid]
            seg_w = w[s:e][valid].reshape(-1, 1)
            out[i] = float(scale) * np.sum(seg_vals * seg_w, axis=0)

        if ncomp == 1:
            return out[:, 0]
        return out

    def integrate_field_along_rays_midpoint(
        self,
        origins_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
        *,
        chunk_size: int = 2048,
        scale: float = 1.0,
    ) -> np.ndarray:
        """One-shot midpoint integration on many rays."""
        mids, weights, offsets = self.adaptive_midpoint_rule(
            origins_xyz,
            direction_xyz,
            t_start,
            t_end,
            chunk_size=chunk_size,
        )
        return self.integrate_midpoint_rule(mids, weights, offsets, scale=scale)

    def linear_pieces_axis_aligned(
        self,
        origin_xyz: np.ndarray,
        direction_xyz_unit: np.ndarray,
        segments: list[RaySegment],
    ) -> list[RayLinearPiece] | None:
        """Build cellwise linear pieces quickly for axis-aligned Cartesian rays."""
        if str(self.tree.tree_coord) != "xyz":
            return None
        if not segments:
            return []

        abs_d = np.abs(direction_xyz_unit)
        axis = int(np.argmax(abs_d))
        if abs_d[axis] < (1.0 - _AXIS_ALIGNED_DIR_TOL):
            return None
        if abs_d[(axis + 1) % 3] > _AXIS_ALIGNED_DIR_TOL:
            return None
        if abs_d[(axis + 2) % 3] > _AXIS_ALIGNED_DIR_TOL:
            return None

        n_seg = len(segments)
        t_bounds = np.empty(2 * n_seg, dtype=float)
        for i, seg in enumerate(segments):
            t_bounds[2 * i] = float(seg.t_enter)
            t_bounds[2 * i + 1] = float(seg.t_exit)

        query_xyz = origin_xyz[None, :] + t_bounds[:, None] * direction_xyz_unit[None, :]
        values, cell_ids = self.interpolator(
            query_xyz,
            query_coord="xyz",
            return_cell_ids=True,
            log_outside_domain=False,
        )
        cid_arr = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
        if np.any(cid_arr < 0):
            return None

        value_arr = np.asarray(values, dtype=float)
        if value_arr.ndim == 1:
            value_arr = value_arr.reshape(-1, 1)
            scalar_field = True
        else:
            scalar_field = False

        out: list[RayLinearPiece] = []
        for i, seg in enumerate(segments):
            t0 = float(seg.t_enter)
            t1 = float(seg.t_exit)
            dt = t1 - t0
            if dt <= 0.0:
                continue
            v0 = value_arr[2 * i]
            v1 = value_arr[2 * i + 1]
            slope = (v1 - v0) / dt
            intercept = v0 - slope * t0
            if scalar_field:
                slope_out: np.ndarray | float = float(slope[0])
                intercept_out: np.ndarray | float = float(intercept[0])
            else:
                slope_out = slope.copy()
                intercept_out = intercept.copy()
            out.append(
                RayLinearPiece(
                    t_start=t0,
                    t_end=t1,
                    cell_id=int(seg.cell_id),
                    tet_id=-1,
                    slope=slope_out,
                    intercept=intercept_out,
                )
            )
        return out

    def linear_pieces_for_cell_segment(
        self,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        cell_id: int,
        t_enter: float,
        t_exit: float,
        *,
        tol: float = 1e-10,
        max_steps: int = 128,
    ) -> list[RayLinearPiece]:
        """Split one ray/cell interval into piecewise-linear tet intervals."""
        cid = int(cell_id)
        if t_exit <= t_enter:
            return []

        interp = self.interpolator
        corner_ids = interp.corners[cid]
        cell_xyz = interp.lookup.points[corner_ids]
        cell_vals = interp.point_values[corner_ids]
        eps = max(_TET_EPS_ABS, _TET_EPS_REL * (1.0 + abs(t_exit - t_enter)))

        t = float(t_enter)
        t_probe = min(float(t_exit), t + eps)
        p_probe = ray_origin + t_probe * ray_dir
        tet_idx = self.find_tet_in_hex(cell_xyz, p_probe, tol=1e-8)
        if tet_idx is None:
            p_mid = ray_origin + (0.5 * (t_enter + t_exit)) * ray_dir
            tet_idx = self.find_tet_in_hex(cell_xyz, p_mid, tol=1e-7)
        if tet_idx is None:
            return []

        pieces: list[RayLinearPiece] = []
        for _ in range(max_steps):
            if t >= t_exit - eps:
                break

            tet = HEX_TETS_INDEX[tet_idx]
            tet_xyz = cell_xyz[tet]
            tet_vals = cell_vals[tet]

            t_next = float(t_exit)
            for face in TET_FACES_INDEX:
                tri = tet_xyz[face]
                t_hit = self.ray_triangle_hit_t(
                    ray_origin,
                    ray_dir,
                    tri,
                    t_min=t + eps,
                    t_max=t_exit,
                    tol=tol,
                )
                if t_hit is not None and t_hit < t_next:
                    t_next = float(t_hit)

            if t_next <= t + eps * 0.25:
                t_next = min(float(t_exit), t + eps)

            slope, intercept = self.tet_ray_linear_coefficients(
                ray_origin,
                ray_dir,
                tet_xyz,
                tet_vals,
            )
            pieces.append(
                RayLinearPiece(
                    t_start=float(t),
                    t_end=float(t_next),
                    cell_id=cid,
                    tet_id=int(tet_idx),
                    slope=slope,
                    intercept=intercept,
                )
            )

            if t_next >= t_exit - eps:
                break

            t = float(t_next)
            next_tet: int | None = None
            for mult in (1.0, 4.0, 16.0, 64.0):
                t_probe = min(float(t_exit), t + mult * eps)
                p_probe = ray_origin + t_probe * ray_dir
                probe = self.find_tet_in_hex(cell_xyz, p_probe, tol=1e-7)
                if probe is not None:
                    next_tet = int(probe)
                    break
            if next_tet is None:
                break
            tet_idx = next_tet

        return pieces

    def linear_pieces(
        self,
        origin_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
        *,
        segments: list[RaySegment] | None = None,
    ) -> list[RayLinearPiece]:
        """Return stitched piecewise-linear `f(t)=m*t+b` intervals on a ray."""
        o = _as_xyz(origin_xyz)
        d = _normalize_direction(direction_xyz)

        ray_segments = segments
        if ray_segments is None:
            ray_segments = self.ray_tracer.trace_prepared(o, d, float(t_start), float(t_end))
        pieces = self.linear_pieces_axis_aligned(o, d, ray_segments)
        if pieces is None:
            pieces = []
            for seg in ray_segments:
                pieces.extend(
                    self.linear_pieces_for_cell_segment(
                        o,
                        d,
                        int(seg.cell_id),
                        float(seg.t_enter),
                        float(seg.t_exit),
                    )
                )
        if not pieces:
            return pieces

        span = abs(float(t_end) - float(t_start))
        stitch_tol = max(1e-10, 1e-8 * (1.0 + span))
        out: list[RayLinearPiece] = [pieces[0]]
        for piece in pieces[1:]:
            prev = out[-1]
            a = float(piece.t_start)
            b = float(piece.t_end)
            if abs(a - prev.t_end) <= stitch_tol:
                a = float(prev.t_end)
            if b <= a:
                continue
            out.append(
                RayLinearPiece(
                    t_start=a,
                    t_end=b,
                    cell_id=int(piece.cell_id),
                    tet_id=int(piece.tet_id),
                    slope=piece.slope,
                    intercept=piece.intercept,
                )
            )

        first = out[0]
        if abs(first.t_start - float(t_start)) <= stitch_tol:
            out[0] = RayLinearPiece(
                t_start=float(t_start),
                t_end=float(first.t_end),
                cell_id=int(first.cell_id),
                tet_id=int(first.tet_id),
                slope=first.slope,
                intercept=first.intercept,
            )
        last = out[-1]
        if abs(last.t_end - float(t_end)) <= stitch_tol:
            out[-1] = RayLinearPiece(
                t_start=float(last.t_start),
                t_end=float(t_end),
                cell_id=int(last.cell_id),
                tet_id=int(last.tet_id),
                slope=last.slope,
                intercept=last.intercept,
            )

        return out
