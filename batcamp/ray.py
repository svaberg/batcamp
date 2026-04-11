#!/usr/bin/env python3
"""Ray-domain seeding helpers for octrees."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import time

import numpy as np
from numba import njit

from .cartesian import _contains_box_xyz
from .octree import Octree
from .spherical import _contains_box_rpa
from .spherical import xyz_to_rpa_components

__all__ = ["OctreeRayTracer", "RaySegments", "render_midpoint_image", "trace_one_ray_kernel"]

logger = logging.getLogger(__name__)

_FACE_AXIS = np.array([0, 0, 1, 1, 2, 2], dtype=np.int8)
_FACE_SIDE = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
_FACE_TANGENTIAL_AXES = np.array(
    [
        [1, 2],
        [1, 2],
        [0, 2],
        [0, 2],
        [0, 1],
        [0, 1],
    ],
    dtype=np.int8,
)
_XYZ_CORNER_BITS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.int8,
)
_RPA_CORNER_BITS = np.array(
    [
        [0, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [0, 0, 1],
    ],
    dtype=np.int8,
)
_EXIT_TOL = 10 ** -12
_BOUNDARY_SHIFT_FACTOR = 10 ** -6
_TREE_COORD_XYZ = np.int8(0)
_TREE_COORD_RPA = np.int8(1)


def _normalize_ray_arrays(origins: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Return flat finite ray arrays plus the leading broadcast shape."""
    o = np.array(origins, dtype=np.float64, order="C")
    d = np.array(directions, dtype=np.float64, order="C")
    if o.ndim == 0 or o.shape[-1] != 3:
        raise ValueError("origins must have shape (..., 3).")
    if d.shape != o.shape:
        raise ValueError("directions must have the same shape as origins.")
    if not np.all(np.isfinite(o)):
        raise ValueError("origins must contain only finite values.")
    if not np.all(np.isfinite(d)):
        raise ValueError("directions must contain only finite values.")
    d_flat = d.reshape(-1, 3)
    if np.any(np.linalg.norm(d_flat, axis=1) <= 0):
        raise ValueError("directions must be nonzero.")
    shape = (1,) if o.ndim == 1 else o.shape[:-1]
    return o.reshape(-1, 3), d_flat, shape


@njit(cache=True)
def _dot3(a: np.ndarray, b: np.ndarray) -> float:
    """Return one 3D dot product."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True)
def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return one 3D cross product."""
    return np.array(
        (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ),
        dtype=np.float64,
    )


def _build_face_corner_order(corner_bits: np.ndarray) -> np.ndarray:
    """Return one `(6, 4)` face-corner table in `[00, 01, 11, 10]` tangential order."""
    face_corners = np.full((6, 4), -1, dtype=np.int8)
    for face_id in range(6):
        axis = int(_FACE_AXIS[face_id])
        side = int(_FACE_SIDE[face_id])
        tangential_axes = _FACE_TANGENTIAL_AXES[face_id]
        targets = (
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 0),
        )
        for pos, (bit_first, bit_second) in enumerate(targets):
            target = np.full(3, -1, dtype=np.int8)
            target[axis] = side
            target[int(tangential_axes[0])] = bit_first
            target[int(tangential_axes[1])] = bit_second
            matches = np.flatnonzero(np.all(corner_bits == target, axis=1))
            if matches.size != 1:
                raise ValueError(f"Could not resolve one unique face corner for face_id={face_id}, target={target}.")
            face_corners[face_id, pos] = np.int8(matches[0])
    return face_corners


def _combine_seed_interval_rows(
    intervals: list[tuple[float, float, int]],
) -> list[tuple[float, float, int]]:
    """Combine same-leaf local ray intervals into one convex cell interval."""
    by_leaf: dict[int, tuple[float, float]] = {}
    for enter, exit_, leaf_id in intervals:
        if leaf_id in by_leaf:
            old_enter, old_exit = by_leaf[leaf_id]
            by_leaf[leaf_id] = (min(old_enter, enter), max(old_exit, exit_))
        else:
            by_leaf[leaf_id] = (float(enter), float(exit_))
    return [(enter, exit_, leaf_id) for leaf_id, (enter, exit_) in by_leaf.items()]


@njit(cache=True)
def _sphere_interval_kernel(origin_xyz: np.ndarray, direction_xyz: np.ndarray, radius: float) -> tuple[bool, float, float]:
    """Return one line-parameter interval inside one sphere centered at the origin."""
    ox = origin_xyz[0]
    oy = origin_xyz[1]
    oz = origin_xyz[2]
    dx = direction_xyz[0]
    dy = direction_xyz[1]
    dz = direction_xyz[2]
    rr = radius * radius
    a = dx * dx + dy * dy + dz * dz
    b = ox * dx + oy * dy + oz * dz
    c = ox * ox + oy * oy + oz * oz - rr
    disc = b * b - a * c
    if disc < 0:
        return False, np.nan, np.nan
    root = math.sqrt(max(0, disc))
    t0 = (-b - root) / a
    t1 = (-b + root) / a
    if t0 <= t1:
        return True, t0, t1
    return True, t1, t0


@njit(cache=True)
def _seed_domain_xyz_kernel(
    origins: np.ndarray,
    directions: np.ndarray,
    clip_lo: float,
    t_hi: float,
    domain_lo: np.ndarray,
    domain_hi: np.ndarray,
) -> np.ndarray:
    """Return clipped Cartesian box midpoints for one flat ray batch."""
    n_rays = int(origins.shape[0])
    seed_xyz = np.full((n_rays, 3), np.nan, dtype=np.float64)
    enter = np.full(n_rays, clip_lo, dtype=np.float64)
    exit_ = np.full(n_rays, t_hi, dtype=np.float64)
    hit = np.ones(n_rays, dtype=np.bool_)

    for axis in range(3):
        for ray_id in range(n_rays):
            if not hit[ray_id]:
                continue
            oa = origins[ray_id, axis]
            da = directions[ray_id, axis]
            slab_lo = domain_lo[axis]
            slab_hi = domain_hi[axis]
            if da == 0:
                if oa < slab_lo or oa > slab_hi:
                    hit[ray_id] = False
                continue
            t0 = (slab_lo - oa) / da
            t1 = (slab_hi - oa) / da
            lo = min(t0, t1)
            hi = max(t0, t1)
            if lo > enter[ray_id]:
                enter[ray_id] = lo
            if hi < exit_[ray_id]:
                exit_[ray_id] = hi
            if exit_[ray_id] < enter[ray_id]:
                hit[ray_id] = False

    for ray_id in range(n_rays):
        if not hit[ray_id]:
            continue
        seed_t = (1 / 2) * (enter[ray_id] + exit_[ray_id])
        seed_xyz[ray_id] = origins[ray_id] + seed_t * directions[ray_id]
    return seed_xyz


@njit(cache=True)
def _seed_domain_rpa_kernel(
    origins: np.ndarray,
    directions: np.ndarray,
    clip_lo: float,
    t_hi: float,
    r_min: float,
    r_max: float,
) -> np.ndarray:
    """Return one front-visible spherical shell seed per flat ray."""
    n_rays = int(origins.shape[0])
    seed_xyz = np.full((n_rays, 3), np.nan, dtype=np.float64)
    tol = _EXIT_TOL * max(r_max, 1)

    for ray_id in range(n_rays):
        hit_outer, t_outer0, t_outer1 = _sphere_interval_kernel(origins[ray_id], directions[ray_id], r_max)
        if not hit_outer:
            continue
        visible_start = max(0, t_outer0)
        visible_end = t_outer1
        if visible_end < visible_start:
            continue
        if r_min > 0:
            hit_inner, t_inner0, t_inner1 = _sphere_interval_kernel(origins[ray_id], directions[ray_id], r_min)
            if hit_inner and t_inner1 >= visible_start:
                if t_inner0 > visible_start:
                    visible_end = min(visible_end, t_inner0)
                else:
                    continue
        visible_start = max(visible_start, clip_lo)
        visible_end = min(visible_end, t_hi)
        if visible_end < visible_start:
            continue

        direction_norm_sq = _dot3(directions[ray_id], directions[ray_id])
        t_closest = -_dot3(origins[ray_id], directions[ray_id]) / direction_norm_sq
        closest_xyz = origins[ray_id] + t_closest * directions[ray_id]
        closest_radius = math.sqrt(_dot3(closest_xyz, closest_xyz))
        seed_t = np.nan
        if (
            t_closest >= (visible_start - tol)
            and t_closest <= (visible_end + tol)
            and closest_radius >= (r_min - tol)
            and closest_radius <= (r_max + tol)
        ):
            seed_t = t_closest
        if not np.isfinite(seed_t):
            r_seed = (1 / 2) * (r_min + r_max)
            hit_seed, t_seed0, t_seed1 = _sphere_interval_kernel(origins[ray_id], directions[ray_id], r_seed)
            if hit_seed:
                if t_seed0 >= (visible_start - tol) and t_seed0 <= (visible_end + tol):
                    seed_t = t_seed0
                if t_seed1 >= (visible_start - tol) and t_seed1 <= (visible_end + tol):
                    if (not np.isfinite(seed_t)) or (t_seed1 < seed_t):
                        seed_t = t_seed1
        if not np.isfinite(seed_t):
            seed_t = (1 / 2) * (visible_start + visible_end)
        seed_xyz[ray_id] = origins[ray_id] + seed_t * directions[ray_id]
    return seed_xyz


@njit(cache=True)
def _lookup_xyz_leaf_kernel(
    point_xyz: np.ndarray,
    hint_cell_id: int,
    tree_coord_code: int,
    cell_child: np.ndarray,
    root_cell_ids: np.ndarray,
    cell_parent: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
) -> int:
    """Resolve one Cartesian query to one runtime cell id, or `-1`."""
    q = np.empty(3, dtype=np.float64)
    if tree_coord_code == _TREE_COORD_XYZ:
        q[:] = point_xyz
    else:
        q[0], q[1], q[2] = xyz_to_rpa_components(point_xyz[0], point_xyz[1], point_xyz[2])

    if not (np.isfinite(q[0]) and np.isfinite(q[1]) and np.isfinite(q[2])):
        return -1
    current = int(hint_cell_id)
    if tree_coord_code == _TREE_COORD_XYZ:
        if not _contains_box_xyz(q, domain_bounds, domain_bounds):
            return -1

        while current >= 0 and not _contains_box_xyz(q, cell_bounds[current], domain_bounds):
            current = int(cell_parent[current])

        if current < 0:
            for root_pos in range(int(root_cell_ids.shape[0])):
                root_cell_id = int(root_cell_ids[root_pos])
                if _contains_box_xyz(q, cell_bounds[root_cell_id], domain_bounds):
                    current = root_cell_id
                    break
        if current < 0:
            return -1

        while True:
            next_cell_id = -1
            has_child = False
            for child_ord in range(8):
                child_id = int(cell_child[current, child_ord])
                if child_id < 0:
                    continue
                has_child = True
                if _contains_box_xyz(q, cell_bounds[child_id], domain_bounds):
                    next_cell_id = child_id
                    break
            if not has_child:
                return int(current)
            if next_cell_id < 0:
                return -1
            current = int(next_cell_id)

    if not _contains_box_rpa(q, domain_bounds, axis2_period, axis2_periodic):
        return -1

    while current >= 0 and not _contains_box_rpa(q, cell_bounds[current], axis2_period, axis2_periodic):
        current = int(cell_parent[current])

    if current < 0:
        for root_pos in range(int(root_cell_ids.shape[0])):
            root_cell_id = int(root_cell_ids[root_pos])
            if _contains_box_rpa(q, cell_bounds[root_cell_id], axis2_period, axis2_periodic):
                current = root_cell_id
                break
    if current < 0:
        return -1

    while True:
        next_cell_id = -1
        has_child = False
        for child_ord in range(8):
            child_id = int(cell_child[current, child_ord])
            if child_id < 0:
                continue
            has_child = True
            if _contains_box_rpa(q, cell_bounds[child_id], axis2_period, axis2_periodic):
                next_cell_id = child_id
                break
        if not has_child:
            return int(current)
        if next_cell_id < 0:
            return -1
        current = int(next_cell_id)


@njit(cache=True)
def _probe_neighbor_leaf_from_normal_kernel(
    point_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    point_scale: float,
    hint_cell_id: int,
    tree_coord_code: int,
    cell_child: np.ndarray,
    root_cell_ids: np.ndarray,
    cell_parent: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
) -> int:
    """Probe slightly across one face normal and resolve the owning leaf."""
    normal_norm = math.sqrt(_dot3(normal_xyz, normal_xyz))
    if normal_norm <= 0:
        return -1
    probe_xyz = point_xyz + ((_BOUNDARY_SHIFT_FACTOR * max(1, point_scale)) / normal_norm) * normal_xyz
    return _lookup_xyz_leaf_kernel(
        probe_xyz,
        hint_cell_id,
        tree_coord_code,
        cell_child,
        root_cell_ids,
        cell_parent,
        cell_bounds,
        domain_bounds,
        axis2_period,
        axis2_periodic,
    )
@njit(cache=True)
def _wrapped_coord_delta(value: float, target: float, period: float, periodic: bool) -> float:
    """Return one coordinate difference, wrapped onto the principal interval when periodic."""
    delta = value - target
    if not periodic or period <= 0:
        return delta
    half_period = (1 / 2) * period
    while delta <= -half_period:
        delta += period
    while delta > half_period:
        delta -= period
    return delta


@njit(cache=True)
def _point_on_cell_face_kernel(
    point_xyz: np.ndarray,
    face_id: int,
    cell_id: int,
    tree_coord_code: int,
    cell_bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
) -> bool:
    """Return whether one boundary point lies on one face of one runtime cell."""
    if tree_coord_code == _TREE_COORD_XYZ:
        point_coord = point_xyz
    else:
        point_coord = np.array(xyz_to_rpa_components(point_xyz[0], point_xyz[1], point_xyz[2]), dtype=np.float64)
    axis = int(_FACE_AXIS[face_id])
    side = int(_FACE_SIDE[face_id])
    face_start = float(cell_bounds[cell_id, axis, 0])
    face_width = float(cell_bounds[cell_id, axis, 1])
    face_stop = face_start + face_width
    coord_value = float(point_coord[axis])
    tol = _EXIT_TOL * max(1, abs(face_start), abs(face_stop), abs(face_width))
    if axis == 2:
        face_delta = _wrapped_coord_delta(coord_value, face_start if side == 0 else face_stop, axis2_period, axis2_periodic)
    else:
        face_delta = coord_value - (face_start if side == 0 else face_stop)
    return abs(face_delta) <= tol


@njit(cache=True)
def _face_exit_subface_kernel(face_patch_center: np.ndarray, point_xyz: np.ndarray) -> int:
    """Return the nearest cached face patch for one boundary point."""
    best_subface = 0
    best_distance_sq = np.inf
    for subface_id in range(4):
        delta = point_xyz - face_patch_center[subface_id]
        distance_sq = _dot3(delta, delta)
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_subface = subface_id
    return int(best_subface)


@njit(cache=True)
def _resolve_boundary_owner_leaf_kernel(
    current_leaf: int,
    point_xyz: np.ndarray,
    face_normal: np.ndarray,
    face_patch_center: np.ndarray,
    active_face_mask: int,
    leaf_valid: np.ndarray,
    next_cell: np.ndarray,
    tree_coord_code: int,
    cell_bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
) -> int:
    """Resolve one boundary owner leaf by crossing the active face topology once."""
    if active_face_mask == 0:
        return -1
    candidate_cell = int(current_leaf)
    for face_id in range(6):
        if (active_face_mask & (1 << face_id)) == 0:
            continue
        if tree_coord_code == _TREE_COORD_XYZ:
            if not _point_on_cell_face_kernel(
                point_xyz,
                face_id,
                candidate_cell,
                tree_coord_code,
                cell_bounds,
                axis2_period,
                axis2_periodic,
            ):
                continue
        patch_owner = int(current_leaf)
        if candidate_cell < leaf_valid.shape[0] and leaf_valid[candidate_cell]:
            patch_owner = int(candidate_cell)
        subface_id = _face_exit_subface_kernel(face_patch_center[patch_owner, face_id], point_xyz)
        candidate_cell = int(next_cell[candidate_cell, face_id, subface_id])
        if candidate_cell < 0:
            return -1
    if candidate_cell < leaf_valid.shape[0] and leaf_valid[candidate_cell] and candidate_cell != current_leaf:
        return int(candidate_cell)
    return -1


@njit(cache=True)
def _point_inside_face_kernel(
    face_xyz: np.ndarray,
    face_edge_normal: np.ndarray,
    face_edge_d: np.ndarray,
    point_xyz: np.ndarray,
    tol: float,
) -> bool:
    """Return whether one point lies inside one convex planar face quad."""
    all_nonnegative = True
    all_nonpositive = True
    for edge_id in range(4):
        sign = _dot3(face_edge_normal[edge_id], point_xyz) - face_edge_d[edge_id]
        if sign < -tol:
            all_nonnegative = False
        if sign > tol:
            all_nonpositive = False
    if all_nonnegative or all_nonpositive:
        return True

    edge_scale = 0
    for edge_id in range(4):
        p0 = face_xyz[edge_id]
        p1 = face_xyz[(edge_id + 1) % 4]
        edge = p1 - p0
        edge_len = math.sqrt(_dot3(edge, edge))
        if edge_len > edge_scale:
            edge_scale = edge_len
    edge_tol = max(tol, (2 * _BOUNDARY_SHIFT_FACTOR) * max(1, edge_scale))
    for edge_id in range(4):
        p0 = face_xyz[edge_id]
        p1 = face_xyz[(edge_id + 1) % 4]
        edge = p1 - p0
        edge_sq = _dot3(edge, edge)
        if edge_sq <= 0:
            closest_xyz = p0
        else:
            weight = _dot3(point_xyz - p0, edge) / edge_sq
            if weight < 0:
                weight = 0
            elif weight > 1:
                weight = 1
            closest_xyz = p0 + weight * edge
        if math.sqrt(_dot3(point_xyz - closest_xyz, point_xyz - closest_xyz)) <= edge_tol:
            return True
    return False


@njit(cache=True)
def _point_inside_cell_kernel(
    face_normal: np.ndarray,
    face_plane_d: np.ndarray,
    point_xyz: np.ndarray,
    tol: float,
) -> bool:
    """Return whether one point lies inside one convex planar-face cell."""
    for face_id in range(6):
        if (_dot3(face_normal[face_id], point_xyz) - face_plane_d[face_id]) > tol:
            return False
    return True


@njit(cache=True)
def _solve_cell_segment_kernel(
    face_xyz: np.ndarray,
    face_normal: np.ndarray,
    face_plane_d: np.ndarray,
    face_edge_normal: np.ndarray,
    face_edge_d: np.ndarray,
    leaf_scale: float,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    current_t: float,
    t_min: float,
) -> tuple[bool, float, float, int, int]:
    """Return one forward cell segment plus one active exit-face mask, or `(False, ...)` if unresolved."""
    point_tol = _EXIT_TOL * max(1, leaf_scale)
    inside_now = _point_inside_cell_kernel(
        face_normal,
        face_plane_d,
        origin_xyz + current_t * direction_xyz,
        point_tol,
    )

    hit_t = np.empty(6, dtype=np.float64)
    hit_face = np.empty(6, dtype=np.int64)
    hit_count = 0
    for face_id in range(6):
        normal_xyz = face_normal[face_id]
        denom = _dot3(normal_xyz, direction_xyz)
        if abs(denom) <= _EXIT_TOL:
            continue
        t_hit = (face_plane_d[face_id] - _dot3(normal_xyz, origin_xyz)) / denom
        face_hit_xyz = origin_xyz + t_hit * direction_xyz
        if _point_inside_face_kernel(face_xyz[face_id], face_edge_normal[face_id], face_edge_d[face_id], face_hit_xyz, point_tol):
            hit_t[hit_count] = t_hit
            hit_face[hit_count] = face_id
            hit_count += 1
    if hit_count == 0:
        return False, np.nan, np.nan, -1, -1

    for i in range(hit_count):
        best = i
        for j in range(i + 1, hit_count):
            if hit_t[j] < hit_t[best]:
                best = j
        if best != i:
            swap_t = hit_t[i]
            hit_t[i] = hit_t[best]
            hit_t[best] = swap_t
            swap_face = hit_face[i]
            hit_face[i] = hit_face[best]
            hit_face[best] = swap_face

    enter_tol = _EXIT_TOL * max(1, abs(current_t))
    segment_enter = current_t if inside_now else np.nan
    exit_t = np.nan
    exit_face = -1
    for hit_id in range(hit_count):
        t_hit = hit_t[hit_id]
        if t_hit < (t_min - enter_tol):
            continue
        if not np.isfinite(segment_enter):
            segment_enter = t_hit
            continue
        if t_hit <= (segment_enter + enter_tol):
            continue
        exit_t = t_hit
        exit_face = int(hit_face[hit_id])
        break
    if np.isfinite(segment_enter) and not np.isfinite(exit_t) and inside_now:
        active_face_mask = 0
        for hit_id in range(hit_count):
            if abs(hit_t[hit_id] - segment_enter) <= enter_tol:
                if exit_face < 0:
                    exit_face = int(hit_face[hit_id])
                active_face_mask |= 1 << int(hit_face[hit_id])
        if active_face_mask != 0:
            return True, segment_enter, segment_enter, exit_face, int(active_face_mask)
    if not np.isfinite(segment_enter) or not np.isfinite(exit_t):
        return False, np.nan, np.nan, -1, -1
    if exit_t <= segment_enter:
        return False, np.nan, np.nan, -1, -1

    exit_tol = _EXIT_TOL * max(1, abs(exit_t))
    active_face_mask = 0
    for hit_id in range(hit_count):
        if abs(hit_t[hit_id] - exit_t) <= exit_tol:
            active_face_mask |= 1 << int(hit_face[hit_id])
    if active_face_mask == 0 and exit_face >= 0:
        active_face_mask = 1 << int(exit_face)
    return True, segment_enter, exit_t, exit_face, int(active_face_mask)


@njit(cache=True)
def _cell_segment_trace_kernel(
    face_xyz: np.ndarray,
    face_normal: np.ndarray,
    face_plane_d: np.ndarray,
    face_edge_normal: np.ndarray,
    face_edge_d: np.ndarray,
    leaf_scale: float,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    current_t: float,
    t_min: float,
) -> tuple[bool, float, float, int, int]:
    """Return one exact forward cell segment with one boundary-shift retry."""
    ok, segment_enter, exit_t, exit_face, active_face_mask = _solve_cell_segment_kernel(
        face_xyz,
        face_normal,
        face_plane_d,
        face_edge_normal,
        face_edge_d,
        leaf_scale,
        origin_xyz,
        direction_xyz,
        current_t,
        t_min,
    )
    if ok:
        return True, segment_enter, exit_t, exit_face, active_face_mask

    direction_norm = math.sqrt(_dot3(direction_xyz, direction_xyz))
    if direction_norm <= 0:
        return False, np.nan, np.nan, -1, -1
    boundary_shift = (_BOUNDARY_SHIFT_FACTOR * max(1, leaf_scale)) / direction_norm
    ok, segment_enter, exit_t, exit_face, active_face_mask = _solve_cell_segment_kernel(
        face_xyz,
        face_normal,
        face_plane_d,
        face_edge_normal,
        face_edge_d,
        leaf_scale,
        origin_xyz + boundary_shift * direction_xyz,
        direction_xyz,
        current_t,
        t_min,
    )
    if not ok:
        return False, np.nan, np.nan, -1, -1
    return True, segment_enter + boundary_shift, exit_t + boundary_shift, exit_face, active_face_mask


@njit(cache=True)
def _touching_neighbor_leaves_kernel(
    leaf_id: int,
    point_xyz: np.ndarray,
    leaf_valid: np.ndarray,
    face_xyz: np.ndarray,
    face_normal: np.ndarray,
    face_plane_d: np.ndarray,
    leaf_scales: np.ndarray,
    next_cell: np.ndarray,
    tree_coord_code: int,
    cell_child: np.ndarray,
    root_cell_ids: np.ndarray,
    cell_parent: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
) -> tuple[np.ndarray, int]:
    """Return unique neighboring leaves whose cells touch one boundary point of one leaf."""
    neighbors = np.full(24, -1, dtype=np.int64)
    count = 0
    point_tol = _EXIT_TOL * max(1, leaf_scales[leaf_id])
    for face_id in range(6):
        normal_xyz = face_normal[leaf_id, face_id]
        face_distance = _dot3(normal_xyz, point_xyz) - face_plane_d[leaf_id, face_id]
        if abs(face_distance) > point_tol:
            continue
        for subface_id in range(4):
            candidate_cell = int(next_cell[leaf_id, face_id, subface_id])
            if candidate_cell < 0:
                continue
            if candidate_cell < leaf_valid.shape[0] and leaf_valid[candidate_cell]:
                candidate = candidate_cell
            else:
                candidate = _probe_neighbor_leaf_from_normal_kernel(
                    point_xyz,
                    normal_xyz,
                    leaf_scales[leaf_id],
                    candidate_cell,
                    tree_coord_code,
                    cell_child,
                    root_cell_ids,
                    cell_parent,
                    cell_bounds,
                    domain_bounds,
                    axis2_period,
                    axis2_periodic,
                )
            if candidate < 0:
                continue
            seen = False
            for idx in range(count):
                if neighbors[idx] == candidate:
                    seen = True
                    break
            if not seen:
                neighbors[count] = candidate
                count += 1
    return neighbors, count


@njit(cache=True)
def _rpa_axis_candidate_leaves_kernel(
    point_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    n_polar: int,
    n_azimuth: int,
    tree_coord_code: int,
    cell_child: np.ndarray,
    root_cell_ids: np.ndarray,
    cell_parent: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
) -> tuple[np.ndarray, int]:
    """Return one finite set of native-sector candidate leaves for one spherical axis point."""
    candidates = np.full(n_azimuth + 1, -1, dtype=np.int64)
    count = 0
    if tree_coord_code != _TREE_COORD_RPA:
        return candidates, count

    radial_xy = math.hypot(point_xyz[0], point_xyz[1])
    radial_scale = max(math.sqrt(_dot3(point_xyz, point_xyz)), 1)
    axis_tol = _BOUNDARY_SHIFT_FACTOR * radial_scale
    if radial_xy > axis_tol:
        return candidates, count

    direction_norm = math.sqrt(_dot3(direction_xyz, direction_xyz))
    if direction_norm > 0:
        probe_xyz = point_xyz + ((_BOUNDARY_SHIFT_FACTOR * radial_scale) / direction_norm) * direction_xyz
        forward_leaf = _lookup_xyz_leaf_kernel(
            probe_xyz,
            -1,
            tree_coord_code,
            cell_child,
            root_cell_ids,
            cell_parent,
            cell_bounds,
            domain_bounds,
            axis2_period,
            axis2_periodic,
        )
        if forward_leaf >= 0:
            candidates[count] = int(forward_leaf)
            count += 1

    r_value = math.sqrt(_dot3(point_xyz, point_xyz))
    polar_center = (1 / 2) * (math.pi / float(n_polar))
    if point_xyz[2] < 0:
        polar_center = math.pi - polar_center
    sin_polar = math.sin(polar_center)
    cos_polar = math.cos(polar_center)
    azimuth_width = (2 * math.pi) / float(n_azimuth)
    for azimuth_id in range(n_azimuth):
        azimuth_center = (float(azimuth_id) + (1 / 2)) * azimuth_width
        query_xyz = np.empty(3, dtype=np.float64)
        query_xyz[0] = r_value * sin_polar * math.cos(azimuth_center)
        query_xyz[1] = r_value * sin_polar * math.sin(azimuth_center)
        query_xyz[2] = r_value * cos_polar
        leaf_id = _lookup_xyz_leaf_kernel(
            query_xyz,
            -1,
            tree_coord_code,
            cell_child,
            root_cell_ids,
            cell_parent,
            cell_bounds,
            domain_bounds,
            axis2_period,
            axis2_periodic,
        )
        if leaf_id < 0:
            continue
        seen = False
        for idx in range(count):
            if candidates[idx] == leaf_id:
                seen = True
                break
        if not seen:
            candidates[count] = int(leaf_id)
            count += 1
    return candidates, count


@njit(cache=True)
def _append_unique_leaf_id(candidates: np.ndarray, count: int, leaf_id: int) -> int:
    """Append one leaf id when it is valid and not already present."""
    if leaf_id < 0:
        return count
    for idx in range(count):
        if int(candidates[idx]) == leaf_id:
            return count
    candidates[count] = leaf_id
    return count + 1


@njit(cache=True)
def _boundary_continuation_leaf_kernel(
    initial_leaf_ids: np.ndarray,
    initial_count: int,
    point_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    exclude_leaf_id: int,
    leaf_valid: np.ndarray,
    face_xyz: np.ndarray,
    face_normal: np.ndarray,
    face_plane_d: np.ndarray,
    face_edge_normal: np.ndarray,
    face_edge_d: np.ndarray,
    leaf_scales: np.ndarray,
    next_cell: np.ndarray,
    n_polar: int,
    n_azimuth: int,
    tree_coord_code: int,
    cell_child: np.ndarray,
    root_cell_ids: np.ndarray,
    cell_parent: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
) -> int:
    """Return one deterministic continuation leaf from one boundary point, or `-1`."""
    queue = np.full(max(1, 4 * leaf_valid.shape[0]), -1, dtype=np.int64)
    head = 0
    tail = 0
    visited = np.zeros(leaf_valid.shape[0], dtype=np.bool_)
    direction_norm = math.sqrt(_dot3(direction_xyz, direction_xyz))
    if direction_norm <= 0:
        return -1
    for idx in range(initial_count):
        leaf_id = int(initial_leaf_ids[idx])
        if leaf_id >= 0:
            queue[tail] = leaf_id
            tail += 1

    valid_enter = np.empty(leaf_valid.shape[0], dtype=np.float64)
    valid_length = np.empty(leaf_valid.shape[0], dtype=np.float64)
    valid_exit = np.empty(leaf_valid.shape[0], dtype=np.float64)
    valid_leaf_ids = np.empty(leaf_valid.shape[0], dtype=np.int64)
    valid_count = 0

    while head < tail:
        leaf_id = int(queue[head])
        head += 1
        if leaf_id < 0 or visited[leaf_id]:
            continue
        visited[leaf_id] = True
        ok, segment_enter, segment_exit, _, _ = _cell_segment_trace_kernel(
            face_xyz[leaf_id],
            face_normal[leaf_id],
            face_plane_d[leaf_id],
            face_edge_normal[leaf_id],
            face_edge_d[leaf_id],
            leaf_scales[leaf_id],
            point_xyz,
            direction_xyz,
            0,
            0,
        )
        if ok:
            if leaf_id != exclude_leaf_id:
                enter_floor = (_BOUNDARY_SHIFT_FACTOR * max(1, leaf_scales[leaf_id])) / direction_norm
                valid_enter[valid_count] = 0 if segment_enter <= enter_floor else segment_enter
                valid_length[valid_count] = segment_exit - segment_enter
                valid_exit[valid_count] = segment_exit
                valid_leaf_ids[valid_count] = leaf_id
                valid_count += 1
                continue
        touching, touching_count = _touching_neighbor_leaves_kernel(
            leaf_id,
            point_xyz,
            leaf_valid,
            face_xyz,
            face_normal,
            face_plane_d,
            leaf_scales,
            next_cell,
            tree_coord_code,
            cell_child,
            root_cell_ids,
            cell_parent,
            cell_bounds,
            domain_bounds,
            axis2_period,
            axis2_periodic,
        )
        for idx in range(touching_count):
            next_id = int(touching[idx])
            if next_id >= 0 and not visited[next_id]:
                queue[tail] = next_id
                tail += 1

    if tree_coord_code == _TREE_COORD_RPA:
        axis_candidates, axis_count = _rpa_axis_candidate_leaves_kernel(
            point_xyz,
            direction_xyz,
            n_polar,
            n_azimuth,
            tree_coord_code,
            cell_child,
            root_cell_ids,
            cell_parent,
            cell_bounds,
            domain_bounds,
            axis2_period,
            axis2_periodic,
            )
        for idx in range(axis_count):
            leaf_id = int(axis_candidates[idx])
            if leaf_id < 0 or visited[leaf_id] or leaf_id == exclude_leaf_id:
                continue
            ok, segment_enter, segment_exit, _, _ = _cell_segment_trace_kernel(
                face_xyz[leaf_id],
                face_normal[leaf_id],
                face_plane_d[leaf_id],
                face_edge_normal[leaf_id],
                face_edge_d[leaf_id],
                leaf_scales[leaf_id],
                point_xyz,
                direction_xyz,
                0,
                0,
            )
            if ok:
                enter_floor = (_BOUNDARY_SHIFT_FACTOR * max(1, leaf_scales[leaf_id])) / direction_norm
                valid_enter[valid_count] = 0 if segment_enter <= enter_floor else segment_enter
                valid_length[valid_count] = segment_exit - segment_enter
                valid_exit[valid_count] = segment_exit
                valid_leaf_ids[valid_count] = leaf_id
                valid_count += 1

    if valid_count == 0:
        return -1

    best_idx = -1
    for idx in range(valid_count):
        if best_idx < 0:
            best_idx = idx
            continue
        if valid_enter[idx] < valid_enter[best_idx]:
            best_idx = idx
            continue
        if valid_enter[idx] > valid_enter[best_idx]:
            continue
        if valid_length[idx] > valid_length[best_idx]:
            best_idx = idx
            continue
        if valid_length[idx] < valid_length[best_idx]:
            continue
        if valid_exit[idx] > valid_exit[best_idx]:
            best_idx = idx
            continue
        if valid_exit[idx] < valid_exit[best_idx]:
            continue
        if valid_leaf_ids[idx] < valid_leaf_ids[best_idx]:
            best_idx = idx

    return int(valid_leaf_ids[best_idx])


@njit(cache=True)
def _launch_leaf_kernel(
    seed_leaf_id: int,
    seed_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    leaf_valid: np.ndarray,
    face_normal: np.ndarray,
    face_patch_center: np.ndarray,
    face_xyz: np.ndarray,
    face_plane_d: np.ndarray,
    face_edge_normal: np.ndarray,
    face_edge_d: np.ndarray,
    leaf_scales: np.ndarray,
    next_cell: np.ndarray,
    n_polar: int,
    n_azimuth: int,
    tree_coord_code: int,
    cell_child: np.ndarray,
    root_cell_ids: np.ndarray,
    cell_parent: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
) -> int:
    """Return the unique leaf to launch into from one seed boundary point."""
    initial_leaf_ids = np.full(1 + 48 + n_azimuth, -1, dtype=np.int64)
    count = 0

    if seed_leaf_id >= 0:
        count = _append_unique_leaf_id(initial_leaf_ids, count, int(seed_leaf_id))
        for face_id in range(6):
            normal_xyz = face_normal[seed_leaf_id, face_id]
            for subface_id in range(4):
                candidate_cell = int(next_cell[seed_leaf_id, face_id, subface_id])
                if candidate_cell < 0:
                    continue
                if candidate_cell < leaf_valid.shape[0] and leaf_valid[candidate_cell]:
                    candidate_leaf = candidate_cell
                else:
                    patch_center = face_patch_center[seed_leaf_id, face_id, subface_id]
                    candidate_leaf = _probe_neighbor_leaf_from_normal_kernel(
                        patch_center,
                        normal_xyz,
                        leaf_scales[seed_leaf_id],
                        candidate_cell,
                        tree_coord_code,
                        cell_child,
                        root_cell_ids,
                        cell_parent,
                        cell_bounds,
                        domain_bounds,
                        axis2_period,
                        axis2_periodic,
                    )
                count = _append_unique_leaf_id(initial_leaf_ids, count, int(candidate_leaf))
        touching_leaf_ids, touching_count = _touching_neighbor_leaves_kernel(
            int(seed_leaf_id),
            seed_xyz,
            leaf_valid,
            face_xyz,
            face_normal,
            face_plane_d,
            leaf_scales,
            next_cell,
            tree_coord_code,
            cell_child,
            root_cell_ids,
            cell_parent,
            cell_bounds,
            domain_bounds,
            axis2_period,
            axis2_periodic,
        )
        for idx in range(touching_count):
            count = _append_unique_leaf_id(initial_leaf_ids, count, int(touching_leaf_ids[idx]))

    if tree_coord_code == _TREE_COORD_RPA:
        axis_leaf_ids, axis_count = _rpa_axis_candidate_leaves_kernel(
            seed_xyz,
            direction_xyz,
            n_polar,
            n_azimuth,
            tree_coord_code,
            cell_child,
            root_cell_ids,
            cell_parent,
            cell_bounds,
            domain_bounds,
            axis2_period,
            axis2_periodic,
        )
        for idx in range(axis_count):
            count = _append_unique_leaf_id(initial_leaf_ids, count, int(axis_leaf_ids[idx]))

    if count == 0:
        return -1

    return _boundary_continuation_leaf_kernel(
        initial_leaf_ids,
        count,
        seed_xyz,
        direction_xyz,
        -1,
        leaf_valid,
        face_xyz,
        face_normal,
        face_plane_d,
        face_edge_normal,
        face_edge_d,
        leaf_scales,
        next_cell,
        n_polar,
        n_azimuth,
        tree_coord_code,
        cell_child,
        root_cell_ids,
        cell_parent,
        cell_bounds,
        domain_bounds,
        axis2_period,
        axis2_periodic,
    )


@njit(cache=True)
def _trace_one_ray_kernel(
    start_leaf_id: int,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    t_min: float,
    t_max: float,
    leaf_valid: np.ndarray,
    face_xyz: np.ndarray,
    face_normal: np.ndarray,
    face_plane_d: np.ndarray,
    face_edge_normal: np.ndarray,
    face_edge_d: np.ndarray,
    leaf_scales: np.ndarray,
    face_patch_center: np.ndarray,
    next_cell: np.ndarray,
    tree_coord_code: int,
    cell_child: np.ndarray,
    root_cell_ids: np.ndarray,
    cell_parent: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
    n_valid_leaf: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk one ray across neighboring leaves using exact cell intersections and boundary-ownership continuation."""
    current_leaf = int(start_leaf_id)
    if current_leaf < 0 or current_leaf >= leaf_valid.shape[0] or not leaf_valid[current_leaf]:
        raise ValueError("start_leaf_id must reference one valid leaf slot.")

    leaf_ids = []
    t_enter_list = []
    t_exit_list = []
    current_t = t_min
    max_steps = int(n_valid_leaf) + 1

    for _ in range(max_steps):
        if current_t >= t_max:
            break
        ok, segment_enter, exit_t, _, active_face_mask = _cell_segment_trace_kernel(
            face_xyz[current_leaf],
            face_normal[current_leaf],
            face_plane_d[current_leaf],
            face_edge_normal[current_leaf],
            face_edge_d[current_leaf],
            leaf_scales[current_leaf],
            origin_xyz,
            direction_xyz,
            current_t,
            t_min,
        )
        if not ok:
            break
        zero_length_tol = _EXIT_TOL * max(1, abs(segment_enter), abs(exit_t))
        if segment_enter >= t_max:
            break
        zero_length_step = exit_t <= (segment_enter + zero_length_tol)
        if not zero_length_step:
            segment_exit = exit_t if exit_t < t_max else t_max
            leaf_ids.append(current_leaf)
            t_enter_list.append(segment_enter)
            t_exit_list.append(segment_exit)
            if exit_t >= t_max:
                break
        exit_xyz = origin_xyz + exit_t * direction_xyz
        if zero_length_step:
            handoff_face_mask = 0
            for face_id in range(6):
                if (active_face_mask & (1 << face_id)) == 0:
                    continue
                if _dot3(direction_xyz, face_normal[current_leaf, face_id]) > 0:
                    handoff_face_mask |= 1 << face_id
            next_leaf_id = -1
            if handoff_face_mask != 0:
                candidate_cell = int(current_leaf)
                for face_id in range(6):
                    if (handoff_face_mask & (1 << face_id)) == 0:
                        continue
                    subface_id = _face_exit_subface_kernel(face_patch_center[current_leaf, face_id], exit_xyz)
                    candidate_cell = int(next_cell[candidate_cell, face_id, subface_id])
                    if candidate_cell < 0:
                        break
                if (
                    candidate_cell >= 0
                    and candidate_cell < leaf_valid.shape[0]
                    and leaf_valid[candidate_cell]
                    and candidate_cell != current_leaf
                ):
                    next_leaf_id = int(candidate_cell)
        else:
            next_leaf_id = _resolve_boundary_owner_leaf_kernel(
                current_leaf,
                exit_xyz,
                face_normal,
                face_patch_center,
                active_face_mask,
                leaf_valid,
                next_cell,
                tree_coord_code,
                cell_bounds,
                axis2_period,
                axis2_periodic,
            )
        if next_leaf_id < 0 or next_leaf_id == current_leaf:
            break
        current_leaf = int(next_leaf_id)
        current_t = exit_t

    n_segment = len(leaf_ids)
    leaf_ids_out = np.empty(n_segment, dtype=np.int64)
    t_enter_out = np.empty(n_segment, dtype=np.float64)
    t_exit_out = np.empty(n_segment, dtype=np.float64)
    for idx in range(n_segment):
        leaf_ids_out[idx] = leaf_ids[idx]
        t_enter_out[idx] = t_enter_list[idx]
        t_exit_out[idx] = t_exit_list[idx]
    return leaf_ids_out, t_enter_out, t_exit_out


@njit(cache=True)
def trace_one_ray_kernel(
    start_leaf_id: int,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    t_min: float,
    t_max: float,
    trace_state: tuple,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk one ray from one packed raw tracing-state bundle."""
    (
        leaf_valid,
        face_xyz,
        face_normal,
        face_plane_d,
        face_edge_normal,
        face_edge_d,
        leaf_scales,
        face_patch_center,
        next_cell,
        n_polar,
        n_azimuth,
        tree_coord_code,
        cell_child,
        root_cell_ids,
        cell_parent,
        cell_bounds,
        domain_bounds,
        axis2_period,
        axis2_periodic,
        n_valid_leaf,
    ) = trace_state
    return _trace_one_ray_kernel(
        start_leaf_id,
        origin_xyz,
        direction_xyz,
        t_min,
        t_max,
        leaf_valid,
        face_xyz,
        face_normal,
        face_plane_d,
        face_edge_normal,
        face_edge_d,
        leaf_scales,
        face_patch_center,
        next_cell,
        tree_coord_code,
        cell_child,
        root_cell_ids,
        cell_parent,
        cell_bounds,
        domain_bounds,
        axis2_period,
        axis2_periodic,
        n_valid_leaf,
    )


@njit(cache=True)
def _trace_rays_kernel(
    origins: np.ndarray,
    directions: np.ndarray,
    seed_xyz: np.ndarray,
    clip_lo: float,
    t_hi: float,
    leaf_valid: np.ndarray,
    face_xyz: np.ndarray,
    face_normal: np.ndarray,
    face_plane_d: np.ndarray,
    face_edge_normal: np.ndarray,
    face_edge_d: np.ndarray,
    leaf_scales: np.ndarray,
    face_patch_center: np.ndarray,
    next_cell: np.ndarray,
    n_polar: int,
    n_azimuth: int,
    tree_coord_code: int,
    cell_child: np.ndarray,
    root_cell_ids: np.ndarray,
    cell_parent: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
    n_valid_leaf: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trace one flat ray batch using compiled launch and walk kernels."""
    n_rays = int(origins.shape[0])
    ray_offsets = np.zeros(n_rays + 1, dtype=np.int64)
    cell_ids = []
    t_enter = []
    t_exit = []

    for ray_id in range(n_rays):
        ray_offsets[ray_id] = len(cell_ids)
        seed = seed_xyz[ray_id]
        if not (np.isfinite(seed[0]) and np.isfinite(seed[1]) and np.isfinite(seed[2])):
            continue

        seed_leaf_id = _lookup_xyz_leaf_kernel(
            seed,
            -1,
            tree_coord_code,
            cell_child,
            root_cell_ids,
            cell_parent,
            cell_bounds,
            domain_bounds,
            axis2_period,
            axis2_periodic,
        )
        if seed_leaf_id < 0:
            continue

        origin = origins[ray_id]
        direction = directions[ray_id]
        direction_norm_sq = _dot3(direction, direction)
        t_seed = _dot3(seed - origin, direction) / direction_norm_sq

        backward_leaf = _launch_leaf_kernel(
            seed_leaf_id,
            seed,
            -direction,
            leaf_valid,
            face_normal,
            face_patch_center,
            face_xyz,
            face_plane_d,
            face_edge_normal,
            face_edge_d,
            leaf_scales,
            next_cell,
            n_polar,
            n_azimuth,
            tree_coord_code,
            cell_child,
            root_cell_ids,
            cell_parent,
            cell_bounds,
            domain_bounds,
            axis2_period,
            axis2_periodic,
        )
        forward_leaf = _launch_leaf_kernel(
            seed_leaf_id,
            seed,
            direction,
            leaf_valid,
            face_normal,
            face_patch_center,
            face_xyz,
            face_plane_d,
            face_edge_normal,
            face_edge_d,
            leaf_scales,
            next_cell,
            n_polar,
            n_azimuth,
            tree_coord_code,
            cell_child,
            root_cell_ids,
            cell_parent,
            cell_bounds,
            domain_bounds,
            axis2_period,
            axis2_periodic,
        )

        backward_limit = max(0, t_seed - clip_lo)
        if backward_limit > 0 and backward_leaf >= 0:
            back_leaf_ids_local, back_enter_local, back_exit_local = _trace_one_ray_kernel(
                int(backward_leaf),
                seed,
                -direction,
                0,
                float(backward_limit),
                leaf_valid,
                face_xyz,
                face_normal,
                face_plane_d,
                face_edge_normal,
                face_edge_d,
                leaf_scales,
                face_patch_center,
                next_cell,
                tree_coord_code,
                cell_child,
                root_cell_ids,
                cell_parent,
                cell_bounds,
                domain_bounds,
                axis2_period,
                axis2_periodic,
                n_valid_leaf,
            )
        else:
            back_leaf_ids_local = np.empty(0, dtype=np.int64)
            back_enter_local = np.empty(0, dtype=np.float64)
            back_exit_local = np.empty(0, dtype=np.float64)

        forward_limit = max(0, t_hi - t_seed)
        if forward_limit > 0 and forward_leaf >= 0:
            forward_leaf_ids_local, forward_enter_local, forward_exit_local = _trace_one_ray_kernel(
                int(forward_leaf),
                seed,
                direction,
                0,
                float(forward_limit),
                leaf_valid,
                face_xyz,
                face_normal,
                face_plane_d,
                face_edge_normal,
                face_edge_d,
                leaf_scales,
                face_patch_center,
                next_cell,
                tree_coord_code,
                cell_child,
                root_cell_ids,
                cell_parent,
                cell_bounds,
                domain_bounds,
                axis2_period,
                axis2_periodic,
                n_valid_leaf,
            )
        else:
            forward_leaf_ids_local = np.empty(0, dtype=np.int64)
            forward_enter_local = np.empty(0, dtype=np.float64)
            forward_exit_local = np.empty(0, dtype=np.float64)

        back_count = int(back_leaf_ids_local.shape[0])
        forward_count = int(forward_leaf_ids_local.shape[0])
        if forward_count > 1:
            direction_norm = math.sqrt(direction_norm_sq)
            tail_leaf_id = int(forward_leaf_ids_local[forward_count - 1])
            tail_floor = (_BOUNDARY_SHIFT_FACTOR * max(1, leaf_scales[tail_leaf_id])) / direction_norm
            tail_length = float(forward_exit_local[forward_count - 1] - forward_enter_local[forward_count - 1])
            if tail_length <= tail_floor:
                forward_count -= 1
        joined = False
        if back_count > 0 and forward_count > 0 and back_leaf_ids_local[0] == forward_leaf_ids_local[0]:
            back_last_exit = t_seed - back_enter_local[0]
            forward_first_enter = t_seed + forward_enter_local[0]
            join_tol = _EXIT_TOL * max(1, abs(back_last_exit), abs(forward_first_enter))
            joined = abs(back_last_exit - forward_first_enter) <= join_tol

        back_stop = 0 if joined else -1
        for local_idx in range(back_count - 1, back_stop, -1):
            cell_ids.append(int(back_leaf_ids_local[local_idx]))
            t_enter.append(t_seed - back_exit_local[local_idx])
            t_exit.append(t_seed - back_enter_local[local_idx])

        if joined:
            cell_ids.append(int(back_leaf_ids_local[0]))
            t_enter.append(t_seed - back_exit_local[0])
            t_exit.append(t_seed + forward_exit_local[0])

        forward_start = 1 if joined else 0
        for local_idx in range(forward_start, forward_count):
            cell_ids.append(int(forward_leaf_ids_local[local_idx]))
            t_enter.append(t_seed + forward_enter_local[local_idx])
            t_exit.append(t_seed + forward_exit_local[local_idx])

    ray_offsets[-1] = len(cell_ids)
    n_segment = len(cell_ids)
    cell_ids_out = np.empty(n_segment, dtype=np.int64)
    t_enter_out = np.empty(n_segment, dtype=np.float64)
    t_exit_out = np.empty(n_segment, dtype=np.float64)
    for idx in range(n_segment):
        cell_ids_out[idx] = cell_ids[idx]
        t_enter_out[idx] = t_enter[idx]
        t_exit_out[idx] = t_exit[idx]
    return ray_offsets, cell_ids_out, t_enter_out, t_exit_out


@dataclass(frozen=True)
class RaySegments:
    """Packed per-ray cell segments with CSR-like ray indexing."""

    ray_offsets: np.ndarray
    cell_ids: np.ndarray
    t_enter: np.ndarray
    t_exit: np.ndarray
    ray_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        ray_offsets = np.asarray(self.ray_offsets, dtype=np.int64)
        cell_ids = np.asarray(self.cell_ids, dtype=np.int64)
        t_enter = np.asarray(self.t_enter, dtype=np.float64)
        t_exit = np.asarray(self.t_exit, dtype=np.float64)
        if ray_offsets.ndim != 1:
            raise ValueError("ray_offsets must be one 1D array.")
        if cell_ids.ndim != 1 or t_enter.ndim != 1 or t_exit.ndim != 1:
            raise ValueError("cell_ids, t_enter, and t_exit must be 1D arrays.")
        if cell_ids.shape != t_enter.shape or cell_ids.shape != t_exit.shape:
            raise ValueError("cell_ids, t_enter, and t_exit must have the same shape.")
        if ray_offsets.size == 0 or ray_offsets[0] != 0:
            raise ValueError("ray_offsets must start at 0.")
        if np.any(np.diff(ray_offsets) < 0):
            raise ValueError("ray_offsets must be nondecreasing.")
        if int(ray_offsets[-1]) != int(cell_ids.size):
            raise ValueError("ray_offsets[-1] must equal the segment count.")
        if np.any(t_exit <= t_enter):
            raise ValueError("Each segment must satisfy t_exit > t_enter.")
        object.__setattr__(self, "ray_offsets", ray_offsets)
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "t_enter", t_enter)
        object.__setattr__(self, "t_exit", t_exit)
        object.__setattr__(self, "ray_shape", tuple(int(v) for v in self.ray_shape))

    @property
    def n_rays(self) -> int:
        """Return the number of rays packed into this segment bundle."""
        return int(self.ray_offsets.size - 1)

    @property
    def segment_length(self) -> np.ndarray:
        """Return one 1D array of per-segment lengths."""
        return self.t_exit - self.t_enter


class OctreeRayTracer:
    """Trace rays against one octree geometry.

    The current implementation only seeds rays inside the global octree domain.
    """

    def __init__(self, tree: Octree) -> None:
        """Bind one tracer to one built octree."""
        if not isinstance(tree, Octree):
            raise TypeError("OctreeRayTracer requires a built Octree as its first argument.")
        logger.info("OctreeRayTracer.__init__: coord=%s", tree.tree_coord)
        t0 = time.perf_counter()
        self.tree = tree
        logger.info("_resolve_trace_coord...")
        t_coord = time.perf_counter()
        resolved_tree_coord = str(tree.tree_coord)
        if resolved_tree_coord == "xyz":
            face_corners = _build_face_corner_order(_XYZ_CORNER_BITS)
            tree_coord_code = int(_TREE_COORD_XYZ)
            axis2_period = 0
            axis2_periodic = False
        elif resolved_tree_coord == "rpa":
            face_corners = _build_face_corner_order(_RPA_CORNER_BITS)
            tree_coord_code = int(_TREE_COORD_RPA)
            axis2_period = 2 * math.pi
            axis2_periodic = True
        else:
            raise NotImplementedError(f"Unsupported tree_coord '{tree.tree_coord}' for OctreeRayTracer.")
        logger.info("_resolve_trace_coord complete in %.2fs", float(time.perf_counter() - t_coord))
        logger.info("_bind_tree_arrays...")
        t_bind = time.perf_counter()
        leaf_slot_count = int(tree.corners.shape[0])
        leaf_valid = tree.cell_levels >= 0
        n_polar = int(tree.leaf_shape[1])
        n_azimuth = int(tree.leaf_shape[2])
        cell_child = tree.cell_child
        cell_parent = tree.cell_parent
        root_cell_ids = np.flatnonzero(cell_parent < 0).astype(np.int64)
        cell_bounds = tree.cell_bounds
        next_cell = tree.cell_neighbor
        logger.info("_bind_tree_arrays complete in %.2fs", float(time.perf_counter() - t_bind))
        logger.info("_prepare_seed_domain...")
        t_seed = time.perf_counter()
        domain_lo, domain_hi = tree.domain_bounds(coord=tree.tree_coord)
        domain_bounds = np.empty((3, 2), dtype=np.float64)
        domain_bounds[:, 0] = domain_lo
        domain_bounds[:, 1] = domain_hi - domain_lo
        if tree_coord_code == int(_TREE_COORD_XYZ):
            seed_domain_xyz_lo = np.asarray(domain_lo, dtype=np.float64)
            seed_domain_xyz_hi = np.asarray(domain_hi, dtype=np.float64)
            seed_domain_r_min = np.nan
            seed_domain_r_max = np.nan
        else:
            if not np.isclose(float(domain_lo[1]), 0, atol=_EXIT_TOL, rtol=0):
                raise NotImplementedError("seed_domain for rpa currently requires polar_min == 0.")
            if not np.isclose(float(domain_hi[1]), math.pi, atol=_EXIT_TOL, rtol=0):
                raise NotImplementedError("seed_domain for rpa currently requires polar_max == pi.")
            if not np.isclose(float(domain_lo[2]), 0, atol=_EXIT_TOL, rtol=0):
                raise NotImplementedError("seed_domain for rpa currently requires azimuth_start == 0.")
            if not np.isclose(float(domain_hi[2] - domain_lo[2]), 2 * math.pi, atol=_EXIT_TOL, rtol=0):
                raise NotImplementedError("seed_domain for rpa currently requires full 2pi azimuth coverage.")
            seed_domain_xyz_lo = np.full(3, np.nan, dtype=np.float64)
            seed_domain_xyz_hi = np.full(3, np.nan, dtype=np.float64)
            seed_domain_r_min = float(domain_lo[0])
            seed_domain_r_max = float(domain_hi[0])
            if seed_domain_r_min < 0 or seed_domain_r_max <= seed_domain_r_min:
                raise ValueError(
                    f"Invalid spherical domain radii r_min={seed_domain_r_min}, r_max={seed_domain_r_max}."
                )
        logger.info("_prepare_seed_domain complete in %.2fs", float(time.perf_counter() - t_seed))
        valid_leaf_ids = np.flatnonzero(leaf_valid).astype(np.int64)
        n_valid_leaf = int(valid_leaf_ids.size)
        leaf_scales = np.full(leaf_slot_count, np.nan, dtype=np.float64)
        face_xyz = np.full((leaf_slot_count, 6, 4, 3), np.nan, dtype=np.float64)
        face_normal = np.full((leaf_slot_count, 6, 3), np.nan, dtype=np.float64)
        face_plane_d = np.full((leaf_slot_count, 6), np.nan, dtype=np.float64)
        face_edge_normal = np.full((leaf_slot_count, 6, 4, 3), np.nan, dtype=np.float64)
        face_edge_d = np.full((leaf_slot_count, 6, 4), np.nan, dtype=np.float64)
        face_patch_center = np.full((leaf_slot_count, 6, 4, 3), np.nan, dtype=np.float64)
        logger.info("build face trace cache...")
        t_geom = time.perf_counter()
        cell_xyz = tree.cell_points(valid_leaf_ids)
        leaf_centers = np.mean(cell_xyz, axis=1)
        leaf_scales[valid_leaf_ids] = np.max(np.linalg.norm(cell_xyz - leaf_centers[:, None, :], axis=2), axis=1)

        valid_face_xyz = cell_xyz[:, face_corners, :]
        valid_face_normal = np.cross(valid_face_xyz[:, :, 1] - valid_face_xyz[:, :, 0], valid_face_xyz[:, :, 2] - valid_face_xyz[:, :, 1])
        valid_face_center = np.mean(valid_face_xyz, axis=2)
        normal_flip = np.sum(valid_face_normal * (valid_face_center - leaf_centers[:, None, :]), axis=2) < 0
        valid_face_normal[normal_flip] *= -1
        valid_face_plane_d = np.sum(valid_face_normal * valid_face_xyz[:, :, 0, :], axis=2)

        valid_edge_p0 = valid_face_xyz
        valid_edge_p1 = np.roll(valid_face_xyz, -1, axis=2)
        valid_edge_vec = valid_edge_p1 - valid_edge_p0
        valid_face_edge_normal = np.cross(valid_edge_vec, valid_face_normal[:, :, None, :])
        valid_face_edge_ref = valid_face_center[:, :, None, :] - valid_edge_p0
        edge_flip = np.sum(valid_face_edge_normal * valid_face_edge_ref, axis=3) < 0
        valid_face_edge_normal[edge_flip] *= -1
        valid_face_edge_d = np.sum(valid_face_edge_normal * valid_edge_p0, axis=3)

        subface_u = np.array([1 / 4, 1 / 4, 3 / 4, 3 / 4], dtype=np.float64)
        subface_v = np.array([1 / 4, 3 / 4, 1 / 4, 3 / 4], dtype=np.float64)
        valid_face_patch_center = (
            ((1 - subface_u)[None, None, :, None] * (1 - subface_v)[None, None, :, None]) * valid_face_xyz[:, :, None, 0, :]
            + ((1 - subface_u)[None, None, :, None] * subface_v[None, None, :, None]) * valid_face_xyz[:, :, None, 1, :]
            + (subface_u[None, None, :, None] * subface_v[None, None, :, None]) * valid_face_xyz[:, :, None, 2, :]
            + (subface_u[None, None, :, None] * (1 - subface_v)[None, None, :, None]) * valid_face_xyz[:, :, None, 3, :]
        )

        face_xyz[valid_leaf_ids] = valid_face_xyz
        face_normal[valid_leaf_ids] = valid_face_normal
        face_plane_d[valid_leaf_ids] = valid_face_plane_d
        face_edge_normal[valid_leaf_ids] = valid_face_edge_normal
        face_edge_d[valid_leaf_ids] = valid_face_edge_d
        face_patch_center[valid_leaf_ids] = valid_face_patch_center
        logger.info("build face trace cache complete in %.2fs", float(time.perf_counter() - t_geom))
        logger.info("_pack_trace_state...")
        t_state = time.perf_counter()
        self._trace_state = (
            leaf_valid,
            face_xyz,
            face_normal,
            face_plane_d,
            face_edge_normal,
            face_edge_d,
            leaf_scales,
            face_patch_center,
            next_cell,
            n_polar,
            n_azimuth,
            tree_coord_code,
            cell_child,
            root_cell_ids,
            cell_parent,
            cell_bounds,
            domain_bounds,
            axis2_period,
            axis2_periodic,
            n_valid_leaf,
        )
        self._seed_domain_state = (
            tree_coord_code,
            seed_domain_xyz_lo,
            seed_domain_xyz_hi,
            seed_domain_r_min,
            seed_domain_r_max,
        )
        logger.info("_pack_trace_state complete in %.2fs", float(time.perf_counter() - t_state))
        logger.info("OctreeRayTracer.__init__ complete in %.2fs", float(time.perf_counter() - t0))

    def trace_kernel_state(self) -> tuple:
        """Return the raw arrays and scalars consumed by `trace_one_ray_kernel`."""
        return self._trace_state

    @staticmethod
    def _normalize_seed(seed_xyz: np.ndarray, ray_shape: tuple[int, ...]) -> np.ndarray:
        """Return one flat seed array matching one normalized ray shape."""
        seed = np.array(seed_xyz, dtype=np.float64, order="C")
        if seed.shape == (3,) and ray_shape == (1,):
            seed = seed.reshape(1, 3)
        if seed.shape != ray_shape + (3,):
            raise ValueError("seed_xyz must have the same shape as the ray origins.")
        if np.any(~(np.isnan(seed) | np.isfinite(seed))):
            raise ValueError("seed_xyz must contain only finite values or NaNs.")
        return seed.reshape(-1, 3)

    @staticmethod
    def _merge_seed_branches(
        backward_leaf_ids: np.ndarray,
        backward_t_enter: np.ndarray,
        backward_t_exit: np.ndarray,
        forward_leaf_ids: np.ndarray,
        forward_t_enter: np.ndarray,
        forward_t_exit: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Merge one backward and one forward seed trace into ascending original-ray order."""
        if backward_leaf_ids.size == 0:
            return forward_leaf_ids, forward_t_enter, forward_t_exit
        if forward_leaf_ids.size == 0:
            return backward_leaf_ids, backward_t_enter, backward_t_exit

        if backward_leaf_ids[-1] != forward_leaf_ids[0]:
            return (
                np.concatenate((backward_leaf_ids, forward_leaf_ids)),
                np.concatenate((backward_t_enter, forward_t_enter)),
                np.concatenate((backward_t_exit, forward_t_exit)),
            )

        join_tol = _EXIT_TOL * max(1, abs(float(backward_t_exit[-1])), abs(float(forward_t_enter[0])))
        if abs(float(backward_t_exit[-1]) - float(forward_t_enter[0])) > join_tol:
            return (
                np.concatenate((backward_leaf_ids, forward_leaf_ids)),
                np.concatenate((backward_t_enter, forward_t_enter)),
                np.concatenate((backward_t_exit, forward_t_exit)),
            )

        merged_leaf = np.concatenate((backward_leaf_ids[:-1], forward_leaf_ids))
        merged_enter = np.concatenate((backward_t_enter[:-1], np.array([backward_t_enter[-1]]), forward_t_enter[1:]))
        merged_exit = np.concatenate((backward_t_exit[:-1], np.array([forward_t_exit[0]]), forward_t_exit[1:]))
        return merged_leaf, merged_enter, merged_exit

    def trace(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        *,
        seed_xyz: np.ndarray | None = None,
        t_min: float = 0,
        t_max: float = np.inf,
    ) -> RaySegments:
        """Trace one batch of seeded rays and return packed per-cell intervals."""
        t_lo = float(t_min)
        t_hi = float(t_max)
        if not math.isfinite(t_lo):
            raise ValueError("t_min must be finite.")
        if math.isnan(t_hi):
            raise ValueError("t_max must not be NaN.")
        if t_hi < t_lo:
            raise ValueError("t_max must be greater than or equal to t_min.")

        o_flat, d_flat, ray_shape = _normalize_ray_arrays(origins, directions)
        seed_flat = (
            self.seed_domain(origins, directions, t_min=t_lo, t_max=t_hi).reshape(-1, 3)
            if seed_xyz is None
            else self._normalize_seed(seed_xyz, ray_shape)
        )
        clip_lo = max(0, t_lo)
        ray_offsets, cell_ids, t_enter, t_exit = _trace_rays_kernel(
            o_flat,
            d_flat,
            seed_flat,
            float(clip_lo),
            float(t_hi),
            *self._trace_state,
        )
        return RaySegments(
            ray_offsets=ray_offsets,
            cell_ids=cell_ids,
            t_enter=t_enter,
            t_exit=t_exit,
            ray_shape=ray_shape,
        )

    def seed_domain(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        *,
        t_min: float = 0,
        t_max: float = np.inf,
    ) -> np.ndarray:
        """Return one in-domain seed point per ray, or `NaN` for misses.

        Returned arrays have the same leading shape as the input ray arrays,
        with one trailing Cartesian axis of length 3.

        For `tree_coord="xyz"`, the octree domain is treated as one axis-aligned
        Cartesian box, and the seed is the midpoint of the clipped domain
        segment.

        For `tree_coord="rpa"`, this first pass assumes one full spherical shell
        with an opaque inner boundary. The seed lies on the front visible shell
        segment only, not a backside interval after crossing the central hole.
        """
        t_lo = float(t_min)
        t_hi = float(t_max)
        if not math.isfinite(t_lo):
            raise ValueError("t_min must be finite.")
        if math.isnan(t_hi):
            raise ValueError("t_max must not be NaN.")
        if t_hi < t_lo:
            raise ValueError("t_max must be greater than or equal to t_min.")

        o_flat, d_flat, shape = _normalize_ray_arrays(origins, directions)
        clip_lo = max(0, t_lo)
        (
            tree_coord_code,
            seed_domain_xyz_lo,
            seed_domain_xyz_hi,
            seed_domain_r_min,
            seed_domain_r_max,
        ) = self._seed_domain_state
        if tree_coord_code == int(_TREE_COORD_XYZ):
            seed_xyz = _seed_domain_xyz_kernel(
                o_flat,
                d_flat,
                float(clip_lo),
                float(t_hi),
                seed_domain_xyz_lo,
                seed_domain_xyz_hi,
            )
            return seed_xyz.reshape(shape + (3,))

        if tree_coord_code != int(_TREE_COORD_RPA):
            raise NotImplementedError(f"Unsupported tree_coord '{self.tree.tree_coord}' for seed_domain.")
        seed_xyz = _seed_domain_rpa_kernel(
            o_flat,
            d_flat,
            float(clip_lo),
            float(t_hi),
            float(seed_domain_r_min),
            float(seed_domain_r_max),
        )
        return seed_xyz.reshape(shape + (3,))

    def __str__(self) -> str:
        """Return a compact human-readable tracer summary."""
        return f"OctreeRayTracer(tree_coord={self.tree.tree_coord})"


def render_midpoint_image(
    interpolator,
    origins: np.ndarray,
    directions: np.ndarray,
    segments: RaySegments,
) -> np.ndarray:
    """Render one midpoint-sampled line integral over packed traced segments."""
    from .interpolator import OctreeInterpolator

    if not isinstance(interpolator, OctreeInterpolator):
        raise TypeError("render_midpoint_image requires one OctreeInterpolator.")

    o_flat, d_flat, ray_shape = _normalize_ray_arrays(origins, directions)
    if tuple(ray_shape) != tuple(segments.ray_shape):
        raise ValueError("segments.ray_shape must match the ray origin/direction shape.")

    n_rays = int(o_flat.shape[0])
    n_components = int(interpolator.n_components)
    accum = np.zeros((n_rays, n_components), dtype=np.float64)
    if segments.cell_ids.size == 0:
        out = accum.reshape(tuple(ray_shape) + interpolator.value_shape)
        if interpolator.value_shape:
            return out
        return out.reshape(tuple(ray_shape))

    counts = np.diff(segments.ray_offsets)
    ray_ids = np.repeat(np.arange(n_rays, dtype=np.int64), counts)
    mid_t = (1 / 2) * (segments.t_enter + segments.t_exit)
    mid_xyz = o_flat[ray_ids] + mid_t[:, None] * d_flat[ray_ids]
    samples = np.asarray(interpolator(mid_xyz, query_coord="xyz", log_outside_domain=False), dtype=np.float64)
    samples_2d = samples.reshape(samples.shape[0], -1)
    np.add.at(accum, ray_ids, samples_2d * segments.segment_length[:, None])

    out = accum.reshape(tuple(ray_shape) + interpolator.value_shape)
    if interpolator.value_shape:
        return out
    return out.reshape(tuple(ray_shape))
