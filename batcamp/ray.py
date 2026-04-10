#!/usr/bin/env python3
"""Ray-domain seeding helpers for octrees."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numba import njit

from .octree import Octree
from .octree import _contains_box
from .spherical import xyz_to_rpa_components

__all__ = ["OctreeRayTracer", "RaySegments", "render_midpoint_image", "trace_one_ray_kernel"]

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
_UNKNOWN_TRANSITION = np.int32(-2)
_NO_TRANSITION = np.int32(-1)
_EXIT_TOL = 1.0e-12
_SUBFACE_TOL = 1.0e-12
_BOUNDARY_SHIFT_FACTOR = 1.0e-6
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
    if np.any(np.linalg.norm(d_flat, axis=1) <= 0.0):
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


def _pack_leaf_key(depth: int, ijk: np.ndarray, axis_bases: np.ndarray) -> np.uint64:
    """Pack one `(depth, axis0, axis1, axis2)` leaf address into one sortable key."""
    key = np.uint64(depth)
    key = key * np.uint64(axis_bases[0]) + np.uint64(ijk[0])
    key = key * np.uint64(axis_bases[1]) + np.uint64(ijk[1])
    key = key * np.uint64(axis_bases[2]) + np.uint64(ijk[2])
    return key


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
    if disc < 0.0:
        return False, np.nan, np.nan
    root = math.sqrt(max(0.0, disc))
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
            if da == 0.0:
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
        seed_t = 0.5 * (enter[ray_id] + exit_[ray_id])
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
    tol = 1e-12 * max(r_max, 1.0)

    for ray_id in range(n_rays):
        hit_outer, t_outer0, t_outer1 = _sphere_interval_kernel(origins[ray_id], directions[ray_id], r_max)
        if not hit_outer:
            continue
        visible_start = max(0.0, t_outer0)
        visible_end = t_outer1
        if visible_end < visible_start:
            continue
        if r_min > 0.0:
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
            r_seed = 0.5 * (r_min + r_max)
            hit_seed, t_seed0, t_seed1 = _sphere_interval_kernel(origins[ray_id], directions[ray_id], r_seed)
            if hit_seed:
                if t_seed0 >= (visible_start - tol) and t_seed0 <= (visible_end + tol):
                    seed_t = t_seed0
                if t_seed1 >= (visible_start - tol) and t_seed1 <= (visible_end + tol):
                    if (not np.isfinite(seed_t)) or (t_seed1 < seed_t):
                        seed_t = t_seed1
        if not np.isfinite(seed_t):
            seed_t = 0.5 * (visible_start + visible_end)
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
    if not _contains_box(q, domain_bounds, axis2_period, axis2_periodic, 0.0):
        return -1

    current = int(hint_cell_id)
    while current >= 0 and not _contains_box(q, cell_bounds[current], axis2_period, axis2_periodic, 1.0e-10):
        current = int(cell_parent[current])

    if current < 0:
        for root_pos in range(int(root_cell_ids.shape[0])):
            root_cell_id = int(root_cell_ids[root_pos])
            if _contains_box(q, cell_bounds[root_cell_id], axis2_period, axis2_periodic, 1.0e-10):
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
            if _contains_box(q, cell_bounds[child_id], axis2_period, axis2_periodic, 1.0e-10):
                next_cell_id = child_id
                break
        if not has_child:
            return int(current)
        if next_cell_id < 0:
            return -1
        current = int(next_cell_id)


@njit(cache=True)
def _leaf_face_xyz(leaf_points: np.ndarray, face_corners: np.ndarray, face_id: int) -> np.ndarray:
    """Return one face's four corners for one leaf."""
    face_xyz = np.empty((4, 3), dtype=np.float64)
    for corner_id in range(4):
        face_xyz[corner_id] = leaf_points[face_corners[face_id, corner_id]]
    return face_xyz


@njit(cache=True)
def _face_normal_kernel(face_xyz: np.ndarray, cell_center_xyz: np.ndarray) -> np.ndarray:
    """Return one outward face normal for one planar face quad."""
    normal = _cross3(face_xyz[1] - face_xyz[0], face_xyz[2] - face_xyz[1])
    face_center = np.empty(3, dtype=np.float64)
    face_center[:] = 0.0
    for corner_id in range(4):
        face_center += face_xyz[corner_id]
    face_center *= 0.25
    if _dot3(normal, face_center - cell_center_xyz) < 0.0:
        normal = -normal
    return normal


@njit(cache=True)
def _point_inside_face_kernel(face_xyz: np.ndarray, normal_xyz: np.ndarray, point_xyz: np.ndarray, tol: float) -> bool:
    """Return whether one point lies inside one convex planar face quad."""
    all_nonnegative = True
    all_nonpositive = True
    edge_scale = 0.0
    for edge_id in range(4):
        p0 = face_xyz[edge_id]
        p1 = face_xyz[(edge_id + 1) % 4]
        edge = p1 - p0
        edge_len = math.sqrt(_dot3(edge, edge))
        if edge_len > edge_scale:
            edge_scale = edge_len
        sign = _dot3(_cross3(edge, point_xyz - p0), normal_xyz)
        if sign < -tol:
            all_nonnegative = False
        if sign > tol:
            all_nonpositive = False
    if all_nonnegative or all_nonpositive:
        return True

    edge_tol = max(tol, (2.0 * _BOUNDARY_SHIFT_FACTOR) * max(1.0, edge_scale))
    for edge_id in range(4):
        p0 = face_xyz[edge_id]
        p1 = face_xyz[(edge_id + 1) % 4]
        edge = p1 - p0
        edge_sq = _dot3(edge, edge)
        if edge_sq <= 0.0:
            closest_xyz = p0
        else:
            weight = _dot3(point_xyz - p0, edge) / edge_sq
            if weight < 0.0:
                weight = 0.0
            elif weight > 1.0:
                weight = 1.0
            closest_xyz = p0 + weight * edge
        if math.sqrt(_dot3(point_xyz - closest_xyz, point_xyz - closest_xyz)) <= edge_tol:
            return True
    return False


@njit(cache=True)
def _point_inside_cell_kernel(
    leaf_points: np.ndarray,
    leaf_center: np.ndarray,
    face_corners: np.ndarray,
    point_xyz: np.ndarray,
    tol: float,
) -> bool:
    """Return whether one point lies inside one convex planar-face cell."""
    for face_id in range(6):
        face_xyz = _leaf_face_xyz(leaf_points, face_corners, face_id)
        normal_xyz = _face_normal_kernel(face_xyz, leaf_center)
        if _dot3(normal_xyz, point_xyz - face_xyz[0]) > tol:
            return False
    return True


@njit(cache=True)
def _solve_cell_segment_kernel(
    leaf_points: np.ndarray,
    leaf_center: np.ndarray,
    leaf_scale: float,
    face_corners: np.ndarray,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    current_t: float,
    t_min: float,
) -> tuple[bool, float, float, int, int]:
    """Return one forward cell segment plus exit face/subface, or `(False, ...)` if unresolved."""
    point_tol = _EXIT_TOL * max(1.0, leaf_scale)
    inside_now = _point_inside_cell_kernel(
        leaf_points,
        leaf_center,
        face_corners,
        origin_xyz + current_t * direction_xyz,
        point_tol,
    )

    hit_t = np.empty(6, dtype=np.float64)
    hit_face = np.empty(6, dtype=np.int64)
    hit_xyz = np.empty((6, 3), dtype=np.float64)
    hit_normal = np.empty((6, 3), dtype=np.float64)
    hit_count = 0
    for face_id in range(6):
        face_xyz = _leaf_face_xyz(leaf_points, face_corners, face_id)
        normal_xyz = _face_normal_kernel(face_xyz, leaf_center)
        denom = _dot3(normal_xyz, direction_xyz)
        if abs(denom) <= _EXIT_TOL:
            continue
        t_hit = _dot3(normal_xyz, face_xyz[0] - origin_xyz) / denom
        face_hit_xyz = origin_xyz + t_hit * direction_xyz
        if _point_inside_face_kernel(face_xyz, normal_xyz, face_hit_xyz, point_tol):
            hit_t[hit_count] = t_hit
            hit_face[hit_count] = face_id
            hit_xyz[hit_count] = face_hit_xyz
            hit_normal[hit_count] = normal_xyz
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
            swap_xyz = hit_xyz[i].copy()
            hit_xyz[i] = hit_xyz[best]
            hit_xyz[best] = swap_xyz
            swap_normal = hit_normal[i].copy()
            hit_normal[i] = hit_normal[best]
            hit_normal[best] = swap_normal

    enter_tol = _EXIT_TOL * max(1.0, abs(current_t))
    segment_enter = current_t if inside_now else np.nan
    exit_t = np.nan
    exit_face = -1
    exit_xyz = np.full(3, np.nan, dtype=np.float64)
    exit_normal = np.full(3, np.nan, dtype=np.float64)
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
        exit_xyz = hit_xyz[hit_id]
        exit_normal = hit_normal[hit_id]
        break
    if not np.isfinite(segment_enter) or not np.isfinite(exit_t):
        return False, np.nan, np.nan, -1, -1
    if exit_t <= segment_enter:
        return False, np.nan, np.nan, -1, -1

    exit_tol = _EXIT_TOL * max(1.0, abs(exit_t))
    degenerate_count = 0
    min_face = exit_face
    for hit_id in range(hit_count):
        if abs(hit_t[hit_id] - exit_t) <= exit_tol:
            degenerate_count += 1
            if hit_face[hit_id] < min_face:
                min_face = int(hit_face[hit_id])
    exit_face = int(min_face)
    subface_id = -1
    if degenerate_count <= 1:
        subface_id = _face_exit_subface_kernel(leaf_points, face_corners, exit_xyz, exit_face, exit_normal)
    return True, segment_enter, exit_t, exit_face, subface_id


@njit(cache=True)
def _cell_segment_trace_kernel(
    leaf_points: np.ndarray,
    leaf_center: np.ndarray,
    leaf_scale: float,
    face_corners: np.ndarray,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    current_t: float,
    t_min: float,
) -> tuple[bool, float, float, int, int]:
    """Return one exact forward cell segment with one boundary-shift retry."""
    ok, segment_enter, exit_t, exit_face, subface_id = _solve_cell_segment_kernel(
        leaf_points,
        leaf_center,
        leaf_scale,
        face_corners,
        origin_xyz,
        direction_xyz,
        current_t,
        t_min,
    )
    if ok:
        return True, segment_enter, exit_t, exit_face, subface_id

    direction_norm = math.sqrt(_dot3(direction_xyz, direction_xyz))
    if direction_norm <= 0.0:
        return False, np.nan, np.nan, -1, -1
    boundary_shift = (_BOUNDARY_SHIFT_FACTOR * max(1.0, leaf_scale)) / direction_norm
    ok, segment_enter, exit_t, exit_face, subface_id = _solve_cell_segment_kernel(
        leaf_points,
        leaf_center,
        leaf_scale,
        face_corners,
        origin_xyz + boundary_shift * direction_xyz,
        direction_xyz,
        current_t,
        t_min,
    )
    if not ok:
        return False, np.nan, np.nan, -1, -1
    return True, segment_enter + boundary_shift, exit_t + boundary_shift, exit_face, subface_id


@njit(cache=True)
def _split_halfspace_bit(
    split0: np.ndarray,
    split1: np.ndarray,
    low_ref: np.ndarray,
    high_ref: np.ndarray,
    hit_xyz: np.ndarray,
    normal_xyz: np.ndarray,
) -> int:
    """Return one side bit for one face split, or `-1` for degenerate hits."""
    split_direction = split1 - split0
    low_sign = _dot3(_cross3(split_direction, low_ref - split0), normal_xyz)
    high_sign = _dot3(_cross3(split_direction, high_ref - split0), normal_xyz)
    hit_sign = _dot3(_cross3(split_direction, hit_xyz - split0), normal_xyz)
    if abs(hit_sign) <= _SUBFACE_TOL:
        return -1
    if low_sign == 0.0 or high_sign == 0.0 or low_sign * high_sign >= 0.0:
        return -1
    if hit_sign * low_sign > 0.0:
        return 0
    return 1


@njit(cache=True)
def _face_exit_subface_kernel(
    leaf_points: np.ndarray,
    face_corners: np.ndarray,
    exit_xyz: np.ndarray,
    face_id: int,
    normal_xyz: np.ndarray,
) -> int:
    """Return the crossed face quadrant, or `-1` for degenerate subface exits."""
    c00 = leaf_points[face_corners[face_id, 0]]
    c01 = leaf_points[face_corners[face_id, 1]]
    c11 = leaf_points[face_corners[face_id, 2]]
    c10 = leaf_points[face_corners[face_id, 3]]

    first_split0 = 0.5 * (c00 + c10)
    first_split1 = 0.5 * (c01 + c11)
    second_split0 = 0.5 * (c00 + c01)
    second_split1 = 0.5 * (c10 + c11)

    first_bit = _split_halfspace_bit(first_split0, first_split1, c00, c10, exit_xyz, normal_xyz)
    if first_bit < 0:
        return -1
    second_bit = _split_halfspace_bit(second_split0, second_split1, c00, c01, exit_xyz, normal_xyz)
    if second_bit < 0:
        return -1
    return 2 * first_bit + second_bit


@njit(cache=True)
def _touching_neighbor_leaves_kernel(
    leaf_id: int,
    point_xyz: np.ndarray,
    face_corners: np.ndarray,
    leaf_points: np.ndarray,
    leaf_centers: np.ndarray,
    leaf_scales: np.ndarray,
    next_leaf: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Return unique neighboring leaves whose cells touch one boundary point of one leaf."""
    neighbors = np.full(24, -1, dtype=np.int64)
    count = 0
    point_tol = _EXIT_TOL * max(1.0, leaf_scales[leaf_id])
    for face_id in range(6):
        face_xyz = _leaf_face_xyz(leaf_points[leaf_id], face_corners, face_id)
        normal_xyz = _face_normal_kernel(face_xyz, leaf_centers[leaf_id])
        face_distance = _dot3(normal_xyz, point_xyz - face_xyz[0])
        if abs(face_distance) > point_tol:
            continue
        for subface_id in range(4):
            candidate = int(next_leaf[leaf_id, face_id, subface_id])
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
    radial_scale = max(math.sqrt(_dot3(point_xyz, point_xyz)), 1.0)
    axis_tol = _BOUNDARY_SHIFT_FACTOR * radial_scale
    if radial_xy > axis_tol:
        return candidates, count

    direction_norm = math.sqrt(_dot3(direction_xyz, direction_xyz))
    if direction_norm > 0.0:
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
    polar_center = 0.5 * (math.pi / float(n_polar))
    if point_xyz[2] < 0.0:
        polar_center = math.pi - polar_center
    sin_polar = math.sin(polar_center)
    cos_polar = math.cos(polar_center)
    azimuth_width = (2.0 * math.pi) / float(n_azimuth)
    for azimuth_id in range(n_azimuth):
        azimuth_center = (float(azimuth_id) + 0.5) * azimuth_width
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
def _best_continuation_leaf_kernel(
    candidate_leaf_ids: np.ndarray,
    candidate_count: int,
    point_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    exclude_leaf_id: int,
    face_corners: np.ndarray,
    leaf_points: np.ndarray,
    leaf_centers: np.ndarray,
    leaf_scales: np.ndarray,
) -> int:
    """Return the best directly valid continuation leaf from one small candidate set."""
    if candidate_count <= 0:
        return -1

    valid_enter = np.empty(candidate_count, dtype=np.float64)
    valid_length = np.empty(candidate_count, dtype=np.float64)
    valid_exit = np.empty(candidate_count, dtype=np.float64)
    valid_leaf_ids = np.empty(candidate_count, dtype=np.int64)
    valid_count = 0
    max_length = 0.0

    for idx in range(candidate_count):
        leaf_id = int(candidate_leaf_ids[idx])
        if leaf_id < 0 or leaf_id == exclude_leaf_id:
            continue
        ok, segment_enter, segment_exit, _, _ = _cell_segment_trace_kernel(
            leaf_points[leaf_id],
            leaf_centers[leaf_id],
            leaf_scales[leaf_id],
            face_corners,
            point_xyz,
            direction_xyz,
            0.0,
            0.0,
        )
        if not ok:
            continue
        length = segment_exit - segment_enter
        valid_enter[valid_count] = segment_enter
        valid_length[valid_count] = length
        valid_exit[valid_count] = segment_exit
        valid_leaf_ids[valid_count] = leaf_id
        valid_count += 1
        if length > max_length:
            max_length = length

    if valid_count == 0:
        return -1

    min_length = 1.0e-6 * max_length
    use_noncollapsed = False
    for idx in range(valid_count):
        if valid_length[idx] > min_length:
            use_noncollapsed = True
            break

    best_idx = -1
    for idx in range(valid_count):
        if use_noncollapsed and valid_length[idx] <= min_length:
            continue
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
def _boundary_continuation_leaf_kernel(
    initial_leaf_ids: np.ndarray,
    initial_count: int,
    point_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    exclude_leaf_id: int,
    face_corners: np.ndarray,
    leaf_valid: np.ndarray,
    leaf_points: np.ndarray,
    leaf_centers: np.ndarray,
    leaf_scales: np.ndarray,
    next_leaf: np.ndarray,
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
    max_length = 0.0

    while head < tail:
        leaf_id = int(queue[head])
        head += 1
        if leaf_id < 0 or visited[leaf_id]:
            continue
        visited[leaf_id] = True
        ok, segment_enter, segment_exit, _, _ = _cell_segment_trace_kernel(
            leaf_points[leaf_id],
            leaf_centers[leaf_id],
            leaf_scales[leaf_id],
            face_corners,
            point_xyz,
            direction_xyz,
            0.0,
            0.0,
        )
        if ok:
            if leaf_id != exclude_leaf_id:
                length = segment_exit - segment_enter
                valid_enter[valid_count] = segment_enter
                valid_length[valid_count] = length
                valid_exit[valid_count] = segment_exit
                valid_leaf_ids[valid_count] = leaf_id
                valid_count += 1
                if length > max_length:
                    max_length = length
                continue
        touching, touching_count = _touching_neighbor_leaves_kernel(
            leaf_id,
            point_xyz,
            face_corners,
            leaf_points,
            leaf_centers,
            leaf_scales,
            next_leaf,
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
                leaf_points[leaf_id],
                leaf_centers[leaf_id],
                leaf_scales[leaf_id],
                face_corners,
                point_xyz,
                direction_xyz,
                0.0,
                0.0,
            )
            if ok:
                length = segment_exit - segment_enter
                valid_enter[valid_count] = segment_enter
                valid_length[valid_count] = length
                valid_exit[valid_count] = segment_exit
                valid_leaf_ids[valid_count] = leaf_id
                valid_count += 1
                if length > max_length:
                    max_length = length

    if valid_count == 0:
        return -1

    min_length = 1.0e-6 * max_length
    use_noncollapsed = False
    for idx in range(valid_count):
        if valid_length[idx] > min_length:
            use_noncollapsed = True
            break

    best_idx = -1
    for idx in range(valid_count):
        if use_noncollapsed and valid_length[idx] <= min_length:
            continue
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
    face_corners: np.ndarray,
    leaf_valid: np.ndarray,
    leaf_points: np.ndarray,
    leaf_centers: np.ndarray,
    leaf_scales: np.ndarray,
    next_leaf: np.ndarray,
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
    initial_leaf_ids = np.full(50 + n_azimuth, -1, dtype=np.int64)
    count = 0

    if seed_leaf_id >= 0:
        count = _append_unique_leaf_id(initial_leaf_ids, count, int(seed_leaf_id))
        for face_id in range(6):
            for subface_id in range(4):
                count = _append_unique_leaf_id(
                    initial_leaf_ids,
                    count,
                    int(next_leaf[seed_leaf_id, face_id, subface_id]),
                )

        touching_leaf_ids, touching_count = _touching_neighbor_leaves_kernel(
            int(seed_leaf_id),
            seed_xyz,
            face_corners,
            leaf_points,
            leaf_centers,
            leaf_scales,
            next_leaf,
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
        face_corners,
        leaf_valid,
        leaf_points,
        leaf_centers,
        leaf_scales,
        next_leaf,
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
    leaf_points: np.ndarray,
    leaf_centers: np.ndarray,
    leaf_scales: np.ndarray,
    face_corners: np.ndarray,
    next_leaf: np.ndarray,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk one ray across neighboring leaves using exact cell intersections and face/subface continuation."""
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
        ok, segment_enter, exit_t, exit_face, subface_id = _cell_segment_trace_kernel(
            leaf_points[current_leaf],
            leaf_centers[current_leaf],
            leaf_scales[current_leaf],
            face_corners,
            origin_xyz,
            direction_xyz,
            current_t,
            t_min,
        )
        if not ok:
            raise ValueError("Failed to resolve a forward cell segment from the current leaf.")
        if segment_enter >= t_max:
            break
        segment_exit = exit_t if exit_t < t_max else t_max
        leaf_ids.append(current_leaf)
        t_enter_list.append(segment_enter)
        t_exit_list.append(segment_exit)
        if exit_t >= t_max:
            break
        exit_xyz = origin_xyz + exit_t * direction_xyz
        next_leaf_id = -1
        if subface_id >= 0:
            candidate = int(next_leaf[current_leaf, exit_face, subface_id])
            if candidate >= 0:
                candidate_ok, _, _, _, _ = _cell_segment_trace_kernel(
                    leaf_points[candidate],
                    leaf_centers[candidate],
                    leaf_scales[candidate],
                    face_corners,
                    exit_xyz,
                    direction_xyz,
                    0.0,
                    0.0,
                )
                if candidate_ok:
                    next_leaf_id = candidate
        if next_leaf_id < 0:
            initial_leaf_ids, initial_count = _touching_neighbor_leaves_kernel(
                current_leaf,
                exit_xyz,
                face_corners,
                leaf_points,
                leaf_centers,
                leaf_scales,
                next_leaf,
            )
            next_leaf_id = _best_continuation_leaf_kernel(
                initial_leaf_ids,
                initial_count,
                exit_xyz,
                direction_xyz,
                current_leaf,
                face_corners,
                leaf_points,
                leaf_centers,
                leaf_scales,
            )
        if next_leaf_id < 0:
            next_leaf_id = _boundary_continuation_leaf_kernel(
                initial_leaf_ids,
                initial_count,
                exit_xyz,
                direction_xyz,
                current_leaf,
                face_corners,
                leaf_valid,
                leaf_points,
                leaf_centers,
                leaf_scales,
                next_leaf,
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
        if next_leaf_id < 0:
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
        leaf_points,
        leaf_centers,
        leaf_scales,
        face_corners,
        next_leaf,
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
        leaf_points,
        leaf_centers,
        leaf_scales,
        face_corners,
        next_leaf,
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


@njit(cache=True)
def _trace_rays_kernel(
    origins: np.ndarray,
    directions: np.ndarray,
    seed_xyz: np.ndarray,
    clip_lo: float,
    t_hi: float,
    leaf_valid: np.ndarray,
    leaf_points: np.ndarray,
    leaf_centers: np.ndarray,
    leaf_scales: np.ndarray,
    face_corners: np.ndarray,
    next_leaf: np.ndarray,
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
            face_corners,
            leaf_valid,
            leaf_points,
            leaf_centers,
            leaf_scales,
            next_leaf,
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
            face_corners,
            leaf_valid,
            leaf_points,
            leaf_centers,
            leaf_scales,
            next_leaf,
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

        backward_limit = max(0.0, t_seed - clip_lo)
        if backward_limit > 0.0 and backward_leaf >= 0:
            back_leaf_ids_local, back_enter_local, back_exit_local = _trace_one_ray_kernel(
                int(backward_leaf),
                seed,
                -direction,
                0.0,
                float(backward_limit),
                leaf_valid,
                leaf_points,
                leaf_centers,
                leaf_scales,
                face_corners,
                next_leaf,
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
        else:
            back_leaf_ids_local = np.empty(0, dtype=np.int64)
            back_enter_local = np.empty(0, dtype=np.float64)
            back_exit_local = np.empty(0, dtype=np.float64)

        forward_limit = max(0.0, t_hi - t_seed)
        if forward_limit > 0.0 and forward_leaf >= 0:
            forward_leaf_ids_local, forward_enter_local, forward_exit_local = _trace_one_ray_kernel(
                int(forward_leaf),
                seed,
                direction,
                0.0,
                float(forward_limit),
                leaf_valid,
                leaf_points,
                leaf_centers,
                leaf_scales,
                face_corners,
                next_leaf,
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
        else:
            forward_leaf_ids_local = np.empty(0, dtype=np.int64)
            forward_enter_local = np.empty(0, dtype=np.float64)
            forward_exit_local = np.empty(0, dtype=np.float64)

        back_count = int(back_leaf_ids_local.shape[0])
        forward_count = int(forward_leaf_ids_local.shape[0])
        joined = False
        if back_count > 0 and forward_count > 0 and back_leaf_ids_local[0] == forward_leaf_ids_local[0]:
            back_last_exit = t_seed - back_enter_local[0]
            forward_first_enter = t_seed + forward_enter_local[0]
            join_tol = _EXIT_TOL * max(1.0, abs(back_last_exit), abs(forward_first_enter))
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
        self.tree = tree
        resolved_tree_coord = str(tree.tree_coord)
        if resolved_tree_coord == "xyz":
            self._face_corners = _build_face_corner_order(_XYZ_CORNER_BITS)
            self._tree_coord_code = int(_TREE_COORD_XYZ)
            self._axis2_period = 0.0
            self._axis2_periodic = False
        elif resolved_tree_coord == "rpa":
            self._face_corners = _build_face_corner_order(_RPA_CORNER_BITS)
            self._tree_coord_code = int(_TREE_COORD_RPA)
            self._axis2_period = 2.0 * math.pi
            self._axis2_periodic = True
        else:
            raise NotImplementedError(f"Unsupported tree_coord '{tree.tree_coord}' for OctreeRayTracer.")
        self._leaf_slot_count = int(tree.corners.shape[0])
        self._leaf_valid = tree.cell_levels >= 0
        self._axis_bases = np.asarray(tree.leaf_shape, dtype=np.uint64) + np.uint64(1)
        self._max_level = int(tree.max_level)
        self._n_polar = int(tree.leaf_shape[1])
        self._n_azimuth = int(tree.leaf_shape[2])
        self._cell_child = np.asarray(tree.cell_child, dtype=np.int64)
        self._cell_parent = np.asarray(tree.cell_parent, dtype=np.int64)
        self._root_cell_ids = np.flatnonzero(self._cell_parent < 0).astype(np.int64)
        self._cell_bounds = np.asarray(tree.cell_bounds, dtype=np.float64)
        domain_lo, domain_hi = tree.domain_bounds(coord=tree.tree_coord)
        self._domain_bounds = np.empty((3, 2), dtype=np.float64)
        self._domain_bounds[:, 0] = domain_lo
        self._domain_bounds[:, 1] = domain_hi - domain_lo
        if self._tree_coord_code == int(_TREE_COORD_XYZ):
            self._seed_domain_xyz_lo = np.asarray(domain_lo, dtype=np.float64)
            self._seed_domain_xyz_hi = np.asarray(domain_hi, dtype=np.float64)
            self._seed_domain_r_min = np.nan
            self._seed_domain_r_max = np.nan
        else:
            if not np.isclose(float(domain_lo[1]), 0.0, atol=1e-12, rtol=0.0):
                raise NotImplementedError("seed_domain for rpa currently requires polar_min == 0.")
            if not np.isclose(float(domain_hi[1]), math.pi, atol=1e-12, rtol=0.0):
                raise NotImplementedError("seed_domain for rpa currently requires polar_max == pi.")
            if not np.isclose(float(domain_lo[2]), 0.0, atol=1e-12, rtol=0.0):
                raise NotImplementedError("seed_domain for rpa currently requires azimuth_start == 0.")
            if not np.isclose(float(domain_hi[2] - domain_lo[2]), 2.0 * math.pi, atol=1e-12, rtol=0.0):
                raise NotImplementedError("seed_domain for rpa currently requires full 2pi azimuth coverage.")
            self._seed_domain_xyz_lo = np.full(3, np.nan, dtype=np.float64)
            self._seed_domain_xyz_hi = np.full(3, np.nan, dtype=np.float64)
            self._seed_domain_r_min = float(domain_lo[0])
            self._seed_domain_r_max = float(domain_hi[0])
            if self._seed_domain_r_min < 0.0 or self._seed_domain_r_max <= self._seed_domain_r_min:
                raise ValueError(
                    f"Invalid spherical domain radii r_min={self._seed_domain_r_min}, r_max={self._seed_domain_r_max}."
                )
        valid_leaf_ids = np.flatnonzero(self._leaf_valid).astype(np.int64)
        self._n_valid_leaf = int(valid_leaf_ids.size)
        valid_depth = tree.cell_depth[valid_leaf_ids]
        valid_ijk = tree.cell_ijk[valid_leaf_ids]
        valid_keys = np.array(
            [_pack_leaf_key(int(valid_depth[i]), valid_ijk[i], self._axis_bases) for i in range(valid_leaf_ids.size)],
            dtype=np.uint64,
        )
        order = np.argsort(valid_keys)
        self._leaf_keys_sorted = valid_keys[order]
        self._leaf_ids_sorted = valid_leaf_ids[order].astype(np.int32, copy=False)
        self._next_leaf = np.full((self._leaf_slot_count, 6, 4), _UNKNOWN_TRANSITION, dtype=np.int32)
        for leaf_id in valid_leaf_ids:
            for face_id in range(6):
                for subface_id in range(4):
                    self._resolve_transition(int(leaf_id), face_id, subface_id)
        self._trace_next_leaf = np.array(self._next_leaf, copy=True)
        self._next_leaf.fill(_UNKNOWN_TRANSITION)
        self._leaf_points = np.full((self._leaf_slot_count, 8, 3), np.nan, dtype=np.float64)
        self._leaf_centers = np.full((self._leaf_slot_count, 3), np.nan, dtype=np.float64)
        self._leaf_scales = np.full(self._leaf_slot_count, np.nan, dtype=np.float64)
        for leaf_id in valid_leaf_ids:
            cell_xyz = np.asarray(tree.cell_points(int(leaf_id)), dtype=np.float64)
            center_xyz = np.mean(cell_xyz, axis=0)
            self._leaf_points[leaf_id] = cell_xyz
            self._leaf_centers[leaf_id] = center_xyz
            self._leaf_scales[leaf_id] = float(np.max(np.linalg.norm(cell_xyz - center_xyz, axis=1)))
        self._trace_state = (
            self._leaf_valid,
            self._leaf_points,
            self._leaf_centers,
            self._leaf_scales,
            self._face_corners,
            self._trace_next_leaf,
            int(self._n_polar),
            int(self._n_azimuth),
            int(self._tree_coord_code),
            self._cell_child,
            self._root_cell_ids,
            self._cell_parent,
            self._cell_bounds,
            self._domain_bounds,
            float(self._axis2_period),
            bool(self._axis2_periodic),
            int(self._n_valid_leaf),
        )

    @staticmethod
    def _normalize_rays(origins: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
        """Return flat finite ray arrays plus the leading broadcast shape."""
        return _normalize_ray_arrays(origins, directions)

    @staticmethod
    def _sphere_interval(origin_xyz: np.ndarray, direction_xyz: np.ndarray, radius: float) -> tuple[bool, float, float]:
        """Return one line-parameter interval inside one sphere centered at the origin."""
        return _sphere_interval_kernel(
            np.asarray(origin_xyz, dtype=np.float64),
            np.asarray(direction_xyz, dtype=np.float64),
            float(radius),
        )

    def _lookup_leaf(self, depth: int, ijk: np.ndarray) -> int:
        """Return one exact leaf-slot id for one `(depth, ijk)` address, or `-1`."""
        if depth < 0 or depth > self._max_level:
            return -1
        if np.any(ijk < 0):
            return -1
        key = _pack_leaf_key(int(depth), ijk, self._axis_bases)
        pos = int(np.searchsorted(self._leaf_keys_sorted, key))
        if pos >= int(self._leaf_keys_sorted.size) or self._leaf_keys_sorted[pos] != key:
            return -1
        return int(self._leaf_ids_sorted[pos])

    def _resolve_transition(self, leaf_id: int, face_id: int, subface_id: int) -> int:
        """Return the neighboring leaf across one face/subface, or `-1` at the domain boundary.

        Face order is `(xmin, xmax, ymin, ymax, zmin, zmax)`.

        Subfaces always split the crossed face into a `2 x 2` grid. Their bits
        are ordered along the two tangential axes of that face:

        - `x*` faces: `(y_bit, z_bit)`
        - `y*` faces: `(x_bit, z_bit)`
        - `z*` faces: `(x_bit, y_bit)`

        with `subface_id = 2 * first_bit + second_bit`.
        """
        leaf = int(leaf_id)
        face = int(face_id)
        subface = int(subface_id)
        if leaf < 0 or leaf >= self._leaf_slot_count or not bool(self._leaf_valid[leaf]):
            raise ValueError("leaf_id must reference one valid leaf slot.")
        if face < 0 or face >= 6:
            raise ValueError("face_id must be in the range [0, 5].")
        if subface < 0 or subface >= 4:
            raise ValueError("subface_id must be in the range [0, 3].")

        cached = int(self._next_leaf[leaf, face, subface])
        if cached != int(_UNKNOWN_TRANSITION):
            return cached

        axis = int(_FACE_AXIS[face])
        side = int(_FACE_SIDE[face])
        tangential_axes = _FACE_TANGENTIAL_AXES[face]
        delta = -1 if side == 0 else 1
        depth = int(self.tree.cell_depth[leaf])
        ijk = self.tree.cell_ijk[leaf]

        next_leaf = _NO_TRANSITION

        same_ijk = np.array(ijk, copy=True)
        same_ijk[axis] += delta
        next_leaf = np.int32(self._lookup_leaf(depth, same_ijk))

        if next_leaf < 0 and depth > 0:
            coarse_ijk = np.right_shift(ijk, 1).astype(np.int64, copy=False)
            coarse_ijk = np.array(coarse_ijk, copy=True)
            coarse_ijk[axis] = (int(ijk[axis]) + delta) >> 1
            next_leaf = np.int32(self._lookup_leaf(depth - 1, coarse_ijk))

        if next_leaf < 0 and depth < self._max_level:
            fine_ijk = np.left_shift(ijk, 1).astype(np.int64, copy=False)
            fine_ijk = np.array(fine_ijk, copy=True)
            fine_ijk[int(tangential_axes[0])] += (subface >> 1) & 1
            fine_ijk[int(tangential_axes[1])] += subface & 1
            fine_ijk[axis] = 2 * int(ijk[axis]) + (-1 if side == 0 else 2)
            next_leaf = np.int32(self._lookup_leaf(depth + 1, fine_ijk))

        self._next_leaf[leaf, face, subface] = np.int32(next_leaf)
        return int(next_leaf)

    def _cell_faces(self, leaf_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return one leaf's explicit corners and geometric center in physical `xyz`."""
        cell_xyz = np.asarray(self.tree.cell_points(int(leaf_id)), dtype=np.float64)
        return cell_xyz, np.mean(cell_xyz, axis=0)

    @staticmethod
    def _face_normal(face_xyz: np.ndarray, cell_center_xyz: np.ndarray) -> np.ndarray:
        """Return one outward face normal for one planar face quad."""
        normal = _cross3(face_xyz[1] - face_xyz[0], face_xyz[2] - face_xyz[1])
        if _dot3(normal, np.mean(face_xyz, axis=0) - cell_center_xyz) < 0.0:
            normal = -normal
        return normal

    @staticmethod
    def _point_inside_face(face_xyz: np.ndarray, normal_xyz: np.ndarray, point_xyz: np.ndarray, tol: float) -> bool:
        """Return whether one point lies inside one convex planar face quad."""
        signs = np.empty(4, dtype=np.float64)
        for edge_id in range(4):
            p0 = face_xyz[edge_id]
            p1 = face_xyz[(edge_id + 1) % 4]
            signs[edge_id] = _dot3(_cross3(p1 - p0, point_xyz - p0), normal_xyz)
        if bool(np.all(signs >= -tol) or np.all(signs <= tol)):
            return True

        edge_scale = float(
            np.max(np.linalg.norm(np.roll(face_xyz, -1, axis=0) - face_xyz, axis=1))
        )
        edge_tol = max(float(tol), (2.0 * _BOUNDARY_SHIFT_FACTOR) * max(1.0, edge_scale))
        for edge_id in range(4):
            p0 = face_xyz[edge_id]
            p1 = face_xyz[(edge_id + 1) % 4]
            edge = p1 - p0
            edge_sq = _dot3(edge, edge)
            if edge_sq <= 0.0:
                distance = float(np.linalg.norm(point_xyz - p0))
            else:
                weight = np.clip(_dot3(point_xyz - p0, edge) / edge_sq, 0.0, 1.0)
                closest_xyz = p0 + weight * edge
                distance = float(np.linalg.norm(point_xyz - closest_xyz))
            if distance <= edge_tol:
                return True
        return False

    def _point_inside_cell(self, leaf_id: int, point_xyz: np.ndarray, tol: float) -> bool:
        """Return whether one point lies inside one convex planar-face cell."""
        cell_xyz, cell_center = self._cell_faces(leaf_id)
        for face_id in range(6):
            face_xyz = cell_xyz[self._face_corners[face_id]]
            normal_xyz = self._face_normal(face_xyz, cell_center)
            if _dot3(normal_xyz, point_xyz - face_xyz[0]) > tol:
                return False
        return True

    def _face_exit_subface(self, leaf_id: int, exit_xyz: np.ndarray, face_id: int, normal_xyz: np.ndarray) -> int:
        """Return the crossed face quadrant using the face's two native tangential axes."""
        cell_xyz, _ = self._cell_faces(leaf_id)
        c00, c01, c11, c10 = cell_xyz[self._face_corners[int(face_id)]]

        first_split0 = 0.5 * (c00 + c10)
        first_split1 = 0.5 * (c01 + c11)
        second_split0 = 0.5 * (c00 + c01)
        second_split1 = 0.5 * (c10 + c11)

        def classify(split0: np.ndarray, split1: np.ndarray, low_ref: np.ndarray, high_ref: np.ndarray) -> int:
            split_direction = split1 - split0
            low_sign = _dot3(_cross3(split_direction, low_ref - split0), normal_xyz)
            high_sign = _dot3(_cross3(split_direction, high_ref - split0), normal_xyz)
            hit_sign = _dot3(_cross3(split_direction, exit_xyz - split0), normal_xyz)
            if abs(hit_sign) <= _SUBFACE_TOL:
                raise ValueError("Degenerate face exit on a subface boundary is not supported yet.")
            if low_sign == 0.0 or high_sign == 0.0 or low_sign * high_sign >= 0.0:
                raise ValueError("Failed to construct a valid face subface split.")
            return 0 if hit_sign * low_sign > 0.0 else 1

        first_bit = classify(first_split0, first_split1, c00, c10)
        second_bit = classify(second_split0, second_split1, c00, c01)
        return 2 * first_bit + second_bit

    def _touching_neighbor_leaves(self, leaf_id: int, point_xyz: np.ndarray) -> list[int]:
        """Return unique neighboring leaves whose cells touch one boundary point of one leaf."""
        cell_xyz, cell_center = self._cell_faces(leaf_id)
        point_tol = _EXIT_TOL * max(1.0, float(np.max(np.linalg.norm(cell_xyz - cell_center, axis=1))))
        neighbors: list[int] = []
        for face_id in range(6):
            face_xyz = cell_xyz[self._face_corners[face_id]]
            normal_xyz = self._face_normal(face_xyz, cell_center)
            face_distance = _dot3(normal_xyz, point_xyz - face_xyz[0])
            if abs(face_distance) > point_tol:
                continue
            for subface_id in range(4):
                next_leaf = int(self._resolve_transition(leaf_id, face_id, subface_id))
                if next_leaf >= 0 and next_leaf not in neighbors:
                    neighbors.append(next_leaf)
        return neighbors

    def _topological_neighbor_leaves(self, leaf_id: int) -> list[int]:
        """Return unique leaves adjacent to one leaf across any face transition."""
        neighbors: list[int] = []
        for face_id in range(6):
            for subface_id in range(4):
                next_leaf = int(self._resolve_transition(int(leaf_id), face_id, subface_id))
                if next_leaf >= 0 and next_leaf not in neighbors:
                    neighbors.append(next_leaf)
        return neighbors

    def _boundary_continuation_leaf(
        self,
        initial_leaf_ids: list[int],
        point_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        *,
        exclude_leaf_ids: tuple[int, ...] = (),
    ) -> int:
        """Return one deterministic continuation leaf from one boundary point, or `-1`."""
        queue = [int(v) for v in initial_leaf_ids]
        visited: set[int] = set()
        valid: list[tuple[float, float, int, float]] = []

        while queue:
            leaf_id = int(queue.pop(0))
            if leaf_id < 0 or leaf_id in visited:
                continue
            visited.add(leaf_id)
            try:
                segment_enter, segment_exit, _, _ = self._cell_segment(
                    leaf_id,
                    point_xyz,
                    direction_xyz,
                    current_t=0.0,
                    t_min=0.0,
                )
            except ValueError as exc:
                message = str(exc)
                if message not in {
                    "The requested leaf does not intersect the ray.",
                    "Failed to resolve a forward cell segment from the current leaf.",
                    "Degenerate cell exit across an edge or corner is not supported yet.",
                    "Degenerate face exit on a subface boundary is not supported yet.",
                }:
                    raise
            else:
                if leaf_id not in exclude_leaf_ids:
                    valid.append(
                        (
                            float(segment_enter),
                            float(segment_exit - segment_enter),
                            int(leaf_id),
                            float(segment_exit),
                        )
                    )
                    continue
            for next_leaf in self._touching_neighbor_leaves(leaf_id, point_xyz):
                if next_leaf not in visited:
                    queue.append(int(next_leaf))

        if self._tree_coord_code == int(_TREE_COORD_RPA):
            for candidate_leaf in self._rpa_axis_candidate_leaves(point_xyz, direction_xyz):
                if candidate_leaf in visited or candidate_leaf in exclude_leaf_ids:
                    continue
                try:
                    segment_enter, segment_exit, _, _ = self._cell_segment(
                        candidate_leaf,
                        point_xyz,
                        direction_xyz,
                        current_t=0.0,
                        t_min=0.0,
                    )
                except ValueError as exc:
                    message = str(exc)
                    if message not in {
                        "The requested leaf does not intersect the ray.",
                        "Failed to resolve a forward cell segment from the current leaf.",
                        "Degenerate cell exit across an edge or corner is not supported yet.",
                        "Degenerate face exit on a subface boundary is not supported yet.",
                    }:
                        raise
                else:
                    valid.append(
                        (
                            float(segment_enter),
                            float(segment_exit - segment_enter),
                            int(candidate_leaf),
                            float(segment_exit),
                        )
                    )
        if not valid:
            return -1
        max_length = max(item[1] for item in valid)
        min_length = 1.0e-6 * max_length
        noncollapsed = [item for item in valid if item[1] > min_length]
        if noncollapsed:
            valid = noncollapsed
        valid.sort(key=lambda item: (item[0], -item[1], -item[3], item[2]))
        return int(valid[0][2])

    def _rpa_axis_candidate_leaves(self, point_xyz: np.ndarray, direction_xyz: np.ndarray | None = None) -> list[int]:
        """Return one finite set of native-sector candidate leaves for one spherical axis point."""
        x = float(point_xyz[0])
        y = float(point_xyz[1])
        z = float(point_xyz[2])
        radial_xy = math.hypot(x, y)
        radial_scale = max(float(np.linalg.norm(point_xyz)), 1.0)
        axis_tol = _BOUNDARY_SHIFT_FACTOR * radial_scale
        if radial_xy > axis_tol:
            return []

        unique_leaf: list[int] = []
        if direction_xyz is not None:
            direction_norm = float(np.linalg.norm(direction_xyz))
            if direction_norm <= 0.0:
                raise ValueError("directions must be nonzero.")
            probe_xyz = np.asarray(point_xyz, dtype=np.float64) + (
                (_BOUNDARY_SHIFT_FACTOR * radial_scale) / direction_norm
            ) * np.asarray(direction_xyz, dtype=np.float64)
            forward_leaf = int(self.tree.lookup_points(probe_xyz.reshape(1, 3), coord="xyz")[0])
            if forward_leaf >= 0:
                unique_leaf.append(forward_leaf)

        r_value, _, _ = xyz_to_rpa_components(x, y, z)
        n_azimuth = int(self.tree.leaf_shape[2])
        n_polar = int(self.tree.leaf_shape[1])
        polar_center = 0.5 * (math.pi / float(n_polar))
        if z < 0.0:
            polar_center = math.pi - polar_center
        azimuth_centers = (np.arange(n_azimuth, dtype=np.float64) + 0.5) * ((2.0 * math.pi) / float(n_azimuth))
        query_rpa = np.column_stack(
            (
                np.full(n_azimuth, r_value, dtype=np.float64),
                np.full(n_azimuth, polar_center, dtype=np.float64),
                np.mod(azimuth_centers, 2.0 * math.pi),
            )
        )
        candidate_leaf = self.tree.lookup_points(query_rpa, coord="rpa").reshape(-1)
        for leaf_id in candidate_leaf:
            resolved = int(leaf_id)
            if resolved >= 0 and resolved not in unique_leaf:
                unique_leaf.append(resolved)
        return unique_leaf

    def _seed_interval_candidates(
        self,
        seed_leaf_id: int,
        seed_xyz: np.ndarray,
        direction_xyz: np.ndarray,
    ) -> list[tuple[float, float, int]]:
        """Return candidate ray intervals near one approximate seed point."""
        candidate_leaf_ids = self._seed_candidate_leaves(seed_leaf_id, seed_xyz, direction_xyz)

        intervals: list[tuple[float, float, int]] = []
        for leaf_id in candidate_leaf_ids:
            try:
                enter, exit_, _, _ = self._cell_segment(
                    leaf_id,
                    seed_xyz,
                    direction_xyz,
                    current_t=0.0,
                    t_min=0.0,
                )
            except ValueError as exc:
                if str(exc) not in {
                    "The requested leaf does not intersect the ray.",
                    "Failed to resolve a forward cell segment from the current leaf.",
                    "Degenerate cell exit across an edge or corner is not supported yet.",
                    "Degenerate face exit on a subface boundary is not supported yet.",
                }:
                    raise
            else:
                intervals.append((float(enter), float(exit_), int(leaf_id)))

            try:
                back_enter, back_exit, _, _ = self._cell_segment(
                    leaf_id,
                    seed_xyz,
                    -direction_xyz,
                    current_t=0.0,
                    t_min=0.0,
                )
            except ValueError as exc:
                if str(exc) not in {
                    "The requested leaf does not intersect the ray.",
                    "Failed to resolve a forward cell segment from the current leaf.",
                    "Degenerate cell exit across an edge or corner is not supported yet.",
                    "Degenerate face exit on a subface boundary is not supported yet.",
                }:
                    raise
            else:
                intervals.append((-float(back_exit), -float(back_enter), int(leaf_id)))
        return intervals

    def _seed_candidate_leaves(
        self,
        seed_leaf_id: int,
        seed_xyz: np.ndarray,
        direction_xyz: np.ndarray,
    ) -> list[int]:
        """Return one finite set of leaves that may own one seed boundary point."""
        candidate_leaf_ids: list[int] = []
        seed_leaf = int(seed_leaf_id)
        if seed_leaf >= 0:
            candidate_leaf_ids.append(seed_leaf)
            for leaf_id in self._topological_neighbor_leaves(seed_leaf):
                if leaf_id not in candidate_leaf_ids:
                    candidate_leaf_ids.append(int(leaf_id))
            for leaf_id in self._touching_neighbor_leaves(seed_leaf, seed_xyz):
                if leaf_id not in candidate_leaf_ids:
                    candidate_leaf_ids.append(int(leaf_id))
        if self._tree_coord_code == int(_TREE_COORD_RPA):
            for leaf_id in self._rpa_axis_candidate_leaves(seed_xyz, direction_xyz):
                if leaf_id not in candidate_leaf_ids:
                    candidate_leaf_ids.append(int(leaf_id))
        return candidate_leaf_ids

    def _launch_leaf(
        self,
        seed_leaf_id: int,
        seed_xyz: np.ndarray,
        direction_xyz: np.ndarray,
    ) -> int:
        """Return the unique leaf to launch into from one seed along one direction."""
        return _launch_leaf_kernel(
            int(seed_leaf_id),
            np.asarray(seed_xyz, dtype=np.float64),
            np.asarray(direction_xyz, dtype=np.float64),
            self._face_corners,
            self._leaf_valid,
            self._leaf_points,
            self._leaf_centers,
            self._leaf_scales,
            self._trace_next_leaf,
            int(self._n_polar),
            int(self._n_azimuth),
            int(self._tree_coord_code),
            self._cell_child,
            self._root_cell_ids,
            self._cell_parent,
            self._cell_bounds,
            self._domain_bounds,
            float(self._axis2_period),
            bool(self._axis2_periodic),
        )

    def _canonicalize_seed(
        self,
        seed_leaf_id: int,
        seed_xyz: np.ndarray,
        direction_xyz: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Return one seed point guaranteed to lie inside one traced cell interval."""
        intervals = self._seed_intervals(int(seed_leaf_id), seed_xyz, direction_xyz)
        if not intervals:
            raise ValueError("Failed to place the seed point inside any traced cell interval.")

        def sort_key(item: tuple[float, float, int]) -> tuple[int, float, float, int]:
            enter, exit_, leaf_id = item
            contains_zero = 0 if (enter <= 0.0 <= exit_) else 1
            midpoint = 0.5 * (enter + exit_)
            length = exit_ - enter
            return contains_zero, -length, abs(midpoint), int(leaf_id)

        best_enter, best_exit, best_leaf = min(intervals, key=sort_key)
        seed_offset = 0.5 * (best_enter + best_exit)
        canonical_seed = np.asarray(seed_xyz, dtype=np.float64) + seed_offset * np.asarray(direction_xyz, dtype=np.float64)
        return canonical_seed, int(best_leaf)

    @staticmethod
    def _combine_seed_intervals(
        intervals: list[tuple[float, float, int]],
    ) -> list[tuple[float, float, int]]:
        """Combine same-leaf local ray intervals into one convex cell interval."""
        return _combine_seed_interval_rows(intervals)

    @staticmethod
    def _usable_seed_intervals(
        intervals: list[tuple[float, float, int]],
    ) -> list[tuple[float, float, int]]:
        """Drop numerically collapsed seed intervals when a real local branch exists."""
        intervals = _combine_seed_interval_rows(intervals)
        if not intervals:
            return []
        max_length = max(exit_ - enter for enter, exit_, _ in intervals)
        min_length = 1.0e-6 * max_length
        usable = [item for item in intervals if (item[1] - item[0]) > min_length]
        if usable:
            return usable
        return intervals

    def _seed_intervals(
        self,
        seed_leaf_id: int,
        seed_xyz: np.ndarray,
        direction_xyz: np.ndarray,
    ) -> list[tuple[float, float, int]]:
        """Return usable local seed intervals for one approximate seed point."""
        return self._usable_seed_intervals(
            self._seed_interval_candidates(int(seed_leaf_id), seed_xyz, direction_xyz)
        )

    def _select_common_seed(
        self,
        seed_leaf_id: int,
        seed_xyz: np.ndarray,
        direction_xyz: np.ndarray,
    ) -> tuple[np.ndarray, int] | None:
        """Return one shared interior seed when one leaf spans both ray directions."""
        intervals = self._seed_intervals(int(seed_leaf_id), seed_xyz, direction_xyz)
        shared = [item for item in intervals if item[0] < 0.0 and item[1] > 0.0]
        if not shared:
            return None

        def sort_key(item: tuple[float, float, int]) -> tuple[float, float, int]:
            enter, exit_, leaf_id = item
            midpoint = 0.5 * (enter + exit_)
            length = exit_ - enter
            return abs(midpoint), -length, int(leaf_id)

        best_enter, best_exit, best_leaf = min(shared, key=sort_key)
        seed_offset = 0.5 * (best_enter + best_exit)
        common_seed = np.asarray(seed_xyz, dtype=np.float64) + seed_offset * np.asarray(direction_xyz, dtype=np.float64)
        return common_seed, int(best_leaf)

    def _select_seed_branch(
        self,
        seed_leaf_id: int,
        seed_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        *,
        branch: str,
    ) -> tuple[np.ndarray, int] | None:
        """Return one interior seed point and leaf for one branch off the approximate seed."""
        intervals = self._seed_intervals(int(seed_leaf_id), seed_xyz, direction_xyz)
        if branch == "forward":
            branch_intervals = [item for item in intervals if item[1] > 0.0]
            if not branch_intervals:
                return None

            def sort_key(item: tuple[float, float, int]) -> tuple[float, float, float, int]:
                enter, exit_, leaf_id = item
                start_distance = max(0.0, enter)
                midpoint = 0.5 * (enter + exit_)
                length = exit_ - enter
                return start_distance, -length, abs(midpoint), int(leaf_id)

        elif branch == "backward":
            branch_intervals = [item for item in intervals if item[0] < 0.0]
            if not branch_intervals:
                return None

            def sort_key(item: tuple[float, float, int]) -> tuple[float, float, float, int]:
                enter, exit_, leaf_id = item
                end_distance = max(0.0, -exit_)
                midpoint = 0.5 * (enter + exit_)
                length = exit_ - enter
                return end_distance, -length, abs(midpoint), int(leaf_id)

        else:
            raise ValueError("branch must be 'forward' or 'backward'.")

        best_enter, best_exit, best_leaf = min(branch_intervals, key=sort_key)
        seed_offset = 0.5 * (best_enter + best_exit)
        branch_seed = np.asarray(seed_xyz, dtype=np.float64) + seed_offset * np.asarray(direction_xyz, dtype=np.float64)
        return branch_seed, int(best_leaf)

    def _cell_segment(
        self,
        leaf_id: int,
        origin_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        *,
        current_t: float,
        t_min: float,
    ) -> tuple[float, float, int, int]:
        """Return one exact forward cell segment and its exit face/subface."""
        cell_xyz, cell_center = self._cell_faces(leaf_id)
        point_tol = _EXIT_TOL * max(1.0, float(np.max(np.linalg.norm(cell_xyz - cell_center, axis=1))))
        def solve(local_origin_xyz: np.ndarray) -> tuple[float, float, int, int]:
            inside_now = self._point_inside_cell(int(leaf_id), local_origin_xyz + float(current_t) * direction_xyz, point_tol)

            face_hits: list[tuple[float, int, np.ndarray, np.ndarray]] = []
            for face_id in range(6):
                face_xyz = cell_xyz[self._face_corners[face_id]]
                normal_xyz = self._face_normal(face_xyz, cell_center)
                denom = _dot3(normal_xyz, direction_xyz)
                if abs(denom) <= _EXIT_TOL:
                    continue
                t_hit = _dot3(normal_xyz, face_xyz[0] - local_origin_xyz) / denom
                hit_xyz = local_origin_xyz + t_hit * direction_xyz
                if self._point_inside_face(face_xyz, normal_xyz, hit_xyz, point_tol):
                    face_hits.append((t_hit, int(face_id), hit_xyz, normal_xyz))

            if not face_hits:
                raise ValueError("The requested leaf does not intersect the ray.")

            face_hits.sort(key=lambda item: item[0])
            segment_enter = float(current_t) if inside_now else np.nan
            exit_t = np.nan
            exit_face = -1
            exit_xyz = np.full(3, np.nan, dtype=np.float64)
            exit_normal = np.full(3, np.nan, dtype=np.float64)
            enter_tol = _EXIT_TOL * max(1.0, abs(float(current_t)))

            for t_hit, face_id, hit_xyz, normal_xyz in face_hits:
                if t_hit < (float(t_min) - enter_tol):
                    continue
                if not np.isfinite(segment_enter):
                    segment_enter = float(t_hit)
                    continue
                if t_hit <= (float(segment_enter) + enter_tol):
                    continue
                exit_t = float(t_hit)
                exit_face = int(face_id)
                exit_xyz = hit_xyz
                exit_normal = normal_xyz
                break

            if not np.isfinite(segment_enter) or not np.isfinite(exit_t):
                raise ValueError("Failed to resolve a forward cell segment from the current leaf.")
            if exit_t <= segment_enter:
                raise ValueError("Degenerate zero-length cell interval is not supported yet.")

            exit_tol = _EXIT_TOL * max(1.0, abs(float(exit_t)))
            degenerate_hits = [face_id for t_hit, face_id, _, _ in face_hits if abs(t_hit - exit_t) <= exit_tol]
            degenerate_face_exit = len(degenerate_hits) > 1
            if len(degenerate_hits) > 1:
                exit_face = int(min(degenerate_hits))

            if degenerate_face_exit:
                subface_id = -1
            else:
                try:
                    subface_id = self._face_exit_subface(leaf_id, exit_xyz, exit_face, exit_normal)
                except ValueError as exc:
                    if str(exc) != "Degenerate face exit on a subface boundary is not supported yet.":
                        raise
                    subface_id = -1
            return float(segment_enter), float(exit_t), int(exit_face), int(subface_id)

        try:
            return solve(origin_xyz)
        except ValueError as exc:
            if str(exc) != "Failed to resolve a forward cell segment from the current leaf.":
                raise

        direction_norm = float(np.linalg.norm(direction_xyz))
        if direction_norm <= 0.0:
            raise ValueError("directions must be nonzero.")
        boundary_shift = (_BOUNDARY_SHIFT_FACTOR * max(1.0, float(np.max(np.linalg.norm(cell_xyz, axis=1))))) / direction_norm
        segment_enter, exit_t, exit_face, subface_id = solve(origin_xyz + boundary_shift * direction_xyz)
        return (
            float(segment_enter + boundary_shift),
            float(exit_t + boundary_shift),
            int(exit_face),
            int(subface_id),
        )

    def _trace_one_ray(
        self,
        start_leaf_id: int,
        origin_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        *,
        t_min: float = 0.0,
        t_max: float = np.inf,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Walk one ray across neighboring planar-face hex cells and return exact cell intervals.

        This is an internal first-pass walker used to validate the topology cache.
        It assumes a non-degenerate ray that crosses one face at a time.
        """
        clip_lo = float(t_min)
        clip_hi = float(t_max)
        if not math.isfinite(clip_lo):
            raise ValueError("t_min must be finite.")
        if math.isnan(clip_hi):
            raise ValueError("t_max must not be NaN.")
        if clip_hi < clip_lo:
            raise ValueError("t_max must be greater than or equal to t_min.")
        o_flat, d_flat, shape = _normalize_ray_arrays(origin_xyz, direction_xyz)
        if shape != (1,):
            raise ValueError("_trace_one_ray requires one origin and one direction vector.")

        return trace_one_ray_kernel(
            int(start_leaf_id),
            o_flat[0],
            d_flat[0],
            float(clip_lo),
            float(clip_hi),
            self._trace_state,
        )

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

        join_tol = _EXIT_TOL * max(1.0, abs(float(backward_t_exit[-1])), abs(float(forward_t_enter[0])))
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
        t_min: float = 0.0,
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
        clip_lo = max(0.0, t_lo)
        (
            leaf_valid,
            leaf_points,
            leaf_centers,
            leaf_scales,
            face_corners,
            next_leaf,
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
        ) = self._trace_state
        ray_offsets, cell_ids, t_enter, t_exit = _trace_rays_kernel(
            o_flat,
            d_flat,
            seed_flat,
            float(clip_lo),
            float(t_hi),
            leaf_valid,
            leaf_points,
            leaf_centers,
            leaf_scales,
            face_corners,
            next_leaf,
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
        t_min: float = 0.0,
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
        clip_lo = max(0.0, t_lo)
        if self._tree_coord_code == int(_TREE_COORD_XYZ):
            seed_xyz = _seed_domain_xyz_kernel(
                o_flat,
                d_flat,
                float(clip_lo),
                float(t_hi),
                self._seed_domain_xyz_lo,
                self._seed_domain_xyz_hi,
            )
            return seed_xyz.reshape(shape + (3,))

        if self._tree_coord_code != int(_TREE_COORD_RPA):
            raise NotImplementedError(f"Unsupported tree_coord '{self.tree.tree_coord}' for seed_domain.")
        seed_xyz = _seed_domain_rpa_kernel(
            o_flat,
            d_flat,
            float(clip_lo),
            float(t_hi),
            float(self._seed_domain_r_min),
            float(self._seed_domain_r_max),
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
    mid_t = 0.5 * (segments.t_enter + segments.t_exit)
    mid_xyz = o_flat[ray_ids] + mid_t[:, None] * d_flat[ray_ids]
    samples = np.asarray(interpolator(mid_xyz, query_coord="xyz", log_outside_domain=False), dtype=np.float64)
    samples_2d = samples.reshape(samples.shape[0], -1)
    np.add.at(accum, ray_ids, samples_2d * segments.segment_length[:, None])

    out = accum.reshape(tuple(ray_shape) + interpolator.value_shape)
    if interpolator.value_shape:
        return out
    return out.reshape(tuple(ray_shape))
