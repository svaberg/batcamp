#!/usr/bin/env python3
"""Cartesian raytracer backend.

This module is the Cartesian backend used by the public ray tracer. It clips
one straight ray to the axis-aligned octree domain, finds the next exit face of
the current leaf, follows the face/subface neighbor graph to determine the
owner of the open interval after that event, and provides the Cartesian direct
midpoint and trilinear accumulators that sit on top of those traced segments.

The emphasis here is exact discrete ownership, not interpolation or rendering.
Given one current leaf and one straight Cartesian ray, these kernels produce
the packed sequence of visited leaves and event times that higher layers later
use for accumulation.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from numba import prange

from . import interpolator_cartesian
from .octree_cartesian import _contains_box
from .octree import Octree
from .octree import BOUNDS_START_SLOT
from .octree import BOUNDS_WIDTH_SLOT
from .octree import _FACE_ID_TO_AXIS
from .octree import _FACE_ID_TO_SIDE
from .octree import _FACE_ID_TO_TANGENTIAL_AXES

# Start with a moderate Cartesian scratch row because straight box crossings
# are cheap and often visit more cells than the spherical default path.
DEFAULT_CROSSING_BUFFER_SIZE = 256

# Scratch trace sentinel returned when one per-ray row fills before the public
# tracer has a chance to grow the buffers and retry.
TRACE_BUFFER_OVERFLOW = -1


def trace_ray(
    tree: Octree,
    origin: np.ndarray,
    direction: np.ndarray,
    start_cell_id: int,
    *,
    t_min: float = 0.0,
    t_max: float = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace one Cartesian ray by exact face crossings plus chained face/subface neighbors.

    Args:
        tree (const): Built Cartesian octree.
        origin (const): Ray origin with shape `(3,)`.
        direction (const): Nonzero ray direction with shape `(3,)`.
        start_cell_id (const): Fallback start cell if the clipped entry lookup lands outside.
        t_min (const): Lower parameter clip.
        t_max (const): Upper parameter clip.

    Returns:
        `(cell_ids, times)` for the traced positive-length and zero-hop crossing path.
    """
    if str(tree.tree_coord) != "xyz":
        raise ValueError("trace_ray supports only tree_coord='xyz'.")
    if not math.isfinite(float(t_min)):
        raise ValueError("t_min must be finite.")
    if float(t_max) < float(t_min):
        raise ValueError("t_max must be greater than or equal to t_min.")

    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if origin.shape != (3,) or direction.shape != (3,):
        raise ValueError("origin and direction must have shape (3,).")
    if not np.any(direction != 0.0):
        raise ValueError("direction must be nonzero.")

    crossing_capacity = DEFAULT_CROSSING_BUFFER_SIZE
    while True:
        cell_ids = np.empty(crossing_capacity, dtype=np.int64)
        times = np.empty(crossing_capacity + 1, dtype=np.float64)
        n_cell, n_time = _trace_ray(
            tree.root_cell_ids,
            tree.cell_child,
            tree.cell_bounds,
            tree.packed_domain_bounds,
            tree.cell_neighbor,
            int(start_cell_id),
            origin,
            direction,
            float(t_min),
            float(t_max),
            cell_ids,
            times,
        )
        if n_cell == -1 or n_time == -1:
            crossing_capacity *= 2
            continue
        if n_cell < 0 or n_time < 0:
            raise ValueError("Cartesian ray trace encountered an invalid crossing.")
        return cell_ids[:n_cell].copy(), times[:n_time].copy()


@njit(cache=True)
def _trace_ray(
    root_cell_ids: np.ndarray,
    cell_child: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
    cell_neighbor: np.ndarray,
    start_cell_id: int,
    origin: np.ndarray,
    direction: np.ndarray,
    t_min: float,
    t_max: float,
    cell_ids_out: np.ndarray,
    times_out: np.ndarray,
) -> tuple[int, int]:
    """Trace one Cartesian ray into fixed scratch buffers.

    Args:
        root_cell_ids (const): Root cell ids.
        cell_child (const): Packed child table.
        cell_bounds (const): Packed leaf bounds.
        domain_bounds (const): Cartesian domain bounds.
        cell_neighbor (const): Packed face/subface neighbor table.
        start_cell_id (const): Fallback start cell if entry lookup lands outside.
        origin (const): Ray origin with shape `(3,)`.
        direction (const): Ray direction with shape `(3,)`.
        t_min (const): Lower parameter clip.
        t_max (const): Upper parameter clip.
        cell_ids_out (output): Scratch cell-id trace buffer.
        times_out (output): Scratch time buffer.

    Returns:
        `(n_cell, n_time)`, `(-1, -1)` for scratch overflow, or `(-2, -2)` for invalid crossings.
    """
    max_cells = int(cell_ids_out.shape[0])
    has_interval, domain_enter, domain_exit = find_domain_interval(origin, direction, domain_bounds)
    if not has_interval:
        return 0, 0
    start_t = max(float(t_min), float(domain_enter))
    stop_t = min(float(t_max), float(domain_exit))
    if not (start_t < stop_t):
        return 0, 0
    start = np.empty(3, dtype=np.float64)
    current = np.empty(3, dtype=np.float64)
    crossing = np.empty(3, dtype=np.float64)
    active_faces = np.empty(3, dtype=np.int64)
    path = np.empty(3, dtype=np.int64)
    active_face_by_axis = np.empty(3, dtype=np.int64)
    active_face_order = np.empty(6, dtype=np.int64)
    for axis in range(3):
        start[axis] = float(origin[axis]) + float(start_t) * float(direction[axis])
        current[axis] = float(start[axis])
    if float(start_t) == float(domain_enter):
        for axis in range(3):
            direction_value = float(direction[axis])
            if direction_value == 0.0:
                continue
            lo_value = float(domain_bounds[axis, BOUNDS_START_SLOT])
            hi_value = lo_value + float(domain_bounds[axis, BOUNDS_WIDTH_SLOT])
            if direction_value > 0.0:
                t0 = (lo_value - float(origin[axis])) / direction_value
                face_value = lo_value
            else:
                t0 = (hi_value - float(origin[axis])) / direction_value
                face_value = hi_value
            if float(t0) == float(domain_enter):
                start[axis] = face_value
                current[axis] = face_value
    current_cell = find_cell(start, cell_child, root_cell_ids, cell_bounds, domain_bounds)
    if current_cell < 0:
        current_cell = int(start_cell_id)
    if current_cell < 0:
        return 0, 0
    n_cell = 0
    n_time = 1
    times_out[0] = float(start_t)
    t_current = float(start_t)
    while current_cell >= 0 and t_current < stop_t:
        t_exit, n_active_face = find_exit(
            cell_bounds,
            current_cell,
            current,
            direction,
            t_current,
            active_faces,
        )
        if n_active_face < 0:
            return -2, -2
        if t_exit > stop_t:
            if n_cell >= max_cells:
                return -1, -1
            cell_ids_out[n_cell] = int(current_cell)
            times_out[n_time] = float(stop_t)
            n_cell += 1
            n_time += 1
            break
        if n_cell >= max_cells:
            return -1, -1
        cell_ids_out[n_cell] = int(current_cell)
        times_out[n_time] = float(t_exit)
        n_cell += 1
        n_time += 1
        _write_crossing(
            cell_bounds,
            current_cell,
            active_faces,
            n_active_face,
            current,
            direction,
            float(t_exit) - float(t_current),
            crossing,
        )
        path_count = walk_faces(
            cell_neighbor,
            domain_bounds,
            cell_bounds,
            current_cell,
            active_faces,
            n_active_face,
            crossing,
            direction,
            path,
            active_face_by_axis,
            active_face_order,
        )
        if path_count < 0:
            return -2, -2
        for path_pos in range(path_count - 1):
            intermediate_cell = int(path[path_pos])
            if intermediate_cell < 0:
                break
            if n_cell >= max_cells:
                return -1, -1
            cell_ids_out[n_cell] = int(intermediate_cell)
            times_out[n_time] = float(t_exit)
            n_cell += 1
            n_time += 1
        if path_count > 0:
            current_cell = int(path[path_count - 1])
        else:
            current_cell = -1
        t_current = float(t_exit)
        for axis in range(3):
            current[axis] = float(crossing[axis])
    return int(n_cell), int(n_time)


@njit(cache=True)
def find_domain_interval(
    origin: np.ndarray,
    direction: np.ndarray,
    domain_bounds: np.ndarray,
) -> tuple[bool, float, float]:
    """Return whether one ray intersects the domain plus the clipped parameter interval.

    Args:
        origin (const): Ray origin with shape `(3,)`.
        direction (const): Ray direction with shape `(3,)`.
        domain_bounds (const): Cartesian domain bounds with shape `(3, 2)`.

    Returns:
        `(has_interval, t_enter, t_exit)`.
    """
    t_enter = -np.inf
    t_exit = np.inf
    for axis in range(3):
        direction_value = float(direction[axis])
        origin_value = float(origin[axis])
        lo_value = float(domain_bounds[axis, BOUNDS_START_SLOT])
        hi_value = lo_value + float(domain_bounds[axis, BOUNDS_WIDTH_SLOT])
        if direction_value == 0.0:
            if origin_value < lo_value or origin_value > hi_value:
                return False, 0.0, 0.0
            continue
        t0 = (lo_value - origin_value) / direction_value
        t1 = (hi_value - origin_value) / direction_value
        if t0 > t1:
            tmp = t0
            t0 = t1
            t1 = tmp
        if t0 > t_enter:
            t_enter = t0
        if t1 < t_exit:
            t_exit = t1
        if t_enter > t_exit:
            return False, 0.0, 0.0
    return True, float(t_enter), float(t_exit)


@njit(cache=True)
def find_cell(
    query: np.ndarray,
    cell_child: np.ndarray,
    root_cell_ids: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
) -> int:
    """Resolve one Cartesian query to its exact half-open owner.

    Args:
        query (const): Cartesian query point with shape `(3,)`.
        cell_child (const): Packed child table.
        root_cell_ids (const): Root cell ids.
        cell_bounds (const): Packed leaf bounds.
        domain_bounds (const): Cartesian domain bounds.

    Returns:
        Owning leaf cell id, or `-1` if the query lies outside the domain.
    """
    if not (np.isfinite(query[0]) and np.isfinite(query[1]) and np.isfinite(query[2])):
        return -1
    if not _contains_box(query, domain_bounds, domain_bounds):
        return -1
    current = -1
    for root_pos in range(int(root_cell_ids.shape[0])):
        root_cell_id = int(root_cell_ids[root_pos])
        if _contains_box(query, cell_bounds[root_cell_id], domain_bounds):
            current = root_cell_id
            break
    if current < 0:
        return -1
    while True:
        next_cell_id = -1
        for child_ord in range(8):
            child_id = int(cell_child[current, child_ord])
            if child_id < 0:
                continue
            if _contains_box(query, cell_bounds[child_id], domain_bounds):
                next_cell_id = child_id
                break
        if next_cell_id < 0:
            return int(current)
        current = int(next_cell_id)


@njit(cache=True)
def find_exit(
    cell_bounds: np.ndarray,
    cell_id: int,
    current: np.ndarray,
    direction: np.ndarray,
    t_current: float,
    active_faces: np.ndarray,
) -> tuple[float, int]:
    """Return one leaf exit crossing as `t_exit` plus active face ids.

    Args:
        cell_bounds (const): Packed leaf bounds.
        cell_id (const): Current leaf id.
        current (const): Current point inside the leaf.
        direction (const): Ray direction.
        t_current (const): Current ray parameter.
        active_faces (output): Scratch face ids written in crossing order.

    Returns:
        `(t_exit, n_active_face)`.
    """
    bounds = cell_bounds[int(cell_id)]
    t_exit = 0.0
    n_active_face = 0
    have_candidate = False
    for axis in range(3):
        direction_value = float(direction[axis])
        if direction_value > 0.0:
            face_id = 2 * axis + 1
            face_value = float(bounds[axis, BOUNDS_START_SLOT] + bounds[axis, BOUNDS_WIDTH_SLOT])
        elif direction_value < 0.0:
            face_id = 2 * axis
            face_value = float(bounds[axis, BOUNDS_START_SLOT])
        else:
            continue
        t_face = float(t_current) + (face_value - float(current[axis])) / direction_value
        if t_face < t_current:
            return np.nan, -1
        if not have_candidate or t_face < t_exit:
            t_exit = float(t_face)
            active_faces[0] = int(face_id)
            n_active_face = 1
            have_candidate = True
            continue
        if t_face == t_exit:
            active_faces[n_active_face] = int(face_id)
            n_active_face += 1
    if not have_candidate:
        return np.nan, -1
    return float(t_exit), int(n_active_face)


@njit(cache=True)
def _fill_active_face_state(
    active_faces: np.ndarray,
    n_active_face: int,
    current_face_id: int,
    active_face_by_axis: np.ndarray,
    active_face_order: np.ndarray,
) -> int:
    """Fill reusable active-face lookup arrays and return the current face order.

    Args:
        active_faces (const): Active face ids for one crossing.
        n_active_face (const): Number of live entries in `active_faces`.
        current_face_id (const): Face currently being resolved.
        active_face_by_axis (output): Axis-to-active-face scratch map.
        active_face_order (output): Face-id-to-order scratch map.

    Returns:
        Order of `current_face_id` inside `active_faces`.
    """
    active_face_by_axis[:] = -1
    active_face_order[:] = -1
    current_face_order = -1
    for order in range(n_active_face):
        face_id = int(active_faces[order])
        active_face_by_axis[int(_FACE_ID_TO_AXIS[face_id])] = face_id
        active_face_order[face_id] = int(order)
        if face_id == int(current_face_id):
            current_face_order = int(order)
    return int(current_face_order)


@njit(cache=True)
def is_on_face(
    cell_bounds: np.ndarray,
    cell_id: int,
    face_id: int,
    crossing: np.ndarray,
) -> bool:
    """Return whether one crossing point lies on one Cartesian face patch carrier.

    Args:
        cell_bounds (const): Packed leaf bounds.
        cell_id (const): Candidate carrier leaf id.
        face_id (const): Face id on that leaf.
        crossing (const): Crossing point.

    Returns:
        Whether the crossing lies on that face carrier.
    """
    bounds = cell_bounds[int(cell_id)]
    axis = int(_FACE_ID_TO_AXIS[int(face_id)])
    side = int(_FACE_ID_TO_SIDE[int(face_id)])
    face_value = float(bounds[axis, BOUNDS_START_SLOT]) if side == 0 else float(bounds[axis, BOUNDS_START_SLOT] + bounds[axis, BOUNDS_WIDTH_SLOT])
    if float(crossing[axis]) != face_value:
        return False
    for tangential_axis in _FACE_ID_TO_TANGENTIAL_AXES[int(face_id)]:
        axis = int(tangential_axis)
        start = float(bounds[axis, BOUNDS_START_SLOT])
        stop = start + float(bounds[axis, BOUNDS_WIDTH_SLOT])
        value = float(crossing[axis])
        if value < start or value > stop:
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
    crossing: np.ndarray,
    direction: np.ndarray,
) -> int:
    """Return the destination-side owning face patch for one face crossing.

    Args:
        cell_neighbor (const): Packed face/subface neighbor table.
        domain_bounds (const): Cartesian domain bounds.
        cell_bounds (const): Packed leaf bounds.
        cell_id (const): Current leaf id.
        face_id (const): Active crossed face on the current leaf.
        active_face_by_axis (const): Axis-to-active-face map for this crossing.
        active_face_order (const): Face-id-to-order map for this crossing.
        current_face_order (const): Order of `face_id` inside the active-face list.
        crossing (const): Crossing point snapped onto crossed faces.
        direction (const): Ray direction.

    Returns:
        Face-patch slot id, `-1` for no owner, or `-2` for ambiguity.
    """
    tangential_axes = _FACE_ID_TO_TANGENTIAL_AXES[int(face_id)]
    row = cell_neighbor[int(cell_id), int(face_id)]
    first_neighbor = -1
    first_slot = -1
    has_multiple_neighbors = False
    for subface_id in range(4):
        neighbor_id = int(row[subface_id])
        if neighbor_id < 0:
            continue
        if first_neighbor < 0:
            first_neighbor = neighbor_id
            first_slot = int(subface_id)
            continue
        if neighbor_id != first_neighbor:
            has_multiple_neighbors = True
            break
    if first_neighbor < 0:
        return 0
    if not has_multiple_neighbors:
        return int(first_slot)
    matched_subface = -1
    matched_neighbor = -1
    current_bounds = cell_bounds[int(cell_id)]
    for subface_id in range(4):
        neighbor_id = int(cell_neighbor[int(cell_id), int(face_id), subface_id])
        if neighbor_id < 0:
            continue
        contains = True
        neighbor_bounds = cell_bounds[neighbor_id]
        for tangential_pos in range(2):
            axis = int(tangential_axes[tangential_pos])
            value = float(crossing[axis])
            current_start = float(current_bounds[axis, BOUNDS_START_SLOT])
            current_stop = current_start + float(current_bounds[axis, BOUNDS_WIDTH_SLOT])
            active_face_id = int(active_face_by_axis[axis])
            direction_value = float(direction[axis])
            implicit_active_side = -1
            if value == current_start and direction_value < 0.0:
                implicit_active_side = 0
            elif value == current_stop and direction_value > 0.0:
                implicit_active_side = 1
            if active_face_id >= 0 or implicit_active_side >= 0:
                neighbor_start = float(neighbor_bounds[axis, BOUNDS_START_SLOT])
                neighbor_stop = neighbor_start + float(neighbor_bounds[axis, BOUNDS_WIDTH_SLOT])
                if active_face_id >= 0:
                    active_side = int(_FACE_ID_TO_SIDE[active_face_id])
                    if int(active_face_order[active_face_id]) < int(current_face_order):
                        active_side = 1 - active_side
                else:
                    active_side = int(implicit_active_side)
                contains_current_interval = neighbor_start <= current_start and current_stop <= neighbor_stop
                if not contains_current_interval:
                    if active_side == 0:
                        if neighbor_start != current_start:
                            contains = False
                            break
                    else:
                        if neighbor_stop != current_stop:
                            contains = False
                            break
                if value < current_start or value > current_stop:
                    contains = False
                    break
                continue
            start = float(neighbor_bounds[axis, BOUNDS_START_SLOT])
            stop = start + float(neighbor_bounds[axis, BOUNDS_WIDTH_SLOT])
            if direction_value > 0.0:
                if value < start or value >= stop:
                    contains = False
                    break
            elif direction_value < 0.0:
                if value <= start or value > stop:
                    contains = False
                    break
            else:
                if start == float(domain_bounds[axis, BOUNDS_START_SLOT]):
                    if value < start or value > stop:
                        contains = False
                        break
                else:
                    if value <= start or value > stop:
                        contains = False
                        break
        if not contains:
            continue
        if matched_neighbor < 0:
            matched_subface = int(subface_id)
            matched_neighbor = int(neighbor_id)
            continue
        if neighbor_id != matched_neighbor:
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
    current: np.ndarray,
    direction: np.ndarray,
    delta_t: float,
    crossing: np.ndarray,
) -> None:
    """Write one snapped crossing point for one leaf exit crossing.

    Args:
        cell_bounds (const): Packed leaf bounds.
        cell_id (const): Current leaf id.
        active_faces (const): Active crossed faces.
        n_active_face (const): Number of live active faces.
        current (const): Current point in the leaf.
        direction (const): Ray direction.
        delta_t (const): Boundary step size from `current`.
        crossing (output): Snapped crossing point.
    """
    for axis in range(3):
        crossing[axis] = float(current[axis]) + float(delta_t) * float(direction[axis])
    bounds = cell_bounds[int(cell_id)]
    for face_pos in range(n_active_face):
        face_id = int(active_faces[face_pos])
        axis = int(_FACE_ID_TO_AXIS[int(face_id)])
        side = int(_FACE_ID_TO_SIDE[int(face_id)])
        if side == 0:
            crossing[axis] = float(bounds[axis, BOUNDS_START_SLOT])
        else:
            crossing[axis] = float(bounds[axis, BOUNDS_START_SLOT] + bounds[axis, BOUNDS_WIDTH_SLOT])


@njit(cache=True)
def walk_faces(
    cell_neighbor: np.ndarray,
    domain_bounds: np.ndarray,
    cell_bounds: np.ndarray,
    start_cell_id: int,
    active_faces: np.ndarray,
    n_active_face: int,
    crossing: np.ndarray,
    direction: np.ndarray,
    path: np.ndarray,
    active_face_by_axis: np.ndarray,
    active_face_order: np.ndarray,
) -> int:
    """Walk one exact Cartesian crossing through the face/subface neighbor graph.

    Args:
        cell_neighbor (const): Packed face/subface neighbor table.
        domain_bounds (const): Cartesian domain bounds.
        cell_bounds (const): Packed leaf bounds.
        start_cell_id (const): Leaf that owns the pre-crossing segment.
        active_faces (const): Active crossed faces for this crossing.
        n_active_face (const): Number of live active faces.
        crossing (const): Snapped crossing point.
        direction (const): Ray direction.
        path (output): Scratch crossing continuation path.
        active_face_by_axis (output): Scratch axis-to-face map.
        active_face_order (output): Scratch face-id-to-order map.

    Returns:
        Number of written cells in `path`, or `-1` on ambiguity.
    """
    current_cell = int(start_cell_id)
    path_count = 0
    for face_pos in range(n_active_face):
        face_id = int(active_faces[face_pos])
        if current_cell < 0:
            break
        if not is_on_face(cell_bounds, current_cell, face_id, crossing):
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
            crossing,
            direction,
        )
        if subface_id < 0:
            return -1
        current_cell = int(cell_neighbor[current_cell, face_id, subface_id])
        path[path_count] = current_cell
        path_count += 1
    return int(path_count)


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
    cell_counts: np.ndarray,
    time_counts: np.ndarray,
    cell_ids_out: np.ndarray,
    times_out: np.ndarray,
) -> None:
    """Trace flat rays into fixed per-ray scratch buffers.

    Args:
        root_cell_ids (const): Root cell ids.
        cell_child (const): Packed child table.
        cell_bounds (const): Packed leaf bounds.
        domain_bounds (const): Cartesian domain bounds.
        cell_neighbor (const): Packed face/subface neighbor table.
        cell_depth (const): Packed cell depths, unused for Cartesian traversal.
        origins (const): Flat origin array with shape `(n_rays, 3)`.
        directions (const): Flat direction array with shape `(n_rays, 3)`.
        t_min (const): Lower parameter clip.
        t_max (const): Upper parameter clip.
        cell_counts (output): Per-ray cell counts.
        time_counts (output): Per-ray time counts.
        cell_ids_out (output): Per-ray scratch cell-id rows.
        times_out (output): Per-ray scratch time rows.
    """
    n_rays = int(origins.shape[0])
    for ray_id in prange(n_rays):
        n_cell, n_time = _trace_ray(
            root_cell_ids,
            cell_child,
            cell_bounds,
            domain_bounds,
            cell_neighbor,
            -1,
            origins[ray_id],
            directions[ray_id],
            t_min,
            t_max,
            cell_ids_out[ray_id],
            times_out[ray_id],
        )
        cell_counts[ray_id] = int(n_cell)
        time_counts[ray_id] = int(n_time)


def midpoint_image(
    tracer,
    interpolator,
    origins,
    directions,
    *,
    t_min: float,
    t_max: float,
    ray_shape: tuple[int, ...],
    geometry_origins,
    geometry_directions,
):
    """Accumulate midpoint-sampled Cartesian rays directly from traced chunks."""
    return tracer.accumulate_chunked(
        interpolator,
        origins,
        directions,
        t_min=t_min,
        t_max=t_max,
        ray_shape=ray_shape,
        accumulator=interpolator_cartesian.accumulate_midpoint_cells,
        label="midpoint_image",
    )


def trilinear_image(
    tracer,
    interpolator,
    origins,
    directions,
    *,
    t_min: float,
    t_max: float,
    ray_shape: tuple[int, ...],
):
    """Accumulate exact Cartesian trilinear ray integrals directly from traced chunks."""
    return tracer.accumulate_chunked(
        interpolator,
        origins,
        directions,
        t_min=t_min,
        t_max=t_max,
        ray_shape=ray_shape,
        accumulator=interpolator_cartesian.accumulate_trilinear_cells,
        label="trilinear_image",
    )
