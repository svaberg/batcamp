#!/usr/bin/env python3
"""Small Cartesian event-walk prototype that uses refined face/subface neighbors."""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from numba import prange

from .cartesian import _contains_box_xyz
from .octree import Octree
from .octree import START
from .octree import WIDTH
from .octree import _FACE_AXIS
from .octree import _FACE_SIDE
from .octree import _FACE_TANGENTIAL_AXES

_MAX_RAY_TRACE_EVENTS = 1000


def _domain_interval_xyz(
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    domain_lo: np.ndarray,
    domain_hi: np.ndarray,
) -> tuple[float, float] | None:
    """Return the clipped Cartesian domain interval for one ray."""
    t_enter = -np.inf
    t_exit = np.inf
    for axis in range(3):
        direction_value = float(direction_xyz[axis])
        origin_value = float(origin_xyz[axis])
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


def _cell_exit_event_xyz(
    cell_bounds: np.ndarray,
    cell_id: int,
    current_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    t_current: float,
) -> tuple[float, tuple[int, ...]]:
    """Return the next exit time and active face ids for one Cartesian leaf."""
    active_faces = np.empty(3, dtype=np.int64)
    t_exit, n_active_face = _cell_exit_event_xyz_raw(
        cell_bounds,
        int(cell_id),
        np.asarray(current_xyz, dtype=np.float64),
        np.asarray(direction_xyz, dtype=np.float64),
        float(t_current),
        active_faces,
    )
    if n_active_face < 0:
        raise ValueError("direction must leave the current leaf along at least one axis.")
    return float(t_exit), tuple(int(active_faces[pos]) for pos in range(n_active_face))


def _event_subface_id(
    cell_neighbor: np.ndarray,
    domain_bounds: np.ndarray,
    cell_bounds: np.ndarray,
    cell_id: int,
    face_id: int,
    active_faces: tuple[int, ...],
    event_xyz: np.ndarray,
    direction_xyz: np.ndarray,
) -> int:
    """Return the neighbor-table face patch whose destination tangential slabs own one event point."""
    active_faces_arr = np.asarray(active_faces, dtype=np.int64)
    active_face_by_axis = np.empty(3, dtype=np.int64)
    active_face_order = np.empty(6, dtype=np.int64)
    current_face_order = _fill_active_face_state_raw(
        active_faces_arr,
        int(active_faces_arr.size),
        int(face_id),
        active_face_by_axis,
        active_face_order,
    )
    subface_id = _event_subface_id_raw(
        cell_neighbor,
        domain_bounds,
        cell_bounds,
        int(cell_id),
        int(face_id),
        active_face_by_axis,
        active_face_order,
        int(current_face_order),
        np.asarray(event_xyz, dtype=np.float64),
        np.asarray(direction_xyz, dtype=np.float64),
    )
    if subface_id == -2:
        raise ValueError("Event face patch is ambiguous under destination-side ownership.")
    if subface_id < 0:
        raise ValueError("Event face patch has no destination-side owner.")
    return int(subface_id)


def _event_on_face(
    cell_bounds: np.ndarray,
    cell_id: int,
    face_id: int,
    event_xyz: np.ndarray,
) -> bool:
    """Return whether one event point lies on one Cartesian leaf face."""
    return bool(
        _event_on_face_raw(
            cell_bounds,
            int(cell_id),
            int(face_id),
            np.asarray(event_xyz, dtype=np.float64),
        )
    )


def walk_event_faces_xyz(
    tree: Octree,
    start_cell_id: int,
    active_faces: tuple[int, ...],
    event_xyz: np.ndarray,
    direction_xyz: np.ndarray,
) -> tuple[int, ...]:
    """Walk one Cartesian face event through the octree face/subface neighbor graph."""
    if str(tree.tree_coord) != "xyz":
        raise ValueError("walk_event_faces_xyz supports only tree_coord='xyz'.")
    active_faces_arr = np.asarray(active_faces, dtype=np.int64)
    path = np.empty(max(1, int(active_faces_arr.size)), dtype=np.int64)
    active_face_by_axis = np.empty(3, dtype=np.int64)
    active_face_order = np.empty(6, dtype=np.int64)
    path_count = _walk_event_faces_xyz_raw(
        tree.cell_neighbor,
        tree._domain_bounds,
        tree.cell_bounds,
        int(start_cell_id),
        active_faces_arr,
        int(active_faces_arr.size),
        np.asarray(event_xyz, dtype=np.float64),
        np.asarray(direction_xyz, dtype=np.float64),
        path,
        active_face_by_axis,
        active_face_order,
    )
    if path_count < 0:
        raise ValueError("Event face patch is ambiguous under destination-side ownership.")
    return tuple(int(path[pos]) for pos in range(path_count))


def _event_xyz_on_active_faces(
    cell_bounds: np.ndarray,
    cell_id: int,
    active_faces: tuple[int, ...],
    current_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    delta_t: float,
) -> np.ndarray:
    """Return one event point snapped onto the crossed faces of the current cell."""
    active_faces_arr = np.asarray(active_faces, dtype=np.int64)
    event_xyz = np.empty(3, dtype=np.float64)
    _event_xyz_on_active_faces_raw(
        cell_bounds,
        int(cell_id),
        active_faces_arr,
        int(active_faces_arr.size),
        np.asarray(current_xyz, dtype=np.float64),
        np.asarray(direction_xyz, dtype=np.float64),
        float(delta_t),
        event_xyz,
    )
    return event_xyz


def trace_xyz_refined_event_path(
    tree: Octree,
    start_cell_id: int,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    *,
    t_min: float = 0.0,
    t_max: float = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace one Cartesian ray by exact face events plus chained face/subface neighbors."""
    if str(tree.tree_coord) != "xyz":
        raise ValueError("trace_xyz_refined_event_path supports only tree_coord='xyz'.")
    if not math.isfinite(float(t_min)):
        raise ValueError("t_min must be finite.")
    if float(t_max) < float(t_min):
        raise ValueError("t_max must be greater than or equal to t_min.")

    origin = np.asarray(origin_xyz, dtype=float)
    direction = np.asarray(direction_xyz, dtype=float)
    if origin.shape != (3,) or direction.shape != (3,):
        raise ValueError("origin_xyz and direction_xyz must have shape (3,).")
    if not np.any(direction != 0.0):
        raise ValueError("direction_xyz must be nonzero.")

    cell_ids = np.empty(_MAX_RAY_TRACE_EVENTS, dtype=np.int64)
    times = np.empty(_MAX_RAY_TRACE_EVENTS + 1, dtype=np.float64)
    n_cell, n_time = _trace_one_xyz_ray_raw(
        tree._root_cell_ids,
        tree.cell_child,
        tree.cell_bounds,
        tree._domain_bounds,
        tree.cell_neighbor,
        int(start_cell_id),
        origin,
        direction,
        float(t_min),
        float(t_max),
        cell_ids,
        times,
    )
    if n_cell < 0 or n_time < 0:
        raise ValueError(
            f"xyz ray trace exceeded _MAX_RAY_TRACE_EVENTS={_MAX_RAY_TRACE_EVENTS} or encountered an invalid event."
        )
    return cell_ids[:n_cell].copy(), times[:n_time].copy()


@njit(cache=True)
def _domain_interval_xyz_raw(
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    domain_bounds: np.ndarray,
) -> tuple[bool, float, float]:
    """Return whether one ray intersects the domain plus the clipped parameter interval."""
    t_enter = -np.inf
    t_exit = np.inf
    for axis in range(3):
        direction_value = float(direction_xyz[axis])
        origin_value = float(origin_xyz[axis])
        lo_value = float(domain_bounds[axis, START])
        hi_value = lo_value + float(domain_bounds[axis, WIDTH])
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
def _find_cell_xyz_single(
    query_xyz: np.ndarray,
    cell_child: np.ndarray,
    root_cell_ids: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
) -> int:
    """Resolve one Cartesian query to its exact half-open owner."""
    if not (np.isfinite(query_xyz[0]) and np.isfinite(query_xyz[1]) and np.isfinite(query_xyz[2])):
        return -1
    if not _contains_box_xyz(query_xyz, domain_bounds, domain_bounds):
        return -1
    current = -1
    for root_pos in range(int(root_cell_ids.shape[0])):
        root_cell_id = int(root_cell_ids[root_pos])
        if _contains_box_xyz(query_xyz, cell_bounds[root_cell_id], domain_bounds):
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
            if _contains_box_xyz(query_xyz, cell_bounds[child_id], domain_bounds):
                next_cell_id = child_id
                break
        if next_cell_id < 0:
            return int(current)
        current = int(next_cell_id)


@njit(cache=True)
def _cell_exit_event_xyz_raw(
    cell_bounds: np.ndarray,
    cell_id: int,
    current_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    t_current: float,
    active_faces: np.ndarray,
) -> tuple[float, int]:
    """Return one leaf exit event as `t_exit` plus active face ids."""
    bounds = cell_bounds[int(cell_id)]
    t_exit = 0.0
    n_active_face = 0
    have_candidate = False
    for axis in range(3):
        direction_value = float(direction_xyz[axis])
        if direction_value > 0.0:
            face_id = 2 * axis + 1
            face_value = float(bounds[axis, START] + bounds[axis, WIDTH])
        elif direction_value < 0.0:
            face_id = 2 * axis
            face_value = float(bounds[axis, START])
        else:
            continue
        t_face = float(t_current) + (face_value - float(current_xyz[axis])) / direction_value
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
def _fill_active_face_state_raw(
    active_faces: np.ndarray,
    n_active_face: int,
    current_face_id: int,
    active_face_by_axis: np.ndarray,
    active_face_order: np.ndarray,
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
def _event_on_face_raw(
    cell_bounds: np.ndarray,
    cell_id: int,
    face_id: int,
    event_xyz: np.ndarray,
) -> bool:
    """Return whether one event point lies on one Cartesian face patch carrier."""
    bounds = cell_bounds[int(cell_id)]
    axis = int(_FACE_AXIS[int(face_id)])
    side = int(_FACE_SIDE[int(face_id)])
    face_value = float(bounds[axis, START]) if side == 0 else float(bounds[axis, START] + bounds[axis, WIDTH])
    if float(event_xyz[axis]) != face_value:
        return False
    for tangential_axis in _FACE_TANGENTIAL_AXES[int(face_id)]:
        axis = int(tangential_axis)
        start = float(bounds[axis, START])
        stop = start + float(bounds[axis, WIDTH])
        value = float(event_xyz[axis])
        if value < start or value > stop:
            return False
    return True


@njit(cache=True)
def _event_subface_id_raw(
    cell_neighbor: np.ndarray,
    domain_bounds: np.ndarray,
    cell_bounds: np.ndarray,
    cell_id: int,
    face_id: int,
    active_face_by_axis: np.ndarray,
    active_face_order: np.ndarray,
    current_face_order: int,
    event_xyz: np.ndarray,
    direction_xyz: np.ndarray,
) -> int:
    """Return the destination-side owning face patch for one face event."""
    tangential_axes = _FACE_TANGENTIAL_AXES[int(face_id)]
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
            value = float(event_xyz[axis])
            current_start = float(current_bounds[axis, START])
            current_stop = current_start + float(current_bounds[axis, WIDTH])
            active_face_id = int(active_face_by_axis[axis])
            direction_value = float(direction_xyz[axis])
            implicit_active_side = -1
            if value == current_start and direction_value < 0.0:
                implicit_active_side = 0
            elif value == current_stop and direction_value > 0.0:
                implicit_active_side = 1
            if active_face_id >= 0 or implicit_active_side >= 0:
                neighbor_start = float(neighbor_bounds[axis, START])
                neighbor_stop = neighbor_start + float(neighbor_bounds[axis, WIDTH])
                if active_face_id >= 0:
                    active_side = int(_FACE_SIDE[active_face_id])
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
            start = float(neighbor_bounds[axis, START])
            stop = start + float(neighbor_bounds[axis, WIDTH])
            if direction_value > 0.0:
                if value < start or value >= stop:
                    contains = False
                    break
            elif direction_value < 0.0:
                if value <= start or value > stop:
                    contains = False
                    break
            else:
                if start == float(domain_bounds[axis, START]):
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
def _event_xyz_on_active_faces_raw(
    cell_bounds: np.ndarray,
    cell_id: int,
    active_faces: np.ndarray,
    n_active_face: int,
    current_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    delta_t: float,
    event_xyz: np.ndarray,
) -> None:
    """Write one snapped event point for one leaf exit event."""
    for axis in range(3):
        event_xyz[axis] = float(current_xyz[axis]) + float(delta_t) * float(direction_xyz[axis])
    bounds = cell_bounds[int(cell_id)]
    for face_pos in range(n_active_face):
        face_id = int(active_faces[face_pos])
        axis = int(_FACE_AXIS[int(face_id)])
        side = int(_FACE_SIDE[int(face_id)])
        if side == 0:
            event_xyz[axis] = float(bounds[axis, START])
        else:
            event_xyz[axis] = float(bounds[axis, START] + bounds[axis, WIDTH])


@njit(cache=True)
def _walk_event_faces_xyz_raw(
    cell_neighbor: np.ndarray,
    domain_bounds: np.ndarray,
    cell_bounds: np.ndarray,
    start_cell_id: int,
    active_faces: np.ndarray,
    n_active_face: int,
    event_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    path: np.ndarray,
    active_face_by_axis: np.ndarray,
    active_face_order: np.ndarray,
) -> int:
    """Walk one exact Cartesian event through the face/subface neighbor graph."""
    current_cell = int(start_cell_id)
    path_count = 0
    for face_pos in range(n_active_face):
        face_id = int(active_faces[face_pos])
        if current_cell < 0:
            break
        if not _event_on_face_raw(cell_bounds, current_cell, face_id, event_xyz):
            continue
        current_face_order = _fill_active_face_state_raw(
            active_faces,
            n_active_face,
            face_id,
            active_face_by_axis,
            active_face_order,
        )
        subface_id = _event_subface_id_raw(
            cell_neighbor,
            domain_bounds,
            cell_bounds,
            current_cell,
            face_id,
            active_face_by_axis,
            active_face_order,
            current_face_order,
            event_xyz,
            direction_xyz,
        )
        if subface_id < 0:
            return -1
        current_cell = int(cell_neighbor[current_cell, face_id, subface_id])
        path[path_count] = current_cell
        path_count += 1
    return int(path_count)


@njit(cache=True)
def _trace_one_xyz_ray_raw(
    root_cell_ids: np.ndarray,
    cell_child: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
    cell_neighbor: np.ndarray,
    start_cell_id: int,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    t_min: float,
    t_max: float,
    cell_ids_out: np.ndarray,
    times_out: np.ndarray,
) -> tuple[int, int]:
    """Trace one Cartesian ray into fixed scratch buffers."""
    has_interval, domain_enter, domain_exit = _domain_interval_xyz_raw(origin_xyz, direction_xyz, domain_bounds)
    if not has_interval:
        return 0, 0
    start_t = max(float(t_min), float(domain_enter))
    stop_t = min(float(t_max), float(domain_exit))
    if not (start_t < stop_t):
        return 0, 0
    start_xyz = np.empty(3, dtype=np.float64)
    current_xyz = np.empty(3, dtype=np.float64)
    event_xyz = np.empty(3, dtype=np.float64)
    active_faces = np.empty(3, dtype=np.int64)
    path = np.empty(3, dtype=np.int64)
    active_face_by_axis = np.empty(3, dtype=np.int64)
    active_face_order = np.empty(6, dtype=np.int64)
    for axis in range(3):
        start_xyz[axis] = float(origin_xyz[axis]) + float(start_t) * float(direction_xyz[axis])
        current_xyz[axis] = float(start_xyz[axis])
    current_cell = _find_cell_xyz_single(start_xyz, cell_child, root_cell_ids, cell_bounds, domain_bounds)
    if current_cell < 0:
        current_cell = int(start_cell_id)
    n_cell = 0
    n_time = 1
    times_out[0] = float(start_t)
    t_current = float(start_t)
    while current_cell >= 0 and t_current < stop_t:
        t_exit, n_active_face = _cell_exit_event_xyz_raw(
            cell_bounds,
            current_cell,
            current_xyz,
            direction_xyz,
            t_current,
            active_faces,
        )
        if n_active_face < 0:
            return -1, -1
        if t_exit > stop_t:
            if n_cell >= _MAX_RAY_TRACE_EVENTS:
                return -1, -1
            cell_ids_out[n_cell] = int(current_cell)
            times_out[n_time] = float(stop_t)
            n_cell += 1
            n_time += 1
            break
        if n_cell >= _MAX_RAY_TRACE_EVENTS:
            return -1, -1
        cell_ids_out[n_cell] = int(current_cell)
        times_out[n_time] = float(t_exit)
        n_cell += 1
        n_time += 1
        _event_xyz_on_active_faces_raw(
            cell_bounds,
            current_cell,
            active_faces,
            n_active_face,
            current_xyz,
            direction_xyz,
            float(t_exit) - float(t_current),
            event_xyz,
        )
        path_count = _walk_event_faces_xyz_raw(
            cell_neighbor,
            domain_bounds,
            cell_bounds,
            current_cell,
            active_faces,
            n_active_face,
            event_xyz,
            direction_xyz,
            path,
            active_face_by_axis,
            active_face_order,
        )
        if path_count < 0:
            return -1, -1
        for path_pos in range(path_count - 1):
            intermediate_cell = int(path[path_pos])
            if intermediate_cell < 0:
                break
            if n_cell >= _MAX_RAY_TRACE_EVENTS:
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
            current_xyz[axis] = float(event_xyz[axis])
    return int(n_cell), int(n_time)


@njit(cache=True, parallel=True)
def trace_xyz_ray_batch_scratch_raw(
    root_cell_ids: np.ndarray,
    cell_child: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
    cell_neighbor: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    t_min: float,
    t_max: float,
    cell_counts: np.ndarray,
    time_counts: np.ndarray,
    cell_ids_out: np.ndarray,
    times_out: np.ndarray,
) -> None:
    """Trace one Cartesian ray batch into fixed per-ray scratch buffers."""
    n_rays = int(origins.shape[0])
    for ray_id in prange(n_rays):
        n_cell, n_time = _trace_one_xyz_ray_raw(
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


@njit(cache=True, parallel=True)
def pack_xyz_ray_batch_raw(
    cell_counts: np.ndarray,
    time_counts: np.ndarray,
    cell_ids_scratch: np.ndarray,
    times_scratch: np.ndarray,
    cell_offsets: np.ndarray,
    time_offsets: np.ndarray,
    cell_ids_out: np.ndarray,
    times_out: np.ndarray,
) -> None:
    """Pack one scratch-traced Cartesian batch into flat output arrays."""
    n_rays = int(cell_counts.shape[0])
    for ray_id in prange(n_rays):
        cell_count = int(cell_counts[ray_id])
        cell_lo = int(cell_offsets[ray_id])
        for cell_pos in range(cell_count):
            cell_ids_out[cell_lo + cell_pos] = int(cell_ids_scratch[ray_id, cell_pos])
        time_count = int(time_counts[ray_id])
        time_lo = int(time_offsets[ray_id])
        for time_pos in range(time_count):
            times_out[time_lo + time_pos] = float(times_scratch[ray_id, time_pos])
