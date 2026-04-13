#!/usr/bin/env python3
"""Small Cartesian event-walk prototype that uses refined face/subface neighbors."""

from __future__ import annotations

import math

import numpy as np

from .octree import Octree
from .octree import START
from .octree import WIDTH
from .octree import _FACE_AXIS
from .octree import _FACE_SIDE
from .octree import _FACE_TANGENTIAL_AXES


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
    bounds = cell_bounds[int(cell_id)]
    candidates: list[tuple[float, int]] = []
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
            raise ValueError("Current leaf does not match the ray state.")
        candidates.append((float(t_face), int(face_id)))
    if not candidates:
        raise ValueError("direction must leave the current leaf along at least one axis.")
    t_exit = min(t_face for t_face, _ in candidates)
    active_faces = tuple(face_id for t_face, face_id in sorted(candidates, key=lambda item: item[1]) if t_face == t_exit)
    return float(t_exit), active_faces


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
    tangential_axes = _FACE_TANGENTIAL_AXES[int(face_id)]
    active_face_by_axis = {
        int(_FACE_AXIS[int(active_face_id)]): int(active_face_id)
        for active_face_id in active_faces
    }
    active_face_order = {
        int(active_face_id): order
        for order, active_face_id in enumerate(active_faces)
    }
    row = np.asarray(cell_neighbor[int(cell_id), int(face_id)], dtype=np.int64)
    nonnegative = row[row >= 0]
    if nonnegative.size == 0:
        return 0
    if np.unique(nonnegative).size == 1:
        return int(np.flatnonzero(row >= 0)[0])
    matched_subface = -1
    matched_neighbor = -1
    negative_only = True
    current_bounds = cell_bounds[int(cell_id)]
    for subface_id in range(4):
        neighbor_id = int(cell_neighbor[int(cell_id), int(face_id), subface_id])
        if neighbor_id < 0:
            continue
        negative_only = False
        contains = True
        neighbor_bounds = cell_bounds[neighbor_id]
        for axis in tangential_axes:
            axis = int(axis)
            value = float(event_xyz[axis])
            current_start = float(current_bounds[axis, START])
            current_stop = current_start + float(current_bounds[axis, WIDTH])
            active_face_id = active_face_by_axis.get(axis)
            direction_value = float(direction_xyz[axis])
            implicit_active_side = None
            if value == current_start and direction_value < 0.0:
                implicit_active_side = 0
            elif value == current_stop and direction_value > 0.0:
                implicit_active_side = 1
            if active_face_id is not None or implicit_active_side is not None:
                neighbor_start = float(neighbor_bounds[axis, START])
                neighbor_stop = neighbor_start + float(neighbor_bounds[axis, WIDTH])
                if active_face_id is not None:
                    active_side = int(_FACE_SIDE[active_face_id])
                    if active_face_order[active_face_id] < active_face_order[int(face_id)]:
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
            raise ValueError("Event face patch is ambiguous under destination-side ownership.")
    if negative_only:
        return 0
    if matched_subface < 0:
        raise ValueError("Event face patch has no destination-side owner.")
    return int(matched_subface)


def _event_on_face(
    cell_bounds: np.ndarray,
    cell_id: int,
    face_id: int,
    event_xyz: np.ndarray,
) -> bool:
    """Return whether one event point lies on one Cartesian leaf face."""
    bounds = cell_bounds[int(cell_id)]
    axis = int(_FACE_AXIS[int(face_id)])
    side = int(_FACE_SIDE[int(face_id)])
    face_value = float(bounds[axis, START]) if side == 0 else float(bounds[axis, START] + bounds[axis, WIDTH])
    if float(event_xyz[axis]) != face_value:
        return False
    for tangential_axis in _FACE_TANGENTIAL_AXES[int(face_id)]:
        tangential_axis = int(tangential_axis)
        start = float(bounds[tangential_axis, START])
        stop = start + float(bounds[tangential_axis, WIDTH])
        value = float(event_xyz[tangential_axis])
        if value < start or value > stop:
            return False
    return True


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
    current_cell = int(start_cell_id)
    path: list[int] = []
    cell_bounds = tree.cell_bounds
    cell_neighbor = tree.cell_neighbor
    domain_bounds = tree._domain_bounds
    for face_id in active_faces:
        if current_cell < 0:
            break
        if not _event_on_face(cell_bounds, current_cell, int(face_id), event_xyz):
            continue
        subface_id = _event_subface_id(
            cell_neighbor,
            domain_bounds,
            cell_bounds,
            current_cell,
            int(face_id),
            active_faces,
            event_xyz,
            direction_xyz,
        )
        current_cell = int(cell_neighbor[current_cell, int(face_id), subface_id])
        path.append(current_cell)
    return tuple(path)


def _event_xyz_on_active_faces(
    cell_bounds: np.ndarray,
    cell_id: int,
    active_faces: tuple[int, ...],
    current_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    delta_t: float,
) -> np.ndarray:
    """Return one event point snapped onto the crossed faces of the current cell."""
    event_xyz = np.asarray(current_xyz, dtype=float) + float(delta_t) * np.asarray(direction_xyz, dtype=float)
    bounds = cell_bounds[int(cell_id)]
    for face_id in active_faces:
        axis = int(_FACE_AXIS[int(face_id)])
        side = int(_FACE_SIDE[int(face_id)])
        if side == 0:
            event_xyz[axis] = float(bounds[axis, START])
        else:
            event_xyz[axis] = float(bounds[axis, START] + bounds[axis, WIDTH])
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

    domain_lo, domain_hi = tree.domain_bounds(coord="xyz")
    interval = _domain_interval_xyz(origin, direction, domain_lo, domain_hi)
    if interval is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    domain_enter, domain_exit = interval
    start_t = max(float(t_min), float(domain_enter))
    stop_t = min(float(t_max), float(domain_exit))
    if not start_t < stop_t:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    start_xyz = origin + start_t * direction
    current_cell = int(tree.lookup_points(start_xyz[None, :], coord="xyz")[0])
    if current_cell < 0:
        current_cell = int(start_cell_id)
    cell_ids: list[int] = []
    times: list[float] = [float(start_t)]
    t_current = float(start_t)
    current_xyz = np.array(start_xyz, dtype=float)
    while current_cell >= 0 and t_current < stop_t:
        t_exit, active_faces = _cell_exit_event_xyz(tree.cell_bounds, current_cell, current_xyz, direction, t_current)
        if t_exit > stop_t:
            cell_ids.append(current_cell)
            times.append(float(stop_t))
            break
        cell_ids.append(current_cell)
        times.append(float(t_exit))
        event_xyz = _event_xyz_on_active_faces(
            tree.cell_bounds,
            current_cell,
            active_faces,
            current_xyz,
            direction,
            float(t_exit) - float(t_current),
        )
        path = walk_event_faces_xyz(tree, current_cell, active_faces, event_xyz, direction)
        for intermediate_cell in path[:-1]:
            if intermediate_cell < 0:
                break
            cell_ids.append(intermediate_cell)
            times.append(float(t_exit))
        current_cell = path[-1] if path else -1
        t_current = float(t_exit)
        current_xyz = np.array(event_xyz, dtype=float)
    return np.asarray(cell_ids, dtype=np.int64), np.asarray(times, dtype=np.float64)
