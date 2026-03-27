#!/usr/bin/env python3
"""Core octree data structures and shared lookup/interpolation utilities."""

from __future__ import annotations

import logging
from pathlib import Path
import time

import numpy as np
from numba import njit
from numba import prange

from .constants import DEFAULT_TREE_COORD
from .constants import SUPPORTED_TREE_COORDS
from .shared_types import GridShape
from .shared_types import LevelCountTable
from .shared_types import TreeCoord

logger = logging.getLogger(__name__)

AXIS0 = 0  # Packed bounds axis index for the first tree coordinate.
AXIS1 = 1  # Packed bounds axis index for the second tree coordinate.
AXIS2 = 2  # Packed bounds axis index for the third tree coordinate.
START = 0  # Packed bounds slot index for interval start.
WIDTH = 1  # Packed bounds slot index for interval width.

_CHILD_IJK_OFFSETS = np.array(
    [[(k >> 2) & 1, (k >> 1) & 1, k & 1] for k in range(8)],
    dtype=np.int64,
)


@njit(cache=True)
def _contains_box(
    q: np.ndarray,
    bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
    tol: float,
) -> bool:
    """Return whether one query lies inside one packed axis-0/1/2 box."""
    for axis in range(AXIS2):
        value = float(q[axis])
        start = float(bounds[axis, START])
        width = float(bounds[axis, WIDTH])
        if value < (start - tol) or value > (start + width + tol):
            return False
    value = float(q[AXIS2])
    start = float(bounds[AXIS2, START])
    width = float(bounds[AXIS2, WIDTH])
    if axis2_periodic:
        if width >= (float(axis2_period) - tol):
            return True
        return ((value - start) % float(axis2_period)) <= (width + tol)
    return value >= (start - tol) and value <= (start + width + tol)


@njit(cache=True, parallel=True)
def _find_cells(
    queries: np.ndarray,
    cell_child: np.ndarray,
    root_cell_ids: np.ndarray,
    cell_parent: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
) -> np.ndarray:
    """Resolve a batch of same-coordinate queries to containing cell ids."""
    n_query = int(queries.shape[0])
    cell_ids = np.full(n_query, -1, dtype=np.int64)
    chunk_size = 1024
    n_chunks = (n_query + chunk_size - 1) // chunk_size
    for chunk_id in prange(n_chunks):
        start = chunk_id * chunk_size
        end = min(n_query, start + chunk_size)
        hint_cell_id = -1
        for i in range(start, end):
            q = queries[i]
            if not (np.isfinite(q[AXIS0]) and np.isfinite(q[AXIS1]) and np.isfinite(q[AXIS2])):
                cell_id = -1
            elif not _contains_box(q, domain_bounds, axis2_period, axis2_periodic, tol=0.0):
                cell_id = -1
            else:
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
                    cell_id = -1
                else:
                    while np.any(cell_child[current] >= 0):
                        next_cell_id = -1
                        for child_ord in range(8):
                            child_id = int(cell_child[current, child_ord])
                            if child_id < 0:
                                continue
                            if _contains_box(q, cell_bounds[child_id], axis2_period, axis2_periodic, 1.0e-10):
                                next_cell_id = child_id
                                break
                        if next_cell_id < 0:
                            current = -1
                            break
                        current = next_cell_id
                    cell_id = int(current)
            cell_ids[i] = cell_id
            hint_cell_id = int(cell_id) if cell_id >= 0 else -1
    return cell_ids


def _cell_row_order(cell_depth: np.ndarray, cell_ijk: np.ndarray) -> np.ndarray:
    """Return sorted `(depth, axis0, axis1, axis2)` row order for packed cell addresses."""
    return np.lexsort(np.column_stack((cell_depth, cell_ijk))[:, ::-1].T)


def _pack_cell_keys(cell_depth: np.ndarray, cell_ijk: np.ndarray, axis_bases: np.ndarray) -> np.ndarray:
    """Pack `(depth, axis0, axis1, axis2)` rows into sortable integer keys."""
    depth = cell_depth.astype(np.uint64, copy=False)
    axis0 = cell_ijk[:, AXIS0].astype(np.uint64, copy=False)
    axis1 = cell_ijk[:, AXIS1].astype(np.uint64, copy=False)
    axis2 = cell_ijk[:, AXIS2].astype(np.uint64, copy=False)
    key = depth * np.uint64(axis_bases[AXIS0]) + axis0
    key = key * np.uint64(axis_bases[AXIS1]) + axis1
    key = key * np.uint64(axis_bases[AXIS2]) + axis2
    return key


def _unpack_cell_keys(keys: np.ndarray, axis_bases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack sortable integer keys into `(depth, cell_ijk)` arrays."""
    key = keys.astype(np.uint64, copy=True)
    axis2 = (key % np.uint64(axis_bases[AXIS2])).astype(np.int64)
    key //= np.uint64(axis_bases[AXIS2])
    axis1 = (key % np.uint64(axis_bases[AXIS1])).astype(np.int64)
    key //= np.uint64(axis_bases[AXIS1])
    axis0 = (key % np.uint64(axis_bases[AXIS0])).astype(np.int64)
    depth = (key // np.uint64(axis_bases[AXIS0])).astype(np.int64)
    return depth, np.column_stack((axis0, axis1, axis2))


def _rebuild_cells(
    depths: np.ndarray,
    cell_ijk: np.ndarray,
    leaf_value: np.ndarray,
    *,
    n_leaf_slots: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build rebuilt octree cell arrays from exact leaf addresses."""
    leaf_ijk_raw = cell_ijk
    if leaf_ijk_raw.ndim != 2 or leaf_ijk_raw.shape[1] != 3:
        raise ValueError("cell_ijk must have shape (n_cells, 3).")

    leaf_order = _cell_row_order(depths, leaf_ijk_raw)
    leaf_depth = depths[leaf_order]
    leaf_ijk = leaf_ijk_raw[leaf_order]

    same_leaf = (
        (leaf_depth[1:] == leaf_depth[:-1])
        & np.all(leaf_ijk[1:] == leaf_ijk[:-1], axis=1)
    )
    if np.any(same_leaf):
        dup = int(np.flatnonzero(same_leaf)[0])
        raise ValueError(
            "Cells overlap at octree address "
            f"{(int(leaf_depth[dup]), *(int(v) for v in leaf_ijk[dup]))}."
        )

    parent_key_parts: list[np.ndarray] = []
    axis_bases = np.max(leaf_ijk, axis=0).astype(np.uint64) + 1
    for shift in range(1, int(np.max(leaf_depth)) + 1):
        mask = leaf_depth >= shift
        if not np.any(mask):
            continue
        parent_key_parts.append(
            _pack_cell_keys(
                leaf_depth[mask] - shift,
                np.right_shift(leaf_ijk[mask], shift),
                axis_bases,
            )
        )

    if parent_key_parts:
        internal_keys = np.unique(np.concatenate(parent_key_parts))
        internal_depth, internal_ijk = _unpack_cell_keys(internal_keys, axis_bases)
    else:
        internal_depth = np.empty(0, dtype=np.int64)
        internal_ijk = np.empty((0, 3), dtype=np.int64)

    all_depth = np.concatenate((leaf_depth, internal_depth))
    all_ijk = np.concatenate((leaf_ijk, internal_ijk), axis=0)
    all_order = _cell_row_order(all_depth, all_ijk)
    all_depth = all_depth[all_order]
    all_ijk = all_ijk[all_order]

    same_cell = (
        (all_depth[1:] == all_depth[:-1])
        & np.all(all_ijk[1:] == all_ijk[:-1], axis=1)
    )
    if np.any(same_cell):
        dup = int(np.flatnonzero(same_cell)[0])
        raise ValueError(
            "Cells overlap across parent/child addresses at "
            f"{(int(all_depth[dup]), *(int(v) for v in all_ijk[dup]))}."
        )

    leaf_slots = int(np.max(leaf_value)) + 1 if n_leaf_slots is None and leaf_value.size else int(n_leaf_slots or 0)
    n_cells = leaf_slots + int(internal_depth.shape[0])
    cell_depth = np.full(n_cells, -1, dtype=np.int64)
    cell_ijk_out = np.full((n_cells, 3), -1, dtype=np.int64)
    cell_depth[leaf_value] = depths
    cell_ijk_out[leaf_value] = leaf_ijk_raw
    start = leaf_slots
    stop = start + int(internal_depth.shape[0])
    cell_depth[start:stop] = internal_depth
    cell_ijk_out[start:stop] = internal_ijk
    return cell_depth, cell_ijk_out


def _build_cell_topology(
    cell_depth: np.ndarray,
    cell_ijk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sparse 8-child references for occupied rebuilt cells."""
    n_cells = int(cell_depth.shape[0])
    cell_child = np.full((n_cells, 8), -1, dtype=np.int64)
    cell_parent = np.full(n_cells, -1, dtype=np.int64)
    occupied = np.flatnonzero(cell_depth >= 0).astype(np.int64)
    occupied_depth = cell_depth[occupied]
    occupied_ijk = cell_ijk[occupied]
    axis_bases = 2 * np.max(occupied_ijk, axis=0).astype(np.uint64) + 2
    occupied_keys = _pack_cell_keys(occupied_depth, occupied_ijk, axis_bases)
    order = np.argsort(occupied_keys)
    sorted_keys = occupied_keys[order]
    sorted_ids = occupied[order]
    for child_ord in range(8):
        child_depth = occupied_depth + 1
        child_ijk = 2 * occupied_ijk + _CHILD_IJK_OFFSETS[child_ord]
        child_keys = _pack_cell_keys(child_depth, child_ijk, axis_bases)
        child_pos = np.searchsorted(sorted_keys, child_keys)
        hits = child_pos < sorted_keys.size
        hits[hits] = sorted_keys[child_pos[hits]] == child_keys[hits]
        if not np.any(hits):
            continue
        parent_ids = occupied[hits]
        child_ids = sorted_ids[child_pos[hits]]
        cell_child[parent_ids, child_ord] = child_ids
        cell_parent[child_ids] = parent_ids
    root_cell_ids = np.flatnonzero(cell_depth == 0).astype(np.int64)
    return cell_child, root_cell_ids, cell_parent


def _rebuild_cell_state(
    cell_levels: np.ndarray,
    cell_ijk: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Rebuild exact occupied cells from leaf addresses."""

    if cell_ijk.ndim != 2 or cell_ijk.shape[1] != 3:
        raise ValueError("Octree leaf cell_ijk must have shape (n_cells, 3).")
    if cell_levels.shape[0] != cell_ijk.shape[0]:
        raise ValueError("Octree leaf level/index arrays must have matching shapes.")
    valid_ids = np.flatnonzero(cell_levels >= 0).astype(np.int64)
    if valid_ids.size == 0:
        raise ValueError("Octree state requires at least one valid leaf cell.")
    depths = cell_levels[valid_ids]
    leaf_ijk_valid = cell_ijk[valid_ids]
    cell_depth, cell_ijk_rt = _rebuild_cells(
        depths,
        leaf_ijk_valid,
        valid_ids,
        n_leaf_slots=int(cell_levels.shape[0]),
    )
    cell_child, root_cell_ids, cell_parent = _build_cell_topology(
        cell_depth,
        cell_ijk_rt,
    )
    return cell_depth, cell_ijk_rt, cell_child, root_cell_ids, cell_parent

class Octree:
    """Adaptive octree summary plus bound lookup entrypoints.

    `level_counts` rows are
    `(level, leaf_count, fine_equivalent_count)`.
    """

    def __init__(
        self,
        *,
        root_shape: GridShape,
        tree_coord: TreeCoord = DEFAULT_TREE_COORD,
        cell_levels: np.ndarray,
        cell_ijk: np.ndarray,
        points: np.ndarray,
        corners: np.ndarray,
    ) -> None:
        """Build one octree from exact leaf addresses plus explicit point/corner geometry."""
        leaf_levels: np.ndarray
        leaf_ijk: np.ndarray
        resolved_tree_coord = str(tree_coord)
        if resolved_tree_coord not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported tree_coord '{resolved_tree_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
            )
        self._root_shape = tuple(int(v) for v in root_shape)
        self._tree_coord = resolved_tree_coord
        leaf_levels = np.asarray(cell_levels, dtype=np.int64)
        leaf_ijk = np.asarray(cell_ijk, dtype=np.int64)
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (n_points, 3).")
        corner_rows = np.asarray(corners, dtype=np.int64)
        if corner_rows.ndim != 2 or corner_rows.shape != (leaf_levels.shape[0], 8):
            raise ValueError("corners must have shape (n_cells, 8) matching cell_levels.")
        max_level = int(np.max(leaf_levels))
        t0 = time.perf_counter()
        (
            self._cell_depth,
            self._cell_ijk,
            self._cell_child,
            self._root_cell_ids,
            self._cell_parent,
        ) = _rebuild_cell_state(
            leaf_levels,
            leaf_ijk,
        )
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "octree materialize: rebuild cell state complete (%.2fs) coord=%s max_level=%d",
                float(time.perf_counter() - t0),
                self._tree_coord,
                max_level,
            )
        self._leaf_slot_count = int(leaf_levels.shape[0])
        if self._tree_coord == "xyz":
            from .cartesian import _attach_cartesian_coord_state

            attach_coord_state = _attach_cartesian_coord_state
        else:
            from .spherical import _attach_spherical_coord_state

            attach_coord_state = _attach_spherical_coord_state
        t0 = time.perf_counter()
        cell_bounds, domain_bounds, axis2_period, axis2_periodic = attach_coord_state(self, points, corner_rows)
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "octree materialize: attach coord state complete (%.2fs) coord=%s",
                float(time.perf_counter() - t0),
                self._tree_coord,
            )
        self._corners = corner_rows
        self._cell_bounds = cell_bounds
        self._domain_bounds = domain_bounds
        self._axis2_period = float(axis2_period)
        self._axis2_periodic = bool(axis2_periodic)

    @property
    def root_shape(self) -> GridShape:
        """Return root-grid shape."""
        return self._root_shape

    @property
    def tree_coord(self) -> TreeCoord:
        """Return tree coordinate system."""
        return self._tree_coord

    @property
    def leaf_shape(self) -> GridShape:
        """Return finest leaf-grid shape."""
        scale = 1 << self.max_level
        return tuple(int(v) * scale for v in self._root_shape)

    @property
    def level_counts(self) -> LevelCountTable:
        """Return `(level, leaf_count, fine_equivalent_count)` rows."""
        valid_levels = self.cell_levels[self.cell_levels >= 0]
        max_level = int(np.max(valid_levels))
        return tuple(
            (
                int(level),
                int(np.count_nonzero(valid_levels == level)),
                int(np.count_nonzero(valid_levels == level) * (8 ** int(max_level - level))),
            )
            for level in sorted(set(int(v) for v in valid_levels.tolist()))
        )

    @property
    def min_level(self) -> int:
        """Return minimum occupied refinement level."""
        valid_levels = self.cell_levels[self.cell_levels >= 0]
        return int(np.min(valid_levels))

    @property
    def max_level(self) -> int:
        """Return maximum occupied refinement level."""
        valid_levels = self.cell_levels[self.cell_levels >= 0]
        return int(np.max(valid_levels))

    @property
    def corners(self) -> np.ndarray:
        """Return leaf-row corner point ids in Tecplot/BATSRUS brick order."""
        return self._corners

    @property
    def cell_bounds(self) -> np.ndarray:
        """Return packed `(n_cells, 3, 2)` start/width bounds for rebuilt cells."""
        return self._cell_bounds

    @property
    def cell_levels(self) -> np.ndarray:
        """Return exact persisted leaf-slot levels, including unused slots as `-1`."""
        return self._cell_depth[: self._leaf_slot_count]

    def save(self, path: str | Path) -> None:
        """Save this tree to a compressed `.npz` file."""
        from .persistence import OctreeState

        state = OctreeState.from_tree(self)
        out_path = Path(path)
        state.save_npz(out_path)
        logger.info("Saved octree to %s", str(out_path))

    @classmethod
    def from_state(
        cls,
        state: "OctreeState",
        *,
        points: np.ndarray,
        corners: np.ndarray,
    ) -> "Octree":
        """Instantiate one tree from exact saved state and explicit point/corner geometry."""
        return cls(
            root_shape=tuple(int(v) for v in state.root_shape),
            tree_coord=state.tree_coord,
            cell_levels=state.cell_levels,
            cell_ijk=state.cell_ijk,
            points=points,
            corners=corners,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        points: np.ndarray,
        corners: np.ndarray,
    ) -> "Octree":
        """Load a tree from `.npz` and bind it to explicit point/corner geometry."""
        from .persistence import OctreeState

        in_path = Path(path)
        state = OctreeState.load_npz(in_path)
        tree = cls.from_state(state, points=points, corners=corners)
        logger.info("Loaded octree from %s", str(in_path))
        return tree

    def __str__(self) -> str:
        """Return a compact human-readable tree summary."""
        leaf_levels = self.cell_levels
        n_leaf_cells = int(np.count_nonzero(leaf_levels >= 0))
        n_runtime_cells = int(np.count_nonzero(self._cell_depth >= 0))
        return (
            "Octree("
            f"tree_coord={self.tree_coord}, "
            f"root_shape={self.root_shape}, "
            f"leaf_shape={self.leaf_shape}, "
            f"leaf_cells={n_leaf_cells}, "
            f"runtime_cells={n_runtime_cells}, "
            f"levels={self.min_level}..{self.max_level}"
            ")"
        )

    @property
    def cell_count(self) -> int:
        """Return number of exact persisted leaf rows."""
        return int(self.cell_levels.shape[0])

    def lookup_points(self, points: np.ndarray, *, coord: TreeCoord) -> np.ndarray:
        """Resolve one batch of query points to leaf cell ids, with `-1` for misses."""
        q = np.array(points, dtype=np.float64, order="C")
        if q.ndim == 0 or q.shape[-1] != 3:
            raise ValueError("points must have shape (..., 3).")
        shape = (1,) if q.ndim == 1 else q.shape[:-1]
        q = q.reshape(-1, 3)
        resolved_coord = str(coord)
        if resolved_coord not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported lookup coord '{resolved_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
            )
        if self.tree_coord == "xyz":
            if resolved_coord != "xyz":
                raise ValueError("Cartesian lookup supports only coord='xyz'.")
            q_local = q
        elif resolved_coord == "rpa":
            q_local = q
        else:
            from .spherical import _xyz_arrays_to_rpa

            q_local = np.column_stack(_xyz_arrays_to_rpa(q[:, 0], q[:, 1], q[:, 2]))
        cell_ids = _find_cells(
            q_local,
            self._cell_child,
            self._root_cell_ids,
            self._cell_parent,
            self._cell_bounds,
            self._domain_bounds,
            self._axis2_period,
            self._axis2_periodic,
        )
        return cell_ids.reshape(shape)

    def domain_bounds(self, *, coord: TreeCoord = "xyz") -> tuple[np.ndarray, np.ndarray]:
        """Return global `(lo, hi)` bounds in the tree's own coordinate system."""
        resolved_coord = str(coord)
        if resolved_coord not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported lookup coord '{resolved_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
            )
        if resolved_coord != self.tree_coord:
            raise ValueError(f"domain_bounds only supports coord={self.tree_coord!r} for this tree.")
        lo = np.array(self._domain_bounds[:, START], dtype=float)
        hi = np.array(self._domain_bounds[:, START] + self._domain_bounds[:, WIDTH], dtype=float)
        return lo, hi
