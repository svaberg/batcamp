#!/usr/bin/env python3
"""Core octree data structures and shared lookup/interpolation utilities."""

from __future__ import annotations

from functools import wraps
import logging
from pathlib import Path
import time

import numpy as np
from batread import Dataset
from numba import njit
from numba import prange

from .constants import XYZ_VARS
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

def timed_info_decorator(func):
    """Decorate one function with a fixed elapsed-time INFO log line."""
    @wraps(func)
    def wrapped(*args, **kwargs):
        logger.debug("%s...", func.__name__)
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        logger.info("%s complete in %.2fs", func.__name__, float(time.perf_counter() - t0))
        return result

    return wrapped


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
    axis0, axis1, axis2 = cell_ijk.astype(np.uint64, copy=False).T
    base0, base1, base2 = axis_bases.astype(np.uint64, copy=False)
    key = depth * base0 + axis0
    key = key * base1 + axis1
    key = key * base2 + axis2
    return key


def _unpack_cell_keys(keys: np.ndarray, axis_bases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack sortable integer keys into `(depth, cell_ijk)` arrays."""
    key = keys.astype(np.uint64, copy=True)
    base0, base1, base2 = axis_bases.astype(np.uint64, copy=False)
    axis2 = (key % base2).astype(np.int64)
    key //= base2
    axis1 = (key % base1).astype(np.int64)
    key //= base1
    axis0 = (key % base0).astype(np.int64)
    depth = (key // base0).astype(np.int64)
    return depth, np.column_stack((axis0, axis1, axis2))


@timed_info_decorator
def _rebuild_cells(
    depths: np.ndarray,
    cell_ijk: np.ndarray,
    leaf_value: np.ndarray,
    *,
    n_leaf_slots: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build rebuilt octree cell arrays from exact leaf addresses."""
    leaf_ijk_raw = cell_ijk
    if leaf_ijk_raw.ndim != 2 or leaf_ijk_raw.shape[1] != 3:
        raise ValueError("cell_ijk must have shape (n_cells, 3).")

    axis_bases = np.max(leaf_ijk_raw, axis=0).astype(np.uint64) + 1
    leaf_keys = _pack_cell_keys(depths, leaf_ijk_raw, axis_bases)
    leaf_order = np.argsort(leaf_keys)
    sorted_leaf_keys = leaf_keys[leaf_order]

    same_leaf = sorted_leaf_keys[1:] == sorted_leaf_keys[:-1]
    if np.any(same_leaf):
        dup = int(np.flatnonzero(same_leaf)[0])
        dup_depth, dup_ijk = _unpack_cell_keys(sorted_leaf_keys[dup : dup + 1], axis_bases)
        raise ValueError(
            "Cells overlap at octree address "
            f"{(int(dup_depth[0]), *(int(v) for v in dup_ijk[0]))}."
        )

    frontier_depth = depths
    frontier_ijk = leaf_ijk_raw
    parent_key_parts: list[np.ndarray] = []
    while True:
        mask = frontier_depth > 0
        if not np.any(mask):
            break
        frontier_depth = frontier_depth[mask] - 1
        frontier_ijk = np.right_shift(frontier_ijk[mask], 1)
        frontier_keys = np.unique(_pack_cell_keys(frontier_depth, frontier_ijk, axis_bases))
        parent_key_parts.append(frontier_keys)
        frontier_depth, frontier_ijk = _unpack_cell_keys(frontier_keys, axis_bases)

    if parent_key_parts:
        internal_keys = np.unique(np.concatenate(parent_key_parts))
        parent_pos = np.searchsorted(sorted_leaf_keys, internal_keys)
        hits = parent_pos < sorted_leaf_keys.size
        hits[hits] = sorted_leaf_keys[parent_pos[hits]] == internal_keys[hits]
        if np.any(hits):
            dup_depth, dup_ijk = _unpack_cell_keys(
                internal_keys[np.flatnonzero(hits)[:1]],
                axis_bases,
            )
            raise ValueError(
                "Cells overlap across parent/child addresses at "
                f"{(int(dup_depth[0]), *(int(v) for v in dup_ijk[0]))}."
            )
        internal_depth, internal_ijk = _unpack_cell_keys(internal_keys, axis_bases)
    else:
        internal_keys = np.empty(0, dtype=np.uint64)
        internal_depth = np.empty(0, dtype=np.int64)
        internal_ijk = np.empty((0, 3), dtype=np.int64)

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
    return cell_depth, cell_ijk_out, internal_keys, internal_depth, internal_ijk


@timed_info_decorator
def _build_cell_topology(
    depths: np.ndarray,
    leaf_ijk: np.ndarray,
    leaf_value: np.ndarray,
    internal_keys: np.ndarray,
    internal_depth: np.ndarray,
    internal_ijk: np.ndarray,
    leaf_slots: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sparse 8-child references for occupied rebuilt cells."""
    n_cells = int(leaf_slots) + int(internal_depth.shape[0])
    cell_child = np.full((n_cells, 8), -1, dtype=np.int64)
    cell_parent = np.full(n_cells, -1, dtype=np.int64)
    axis_bases = np.max(leaf_ijk, axis=0).astype(np.uint64) + 1

    leaf_mask = depths > 0
    leaf_child_ids = leaf_value[leaf_mask]
    leaf_parent_keys = _pack_cell_keys(
        depths[leaf_mask] - 1,
        np.right_shift(leaf_ijk[leaf_mask], 1),
        axis_bases,
    )
    leaf_child_ord = (
        ((leaf_ijk[leaf_mask, AXIS0] & 1) << 2)
        | ((leaf_ijk[leaf_mask, AXIS1] & 1) << 1)
        | (leaf_ijk[leaf_mask, AXIS2] & 1)
    ).astype(np.int64)

    internal_mask = internal_depth > 0
    internal_child_ids = int(leaf_slots) + np.flatnonzero(internal_mask).astype(np.int64)
    internal_parent_keys = _pack_cell_keys(
        internal_depth[internal_mask] - 1,
        np.right_shift(internal_ijk[internal_mask], 1),
        axis_bases,
    )
    internal_child_ord = (
        ((internal_ijk[internal_mask, AXIS0] & 1) << 2)
        | ((internal_ijk[internal_mask, AXIS1] & 1) << 1)
        | (internal_ijk[internal_mask, AXIS2] & 1)
    ).astype(np.int64)

    child_ids = np.concatenate((leaf_child_ids, internal_child_ids))
    parent_keys = np.concatenate((leaf_parent_keys, internal_parent_keys))
    child_ord = np.concatenate((leaf_child_ord, internal_child_ord))
    parent_pos = np.searchsorted(internal_keys, parent_keys)
    hits = parent_pos < internal_keys.size
    hits[hits] = internal_keys[parent_pos[hits]] == parent_keys[hits]
    parent_ids = int(leaf_slots) + parent_pos[hits]
    child_ids = child_ids[hits]
    child_ord = child_ord[hits]
    cell_child[parent_ids, child_ord] = child_ids
    cell_parent[child_ids] = parent_ids
    root_cell_ids = np.concatenate(
        (
            leaf_value[depths == 0],
            int(leaf_slots) + np.flatnonzero(internal_depth == 0).astype(np.int64),
        )
    )
    return cell_child, root_cell_ids, cell_parent


def _rebuild_cell_state(
    cell_levels: np.ndarray,
    cell_ijk: np.ndarray,
    tree_coord: str,
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
    cell_depth, cell_ijk_rt, internal_keys, internal_depth, internal_ijk = _rebuild_cells(
        depths,
        leaf_ijk_valid,
        valid_ids,
        n_leaf_slots=int(cell_levels.shape[0]),
    )
    cell_child, root_cell_ids, cell_parent = _build_cell_topology(
        depths,
        leaf_ijk_valid,
        valid_ids,
        internal_keys,
        internal_depth,
        internal_ijk,
        int(cell_levels.shape[0]),
    )
    return cell_depth, cell_ijk_rt, cell_child, root_cell_ids, cell_parent

class Octree:
    """Adaptive octree summary plus bound lookup entrypoints.

    `level_counts` rows are
    `(level, leaf_count, fine_equivalent_count)`.
    """

    def __init__(
        self,
        points: np.ndarray,
        corners: np.ndarray,
        *,
        tree_coord: TreeCoord | None = None,
        axis_rho_tol: float = 1e-12,
        level_rtol: float = 1e-4,
        level_atol: float = 1e-9,
    ) -> None:
        """Build one octree from explicit point coordinates and cell corners."""
        from .builder import _build_octree_state

        state = _build_octree_state(
            points,
            corners,
            tree_coord=tree_coord,
            axis_rho_tol=axis_rho_tol,
            level_rtol=level_rtol,
            level_atol=level_atol,
            cell_levels=None,
        )
        logger.info("_init_from_state: coord=%s", state.tree_coord)
        logger.debug("_init_from_state...")
        t0 = time.perf_counter()
        self._init_from_state(
            root_shape=state.root_shape,
            tree_coord=state.tree_coord,
            cell_levels=state.cell_levels,
            cell_ijk=state.cell_ijk,
            points=points,
            corners=corners,
        )
        logger.info("_init_from_state complete in %.2fs", float(time.perf_counter() - t0))

    def _init_from_state(
        self,
        *,
        root_shape: GridShape,
        tree_coord: TreeCoord,
        cell_levels: np.ndarray,
        cell_ijk: np.ndarray,
        points: np.ndarray,
        corners: np.ndarray,
    ) -> None:
        """Materialize one octree from exact leaf addresses and explicit geometry."""
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
        logger.info("_rebuild_cell_state: coord=%s max_level=%d", self._tree_coord, max_level)
        logger.debug("_rebuild_cell_state...")
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
            self._tree_coord,
        )
        logger.info("_rebuild_cell_state complete in %.2fs", float(time.perf_counter() - t0))
        self._leaf_slot_count = int(leaf_levels.shape[0])
        if self._tree_coord == "xyz":
            from .cartesian import _attach_cartesian_coord_state

            attach_coord_state = _attach_cartesian_coord_state
        else:
            from .spherical import _attach_spherical_coord_state

            attach_coord_state = _attach_spherical_coord_state
        logger.info("%s: coord=%s", attach_coord_state.__name__, self._tree_coord)
        logger.debug("%s...", attach_coord_state.__name__)
        t0 = time.perf_counter()
        cell_bounds, domain_bounds, axis2_period, axis2_periodic = attach_coord_state(self, points, corner_rows)
        logger.info("%s complete in %.2fs", attach_coord_state.__name__, float(time.perf_counter() - t0))
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

    @property
    def cell_depth(self) -> np.ndarray:
        """Return rebuilt runtime cell depths."""
        return self._cell_depth

    @property
    def cell_ijk(self) -> np.ndarray:
        """Return rebuilt runtime cell addresses."""
        return self._cell_ijk

    @property
    def radial_edges(self) -> np.ndarray:
        """Return finest radial edge locations for spherical trees."""
        return self._radial_edges

    @radial_edges.setter
    def radial_edges(self, radial_edges: np.ndarray) -> None:
        """Store finest radial edge locations for spherical trees."""
        self._radial_edges = radial_edges

    def save(self, path: str | Path) -> None:
        """Save this tree to a compressed `.npz` file."""
        from .persistence import OctreeState

        state = OctreeState.from_tree(self)
        out_path = Path(path)
        state.save_npz(out_path)
        logger.info("Saved octree to %s", str(out_path))

    @classmethod
    def from_ds(
        cls,
        ds: Dataset,
        *,
        tree_coord: TreeCoord | None = None,
        axis_rho_tol: float = 1e-12,
        level_rtol: float = 1e-4,
        level_atol: float = 1e-9,
    ) -> "Octree":
        """Build one tree from a dataset by extracting explicit points and corners."""
        from .builder import _warn_if_blocks_aux_mismatch

        if ds.corners is None:
            raise ValueError("Dataset has no corners; cannot build octree.")
        logger.debug("from_ds...")
        t0 = time.perf_counter()
        points = np.column_stack(tuple(np.asarray(ds[name], dtype=np.float64) for name in XYZ_VARS))
        corners = np.asarray(ds.corners, dtype=np.int64)
        _warn_if_blocks_aux_mismatch(ds, int(corners.shape[0]))
        logger.info(
            "from_ds: n_points=%d n_cells=%d",
            int(points.shape[0]),
            int(corners.shape[0]),
        )
        tree = cls(
            points,
            corners,
            tree_coord=tree_coord,
            axis_rho_tol=axis_rho_tol,
            level_rtol=level_rtol,
            level_atol=level_atol,
        )
        logger.info("from_ds complete in %.2fs", float(time.perf_counter() - t0))
        return tree

    @classmethod
    def from_state(
        cls,
        state: "OctreeState",
        *,
        points: np.ndarray,
        corners: np.ndarray,
    ) -> "Octree":
        """Instantiate one tree from exact saved state and explicit point/corner geometry."""
        tree = cls.__new__(cls)
        tree._init_from_state(
            root_shape=tuple(int(v) for v in state.root_shape),
            tree_coord=state.tree_coord,
            cell_levels=state.cell_levels,
            cell_ijk=state.cell_ijk,
            points=points,
            corners=corners,
        )
        return tree

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
            from .spherical import xyz_arrays_to_rpa

            q_local = np.column_stack(xyz_arrays_to_rpa(q[:, 0], q[:, 1], q[:, 2]))
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
