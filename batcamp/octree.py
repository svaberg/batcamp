#!/usr/bin/env python3
"""Core octree data structures and shared lookup/interpolation utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from batread import Dataset
from numba import njit
from numba import prange

from .constants import DEFAULT_TREE_COORD
from .constants import SUPPORTED_TREE_COORDS
from .constants import XYZ_VARS
from .shared_types import GridShape
from .shared_types import LevelCountTable
from .shared_types import TreeCoord

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .face_neighbors import OctreeFaceNeighbors

AXIS0 = 0  # Packed bounds axis index for the first tree coordinate.
AXIS1 = 1  # Packed bounds axis index for the second tree coordinate.
AXIS2 = 2  # Packed bounds axis index for the third tree coordinate.
START = 0  # Packed bounds slot index for interval start.
WIDTH = 1  # Packed bounds slot index for interval width.


_TRILINEAR_TARGET_BITS = np.array(
    [[k & 1, (k >> 1) & 1, (k >> 2) & 1] for k in range(8)],
    dtype=np.int8,
)
_CHILD_IJK_OFFSETS = np.array(
    [[(k >> 2) & 1, (k >> 1) & 1, k & 1] for k in range(8)],
    dtype=np.int64,
)

def _bound_xyz_and_leaf_levels(
    tree: "Octree",
    ds: Dataset,
    corners: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the bound dataset arrays and exact leaf levels needed for coordinate state."""
    x, y, z = (np.asarray(ds[name], dtype=np.float64) for name in XYZ_VARS)
    return corners, x, y, z, tree.cell_levels


def _trilinear_corner_order(
    corner_axes: np.ndarray,
    *,
    cell_bounds: np.ndarray,
    axis2_periodic: bool,
) -> np.ndarray:
    """Build logical trilinear corner indices from per-corner axis coordinates."""
    tiny = np.finfo(float).tiny
    axis0 = corner_axes[:, :, AXIS0]
    axis1 = corner_axes[:, :, AXIS1]
    axis2 = corner_axes[:, :, AXIS2]
    axis0_start = cell_bounds[:, AXIS0, START]
    axis0_width = cell_bounds[:, AXIS0, WIDTH]
    axis1_start = cell_bounds[:, AXIS1, START]
    axis1_width = cell_bounds[:, AXIS1, WIDTH]
    axis2_start = cell_bounds[:, AXIS2, START]
    axis2_width = cell_bounds[:, AXIS2, WIDTH]

    axis0_mid = axis0_start[:, None] + 0.5 * axis0_width[:, None]
    axis1_mid = axis1_start[:, None] + 0.5 * axis1_width[:, None]
    bit0 = (axis0 >= axis0_mid).astype(np.int8)
    bit1 = (axis1 >= axis1_mid).astype(np.int8)
    if axis2_periodic:
        axis2_full = axis2_width >= (2.0 * np.pi - 1.0e-10)
        axis2_tiny = axis2_width <= tiny
        axis2_rel = np.mod(axis2 - axis2_start[:, None], 2.0 * np.pi)
        axis2_rel = np.where(
            (~axis2_full)[:, None],
            np.clip(axis2_rel, 0.0, axis2_width[:, None]),
            axis2_rel,
        )
        axis2_mid = 0.5 * axis2_width[:, None]
        bit2 = np.zeros_like(bit0, dtype=np.int8)
        valid_axis2 = ~axis2_tiny
        if np.any(valid_axis2):
            bit2[valid_axis2] = (axis2_rel[valid_axis2] >= axis2_mid[valid_axis2]).astype(np.int8)
    else:
        axis2_mid = axis2_start[:, None] + 0.5 * axis2_width[:, None]
        bit2 = (axis2 >= axis2_mid).astype(np.int8)

    bin_id = bit0 + (bit1 << 1) + (bit2 << 2)
    bit_trip = np.stack((bit0, bit1, bit2), axis=2)
    n_cells = int(axis0.shape[0])
    bin_to_corner = np.empty((n_cells, 8), dtype=np.int64)
    for k in range(8):
        eq = bin_id == k
        has = np.any(eq, axis=1)
        pick = np.argmax(eq, axis=1).astype(np.int64)
        missing = ~has
        if np.any(missing):
            d = np.sum((bit_trip[missing] - _TRILINEAR_TARGET_BITS[k]) ** 2, axis=2)
            pick[missing] = np.argmin(d, axis=1)
        bin_to_corner[:, k] = pick
    return bin_to_corner


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
    """Return lexicographic `(depth, axis0, axis1, axis2)` row order for packed cell addresses."""
    return np.lexsort(np.column_stack((cell_depth, cell_ijk))[:, ::-1].T)


def _rebuild_cells(
    depths: np.ndarray,
    cell_ijk: np.ndarray,
    leaf_value: np.ndarray,
    *,
    tree_depth: int,
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

    parent_depth_parts: list[np.ndarray] = []
    parent_ijk_parts: list[np.ndarray] = []
    for parent_depth in range(int(tree_depth)):
        mask = depths > int(parent_depth)
        if not np.any(mask):
            continue
        up = depths[mask] - int(parent_depth)
        parent_ijk = np.right_shift(leaf_ijk_raw[mask], up[:, None])
        parent_cells = np.column_stack(
            (
                np.full(int(np.count_nonzero(mask)), int(parent_depth), dtype=np.int64),
                parent_ijk,
            )
        )
        parent_cells = np.unique(parent_cells, axis=0)
        parent_depth_parts.append(parent_cells[:, 0].astype(np.int64, copy=False))
        parent_ijk_parts.append(parent_cells[:, 1:].astype(np.int64, copy=False))

    if parent_depth_parts:
        internal_depth = np.concatenate(parent_depth_parts)
        internal_ijk = np.concatenate(parent_ijk_parts, axis=0)
        internal_order = _cell_row_order(internal_depth, internal_ijk)
        internal_depth = internal_depth[internal_order]
        internal_ijk = internal_ijk[internal_order]
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
    key_to_cell = {
        (int(cell_depth[idx]), *(int(v) for v in cell_ijk[idx])): int(idx)
        for idx in occupied
    }
    for idx in occupied:
        depth = int(cell_depth[idx])
        parent_ijk = cell_ijk[idx]
        for child_ord in range(8):
            child_ijk = 2 * parent_ijk + _CHILD_IJK_OFFSETS[child_ord]
            child_key = (depth + 1, *(int(v) for v in child_ijk))
            child_idx = key_to_cell.get(child_key)
            if child_idx is not None:
                cell_child[idx, child_ord] = int(child_idx)
                cell_parent[int(child_idx)] = int(idx)
    root_cell_ids = np.flatnonzero(cell_depth == 0).astype(np.int64)
    return cell_child, root_cell_ids, cell_parent


def _rebuild_cell_state(
    cell_levels: np.ndarray,
    cell_ijk: np.ndarray,
    *,
    max_level: int,
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
        tree_depth=int(max_level),
        n_leaf_slots=int(cell_levels.shape[0]),
    )
    cell_child, root_cell_ids, cell_parent = _build_cell_topology(
        cell_depth,
        cell_ijk_rt,
    )
    return cell_depth, cell_ijk_rt, cell_child, root_cell_ids, cell_parent

def _build_trilinear_geometry(
    tree: "Octree",
    ds: Dataset,
    corners_all: np.ndarray,
    cell_bounds: np.ndarray,
) -> np.ndarray:
    """Build the leaf-cell trilinear interpolation arrays from the bound dataset."""
    leaf_cell_ids = np.flatnonzero(tree.cell_levels >= 0).astype(np.int64)
    corners = corners_all[leaf_cell_ids]
    leaf_bounds = cell_bounds[leaf_cell_ids]
    x = np.asarray(ds[XYZ_VARS[0]], dtype=np.float64)
    y = np.asarray(ds[XYZ_VARS[1]], dtype=np.float64)
    z = np.asarray(ds[XYZ_VARS[2]], dtype=np.float64)
    if tree.tree_coord == "xyz":
        corner_axes = np.stack((x[corners], y[corners], z[corners]), axis=2)
        axis2_periodic = False
    else:
        from .spherical import _xyz_arrays_to_rpa

        point_r, point_p, point_a = _xyz_arrays_to_rpa(x, y, z)
        corner_axes = np.stack((point_r[corners], point_p[corners], point_a[corners]), axis=2)
        axis2_periodic = True
    leaf_bin_to_corner = _trilinear_corner_order(
        corner_axes,
        cell_bounds=leaf_bounds,
        axis2_periodic=axis2_periodic,
    )
    n_leaf_slots = int(corners_all.shape[0])
    interp_corners = np.full((n_leaf_slots, 8), -1, dtype=np.int64)
    interp_corners[leaf_cell_ids] = np.take_along_axis(corners, leaf_bin_to_corner, axis=1)
    return interp_corners


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
        ds: Dataset,
    ) -> None:
        """Build one octree directly from exact leaf addresses."""
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

        valid_levels = leaf_levels[leaf_levels >= 0]
        if valid_levels.size == 0:
            raise ValueError("Octree state requires at least one valid cell level.")
        max_level = int(np.max(valid_levels))
        (
            self._cell_depth,
            self._cell_ijk,
            self._cell_child,
            self._root_cell_ids,
            self._cell_parent,
        ) = _rebuild_cell_state(
            leaf_levels,
            leaf_ijk,
            max_level=max_level,
        )
        self._leaf_slot_count = int(leaf_levels.shape[0])
        if ds.corners is None:
            raise ValueError("Dataset has no corners; cannot bind octree lookup.")
        if not set(XYZ_VARS).issubset(set(ds.variables)):
            raise ValueError("Dataset must provide X/Y/Z variables to bind octree lookup.")
        corners = np.asarray(ds.corners, dtype=np.int64)
        cell_bounds, domain_bounds, axis2_period, axis2_periodic = self._coord_backend(str(self._tree_coord))._attach_coord_state(
            self, ds, corners
        )
        interp_corners = _build_trilinear_geometry(self, ds, corners, cell_bounds)
        self._ds = ds
        self._corners = interp_corners
        self._cell_bounds = cell_bounds
        self._domain_bounds = domain_bounds
        self._axis2_period = float(axis2_period)
        self._axis2_periodic = bool(axis2_periodic)
        self._face_neighbors_by_max_level: dict[int, OctreeFaceNeighbors] = {}

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
    def ds(self) -> Dataset:
        """Return bound dataset."""
        return self._ds

    @property
    def corners(self) -> np.ndarray:
        """Return logical trilinear corner point ids for leaf rows."""
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
        ds: Dataset,
    ) -> "Octree":
        """Instantiate one tree from exact saved state."""
        return cls(
            root_shape=tuple(int(v) for v in state.root_shape),
            tree_coord=state.tree_coord,
            cell_levels=state.cell_levels,
            cell_ijk=state.cell_ijk,
            ds=ds,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        ds: Dataset,
    ) -> "Octree":
        """Load a tree from `.npz` and bind it to the given dataset."""
        from .persistence import OctreeState

        in_path = Path(path)
        state = OctreeState.load_npz(in_path)
        tree = cls.from_state(state, ds=ds)
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

    @staticmethod
    def _coord_backend(tree_coord: str) -> type:
        """Return the geometry-specific support class for one tree coordinate."""
        if tree_coord == "xyz":
            from .cartesian import _CartesianCoordSupport

            return _CartesianCoordSupport
        if tree_coord == "rpa":
            from .spherical import _SphericalCoordSupport

            return _SphericalCoordSupport
        raise ValueError(
            f"Unsupported tree_coord '{tree_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
        )

    @property
    def cell_count(self) -> int:
        """Return number of exact persisted leaf rows."""
        return int(self.cell_levels.shape[0])

    def lookup_points(self, points: np.ndarray, *, coord: TreeCoord) -> np.ndarray:
        """Resolve one batch of query points to leaf cell ids, with `-1` for misses."""
        q = np.array(points, dtype=np.float64, order="C")
        if q.ndim == 1:
            if q.size != 3:
                raise ValueError("points must have shape (..., 3).")
            shape = (1,)
            q = q.reshape(1, 3)
        else:
            if q.shape[-1] != 3:
                raise ValueError("points must have shape (..., 3).")
            shape = q.shape[:-1]
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

    def face_neighbors(self, *, max_level: int | None = None) -> "OctreeFaceNeighbors":
        """Return the lazily built face-neighbor graph for one level cutoff."""
        valid_levels = self.cell_levels[self.cell_levels >= 0]
        target_max_level = int(np.max(valid_levels) if max_level is None else max_level)
        cache = self._face_neighbors_by_max_level
        face_neighbors = cache.get(target_max_level)
        if face_neighbors is None:
            from .face_neighbors import _build_face_neighbors

            face_neighbors = _build_face_neighbors(
                tree=self,
                target_max_level=target_max_level,
            )
            cache[int(face_neighbors.max_level)] = face_neighbors
        return face_neighbors

    def domain_bounds(self, *, coord: TreeCoord = "xyz") -> tuple[np.ndarray, np.ndarray]:
        """Return global `(lo, hi)` bounds for the bound tree in requested coord."""
        resolved_coord = str(coord)
        if resolved_coord not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported lookup coord '{resolved_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
            )

        backend = self._coord_backend(self.tree_coord)
        if resolved_coord == "xyz":
            return backend._domain_bounds_xyz(self)
        return backend._domain_bounds_rpa(self)
