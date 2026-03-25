#!/usr/bin/env python3
"""Core octree data structures and shared lookup/ray utilities."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import ClassVar
from typing import NamedTuple
from typing import TYPE_CHECKING
from typing import Literal
from typing import TypeAlias

import numpy as np
from batread import Dataset

SUPPORTED_TREE_COORDS = ("rpa", "xyz")
DEFAULT_TREE_COORD = "xyz"
_DEFAULT_BOUND_AXIS_RHO_TOL = 1e-12

TreeCoord: TypeAlias = Literal["rpa", "xyz"]
"""Coordinate-system tag used by octree builder/lookup dispatch."""

GridShape: TypeAlias = tuple[int, int, int]
"""Grid extents `(n_axis0, n_axis1, n_axis2)`."""

GridIndex: TypeAlias = tuple[int, int, int]
"""Discrete cell/bin index triplet `(i_axis0, i_axis1, i_axis2)`."""

GridPath: TypeAlias = tuple[GridIndex, ...]
"""Root-to-leaf sequence of `GridIndex` entries."""

LevelCountRow: TypeAlias = tuple[int, int, int]
"""Tuple meaning `(level, leaf_count, fine_equivalent_count)`."""

LevelCountTable: TypeAlias = tuple[LevelCountRow, ...]
"""Sorted collection of per-level count rows."""

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .face_neighbors import OctreeFaceNeighbors


class LookupGeometryState(NamedTuple):
    """Bound point/cell arrays and packed lookup state owned by one octree."""

    points: np.ndarray
    corners: np.ndarray
    cell_centers: np.ndarray
    lookup_state: object


def _level_metadata_from_leaves(
    root_shape: GridShape,
    cell_levels: np.ndarray,
) -> tuple[GridShape, bool, LevelCountTable, int, int]:
    """Rebuild tree summary metadata from exact leaf levels."""
    levels = np.asarray(cell_levels, dtype=np.int64)
    valid = levels >= 0
    valid_levels = levels[valid]
    if valid_levels.size == 0:
        raise ValueError("Octree state requires at least one valid cell level.")
    min_level = int(np.min(valid_levels))
    max_level = int(np.max(valid_levels))
    scale = 1 << max_level
    leaf_shape = tuple(int(v) * scale for v in root_shape)
    level_counts = tuple(
        (
            int(level),
            int(np.count_nonzero(valid_levels == level)),
            int(np.count_nonzero(valid_levels == level) * (8 ** int(max_level - level))),
        )
        for level in sorted(set(int(v) for v in valid_levels.tolist()))
    )
    weighted_cells = int(sum(item[2] for item in level_counts))
    is_full = int(np.count_nonzero(valid)) == int(levels.size) and weighted_cells == int(np.prod(leaf_shape))
    return leaf_shape, bool(is_full), level_counts, min_level, max_level


def _node_state_from_leaves(
    cell_levels: np.ndarray,
    cell_i0: np.ndarray,
    cell_i1: np.ndarray,
    cell_i2: np.ndarray,
    *,
    max_level: int,
) -> tuple[np.ndarray, ...]:
    """Rebuild exact occupied nonleaves and leaf-node maps from leaf addresses."""
    from .builder import _build_child_table
    from .builder import _build_node_arrays

    levels = np.asarray(cell_levels, dtype=np.int64)
    i0_all = np.asarray(cell_i0, dtype=np.int64)
    i1_all = np.asarray(cell_i1, dtype=np.int64)
    i2_all = np.asarray(cell_i2, dtype=np.int64)
    if not (levels.shape == i0_all.shape == i1_all.shape == i2_all.shape):
        raise ValueError("Octree leaf level/index arrays must have matching shapes.")
    valid_ids = np.flatnonzero(levels >= 0).astype(np.int64)
    if valid_ids.size == 0:
        raise ValueError("Octree state requires at least one valid leaf cell.")
    depths = np.asarray(levels[valid_ids], dtype=np.int64)
    leaf_i0 = np.asarray(i0_all[valid_ids], dtype=np.int64)
    leaf_i1 = np.asarray(i1_all[valid_ids], dtype=np.int64)
    leaf_i2 = np.asarray(i2_all[valid_ids], dtype=np.int64)
    node_depth, node_i0, node_i1, node_i2, node_value = _build_node_arrays(
        depths,
        leaf_i0,
        leaf_i1,
        leaf_i2,
        valid_ids,
        tree_depth=int(max_level),
        label="Restored",
    )
    node_child, root_node_ids, node_parent = _build_child_table(
        node_depth,
        node_i0,
        node_i1,
        node_i2,
        node_value,
    )
    cell_node_id = np.full(levels.shape[0], -1, dtype=np.int64)
    leaf_mask = node_value >= 0
    cell_node_id[node_value[leaf_mask]] = np.flatnonzero(leaf_mask).astype(np.int64)
    return node_depth, node_i0, node_i1, node_i2, node_value, node_child, root_node_ids, node_parent, cell_node_id


class Octree:
    """Adaptive octree summary plus bound lookup/ray-query entrypoints.

    `level_counts` rows are
    `(level, leaf_count, fine_equivalent_count)`.
    """

    X_VAR: ClassVar[str] = "X [R]"
    Y_VAR: ClassVar[str] = "Y [R]"
    Z_VAR: ClassVar[str] = "Z [R]"
    XY_VARS: ClassVar[tuple[str, str]] = (X_VAR, Y_VAR)
    XYZ_VARS: ClassVar[tuple[str, str, str]] = (X_VAR, Y_VAR, Z_VAR)

    def __init__(
        self,
        *,
        root_shape: GridShape,
        tree_coord: TreeCoord = DEFAULT_TREE_COORD,
        cell_levels: np.ndarray,
        cell_i0: np.ndarray,
        cell_i1: np.ndarray,
        cell_i2: np.ndarray,
        ds: Dataset | None = None,
        axis_rho_tol: float = _DEFAULT_BOUND_AXIS_RHO_TOL,
    ) -> None:
        """Build one octree directly from exact leaf addresses."""
        resolved_tree_coord = str(tree_coord)
        if resolved_tree_coord not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported tree_coord '{resolved_tree_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
            )
        self.root_shape = tuple(int(v) for v in root_shape)
        self.tree_coord = resolved_tree_coord
        self.cell_levels = np.asarray(cell_levels, dtype=np.int64)
        self.axis_rho_tol = float(axis_rho_tol)
        self.ds = None

        self.leaf_shape, self.is_full, self.level_counts, self.min_level, self.max_level = _level_metadata_from_leaves(
            self.root_shape,
            self.cell_levels,
        )
        self._i0 = np.asarray(cell_i0, dtype=np.int64)
        self._i1 = np.asarray(cell_i1, dtype=np.int64)
        self._i2 = np.asarray(cell_i2, dtype=np.int64)
        (
            self._node_depth,
            self._node_i0,
            self._node_i1,
            self._node_i2,
            self._node_value,
            self._node_child,
            self._root_node_ids,
            self._node_parent,
            self._cell_node_id,
        ) = _node_state_from_leaves(
            self.cell_levels,
            self._i0,
            self._i1,
            self._i2,
            max_level=self.max_level,
        )
        if ds is not None:
            self.bind(ds, axis_rho_tol=self.axis_rho_tol)

    @property
    def levels(self) -> tuple[int, ...]:
        """Return the sorted refinement levels present in this tree."""
        return tuple(int(level) for level, _count, _expected in self.level_counts)

    @property
    def is_uniform(self) -> bool:
        """Return `True` when all cells are at one refinement level."""
        return int(self.min_level) == int(self.max_level)

    @property
    def depth(self) -> int:
        """Return the maximum root-relative level.

        `depth` is kept as a read-only alias for `max_level` so the tree has
        one refinement coordinate system internally.
        """
        return int(self.max_level)

    def bind(
        self,
        ds: Dataset,
        *,
        axis_rho_tol: float | None = None,
    ) -> None:
        """Attach a dataset to this tree so lookup and ray methods can run."""
        if ds.corners is None:
            raise ValueError("Dataset has no corners; cannot bind octree lookup.")
        if not set(self.XYZ_VARS).issubset(set(ds.variables)):
            raise ValueError("Dataset must provide X/Y/Z variables to bind octree lookup.")
        next_axis_rho_tol = float(self.axis_rho_tol) if axis_rho_tol is None else float(axis_rho_tol)

        self.ds = ds
        self.axis_rho_tol = next_axis_rho_tol

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
        ds: Dataset | None = None,
        axis_rho_tol: float | None = None,
        bind: bool = True,
    ) -> "Octree":
        """Instantiate one tree from exact saved state."""
        if bind and ds is None:
            raise ValueError("Octree.from_state requires ds when bind=True.")
        if str(state.tree_coord) not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported tree_coord '{state.tree_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
            )
        resolved_axis_rho_tol = float(_DEFAULT_BOUND_AXIS_RHO_TOL) if axis_rho_tol is None else float(axis_rho_tol)
        return cls(
            root_shape=tuple(int(v) for v in state.root_shape),
            tree_coord=state.tree_coord,
            cell_levels=state.cell_levels,
            cell_i0=state.cell_i0,
            cell_i1=state.cell_i1,
            cell_i2=state.cell_i2,
            ds=ds if bind else None,
            axis_rho_tol=resolved_axis_rho_tol,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        ds: Dataset,
        axis_rho_tol: float | None = None,
    ) -> "Octree":
        """Load a tree from `.npz` and bind it to the given dataset."""
        from .persistence import OctreeState

        in_path = Path(path)
        state = OctreeState.load_npz(in_path)
        tree = cls.from_state(state, ds=ds, axis_rho_tol=axis_rho_tol, bind=True)
        logger.info("Loaded octree from %s", str(in_path))
        return tree

    def __str__(self) -> str:
        """Return human-readable summary text."""
        leaf_levels = ", ".join(
            f"L{level}:{count} (fine-equiv {expected})"
            for level, count, expected in self.level_counts
        )
        shape_kind = "uniform" if self.is_uniform else "adaptive"
        return (
            f"Octree ({shape_kind}): "
            f"tree_coord={self.tree_coord}, "
            f"finest_leaf_grid={self.leaf_shape}, root_grid={self.root_shape}, "
            f"max_level={self.max_level}, full={self.is_full}, "
            f"levels={self.min_level}..{self.max_level}; leaf_levels[{leaf_levels}]"
        )

    @staticmethod
    def _lookup_backend(tree_coord: str) -> type:
        """Return the geometry-specific lookup helper for one tree coordinate."""
        if tree_coord == "xyz":
            from .cartesian import _CartesianCellLookup

            return _CartesianCellLookup
        if tree_coord == "rpa":
            from .spherical import _SphericalCellLookup

            return _SphericalCellLookup
        raise ValueError(
            f"Unsupported tree_coord '{tree_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
        )

    def build_lookup(self) -> None:
        """Build the per-cell lookup data used by query methods."""
        if self.ds is None or self.ds.corners is None:
            raise ValueError("Octree is not bound to a dataset. Call bind(...) before lookup.")
        self._lookup_backend(str(self.tree_coord))._init_lookup_state(self)

    def _require_lookup(self) -> "Octree":
        """Ensure lookup data is built, then return `self`."""
        if self.ds is None or self.ds.corners is None:
            raise ValueError("Octree is not bound to a dataset. Build and bind it before lookup.")
        self.build_lookup()
        return self

    @property
    def cell_count(self) -> int:
        """Return number of leaf cells available in the bound lookup."""
        self._require_lookup()
        return int(self._cell_centers.shape[0])

    @property
    def cell_centers(self) -> np.ndarray:
        """Return leaf-cell centers in Cartesian coordinates."""
        self._require_lookup()
        return self._cell_centers

    def lookup_geometry(self) -> LookupGeometryState:
        """Return bound point/cell arrays plus packed lookup state."""
        self._require_lookup()
        required = ("_cell_centers", "_lookup_state")
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise ValueError(f"Octree lookup geometry is incomplete: missing {missing}.")
        if self.ds is None or self.ds.corners is None:
            raise ValueError("Octree is not bound to a dataset. Build and bind it before lookup.")
        return LookupGeometryState(
            points=np.column_stack(
                (
                    np.asarray(self.ds[self.X_VAR], dtype=np.float64),
                    np.asarray(self.ds[self.Y_VAR], dtype=np.float64),
                    np.asarray(self.ds[self.Z_VAR], dtype=np.float64),
                )
            ),
            corners=np.asarray(self.ds.corners, dtype=np.int64),
            cell_centers=self._cell_centers,
            lookup_state=self._lookup_state,
        )

    def _frontier_nodes(self, max_level: int) -> tuple[np.ndarray, ...]:
        """Return unique frontier nodes by truncating leaves to one level cutoff."""
        if self.cell_levels is None:
            raise ValueError("Octree has no cell_levels; cannot build frontier nodes.")
        required = ("_i0", "_i1", "_i2")
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise ValueError(f"Octree frontier nodes require exact leaf addresses: missing {missing}.")

        levels_all = self.cell_levels
        valid = levels_all >= 0
        if not np.any(valid):
            raise ValueError("Octree contains no valid cells (all levels are < 0).")

        cell_ids = np.flatnonzero(valid)
        i0_all = self._i0
        i1_all = self._i1
        i2_all = self._i2
        if not (levels_all.shape == i0_all.shape == i1_all.shape == i2_all.shape):
            raise ValueError("Cell level/index arrays must have matching shapes.")

        levels_valid = levels_all[valid]
        active_levels = np.minimum(levels_valid, int(max_level))
        shift = levels_valid - active_levels
        active_i0 = np.right_shift(i0_all[valid], shift)
        active_i1 = np.right_shift(i1_all[valid], shift)
        active_i2 = np.right_shift(i2_all[valid], shift)

        keys = np.column_stack((active_levels, active_i0, active_i1, active_i2))
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

        node_cell_ids = np.full(unique_keys.shape[0], -1, dtype=np.int64)
        cell_to_node_id = np.full(levels_all.shape[0], -1, dtype=np.int64)
        for row, node_id in enumerate(inverse):
            nid = int(node_id)
            if node_cell_ids[nid] < 0:
                node_cell_ids[nid] = int(cell_ids[row])
            cell_to_node_id[int(cell_ids[row])] = nid

        levels = unique_keys[:, 0]
        i0 = unique_keys[:, 1]
        i1 = unique_keys[:, 2]
        i2 = unique_keys[:, 3]
        return levels, i0, i1, i2, node_cell_ids, cell_to_node_id

    def face_neighbors(self, *, max_level: int | None = None) -> "OctreeFaceNeighbors":
        """Return the lazily built face-neighbor graph for one level cutoff."""
        if self.cell_levels is None:
            raise ValueError("Octree has no cell_levels; cannot build face neighbors.")
        valid_levels = self.cell_levels[self.cell_levels >= 0]
        if valid_levels.size == 0:
            raise ValueError("Octree contains no valid cell levels (all < 0).")

        target_max_level = int(np.max(valid_levels) if max_level is None else max_level)
        cache = getattr(self, "_face_neighbors_by_max_level", None)
        if cache is None:
            cache = {}
            self._face_neighbors_by_max_level = cache
        face_neighbors = cache.get(target_max_level)
        if face_neighbors is None:
            from .face_neighbors import _build_face_neighbors_from_frontier

            frontier_nodes = self._frontier_nodes(target_max_level)
            face_neighbors = _build_face_neighbors_from_frontier(
                root_shape=self.root_shape,
                tree_coord=self.tree_coord,
                target_max_level=target_max_level,
                frontier_nodes=frontier_nodes,
            )
            cache[int(face_neighbors.max_level)] = face_neighbors
        return face_neighbors

    def cell_bounds(
        self,
        cell_id: int,
        *,
        coord: TreeCoord = "xyz",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(lo, hi)` bounds for one cell in requested coord."""
        self._require_lookup()
        cid = int(cell_id)
        n_cells = int(self._cell_centers.shape[0])
        if cid < 0 or cid >= n_cells:
            raise ValueError(f"Invalid cell_id {cid}; expected [0, {n_cells - 1}].")

        resolved_coord = str(coord)
        if resolved_coord not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported lookup coord '{resolved_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
            )

        backend = self._lookup_backend(str(self.tree_coord))
        if resolved_coord == "xyz":
            return backend._cell_bounds_xyz(self, cid)
        return backend._cell_bounds_rpa(self, cid)

    def domain_bounds(self, *, coord: TreeCoord = "xyz") -> tuple[np.ndarray, np.ndarray]:
        """Return global `(lo, hi)` bounds for the bound tree in requested coord."""
        self._require_lookup()
        resolved_coord = str(coord)
        if resolved_coord not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported lookup coord '{resolved_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
            )

        backend = self._lookup_backend(str(self.tree_coord))
        if resolved_coord == "xyz":
            return backend._domain_bounds_xyz(self)
        return backend._domain_bounds_rpa(self)

    def lookup_cell_id(
        self,
        point: np.ndarray,
        *,
        coord: str,
    ) -> int:
        """Resolve one query point to a leaf `cell_id` (or `-1`)."""
        return self._lookup_backend(str(self.tree_coord)).lookup_cell_id(self, point, coord=coord)

    def hit_from_chosen(self, chosen: int, *, allow_invalid_level: bool = False) -> "LookupHit | None":
        """Materialize lookup metadata from one chosen cell id."""
        return self._lookup_backend(str(self.tree_coord)).hit_from_chosen(
            self,
            chosen,
            allow_invalid_level=allow_invalid_level,
        )

    def lookup_point(
        self,
        point: np.ndarray,
        *,
        coord: TreeCoord,
    ) -> "LookupHit | None":
        """Find which cell contains one point, or return `None` if not found."""
        q = np.array(point, dtype=float).reshape(3)
        resolved_coord = str(coord)
        if resolved_coord not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported lookup coord '{resolved_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
            )
        self._require_lookup()
        chosen = self.lookup_cell_id(q, coord=resolved_coord)
        return self.hit_from_chosen(int(chosen))

    def contains_cell(
        self,
        cell_id: int,
        point: np.ndarray,
        *,
        coord: TreeCoord,
        tol: float = 1e-10,
    ) -> bool:
        """Return whether one point lies inside one cell."""
        return self._lookup_backend(str(self.tree_coord)).contains_cell(
            self,
            cell_id,
            point,
            coord=coord,
            tol=tol,
        )

    def hit_from_cell_id(self, cell_id: int) -> "LookupHit":
        """Return lookup metadata for a known cell id."""
        cid = int(cell_id)
        self._require_lookup()
        n_cells = int(self._cell_centers.shape[0])
        if cid < 0 or cid >= n_cells:
            raise ValueError(f"Invalid cell_id {cid}; expected [0, {n_cells - 1}].")
        hit = self.hit_from_chosen(cid, allow_invalid_level=True)
        if hit is None:
            raise ValueError(f"Invalid cell_id {cid}; cannot materialize LookupHit.")
        return hit

@dataclass(frozen=True)
class LookupHit:
    """Resolved lookup metadata for one query point."""

    cell_id: int
    level: int
    i0: int
    i1: int
    i2: int
    path: GridPath
    center_xyz: tuple[float, float, float]
