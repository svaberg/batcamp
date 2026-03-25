#!/usr/bin/env python3
"""Core octree data structures and shared lookup/ray utilities."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import logging
from pathlib import Path
from typing import ClassVar
from typing import NamedTuple
from typing import TYPE_CHECKING
from typing import Literal
from typing import TypeAlias

import numpy as np
from batread.dataset import Dataset

DEFAULT_MIN_VALID_CELL_FRACTION = 0.5
DEFAULT_AXIS_RHO_TOL = 1e-12
OCTREE_FILE_VERSION = 4
SUPPORTED_TREE_COORDS = ("rpa", "xyz")
DEFAULT_TREE_COORD = "xyz"

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


def infer_tree_coord_from_geometry(ds: Dataset, *, sample_size: int = 2048) -> TreeCoord:
    """Guess whether the mesh is Cartesian (`xyz`) or spherical-like (`rpa`)."""
    corners = getattr(ds, "corners", None)
    if corners is None:
        raise ValueError("Dataset has no cell connectivity (corners).")
    corners_arr = np.asarray(corners, dtype=np.int64)
    if corners_arr.ndim != 2 or corners_arr.shape[0] == 0:
        return "rpa"

    if corners_arr.shape[0] > int(sample_size):
        idx = np.linspace(0, corners_arr.shape[0] - 1, int(sample_size), dtype=np.int64)
        sample = corners_arr[idx]
    else:
        sample = corners_arr

    x = np.asarray(ds[Octree.X_VAR], dtype=float)
    y = np.asarray(ds[Octree.Y_VAR], dtype=float)
    z = np.asarray(ds[Octree.Z_VAR], dtype=float)
    xr = np.round(x[sample], 12)
    yr = np.round(y[sample], 12)
    zr = np.round(z[sample], 12)

    ux = np.array([np.unique(row).size for row in xr], dtype=np.int64)
    uy = np.array([np.unique(row).size for row in yr], dtype=np.int64)
    uz = np.array([np.unique(row).size for row in zr], dtype=np.int64)
    axis_like = (ux <= 2) & (uy <= 2) & (uz <= 2)
    frac_axis_like = float(np.mean(axis_like)) if axis_like.size > 0 else 0.0
    return "xyz" if frac_axis_like >= 0.98 else "rpa"


@dataclass
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

    leaf_shape: GridShape
    root_shape: GridShape
    is_full: bool
    level_counts: LevelCountTable
    min_level: int
    max_level: int
    tree_coord: TreeCoord = DEFAULT_TREE_COORD
    cell_levels: np.ndarray | None = None
    ds: Dataset | None = field(default=None, repr=False)
    axis_rho_tol: float = field(default=DEFAULT_AXIS_RHO_TOL, repr=False)
    _lookup_ready: bool = field(default=False, init=False, repr=False)

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
        ds_changed = self.ds is not ds
        tol_changed = not np.isclose(float(self.axis_rho_tol), next_axis_rho_tol, rtol=0.0, atol=0.0)

        self.ds = ds
        self.axis_rho_tol = next_axis_rho_tol
        self._corners = np.asarray(ds.corners, dtype=np.int64)
        self._points = np.column_stack(
            (
                np.asarray(ds[self.X_VAR], dtype=float),
                np.asarray(ds[self.Y_VAR], dtype=float),
                np.asarray(ds[self.Z_VAR], dtype=float),
            )
        )
        if ds_changed or tol_changed:
            self._lookup_ready = False

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
        resolved_axis_rho_tol = float(state.axis_rho_tol) if axis_rho_tol is None else float(axis_rho_tol)
        tree = cls(
            leaf_shape=state.leaf_shape,
            root_shape=state.root_shape,
            is_full=state.is_full,
            level_counts=state.level_counts,
            min_level=state.min_level,
            max_level=state.max_level,
            tree_coord=state.tree_coord,
            cell_levels=np.asarray(state.cell_levels, dtype=np.int64),
            axis_rho_tol=resolved_axis_rho_tol,
        )
        for field_name, attr_name, dtype, _shape in state.ARRAY_SPECS:
            if attr_name == "cell_levels":
                continue
            setattr(tree, attr_name, np.asarray(getattr(state, field_name), dtype=dtype))
        for field_name, attr_name, _default in state.FLOAT_SCALAR_SPECS:
            setattr(tree, attr_name, float(getattr(state, field_name)))
        if bind:
            tree.bind(ds, axis_rho_tol=resolved_axis_rho_tol)
        return tree

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
        if self._lookup_ready:
            return self
        if self.ds is None or self.ds.corners is None:
            raise ValueError("Octree is not bound to a dataset. Build and bind it before lookup.")
        self.build_lookup()
        self._lookup_ready = True
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
        return np.asarray(self._cell_centers, dtype=float)

    def _lookup_geometry(self) -> LookupGeometryState:
        """Return bound point/cell arrays plus packed lookup state."""
        self._require_lookup()
        required = ("_points", "_corners", "_cell_centers", "_lookup_state")
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise ValueError(f"Octree lookup geometry is incomplete: missing {missing}.")
        return LookupGeometryState(
            points=np.asarray(self._points, dtype=np.float64),
            corners=np.asarray(self._corners, dtype=np.int64),
            cell_centers=np.asarray(self._cell_centers, dtype=np.float64),
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

        levels_all = np.asarray(self.cell_levels, dtype=np.int64)
        valid = levels_all >= 0
        if not np.any(valid):
            raise ValueError("Octree contains no valid cells (all levels are < 0).")

        cell_ids = np.flatnonzero(valid).astype(np.int64)
        i0_all = np.asarray(self._i0, dtype=np.int64)
        i1_all = np.asarray(self._i1, dtype=np.int64)
        i2_all = np.asarray(self._i2, dtype=np.int64)
        if not (levels_all.shape == i0_all.shape == i1_all.shape == i2_all.shape):
            raise ValueError("Cell level/index arrays must have matching shapes.")

        levels_valid = levels_all[valid]
        active_levels = np.minimum(levels_valid, int(max_level))
        shift = levels_valid - active_levels
        active_i0 = np.right_shift(i0_all[valid], shift)
        active_i1 = np.right_shift(i1_all[valid], shift)
        active_i2 = np.right_shift(i2_all[valid], shift)

        keys = np.column_stack((active_levels, active_i0, active_i1, active_i2)).astype(np.int64)
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

        node_cell_ids = np.full(unique_keys.shape[0], -1, dtype=np.int64)
        cell_to_node_id = np.full(levels_all.shape[0], -1, dtype=np.int64)
        for row, node_id in enumerate(inverse):
            nid = int(node_id)
            if node_cell_ids[nid] < 0:
                node_cell_ids[nid] = int(cell_ids[row])
            cell_to_node_id[int(cell_ids[row])] = nid

        levels = np.asarray(unique_keys[:, 0], dtype=np.int64)
        i0 = np.asarray(unique_keys[:, 1], dtype=np.int64)
        i1 = np.asarray(unique_keys[:, 2], dtype=np.int64)
        i2 = np.asarray(unique_keys[:, 3], dtype=np.int64)
        return levels, i0, i1, i2, node_cell_ids, cell_to_node_id

    def face_neighbors(self, *, max_level: int | None = None) -> "OctreeFaceNeighbors":
        """Return the lazily built face-neighbor graph for one level cutoff."""
        if self.cell_levels is None:
            raise ValueError("Octree has no cell_levels; cannot build face neighbors.")
        valid_levels = np.asarray(self.cell_levels, dtype=np.int64)
        valid_levels = valid_levels[valid_levels >= 0]
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
