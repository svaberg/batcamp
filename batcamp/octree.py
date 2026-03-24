#!/usr/bin/env python3
"""Core octree data structures and shared lookup/ray utilities."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import logging
from pathlib import Path
from typing import ClassVar
from typing import TYPE_CHECKING
from typing import cast
from typing import Literal
from typing import TypeAlias

import numpy as np
from batread.dataset import Dataset

DEFAULT_MIN_VALID_CELL_FRACTION = 0.5
DEFAULT_AXIS_RHO_TOL = 1e-12
OCTREE_FILE_VERSION = 3
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

    TREE_COORD: ClassVar[str | None] = None
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

    @classmethod
    def from_dataset(
        cls,
        ds: Dataset,
        *,
        tree_coord: TreeCoord | None = None,
        axis_rho_tol: float = DEFAULT_AXIS_RHO_TOL,
        level_rtol: float = 1e-4,
        level_atol: float = 1e-9,
    ) -> "Octree":
        """Build a tree from a dataset and bind it so lookup methods can run."""
        resolved_tree_coord: TreeCoord
        if cls is not Octree and cls.TREE_COORD is not None:
            if tree_coord is None:
                resolved_tree_coord = cls.TREE_COORD
            else:
                resolved_tree_coord = cast(TreeCoord, str(tree_coord))
            if resolved_tree_coord != cls.TREE_COORD:
                raise ValueError(
                    f"{cls.__name__} requires tree_coord='{cls.TREE_COORD}', got '{resolved_tree_coord}'."
                )
        else:
            if tree_coord is None:
                resolved_tree_coord = infer_tree_coord_from_geometry(ds)
            else:
                resolved_tree_coord = cast(TreeCoord, str(tree_coord))
        from .builder import OctreeBuilder

        builder = OctreeBuilder(level_rtol=level_rtol, level_atol=level_atol)
        return builder.build(ds, tree_coord=resolved_tree_coord, axis_rho_tol=axis_rho_tol)

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
        next_axis_rho_tol = float(self.axis_rho_tol) if axis_rho_tol is None else float(axis_rho_tol)
        ds_changed = self.ds is not ds
        tol_changed = not np.isclose(float(self.axis_rho_tol), next_axis_rho_tol, rtol=0.0, atol=0.0)

        self.ds = ds
        self.axis_rho_tol = next_axis_rho_tol
        if ds_changed or tol_changed:
            self._lookup_ready = False

    def save(self, path: str | Path) -> None:
        """Save this tree to a compressed `.npz` file."""
        from .persistence import OctreeArrayState
        from .persistence import OctreePersistenceState

        state = OctreePersistenceState.from_octree(self)
        arrays = OctreeArrayState.from_tree(self)
        out_path = Path(path)
        state.save_npz(
            out_path,
            arrays=arrays,
        )
        logger.info("Saved octree to %s", str(out_path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        ds: Dataset,
        axis_rho_tol: float | None = None,
    ) -> "Octree":
        """Load a tree from `.npz` and bind it to the given dataset."""
        from .persistence import OctreePersistenceState

        in_path = Path(path)
        state, array_state = OctreePersistenceState.load_npz(in_path)
        tree_coord = str(state.tree_coord)
        tree_cls = octree_class_for_coord(tree_coord)
        if cls is not Octree and cls.TREE_COORD is not None and cls.TREE_COORD != tree_coord:
            raise ValueError(
                f"{cls.__name__} cannot load tree_coord='{tree_coord}'."
            )
        if cls is not Octree:
            tree_cls = cls
        tree = state.instantiate_tree(tree_cls, arrays=array_state)
        tree.bind(ds, axis_rho_tol=axis_rho_tol)
        if axis_rho_tol is not None:
            tree.axis_rho_tol = float(axis_rho_tol)
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

    def build_lookup(
        self,
    ) -> None:
        """Build the per-cell lookup data used by query methods."""
        raise NotImplementedError("Lookup must be implemented by concrete octree subclasses.")

    def _require_lookup(self) -> "Octree":
        """Ensure lookup data is built, then return `self`."""
        if self._lookup_ready:
            return self
        if self.ds is None or self.ds.corners is None:
            raise ValueError("Octree is not bound to a dataset. Call Octree.from_dataset(...) or bind(...).")
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

    @property
    def lookup_state(self) -> object:
        """Return compiled lookup state used by numba kernels."""
        # TODO: This still leaks packed lookup internals to interpolator/ray code.
        # The intended boundary is exact tree state on Octree, not foreign modules
        # reaching through extra private cache slots on the tree.
        self._require_lookup()
        state = getattr(self, "_lookup_state", None)
        if state is None:
            raise ValueError("Lookup state is unavailable for this tree.")
        return state

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
            from .face_neighbors import build_face_neighbors

            face_neighbors = build_face_neighbors(self, max_level=target_max_level)
            self._cache_face_neighbors(face_neighbors)
        return face_neighbors

    def _cache_face_neighbors(self, face_neighbors: "OctreeFaceNeighbors") -> None:
        """Store one prebuilt face-neighbor graph in the lazy octree cache."""
        cache = getattr(self, "_face_neighbors_by_max_level", None)
        if cache is None:
            cache = {}
            self._face_neighbors_by_max_level = cache
        cache[int(face_neighbors.max_level)] = face_neighbors

    @property
    def points(self) -> np.ndarray:
        """Return node coordinates used by lookup/interpolation setup."""
        self._require_lookup()
        pts = getattr(self, "_points", None)
        if pts is None:
            raise ValueError("Lookup points are unavailable for this tree.")
        return np.asarray(pts, dtype=float)

    @property
    def cell_phi_start(self) -> np.ndarray:
        """Return per-cell azimuth start for spherical lookup trees."""
        self._require_lookup()
        starts = getattr(self, "_cell_phi_start", None)
        if starts is None:
            raise ValueError("cell_phi_start is only available for spherical trees.")
        return np.asarray(starts, dtype=float)

    @property
    def cell_phi_width(self) -> np.ndarray:
        """Return per-cell wrapped azimuth width for spherical lookup trees."""
        self._require_lookup()
        widths = getattr(self, "_cell_phi_width", None)
        if widths is None:
            raise ValueError("cell_phi_width is only available for spherical trees.")
        return np.asarray(widths, dtype=float)

    def _cell_bounds_xyz(self, cell_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return one cell's Cartesian bounds for subclasses."""
        raise NotImplementedError

    def _cell_bounds_rpa(self, cell_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return one cell's spherical bounds for subclasses."""
        raise NotImplementedError

    def _domain_bounds_xyz(self) -> tuple[np.ndarray, np.ndarray]:
        """Return global Cartesian bounds for subclasses."""
        raise NotImplementedError

    def _domain_bounds_rpa(self) -> tuple[np.ndarray, np.ndarray]:
        """Return global spherical bounds for subclasses."""
        raise NotImplementedError

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

        if resolved_coord == "xyz":
            return self._cell_bounds_xyz(cid)
        return self._cell_bounds_rpa(cid)

    def domain_bounds(self, *, coord: TreeCoord = "xyz") -> tuple[np.ndarray, np.ndarray]:
        """Return global `(lo, hi)` bounds for the bound tree in requested coord."""
        self._require_lookup()
        resolved_coord = str(coord)
        if resolved_coord not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported lookup coord '{resolved_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
            )

        if resolved_coord == "xyz":
            return self._domain_bounds_xyz()
        return self._domain_bounds_rpa()

    def lookup_cell_id(
        self,
        point: np.ndarray,
        *,
        coord: str,
    ) -> int:
        """Resolve one query point to a leaf `cell_id` (or `-1`)."""
        raise NotImplementedError

    def hit_from_chosen(self, chosen: int, *, allow_invalid_level: bool = False) -> "LookupHit | None":
        """Materialize lookup metadata from one chosen cell id."""
        raise NotImplementedError

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
        raise NotImplementedError

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

def octree_class_for_coord(tree_coord: str) -> type[Octree]:
    """Return the octree class that matches `tree_coord`."""
    from .cartesian import CartesianOctree
    from .spherical import SphericalOctree

    if tree_coord == "rpa":
        return SphericalOctree
    if tree_coord == "xyz":
        return CartesianOctree
    raise ValueError(
        f"Unsupported tree_coord '{tree_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
    )


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
