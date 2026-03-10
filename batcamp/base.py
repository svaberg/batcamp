#!/usr/bin/env python3
"""Core octree data structures and shared lookup/ray utilities."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import logging
from pathlib import Path
from typing import ClassVar
from typing import cast
from typing import Literal
from typing import TypeAlias

import numpy as np
from starwinds_readplt.dataset import Dataset

DEFAULT_MIN_VALID_CELL_FRACTION = 0.5
DEFAULT_AXIS_RHO_TOL = 1e-12
OCTREE_FILE_VERSION = 1
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


@dataclass
class Octree:
    """Adaptive octree summary plus bound lookup/ray-query entrypoints.

    `level_counts` rows are
    `(level, leaf_count, fine_equivalent_count)`.
    """

    TREE_COORD: ClassVar[str | None] = None

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
        """Build and bind an octree directly from a plain dataset.

        Consumes:
        - `ds`: source dataset with points/corners.
        - Coordinate-system and level-inference tolerance parameters.
        Returns:
        - Built and bound octree instance.
        """
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
            resolved_tree_coord = DEFAULT_TREE_COORD if tree_coord is None else cast(TreeCoord, str(tree_coord))
        from .builder import OctreeBuilder

        builder = OctreeBuilder(level_rtol=level_rtol, level_atol=level_atol)
        allow_default_fallback = bool(cls is Octree and tree_coord is None)
        try:
            return builder.build(ds, tree_coord=resolved_tree_coord, axis_rho_tol=axis_rho_tol)
        except ValueError:
            if not allow_default_fallback:
                raise
            alt_tree_coord: TreeCoord = "rpa" if resolved_tree_coord == "xyz" else "xyz"
            logger.info(
                "Falling back to tree_coord='%s' after default tree_coord='%s' failed in Octree.from_dataset.",
                alt_tree_coord,
                resolved_tree_coord,
            )
            return builder.build(ds, tree_coord=alt_tree_coord, axis_rho_tol=axis_rho_tol)

    @property
    def levels(self) -> tuple[int, ...]:
        """Expose sorted refinement levels present in this tree.

        Consumes:
        - `self.level_counts`.
        Returns:
        - Tuple of level integers.
        """
        return tuple(int(level) for level, _count, _expected in self.level_counts)

    @property
    def is_uniform(self) -> bool:
        """Report whether the tree has one refinement level.

        Consumes:
        - `self.min_level`, `self.max_level`.
        Returns:
        - `True` when min/max levels are equal, else `False`.
        """
        return int(self.min_level) == int(self.max_level)

    @property
    def depth(self) -> int:
        """Return octree depth derived from `root_shape` and `leaf_shape`."""
        return _depth_from_shapes(self.leaf_shape, self.root_shape)

    def bind(
        self,
        ds: Dataset,
        *,
        axis_rho_tol: float | None = None,
    ) -> None:
        """Bind this octree to dataset geometry for lookups/ray queries.

        Consumes:
        - `ds`: dataset with point variables.
        - Optional `axis_rho_tol`.
        Returns:
        - `None`; binds geometry and invalidates cached lookup state.
        """
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
        """Save octree metadata to a compressed `.npz` file.

        Consumes:
        - Output `path`.
        Returns:
        - `None`; writes one `.npz` persistence file.
        """
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
        """Load octree metadata from `.npz` and bind dataset geometry.

        Consumes:
        - Input file `path` and bound dataset `ds`.
        Returns:
        - Loaded `Octree` (or subclass) instance.
        """
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

    def summary(self) -> str:
        """Return compact summary text for this tree.

        Consumes:
        - Octree summary fields on `self`.
        Returns:
        - Single summary string.
        """
        return format_octree_summary(self)

    def __str__(self) -> str:
        """Return human-readable summary text."""
        return self.summary()

    def build_lookup(
        self,
    ) -> None:
        """Construct a bound cell-lookup object for this tree.

        Consumes:
        - Bound tree geometry.
        Returns:
        - `None`; concrete subclasses populate lookup state on `self`.
        """
        raise NotImplementedError("Lookup must be implemented by concrete octree subclasses.")

    def _require_lookup(self) -> "Octree":
        """Return cached lookup state, building it lazily if needed.

        Consumes:
        - Bound dataset state on `self`.
        Returns:
        - Bound octree instance with lookup state initialized.
        """
        if self._lookup_ready:
            return self
        if self.ds is None or self.ds.corners is None:
            raise ValueError("Octree is not bound to a dataset. Call Octree.from_dataset(...) or bind(...).")
        self.build_lookup()
        self._lookup_ready = True
        return self

    @property
    def lookup(self) -> "Octree":
        """Expose bound lookup state (the octree object itself).

        Consumes:
        - Cached/bound lookup state on `self`.
        Returns:
        - `Octree` with lookup state initialized.
        """
        return self._require_lookup()

    def lookup_cell_id(
        self,
        point: np.ndarray,
        *,
        space: str,
    ) -> int:
        """Resolve one query point to a leaf `cell_id` (or `-1`)."""
        raise NotImplementedError

    def hit_from_chosen(self, chosen: int, *, allow_invalid_depth: bool = False) -> "LookupHit | None":
        """Materialize lookup metadata from one chosen cell id."""
        raise NotImplementedError

    def lookup_point(
        self,
        point: np.ndarray,
        *,
        space: TreeCoord,
    ) -> "LookupHit | None":
        """Lookup one query point in the requested coordinate space.

        Consumes:
        - `point`: query coordinate triple.
        - `space`: `"xyz"` or `"rpa"`.
        Returns:
        - `LookupHit` if a cell is resolved, else `None`.
        """
        q = np.array(point, dtype=float).reshape(3)
        resolved_space = str(space)
        if resolved_space not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported lookup space '{resolved_space}'; expected one of {SUPPORTED_TREE_COORDS}."
            )
        self._require_lookup()
        chosen = self.lookup_cell_id(q, space=resolved_space)
        return self.hit_from_chosen(int(chosen))

    def contains_cell(
        self,
        cell_id: int,
        point: np.ndarray,
        *,
        space: TreeCoord,
        tol: float = 1e-10,
    ) -> bool:
        """Containment test of one query point against one leaf cell.

        Consumes:
        - `cell_id`: integer leaf cell id.
        - `point`: query coordinate triple.
        - `space`: `"xyz"` or `"rpa"`.
        - Optional containment tolerance `tol`.
        Returns:
        - `True` when the point lies inside/on the cell bounds, else `False`.
        """
        raise NotImplementedError

    def hit_from_cell_id(self, cell_id: int) -> "LookupHit":
        """Materialize `LookupHit` metadata from a known cell id.

        Consumes:
        - `cell_id`: integer leaf-cell id.
        Returns:
        - `LookupHit` for that id, or raises `ValueError` when invalid.
        """
        cid = int(cell_id)
        self._require_lookup()
        n_cells = int(self._cell_centers.shape[0])
        if cid < 0 or cid >= n_cells:
            raise ValueError(f"Invalid cell_id {cid}; expected [0, {n_cells - 1}].")
        hit = self.hit_from_chosen(cid, allow_invalid_depth=True)
        if hit is None:
            raise ValueError(f"Invalid cell_id {cid}; cannot materialize LookupHit.")
        return hit

    def lookup_local(self, xyz: np.ndarray, near_cid: int | None = None) -> "LookupHit | None":
        """Lookup in xyz using a nearby-cell hint, then fallback to full lookup.

        Consumes:
        - Cartesian query `xyz` and optional nearby `cell_id` hint.
        Returns:
        - `LookupHit` if resolved, else `None`.
        """
        q = np.array(xyz, dtype=float)
        x = float(q[0])
        y = float(q[1])
        z = float(q[2])
        if near_cid is not None and int(near_cid) >= 0:
            near = int(near_cid)
            if self.contains_cell(near, q, space="xyz"):
                return self.hit_from_cell_id(near)
        return self.lookup_point(np.array([x, y, z], dtype=float), space="xyz")

def octree_class_for_coord(tree_coord: str) -> type[Octree]:
    """Resolve coordinate-system tag to the concrete octree class.

    Consumes:
    - `tree_coord` tag (`"rpa"` or `"xyz"`).
    Returns:
    - Matching `Octree` subclass type.
    """
    from .cartesian import CartesianOctree
    from .spherical import SphericalOctree

    if tree_coord == "rpa":
        return SphericalOctree
    if tree_coord == "xyz":
        return CartesianOctree
    raise ValueError(
        f"Unsupported tree_coord '{tree_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
    )


def _depth_from_shapes(leaf_shape: GridShape, root_shape: GridShape) -> int:
    """Compute dyadic octree depth from leaf/root grid shapes."""
    depths: list[int] = []
    for leaf, root in zip(leaf_shape, root_shape):
        if int(root) <= 0 or int(leaf) < int(root) or (int(leaf) % int(root)) != 0:
            raise ValueError(f"Invalid leaf/root shape pair: leaf={leaf_shape}, root={root_shape}.")
        ratio = int(leaf) // int(root)
        depth = 0
        while ratio > 1 and (ratio % 2) == 0:
            ratio //= 2
            depth += 1
        if ratio != 1:
            raise ValueError(
                f"Non-dyadic leaf/root shape pair: leaf={leaf_shape}, root={root_shape}."
            )
        depths.append(depth)
    return min(depths) if depths else 0


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

def format_octree_summary(tree: Octree) -> str:
    """Format one-line summary text for an `Octree` object.

    Consumes:
    - `tree`: octree summary object.
    Returns:
    - Single formatted summary string.
    """
    leaf_levels = ", ".join(
        f"L{level}:{count} (fine-equiv {expected})"
        for level, count, expected in tree.level_counts
    )
    shape_kind = "uniform" if tree.is_uniform else "adaptive"
    out = (
        f"Octree ({shape_kind}): "
        f"tree_coord={tree.tree_coord}, "
        f"finest_leaf_grid={tree.leaf_shape}, root_grid={tree.root_shape}, "
        f"depth={tree.depth}, full={tree.is_full}, "
        f"levels={tree.min_level}..{tree.max_level}; leaf_levels[{leaf_levels}]"
    )
    return out
