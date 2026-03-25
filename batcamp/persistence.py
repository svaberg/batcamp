#!/usr/bin/env python3
"""Octree persistence dataclasses and `.npz` save/load helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .octree import TreeCoord
from .octree import GridShape
from .octree import LevelCountTable
from .octree import OCTREE_FILE_VERSION
from .octree import Octree
from .octree import SUPPORTED_TREE_COORDS

_REQUIRED_TREE_ATTRS = ("_i0", "_i1", "_i2", "_node_depth", "_node_i0", "_node_i1", "_node_i2", "_node_value")

_ARRAY_SPECS = (
    ("cell_levels", "cell_levels", np.int64, (0,)),
    ("cell_i0", "_i0", np.int64, (0,)),
    ("cell_i1", "_i1", np.int64, (0,)),
    ("cell_i2", "_i2", np.int64, (0,)),
    ("cell_centers", "_cell_centers", np.float64, (0, 3)),
    ("cell_x_min", "_cell_x_min", np.float64, (0,)),
    ("cell_x_max", "_cell_x_max", np.float64, (0,)),
    ("cell_y_min", "_cell_y_min", np.float64, (0,)),
    ("cell_y_max", "_cell_y_max", np.float64, (0,)),
    ("cell_z_min", "_cell_z_min", np.float64, (0,)),
    ("cell_z_max", "_cell_z_max", np.float64, (0,)),
    ("xyz_min", "_xyz_min", np.float64, (0,)),
    ("xyz_max", "_xyz_max", np.float64, (0,)),
    ("cell_r_min", "_cell_r_min", np.float64, (0,)),
    ("cell_r_max", "_cell_r_max", np.float64, (0,)),
    ("cell_theta_min", "_cell_theta_min", np.float64, (0,)),
    ("cell_theta_max", "_cell_theta_max", np.float64, (0,)),
    ("cell_phi_start", "_cell_phi_start", np.float64, (0,)),
    ("cell_phi_width", "_cell_phi_width", np.float64, (0,)),
    ("node_depth", "_node_depth", np.int64, (0,)),
    ("node_i0", "_node_i0", np.int64, (0,)),
    ("node_i1", "_node_i1", np.int64, (0,)),
    ("node_i2", "_node_i2", np.int64, (0,)),
    ("node_value", "_node_value", np.int64, (0,)),
    ("node_child", "_node_child", np.int64, (0, 8)),
    ("root_node_ids", "_root_node_ids", np.int64, (0,)),
    ("node_parent", "_node_parent", np.int64, (0,)),
    ("cell_node_id", "_cell_node_id", np.int64, (0,)),
    ("node_x_min", "_node_x_min", np.float64, (0,)),
    ("node_x_max", "_node_x_max", np.float64, (0,)),
    ("node_y_min", "_node_y_min", np.float64, (0,)),
    ("node_y_max", "_node_y_max", np.float64, (0,)),
    ("node_z_min", "_node_z_min", np.float64, (0,)),
    ("node_z_max", "_node_z_max", np.float64, (0,)),
    ("node_r_min", "_node_r_min", np.float64, (0,)),
    ("node_r_max", "_node_r_max", np.float64, (0,)),
    ("node_theta_min", "_node_theta_min", np.float64, (0,)),
    ("node_theta_max", "_node_theta_max", np.float64, (0,)),
    ("node_phi_start", "_node_phi_start", np.float64, (0,)),
    ("node_phi_width", "_node_phi_width", np.float64, (0,)),
    ("radial_edges", "_radial_edges", np.float64, (0,)),
)

_FLOAT_SCALAR_SPECS = (
    ("r_min", "_r_min", np.nan),
    ("r_max", "_r_max", np.nan),
)

_REQUIRED_FILE_KEYS = (
    "version",
    "tree_coord",
    "leaf_shape",
    "root_shape",
    "is_full",
    "level_counts",
    "min_level",
    "max_level",
    "axis_rho_tol",
) + tuple(name for name, _attr, _dtype, _shape in _ARRAY_SPECS) + tuple(
    name for name, _attr, _default in _FLOAT_SCALAR_SPECS
)


def _empty_array(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Return one empty array matching the persisted field shape/dtype."""
    return np.empty(shape, dtype=dtype)


def _array_state_from_tree(tree: Octree) -> dict[str, object]:
    """Collect persisted array/scalar payload from one in-memory tree."""
    payload: dict[str, object] = {}
    for field_name, attr_name, dtype, shape in _ARRAY_SPECS:
        default = _empty_array(shape, dtype)
        payload[field_name] = np.asarray(getattr(tree, attr_name, default), dtype=dtype)
    for field_name, attr_name, default in _FLOAT_SCALAR_SPECS:
        payload[field_name] = float(getattr(tree, attr_name, default))
    return payload


def _array_state_from_npz(data: np.lib.npyio.NpzFile) -> dict[str, object]:
    """Collect persisted array/scalar payload from one `.npz` archive."""
    payload: dict[str, object] = {}
    for field_name, _attr_name, dtype, _shape in _ARRAY_SPECS:
        payload[field_name] = np.asarray(data[field_name], dtype=dtype)
    for field_name, _attr_name, _default in _FLOAT_SCALAR_SPECS:
        payload[field_name] = float(data[field_name])
    return payload


def _array_state_npz_payload(arrays: "OctreeArrayState") -> dict[str, object]:
    """Convert one array-state dataclass to `np.savez_compressed` payload."""
    payload: dict[str, object] = {}
    for field_name, _attr_name, dtype, _shape in _ARRAY_SPECS:
        payload[field_name] = np.asarray(getattr(arrays, field_name), dtype=dtype)
    for field_name, _attr_name, _default in _FLOAT_SCALAR_SPECS:
        payload[field_name] = float(getattr(arrays, field_name))
    return payload


@dataclass(frozen=True)
class OctreeArrayState:
    """Array payload persisted alongside octree core metadata."""

    cell_levels: np.ndarray
    cell_i0: np.ndarray
    cell_i1: np.ndarray
    cell_i2: np.ndarray
    cell_centers: np.ndarray
    cell_x_min: np.ndarray
    cell_x_max: np.ndarray
    cell_y_min: np.ndarray
    cell_y_max: np.ndarray
    cell_z_min: np.ndarray
    cell_z_max: np.ndarray
    xyz_min: np.ndarray
    xyz_max: np.ndarray
    cell_r_min: np.ndarray
    cell_r_max: np.ndarray
    r_min: float
    r_max: float
    cell_theta_min: np.ndarray
    cell_theta_max: np.ndarray
    cell_phi_start: np.ndarray
    cell_phi_width: np.ndarray
    node_depth: np.ndarray
    node_i0: np.ndarray
    node_i1: np.ndarray
    node_i2: np.ndarray
    node_value: np.ndarray
    node_child: np.ndarray
    root_node_ids: np.ndarray
    node_parent: np.ndarray
    cell_node_id: np.ndarray
    node_x_min: np.ndarray
    node_x_max: np.ndarray
    node_y_min: np.ndarray
    node_y_max: np.ndarray
    node_z_min: np.ndarray
    node_z_max: np.ndarray
    node_r_min: np.ndarray
    node_r_max: np.ndarray
    node_theta_min: np.ndarray
    node_theta_max: np.ndarray
    node_phi_start: np.ndarray
    node_phi_width: np.ndarray
    radial_edges: np.ndarray

    @classmethod
    def from_tree(cls, tree: Octree) -> "OctreeArrayState":
        """Capture array payload from one in-memory tree."""
        if tree.cell_levels is None:
            raise ValueError("Cannot persist octree without cell_levels.")
        missing = [name for name in _REQUIRED_TREE_ATTRS if not hasattr(tree, name)]
        if missing:
            raise ValueError(f"Cannot persist octree without exact tree state: missing {missing}.")
        return cls(**_array_state_from_tree(tree))


@dataclass(frozen=True)
class OctreePersistenceState:
    """Versioned octree core metadata used by save/load operations."""

    leaf_shape: GridShape
    root_shape: GridShape
    is_full: bool
    level_counts: LevelCountTable
    min_level: int
    max_level: int
    tree_coord: TreeCoord
    axis_rho_tol: float

    @classmethod
    def from_octree(cls, tree: Octree) -> "OctreePersistenceState":
        """Capture persistence-safe core metadata from one octree object."""
        tree_coord = str(tree.tree_coord)
        if tree_coord not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported tree_coord '{tree_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
            )
        return cls(
            leaf_shape=tuple(int(v) for v in tree.leaf_shape),
            root_shape=tuple(int(v) for v in tree.root_shape),
            is_full=bool(tree.is_full),
            level_counts=tuple(tuple(int(v) for v in row) for row in tree.level_counts),
            min_level=int(tree.min_level),
            max_level=int(tree.max_level),
            tree_coord=tree_coord,
            axis_rho_tol=float(tree.axis_rho_tol),
        )

    def save_npz(self, path: Path, *, arrays: OctreeArrayState) -> None:
        """Persist one octree snapshot to a compressed `.npz` file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            version=int(OCTREE_FILE_VERSION),
            tree_coord=str(self.tree_coord),
            leaf_shape=np.asarray(self.leaf_shape, dtype=np.int64),
            root_shape=np.asarray(self.root_shape, dtype=np.int64),
            is_full=bool(self.is_full),
            level_counts=np.asarray(self.level_counts, dtype=np.int64),
            min_level=int(self.min_level),
            max_level=int(self.max_level),
            axis_rho_tol=float(self.axis_rho_tol),
            **_array_state_npz_payload(arrays),
        )

    @classmethod
    def load_npz(cls, path: Path) -> tuple["OctreePersistenceState", OctreeArrayState]:
        """Load one persisted octree snapshot and array payload."""
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in _REQUIRED_FILE_KEYS if key not in data]
            if missing:
                raise ValueError(f"Missing required octree fields: {missing}.")

            version = int(data["version"])
            if version != OCTREE_FILE_VERSION:
                raise ValueError(
                    f"Unsupported octree file version {version}; expected {OCTREE_FILE_VERSION}."
                )

            tree_coord = str(data["tree_coord"])
            if tree_coord not in SUPPORTED_TREE_COORDS:
                raise ValueError(
                    f"Unsupported tree_coord '{tree_coord}' in octree file; "
                    f"expected one of {SUPPORTED_TREE_COORDS}."
                )

            state = cls(
                leaf_shape=tuple(int(v) for v in np.asarray(data["leaf_shape"], dtype=np.int64).tolist()),
                root_shape=tuple(int(v) for v in np.asarray(data["root_shape"], dtype=np.int64).tolist()),
                is_full=bool(data["is_full"]),
                level_counts=tuple(
                    tuple(int(v) for v in row)
                    for row in np.asarray(data["level_counts"], dtype=np.int64).tolist()
                ),
                min_level=int(data["min_level"]),
                max_level=int(data["max_level"]),
                tree_coord=tree_coord,
                axis_rho_tol=float(data["axis_rho_tol"]),
            )
            arrays = OctreeArrayState(**_array_state_from_npz(data))
            return state, arrays

    def instantiate_tree(
        self,
        tree_cls: type[Octree],
        *,
        arrays: OctreeArrayState,
    ) -> Octree:
        """Instantiate one octree object from loaded metadata."""
        tree = tree_cls(
            leaf_shape=self.leaf_shape,
            root_shape=self.root_shape,
            is_full=self.is_full,
            level_counts=self.level_counts,
            min_level=self.min_level,
            max_level=self.max_level,
            tree_coord=self.tree_coord,
            cell_levels=arrays.cell_levels,
            axis_rho_tol=self.axis_rho_tol,
        )
        for field_name, attr_name, dtype, _shape in _ARRAY_SPECS:
            if attr_name == "cell_levels":
                continue
            setattr(tree, attr_name, np.asarray(getattr(arrays, field_name), dtype=dtype))
        for field_name, attr_name, _default in _FLOAT_SCALAR_SPECS:
            setattr(tree, attr_name, float(getattr(arrays, field_name)))
        return tree
