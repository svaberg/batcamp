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


@dataclass(frozen=True)
class OctreeArrayState:
    """Array payload persisted alongside octree core metadata."""

    cell_levels: np.ndarray
    cell_i0: np.ndarray
    cell_i1: np.ndarray
    cell_i2: np.ndarray
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
        required = ("_i0", "_i1", "_i2", "_node_depth", "_node_i0", "_node_i1", "_node_i2", "_node_value")
        missing = [name for name in required if not hasattr(tree, name)]
        if missing:
            raise ValueError(f"Cannot persist octree without exact tree state: missing {missing}.")
        return cls(
            cell_levels=np.asarray(tree.cell_levels, dtype=np.int64),
            cell_i0=np.asarray(tree._i0, dtype=np.int64),
            cell_i1=np.asarray(tree._i1, dtype=np.int64),
            cell_i2=np.asarray(tree._i2, dtype=np.int64),
            node_depth=np.asarray(tree._node_depth, dtype=np.int64),
            node_i0=np.asarray(tree._node_i0, dtype=np.int64),
            node_i1=np.asarray(tree._node_i1, dtype=np.int64),
            node_i2=np.asarray(tree._node_i2, dtype=np.int64),
            node_value=np.asarray(tree._node_value, dtype=np.int64),
            node_child=np.asarray(getattr(tree, "_node_child", np.empty((0, 8), dtype=np.int64)), dtype=np.int64),
            root_node_ids=np.asarray(getattr(tree, "_root_node_ids", np.empty((0,), dtype=np.int64)), dtype=np.int64),
            node_parent=np.asarray(getattr(tree, "_node_parent", np.empty((0,), dtype=np.int64)), dtype=np.int64),
            cell_node_id=np.asarray(getattr(tree, "_cell_node_id", np.empty((0,), dtype=np.int64)), dtype=np.int64),
            node_x_min=np.asarray(getattr(tree, "_node_x_min", np.empty((0,), dtype=np.float64)), dtype=np.float64),
            node_x_max=np.asarray(getattr(tree, "_node_x_max", np.empty((0,), dtype=np.float64)), dtype=np.float64),
            node_y_min=np.asarray(getattr(tree, "_node_y_min", np.empty((0,), dtype=np.float64)), dtype=np.float64),
            node_y_max=np.asarray(getattr(tree, "_node_y_max", np.empty((0,), dtype=np.float64)), dtype=np.float64),
            node_z_min=np.asarray(getattr(tree, "_node_z_min", np.empty((0,), dtype=np.float64)), dtype=np.float64),
            node_z_max=np.asarray(getattr(tree, "_node_z_max", np.empty((0,), dtype=np.float64)), dtype=np.float64),
            node_r_min=np.asarray(getattr(tree, "_node_r_min", np.empty((0,), dtype=np.float64)), dtype=np.float64),
            node_r_max=np.asarray(getattr(tree, "_node_r_max", np.empty((0,), dtype=np.float64)), dtype=np.float64),
            node_theta_min=np.asarray(getattr(tree, "_node_theta_min", np.empty((0,), dtype=np.float64)), dtype=np.float64),
            node_theta_max=np.asarray(getattr(tree, "_node_theta_max", np.empty((0,), dtype=np.float64)), dtype=np.float64),
            node_phi_start=np.asarray(getattr(tree, "_node_phi_start", np.empty((0,), dtype=np.float64)), dtype=np.float64),
            node_phi_width=np.asarray(getattr(tree, "_node_phi_width", np.empty((0,), dtype=np.float64)), dtype=np.float64),
            radial_edges=np.asarray(getattr(tree, "_radial_edges", np.empty((0,), dtype=np.float64)), dtype=np.float64),
        )


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
            cell_levels=np.asarray(arrays.cell_levels, dtype=np.int64),
            cell_i0=np.asarray(arrays.cell_i0, dtype=np.int64),
            cell_i1=np.asarray(arrays.cell_i1, dtype=np.int64),
            cell_i2=np.asarray(arrays.cell_i2, dtype=np.int64),
            node_depth=np.asarray(arrays.node_depth, dtype=np.int64),
            node_i0=np.asarray(arrays.node_i0, dtype=np.int64),
            node_i1=np.asarray(arrays.node_i1, dtype=np.int64),
            node_i2=np.asarray(arrays.node_i2, dtype=np.int64),
            node_value=np.asarray(arrays.node_value, dtype=np.int64),
            node_child=np.asarray(arrays.node_child, dtype=np.int64),
            root_node_ids=np.asarray(arrays.root_node_ids, dtype=np.int64),
            node_parent=np.asarray(arrays.node_parent, dtype=np.int64),
            cell_node_id=np.asarray(arrays.cell_node_id, dtype=np.int64),
            node_x_min=np.asarray(arrays.node_x_min, dtype=np.float64),
            node_x_max=np.asarray(arrays.node_x_max, dtype=np.float64),
            node_y_min=np.asarray(arrays.node_y_min, dtype=np.float64),
            node_y_max=np.asarray(arrays.node_y_max, dtype=np.float64),
            node_z_min=np.asarray(arrays.node_z_min, dtype=np.float64),
            node_z_max=np.asarray(arrays.node_z_max, dtype=np.float64),
            node_r_min=np.asarray(arrays.node_r_min, dtype=np.float64),
            node_r_max=np.asarray(arrays.node_r_max, dtype=np.float64),
            node_theta_min=np.asarray(arrays.node_theta_min, dtype=np.float64),
            node_theta_max=np.asarray(arrays.node_theta_max, dtype=np.float64),
            node_phi_start=np.asarray(arrays.node_phi_start, dtype=np.float64),
            node_phi_width=np.asarray(arrays.node_phi_width, dtype=np.float64),
            radial_edges=np.asarray(arrays.radial_edges, dtype=np.float64),
        )

    @classmethod
    def load_npz(cls, path: Path) -> tuple["OctreePersistenceState", OctreeArrayState]:
        """Load one persisted octree snapshot and array payload."""
        required = (
            "version",
            "tree_coord",
            "leaf_shape",
            "root_shape",
            "is_full",
            "level_counts",
            "min_level",
            "max_level",
            "axis_rho_tol",
            "cell_levels",
            "cell_i0",
            "cell_i1",
            "cell_i2",
            "node_depth",
            "node_i0",
            "node_i1",
            "node_i2",
            "node_value",
            "node_child",
            "root_node_ids",
            "node_parent",
            "cell_node_id",
            "node_x_min",
            "node_x_max",
            "node_y_min",
            "node_y_max",
            "node_z_min",
            "node_z_max",
            "node_r_min",
            "node_r_max",
            "node_theta_min",
            "node_theta_max",
            "node_phi_start",
            "node_phi_width",
            "radial_edges",
        )
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in required if key not in data]
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
            arrays = OctreeArrayState(
                cell_levels=np.asarray(data["cell_levels"], dtype=np.int64),
                cell_i0=np.asarray(data["cell_i0"], dtype=np.int64),
                cell_i1=np.asarray(data["cell_i1"], dtype=np.int64),
                cell_i2=np.asarray(data["cell_i2"], dtype=np.int64),
                node_depth=np.asarray(data["node_depth"], dtype=np.int64),
                node_i0=np.asarray(data["node_i0"], dtype=np.int64),
                node_i1=np.asarray(data["node_i1"], dtype=np.int64),
                node_i2=np.asarray(data["node_i2"], dtype=np.int64),
                node_value=np.asarray(data["node_value"], dtype=np.int64),
                node_child=np.asarray(data["node_child"], dtype=np.int64),
                root_node_ids=np.asarray(data["root_node_ids"], dtype=np.int64),
                node_parent=np.asarray(data["node_parent"], dtype=np.int64),
                cell_node_id=np.asarray(data["cell_node_id"], dtype=np.int64),
                node_x_min=np.asarray(data["node_x_min"], dtype=np.float64),
                node_x_max=np.asarray(data["node_x_max"], dtype=np.float64),
                node_y_min=np.asarray(data["node_y_min"], dtype=np.float64),
                node_y_max=np.asarray(data["node_y_max"], dtype=np.float64),
                node_z_min=np.asarray(data["node_z_min"], dtype=np.float64),
                node_z_max=np.asarray(data["node_z_max"], dtype=np.float64),
                node_r_min=np.asarray(data["node_r_min"], dtype=np.float64),
                node_r_max=np.asarray(data["node_r_max"], dtype=np.float64),
                node_theta_min=np.asarray(data["node_theta_min"], dtype=np.float64),
                node_theta_max=np.asarray(data["node_theta_max"], dtype=np.float64),
                node_phi_start=np.asarray(data["node_phi_start"], dtype=np.float64),
                node_phi_width=np.asarray(data["node_phi_width"], dtype=np.float64),
                radial_edges=np.asarray(data["radial_edges"], dtype=np.float64),
            )
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
        tree._i0 = np.asarray(arrays.cell_i0, dtype=np.int64)
        tree._i1 = np.asarray(arrays.cell_i1, dtype=np.int64)
        tree._i2 = np.asarray(arrays.cell_i2, dtype=np.int64)
        tree._node_depth = np.asarray(arrays.node_depth, dtype=np.int64)
        tree._node_i0 = np.asarray(arrays.node_i0, dtype=np.int64)
        tree._node_i1 = np.asarray(arrays.node_i1, dtype=np.int64)
        tree._node_i2 = np.asarray(arrays.node_i2, dtype=np.int64)
        tree._node_value = np.asarray(arrays.node_value, dtype=np.int64)
        tree._node_child = np.asarray(arrays.node_child, dtype=np.int64)
        tree._root_node_ids = np.asarray(arrays.root_node_ids, dtype=np.int64)
        tree._node_parent = np.asarray(arrays.node_parent, dtype=np.int64)
        tree._cell_node_id = np.asarray(arrays.cell_node_id, dtype=np.int64)
        tree._node_x_min = np.asarray(arrays.node_x_min, dtype=np.float64)
        tree._node_x_max = np.asarray(arrays.node_x_max, dtype=np.float64)
        tree._node_y_min = np.asarray(arrays.node_y_min, dtype=np.float64)
        tree._node_y_max = np.asarray(arrays.node_y_max, dtype=np.float64)
        tree._node_z_min = np.asarray(arrays.node_z_min, dtype=np.float64)
        tree._node_z_max = np.asarray(arrays.node_z_max, dtype=np.float64)
        tree._node_r_min = np.asarray(arrays.node_r_min, dtype=np.float64)
        tree._node_r_max = np.asarray(arrays.node_r_max, dtype=np.float64)
        tree._node_theta_min = np.asarray(arrays.node_theta_min, dtype=np.float64)
        tree._node_theta_max = np.asarray(arrays.node_theta_max, dtype=np.float64)
        tree._node_phi_start = np.asarray(arrays.node_phi_start, dtype=np.float64)
        tree._node_phi_width = np.asarray(arrays.node_phi_width, dtype=np.float64)
        tree._radial_edges = np.asarray(arrays.radial_edges, dtype=np.float64)
        return tree
