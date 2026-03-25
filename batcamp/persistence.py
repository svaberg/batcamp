#!/usr/bin/env python3
"""Minimal octree state save/load helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .octree import Octree
from .shared_types import GridShape
from .shared_types import TreeCoord

OCTREE_FILE_VERSION = 5

_REQUIRED_TREE_ATTRS = ("_i0", "_i1", "_i2")

_REQUIRED_FILE_KEYS = (
    "version",
    "tree_coord",
    "root_shape",
    "cell_levels",
    "cell_i0",
    "cell_i1",
    "cell_i2",
)


@dataclass(frozen=True)
class OctreeState:
    """Minimal exact octree state shared by build/save/load."""

    tree_coord: TreeCoord
    root_shape: GridShape
    cell_levels: np.ndarray
    cell_i0: np.ndarray
    cell_i1: np.ndarray
    cell_i2: np.ndarray

    @classmethod
    def from_tree(cls, tree: Octree) -> "OctreeState":
        """Capture one octree as minimal persisted state."""
        if tree.cell_levels is None:
            raise ValueError("Cannot persist octree without cell_levels.")
        missing = [name for name in _REQUIRED_TREE_ATTRS if not hasattr(tree, name)]
        if missing:
            raise ValueError(f"Cannot persist octree without exact leaf addresses: missing {missing}.")
        return cls(
            tree_coord=str(tree.tree_coord),
            root_shape=tuple(int(v) for v in tree.root_shape),
            cell_levels=np.asarray(tree.cell_levels, dtype=np.int64),
            cell_i0=np.asarray(tree._i0, dtype=np.int64),
            cell_i1=np.asarray(tree._i1, dtype=np.int64),
            cell_i2=np.asarray(tree._i2, dtype=np.int64),
        )

    def save_npz(self, path: str | Path) -> None:
        """Persist one octree state to a compressed `.npz` file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            version=int(OCTREE_FILE_VERSION),
            tree_coord=str(self.tree_coord),
            root_shape=np.asarray(self.root_shape, dtype=np.int64),
            cell_levels=np.asarray(self.cell_levels, dtype=np.int64),
            cell_i0=np.asarray(self.cell_i0, dtype=np.int64),
            cell_i1=np.asarray(self.cell_i1, dtype=np.int64),
            cell_i2=np.asarray(self.cell_i2, dtype=np.int64),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "OctreeState":
        """Load one persisted octree state from `.npz`."""
        with np.load(Path(path), allow_pickle=False) as data:
            missing = [key for key in _REQUIRED_FILE_KEYS if key not in data]
            if missing:
                raise ValueError(f"Missing required octree fields: {missing}.")

            version = int(data["version"])
            if version != OCTREE_FILE_VERSION:
                raise ValueError(
                    f"Unsupported octree file version {version}; expected {OCTREE_FILE_VERSION}."
                )

            return cls(
                tree_coord=str(data["tree_coord"]),
                root_shape=tuple(int(v) for v in np.asarray(data["root_shape"], dtype=np.int64).tolist()),
                cell_levels=np.asarray(data["cell_levels"], dtype=np.int64),
                cell_i0=np.asarray(data["cell_i0"], dtype=np.int64),
                cell_i1=np.asarray(data["cell_i1"], dtype=np.int64),
                cell_i2=np.asarray(data["cell_i2"], dtype=np.int64),
            )
