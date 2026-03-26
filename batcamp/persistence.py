#!/usr/bin/env python3
"""Minimal octree state save/load helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .octree import Octree
from .shared_types import GridShape
from .shared_types import TreeCoord

OCTREE_FILE_VERSION = 6


@dataclass(frozen=True)
class OctreeState:
    """Minimal exact octree state shared by build/save/load."""

    tree_coord: TreeCoord
    root_shape: GridShape
    cell_levels: np.ndarray
    cell_ijk: np.ndarray

    @classmethod
    def from_tree(cls, tree: Octree) -> "OctreeState":
        """Capture one octree as minimal persisted state."""
        leaf_row_count = int(tree.cell_levels.shape[0])
        return cls(
            tree_coord=str(tree.tree_coord),
            root_shape=tuple(int(v) for v in tree.root_shape),
            cell_levels=np.asarray(tree.cell_levels, dtype=np.int64),
            cell_ijk=np.asarray(tree._cell_ijk[:leaf_row_count], dtype=np.int64),
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
            cell_ijk=np.asarray(self.cell_ijk, dtype=np.int64),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "OctreeState":
        """Load one persisted octree state from `.npz`."""
        with np.load(Path(path), allow_pickle=False) as data:
            missing = [
                key
                for key in ("version", "tree_coord", "root_shape", "cell_levels", "cell_ijk")
                if key not in data
            ]
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
                cell_ijk=np.asarray(data["cell_ijk"], dtype=np.int64),
            )
