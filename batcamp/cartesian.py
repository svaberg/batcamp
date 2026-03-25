#!/usr/bin/env python3
"""Cartesian coordinate support for octree lookup.

Important modeling assumption for ``tree_coord="xyz"``:
leaf cells are treated as axis-aligned Cartesian slabs from
per-cell corner min/max values. Lookup/ray/interpolation kernels in the
Cartesian backend are exact under this axis-aligned representation.
"""

from __future__ import annotations
from numba import njit
import numpy as np

from .constants import XYZ_VARS
from .octree import LookupKernelState
from .octree import Octree
from .octree import _contains_lookup_cell
from .octree import _lookup_cell_id_kernel

_LOOKUP_CONTAIN_TOL = 1e-10
_MISSING_NODE_VALUE = -1

CartesianLookupKernelState = LookupKernelState


@njit(cache=False)
def _lookup_xyz_cell_id_kernel(
    x: float,
    y: float,
    z: float,
    lookup_state: LookupKernelState,
    prev_cid: int = -1,
) -> int:
    """Resolve one Cartesian query to a cell id by descending sparse child containment."""
    return _lookup_cell_id_kernel(x, y, z, lookup_state, prev_cid, _LOOKUP_CONTAIN_TOL)


class _CartesianCoordSupport:
    """Cartesian geometry support for octree lookup on axis-aligned slab cells."""

    def _bind_geometry(self) -> None:
        """Attach Cartesian bound geometry derived from exact leaf addresses."""
        required = (
            "_i0",
            "_i1",
            "_i2",
            "_node_depth",
            "_node_i0",
            "_node_i1",
            "_node_i2",
            "_node_value",
            "_node_child",
            "_root_node_ids",
            "_node_parent",
            "_cell_node_id",
        )
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise ValueError(f"Cartesian lookup requires exact tree state: missing {missing}.")
        # Cast once at the dataset->kernel boundary: corner ids must index cleanly
        # and coordinate arrays feed float64 lookup kernels.
        corners = np.asarray(self.ds.corners, dtype=np.int64)
        x = np.asarray(self.ds[XYZ_VARS[0]], dtype=np.float64)
        y = np.asarray(self.ds[XYZ_VARS[1]], dtype=np.float64)
        z = np.asarray(self.ds[XYZ_VARS[2]], dtype=np.float64)
        cell_x = x[corners]
        cell_y = y[corners]
        cell_z = z[corners]
        self._cell_x_min = np.min(cell_x, axis=1)
        self._cell_x_max = np.max(cell_x, axis=1)
        self._cell_y_min = np.min(cell_y, axis=1)
        self._cell_y_max = np.max(cell_y, axis=1)
        self._cell_z_min = np.min(cell_z, axis=1)
        self._cell_z_max = np.max(cell_z, axis=1)

        n_cells = int(corners.shape[0])
        if self.cell_levels is None or self.cell_levels.shape[0] != n_cells:
            raise ValueError("Cartesian lookup requires exact cell_levels.")
        self._cell_level = self.cell_levels
        self._cell_valid = self._cell_level >= 0
        n_nodes = int(self._node_value.shape[0])
        self._node_x_min = np.full(n_nodes, np.inf, dtype=np.float64)
        self._node_x_max = np.full(n_nodes, -np.inf, dtype=np.float64)
        self._node_y_min = np.full(n_nodes, np.inf, dtype=np.float64)
        self._node_y_max = np.full(n_nodes, -np.inf, dtype=np.float64)
        self._node_z_min = np.full(n_nodes, np.inf, dtype=np.float64)
        self._node_z_max = np.full(n_nodes, -np.inf, dtype=np.float64)
        leaf_mask = self._node_value >= 0
        leaf_node_ids = np.flatnonzero(leaf_mask)
        leaf_cell_ids = self._node_value[leaf_mask]
        self._node_x_min[leaf_node_ids] = self._cell_x_min[leaf_cell_ids]
        self._node_x_max[leaf_node_ids] = self._cell_x_max[leaf_cell_ids]
        self._node_y_min[leaf_node_ids] = self._cell_y_min[leaf_cell_ids]
        self._node_y_max[leaf_node_ids] = self._cell_y_max[leaf_cell_ids]
        self._node_z_min[leaf_node_ids] = self._cell_z_min[leaf_cell_ids]
        self._node_z_max[leaf_node_ids] = self._cell_z_max[leaf_cell_ids]
        for nid in range(n_nodes - 1, -1, -1):
            parent = int(self._node_parent[nid])
            if parent < 0:
                continue
            self._node_x_min[parent] = min(self._node_x_min[parent], self._node_x_min[nid])
            self._node_x_max[parent] = max(self._node_x_max[parent], self._node_x_max[nid])
            self._node_y_min[parent] = min(self._node_y_min[parent], self._node_y_min[nid])
            self._node_y_max[parent] = max(self._node_y_max[parent], self._node_y_max[nid])
            self._node_z_min[parent] = min(self._node_z_min[parent], self._node_z_min[nid])
            self._node_z_max[parent] = max(self._node_z_max[parent], self._node_z_max[nid])
        root_ids = self._root_node_ids
        self._xyz_min = np.array(
            [
                float(np.min(self._node_x_min[root_ids])),
                float(np.min(self._node_y_min[root_ids])),
                float(np.min(self._node_z_min[root_ids])),
            ],
            dtype=np.float64,
        )
        self._xyz_max = np.array(
            [
                float(np.max(self._node_x_max[root_ids])),
                float(np.max(self._node_y_max[root_ids])),
                float(np.max(self._node_z_max[root_ids])),
            ],
            dtype=np.float64,
        )
        self._coord_state = LookupKernelState(
            cell_axis0_start=self._cell_x_min,
            cell_axis0_width=self._cell_x_max - self._cell_x_min,
            cell_axis1_start=self._cell_y_min,
            cell_axis1_width=self._cell_y_max - self._cell_y_min,
            cell_axis2_start=self._cell_z_min,
            cell_axis2_width=self._cell_z_max - self._cell_z_min,
            cell_valid=self._cell_valid,
            domain_axis0_start=float(self._xyz_min[0]),
            domain_axis0_width=float(self._xyz_max[0] - self._xyz_min[0]),
            domain_axis1_start=float(self._xyz_min[1]),
            domain_axis1_width=float(self._xyz_max[1] - self._xyz_min[1]),
            domain_axis2_start=float(self._xyz_min[2]),
            domain_axis2_width=float(self._xyz_max[2] - self._xyz_min[2]),
            axis2_period=0.0,
            axis2_periodic=False,
            node_value=self._node_value,
            node_child=self._node_child,
            root_node_ids=self._root_node_ids,
            node_parent=self._node_parent,
            cell_node_id=self._cell_node_id,
            node_axis0_start=self._node_x_min,
            node_axis0_width=self._node_x_max - self._node_x_min,
            node_axis1_start=self._node_y_min,
            node_axis1_width=self._node_y_max - self._node_y_min,
            node_axis2_start=self._node_z_min,
            node_axis2_width=self._node_z_max - self._node_z_min,
        )

    def _domain_bounds_xyz(self) -> tuple[np.ndarray, np.ndarray]:
        return self._xyz_min, self._xyz_max

    def _domain_bounds_rpa(self) -> tuple[np.ndarray, np.ndarray]:
        pts = np.column_stack(
            (
                self.ds[XYZ_VARS[0]],
                self.ds[XYZ_VARS[1]],
                self.ds[XYZ_VARS[2]],
            )
        )
        r = np.linalg.norm(pts, axis=1)
        theta = np.arccos(np.clip(pts[:, 2] / np.maximum(r, np.finfo(float).tiny), -1.0, 1.0))
        phi = np.mod(np.arctan2(pts[:, 1], pts[:, 0]), 2.0 * np.pi)
        return (
            np.array([float(np.min(r)), float(np.min(theta)), float(np.min(phi))], dtype=float),
            np.array([float(np.max(r)), float(np.max(theta)), float(np.max(phi))], dtype=float),
        )

    def _contains_xyz_cell(
        self,
        cell_id: int,
        x: float,
        y: float,
        z: float,
        *,
        tol: float = _LOOKUP_CONTAIN_TOL,
    ) -> bool:
        """Return whether one `(x, y, z)` point lies inside one cell."""
        return bool(
            _contains_lookup_cell(
                int(cell_id),
                float(x),
                float(y),
                float(z),
                self._coord_state,
                float(tol),
            )
        )

    def _lookup_xyz_cell_id(self, x: float, y: float, z: float) -> int:
        """Return the containing cell id for `(x, y, z)`, or `-1`."""
        return int(
            _lookup_xyz_cell_id_kernel(
                float(x),
                float(y),
                float(z),
                self._coord_state,
            )
        )
