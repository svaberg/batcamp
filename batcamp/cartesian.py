#!/usr/bin/env python3
"""Cartesian octree and Cartesian lookup implementation.

Important modeling assumption for ``tree_coord="xyz"``:
leaf cells are treated as axis-aligned Cartesian slabs from
per-cell corner min/max values. Lookup/ray/interpolation kernels in the
Cartesian backend are exact under this axis-aligned representation.
"""

from __future__ import annotations

from typing import ClassVar
from typing import NamedTuple

import math
from numba import njit
import numpy as np

from .octree import GridIndex
from .octree import GridPath
from .octree import LookupHit
from .octree import Octree

_LOOKUP_CONTAIN_TOL = 1e-10
_MISSING_NODE_VALUE = -1
_INTERNAL_NODE_VALUE = -2


class CartesianLookupKernelState(NamedTuple):
    """Arrays used by compiled Cartesian lookup code under a slab cell model."""

    cell_centers: np.ndarray
    cell_x_min: np.ndarray
    cell_x_max: np.ndarray
    cell_y_min: np.ndarray
    cell_y_max: np.ndarray
    cell_z_min: np.ndarray
    cell_z_max: np.ndarray
    cell_valid: np.ndarray
    xyz_min: np.ndarray
    xyz_max: np.ndarray
    xyz_span: np.ndarray
    bin_shape: np.ndarray
    bin_offsets: np.ndarray
    bin_cell_ids: np.ndarray
    max_radius: int
    leaf_shape: np.ndarray
    tree_depth: int
    cell_i0: np.ndarray
    cell_i1: np.ndarray
    cell_i2: np.ndarray
    node_depth: np.ndarray
    node_i0: np.ndarray
    node_i1: np.ndarray
    node_i2: np.ndarray
    node_value: np.ndarray


@njit(cache=True)
def _contains_xyz_cell(
    cid: int,
    x: float,
    y: float,
    z: float,
    lookup_state: CartesianLookupKernelState,
    tol: float = _LOOKUP_CONTAIN_TOL,
) -> bool:
    """Check one Cartesian query against one cell slab bounds."""
    if not lookup_state.cell_valid[cid]:
        return False
    if x < (lookup_state.cell_x_min[cid] - tol) or x > (lookup_state.cell_x_max[cid] + tol):
        return False
    if y < (lookup_state.cell_y_min[cid] - tol) or y > (lookup_state.cell_y_max[cid] + tol):
        return False
    if z < (lookup_state.cell_z_min[cid] - tol) or z > (lookup_state.cell_z_max[cid] + tol):
        return False
    return True


@njit(cache=True)
def _lookup_fine_index(q: float, q_min: float, q_span: float, n_fine: int) -> int:
    """Map one coordinate to its finest-grid index, clamped to valid range."""
    idx = int(math.floor(((q - q_min) / q_span) * n_fine))
    if idx < 0:
        return 0
    if idx >= n_fine:
        return int(n_fine - 1)
    return idx


@njit(cache=True)
def _find_node_value(
    depth: int,
    i0: int,
    i1: int,
    i2: int,
    lookup_state: CartesianLookupKernelState,
) -> int:
    """Binary-search one occupied Cartesian node by `(depth, i0, i1, i2)`."""
    lo = 0
    hi = int(lookup_state.node_depth.shape[0]) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        m_depth = int(lookup_state.node_depth[mid])
        if depth < m_depth:
            hi = mid - 1
            continue
        if depth > m_depth:
            lo = mid + 1
            continue

        m_i0 = int(lookup_state.node_i0[mid])
        if i0 < m_i0:
            hi = mid - 1
            continue
        if i0 > m_i0:
            lo = mid + 1
            continue

        m_i1 = int(lookup_state.node_i1[mid])
        if i1 < m_i1:
            hi = mid - 1
            continue
        if i1 > m_i1:
            lo = mid + 1
            continue

        m_i2 = int(lookup_state.node_i2[mid])
        if i2 < m_i2:
            hi = mid - 1
            continue
        if i2 > m_i2:
            lo = mid + 1
            continue

        return int(lookup_state.node_value[mid])
    return _MISSING_NODE_VALUE


@njit(cache=True)
def _lookup_xyz_cell_id_kernel(
    x: float,
    y: float,
    z: float,
    lookup_state: CartesianLookupKernelState,
    prev_cid: int = -1,
) -> int:
    """Resolve one Cartesian query to a cell id by descending occupied octree nodes."""
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return -1
    if prev_cid >= 0 and _contains_xyz_cell(int(prev_cid), x, y, z, lookup_state):
        return int(prev_cid)

    inside_bbox = bool(
        (x >= lookup_state.xyz_min[0])
        and (x <= lookup_state.xyz_max[0])
        and (y >= lookup_state.xyz_min[1])
        and (y <= lookup_state.xyz_max[1])
        and (z >= lookup_state.xyz_min[2])
        and (z <= lookup_state.xyz_max[2])
    )
    if not inside_bbox:
        return -1

    fine_i0 = _lookup_fine_index(x, lookup_state.xyz_min[0], lookup_state.xyz_span[0], int(lookup_state.leaf_shape[0]))
    fine_i1 = _lookup_fine_index(y, lookup_state.xyz_min[1], lookup_state.xyz_span[1], int(lookup_state.leaf_shape[1]))
    fine_i2 = _lookup_fine_index(z, lookup_state.xyz_min[2], lookup_state.xyz_span[2], int(lookup_state.leaf_shape[2]))

    tree_depth = int(lookup_state.tree_depth)
    for depth in range(tree_depth + 1):
        shift = tree_depth - depth
        i0 = fine_i0 >> shift
        i1 = fine_i1 >> shift
        i2 = fine_i2 >> shift
        node_value = _find_node_value(depth, i0, i1, i2, lookup_state)
        if node_value == _MISSING_NODE_VALUE:
            return -1
        if node_value >= 0:
            cid = int(node_value)
            if _contains_xyz_cell(cid, x, y, z, lookup_state):
                return cid
            return -1
    return -1


class _CartesianCellLookup:
    """Leaf-cell lookup accelerator for Cartesian `(x, y, z)` octrees.

    The accelerator stores each cell as an axis-aligned box in Cartesian space.
    """

    def _init_lookup_state(self, tree: Octree) -> None:
        """Build per-cell lookup arrays from a bound Cartesian tree.

        Cell geometry is reduced to per-axis min/max slab bounds.
        """
        if tree.ds is None or tree.ds.corners is None:
            raise ValueError("Lookup requires a bound octree with dataset and corners.")
        ds = tree.ds
        corners = np.array(ds.corners, dtype=np.int64)
        if not set(Octree.XYZ_VARS).issubset(set(ds.variables)):
            raise ValueError("Lookup requires X/Y/Z variables.")

        self.tree = tree
        self._corners = np.array(corners, dtype=np.int64)
        self._points = np.column_stack(
            (
                np.array(ds[Octree.X_VAR], dtype=float),
                np.array(ds[Octree.Y_VAR], dtype=float),
                np.array(ds[Octree.Z_VAR], dtype=float),
            )
        )
        cell_xyz = self._points[self._corners]
        self._cell_centers = np.mean(cell_xyz, axis=1)
        self._cell_x_min = np.min(cell_xyz[:, :, 0], axis=1)
        self._cell_x_max = np.max(cell_xyz[:, :, 0], axis=1)
        self._cell_y_min = np.min(cell_xyz[:, :, 1], axis=1)
        self._cell_y_max = np.max(cell_xyz[:, :, 1], axis=1)
        self._cell_z_min = np.min(cell_xyz[:, :, 2], axis=1)
        self._cell_z_max = np.max(cell_xyz[:, :, 2], axis=1)
        tiny = np.finfo(float).tiny
        self._cell_dx = np.maximum(self._cell_x_max - self._cell_x_min, tiny)
        self._cell_dy = np.maximum(self._cell_y_max - self._cell_y_min, tiny)
        self._cell_dz = np.maximum(self._cell_z_max - self._cell_z_min, tiny)
        self._xyz_min = np.array(
            [
                float(np.min(self._cell_x_min)),
                float(np.min(self._cell_y_min)),
                float(np.min(self._cell_z_min)),
            ],
            dtype=float,
        )
        self._xyz_max = np.array(
            [
                float(np.max(self._cell_x_max)),
                float(np.max(self._cell_y_max)),
                float(np.max(self._cell_z_max)),
            ],
            dtype=float,
        )
        self._xyz_span = np.maximum(self._xyz_max - self._xyz_min, tiny)

        n_cells = int(self._corners.shape[0])
        if tree.cell_levels is not None and tree.cell_levels.shape[0] == n_cells:
            self._cell_level = np.array(tree.cell_levels, dtype=np.int64)
        else:
            self._cell_level = np.full(n_cells, int(tree.max_level), dtype=np.int64)
        self._cell_valid = self._cell_level >= 0

        self._leaf_shape = np.asarray(tree.leaf_shape, dtype=np.int64)
        self._tree_depth = int(tree.depth)
        self._cell_depth = np.full(n_cells, -1, dtype=np.int64)
        self._i0 = np.full(n_cells, -1, dtype=np.int64)
        self._i1 = np.full(n_cells, -1, dtype=np.int64)
        self._i2 = np.full(n_cells, -1, dtype=np.int64)
        self._init_exact_addresses()
        self._bin_shape = np.array([1, 1, 1], dtype=np.int64)
        self._bin_offsets = np.zeros(2, dtype=np.int64)
        self._bin_cell_ids = np.empty((0,), dtype=np.int64)
        self._max_radius = 0
        self._lookup_state = CartesianLookupKernelState(
            cell_centers=self._cell_centers,
            cell_x_min=self._cell_x_min,
            cell_x_max=self._cell_x_max,
            cell_y_min=self._cell_y_min,
            cell_y_max=self._cell_y_max,
            cell_z_min=self._cell_z_min,
            cell_z_max=self._cell_z_max,
            cell_valid=self._cell_valid,
            xyz_min=self._xyz_min,
            xyz_max=self._xyz_max,
            xyz_span=self._xyz_span,
            bin_shape=self._bin_shape,
            bin_offsets=self._bin_offsets,
            bin_cell_ids=self._bin_cell_ids,
            max_radius=int(self._max_radius),
            leaf_shape=self._leaf_shape,
            tree_depth=int(self._tree_depth),
            cell_i0=self._i0,
            cell_i1=self._i1,
            cell_i2=self._i2,
            node_depth=self._node_depth,
            node_i0=self._node_i0,
            node_i1=self._node_i1,
            node_i2=self._node_i2,
            node_value=self._node_value,
        )

    def _init_exact_addresses(self) -> None:
        """Derive exact Cartesian `(depth, i0, i1, i2)` addresses from cell bounds."""
        n_cells = int(self._corners.shape[0])
        valid_ids = np.flatnonzero(self._cell_valid).astype(np.int64)
        axis_tol = np.array(
            [1e-9 * max(float(self._xyz_span[k]), 1.0) for k in range(3)],
            dtype=float,
        )

        node_values: dict[tuple[int, int, int, int], int] = {}
        for cid in valid_ids.tolist():
            level = int(self._cell_level[cid])
            depth = int(level)
            if depth < 0:
                raise ValueError(
                    f"Derived negative level {level}; max_level={self.tree.max_level}."
                )
            self._cell_depth[cid] = depth
            scale = 1 << depth
            n0 = int(self.tree.root_shape[0]) * scale
            n1 = int(self.tree.root_shape[1]) * scale
            n2 = int(self.tree.root_shape[2]) * scale
            d0 = float(self._xyz_span[0] / n0)
            d1 = float(self._xyz_span[1] / n1)
            d2 = float(self._xyz_span[2] / n2)

            i0 = int(round((float(self._cell_x_min[cid]) - float(self._xyz_min[0])) / d0))
            i1 = int(round((float(self._cell_y_min[cid]) - float(self._xyz_min[1])) / d1))
            i2 = int(round((float(self._cell_z_min[cid]) - float(self._xyz_min[2])) / d2))
            if i0 < 0 or i0 >= n0 or i1 < 0 or i1 >= n1 or i2 < 0 or i2 >= n2:
                raise ValueError(f"Cartesian cell {cid} address is outside inferred grid bounds.")

            x0 = float(self._xyz_min[0] + i0 * d0)
            x1 = float(x0 + d0)
            y0 = float(self._xyz_min[1] + i1 * d1)
            y1 = float(y0 + d1)
            z0 = float(self._xyz_min[2] + i2 * d2)
            z1 = float(z0 + d2)
            if not np.isclose(float(self._cell_x_min[cid]), x0, rtol=0.0, atol=float(axis_tol[0])):
                raise ValueError(f"Cartesian cell {cid} x-min does not align with inferred octree grid.")
            if not np.isclose(float(self._cell_x_max[cid]), x1, rtol=0.0, atol=float(axis_tol[0])):
                raise ValueError(f"Cartesian cell {cid} x-max does not align with inferred octree grid.")
            if not np.isclose(float(self._cell_y_min[cid]), y0, rtol=0.0, atol=float(axis_tol[1])):
                raise ValueError(f"Cartesian cell {cid} y-min does not align with inferred octree grid.")
            if not np.isclose(float(self._cell_y_max[cid]), y1, rtol=0.0, atol=float(axis_tol[1])):
                raise ValueError(f"Cartesian cell {cid} y-max does not align with inferred octree grid.")
            if not np.isclose(float(self._cell_z_min[cid]), z0, rtol=0.0, atol=float(axis_tol[2])):
                raise ValueError(f"Cartesian cell {cid} z-min does not align with inferred octree grid.")
            if not np.isclose(float(self._cell_z_max[cid]), z1, rtol=0.0, atol=float(axis_tol[2])):
                raise ValueError(f"Cartesian cell {cid} z-max does not align with inferred octree grid.")

            self._i0[cid] = i0
            self._i1[cid] = i1
            self._i2[cid] = i2

            leaf_key = (depth, i0, i1, i2)
            if leaf_key in node_values:
                raise ValueError(f"Cartesian cells overlap at octree address {leaf_key}.")
            node_values[leaf_key] = cid

            for parent_depth in range(depth - 1, -1, -1):
                shift = depth - parent_depth
                parent_key = (parent_depth, i0 >> shift, i1 >> shift, i2 >> shift)
                existing = node_values.get(parent_key)
                if existing is None:
                    node_values[parent_key] = _INTERNAL_NODE_VALUE
                elif existing >= 0:
                    raise ValueError(f"Cartesian cells overlap across parent/child addresses at {parent_key}.")

        node_keys = np.array(list(node_values.keys()), dtype=np.int64)
        if node_keys.size == 0:
            raise ValueError("Cartesian lookup requires at least one valid cell address.")
        node_vals = np.array([node_values[tuple(key.tolist())] for key in node_keys], dtype=np.int64)
        order = np.lexsort((node_keys[:, 3], node_keys[:, 2], node_keys[:, 1], node_keys[:, 0]))
        node_keys = node_keys[order]
        node_vals = node_vals[order]
        self._node_depth = node_keys[:, 0].astype(np.int64, copy=False)
        self._node_i0 = node_keys[:, 1].astype(np.int64, copy=False)
        self._node_i1 = node_keys[:, 2].astype(np.int64, copy=False)
        self._node_i2 = node_keys[:, 3].astype(np.int64, copy=False)
        self._node_value = node_vals

    @staticmethod
    def _path(i0: int, i1: int, i2: int, level: int) -> GridPath:
        """Construct root-to-leaf index path for one leaf coordinate triplet."""
        out: list[GridIndex] = []
        for path_level in range(level + 1):
            shift = level - path_level
            out.append((i0 >> shift, i1 >> shift, i2 >> shift))
        return tuple(out)

    def _cell_bounds_xyz(self, cell_id: int) -> tuple[np.ndarray, np.ndarray]:
        cid = int(cell_id)
        return (
            np.array([self._cell_x_min[cid], self._cell_y_min[cid], self._cell_z_min[cid]], dtype=float),
            np.array([self._cell_x_max[cid], self._cell_y_max[cid], self._cell_z_max[cid]], dtype=float),
        )

    def _cell_bounds_rpa(self, cell_id: int) -> tuple[np.ndarray, np.ndarray]:
        cid = int(cell_id)
        corners = np.asarray(self._corners[cid], dtype=np.int64)
        pts = np.asarray(self._points[corners], dtype=float)
        r = np.linalg.norm(pts, axis=1)
        theta = np.arccos(np.clip(pts[:, 2] / np.maximum(r, np.finfo(float).tiny), -1.0, 1.0))
        phi = np.mod(np.arctan2(pts[:, 1], pts[:, 0]), 2.0 * np.pi)
        return (
            np.array([float(np.min(r)), float(np.min(theta)), float(np.min(phi))], dtype=float),
            np.array([float(np.max(r)), float(np.max(theta)), float(np.max(phi))], dtype=float),
        )

    def _domain_bounds_xyz(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self._xyz_min, dtype=float), np.asarray(self._xyz_max, dtype=float)

    def _domain_bounds_rpa(self) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(self._points, dtype=float)
        r = np.linalg.norm(pts, axis=1)
        theta = np.arccos(np.clip(pts[:, 2] / np.maximum(r, np.finfo(float).tiny), -1.0, 1.0))
        phi = np.mod(np.arctan2(pts[:, 1], pts[:, 0]), 2.0 * np.pi)
        return (
            np.array([float(np.min(r)), float(np.min(theta)), float(np.min(phi))], dtype=float),
            np.array([float(np.max(r)), float(np.max(theta)), float(np.max(phi))], dtype=float),
        )

    def contains_cell(
        self,
        cell_id: int,
        point: np.ndarray,
        *,
        coord: str,
        tol: float = _LOOKUP_CONTAIN_TOL,
    ) -> bool:
        """Return whether one point lies inside one Cartesian cell slab."""
        resolved = str(coord)
        if resolved != "xyz":
            raise ValueError("Cartesian lookup supports only coord='xyz'.")
        q = np.array(point, dtype=float).reshape(3)
        return self._contains_xyz_cell(
            int(cell_id),
            float(q[0]),
            float(q[1]),
            float(q[2]),
            tol=float(tol),
        )

    def lookup_cell_id(
        self,
        point: np.ndarray,
        *,
        coord: str,
    ) -> int:
        """Return the containing Cartesian cell id, or `-1` when not found."""
        resolved = str(coord)
        if resolved != "xyz":
            raise ValueError("Cartesian lookup supports only coord='xyz'.")
        q = np.array(point, dtype=float).reshape(3)
        return self._lookup_xyz_cell_id(float(q[0]), float(q[1]), float(q[2]))

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
            _contains_xyz_cell(
                int(cell_id),
                float(x),
                float(y),
                float(z),
                self._lookup_state,
                float(tol),
            )
        )

    def cell_step_hint(self, cell_id: int) -> float:
        """Return an initial step size hint for Python ray tracing."""
        cid = int(cell_id)
        return float(max(float(self._cell_dx[cid]), float(self._cell_dy[cid]), float(self._cell_dz[cid]), 1e-6))

    def _lookup_xyz_cell_id(self, x: float, y: float, z: float) -> int:
        """Return the containing cell id for `(x, y, z)`, or `-1`."""
        return int(
            _lookup_xyz_cell_id_kernel(
                float(x),
                float(y),
                float(z),
                self._lookup_state,
            )
        )

    def hit_from_chosen(self, chosen: int, *, allow_invalid_level: bool = False) -> LookupHit | None:
        """Build a `LookupHit` from an internal cell id."""
        if chosen < 0:
            return None
        center = self._cell_centers[chosen]
        level = int(self._cell_level[chosen])
        if level < 0 and not allow_invalid_level:
            return None
        if level < 0:
            path_level = int(self.tree.max_level)
        else:
            path_level = int(level)
            if path_level < 0:
                raise ValueError(f"Derived negative level {level}; max_level={self.tree.max_level}.")
        cell_i0 = int(self._i0[chosen])
        cell_i1 = int(self._i1[chosen])
        cell_i2 = int(self._i2[chosen])
        return LookupHit(
            cell_id=int(chosen),
            level=level,
            i0=cell_i0,
            i1=cell_i1,
            i2=cell_i2,
            path=self._path(cell_i0, cell_i1, cell_i2, path_level),
            center_xyz=(float(center[0]), float(center[1]), float(center[2])),
        )

class CartesianOctree(_CartesianCellLookup, Octree):
    """Octree specialization for Cartesian `(x, y, z)` datasets.

    For this backend, cells are represented as axis-aligned Cartesian boxes.
    """

    TREE_COORD: ClassVar[str | None] = "xyz"

    def build_lookup(
        self,
    ) -> None:
        """Build lookup arrays for this Cartesian tree."""
        if self.ds is None or self.ds.corners is None:
            raise ValueError("Octree is not bound to a dataset. Call bind(...) before lookup.")
        self._init_lookup_state(self)
