#!/usr/bin/env python3
"""Cartesian octree and Cartesian lookup implementation."""

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
_DEFAULT_LOOKUP_MAX_RADIUS = 2


class CartesianLookupKernelState(NamedTuple):
    """Arrays used by compiled Cartesian lookup code."""

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


@njit(cache=True)
def _contains_xyz_cell(
    cid: int,
    x: float,
    y: float,
    z: float,
    lookup_state: CartesianLookupKernelState,
    tol: float = _LOOKUP_CONTAIN_TOL,
) -> bool:
    """Check one Cartesian query against one cell AABB bounds."""
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
def _lookup_xyz_cell_id_kernel(
    x: float,
    y: float,
    z: float,
    lookup_state: CartesianLookupKernelState,
    prev_cid: int = -1,
) -> int:
    """Resolve one Cartesian query to a cell id using bin neighborhoods."""
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

    nbin = int(lookup_state.bin_shape[0])
    fx = (x - lookup_state.xyz_min[0]) / lookup_state.xyz_span[0]
    fy = (y - lookup_state.xyz_min[1]) / lookup_state.xyz_span[1]
    fz = (z - lookup_state.xyz_min[2]) / lookup_state.xyz_span[2]
    bx = int(math.floor(fx * nbin))
    by = int(math.floor(fy * nbin))
    bz = int(math.floor(fz * nbin))
    if bx < 0:
        bx = 0
    elif bx >= nbin:
        bx = nbin - 1
    if by < 0:
        by = 0
    elif by >= nbin:
        by = nbin - 1
    if bz < 0:
        bz = 0
    elif bz >= nbin:
        bz = nbin - 1

    for radius in range(int(lookup_state.max_radius) + 1):
        inside_best = -1
        inside_best_d2 = np.inf
        for dx in range(-radius, radius + 1):
            ix = bx + dx
            if ix < 0 or ix >= nbin:
                continue
            for dy in range(-radius, radius + 1):
                iy = by + dy
                if iy < 0 or iy >= nbin:
                    continue
                for dz in range(-radius, radius + 1):
                    iz = bz + dz
                    if iz < 0 or iz >= nbin:
                        continue
                    key = (ix * nbin + iy) * nbin + iz
                    start = int(lookup_state.bin_offsets[key])
                    end = int(lookup_state.bin_offsets[key + 1])
                    for pos in range(start, end):
                        cid = int(lookup_state.bin_cell_ids[pos])
                        cx = lookup_state.cell_centers[cid, 0]
                        cy = lookup_state.cell_centers[cid, 1]
                        cz = lookup_state.cell_centers[cid, 2]
                        d2 = (cx - x) * (cx - x) + (cy - y) * (cy - y) + (cz - z) * (cz - z)
                        if _contains_xyz_cell(cid, x, y, z, lookup_state) and d2 < inside_best_d2:
                            inside_best_d2 = d2
                            inside_best = cid
        if inside_best >= 0:
            return int(inside_best)

    n_cells = int(lookup_state.cell_centers.shape[0])
    all_inside_best = -1
    all_inside_best_d2 = np.inf
    for cid in range(n_cells):
        cx = lookup_state.cell_centers[cid, 0]
        cy = lookup_state.cell_centers[cid, 1]
        cz = lookup_state.cell_centers[cid, 2]
        d2 = (cx - x) * (cx - x) + (cy - y) * (cy - y) + (cz - z) * (cz - z)
        if _contains_xyz_cell(cid, x, y, z, lookup_state) and d2 < all_inside_best_d2:
            all_inside_best_d2 = d2
            all_inside_best = cid
    if all_inside_best >= 0:
        return int(all_inside_best)
    return -1


class _CartesianCellLookup:
    """Leaf-cell lookup accelerator for Cartesian `(x, y, z)` octrees."""

    def _init_lookup_state(self, tree: Octree) -> None:
        """Build per-cell lookup arrays from a bound Cartesian tree."""
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
                np.array(ds.variable(Octree.X_VAR), dtype=float),
                np.array(ds.variable(Octree.Y_VAR), dtype=float),
                np.array(ds.variable(Octree.Z_VAR), dtype=float),
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
            self._cell_level_rel = np.array(tree.cell_levels, dtype=np.int64)
        else:
            self._cell_level_rel = np.full(n_cells, int(tree.max_level), dtype=np.int64)
        self._cell_valid = self._cell_level_rel >= 0

        # Approximate leaf-grid indices from normalized center coordinates.
        lx, ly, lz = int(tree.leaf_shape[0]), int(tree.leaf_shape[1]), int(tree.leaf_shape[2])
        fx = (self._cell_centers[:, 0] - self._xyz_min[0]) / self._xyz_span[0]
        fy = (self._cell_centers[:, 1] - self._xyz_min[1]) / self._xyz_span[1]
        fz = (self._cell_centers[:, 2] - self._xyz_min[2]) / self._xyz_span[2]
        self._i0 = np.clip(np.floor(fx * lx), 0, max(lx - 1, 0)).astype(np.int64)
        self._i1 = np.clip(np.floor(fy * ly), 0, max(ly - 1, 0)).astype(np.int64)
        self._i2 = np.clip(np.floor(fz * lz), 0, max(lz - 1, 0)).astype(np.int64)

        nbin = max(4, int(round(n_cells ** (1.0 / 3.0))))
        self._bin_shape = np.array([nbin, nbin, nbin], dtype=np.int64)
        bxyz = np.floor(((self._cell_centers - self._xyz_min[None, :]) / self._xyz_span[None, :]) * nbin).astype(
            np.int64
        )
        bxyz = np.clip(bxyz, 0, nbin - 1)
        keys = (bxyz[:, 0] * nbin + bxyz[:, 1]) * nbin + bxyz[:, 2]
        n_bins = int(nbin * nbin * nbin)
        bin_lists: list[list[int]] = [[] for _ in range(n_bins)]
        for cid in range(n_cells):
            bin_lists[int(keys[cid])].append(cid)

        bin_counts = np.zeros(n_bins, dtype=np.int64)
        for key, ids in enumerate(bin_lists):
            if not ids:
                continue
            arr = np.array(ids, dtype=np.int64)
            order = np.argsort(np.linalg.norm(self._cell_centers[arr], axis=1))
            sorted_ids = arr[order]
            bin_lists[key] = sorted_ids.tolist()
            bin_counts[key] = int(sorted_ids.size)

        bin_offsets = np.zeros(n_bins + 1, dtype=np.int64)
        if n_bins > 0:
            np.cumsum(bin_counts, out=bin_offsets[1:])
        self._bin_counts = bin_counts
        self._bin_offsets = bin_offsets
        total_refs = int(bin_offsets[-1])
        self._bin_cell_ids = np.empty(total_refs, dtype=np.int64)
        for key in range(n_bins):
            start = int(bin_offsets[key])
            end = int(bin_offsets[key + 1])
            if end > start:
                self._bin_cell_ids[start:end] = np.array(bin_lists[key], dtype=np.int64)

        self._max_radius = int(_DEFAULT_LOOKUP_MAX_RADIUS)
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
        )

    @staticmethod
    def _path(i0: int, i1: int, i2: int, depth: int) -> GridPath:
        """Construct root-to-leaf index path for one leaf coordinate triplet."""
        out: list[GridIndex] = []
        for level in range(depth + 1):
            shift = depth - level
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
        """Return whether one point lies inside one Cartesian cell."""
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

    def hit_from_chosen(self, chosen: int, *, allow_invalid_depth: bool = False) -> LookupHit | None:
        """Build a `LookupHit` from an internal cell id."""
        if chosen < 0:
            return None
        center = self._cell_centers[chosen]
        level = int(self._cell_level_rel[chosen])
        if level < 0 and not allow_invalid_depth:
            return None
        if level < 0:
            depth = int(self.tree.depth)
        else:
            depth = int(self.tree.depth - (int(self.tree.max_level) - level))
            if depth < 0:
                raise ValueError(
                    f"Derived negative tree depth for level {level}; "
                    f"tree.depth={self.tree.depth}, max_level={self.tree.max_level}."
                )
        cell_i0 = int(self._i0[chosen])
        cell_i1 = int(self._i1[chosen])
        cell_i2 = int(self._i2[chosen])
        return LookupHit(
            cell_id=int(chosen),
            level=level,
            i0=cell_i0,
            i1=cell_i1,
            i2=cell_i2,
            path=self._path(cell_i0, cell_i1, cell_i2, depth),
            center_xyz=(float(center[0]), float(center[1]), float(center[2])),
        )

class CartesianOctree(_CartesianCellLookup, Octree):
    """Octree specialization placeholder for Cartesian `(x, y, z)` datasets."""

    TREE_COORD: ClassVar[str | None] = "xyz"

    def build_lookup(
        self,
    ) -> None:
        """Build lookup arrays for this Cartesian tree."""
        if self.ds is None or self.ds.corners is None:
            raise ValueError("Octree is not bound to a dataset. Call bind(...) before lookup.")
        self._init_lookup_state(self)
