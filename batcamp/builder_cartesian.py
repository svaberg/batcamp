#!/usr/bin/env python3
"""Cartesian (`xyz`) octree level and shape inference utilities."""

from __future__ import annotations

import numpy as np
from batread import Dataset

from .constants import XYZ_VARS
from .builder import LevelShapeStatsMap
from .builder import _median_positive
from .builder import _resolve_cell_levels


class CartesianOctreeBuilder:
    """Coordinate-specific Cartesian inference strategy used by `OctreeBuilder`."""

    @staticmethod
    def infer_xyz_levels_from_cell_spans(
        dx: np.ndarray,
        dy: np.ndarray,
        dz: np.ndarray,
        *,
        rtol: float = 2e-2,
        atol: float = 1e-10,
    ) -> np.ndarray:
        """Infer dyadic xyz refinement levels from per-cell axis-aligned spans."""
        levels = np.full(dx.shape, -1, dtype=np.int64)
        tiny = max(float(atol), 1e-12)
        valid = (dx > tiny) & (dy > tiny) & (dz > tiny) & np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dz)
        if not np.any(valid):
            return levels

        coarse_dx = float(np.max(dx[valid]))
        coarse_dy = float(np.max(dy[valid]))
        coarse_dz = float(np.max(dz[valid]))

        raw_x = np.log2(coarse_dx / dx[valid])
        raw_y = np.log2(coarse_dy / dy[valid])
        raw_z = np.log2(coarse_dz / dz[valid])
        guess = np.maximum(np.rint((raw_x + raw_y + raw_z) / 3.0).astype(np.int64), 0)

        exp_x = coarse_dx / np.exp2(guess)
        exp_y = coarse_dy / np.exp2(guess)
        exp_z = coarse_dz / np.exp2(guess)
        ok = (
            np.isclose(dx[valid], exp_x, rtol=rtol, atol=atol)
            & np.isclose(dy[valid], exp_y, rtol=rtol, atol=atol)
            & np.isclose(dz[valid], exp_z, rtol=rtol, atol=atol)
        )
        levels[valid] = np.where(ok, guess, -1)
        return levels

    @staticmethod
    def infer_xyz_level_shapes(
        ds: Dataset,
        corners: np.ndarray,
        cell_levels: np.ndarray,
    ) -> LevelShapeStatsMap:
        """Infer per-level axis counts/spacings for Cartesian octrees."""
        x = np.asarray(ds[XYZ_VARS[0]], dtype=float)
        y = np.asarray(ds[XYZ_VARS[1]], dtype=float)
        z = np.asarray(ds[XYZ_VARS[2]], dtype=float)
        cell_x = x[corners]
        cell_y = y[corners]
        cell_z = z[corners]
        dx = np.ptp(cell_x, axis=1)
        dy = np.ptp(cell_y, axis=1)
        dz = np.ptp(cell_z, axis=1)

        x_min = float(np.min(np.min(cell_x, axis=1)))
        x_max = float(np.max(np.max(cell_x, axis=1)))
        y_min = float(np.min(np.min(cell_y, axis=1)))
        y_max = float(np.max(np.max(cell_y, axis=1)))
        z_min = float(np.min(np.min(cell_z, axis=1)))
        z_max = float(np.max(np.max(cell_z, axis=1)))

        span_x = max(x_max - x_min, np.finfo(float).tiny)
        span_y = max(y_max - y_min, np.finfo(float).tiny)
        span_z = max(z_max - z_min, np.finfo(float).tiny)

        out: LevelShapeStatsMap = {}
        unique_levels = sorted(set(int(v) for v in cell_levels.tolist() if int(v) >= 0))
        if not unique_levels:
            raise ValueError("No valid (>=0) cell levels available for tree inference.")

        for level in unique_levels:
            mask = cell_levels == level
            med_dx = _median_positive(dx[mask])
            med_dy = _median_positive(dy[mask])
            med_dz = _median_positive(dz[mask])
            n_x = int(round(span_x / med_dx))
            n_y = int(round(span_y / med_dy))
            n_z = int(round(span_z / med_dz))
            if n_x <= 0 or n_y <= 0 or n_z <= 0:
                raise ValueError(
                    f"Invalid xyz counts inferred at level {level}: n_x={n_x}, n_y={n_y}, n_z={n_z}."
                )
            ref_dx = span_x / n_x
            ref_dy = span_y / n_y
            ref_dz = span_z / n_z
            if not np.isclose(med_dx, ref_dx, rtol=2e-2, atol=1e-9):
                raise ValueError(f"Level {level} has inconsistent dx={med_dx:.6e} vs inferred {ref_dx:.6e}.")
            if not np.isclose(med_dy, ref_dy, rtol=2e-2, atol=1e-9):
                raise ValueError(f"Level {level} has inconsistent dy={med_dy:.6e} vs inferred {ref_dy:.6e}.")
            if not np.isclose(med_dz, ref_dz, rtol=2e-2, atol=1e-9):
                raise ValueError(f"Level {level} has inconsistent dz={med_dz:.6e} vs inferred {ref_dz:.6e}.")
            out[level] = (n_y, n_z, med_dy, med_dz, int(np.count_nonzero(mask)))
        return out

    def __init__(self, *, level_rtol: float = 1e-4, level_atol: float = 1e-9) -> None:
        """Store level-inference tolerances."""
        self.level_rtol = float(level_rtol)
        self.level_atol = float(level_atol)

    def infer_level_shapes(
        self,
        ds: Dataset,
        corners: np.ndarray,
        *,
        cell_levels: np.ndarray | None = None,
    ) -> tuple[LevelShapeStatsMap, np.ndarray, int]:
        """Infer Cartesian level-shape map and validated levels."""
        x = np.asarray(ds[XYZ_VARS[0]], dtype=float)
        y = np.asarray(ds[XYZ_VARS[1]], dtype=float)
        z = np.asarray(ds[XYZ_VARS[2]], dtype=float)
        cell_x = x[corners]
        cell_y = y[corners]
        cell_z = z[corners]
        dx = np.ptp(cell_x, axis=1)
        dy = np.ptp(cell_y, axis=1)
        dz = np.ptp(cell_z, axis=1)

        inferred_levels: np.ndarray | None = None
        if cell_levels is None:
            inferred_levels = self.infer_xyz_levels_from_cell_spans(
                dx,
                dy,
                dz,
                rtol=max(2e-2, float(self.level_rtol)),
                atol=max(1e-10, float(self.level_atol)),
            )
        levels, _min_level, max_level = _resolve_cell_levels(
            inferred_levels=inferred_levels,
            cell_levels=cell_levels,
            expected_shape=dx.shape,
        )
        level_shapes = self.infer_xyz_level_shapes(ds, corners, levels)
        return level_shapes, levels, max_level

    @staticmethod
    def infer_leaf_shape(
        ds: Dataset,
        corners: np.ndarray,
        cell_levels: np.ndarray,
        *,
        max_level: int,
    ) -> tuple[int, int, int]:
        """Infer finest Cartesian `(n_x, n_y, n_z)` counts from geometry at `max_level`."""
        x = np.asarray(ds[XYZ_VARS[0]], dtype=float)
        y = np.asarray(ds[XYZ_VARS[1]], dtype=float)
        z = np.asarray(ds[XYZ_VARS[2]], dtype=float)
        cell_x = x[corners]
        cell_y = y[corners]
        cell_z = z[corners]
        dx = np.ptp(cell_x, axis=1)
        dy = np.ptp(cell_y, axis=1)
        dz = np.ptp(cell_z, axis=1)
        mask = np.asarray(cell_levels, dtype=np.int64) == int(max_level)
        if not np.any(mask):
            raise ValueError(f"No cells found at max_level={max_level}.")

        x_min = float(np.min(np.min(cell_x, axis=1)))
        x_max = float(np.max(np.max(cell_x, axis=1)))
        y_min = float(np.min(np.min(cell_y, axis=1)))
        y_max = float(np.max(np.max(cell_y, axis=1)))
        z_min = float(np.min(np.min(cell_z, axis=1)))
        z_max = float(np.max(np.max(cell_z, axis=1)))
        span_x = max(x_max - x_min, np.finfo(float).tiny)
        span_y = max(y_max - y_min, np.finfo(float).tiny)
        span_z = max(z_max - z_min, np.finfo(float).tiny)

        finest_dx = _median_positive(dx[mask])
        finest_dy = _median_positive(dy[mask])
        finest_dz = _median_positive(dz[mask])
        n_x = int(round(span_x / finest_dx))
        n_y = int(round(span_y / finest_dy))
        n_z = int(round(span_z / finest_dz))
        if n_x <= 0 or n_y <= 0 or n_z <= 0:
            raise ValueError(
                f"Invalid finest Cartesian shape inferred: n_x={n_x}, n_y={n_y}, n_z={n_z}."
            )
        return n_x, n_y, n_z

    @staticmethod
    def populate_tree_state(
        *,
        leaf_shape: tuple[int, int, int],
        max_level: int,
        cell_levels: np.ndarray | None,
        ds: Dataset,
        corners: np.ndarray,
    ) -> dict[str, object]:
        """Return exact Cartesian octree state for one built tree."""
        if cell_levels is None:
            raise ValueError("Cartesian tree state requires cell_levels.")

        points = np.column_stack(
            (
                np.asarray(ds[XYZ_VARS[0]], dtype=float),
                np.asarray(ds[XYZ_VARS[1]], dtype=float),
                np.asarray(ds[XYZ_VARS[2]], dtype=float),
            )
        )
        corners_arr = np.asarray(corners, dtype=np.int64)
        cell_xyz = points[corners_arr]
        cell_x_min = np.min(cell_xyz[:, :, 0], axis=1)
        cell_x_max = np.max(cell_xyz[:, :, 0], axis=1)
        cell_y_min = np.min(cell_xyz[:, :, 1], axis=1)
        cell_y_max = np.max(cell_xyz[:, :, 1], axis=1)
        cell_z_min = np.min(cell_xyz[:, :, 2], axis=1)
        cell_z_max = np.max(cell_xyz[:, :, 2], axis=1)

        xyz_min = np.array(
            [
                float(np.min(cell_x_min)),
                float(np.min(cell_y_min)),
                float(np.min(cell_z_min)),
            ],
            dtype=float,
        )
        xyz_max = np.array(
            [
                float(np.max(cell_x_max)),
                float(np.max(cell_y_max)),
                float(np.max(cell_z_max)),
            ],
            dtype=float,
        )
        xyz_span = np.maximum(xyz_max - xyz_min, np.finfo(float).tiny)
        leaf_shape_arr = np.asarray(leaf_shape, dtype=np.int64)
        fine_step = xyz_span / np.asarray(leaf_shape_arr, dtype=float)
        float32_eps = float(np.finfo(np.float32).eps)
        axis_tol = np.empty(3, dtype=float)
        for k in range(3):
            coord_scale = max(abs(float(xyz_min[k])), abs(float(xyz_max[k])), 1.0)
            axis_tol[k] = min(
                0.25 * float(fine_step[k]),
                max(8.0 * float32_eps * coord_scale, 1e-9 * max(float(xyz_span[k]), 1.0)),
            )

        levels = np.asarray(cell_levels, dtype=np.int64)
        valid_ids = np.flatnonzero(levels >= 0).astype(np.int64)
        if valid_ids.size == 0:
            raise ValueError("Cartesian tree state requires at least one valid cell level.")

        depths = np.asarray(levels[valid_ids], dtype=np.int64)
        tree_depth = int(max_level)
        shifts = np.asarray(tree_depth - depths, dtype=np.int64)
        if np.any(shifts < 0):
            bad = int(valid_ids[np.flatnonzero(shifts < 0)[0]])
            raise ValueError(f"Cartesian cell {bad} depth exceeds tree_depth={tree_depth}.")
        width_units = np.left_shift(np.ones_like(shifts, dtype=np.int64), shifts)

        x0_f = np.rint((cell_x_min[valid_ids] - float(xyz_min[0])) / float(fine_step[0])).astype(np.int64)
        x1_f = np.rint((cell_x_max[valid_ids] - float(xyz_min[0])) / float(fine_step[0])).astype(np.int64)
        y0_f = np.rint((cell_y_min[valid_ids] - float(xyz_min[1])) / float(fine_step[1])).astype(np.int64)
        y1_f = np.rint((cell_y_max[valid_ids] - float(xyz_min[1])) / float(fine_step[1])).astype(np.int64)
        z0_f = np.rint((cell_z_min[valid_ids] - float(xyz_min[2])) / float(fine_step[2])).astype(np.int64)
        z1_f = np.rint((cell_z_max[valid_ids] - float(xyz_min[2])) / float(fine_step[2])).astype(np.int64)

        fine_n0 = int(leaf_shape_arr[0])
        fine_n1 = int(leaf_shape_arr[1])
        fine_n2 = int(leaf_shape_arr[2])
        in_bounds = (
            (x0_f >= 0) & (x1_f <= fine_n0)
            & (y0_f >= 0) & (y1_f <= fine_n1)
            & (z0_f >= 0) & (z1_f <= fine_n2)
        )
        if not np.all(in_bounds):
            bad = int(valid_ids[np.flatnonzero(~in_bounds)[0]])
            raise ValueError(f"Cartesian cell {bad} address is outside inferred grid bounds.")

        spans_ok = (
            ((x1_f - x0_f) == width_units)
            & ((y1_f - y0_f) == width_units)
            & ((z1_f - z0_f) == width_units)
        )
        if not np.all(spans_ok):
            bad = int(valid_ids[np.flatnonzero(~spans_ok)[0]])
            raise ValueError(f"Cartesian cell {bad} width does not match inferred level {int(levels[bad])}.")

        aligned = (
            ((x0_f % width_units) == 0)
            & ((y0_f % width_units) == 0)
            & ((z0_f % width_units) == 0)
        )
        if not np.all(aligned):
            bad = int(valid_ids[np.flatnonzero(~aligned)[0]])
            raise ValueError(f"Cartesian cell {bad} fine-grid origin is not aligned to its inferred level.")

        x0_snap = float(xyz_min[0]) + x0_f * float(fine_step[0])
        x1_snap = float(xyz_min[0]) + x1_f * float(fine_step[0])
        y0_snap = float(xyz_min[1]) + y0_f * float(fine_step[1])
        y1_snap = float(xyz_min[1]) + y1_f * float(fine_step[1])
        z0_snap = float(xyz_min[2]) + z0_f * float(fine_step[2])
        z1_snap = float(xyz_min[2]) + z1_f * float(fine_step[2])
        snap_ok = (
            np.isclose(cell_x_min[valid_ids], x0_snap, rtol=0.0, atol=float(axis_tol[0]))
            & np.isclose(cell_x_max[valid_ids], x1_snap, rtol=0.0, atol=float(axis_tol[0]))
            & np.isclose(cell_y_min[valid_ids], y0_snap, rtol=0.0, atol=float(axis_tol[1]))
            & np.isclose(cell_y_max[valid_ids], y1_snap, rtol=0.0, atol=float(axis_tol[1]))
            & np.isclose(cell_z_min[valid_ids], z0_snap, rtol=0.0, atol=float(axis_tol[2]))
            & np.isclose(cell_z_max[valid_ids], z1_snap, rtol=0.0, atol=float(axis_tol[2]))
        )
        if not np.all(snap_ok):
            bad = int(valid_ids[np.flatnonzero(~snap_ok)[0]])
            raise ValueError(f"Cartesian cell {bad} bounds do not align with inferred octree grid.")

        cell_ijk = np.full((levels.shape[0], 3), -1, dtype=np.int64)
        i0 = np.right_shift(x0_f, shifts)
        i1 = np.right_shift(y0_f, shifts)
        i2 = np.right_shift(z0_f, shifts)
        cell_ijk[valid_ids, 0] = i0
        cell_ijk[valid_ids, 1] = i1
        cell_ijk[valid_ids, 2] = i2

        return {
            "cell_levels": np.asarray(cell_levels, dtype=np.int64),
            "cell_ijk": np.asarray(cell_ijk, dtype=np.int64),
        }
