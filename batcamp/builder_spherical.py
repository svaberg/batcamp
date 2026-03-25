#!/usr/bin/env python3
"""Spherical (`rpa`) octree level and shape inference utilities."""

from __future__ import annotations

import math

import numpy as np
from batread.dataset import Dataset

from .builder import LevelShapeStatsMap
from .builder import _median_positive
from .builder import _resolve_cell_levels
from .octree import DEFAULT_AXIS_RHO_TOL
from .octree import Octree


class SphericalOctreeBuilder:
    """Coordinate-specific spherical inference strategy used by `OctreeBuilder`."""

    @staticmethod
    def _circular_span(cell_phi: np.ndarray) -> np.ndarray:
        """Compute minimal wrapped angular span for each row of azimuth samples."""
        ordered = np.sort(np.mod(cell_phi, 2.0 * math.pi), axis=1)
        wrapped = np.concatenate((ordered, ordered[:, :1] + 2.0 * math.pi), axis=1)
        gaps = np.diff(wrapped, axis=1)
        return 2.0 * math.pi - np.max(gaps, axis=1)

    @staticmethod
    def _circular_mean(cell_phi: np.ndarray) -> np.ndarray:
        """Compute circular mean for each row of azimuth samples."""
        mean_complex = np.mean(np.exp(1j * cell_phi), axis=1)
        return np.mod(np.angle(mean_complex), 2.0 * math.pi)

    @staticmethod
    def _circular_span_and_mean(
        cell_phi: np.ndarray,
        *,
        ignore_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-cell circular span and mean with optional corner masking."""
        if ignore_mask is None:
            return (
                SphericalOctreeBuilder._circular_span(cell_phi),
                SphericalOctreeBuilder._circular_mean(cell_phi),
            )
        if ignore_mask.shape != cell_phi.shape:
            raise ValueError(
                f"ignore_mask shape {ignore_mask.shape} does not match cell_phi {cell_phi.shape}"
            )

        n_cells = cell_phi.shape[0]
        span = np.empty(n_cells, dtype=float)
        center = np.empty(n_cells, dtype=float)
        row_has_mask = np.any(ignore_mask, axis=1)
        row_no_mask = ~row_has_mask

        if np.any(row_no_mask):
            span[row_no_mask] = SphericalOctreeBuilder._circular_span(cell_phi[row_no_mask])
            center[row_no_mask] = SphericalOctreeBuilder._circular_mean(cell_phi[row_no_mask])

        for cell_id in np.flatnonzero(row_has_mask):
            vals = cell_phi[cell_id, ~ignore_mask[cell_id]]
            if vals.size < 2:
                span[cell_id] = 0.0
                center[cell_id] = np.nan
                continue
            vals = vals.reshape(1, -1)
            span[cell_id] = SphericalOctreeBuilder._circular_span(vals)[0]
            center[cell_id] = SphericalOctreeBuilder._circular_mean(vals)[0]
        return span, center

    @staticmethod
    def _cluster_close_values(values: np.ndarray, *, atol: float) -> tuple[np.ndarray, np.ndarray]:
        """Cluster sorted boundary values within one absolute tolerance."""
        # TODO: Drive boundary clustering from the most probable spherical edge
        # lattice implied by the whole dataset, not this one absolute cutoff.
        ordered = np.sort(np.asarray(values, dtype=float).reshape(-1))
        if ordered.size == 0:
            return ordered, ordered
        clusters: list[list[float]] = [[float(ordered[0])]]
        prev = float(ordered[0])
        for value in ordered[1:]:
            current = float(value)
            if abs(current - prev) <= float(atol):
                clusters[-1].append(current)
                prev = current
                continue
            clusters.append([current])
            prev = current

        centers = np.empty(len(clusters), dtype=float)
        tolerances = np.empty(len(clusters), dtype=float)
        for idx, cluster in enumerate(clusters):
            arr = np.asarray(cluster, dtype=float)
            center = float(np.mean(arr))
            centers[idx] = center
            tolerances[idx] = max(float(np.max(np.abs(arr - center))), float(atol))
        return centers, tolerances

    @staticmethod
    def _minimal_phi_interval(values: np.ndarray) -> tuple[float, float]:
        """Return the smallest wrapped azimuth interval covering the samples."""
        vals = np.sort(np.mod(np.asarray(values, dtype=float), 2.0 * math.pi))
        if vals.size == 0:
            return 0.0, 2.0 * math.pi
        if vals.size == 1:
            return float(vals[0]), 0.0
        wrapped = np.concatenate((vals, vals[:1] + 2.0 * math.pi))
        gaps = np.diff(wrapped)
        k = int(np.argmax(gaps))
        start = float(wrapped[k + 1] % (2.0 * math.pi))
        width = float((2.0 * math.pi) - gaps[k])
        return start, width

    @staticmethod
    def _axis_corner_mask(ds: Dataset, corners: np.ndarray, *, axis_rho_tol: float) -> np.ndarray:
        """Mark corners near the polar axis where azimuth is singular."""
        names = set(ds.variables)
        if not set(Octree.XY_VARS).issubset(names):
            return np.zeros(corners.shape, dtype=bool)
        x = np.asarray(ds[Octree.X_VAR], dtype=float)
        y = np.asarray(ds[Octree.Y_VAR], dtype=float)
        rho = np.hypot(x, y)
        return rho[corners] <= float(axis_rho_tol)

    @staticmethod
    def _extract_phi(ds: Dataset) -> np.ndarray:
        """Extract wrapped azimuth values from dataset fields."""
        variable_names = set(ds.variables)
        if "Lon [deg]" in variable_names:
            lon_deg = np.asarray(ds["Lon [deg]"], dtype=float)
            return np.deg2rad(np.mod(lon_deg, 360.0))
        if "Lon [rad]" in variable_names:
            lon_rad = np.asarray(ds["Lon [rad]"], dtype=float)
            return np.mod(lon_rad, 2.0 * math.pi)
        if "phi [rad]" in variable_names:
            phi_rad = np.asarray(ds["phi [rad]"], dtype=float)
            return np.mod(phi_rad, 2.0 * math.pi)
        if set(Octree.XY_VARS).issubset(variable_names):
            x = np.asarray(ds[Octree.X_VAR], dtype=float)
            y = np.asarray(ds[Octree.Y_VAR], dtype=float)
            return np.mod(np.arctan2(y, x), 2.0 * math.pi)
        raise ValueError(
            "Could not determine phi. Need either (X [R], Y [R]) or Lon/phi fields. "
            f"Available variables are {list(ds.variables)}."
        )

    @staticmethod
    def infer_level_expectation(
        delta_phi: np.ndarray,
        *,
        rtol: float = 1e-4,
        atol: float = 1e-9,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Infer dyadic levels and expected spans from observed `delta_phi`."""
        # TODO: Replace these local span tolerances with one inference step that
        # fits the most probable dyadic angular spacing from noisy observed cells.
        levels = np.full(delta_phi.shape, -1, dtype=np.int64)
        expected = np.full(delta_phi.shape, np.nan, dtype=float)
        positive = delta_phi > max(float(atol), 1e-12)
        if not np.any(positive):
            return levels, expected, float("nan")

        coarse = float(np.max(delta_phi[positive]))
        raw_level = np.log2(coarse / delta_phi[positive])
        guess = np.maximum(np.rint(raw_level).astype(np.int64), 0)
        expected_pos = coarse / np.exp2(guess)
        ok = np.isclose(delta_phi[positive], expected_pos, rtol=rtol, atol=atol)
        levels[positive] = np.where(ok, guess, -1)
        expected[positive] = expected_pos
        return levels, expected, coarse

    @staticmethod
    def infer_levels_from_span(
        delta_phi: np.ndarray,
        *,
        rtol: float = 1e-4,
        atol: float = 1e-9,
    ) -> np.ndarray:
        """Infer integer dyadic levels from per-cell azimuth spans."""
        levels, _expected, _coarse = SphericalOctreeBuilder.infer_level_expectation(
            delta_phi,
            rtol=rtol,
            atol=atol,
        )
        return levels

    @staticmethod
    def compute_delta_phi_and_levels(
        ds: Dataset,
        *,
        corners: np.ndarray | None = None,
        rtol: float = 1e-4,
        atol: float = 1e-9,
        axis_rho_tol: float = DEFAULT_AXIS_RHO_TOL,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute per-cell `delta_phi` and inferred dyadic refinement levels."""
        source_corners = ds.corners if corners is None else corners
        if source_corners is None:
            raise ValueError("Dataset has no corners; cannot compute delta_phi levels.")
        corners_arr = np.asarray(source_corners, dtype=np.int64)
        if corners_arr.ndim != 2:
            raise ValueError(f"Expected 2D corner array, got shape {corners_arr.shape}.")
        if corners_arr.shape[1] < 3:
            raise ValueError("Need at least 3 corners per cell to estimate delta_phi.")

        phi = SphericalOctreeBuilder._extract_phi(ds)
        cell_phi = phi[corners_arr]
        axis_mask = SphericalOctreeBuilder._axis_corner_mask(ds, corners_arr, axis_rho_tol=axis_rho_tol)
        delta_phi, center_phi = SphericalOctreeBuilder._circular_span_and_mean(cell_phi, ignore_mask=axis_mask)
        levels, expected, coarse = SphericalOctreeBuilder.infer_level_expectation(
            delta_phi,
            rtol=rtol,
            atol=atol,
        )
        return delta_phi, center_phi, levels, expected, coarse

    @staticmethod
    def infer_level_angular_shapes(
        ds: Dataset,
        corners: np.ndarray,
        delta_phi: np.ndarray,
        cell_levels: np.ndarray,
    ) -> LevelShapeStatsMap:
        """Infer per-level angular counts/spacings from spherical mesh geometry."""
        x = np.asarray(ds[Octree.X_VAR], dtype=float)
        y = np.asarray(ds[Octree.Y_VAR], dtype=float)
        z = np.asarray(ds[Octree.Z_VAR], dtype=float)
        r = np.sqrt(x * x + y * y + z * z)
        theta = np.arccos(np.clip(z / np.maximum(r, np.finfo(float).tiny), -1.0, 1.0))
        delta_theta = np.ptp(theta[corners], axis=1)

        out: LevelShapeStatsMap = {}
        unique_levels = sorted(set(int(v) for v in cell_levels.tolist() if int(v) >= 0))
        if not unique_levels:
            raise ValueError("No valid (>=0) cell levels available for tree inference.")

        for level in unique_levels:
            mask = cell_levels == level
            med_dphi = _median_positive(delta_phi[mask])
            med_dtheta = _median_positive(delta_theta[mask])
            n_phi = int(round((2.0 * math.pi) / med_dphi))
            n_theta = int(round(math.pi / med_dtheta))
            if n_phi <= 0 or n_theta <= 0:
                raise ValueError(
                    f"Invalid angular counts inferred at level {level}: n_theta={n_theta}, n_phi={n_phi}."
                )

            ref_dphi = (2.0 * math.pi) / n_phi
            ref_dtheta = math.pi / n_theta
            # TODO: Infer the likeliest angular grid globally instead of relying
            # on these fixed validation thresholds for per-level medians.
            if not np.isclose(med_dphi, ref_dphi, rtol=2e-2, atol=1e-9):
                raise ValueError(
                    f"Level {level} has inconsistent dphi={med_dphi:.6e} vs inferred {ref_dphi:.6e}."
                )
            if not np.isclose(med_dtheta, ref_dtheta, rtol=2e-2, atol=1e-9):
                raise ValueError(
                    f"Level {level} has inconsistent dtheta={med_dtheta:.6e} vs inferred {ref_dtheta:.6e}."
                )
            out[level] = (n_theta, n_phi, med_dtheta, med_dphi, int(np.count_nonzero(mask)))
        return out

    def __init__(self, *, level_rtol: float = 1e-4, level_atol: float = 1e-9) -> None:
        """Store dyadic level-inference tolerances."""
        self.level_rtol = float(level_rtol)
        self.level_atol = float(level_atol)

    def infer_level_shapes(
        self,
        ds: Dataset,
        corners: np.ndarray,
        *,
        cell_levels: np.ndarray | None = None,
        axis_rho_tol: float = DEFAULT_AXIS_RHO_TOL,
    ) -> tuple[LevelShapeStatsMap, np.ndarray, int, int]:
        """Infer spherical level-shape map and validated levels."""
        delta_phi, _center_phi, auto_levels, _expected, _coarse = self.compute_delta_phi_and_levels(
            ds,
            corners=corners,
            rtol=self.level_rtol,
            atol=self.level_atol,
            axis_rho_tol=axis_rho_tol,
        )
        levels, min_level, max_level = _resolve_cell_levels(
            inferred_levels=auto_levels,
            cell_levels=cell_levels,
            expected_shape=auto_levels.shape,
        )
        level_shapes = self.infer_level_angular_shapes(ds, corners, delta_phi, levels)
        return level_shapes, levels, min_level, max_level

    @staticmethod
    def infer_leaf_shape(
        level_shapes: LevelShapeStatsMap,
    ) -> tuple[tuple[int, int, int], int]:
        """Infer finest spherical leaf shape and weighted finest-cell count."""
        max_level = max(level_shapes)
        n_axis1_f = int(level_shapes[max_level][0])
        n_axis2_f = int(level_shapes[max_level][1])
        weighted_cells = 0
        for level, (_n_axis1, _n_axis2, _d_axis1, _d_axis2, count) in level_shapes.items():
            weighted_cells += int(count) * (8 ** int(max_level - level))

        denom = int(n_axis1_f * n_axis2_f)
        if denom <= 0:
            raise ValueError("Invalid finest angular denominator while inferring n_axis0.")
        n_axis0_float = weighted_cells / float(denom)
        n_axis0 = int(round(n_axis0_float))
        if not np.isclose(n_axis0_float, float(n_axis0), rtol=0.0, atol=1e-9):
            raise ValueError(
                "Could not infer integer finest n_axis0 from weighted cell counts: "
                f"weighted={weighted_cells}, n_axis1={n_axis1_f}, n_axis2={n_axis2_f}."
            )
        return (n_axis0, n_axis1_f, n_axis2_f), int(weighted_cells)

    def infer_tree_geometry(
        self,
        ds: Dataset,
        corners: np.ndarray,
        *,
        cell_levels: np.ndarray | None = None,
        axis_rho_tol: float = DEFAULT_AXIS_RHO_TOL,
    ) -> tuple[LevelShapeStatsMap, np.ndarray, int, int, tuple[int, int, int], int]:
        """Infer spherical levels, per-level shapes, and finest leaf shape."""
        level_shapes, levels, min_level, max_level = self.infer_level_shapes(
            ds,
            corners,
            cell_levels=cell_levels,
            axis_rho_tol=axis_rho_tol,
        )
        leaf_shape, weighted_cells = self.infer_leaf_shape(level_shapes)
        return level_shapes, levels, min_level, max_level, leaf_shape, weighted_cells

    @staticmethod
    def populate_tree_state(
        *,
        leaf_shape: tuple[int, int, int],
        max_level: int,
        cell_levels: np.ndarray | None,
        axis_rho_tol: float,
        ds: Dataset,
        corners: np.ndarray,
    ) -> dict[str, object]:
        """Return exact spherical octree state for one built tree."""
        if cell_levels is None:
            raise ValueError("Spherical tree state requires cell_levels.")

        points = np.column_stack(
            (
                np.asarray(ds[Octree.X_VAR], dtype=float),
                np.asarray(ds[Octree.Y_VAR], dtype=float),
                np.asarray(ds[Octree.Z_VAR], dtype=float),
            )
        )
        corners_arr = np.asarray(corners, dtype=np.int64)
        cell_centers = np.mean(points[corners_arr], axis=1)
        levels = np.asarray(cell_levels, dtype=np.int64)
        valid = levels >= 0
        valid_ids = np.flatnonzero(valid).astype(np.int64)
        if valid_ids.size == 0:
            raise ValueError("Spherical tree state requires at least one valid cell level.")

        points_r = np.linalg.norm(points, axis=1)
        cell_r_min = np.min(points_r[corners_arr], axis=1)
        cell_r_max = np.max(points_r[corners_arr], axis=1)
        theta_points = np.arccos(
            np.clip(points[:, 2] / np.maximum(points_r, np.finfo(float).tiny), -1.0, 1.0)
        )
        cell_theta_min = np.min(theta_points[corners_arr], axis=1)
        cell_theta_max = np.max(theta_points[corners_arr], axis=1)
        phi_points = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * math.pi)
        axis_mask = SphericalOctreeBuilder._axis_corner_mask(
            ds,
            corners_arr,
            axis_rho_tol=float(axis_rho_tol),
        )
        phi_start = np.empty(levels.shape[0], dtype=float)
        phi_width = np.empty(levels.shape[0], dtype=float)
        phi_corners = phi_points[corners_arr]
        for cid in range(levels.shape[0]):
            vals = phi_corners[cid, ~axis_mask[cid]]
            if vals.size < 2:
                vals = phi_corners[cid]
            start, width = SphericalOctreeBuilder._minimal_phi_interval(vals)
            phi_start[cid] = start
            phi_width[cid] = width

        r_min = float(np.min(cell_r_min))
        r_max = float(np.max(cell_r_max))
        # TODO: Infer the radial edge lattice from clustered observed boundaries
        # instead of depending on this hard-coded clustering tolerance.
        radial_tol = 1e-7 * max(float(r_max - r_min), 1.0)
        radial_edges, radial_edge_tol = SphericalOctreeBuilder._cluster_close_values(
            np.concatenate((cell_r_min[valid], cell_r_max[valid])),
            atol=radial_tol,
        )
        expected_edges = int(leaf_shape[0]) + 1
        if radial_edges.size != expected_edges:
            raise ValueError(
                "Spherical radial edge count does not match leaf_shape: "
                f"edges={int(radial_edges.size)}, expected={expected_edges}."
            )

        tree_depth = int(max_level)
        depths = np.asarray(levels[valid_ids], dtype=np.int64)
        shifts = np.asarray(tree_depth - depths, dtype=np.int64)
        if np.any(shifts < 0):
            bad = int(valid_ids[np.flatnonzero(shifts < 0)[0]])
            raise ValueError(f"Spherical cell {bad} depth exceeds tree_depth={tree_depth}.")
        width_units = np.left_shift(np.ones_like(shifts, dtype=np.int64), shifts)

        d_theta_f = math.pi / float(int(leaf_shape[1]))
        d_phi_f = (2.0 * math.pi) / float(int(leaf_shape[2]))
        # TODO: Replace these angular snap tolerances with explicit inferred
        # theta/phi edge sets so noisy data maps to the most probable octree.
        theta_tol = max(1e-7 * math.pi, 2e-5 * d_theta_f)
        phi_tol = max(1e-7 * 2.0 * math.pi, 2e-5 * d_phi_f)
        r0_search = np.searchsorted(radial_edges, cell_r_min[valid_ids], side="left").astype(np.int64)
        r1_search = np.searchsorted(radial_edges, cell_r_max[valid_ids], side="left").astype(np.int64)
        r0_next = np.clip(r0_search, 0, radial_edges.size - 1)
        r1_next = np.clip(r1_search, 0, radial_edges.size - 1)
        r0_prev = np.clip(r0_search - 1, 0, radial_edges.size - 1)
        r1_prev = np.clip(r1_search - 1, 0, radial_edges.size - 1)
        r0_use_prev = (r0_search > 0) & (
            np.abs(radial_edges[r0_prev] - cell_r_min[valid_ids])
            <= np.abs(radial_edges[r0_next] - cell_r_min[valid_ids])
        )
        r1_use_prev = (r1_search > 0) & (
            np.abs(radial_edges[r1_prev] - cell_r_max[valid_ids])
            <= np.abs(radial_edges[r1_next] - cell_r_max[valid_ids])
        )
        r0_f = np.where(r0_use_prev, r0_prev, r0_next).astype(np.int64)
        r1_f = np.where(r1_use_prev, r1_prev, r1_next).astype(np.int64)
        n_r_fine = int(leaf_shape[0])
        if np.any(r0_f < 0) or np.any(r1_f > n_r_fine):
            bad = int(valid_ids[np.flatnonzero((r0_f < 0) | (r1_f > n_r_fine))[0]])
            raise ValueError(f"Spherical cell {bad} radial address is outside the inferred octree grid.")
        r0_ok = np.abs(radial_edges[r0_f] - cell_r_min[valid_ids]) <= radial_edge_tol[r0_f]
        r1_ok = np.abs(radial_edges[r1_f] - cell_r_max[valid_ids]) <= radial_edge_tol[r1_f]
        if np.any(~r0_ok):
            bad = int(valid_ids[np.flatnonzero(~r0_ok)[0]])
            raise ValueError(f"Spherical cell {bad} r-min does not align with the inferred octree grid.")
        if np.any(~r1_ok):
            bad = int(valid_ids[np.flatnonzero(~r1_ok)[0]])
            raise ValueError(f"Spherical cell {bad} r-max does not align with the inferred octree grid.")

        width_phi = np.asarray(phi_width[valid_ids], dtype=float)
        if np.any(width_phi >= (2.0 * math.pi - phi_tol)):
            bad = int(valid_ids[np.flatnonzero(width_phi >= (2.0 * math.pi - phi_tol))[0]])
            raise ValueError(f"Spherical cell {bad} spans the full azimuth and has no unique octree address.")

        i1_f = np.rint(cell_theta_min[valid_ids] / d_theta_f).astype(np.int64)
        i1_hi = np.rint(cell_theta_max[valid_ids] / d_theta_f).astype(np.int64)
        i2_f = np.rint(phi_start[valid_ids] / d_phi_f).astype(np.int64)
        i2_hi = np.rint((phi_start[valid_ids] + width_phi) / d_phi_f).astype(np.int64)
        n_theta_fine = int(leaf_shape[1])
        n_phi_fine = int(leaf_shape[2])
        in_bounds = (
            (i1_f >= 0) & (i1_hi <= n_theta_fine)
            & (i2_f >= 0) & (i2_hi <= n_phi_fine)
        )
        if not np.all(in_bounds):
            bad = int(valid_ids[np.flatnonzero(~in_bounds)[0]])
            raise ValueError(f"Spherical cell {bad} address is outside the inferred octree grid.")
        if np.any(~np.isclose(cell_theta_min[valid_ids], i1_f * d_theta_f, rtol=0.0, atol=theta_tol)):
            bad = int(valid_ids[np.flatnonzero(~np.isclose(cell_theta_min[valid_ids], i1_f * d_theta_f, rtol=0.0, atol=theta_tol))[0]])
            raise ValueError(f"Spherical cell {bad} theta-min does not align with the inferred octree grid.")
        if np.any(~np.isclose(cell_theta_max[valid_ids], i1_hi * d_theta_f, rtol=0.0, atol=theta_tol)):
            bad = int(valid_ids[np.flatnonzero(~np.isclose(cell_theta_max[valid_ids], i1_hi * d_theta_f, rtol=0.0, atol=theta_tol))[0]])
            raise ValueError(f"Spherical cell {bad} theta-max does not align with the inferred octree grid.")
        phi_delta = np.abs((phi_start[valid_ids] - (i2_f * d_phi_f) + math.pi) % (2.0 * math.pi) - math.pi)
        if np.any(phi_delta > phi_tol):
            bad = int(valid_ids[np.flatnonzero(phi_delta > phi_tol)[0]])
            raise ValueError(f"Spherical cell {bad} phi-start does not align with the inferred octree grid.")
        if np.any(~np.isclose(width_phi, (i2_hi - i2_f) * d_phi_f, rtol=0.0, atol=phi_tol)):
            bad = int(valid_ids[np.flatnonzero(~np.isclose(width_phi, (i2_hi - i2_f) * d_phi_f, rtol=0.0, atol=phi_tol))[0]])
            raise ValueError(f"Spherical cell {bad} phi-width does not align with the inferred octree grid.")

        spans_ok = (
            ((r1_f - r0_f) == width_units)
            & ((i1_hi - i1_f) == width_units)
            & ((i2_hi - i2_f) == width_units)
        )
        if not np.all(spans_ok):
            bad = int(valid_ids[np.flatnonzero(~spans_ok)[0]])
            raise ValueError(f"Spherical cell {bad} width does not match inferred level {int(levels[bad])}.")

        aligned = (
            ((r0_f % width_units) == 0)
            & ((i1_f % width_units) == 0)
            & ((i2_f % width_units) == 0)
        )
        if not np.all(aligned):
            bad = int(valid_ids[np.flatnonzero(~aligned)[0]])
            raise ValueError(f"Spherical cell {bad} fine-grid origin is not aligned to its inferred level.")

        cell_i0 = np.full(levels.shape[0], -1, dtype=np.int64)
        cell_i1 = np.full(levels.shape[0], -1, dtype=np.int64)
        cell_i2 = np.full(levels.shape[0], -1, dtype=np.int64)
        i0 = np.right_shift(r0_f, shifts)
        i1 = np.right_shift(i1_f, shifts)
        i2 = np.right_shift(i2_f, shifts)
        cell_i0[valid_ids] = i0
        cell_i1[valid_ids] = i1
        cell_i2[valid_ids] = i2

        return {
            "cell_levels": np.asarray(cell_levels, dtype=np.int64),
            "cell_i0": np.asarray(cell_i0, dtype=np.int64),
            "cell_i1": np.asarray(cell_i1, dtype=np.int64),
            "cell_i2": np.asarray(cell_i2, dtype=np.int64),
        }
