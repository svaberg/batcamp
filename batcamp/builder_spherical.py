#!/usr/bin/env python3
"""Spherical (`rpa`) octree level and shape inference utilities."""

from __future__ import annotations

import logging
import time

import numpy as np

from .builder import DEFAULT_AXIS_TOL
from .builder import DEFAULT_LEVEL_ATOL
from .builder import DEFAULT_LEVEL_RTOL
from .builder import DEFAULT_POSITIVE_TINY
from .builder import LevelShapeStatsMap
from .builder import SHAPE_MATCH_ATOL
from .builder import SHAPE_MATCH_RTOL
from .builder import median_positive
from .builder import _resolve_cell_levels

logger = logging.getLogger(__name__)

SNAP_TOPOLOGY_WARNING_FRACTION = 1e-2
"""Warn only when a snap residual reaches this fraction of a topology-changing margin."""


def cluster_close_values(values: np.ndarray, *, atol: float) -> tuple[np.ndarray, np.ndarray]:
    """Cluster sorted boundary values within one absolute tolerance."""
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


def nearest_cluster_indices(centers: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map each value to the nearest sorted cluster center index."""
    search = np.searchsorted(centers, values, side="left").astype(np.int64)
    next_idx = np.clip(search, 0, centers.size - 1)
    prev_idx = np.clip(search - 1, 0, centers.size - 1)
    use_prev = (search > 0) & (
        np.abs(centers[prev_idx] - values) <= np.abs(centers[next_idx] - values)
    )
    return np.where(use_prev, prev_idx, next_idx).astype(np.int64)


def minimal_azimuth_intervals(
    cell_azimuth: np.ndarray,
    *,
    ignore_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return minimal wrapped azimuth interval start/width for each cell row."""
    def intervals_from_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vals = np.sort(np.mod(values, 2.0 * np.pi), axis=1)
        wrapped = np.concatenate((vals, vals[:, :1] + 2.0 * np.pi), axis=1)
        gaps = np.diff(wrapped, axis=1)
        gap_id = np.argmax(gaps, axis=1)
        start = np.take_along_axis(wrapped[:, 1:], gap_id[:, None], axis=1)[:, 0]
        width = (2.0 * np.pi) - gaps[np.arange(gaps.shape[0]), gap_id]
        return np.mod(start, 2.0 * np.pi), width

    if ignore_mask is None:
        return intervals_from_rows(cell_azimuth)
    if ignore_mask.shape != cell_azimuth.shape:
        raise ValueError(
            f"ignore_mask shape {ignore_mask.shape} does not match cell_azimuth {cell_azimuth.shape}"
        )

    n_cells = cell_azimuth.shape[0]
    start = np.empty(n_cells, dtype=float)
    width = np.empty(n_cells, dtype=float)
    row_has_mask = np.any(ignore_mask, axis=1)
    row_no_mask = ~row_has_mask

    if np.any(row_no_mask):
        row_start, row_width = intervals_from_rows(cell_azimuth[row_no_mask])
        start[row_no_mask] = row_start
        width[row_no_mask] = row_width

    for cell_id in np.flatnonzero(row_has_mask):
        vals = cell_azimuth[cell_id, ~ignore_mask[cell_id]]
        if vals.size < 2:
            vals = cell_azimuth[cell_id]
        row_start, row_width = intervals_from_rows(np.asarray(vals, dtype=float).reshape(1, -1))
        start[cell_id] = row_start[0]
        width[cell_id] = row_width[0]
    return start, width


def infer_level_expectation(
    azimuth_span: np.ndarray,
    *,
    rtol: float = DEFAULT_LEVEL_RTOL,
    atol: float = DEFAULT_LEVEL_ATOL,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Infer dyadic levels and expected spans from observed azimuth spans."""
    levels = np.full(azimuth_span.shape, -1, dtype=np.int64)
    expected = np.full(azimuth_span.shape, np.nan, dtype=float)
    positive = azimuth_span > max(float(atol), DEFAULT_POSITIVE_TINY)
    if not np.any(positive):
        return levels, expected, float("nan")

    coarse = float(np.max(azimuth_span[positive]))
    raw_level = np.log2(coarse / azimuth_span[positive])
    guess = np.maximum(np.rint(raw_level).astype(np.int64), 0)
    expected_pos = coarse / np.exp2(guess)
    ok = np.isclose(azimuth_span[positive], expected_pos, rtol=rtol, atol=atol)
    expected[positive] = expected_pos
    if not np.all(ok):
        bad_ids = np.flatnonzero(positive)
        bad_mask = ~ok
        bad_local_ids = bad_ids[bad_mask]
        residual = np.abs(azimuth_span[bad_local_ids] - expected[bad_local_ids])
        tol_scale = np.maximum(float(atol), np.abs(expected[bad_local_ids]) * float(rtol))
        max_local = int(np.argmax(residual))
        worst_cell = int(bad_local_ids[max_local])
        worst_residual = float(residual[max_local])
        worst_tol = float(tol_scale[max_local])
        worst_ratio = worst_residual / worst_tol if worst_tol > 0.0 else float("inf")
        raise ValueError(
            "Could not infer one dyadic spherical refinement level per cell from azimuth span. "
            f"mismatch_count={int(np.count_nonzero(bad_mask))} "
            f"max_abs_residual={worst_residual:.17g} "
            f"max_tol={worst_tol:.17g} "
            f"max_residual_over_tol={worst_ratio:.6g} "
            f"worst_cell={worst_cell} "
            f"observed_span={float(azimuth_span[worst_cell]):.17g} "
            f"expected_span={float(expected[worst_cell]):.17g} "
            f"inferred_level={int(guess[bad_mask][max_local])}."
        )
    levels[positive] = guess
    return levels, expected, coarse


def infer_level_angular_shapes(
    azimuth_span: np.ndarray,
    cell_polar_min: np.ndarray,
    cell_polar_max: np.ndarray,
    cell_levels: np.ndarray,
) -> LevelShapeStatsMap:
    """Infer per-level angular counts/spacings from spherical mesh geometry."""
    delta_polar = np.asarray(cell_polar_max, dtype=float) - np.asarray(cell_polar_min, dtype=float)

    out: LevelShapeStatsMap = {}
    unique_levels = sorted(set(int(v) for v in cell_levels.tolist() if int(v) >= 0))
    if not unique_levels:
        raise ValueError("No valid (>=0) cell levels available for tree inference.")

    for level in unique_levels:
        mask = cell_levels == level
        med_dazimuth = median_positive(azimuth_span[mask])
        med_dpolar = median_positive(delta_polar[mask])
        n_azimuth = int(round((2.0 * np.pi) / med_dazimuth))
        n_polar = int(round(np.pi / med_dpolar))
        if n_azimuth <= 0 or n_polar <= 0:
            raise ValueError(
                f"Invalid angular counts inferred at level {level}: n_polar={n_polar}, n_azimuth={n_azimuth}."
            )

        ref_dazimuth = (2.0 * np.pi) / n_azimuth
        ref_dpolar = np.pi / n_polar
        if not np.isclose(med_dazimuth, ref_dazimuth, rtol=SHAPE_MATCH_RTOL, atol=SHAPE_MATCH_ATOL):
            raise ValueError(
                f"Level {int(level)} has inconsistent dazimuth={float(med_dazimuth):.17g} "
                f"vs inferred={float(ref_dazimuth):.17g}; "
                f"abs_residual={abs(float(med_dazimuth - ref_dazimuth)):.17g} "
                f"rtol={float(SHAPE_MATCH_RTOL):.3g} atol={float(SHAPE_MATCH_ATOL):.3g}."
            )
        if not np.isclose(med_dpolar, ref_dpolar, rtol=SHAPE_MATCH_RTOL, atol=SHAPE_MATCH_ATOL):
            raise ValueError(
                f"Level {int(level)} has inconsistent dpolar={float(med_dpolar):.17g} "
                f"vs inferred={float(ref_dpolar):.17g}; "
                f"abs_residual={abs(float(med_dpolar - ref_dpolar)):.17g} "
                f"rtol={float(SHAPE_MATCH_RTOL):.3g} atol={float(SHAPE_MATCH_ATOL):.3g}."
            )
        out[level] = (n_polar, n_azimuth, med_dpolar, med_dazimuth, int(np.count_nonzero(mask)))
    return out


def infer_levels(
    points: np.ndarray,
    corners: np.ndarray,
    *,
    cell_levels: np.ndarray | None = None,
    axis_tol: float = DEFAULT_AXIS_TOL,
    level_rtol: float = DEFAULT_LEVEL_RTOL,
    level_atol: float = DEFAULT_LEVEL_ATOL,
) -> tuple[np.ndarray, int, tuple[int, int, int], dict[str, object]]:
    """Infer spherical levels and return the finest spherical shape plus reusable builder state."""
    logger.info("_observed_spherical_bounds...")
    t0 = time.perf_counter()
    cell_log_r_min, cell_log_r_max, cell_polar_min, cell_polar_max, azimuth_start, azimuth_width = (
        _observed_spherical_bounds(
            points,
            corners,
            axis_tol=axis_tol,
        )
    )
    logger.info("_observed_spherical_bounds complete in %.2fs", float(time.perf_counter() - t0))
    logger.info("infer_level_expectation...")
    t0 = time.perf_counter()
    auto_levels, _expected, _coarse = infer_level_expectation(
        azimuth_width,
        rtol=level_rtol,
        atol=level_atol,
    )
    logger.info("infer_level_expectation complete in %.2fs", float(time.perf_counter() - t0))
    levels, max_level = _resolve_cell_levels(
        inferred_levels=auto_levels,
        cell_levels=cell_levels,
        expected_shape=auto_levels.shape,
    )
    try:
        logger.info("infer_level_angular_shapes...")
        t0 = time.perf_counter()
        level_shapes = infer_level_angular_shapes(
            azimuth_width,
            cell_polar_min,
            cell_polar_max,
            levels,
        )
        logger.info("infer_level_angular_shapes complete in %.2fs", float(time.perf_counter() - t0))
        logger.info("infer_leaf_shape...")
        t0 = time.perf_counter()
        angular_leaf_shape = infer_leaf_shape(level_shapes)
        logger.info("infer_leaf_shape complete in %.2fs", float(time.perf_counter() - t0))
        logger.info("infer_log_radial_state...")
        t0 = time.perf_counter()
        radial_state = infer_log_radial_state(
            points,
            corners,
            levels,
            n_axis0_f=int(angular_leaf_shape[0]),
        )
        leaf_shape = (
            int(radial_state["n_axis0_f"]),
            int(angular_leaf_shape[1]),
            int(angular_leaf_shape[2]),
        )
        logger.info("infer_log_radial_state complete in %.2fs", float(time.perf_counter() - t0))
    except ValueError as exc:
        raise ValueError(
            "Could not build a spherical octree from these points and corners. "
            "The geometry does not match the current spherical builder assumptions."
        ) from exc
    return levels, max_level, leaf_shape, {
        "radial_state": radial_state,
        "observed_bounds": (
            cell_log_r_min,
            cell_log_r_max,
            cell_polar_min,
            cell_polar_max,
            azimuth_start,
            azimuth_width,
        ),
    }


def infer_leaf_shape(
    level_shapes: LevelShapeStatsMap,
) -> tuple[int, int, int]:
    """Infer finest spherical leaf shape."""
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
    return n_axis0, n_axis1_f, n_axis2_f


def recover_log_radial_lattice(
    points: np.ndarray,
    corners: np.ndarray,
    cell_levels: np.ndarray,
    *,
    n_axis0_f: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover one integer fine-grid coordinate for each observed radial boundary."""
    logger.info("recover_log_radial_lattice: geometry...")
    t0 = time.perf_counter()
    valid = np.asarray(cell_levels, dtype=np.int64) >= 0
    if not np.any(valid):
        raise ValueError("No valid (>=0) cell levels available to recover the radial lattice.")
    point_r = np.linalg.norm(points, axis=1)
    if np.any(point_r <= 0.0):
        raise ValueError("Spherical builder requires strictly positive radius for log-radial reconstruction.")
    point_log_r = np.log(point_r)
    corners_arr = np.asarray(corners, dtype=np.int64)
    cell_log_r_min = np.min(point_log_r[corners_arr], axis=1)
    cell_log_r_max = np.max(point_log_r[corners_arr], axis=1)
    logger.info("recover_log_radial_lattice: geometry complete in %.2fs", float(time.perf_counter() - t0))

    logger.info("recover_log_radial_lattice: cluster radial edges...")
    t0 = time.perf_counter()
    log_r_min = float(np.min(cell_log_r_min[valid]))
    log_r_max = float(np.max(cell_log_r_max[valid]))
    radial_tol = 1e-7 * max(float(log_r_max - log_r_min), 1.0)
    radial_edges, radial_edge_tol = cluster_close_values(
        np.concatenate((cell_log_r_min[valid], cell_log_r_max[valid])),
        atol=radial_tol,
    )
    if radial_edges.size < 2:
        raise ValueError("Could not infer at least one radial shell from log-r boundaries.")
    logger.info("recover_log_radial_lattice: cluster radial edges complete in %.2fs", float(time.perf_counter() - t0))

    logger.info("recover_log_radial_lattice: assign edge ids...")
    t0 = time.perf_counter()
    valid_ids = np.flatnonzero(valid).astype(np.int64)
    r0_edge_id = nearest_cluster_indices(radial_edges, cell_log_r_min[valid_ids])
    r1_edge_id = nearest_cluster_indices(radial_edges, cell_log_r_max[valid_ids])
    max_level = int(np.max(cell_levels[valid]))
    width_units = np.left_shift(np.ones(valid_ids.size, dtype=np.int64), max_level - cell_levels[valid_ids])
    logger.info("recover_log_radial_lattice: assign edge ids complete in %.2fs", float(time.perf_counter() - t0))

    logger.info("recover_log_radial_lattice: dedupe interval constraints...")
    t0 = time.perf_counter()
    bad_span = r1_edge_id <= r0_edge_id
    if np.any(bad_span):
        bad = int(valid_ids[np.flatnonzero(bad_span)[0]])
        raise ValueError(f"Spherical cell {bad} has non-positive radial span.")
    n_edges = int(radial_edges.size)
    pair_key = r0_edge_id.astype(np.int64) * np.int64(n_edges) + r1_edge_id.astype(np.int64)
    order = np.argsort(pair_key, kind="mergesort")
    sorted_key = pair_key[order]
    sorted_width = width_units[order]
    group_start = np.empty(sorted_key.size, dtype=bool)
    group_start[0] = True
    group_start[1:] = sorted_key[1:] != sorted_key[:-1]
    group_ids = np.flatnonzero(group_start)
    group_min = np.minimum.reduceat(sorted_width, group_ids)
    group_max = np.maximum.reduceat(sorted_width, group_ids)
    inconsistent = group_min != group_max
    if np.any(inconsistent):
        bad_group = int(np.flatnonzero(inconsistent)[0])
        key = int(sorted_key[group_ids[bad_group]])
        start_edge = key // n_edges
        stop_edge = key % n_edges
        raise ValueError(
            "Spherical radial intervals imply conflicting widths for the same recovered edge span. "
            f"edge_span=[{start_edge}, {stop_edge}] "
            f"min_width={int(group_min[bad_group])} max_width={int(group_max[bad_group])}."
        )
    unique_key = sorted_key[group_ids]
    unique_width = group_min.astype(np.int64, copy=False)
    start_edge = (unique_key // n_edges).astype(np.int64, copy=False)
    stop_edge = (unique_key % n_edges).astype(np.int64, copy=False)
    logger.info(
        "recover_log_radial_lattice: dedupe interval constraints complete in %.2fs",
        float(time.perf_counter() - t0),
    )

    logger.info("recover_log_radial_lattice: solve integer difference constraints...")
    t0 = time.perf_counter()
    values = np.full(n_edges, int(n_axis0_f), dtype=np.int64)
    values[0] = 0
    values[-1] = int(n_axis0_f)
    fixed = np.zeros(n_edges, dtype=bool)
    fixed[0] = True
    fixed[-1] = True

    edge_u = np.concatenate((
        stop_edge,
        start_edge,
        np.arange(1, n_edges, dtype=np.int64),
    ))
    edge_v = np.concatenate((
        start_edge,
        stop_edge,
        np.arange(0, n_edges - 1, dtype=np.int64),
    ))
    edge_c = np.concatenate((
        -unique_width,
        unique_width,
        np.zeros(n_edges - 1, dtype=np.int64),
    ))

    n_nodes = n_edges
    for _ in range(max(n_nodes - 1, 0)):
        changed = False
        for u, v, c in zip(edge_u, edge_v, edge_c, strict=True):
            candidate = int(values[int(u)] + c)
            vi = int(v)
            if fixed[vi]:
                if candidate < int(values[vi]):
                    raise ValueError(
                        "Could not recover a self-consistent log-radial lattice from spherical cell boundaries."
                    )
                continue
            if candidate < int(values[vi]):
                values[vi] = candidate
                changed = True
        if not changed:
            break
    for u, v, c in zip(edge_u, edge_v, edge_c, strict=True):
        candidate = int(values[int(u)] + c)
        if candidate < int(values[int(v)]):
            raise ValueError(
                "Could not recover a self-consistent log-radial lattice from spherical cell boundaries."
            )
    logger.info(
        "recover_log_radial_lattice: solve integer difference constraints complete in %.2fs",
        float(time.perf_counter() - t0),
    )

    edge_units = values
    if int(edge_units[-1]) != int(n_axis0_f):
        raise ValueError(
            "Recovered log-radial lattice does not end on the inferred finest radial count: "
            f"got={int(edge_units[-1])}, expected={int(n_axis0_f)}."
        )
    return radial_edges, radial_edge_tol, edge_units, r0_edge_id, r1_edge_id, valid_ids


def infer_log_radial_state(
    points: np.ndarray,
    corners: np.ndarray,
    cell_levels: np.ndarray,
    *,
    n_axis0_f: int,
) -> dict[str, object]:
    """Recover and return reusable log-radial lattice state for one inferred finest count."""
    radial_edges, radial_edge_tol, edge_units, r0_edge_id, r1_edge_id, valid_ids = recover_log_radial_lattice(
        points,
        corners,
        cell_levels,
        n_axis0_f=n_axis0_f,
    )
    return {
        "n_axis0_f": int(n_axis0_f),
        "radial_edges": radial_edges,
        "radial_edge_tol": radial_edge_tol,
        "edge_units": edge_units,
        "r0_edge_id": r0_edge_id,
        "r1_edge_id": r1_edge_id,
        "valid_ids": valid_ids,
    }


def snap_polar_bounds(
    cell_polar_min: np.ndarray,
    cell_polar_max: np.ndarray,
    *,
    d_polar_f: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Snap observed polar bounds to the nearest fine-grid lines with progressive tolerance."""
    i0 = np.rint(cell_polar_min / d_polar_f).astype(np.int64)
    i1 = np.rint(cell_polar_max / d_polar_f).astype(np.int64)
    base_tol = max(1e-7 * np.pi, 2e-5 * d_polar_f)
    max_tol = 0.49 * d_polar_f
    tol = float(base_tol)
    while tol <= max_tol:
        min_ok = np.isclose(cell_polar_min, i0 * d_polar_f, rtol=0.0, atol=tol)
        max_ok = np.isclose(cell_polar_max, i1 * d_polar_f, rtol=0.0, atol=tol)
        if np.all(min_ok) and np.all(max_ok):
            return i0, i1, tol
        tol *= 2.0
    raise ValueError("Could not snap polar bounds onto the inferred spherical grid.")


def validate_one_level_neighbors(
    *,
    leaf_shape: tuple[int, int, int],
    cell_levels: np.ndarray,
    valid_ids: np.ndarray,
    r0_f: np.ndarray,
    i1_f: np.ndarray,
    i2_f: np.ndarray,
    width_units: np.ndarray,
) -> None:
    """Require face-neighboring spherical cells to differ by at most one level."""
    owner = np.full(tuple(int(v) for v in leaf_shape), -1, dtype=np.int64)
    for row, cell_id in enumerate(valid_ids):
        width = int(width_units[row])
        r0 = int(r0_f[row])
        p0 = int(i1_f[row])
        a0 = int(i2_f[row])
        owner[r0:r0 + width, p0:p0 + width, a0:a0 + width] = int(cell_id)

    def check_axis(axis: int, *, periodic: bool = False) -> tuple[int, int] | None:
        slicer0 = [slice(None), slice(None), slice(None)]
        slicer1 = [slice(None), slice(None), slice(None)]
        slicer0[axis] = slice(0, owner.shape[axis] - 1)
        slicer1[axis] = slice(1, owner.shape[axis])
        left = owner[tuple(slicer0)]
        right = owner[tuple(slicer1)]
        face_mask = (left >= 0) & (right >= 0) & (left != right)
        if periodic:
            wrap_left = np.take(owner, owner.shape[axis] - 1, axis=axis)
            wrap_right = np.take(owner, 0, axis=axis)
            wrap_mask = (wrap_left >= 0) & (wrap_right >= 0) & (wrap_left != wrap_right)
            if np.any(wrap_mask):
                left = np.concatenate((left[face_mask], wrap_left[wrap_mask]))
                right = np.concatenate((right[face_mask], wrap_right[wrap_mask]))
            else:
                left = left[face_mask]
                right = right[face_mask]
        else:
            left = left[face_mask]
            right = right[face_mask]
        if left.size == 0:
            return None
        level_diff = np.abs(cell_levels[left] - cell_levels[right])
        if np.any(level_diff > 1):
            bad = int(np.flatnonzero(level_diff > 1)[0])
            return int(left[bad]), int(right[bad])
        return None

    for axis, periodic in ((0, False), (1, False), (2, True)):
        bad_pair = check_axis(axis, periodic=periodic)
        if bad_pair is not None:
            left_id, right_id = bad_pair
            raise ValueError(
                "Spherical neighboring cells differ by more than one refinement level: "
                f"cells {left_id} and {right_id}."
            )


def _observed_spherical_bounds(
    points: np.ndarray,
    corners: np.ndarray,
    *,
    axis_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Measure per-cell spherical bounds from explicit corner geometry."""
    corners_arr = np.asarray(corners, dtype=np.int64)
    if corners_arr.ndim != 2:
        raise ValueError(f"Expected 2D corner array, got shape {corners_arr.shape}.")
    if corners_arr.shape[1] < 3:
        raise ValueError("Need at least 3 corners per cell to estimate spherical bounds.")
    points_r = np.linalg.norm(points, axis=1)
    if np.any(points_r <= 0.0):
        raise ValueError("Spherical tree state requires strictly positive radius for log-radial reconstruction.")

    points_log_r = np.log(points_r)
    cell_log_r_min = np.min(points_log_r[corners_arr], axis=1)
    cell_log_r_max = np.max(points_log_r[corners_arr], axis=1)

    polar_points = np.arccos(
        np.clip(points[:, 2] / np.maximum(points_r, np.finfo(float).tiny), -1.0, 1.0)
    )
    cell_polar_min = np.min(polar_points[corners_arr], axis=1)
    cell_polar_max = np.max(polar_points[corners_arr], axis=1)

    azimuth_points = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
    axis_mask = np.hypot(points[:, 0], points[:, 1])[corners_arr] <= axis_tol
    azimuth_start, azimuth_width = minimal_azimuth_intervals(
        azimuth_points[corners_arr],
        ignore_mask=axis_mask,
    )
    return cell_log_r_min, cell_log_r_max, cell_polar_min, cell_polar_max, azimuth_start, azimuth_width


def _format_observed_rpa_limits(
    *,
    cell_id: int,
    cell_log_r_min: np.ndarray,
    cell_log_r_max: np.ndarray,
    cell_polar_min: np.ndarray,
    cell_polar_max: np.ndarray,
    azimuth_start: np.ndarray,
    azimuth_width: np.ndarray,
) -> str:
    """Return a compact description of one cell's observed spherical bounds."""
    r_min = float(np.exp(cell_log_r_min[cell_id]))
    r_max = float(np.exp(cell_log_r_max[cell_id]))
    polar_min = float(cell_polar_min[cell_id])
    polar_max = float(cell_polar_max[cell_id])
    azimuth0 = float(azimuth_start[cell_id])
    azimuth_w = float(azimuth_width[cell_id])
    azimuth1 = azimuth0 + azimuth_w
    return (
        "observed_rpa="
        f"(r=[{r_min:.17g}, {r_max:.17g}], "
        f"polar=[{polar_min:.17g}, {polar_max:.17g}], "
        f"azimuth_start={azimuth0:.17g}, "
        f"azimuth_width={azimuth_w:.17g}, "
        f"azimuth_stop_unwrapped={azimuth1:.17g})"
    )


def _log_cellwise_mismatch(
    *,
    summary_label: str,
    action: str,
    mask: np.ndarray,
    valid_ids: np.ndarray,
    observed: np.ndarray,
    snapped: np.ndarray,
    tol: np.ndarray | float,
    topology_margin: np.ndarray | float,
    detail_builder,
) -> None:
    """Log one warning summary and a few debug examples for cellwise residual mismatches."""
    bad_idx = np.flatnonzero(mask)
    if bad_idx.size == 0:
        return
    residual = np.abs(observed[bad_idx] - snapped[bad_idx])
    tol_arr = np.broadcast_to(np.asarray(tol, dtype=float), observed.shape)
    bad_tol = tol_arr[bad_idx]
    topology_arr = np.broadcast_to(np.asarray(topology_margin, dtype=float), observed.shape)
    bad_topology = topology_arr[bad_idx]
    max_local = int(np.argmax(residual))
    worst_idx = int(bad_idx[max_local])
    worst_cell = int(valid_ids[worst_idx])
    worst_residual = float(residual[max_local])
    worst_tol = float(bad_tol[max_local])
    worst_ratio = worst_residual / worst_tol if worst_tol > 0.0 else float("inf")
    worst_topology = float(bad_topology[max_local])
    worst_topology_ratio = worst_residual / worst_topology if worst_topology > 0.0 else float("inf")
    summary = (
        "%s for %d cells; %s. max_abs_residual=%.17g max_tol=%.17g "
        "max_residual_over_tol=%.6g max_topology_margin=%.17g "
        "max_residual_over_topology_margin=%.6g worst_cell=%d"
    )
    log_fn = logger.warning if worst_topology_ratio >= SNAP_TOPOLOGY_WARNING_FRACTION else logger.debug
    log_fn(
        summary,
        summary_label,
        int(bad_idx.size),
        action,
        worst_residual,
        worst_tol,
        worst_ratio,
        worst_topology,
        worst_topology_ratio,
        worst_cell,
    )
    for idx in bad_idx[:3]:
        cell_id = int(valid_ids[int(idx)])
        logger.debug("%s cell %d: %s", summary_label, cell_id, detail_builder(int(idx), cell_id))


def _cluster_switch_margin(centers: np.ndarray, edge_ids: np.ndarray) -> np.ndarray:
    """Return distance from each assigned center to the nearest address-switch midpoint."""
    centers_arr = np.asarray(centers, dtype=float)
    ids = np.asarray(edge_ids, dtype=np.int64)
    out = np.full(ids.shape, np.inf, dtype=float)
    if centers_arr.size < 2:
        return out
    left_gap = np.full(centers_arr.shape, np.inf, dtype=float)
    right_gap = np.full(centers_arr.shape, np.inf, dtype=float)
    left_gap[1:] = centers_arr[1:] - centers_arr[:-1]
    right_gap[:-1] = centers_arr[1:] - centers_arr[:-1]
    out = 0.5 * np.minimum(left_gap[ids], right_gap[ids])
    return out


def _recover_radial_addresses(
    *,
    leaf_shape: tuple[int, int, int],
    cell_log_r_min: np.ndarray,
    cell_log_r_max: np.ndarray,
    cell_polar_min: np.ndarray,
    cell_polar_max: np.ndarray,
    azimuth_start: np.ndarray,
    azimuth_width: np.ndarray,
    radial_state: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map observed spherical radial bounds onto the inferred fine radial grid."""
    radial_edges = np.asarray(radial_state["radial_edges"], dtype=float)
    radial_edge_tol = np.asarray(radial_state["radial_edge_tol"], dtype=float)
    radial_edge_units = np.asarray(radial_state["edge_units"], dtype=np.int64)
    r0_edge_id = np.asarray(radial_state["r0_edge_id"], dtype=np.int64)
    r1_edge_id = np.asarray(radial_state["r1_edge_id"], dtype=np.int64)
    valid_ids = np.asarray(radial_state["valid_ids"], dtype=np.int64)
    r0_f = radial_edge_units[r0_edge_id]
    r1_f = radial_edge_units[r1_edge_id]
    n_r_fine = int(leaf_shape[0])
    if np.any(r0_f < 0) or np.any(r1_f > n_r_fine):
        bad_idx = int(np.flatnonzero((r0_f < 0) | (r1_f > n_r_fine))[0])
        bad = int(valid_ids[bad_idx])
        raise ValueError(
            f"Spherical cell {bad} radial address is outside the inferred octree grid. "
            f"r_addr=[{int(r0_f[bad_idx])}, {int(r1_f[bad_idx])}] n_r_fine={n_r_fine} "
            + _format_observed_rpa_limits(
                cell_id=bad,
                cell_log_r_min=cell_log_r_min,
                cell_log_r_max=cell_log_r_max,
                cell_polar_min=cell_polar_min,
                cell_polar_max=cell_polar_max,
                azimuth_start=azimuth_start,
                azimuth_width=azimuth_width,
            )
        )

    r0_ok = np.abs(radial_edges[r0_edge_id] - cell_log_r_min[valid_ids]) <= radial_edge_tol[r0_edge_id]
    r1_ok = np.abs(radial_edges[r1_edge_id] - cell_log_r_max[valid_ids]) <= radial_edge_tol[r1_edge_id]
    if np.any(~r0_ok):
        _log_cellwise_mismatch(
            summary_label="Spherical r-min snap mismatch",
            action="keeping nearest recovered radial edges",
            mask=~r0_ok,
            valid_ids=valid_ids,
            observed=cell_log_r_min[valid_ids],
            snapped=radial_edges[r0_edge_id],
            tol=radial_edge_tol[r0_edge_id],
            topology_margin=_cluster_switch_margin(radial_edges, r0_edge_id),
            detail_builder=lambda idx, cell_id: (
                f"observed_log_r_min={float(cell_log_r_min[cell_id]):.17g} "
                f"snapped_log_r_min={float(radial_edges[r0_edge_id[idx]]):.17g} "
                f"tol={float(radial_edge_tol[r0_edge_id[idx]]):.17g} "
                + _format_observed_rpa_limits(
                    cell_id=cell_id,
                    cell_log_r_min=cell_log_r_min,
                    cell_log_r_max=cell_log_r_max,
                    cell_polar_min=cell_polar_min,
                    cell_polar_max=cell_polar_max,
                    azimuth_start=azimuth_start,
                    azimuth_width=azimuth_width,
                )
            ),
        )
    if np.any(~r1_ok):
        _log_cellwise_mismatch(
            summary_label="Spherical r-max snap mismatch",
            action="keeping nearest recovered radial edges",
            mask=~r1_ok,
            valid_ids=valid_ids,
            observed=cell_log_r_max[valid_ids],
            snapped=radial_edges[r1_edge_id],
            tol=radial_edge_tol[r1_edge_id],
            topology_margin=_cluster_switch_margin(radial_edges, r1_edge_id),
            detail_builder=lambda idx, cell_id: (
                f"observed_log_r_max={float(cell_log_r_max[cell_id]):.17g} "
                f"snapped_log_r_max={float(radial_edges[r1_edge_id[idx]]):.17g} "
                f"tol={float(radial_edge_tol[r1_edge_id[idx]]):.17g} "
                + _format_observed_rpa_limits(
                    cell_id=cell_id,
                    cell_log_r_min=cell_log_r_min,
                    cell_log_r_max=cell_log_r_max,
                    cell_polar_min=cell_polar_min,
                    cell_polar_max=cell_polar_max,
                    azimuth_start=azimuth_start,
                    azimuth_width=azimuth_width,
                )
            ),
        )
    return valid_ids, r0_f, r1_f


def _snap_angular_addresses(
    *,
    leaf_shape: tuple[int, int, int],
    valid_ids: np.ndarray,
    cell_log_r_min: np.ndarray,
    cell_log_r_max: np.ndarray,
    cell_polar_min: np.ndarray,
    cell_polar_max: np.ndarray,
    azimuth_start: np.ndarray,
    azimuth_width: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Snap observed polar/azimuth bounds onto the inferred fine angular grid."""
    d_polar_f = np.pi / float(int(leaf_shape[1]))
    d_azimuth_f = (2.0 * np.pi) / float(int(leaf_shape[2]))
    width_azimuth = np.asarray(azimuth_width[valid_ids], dtype=float)
    azimuth_tol = max(1e-7 * 2.0 * np.pi, 2e-5 * d_azimuth_f)
    if np.any(width_azimuth >= (2.0 * np.pi - azimuth_tol)):
        bad = int(valid_ids[np.flatnonzero(width_azimuth >= (2.0 * np.pi - azimuth_tol))[0]])
        raise ValueError(
            f"Spherical cell {bad} spans the full azimuth and has no unique octree address. "
            + _format_observed_rpa_limits(
                cell_id=bad,
                cell_log_r_min=cell_log_r_min,
                cell_log_r_max=cell_log_r_max,
                cell_polar_min=cell_polar_min,
                cell_polar_max=cell_polar_max,
                azimuth_start=azimuth_start,
                azimuth_width=azimuth_width,
            )
        )

    try:
        i1_f, i1_hi, polar_tol = snap_polar_bounds(
            cell_polar_min[valid_ids],
            cell_polar_max[valid_ids],
            d_polar_f=d_polar_f,
        )
    except ValueError as exc:
        raise ValueError(
            "Could not build a spherical octree from these points and corners. "
            "The geometry does not match the current spherical builder assumptions."
        ) from exc

    i2_f = np.rint(azimuth_start[valid_ids] / d_azimuth_f).astype(np.int64)
    i2_hi = np.rint((azimuth_start[valid_ids] + width_azimuth) / d_azimuth_f).astype(np.int64)
    n_polar_fine = int(leaf_shape[1])
    n_azimuth_fine = int(leaf_shape[2])
    in_bounds = (
        (i1_f >= 0) & (i1_hi <= n_polar_fine)
        & (i2_f >= 0) & (i2_hi <= n_azimuth_fine)
    )
    if not np.all(in_bounds):
        bad = int(valid_ids[np.flatnonzero(~in_bounds)[0]])
        raise ValueError(
            f"Spherical cell {bad} address is outside the inferred octree grid. "
            + _format_observed_rpa_limits(
                cell_id=bad,
                cell_log_r_min=cell_log_r_min,
                cell_log_r_max=cell_log_r_max,
                cell_polar_min=cell_polar_min,
                cell_polar_max=cell_polar_max,
                azimuth_start=azimuth_start,
                azimuth_width=azimuth_width,
            )
        )

    polar_min_ok = np.isclose(cell_polar_min[valid_ids], i1_f * d_polar_f, rtol=0.0, atol=polar_tol)
    polar_max_ok = np.isclose(cell_polar_max[valid_ids], i1_hi * d_polar_f, rtol=0.0, atol=polar_tol)
    azimuth_delta = np.abs((azimuth_start[valid_ids] - (i2_f * d_azimuth_f) + np.pi) % (2.0 * np.pi) - np.pi)
    azimuth_start_ok = azimuth_delta <= azimuth_tol
    azimuth_width_ok = np.isclose(width_azimuth, (i2_hi - i2_f) * d_azimuth_f, rtol=0.0, atol=azimuth_tol)

    if np.any(~polar_min_ok):
        _log_cellwise_mismatch(
            summary_label="Spherical polar-min snap mismatch",
            action="keeping nearest snapped polar bounds",
            mask=~polar_min_ok,
            valid_ids=valid_ids,
            observed=cell_polar_min[valid_ids],
            snapped=i1_f * d_polar_f,
            tol=polar_tol,
            topology_margin=0.5 * d_polar_f,
            detail_builder=lambda idx, cell_id: (
                f"observed_polar_min={float(cell_polar_min[cell_id]):.17g} "
                f"snapped_polar_min={float(i1_f[idx] * d_polar_f):.17g} "
                f"tol={float(polar_tol):.17g} "
                + _format_observed_rpa_limits(
                    cell_id=cell_id,
                    cell_log_r_min=cell_log_r_min,
                    cell_log_r_max=cell_log_r_max,
                    cell_polar_min=cell_polar_min,
                    cell_polar_max=cell_polar_max,
                    azimuth_start=azimuth_start,
                    azimuth_width=azimuth_width,
                )
            ),
        )
    if np.any(~polar_max_ok):
        _log_cellwise_mismatch(
            summary_label="Spherical polar-max snap mismatch",
            action="keeping nearest snapped polar bounds",
            mask=~polar_max_ok,
            valid_ids=valid_ids,
            observed=cell_polar_max[valid_ids],
            snapped=i1_hi * d_polar_f,
            tol=polar_tol,
            topology_margin=0.5 * d_polar_f,
            detail_builder=lambda idx, cell_id: (
                f"observed_polar_max={float(cell_polar_max[cell_id]):.17g} "
                f"snapped_polar_max={float(i1_hi[idx] * d_polar_f):.17g} "
                f"tol={float(polar_tol):.17g} "
                + _format_observed_rpa_limits(
                    cell_id=cell_id,
                    cell_log_r_min=cell_log_r_min,
                    cell_log_r_max=cell_log_r_max,
                    cell_polar_min=cell_polar_min,
                    cell_polar_max=cell_polar_max,
                    azimuth_start=azimuth_start,
                    azimuth_width=azimuth_width,
                )
            ),
        )
    if np.any(~azimuth_start_ok):
        _log_cellwise_mismatch(
            summary_label="Spherical azimuth-start snap mismatch",
            action="keeping nearest snapped azimuth starts",
            mask=~azimuth_start_ok,
            valid_ids=valid_ids,
            observed=azimuth_start[valid_ids],
            snapped=i2_f * d_azimuth_f,
            tol=azimuth_tol,
            topology_margin=0.5 * d_azimuth_f,
            detail_builder=lambda idx, cell_id: (
                f"observed_azimuth_start={float(azimuth_start[cell_id]):.17g} "
                f"snapped_azimuth_start={float(i2_f[idx] * d_azimuth_f):.17g} "
                f"tol={float(azimuth_tol):.17g} "
                + _format_observed_rpa_limits(
                    cell_id=cell_id,
                    cell_log_r_min=cell_log_r_min,
                    cell_log_r_max=cell_log_r_max,
                    cell_polar_min=cell_polar_min,
                    cell_polar_max=cell_polar_max,
                    azimuth_start=azimuth_start,
                    azimuth_width=azimuth_width,
                )
            ),
        )
    if np.any(~azimuth_width_ok):
        _log_cellwise_mismatch(
            summary_label="Spherical azimuth-width snap mismatch",
            action="keeping nearest snapped azimuth widths",
            mask=~azimuth_width_ok,
            valid_ids=valid_ids,
            observed=width_azimuth,
            snapped=(i2_hi - i2_f) * d_azimuth_f,
            tol=azimuth_tol,
            topology_margin=0.5 * d_azimuth_f,
            detail_builder=lambda idx, cell_id: (
                f"observed_azimuth_width={float(width_azimuth[idx]):.17g} "
                f"snapped_azimuth_width={float((i2_hi[idx] - i2_f[idx]) * d_azimuth_f):.17g} "
                f"tol={float(azimuth_tol):.17g} "
                + _format_observed_rpa_limits(
                    cell_id=cell_id,
                    cell_log_r_min=cell_log_r_min,
                    cell_log_r_max=cell_log_r_max,
                    cell_polar_min=cell_polar_min,
                    cell_polar_max=cell_polar_max,
                    azimuth_start=azimuth_start,
                    azimuth_width=azimuth_width,
                )
            ),
        )
    return i1_f, i1_hi, i2_f, i2_hi


def populate_tree_state(
    *,
    leaf_shape: tuple[int, int, int],
    max_level: int,
    cell_levels: np.ndarray | None,
    inferred_state: dict[str, object],
) -> dict[str, object]:
    """Return exact spherical octree state for one built tree."""
    if cell_levels is None:
        raise ValueError("Spherical tree state requires cell_levels.")
    levels = np.asarray(cell_levels, dtype=np.int64)
    valid = levels >= 0
    if not np.any(valid):
        raise ValueError("Spherical tree state requires at least one valid cell level.")

    logger.info("_observed_spherical_bounds...")
    t0 = time.perf_counter()
    (
        cell_log_r_min,
        cell_log_r_max,
        cell_polar_min,
        cell_polar_max,
        azimuth_start,
        azimuth_width,
    ) = inferred_state["observed_bounds"]
    logger.info("_observed_spherical_bounds complete in %.2fs", float(time.perf_counter() - t0))

    tree_depth = int(max_level)
    logger.info("_recover_radial_addresses...")
    t0 = time.perf_counter()
    valid_ids, r0_f, r1_f = _recover_radial_addresses(
        leaf_shape=leaf_shape,
        cell_log_r_min=cell_log_r_min,
        cell_log_r_max=cell_log_r_max,
        cell_polar_min=cell_polar_min,
        cell_polar_max=cell_polar_max,
        azimuth_start=azimuth_start,
        azimuth_width=azimuth_width,
        radial_state=inferred_state["radial_state"],
    )
    logger.info("_recover_radial_addresses complete in %.2fs", float(time.perf_counter() - t0))
    depths = levels[valid_ids]
    shifts = np.asarray(tree_depth - depths, dtype=np.int64)
    if np.any(shifts < 0):
        bad = int(valid_ids[np.flatnonzero(shifts < 0)[0]])
        raise ValueError(f"Spherical cell {bad} depth exceeds tree_depth={tree_depth}.")
    width_units = np.left_shift(np.ones_like(shifts, dtype=np.int64), shifts)

    logger.info("_snap_angular_addresses...")
    t0 = time.perf_counter()
    i1_f, i1_hi, i2_f, i2_hi = _snap_angular_addresses(
        leaf_shape=leaf_shape,
        valid_ids=valid_ids,
        cell_log_r_min=cell_log_r_min,
        cell_log_r_max=cell_log_r_max,
        cell_polar_min=cell_polar_min,
        cell_polar_max=cell_polar_max,
        azimuth_start=azimuth_start,
        azimuth_width=azimuth_width,
    )
    logger.info("_snap_angular_addresses complete in %.2fs", float(time.perf_counter() - t0))

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

    cell_ijk = np.full((levels.shape[0], 3), -1, dtype=np.int64)
    i0 = np.right_shift(r0_f, shifts)
    i1 = np.right_shift(i1_f, shifts)
    i2 = np.right_shift(i2_f, shifts)
    cell_ijk[valid_ids, 0] = i0
    cell_ijk[valid_ids, 1] = i1
    cell_ijk[valid_ids, 2] = i2
    logger.info("validate_one_level_neighbors...")
    t0 = time.perf_counter()
    validate_one_level_neighbors(
        leaf_shape=leaf_shape,
        cell_levels=levels,
        valid_ids=valid_ids,
        r0_f=r0_f,
        i1_f=i1_f,
        i2_f=i2_f,
        width_units=width_units,
    )
    logger.info("validate_one_level_neighbors complete in %.2fs", float(time.perf_counter() - t0))

    return {
        "cell_levels": np.asarray(cell_levels, dtype=np.int64),
        "cell_ijk": np.asarray(cell_ijk, dtype=np.int64),
    }
