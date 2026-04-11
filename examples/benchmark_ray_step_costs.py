#!/usr/bin/env python3
"""Profile one warmed ray trace by per-cell step category."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from batread.dataset import Dataset

from batcamp import OctreeRayTracer
from batcamp.constants import XYZ_VARS
from batcamp.ray import _cell_segment_trace_kernel
from batcamp.ray import _launch_leaf_kernel
from batcamp.ray import _lookup_xyz_leaf_kernel
from batcamp.ray import _resolve_boundary_owner_leaf_kernel

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_helpers import _build_octree
    from benchmark_helpers import _configure_builder_logging
    from benchmark_helpers import _configure_progress_logging
    from benchmark_helpers import _ProgressReporter
    from benchmark_helpers import _time_call
    from benchmark_helpers import resolve_data_file
else:
    from .benchmark_helpers import _build_octree
    from .benchmark_helpers import _configure_builder_logging
    from .benchmark_helpers import _configure_progress_logging
    from .benchmark_helpers import _ProgressReporter
    from .benchmark_helpers import _time_call
    from .benchmark_helpers import resolve_data_file


def _plane_axis_points(lo: float, hi: float, n: int) -> np.ndarray:
    """Return one regular image-axis coordinate array at pixel centers."""
    step = (float(hi) - float(lo)) / float(n)
    return float(lo) + (np.arange(int(n), dtype=float) + 0.5) * step


def _ray_setup(
    *,
    n_plane: int,
    bounds: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build one `(z, y)` batch of parallel +x rays."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    y = _plane_axis_points(ymin, ymax, int(n_plane))
    z = _plane_axis_points(zmin, zmax, int(n_plane))
    yg, zg = np.meshgrid(y, z, indexing="xy")
    xg = np.full_like(yg, float(xmin), dtype=float)
    origins = np.stack((xg, yg, zg), axis=-1)
    directions = np.zeros_like(origins)
    directions[..., 0] = 1.0
    return origins, directions, float(xmax - xmin)


def _profile_trace_branch(
    start_leaf_id: int,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    t_min: float,
    t_max: float,
    profile: dict[str, float | int],
    leaf_valid: np.ndarray,
    face_xyz: np.ndarray,
    face_normal: np.ndarray,
    face_plane_d: np.ndarray,
    face_edge_normal: np.ndarray,
    face_edge_d: np.ndarray,
    leaf_scales: np.ndarray,
    face_patch_center: np.ndarray,
    next_cell: np.ndarray,
    tree_coord_code: int,
    cell_child: np.ndarray,
    root_cell_ids: np.ndarray,
    cell_parent: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
    n_valid_leaf: int,
) -> None:
    """Accumulate one branch cost breakdown with the live step logic."""
    current_leaf = int(start_leaf_id)
    current_t = float(t_min)
    max_steps = int(n_valid_leaf) + 1

    for _ in range(max_steps):
        if current_t >= float(t_max):
            break

        profile["steps"] += 1

        t0 = time.perf_counter()
        ok, segment_enter, exit_t, _, active_face_mask = _cell_segment_trace_kernel(
            face_xyz[current_leaf],
            face_normal[current_leaf],
            face_plane_d[current_leaf],
            face_edge_normal[current_leaf],
            face_edge_d[current_leaf],
            leaf_scales[current_leaf],
            origin_xyz,
            direction_xyz,
            current_t,
            t_min,
        )
        profile["cell_segment_s"] += float(time.perf_counter() - t0)

        if not ok or segment_enter >= float(t_max) or exit_t >= float(t_max):
            break

        exit_xyz = origin_xyz + exit_t * direction_xyz
        profile["ownership_calls"] += 1
        t0 = time.perf_counter()
        next_leaf_id = _resolve_boundary_owner_leaf_kernel(
            current_leaf,
            exit_xyz,
            direction_xyz,
            face_normal,
            face_patch_center,
            int(active_face_mask),
            leaf_valid,
            next_cell,
            leaf_scales,
            tree_coord_code,
            cell_child,
            root_cell_ids,
            cell_parent,
            cell_bounds,
            domain_bounds,
            axis2_period,
            axis2_periodic,
        )
        if next_leaf_id >= 0 and next_leaf_id != current_leaf:
            profile["ownership_success"] += 1
        profile["ownership_s"] += float(time.perf_counter() - t0)

        if next_leaf_id < 0 or next_leaf_id == current_leaf:
            break

        current_leaf = int(next_leaf_id)
        current_t = float(exit_t)


def _ray_step_cost_report(
    tracer: OctreeRayTracer,
    *,
    n_plane: int,
    bounds: tuple[float, float, float, float, float, float],
) -> dict[str, float | int]:
    """Return one per-cell step cost report for one warmed ray plane."""
    origins, directions, t_end = _ray_setup(n_plane=int(n_plane), bounds=bounds)
    tracer.trace(origins, directions, t_min=0.0, t_max=float(t_end))

    seed_xyz = tracer.seed_domain(origins, directions, t_min=0.0, t_max=float(t_end)).reshape(-1, 3)
    origins_flat = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    directions_flat = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    (
        leaf_valid,
        face_xyz,
        face_normal,
        face_plane_d,
        face_edge_normal,
        face_edge_d,
        leaf_scales,
        face_patch_center,
        next_cell,
        n_polar,
        n_azimuth,
        tree_coord_code,
        cell_child,
        root_cell_ids,
        cell_parent,
        cell_bounds,
        domain_bounds,
        axis2_period,
        axis2_periodic,
        n_valid_leaf,
    ) = tracer.trace_kernel_state()

    profile: dict[str, float | int] = {
        "sample_resolution": int(n_plane),
        "sample_rays": int(origins_flat.shape[0]),
        "steps": 0,
        "ownership_calls": 0,
        "ownership_success": 0,
        "cell_segment_s": 0.0,
        "ownership_s": 0.0,
    }

    for ray_id in range(int(origins_flat.shape[0])):
        seed = seed_xyz[ray_id]
        if not np.all(np.isfinite(seed)):
            continue

        seed_leaf_id = _lookup_xyz_leaf_kernel(
            seed,
            -1,
            tree_coord_code,
            cell_child,
            root_cell_ids,
            cell_parent,
            cell_bounds,
            domain_bounds,
            axis2_period,
            axis2_periodic,
        )
        if seed_leaf_id < 0:
            continue

        origin_xyz = origins_flat[ray_id]
        direction_xyz = directions_flat[ray_id]
        direction_norm_sq = float(np.dot(direction_xyz, direction_xyz))
        t_seed = float(np.dot(seed - origin_xyz, direction_xyz) / direction_norm_sq)

        backward_leaf = _launch_leaf_kernel(
            int(seed_leaf_id),
            seed,
            -direction_xyz,
            leaf_valid,
            face_normal,
            face_patch_center,
            face_xyz,
            face_plane_d,
            face_edge_normal,
            face_edge_d,
            leaf_scales,
            next_cell,
            n_polar,
            n_azimuth,
            tree_coord_code,
            cell_child,
            root_cell_ids,
            cell_parent,
            cell_bounds,
            domain_bounds,
            axis2_period,
            axis2_periodic,
        )
        forward_leaf = _launch_leaf_kernel(
            int(seed_leaf_id),
            seed,
            direction_xyz,
            leaf_valid,
            face_normal,
            face_patch_center,
            face_xyz,
            face_plane_d,
            face_edge_normal,
            face_edge_d,
            leaf_scales,
            next_cell,
            n_polar,
            n_azimuth,
            tree_coord_code,
            cell_child,
            root_cell_ids,
            cell_parent,
            cell_bounds,
            domain_bounds,
            axis2_period,
            axis2_periodic,
        )

        backward_limit = max(0.0, t_seed)
        if backward_limit > 0.0 and backward_leaf >= 0:
            _profile_trace_branch(
                int(backward_leaf),
                seed,
                -direction_xyz,
                0.0,
                float(backward_limit),
                profile,
                leaf_valid,
                face_xyz,
                face_normal,
                face_plane_d,
                face_edge_normal,
                face_edge_d,
                leaf_scales,
                face_patch_center,
                next_cell,
                tree_coord_code,
                cell_child,
                root_cell_ids,
                cell_parent,
                cell_bounds,
                domain_bounds,
                axis2_period,
                axis2_periodic,
                n_valid_leaf,
            )

        forward_limit = max(0.0, float(t_end) - t_seed)
        if forward_limit > 0.0 and forward_leaf >= 0:
            _profile_trace_branch(
                int(forward_leaf),
                seed,
                direction_xyz,
                0.0,
                float(forward_limit),
                profile,
                leaf_valid,
                face_xyz,
                face_normal,
                face_plane_d,
                face_edge_normal,
                face_edge_d,
                leaf_scales,
                face_patch_center,
                next_cell,
                tree_coord_code,
                cell_child,
                root_cell_ids,
                cell_parent,
                cell_bounds,
                domain_bounds,
                axis2_period,
                axis2_periodic,
                n_valid_leaf,
            )

    total_step_s = (
        float(profile["cell_segment_s"])
        + float(profile["ownership_s"])
    )
    profile["total_profiled_s"] = total_step_s
    profile["cell_segment_frac"] = float(profile["cell_segment_s"]) / total_step_s if total_step_s > 0.0 else np.nan
    profile["ownership_frac"] = float(profile["ownership_s"]) / total_step_s if total_step_s > 0.0 else np.nan
    return profile


def _write_report(report: dict[str, float | int], out_path: Path) -> None:
    """Write one markdown report with the aggregated step-cost breakdown."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ray Step Cost Report",
        "",
        f"- sampled ray plane: `{int(report['sample_resolution'])}x{int(report['sample_resolution'])}`",
        f"- profiled rays: `{int(report['sample_rays'])}`",
        f"- profiled cell steps: `{int(report['steps'])}`",
        f"- ownership success: `{int(report['ownership_success'])}/{int(report['ownership_calls'])}`",
        "",
        "| step piece | share | seconds | calls |",
        "|---|---:|---:|---:|",
        (
            f"| current cell segment solve | {100.0 * float(report['cell_segment_frac']):.1f}% | "
            f"{float(report['cell_segment_s']):.6f} | {int(report['steps'])} |"
        ),
        (
            f"| ownership continuation | {100.0 * float(report['ownership_frac']):.1f}% | "
            f"{float(report['ownership_s']):.6f} | {int(report['ownership_calls'])} |"
        ),
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile one warmed ray trace by per-cell step category.")
    parser.add_argument(
        "--dataset",
        default="3d__var_1_n00000000.plt",
        help="Dataset basename to resolve from sample_data or pooch.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=31,
        help="Square ray-plane resolution used for the step profile.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Output directory for the markdown report and progress log.",
    )
    args = parser.parse_args()

    if int(args.resolution) <= 0:
        raise ValueError("resolution must be positive.")

    repo_root = Path(__file__).resolve().parent.parent
    out_root = (repo_root / str(args.output_dir)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    progress_log_path = out_root / "benchmark_ray_step_costs.log"
    progress_log_path.write_text("", encoding="utf-8")
    _configure_progress_logging(log_path=progress_log_path)
    _configure_builder_logging(log_path=progress_log_path)
    progress = _ProgressReporter(log_path=progress_log_path)

    progress.note(f"output_dir={out_root}")
    progress.start("resolve data file")
    data_path, resolve_s = _time_call(resolve_data_file, repo_root, str(args.dataset))
    progress.complete("resolve data file", resolve_s, detail=f"-> {data_path}")
    progress.start("read dataset")
    ds, read_s = _time_call(Dataset.from_file, str(data_path))
    progress.complete("read dataset", read_s)
    progress.start("prepare octree")
    tree, tree_s = _time_call(_build_octree, ds)
    progress.complete("prepare octree", tree_s, detail=f"coord={tree.tree_coord}")
    progress.start("build ray tracer")
    tracer, tracer_s = _time_call(OctreeRayTracer, tree)
    progress.complete("build ray tracer", tracer_s)

    xyz = np.column_stack(tuple(np.asarray(ds[name], dtype=float) for name in XYZ_VARS))
    dmin = np.min(xyz, axis=0)
    dmax = np.max(xyz, axis=0)
    bounds = (
        float(dmin[0]),
        float(dmax[0]),
        float(dmin[1]),
        float(dmax[1]),
        float(dmin[2]),
        float(dmax[2]),
    )

    progress.start(f"profile ray step costs at {int(args.resolution)}x{int(args.resolution)}")
    report, profile_s = _time_call(
        _ray_step_cost_report,
        tracer,
        n_plane=int(args.resolution),
        bounds=bounds,
    )
    progress.complete(
        f"profile ray step costs at {int(args.resolution)}x{int(args.resolution)}",
        profile_s,
        detail=(
            f"segment={100.0 * float(report['cell_segment_frac']):.1f}% "
            f"ownership={100.0 * float(report['ownership_frac']):.1f}%"
        ),
    )
    progress.note(
        f"steps={int(report['steps'])} ownership={int(report['ownership_success'])}/{int(report['ownership_calls'])}"
    )

    out_path = out_root / f"benchmark_ray_step_costs_{Path(str(args.dataset)).stem}_{int(args.resolution)}x{int(args.resolution)}.md"
    _write_report(report, out_path)
    progress.note(f"report={out_path}")


if __name__ == "__main__":
    main()
