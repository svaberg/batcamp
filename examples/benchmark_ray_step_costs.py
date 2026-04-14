#!/usr/bin/env python3
"""Profile one warmed xyz ray trace through public crossing-trace statistics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from batread.dataset import Dataset

from batcamp import OctreeRayTracer
from batcamp.constants import XYZ_VARS

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


def _trace_report(
    tracer: OctreeRayTracer,
    *,
    n_plane: int,
    bounds: tuple[float, float, float, float, float, float],
) -> dict[str, float | int]:
    """Return one public crossing-trace timing and segment-statistics report."""
    origins, directions, t_end = _ray_setup(n_plane=int(n_plane), bounds=bounds)

    tracer.trace(origins, directions, t_min=0.0, t_max=float(t_end))
    t0 = time.perf_counter()
    segments = tracer.trace(origins, directions, t_min=0.0, t_max=float(t_end))
    trace_s = float(time.perf_counter() - t0)

    ray_counts = np.diff(segments.ray_offsets)
    time_counts = np.diff(segments.time_offsets)
    if not np.array_equal(time_counts, np.where(ray_counts == 0, 0, ray_counts + 1)):
        raise ValueError("RaySegments has inconsistent packed counts.")

    n_rays = int(ray_counts.size)
    nonempty_mask = ray_counts > 0
    nonempty_rays = int(np.count_nonzero(nonempty_mask))
    raw_segments = int(segments.cell_ids.size)
    positive_segments = 0
    zero_length_hops = 0
    total_length = 0.0
    positive_counts = np.zeros(n_rays, dtype=np.int64)

    for ray_id in range(n_rays):
        time_lo = int(segments.time_offsets[ray_id])
        time_hi = int(segments.time_offsets[ray_id + 1])
        ray_times = segments.times[time_lo:time_hi]
        if ray_times.size == 0:
            continue
        ray_lengths = np.diff(ray_times)
        positive_mask = ray_lengths > 0.0
        positive_count = int(np.count_nonzero(positive_mask))
        zero_count = int(ray_lengths.size - positive_count)
        positive_counts[ray_id] = positive_count
        positive_segments += positive_count
        zero_length_hops += zero_count
        total_length += float(np.sum(ray_lengths[positive_mask], dtype=np.float64))

    rays_per_s = float(n_rays) / trace_s if trace_s > 0.0 else np.nan
    positive_segments_per_s = float(positive_segments) / trace_s if trace_s > 0.0 else np.nan
    mean_positive_segments = float(np.mean(positive_counts[nonempty_mask])) if nonempty_rays else 0.0
    max_positive_segments = int(np.max(positive_counts)) if positive_counts.size else 0
    mean_length = total_length / float(nonempty_rays) if nonempty_rays else 0.0

    return {
        "sample_resolution": int(n_plane),
        "sample_rays": n_rays,
        "nonempty_rays": nonempty_rays,
        "raw_segments": raw_segments,
        "positive_segments": int(positive_segments),
        "zero_length_hops": int(zero_length_hops),
        "trace_s": trace_s,
        "rays_per_s": rays_per_s,
        "positive_segments_per_s": positive_segments_per_s,
        "mean_positive_segments_per_ray": mean_positive_segments,
        "max_positive_segments_per_ray": max_positive_segments,
        "total_length": total_length,
        "mean_length_per_nonempty_ray": mean_length,
    }


def _write_report(report: dict[str, float | int], out_path: Path) -> None:
    """Write one markdown report with the public crossing-trace benchmark summary."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ray Trace Cost Report",
        "",
        "- scope: `xyz` public crossing tracer only",
        f"- sampled ray plane: `{int(report['sample_resolution'])}x{int(report['sample_resolution'])}`",
        f"- sampled rays: `{int(report['sample_rays'])}`",
        f"- nonempty rays: `{int(report['nonempty_rays'])}`",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| trace wall time (s) | {float(report['trace_s']):.6f} |",
        f"| rays / s | {float(report['rays_per_s']):.1f} |",
        f"| positive segments / s | {float(report['positive_segments_per_s']):.1f} |",
        f"| raw packed segments | {int(report['raw_segments'])} |",
        f"| positive segments | {int(report['positive_segments'])} |",
        f"| zero-length hops | {int(report['zero_length_hops'])} |",
        f"| mean positive segments / nonempty ray | {float(report['mean_positive_segments_per_ray']):.3f} |",
        f"| max positive segments / ray | {int(report['max_positive_segments_per_ray'])} |",
        f"| total traced length | {float(report['total_length']):.6f} |",
        f"| mean traced length / nonempty ray | {float(report['mean_length_per_nonempty_ray']):.6f} |",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile one warmed xyz ray trace through public crossing-trace statistics.")
    parser.add_argument(
        "--dataset",
        default="3d__var_2_n00006003.plt",
        help="XYZ dataset basename to resolve from sample_data or pooch.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=31,
        help="Square ray-plane resolution used for the trace profile.",
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
    if str(tree.tree_coord) != "xyz":
        raise NotImplementedError("benchmark_ray_step_costs currently supports only tree_coord='xyz'.")

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

    progress.start(f"profile crossing trace at {int(args.resolution)}x{int(args.resolution)}")
    report, profile_s = _time_call(
        _trace_report,
        tracer,
        n_plane=int(args.resolution),
        bounds=bounds,
    )
    progress.complete(
        f"profile crossing trace at {int(args.resolution)}x{int(args.resolution)}",
        profile_s,
        detail=(
            f"trace_s={float(report['trace_s']):.6f} "
            f"positive_segments={int(report['positive_segments'])} "
            f"zero_hops={int(report['zero_length_hops'])}"
        ),
    )

    out_path = out_root / f"benchmark_ray_step_costs_{Path(str(args.dataset)).stem}_{int(args.resolution)}x{int(args.resolution)}.md"
    _write_report(report, out_path)
    progress.note(f"report={out_path}")


if __name__ == "__main__":
    main()
