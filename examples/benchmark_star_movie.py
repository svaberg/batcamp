#!/usr/bin/env python3
"""Render one orbiting ray-traced movie around one star-centered dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from batread.dataset import Dataset

from batcamp import camera_rays
from batcamp import OctreeInterpolator
from batcamp import OctreeRayTracer
from benchmark_helpers import _build_octree
from benchmark_helpers import _configure_builder_logging
from benchmark_helpers import _configure_progress_logging
from benchmark_helpers import _ProgressReporter
from benchmark_helpers import _time_call
from benchmark_helpers import DatasetCase
from benchmark_helpers import resolve_data_file

MOVIE_CASES = (
    DatasetCase("local_example", "3d__var_1_n00000000.plt"),
    DatasetCase("local_xyz", "3d__var_2_n00006003.plt"),
    DatasetCase("local_rpa", "3d__var_2_n00060005.plt"),
    DatasetCase("sc", "3d__var_4_n00044000.plt"),
    DatasetCase("ih", "3d__var_4_n00005000.plt"),
)
DEFAULT_DISTANCE_MULTIPLIER = 2.0
DEFAULT_VIEW_WIDTH_MULTIPLIER = 1.5
DEFAULT_ELEVATION_DEG = 15.0


def _xyz_box_corners(domain_bounds: np.ndarray) -> np.ndarray:
    """Return the eight Cartesian box corners for one xyz domain."""
    domain_lo = np.asarray(domain_bounds[:, 0], dtype=float)
    domain_hi = domain_lo + np.asarray(domain_bounds[:, 1], dtype=float)
    return np.array(
        [
            [x, y, z]
            for x in (float(domain_lo[0]), float(domain_hi[0]))
            for y in (float(domain_lo[1]), float(domain_hi[1]))
            for z in (float(domain_lo[2]), float(domain_hi[2]))
        ],
        dtype=float,
    )


def _domain_radius(tree) -> float:
    """Return one outer radius scale for camera placement."""
    if str(tree.tree_coord) == "rpa":
        return float(np.sum(tree.packed_domain_bounds[0], dtype=float))
    corners = _xyz_box_corners(np.asarray(tree.packed_domain_bounds, dtype=float))
    return float(np.max(np.linalg.norm(corners, axis=1)))


def _orbit_origin(*, radius: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Return one camera origin on a circular orbit."""
    azimuth = np.deg2rad(float(azimuth_deg))
    elevation = np.deg2rad(float(elevation_deg))
    cos_elevation = float(np.cos(elevation))
    return np.array(
        [
            float(radius) * cos_elevation * float(np.cos(azimuth)),
            float(radius) * cos_elevation * float(np.sin(azimuth)),
            float(radius) * float(np.sin(elevation)),
        ],
        dtype=float,
    )


def _frame_paths(out_root: Path, *, case_label: str) -> tuple[Path, Path, Path, Path]:
    """Return frame directory plus movie/report artifacts."""
    frame_dir = out_root / f"benchmark_star_movie_{case_label}_frames"
    movie_path = out_root / f"benchmark_star_movie_{case_label}.mp4"
    csv_path = out_root / f"benchmark_star_movie_{case_label}_frame_times.csv"
    report_path = out_root / f"benchmark_star_movie_{case_label}_timing_report.md"
    return frame_dir, movie_path, csv_path, report_path


def _frame_azimuth_deg(frame: int, n_frames: int) -> float:
    """Return one orbit azimuth for one frame index."""
    return 360.0 * (float(frame) + 0.5) / float(n_frames)


def _render_frame(
    tracer: OctreeRayTracer,
    interp: OctreeInterpolator,
    *,
    origin: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    nx: int,
    ny: int,
    width: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one orbiting parallel camera frame."""
    origins, directions = camera_rays(
        origin=origin,
        target=target,
        up=up,
        nx=int(nx),
        ny=int(ny),
        width=float(width),
        height=float(height),
        projection="parallel",
    )
    return tracer.trilinear_image(interp, origins, directions)


def _display_norm(image: np.ndarray) -> LogNorm:
    """Return one stable log norm from one positive image."""
    positive = np.asarray(image[np.isfinite(image) & (image > 0.0)], dtype=float)
    if positive.size == 0:
        raise ValueError("movie frame must contain positive finite values.")
    vmin = float(np.quantile(positive, 0.02))
    vmax = float(np.quantile(positive, 0.995))
    vmin = max(vmin, float(np.min(positive)))
    vmax = max(vmax, vmin * 1.000001)
    return LogNorm(vmin=vmin, vmax=vmax)


def _save_frame(path: Path, image: np.ndarray, *, norm: LogNorm) -> None:
    """Save one rendered frame as one PNG."""
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("black")
    out = np.array(image, dtype=float, copy=True)
    out[~np.isfinite(out) | (out <= 0.0)] = np.nan
    rgba = cmap(norm(np.ma.masked_invalid(out)))
    plt.imsave(path, rgba, origin="lower")


def _write_movie(frame_dir: Path, movie_path: Path, *, fps: int) -> None:
    """Encode one PNG frame sequence to one MP4 movie with ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required to encode benchmark_star_movie output.")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(int(fps)),
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-pix_fmt",
            "yuv420p",
            str(movie_path),
        ],
        check=True,
        text=True,
    )


def _write_report(
    report_path: Path,
    rows: list[dict[str, float | int | str]],
    *,
    case_label: str,
    variable: str,
    build_s: float,
    interp_s: float,
    tracer_s: float,
    encode_s: float,
    movie_path: Path,
) -> None:
    """Write one concise timing report."""
    ray_times = np.asarray([float(row["ray_s"]) for row in rows], dtype=float)
    lines = [
        f"# benchmark_star_movie {case_label}",
        "",
        f"- variable: `{variable}`",
        f"- frames: `{len(rows)}`",
        f"- mean ray_s: `{float(np.mean(ray_times)):.6f}`",
        f"- median ray_s: `{float(np.median(ray_times)):.6f}`",
        f"- max ray_s: `{float(np.max(ray_times)):.6f}`",
        f"- total ray_s: `{float(np.sum(ray_times)):.6f}`",
        f"- build_s: `{float(build_s):.6f}`",
        f"- interp_s: `{float(interp_s):.6f}`",
        f"- tracer_s: `{float(tracer_s):.6f}`",
        f"- encode_s: `{float(encode_s):.6f}`",
        f"- movie: `{movie_path.name}`",
        "",
        "| frame | azimuth_deg | ray_s | count_median | count_max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['frame'])} | {float(row['azimuth_deg']):.3f} | "
            f"{float(row['ray_s']):.6f} | {float(row['count_median']):.1f} | {int(row['count_max'])} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_case(
    *,
    case: DatasetCase,
    repo_root: Path,
    out_root: Path,
    progress: _ProgressReporter,
    n_frames: int,
    nx: int,
    ny: int,
    fps: int,
    distance_multiplier: float,
    view_width_multiplier: float,
    elevation_deg: float,
    variable: str,
) -> None:
    """Render one orbit movie for one dataset case."""
    data_path = resolve_data_file(repo_root, case.file_name)
    progress.start(f"[{case.label}] load dataset")
    ds, load_s = _time_call(Dataset.from_file, str(data_path))
    progress.complete(f"[{case.label}] load dataset", load_s, detail=f"path={data_path}")

    progress.start(f"[{case.label}] build octree")
    tree, build_s = _time_call(_build_octree, ds)
    progress.complete(f"[{case.label}] build octree", build_s, detail=f"coord={tree.tree_coord}")

    progress.start(f"[{case.label}] build interpolator")
    interp, interp_s = _time_call(OctreeInterpolator, tree, np.asarray(ds[variable], dtype=float))
    progress.complete(f"[{case.label}] build interpolator", interp_s)

    progress.start(f"[{case.label}] build tracer")
    tracer, tracer_s = _time_call(OctreeRayTracer, tree)
    progress.complete(f"[{case.label}] build tracer", tracer_s)

    domain_radius = _domain_radius(tree)
    camera_radius = float(distance_multiplier) * domain_radius
    view_width = float(view_width_multiplier) * domain_radius
    view_height = view_width * float(ny) / float(nx)
    target = np.zeros(3, dtype=float)
    up = np.array([0.0, 0.0, 1.0], dtype=float)

    frame_dir, movie_path, csv_path, report_path = _frame_paths(out_root, case_label=case.label)
    frame_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in frame_dir.glob("frame_*.png"):
        frame_path.unlink()
    movie_path.unlink(missing_ok=True)
    csv_path.unlink(missing_ok=True)
    report_path.unlink(missing_ok=True)

    warm_azimuth_deg = _frame_azimuth_deg(0, int(n_frames))
    warm_origin = _orbit_origin(
        radius=camera_radius,
        azimuth_deg=warm_azimuth_deg,
        elevation_deg=elevation_deg,
    )
    progress.start(f"[{case.label}] warm frame")
    try:
        (warm_image, warm_counts), warm_s = _time_call(
            _render_frame,
            tracer,
            interp,
            origin=warm_origin,
            target=target,
            up=up,
            nx=int(nx),
            ny=int(ny),
            width=float(view_width),
            height=float(view_height),
        )
    except ValueError as exc:
        raise ValueError(
            f"{case.label} warm frame failed at azimuth_deg={float(warm_azimuth_deg):.3f}"
        ) from exc
    norm = _display_norm(warm_image)
    progress.complete(
        f"[{case.label}] warm frame",
        warm_s,
        detail=f"norm=[{float(norm.vmin):.3e}, {float(norm.vmax):.3e}]",
    )

    rows: list[dict[str, float | int | str]] = []
    csv_rows: list[list[str]] = []
    progress.note(
        f"[{case.label}] orbit frames={int(n_frames)} nx={int(nx)} ny={int(ny)} "
        f"radius={camera_radius:.3f} width={view_width:.3f} elevation_deg={float(elevation_deg):.3f}"
    )
    for frame in range(int(n_frames)):
        azimuth_deg = _frame_azimuth_deg(frame, int(n_frames))
        origin = _orbit_origin(
            radius=camera_radius,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
        )
        try:
            (image, counts), ray_s = _time_call(
                _render_frame,
                tracer,
                interp,
                origin=origin,
                target=target,
                up=up,
                nx=int(nx),
                ny=int(ny),
                width=float(view_width),
                height=float(view_height),
            )
        except ValueError as exc:
            raise ValueError(
                f"{case.label} frame={int(frame)} azimuth_deg={float(azimuth_deg):.3f} failed"
            ) from exc
        frame_path = frame_dir / f"frame_{frame:04d}.png"
        _save_frame(frame_path, image, norm=norm)
        row = {
            "frame": int(frame),
            "azimuth_deg": float(azimuth_deg),
            "ray_s": float(ray_s),
            "count_median": float(np.median(np.asarray(counts, dtype=float))),
            "count_max": int(np.max(np.asarray(counts, dtype=np.int64))),
            "frame_path": frame_path.name,
        }
        rows.append(row)
        csv_rows.append(
            [
                str(int(row["frame"])),
                f"{float(row['azimuth_deg']):.6f}",
                f"{float(row['ray_s']):.6f}",
                f"{float(row['count_median']):.1f}",
                str(int(row["count_max"])),
                str(row["frame_path"]),
            ]
        )
        if frame == 0 or frame == int(n_frames) - 1 or (frame + 1) % 10 == 0:
            progress.note(
                f"[{case.label}] frame {frame + 1}/{int(n_frames)} "
                f"ray={float(ray_s):.2f}s count_max={int(row['count_max'])}"
            )

    csv_path.write_text(
        "frame,azimuth_deg,ray_s,count_median,count_max,frame_path\n"
        + "\n".join(",".join(row) for row in csv_rows)
        + "\n",
        encoding="utf-8",
    )

    progress.start(f"[{case.label}] encode movie")
    t_encode = time.perf_counter()
    _write_movie(frame_dir, movie_path, fps=int(fps))
    encode_s = float(time.perf_counter() - t_encode)
    progress.complete(f"[{case.label}] encode movie", encode_s, detail=f"path={movie_path.name}")

    _write_report(
        report_path,
        rows,
        case_label=case.label,
        variable=variable,
        build_s=build_s,
        interp_s=interp_s,
        tracer_s=tracer_s,
        encode_s=encode_s,
        movie_path=movie_path,
    )
    progress.note(f"[{case.label}] done -> benchmark_star_movie_{case.label}_*")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one orbiting star movie with the batcamp ray tracer.")
    parser.add_argument(
        "--frames",
        type=int,
        default=96,
        help="Number of movie frames to render (default: 96).",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=256,
        help="Frame width in pixels (default: 256).",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=256,
        help="Frame height in pixels (default: 256).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Movie frames per second (default: 24).",
    )
    parser.add_argument(
        "--distance-multiplier",
        type=float,
        default=DEFAULT_DISTANCE_MULTIPLIER,
        help=f"Camera radius as a multiple of the dataset outer radius (default: {DEFAULT_DISTANCE_MULTIPLIER:.1f}).",
    )
    parser.add_argument(
        "--view-width-multiplier",
        type=float,
        default=DEFAULT_VIEW_WIDTH_MULTIPLIER,
        help=f"Image-plane width as a multiple of the dataset outer radius (default: {DEFAULT_VIEW_WIDTH_MULTIPLIER:.1f}).",
    )
    parser.add_argument(
        "--elevation-deg",
        type=float,
        default=DEFAULT_ELEVATION_DEG,
        help=f"Orbit elevation in degrees (default: {DEFAULT_ELEVATION_DEG:.1f}).",
    )
    parser.add_argument(
        "--variable",
        default="Rho [g/cm^3]",
        help="Dataset variable to ray-integrate.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Output directory for frames, movie, and timing report.",
    )
    args = parser.parse_args()

    if int(args.frames) <= 0:
        raise ValueError("frames must be positive.")
    if int(args.nx) <= 0 or int(args.ny) <= 0:
        raise ValueError("nx and ny must be positive.")
    if int(args.fps) <= 0:
        raise ValueError("fps must be positive.")
    if float(args.distance_multiplier) <= 0.0:
        raise ValueError("distance_multiplier must be positive.")
    if float(args.view_width_multiplier) <= 0.0:
        raise ValueError("view_width_multiplier must be positive.")

    repo_root = Path(__file__).resolve().parent.parent
    out_root = (repo_root / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    progress_log_path = out_root / "benchmark_star_movie.log"
    progress_log_path.write_text("", encoding="utf-8")
    _configure_progress_logging(log_path=progress_log_path)
    _configure_builder_logging(log_path=progress_log_path)
    progress = _ProgressReporter(log_path=progress_log_path)

    progress.note(f"output_dir={out_root}")
    for case in MOVIE_CASES:
        _run_case(
            case=case,
            repo_root=repo_root,
            out_root=out_root,
            progress=progress,
            n_frames=int(args.frames),
            nx=int(args.nx),
            ny=int(args.ny),
            fps=int(args.fps),
            distance_multiplier=float(args.distance_multiplier),
            view_width_multiplier=float(args.view_width_multiplier),
            elevation_deg=float(args.elevation_deg),
            variable=str(args.variable),
        )


if __name__ == "__main__":
    main()
