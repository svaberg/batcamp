#!/usr/bin/env python3
"""Render a simple perspective flyby of the sample star with the FOV camera."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tarfile
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pooch
from batread.dataset import Dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from batcamp import FovCamera
from batcamp import OctreeInterpolator
from batcamp import OctreeRayInterpolator


_G2211_URL = "https://zenodo.org/records/7110555/files/run-Sun-G2211.tar.gz"
_G2211_SHA256 = "c31a32aab08cc20d5b643bba734fd7220e6b369e691f55f88a3a08cc5b2a2136"
_DEFAULT_SC_MEMBER = "run-Sun-G2211/SC/IO2/3d__var_4_n00044000.plt"
_DEFAULT_FIELD = "Rho [g/cm^3]"
_DEFAULT_RESOLUTION = 256
_DEFAULT_STEPS = 48
_DEFAULT_OUTPUT = _REPO_ROOT / "artifacts" / "ray_flyby"
_DEFAULT_VERTICAL_FOV_DEGREES = 70.0
_DEFAULT_CLOSEST_DISTANCE_FACTOR = 1.15
_DEFAULT_PATH_HALF_LENGTH_FACTOR = 6.0
_DEFAULT_CHUNK_SIZE = 4096


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=_DEFAULT_RESOLUTION, help="square output resolution")
    parser.add_argument("--steps", type=int, default=_DEFAULT_STEPS, help="number of flyby frames")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT, help="directory for output frames")
    return parser.parse_args()


def _time_call(fn, *args):
    """Return `(result, seconds)` for one call."""
    start = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - start


def _load_ray_context() -> tuple[OctreeInterpolator, OctreeRayInterpolator, dict[str, float | str]]:
    data_path, resolve_s = _time_call(_resolve_sc_data_file)
    ds, read_s = _time_call(Dataset.from_file, str(data_path))
    interp, interp_s = _time_call(OctreeInterpolator, ds, [_DEFAULT_FIELD])
    ray, ray_s = _time_call(OctreeRayInterpolator, interp)
    timing = {
        "data_path": str(data_path),
        "resolve_s": resolve_s,
        "read_s": read_s,
        "interp_s": interp_s,
        "ray_s": ray_s,
    }
    return interp, ray, timing


def _resolve_sc_data_file() -> Path:
    """Resolve the SC sample file from local cache or fetch it once via pooch."""
    archive_path = Path(
        pooch.retrieve(
            url=_G2211_URL,
            known_hash=_G2211_SHA256,
            progressbar=False,
        )
    )
    with tarfile.open(archive_path, "r:gz") as tar:
        if _DEFAULT_SC_MEMBER not in {m.name for m in tar.getmembers() if m.isfile()}:
            raise FileNotFoundError(_DEFAULT_SC_MEMBER)
    extracted = pooch.retrieve(
        url=_G2211_URL,
        known_hash=_G2211_SHA256,
        progressbar=False,
        processor=pooch.Untar(members=[_DEFAULT_SC_MEMBER]),
    )
    if isinstance(extracted, (list, tuple)):
        extracted = extracted[0]
    return Path(extracted)


def _flyby_eye_positions(center_xyz: np.ndarray, star_radius: float, n_steps: int) -> np.ndarray:
    """Return a straight flyby path that passes close to the stellar surface."""
    impact = float(_DEFAULT_CLOSEST_DISTANCE_FACTOR) * float(star_radius)
    tilt = np.deg2rad(18.0)
    y0 = impact * np.cos(tilt)
    z0 = impact * np.sin(tilt)
    x_half = float(_DEFAULT_PATH_HALF_LENGTH_FACTOR) * float(star_radius)
    x = np.linspace(-x_half, x_half, int(n_steps), dtype=float)
    eyes = np.column_stack(
        (
            center_xyz[0] + x,
            np.full(x.shape, center_xyz[1] + y0, dtype=float),
            np.full(x.shape, center_xyz[2] + z0, dtype=float),
        )
    )
    return eyes


def _frame_camera(eye_xyz: np.ndarray, center_xyz: np.ndarray, r_max: float) -> FovCamera:
    """Build one perspective camera aimed at the star center."""
    eye = np.asarray(eye_xyz, dtype=float).reshape(3)
    center = np.asarray(center_xyz, dtype=float).reshape(3)
    t_end = float(np.linalg.norm(eye - center) + 1.2 * float(r_max))
    return FovCamera(
        eye_xyz=eye,
        target_xyz=center,
        up_hint_xyz=np.array([0.0, 0.0, 1.0], dtype=float),
        vertical_fov_degrees=float(_DEFAULT_VERTICAL_FOV_DEGREES),
        t_end=t_end,
    )


def _render_frame_values(
    ray: OctreeRayInterpolator,
    *,
    eye_xyz: np.ndarray,
    center_xyz: np.ndarray,
    r_max: float,
    resolution: int,
) -> np.ndarray:
    """Return one `(z, y)` frame of integrated density."""
    camera = _frame_camera(eye_xyz, center_xyz, r_max)
    origins, directions, t_end, image_shape = camera.rays(ny=int(resolution), nz=int(resolution))
    values = np.asarray(
        ray.integrate_field_along_rays(
            origins,
            directions,
            0.0,
            t_end,
            chunk_size=_DEFAULT_CHUNK_SIZE,
        ),
        dtype=float,
    ).reshape(image_shape)
    return values


def _image_extent(*, resolution: int) -> tuple[float, float, float, float]:
    """Return image-plane extent in camera coordinates."""
    half_height = float(np.tan(0.5 * np.deg2rad(_DEFAULT_VERTICAL_FOV_DEGREES)))
    half_width = (float(resolution) / float(resolution)) * half_height
    return (-half_width, half_width, -half_height, half_height)


def _true_log_limits(frames: list[np.ndarray]) -> tuple[float, float]:
    """Return exact global log10 min/max over all positive finite frame values."""
    logs: list[np.ndarray] = []
    for frame in frames:
        positive = frame[np.isfinite(frame) & (frame > 0.0)]
        if positive.size > 0:
            logs.append(np.log10(positive))
    if not logs:
        return -30.0, -10.0
    merged = np.concatenate(logs)
    vmin = float(np.min(merged))
    vmax = float(np.max(merged))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _save_frame(
    values: np.ndarray,
    *,
    eye_xyz: np.ndarray,
    center_xyz: np.ndarray,
    vmin: float,
    vmax: float,
    output_path: Path,
    frame_index: int,
    n_steps: int,
    extent: tuple[float, float, float, float],
) -> None:
    """Save one rendered frame."""
    positive = np.isfinite(values) & (values > 0.0)
    plot = np.full(values.shape, np.nan, dtype=float)
    plot[positive] = np.log10(values[positive])

    cmap = plt.colormaps["magma"].copy()
    cmap.set_bad("black")

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    im = ax.imshow(plot, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect="equal")
    ax.set_xlabel("image x", color="white")
    ax.set_ylabel("image y", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    distance = float(np.linalg.norm(np.asarray(eye_xyz, dtype=float) - np.asarray(center_xyz, dtype=float)))
    ax.set_title(
        f"Flyby frame {frame_index + 1}/{n_steps}  |  eye distance {distance:.2f} R",
        color="white",
        pad=10.0,
    )
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, fraction=0.05)
    cb.set_label("log10(column density)", color="white")
    cb.ax.xaxis.set_tick_params(color="white", labelcolor="white")
    cb.outline.set_edgecolor("white")
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def _write_timing_report(
    output_path: Path,
    *,
    resolution: int,
    steps: int,
    setup: dict[str, float | str],
    vmin: float,
    vmax: float,
    frame_render_s: list[float],
    frame_save_s: list[float],
    total_s: float,
) -> None:
    """Write one markdown timing report."""
    render = np.asarray(frame_render_s, dtype=float)
    save = np.asarray(frame_save_s, dtype=float)
    lines = [
        "# Flyby Timing Report",
        "",
        f"- resolution: `{resolution}x{resolution}`",
        f"- steps: `{steps}`",
        f"- data_path: `{setup['data_path']}`",
        f"- log10_vmin: `{float(vmin):.6f}`",
        f"- log10_vmax: `{float(vmax):.6f}`",
        "",
        "## Setup",
        "",
        f"- resolve_s: `{float(setup['resolve_s']):.6f}`",
        f"- read_s: `{float(setup['read_s']):.6f}`",
        f"- interp_s: `{float(setup['interp_s']):.6f}`",
        f"- ray_s: `{float(setup['ray_s']):.6f}`",
        "",
        "## Frame Loop",
        "",
        "- octree/interpolator/ray objects are built once before the frame loop",
        "- per-frame work only builds camera rays, integrates, and saves the image",
        "- color scale uses the true global min/max over the rendered frames",
        "",
        f"- render_mean_s: `{float(np.mean(render)):.6f}`",
        f"- render_max_s: `{float(np.max(render)):.6f}`",
        f"- save_mean_s: `{float(np.mean(save)):.6f}`",
        f"- save_max_s: `{float(np.max(save)):.6f}`",
        f"- total_render_s: `{float(np.sum(render)):.6f}`",
        f"- total_save_s: `{float(np.sum(save)):.6f}`",
        f"- total_wall_s: `{float(total_s):.6f}`",
        "",
        "## Per Frame",
        "",
        "| frame | render_s | save_s | total_s |",
        "|---:|---:|---:|---:|",
    ]
    for i, (r_s, s_s) in enumerate(zip(frame_render_s, frame_save_s, strict=True)):
        lines.append(f"| {i} | {float(r_s):.6f} | {float(s_s):.6f} | {float(r_s + s_s):.6f} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    total_start = time.perf_counter()
    args = _parse_args()
    resolution = int(args.resolution)
    steps = int(args.steps)
    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    if steps <= 1:
        raise ValueError("steps must be greater than 1.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    interp, ray, setup_timing = _load_ray_context()
    dmin, dmax = interp.tree.domain_bounds(coord="xyz")
    center_xyz = 0.5 * (np.asarray(dmin, dtype=float).reshape(3) + np.asarray(dmax, dtype=float).reshape(3))
    r_lo, r_hi = interp.tree.domain_bounds(coord="rpa")
    r_star = float(np.asarray(r_lo, dtype=float).reshape(3)[0])
    r_max = float(np.asarray(r_hi, dtype=float).reshape(3)[0])
    extent = _image_extent(resolution=resolution)

    eyes_xyz = _flyby_eye_positions(center_xyz, r_star, steps)
    frames: list[np.ndarray] = []
    frame_render_s: list[float] = []
    for i, eye_xyz in enumerate(eyes_xyz):
        print(f"Rendering frame {i + 1}/{steps}", flush=True)
        render_start = time.perf_counter()
        values = _render_frame_values(
            ray,
            eye_xyz=eye_xyz,
            center_xyz=center_xyz,
            r_max=r_max,
            resolution=resolution,
        )
        frame_render_s.append(time.perf_counter() - render_start)
        frames.append(values)

    vmin, vmax = _true_log_limits(frames)
    print(f"Using log10 scale [{vmin:.3f}, {vmax:.3f}]")

    frame_save_s: list[float] = []
    for i, (eye_xyz, values) in enumerate(zip(eyes_xyz, frames, strict=True)):
        print(f"Saving frame {i + 1}/{steps}", flush=True)
        save_start = time.perf_counter()
        _save_frame(
            values,
            eye_xyz=eye_xyz,
            center_xyz=center_xyz,
            vmin=vmin,
            vmax=vmax,
            output_path=output_dir / f"frame_{i:04d}.png",
            frame_index=i,
            n_steps=steps,
            extent=extent,
        )
        frame_save_s.append(time.perf_counter() - save_start)

    _write_timing_report(
        output_dir / "timing_report.md",
        resolution=resolution,
        steps=steps,
        setup=setup_timing,
        vmin=vmin,
        vmax=vmax,
        frame_render_s=frame_render_s,
        frame_save_s=frame_save_s,
        total_s=time.perf_counter() - total_start,
    )


if __name__ == "__main__":
    main()
