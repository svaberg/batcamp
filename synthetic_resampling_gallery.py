#!/usr/bin/env python3
"""Generate one gallery of synthetic plane-resampling patterns."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from batcamp import Octree
from batcamp import OctreeBuilder
from batcamp import OctreeInterpolator
from batcamp.constants import XYZ_VARS
from tests.fake_dataset import FakeDataset
from tests.fake_dataset import build_cartesian_hex_mesh


def _build_ring_pattern_dataset(
    *,
    mesh_nxy: int,
    ring_radius: float,
    ring_width: float,
    center_x: float,
    center_y: float,
) -> FakeDataset:
    """Build one thin slab whose midplane contains one smooth annular peak."""
    points, corners = build_cartesian_hex_mesh(
        x_edges=np.linspace(-1.0, 1.0, int(mesh_nxy) + 1, dtype=float),
        y_edges=np.linspace(-1.0, 1.0, int(mesh_nxy) + 1, dtype=float),
        z_edges=np.array([-0.25, 0.25], dtype=float),
    )
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    radius = np.sqrt((x - float(center_x)) ** 2 + (y - float(center_y)) ** 2)
    pattern = 0.05 + np.exp(-((radius - float(ring_radius)) ** 2) / (2.0 * float(ring_width) ** 2))
    return FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: x,
            XYZ_VARS[1]: y,
            XYZ_VARS[2]: z,
            "Pattern": pattern,
        },
    )


def _build_checkerboard_pattern_dataset(
    *,
    mesh_nxy: int,
    tile_count: int,
    x_offset: float,
    y_offset: float,
) -> FakeDataset:
    """Build one thin slab whose midplane contains one smooth alternating checkerboard."""
    points, corners = build_cartesian_hex_mesh(
        x_edges=np.linspace(-1.0, 1.0, int(mesh_nxy) + 1, dtype=float),
        y_edges=np.linspace(-1.0, 1.0, int(mesh_nxy) + 1, dtype=float),
        z_edges=np.array([-0.25, 0.25], dtype=float),
    )
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    freq = 0.5 * float(tile_count)
    pattern = 1.0 + 0.75 * np.sin(2.0 * np.pi * freq * (x - float(x_offset))) * np.sin(
        2.0 * np.pi * freq * (y - float(y_offset))
    )
    return FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: x,
            XYZ_VARS[1]: y,
            XYZ_VARS[2]: z,
            "Pattern": pattern,
        },
    )


def _build_adaptive_slab_points_and_corners() -> tuple[np.ndarray, np.ndarray]:
    """Build one dyadic slab with six coarse octants and one refined northeast column."""
    x_edges = np.linspace(-1.0, 1.0, 9, dtype=float)
    y_edges = np.linspace(-1.0, 1.0, 9, dtype=float)
    z_edges = np.linspace(-0.25, 0.25, 9, dtype=float)
    points, _unused = build_cartesian_hex_mesh(x_edges=x_edges, y_edges=y_edges, z_edges=z_edges)
    node_index = np.arange(points.shape[0], dtype=np.int64).reshape(x_edges.size, y_edges.size, z_edges.size)

    corners: list[list[int]] = []

    def add_cell(ix0: int, ix1: int, iy0: int, iy1: int, iz0: int, iz1: int) -> None:
        corners.append(
            [
                int(node_index[ix0, iy0, iz0]),
                int(node_index[ix1, iy0, iz0]),
                int(node_index[ix0, iy1, iz0]),
                int(node_index[ix1, iy1, iz0]),
                int(node_index[ix0, iy0, iz1]),
                int(node_index[ix1, iy0, iz1]),
                int(node_index[ix0, iy1, iz1]),
                int(node_index[ix1, iy1, iz1]),
            ]
        )

    for ix0, ix1 in ((0, 4), (4, 8)):
        for iy0, iy1 in ((0, 4), (4, 8)):
            for iz0, iz1 in ((0, 4), (4, 8)):
                if ix0 == 4 and iy0 == 4:
                    continue
                add_cell(ix0, ix1, iy0, iy1, iz0, iz1)
    for ix in range(4, 8):
        for iy in range(4, 8):
            for iz in range(0, 8):
                add_cell(ix, ix + 1, iy, iy + 1, iz, iz + 1)
    return points, np.array(corners, dtype=np.int64)


def _build_adaptive_ring_pattern_dataset(
    *,
    ring_radius: float,
    ring_width: float,
    center_x: float,
    center_y: float,
) -> FakeDataset:
    """Build one adaptive slab with a ring concentrated in the refined patch."""
    points, corners = _build_adaptive_slab_points_and_corners()
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    radius = np.sqrt((x - float(center_x)) ** 2 + (y - float(center_y)) ** 2)
    pattern = 0.05 + np.exp(-((radius - float(ring_radius)) ** 2) / (2.0 * float(ring_width) ** 2))
    return FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: x,
            XYZ_VARS[1]: y,
            XYZ_VARS[2]: z,
            "Pattern": pattern,
        },
    )


def _build_adaptive_checkerboard_pattern_dataset(
    *,
    tile_count: int,
    x_offset: float,
    y_offset: float,
) -> FakeDataset:
    """Build one adaptive slab with a checkerboard over the refined northeast patch."""
    points, corners = _build_adaptive_slab_points_and_corners()
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    tile_size = 1.0 / float(tile_count)
    ix = np.floor(np.clip((x - float(x_offset)) / tile_size, 0.0, float(tile_count) - 1.0e-6)).astype(np.int64)
    iy = np.floor(np.clip((y - float(y_offset)) / tile_size, 0.0, float(tile_count) - 1.0e-6)).astype(np.int64)
    in_patch = (x >= float(x_offset)) & (x <= float(x_offset) + 1.0) & (y >= float(y_offset)) & (y <= float(y_offset) + 1.0)
    parity = (ix + iy) % 2
    pattern = np.where(in_patch, np.where(parity == 0, 1.8, 0.2), 1.0)
    return FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: x,
            XYZ_VARS[1]: y,
            XYZ_VARS[2]: z,
            "Pattern": pattern,
        },
    )


def _resample_xy_plane(
    interp: OctreeInterpolator,
    *,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample one scalar field onto one regular `xy` plane just above `z=0`."""
    dmin, dmax = interp.tree.domain_bounds(coord="xyz")
    x = np.linspace(float(dmin[0]), float(dmax[0]), int(resolution), dtype=float)
    y = np.linspace(float(dmin[1]), float(dmax[1]), int(resolution), dtype=float)
    xg, yg = np.meshgrid(x, y, indexing="xy")
    zg = np.full_like(xg, 0.03125, dtype=float)
    query = np.column_stack((xg.ravel(), yg.ravel(), zg.ravel()))
    img = np.asarray(
        interp(query, query_coord="xyz", log_outside_domain=False),
        dtype=float,
    ).reshape(int(resolution), int(resolution))
    return xg, yg, img


def _save_image(
    path: Path,
    *,
    xg: np.ndarray,
    yg: np.ndarray,
    img: np.ndarray,
    title: str,
) -> None:
    """Save one resampled scalar image with one colorbar."""
    finite = np.isfinite(img)
    vals = img[finite]
    if vals.size == 0:
        raise ValueError(f"No finite values for {path.name}.")
    fig, ax = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
    im = ax.imshow(
        img,
        origin="lower",
        extent=(float(np.min(xg)), float(np.max(xg)), float(np.min(yg)), float(np.max(yg))),
        cmap="viridis",
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _ring_cases() -> list[tuple[str, FakeDataset]]:
    """Return one batch of concentric-ring datasets."""
    out: list[tuple[str, FakeDataset]] = []
    for ring_radius in (0.28, 0.42, 0.56, 0.70):
        for ring_width in (0.05, 0.09):
            label = f"rings_r{ring_radius:.2f}_w{ring_width:.2f}"
            ds = _build_ring_pattern_dataset(
                mesh_nxy=64,
                ring_radius=ring_radius,
                ring_width=ring_width,
                center_x=0.0,
                center_y=0.0,
            )
            out.append((label, ds))
    return out


def _checker_cases() -> list[tuple[str, FakeDataset]]:
    """Return one batch of checkerboard datasets."""
    out: list[tuple[str, FakeDataset]] = []
    for tile_count in (4, 6, 8, 10):
        for x_offset, y_offset in ((0.00, 0.00), (0.08, 0.00), (0.08, 0.08)):
            label = f"checker_t{tile_count:02d}_ox{x_offset:.2f}_oy{y_offset:.2f}"
            ds = _build_checkerboard_pattern_dataset(
                mesh_nxy=64,
                tile_count=tile_count,
                x_offset=x_offset,
                y_offset=y_offset,
            )
            out.append((label, ds))
    return out


def _adaptive_ring_cases() -> list[tuple[str, FakeDataset]]:
    """Return one batch of adaptive ring datasets."""
    out: list[tuple[str, FakeDataset]] = []
    for ring_radius in (0.18, 0.24, 0.30):
        for ring_width in (0.04, 0.07):
            label = f"adaptive_rings_r{ring_radius:.2f}_w{ring_width:.2f}"
            ds = _build_adaptive_ring_pattern_dataset(
                ring_radius=ring_radius,
                ring_width=ring_width,
                center_x=0.5,
                center_y=0.5,
            )
            out.append((label, ds))
    return out


def _adaptive_checker_cases() -> list[tuple[str, FakeDataset]]:
    """Return one batch of adaptive checkerboard datasets."""
    out: list[tuple[str, FakeDataset]] = []
    for tile_count in (2,):
        for x_offset, y_offset in ((0.00, 0.00), (0.10, 0.00), (0.10, 0.10)):
            label = f"adaptive_checker_t{tile_count:02d}_ox{x_offset:.2f}_oy{y_offset:.2f}"
            ds = _build_adaptive_checkerboard_pattern_dataset(
                tile_count=tile_count,
                x_offset=x_offset,
                y_offset=y_offset,
            )
            out.append((label, ds))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic resampling pattern gallery.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/synthetic_patterns_gallery",
        help="Directory where PNGs are written.",
    )
    parser.add_argument(
        "--min-resolution",
        type=int,
        default=2,
        help="Smallest square output resolution.",
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=1024,
        help="Largest square output resolution.",
    )
    parser.add_argument(
        "--max-seconds-per-image",
        type=float,
        default=0.5,
        help="Stop increasing resolution for one pattern once one image exceeds this wall time.",
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=0,
        help="Optional positive cap on how many pattern cases to generate.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    cases = _adaptive_ring_cases() + _adaptive_checker_cases() + _ring_cases() + _checker_cases()
    if int(args.limit_cases) > 0:
        cases = cases[: int(args.limit_cases)]

    min_resolution = int(args.min_resolution)
    max_resolution = int(args.max_resolution)
    if min_resolution <= 0:
        raise ValueError("min_resolution must be positive.")
    if max_resolution < min_resolution:
        raise ValueError("max_resolution must be >= min_resolution.")
    if float(args.max_seconds_per_image) <= 0.0:
        raise ValueError("max_seconds_per_image must be positive.")

    for label, ds in cases:
        interp = OctreeInterpolator(OctreeBuilder().build(ds, tree_coord="xyz"), ["Pattern"])
        resolution = int(min_resolution)
        while resolution <= max_resolution:
            t0 = perf_counter()
            xg, yg, img = _resample_xy_plane(interp, resolution=resolution)
            out_path = out_dir / f"{label}_{resolution:03d}.png"
            _save_image(
                out_path,
                xg=xg,
                yg=yg,
                img=img,
                title=f"{label} @ {resolution}x{resolution}",
            )
            elapsed_s = float(perf_counter() - t0)
            finite = np.isfinite(img)
            vals = img[finite]
            print(
                (
                    f"{out_path} finite={int(finite.sum())} "
                    f"min={float(np.min(vals)):.6f} max={float(np.max(vals)):.6f} "
                    f"seconds={elapsed_s:.3f}"
                ),
                flush=True,
            )
            if elapsed_s > float(args.max_seconds_per_image):
                print(
                    (
                        f"stop {label} at {resolution}x{resolution}: "
                        f"{elapsed_s:.3f}s > {float(args.max_seconds_per_image):.3f}s"
                    ),
                    flush=True,
                )
                break
            resolution *= 2


if __name__ == "__main__":
    main()
