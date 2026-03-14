from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pooch
from batread.dataset import Dataset

from batcamp import OctreeInterpolator
from batcamp import OctreeRayInterpolator
from batcamp.ray import _has_xyz_lookup_kernel


_URL = "https://zenodo.org/records/7110555/files/run-Sun-G2211.tar.gz"
_HASH = "c31a32aab08cc20d5b643bba734fd7220e6b369e691f55f88a3a08cc5b2a2136"
_MEMBERS = {
    "SC": "run-Sun-G2211/SC/IO2/3d__var_4_n00044000.plt",
    "IH": "run-Sun-G2211/IH/IO2/3d__var_4_n00005000.plt",
}


def parse_grid_sizes(spec: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for chunk in spec.split(","):
        part = chunk.strip().lower()
        if not part:
            continue
        if "x" not in part:
            raise ValueError(f"Invalid grid token '{part}', expected NyxNz format.")
        a, b = part.split("x", 1)
        ny = int(a.strip())
        nz = int(b.strip())
        if ny <= 0 or nz <= 0:
            raise ValueError(f"Grid values must be positive: {part}")
        out.append((ny, nz))
    if not out:
        raise ValueError("No valid grid sizes parsed.")
    return out


def camera(tree, *, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray, float, tuple[float, float, float, float, float, float]]:
    dmin, dmax = tree.domain_bounds(coord="xyz")
    xmin, xmax = float(dmin[0]), float(dmax[0])
    ymin, ymax = float(dmin[1]), float(dmax[1])
    zmin, zmax = float(dmin[2]), float(dmax[2])

    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)
    yg, zg = np.meshgrid(y, z, indexing="xy")
    x0 = xmin + 1.0e-6 * (xmax - xmin)
    origins = np.column_stack((np.full(yg.size, x0, dtype=float), yg.ravel(), zg.ravel()))
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    t_end = (xmax - xmin) * 0.999999
    return origins, direction, t_end, (xmin, xmax, ymin, ymax, zmin, zmax)


def baseline_grid_sum(
    interp: OctreeInterpolator,
    bounds: tuple[float, float, float, float, float, float],
    *,
    ny: int,
    nz: int,
    nx: int,
) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)
    xg, yg, zg = np.meshgrid(x, y, z, indexing="ij")
    q = np.column_stack((xg.ravel(), yg.ravel(), zg.ravel()))
    vals = np.asarray(
        interp(q, query_coord="xyz", log_outside_domain=False),
        dtype=float,
    ).reshape(nx, ny, nz)
    return np.trapezoid(vals, x=x, axis=0)


def time_hot_call(fn) -> float:
    _ = fn()
    t0 = time.perf_counter()
    _ = fn()
    return time.perf_counter() - t0


def path_diagnostics(interp: OctreeInterpolator, *, axis_aligned: bool = True) -> str:
    has_xyz_lookup = bool(_has_xyz_lookup_kernel(interp.tree))
    has_xyz_interp = bool(interp.xyz_interp_state is not None)
    scalar = bool(int(interp.n_value_components) == 1)
    can_xyz_scalar = bool(has_xyz_lookup and has_xyz_interp and scalar)
    if str(interp.tree.tree_coord) == "rpa":
        mode = "rpa"
    elif can_xyz_scalar and axis_aligned:
        mode = "xyz_axis_kernel"
    elif can_xyz_scalar:
        mode = "xyz_general_kernel"
    else:
        mode = "fallback"
    return mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Short-loop ray performance microbenchmark.")
    parser.add_argument("--grid-sizes", default="3x3,5x5", help="Comma-separated NyxNz list (default: 3x3,5x5).")
    parser.add_argument("--max-rays", type=int, default=50, help="Hard cap on rays per run (default: 50).")
    parser.add_argument("--baseline-nx", type=int, default=256, help="Nx for grid-sum baseline (default: 256).")
    args = parser.parse_args()

    grids = parse_grid_sizes(args.grid_sizes)
    for ny, nz in grids:
        n_rays = ny * nz
        if n_rays > int(args.max_rays):
            raise ValueError(f"Grid {ny}x{nz} has {n_rays} rays, exceeds cap {args.max_rays}.")

    print("| label | grid | rays | tree_coord | path | ray_s | baseline_s | ratio_ray_over_base |")
    print("|---|---:|---:|---|---|---:|---:|---:|")

    for label, member in _MEMBERS.items():
        path = Path(
            pooch.retrieve(
                url=_URL,
                known_hash=_HASH,
                progressbar=False,
                processor=pooch.Untar(members=[member]),
            )[0]
        )
        ds = Dataset.from_file(str(path))
        interp = OctreeInterpolator(ds, ["Rho [g/cm^3]"])
        ray = OctreeRayInterpolator(interp)

        for ny, nz in grids:
            origins, direction, t_end, bounds = camera(interp.tree, ny=ny, nz=nz)
            n_rays = int(origins.shape[0])
            path_mode = path_diagnostics(interp, axis_aligned=True)

            ray_s = time_hot_call(
                lambda: ray.integrate_field_along_rays(
                    origins,
                    direction,
                    0.0,
                    float(t_end),
                    chunk_size=4096,
                )
            )
            baseline_s = time_hot_call(
                lambda: baseline_grid_sum(
                    interp,
                    bounds,
                    ny=ny,
                    nz=nz,
                    nx=int(args.baseline_nx),
                )
            )
            ratio = float(ray_s / max(baseline_s, 1.0e-15))
            print(
                f"| {label} | {ny}x{nz} | {n_rays} | {interp.tree.tree_coord} | {path_mode} | "
                f"{ray_s:.4f} | {baseline_s:.4f} | {ratio:.2f} |"
            )


if __name__ == "__main__":
    main()
