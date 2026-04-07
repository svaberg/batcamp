from __future__ import annotations

from time import perf_counter

import numpy as np
import pytest
from batread import Dataset
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from sample_data_helper import data_file
from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp.constants import XYZ_VARS

pytestmark = [
    pytest.mark.design_lockin,
    pytest.mark.perf,
]


_RHO = "Rho [g/cm^3]"
_PLANE_RAMP = (2, 4, 8, 16, 32, 64)


def _xyz_points(ds: Dataset) -> np.ndarray:
    """Return dataset point coordinates as one dense `(n, 3)` array."""
    return np.column_stack(
        [
            np.asarray(ds[XYZ_VARS[0]], dtype=float),
            np.asarray(ds[XYZ_VARS[1]], dtype=float),
            np.asarray(ds[XYZ_VARS[2]], dtype=float),
        ]
    )


def _xy_plane_queries(xyz: np.ndarray, *, resolution: int) -> np.ndarray:
    """Return one regular `xy` plane through the dataset midplane."""
    dmin = np.min(xyz, axis=0)
    dmax = np.max(xyz, axis=0)
    x = np.linspace(float(dmin[0]), float(dmax[0]), int(resolution), dtype=float)
    y = np.linspace(float(dmin[1]), float(dmax[1]), int(resolution), dtype=float)
    xg, yg = np.meshgrid(x, y, indexing="xy")
    zg = np.full_like(xg, 0.5 * float(dmin[2] + dmax[2]))
    return np.column_stack((xg.ravel(), yg.ravel(), zg.ravel()))


def test_build_and_query_runtime_guardrail() -> None:
    """Performance guardrail for octree interpolator build/query on representative data."""
    input_file = data_file("3d__var_4_n00005000.plt")
    assert input_file.exists(), f"Missing sample file: {input_file}"

    ds = Dataset.from_file(str(input_file))
    rng = np.random.default_rng(0)

    t0 = perf_counter()
    tree = Octree.from_ds(ds)
    interp = OctreeInterpolator(tree, np.asarray(ds[_RHO], dtype=float))
    build_s = perf_counter() - t0

    xyz = np.column_stack(
        [
            np.asarray(ds[XYZ_VARS[0]], dtype=float),
            np.asarray(ds[XYZ_VARS[1]], dtype=float),
            np.asarray(ds[XYZ_VARS[2]], dtype=float),
        ]
    )
    n_query = min(4000, xyz.shape[0])
    q = xyz[rng.choice(xyz.shape[0], size=n_query, replace=False)]

    t0 = perf_counter()
    vals = interp(q)
    query_s = perf_counter() - t0

    # Guardrails are intentionally loose to avoid flaky failures across machines,
    # while still catching major regressions.
    assert build_s < 60.0, f"octree build took {build_s:.3f}s (expected < 60s)"
    assert query_s < 10.0, f"octree query took {query_s:.3f}s for {n_query} points (expected < 10s)"
    assert np.isfinite(vals).any(), "expected at least some finite interpolation values"


@pytest.mark.parametrize(
    ("file_name", "tree_coord"),
    [
        ("3d__var_2_n00006003.plt", "xyz"),
        ("3d__var_1_n00000000.plt", "rpa"),
    ],
    ids=["cartesian_midplane", "spherical_midplane"],
)
def test_resampling_ramp_faster_than_scipy_linearnd(file_name: str, tree_coord: str) -> None:
    """Octree plane-resampling ramp should beat SciPy Delaunay+LinearND end to end."""
    input_file = data_file(file_name)
    assert input_file.exists(), f"Missing sample file: {input_file}"

    ds = Dataset.from_file(str(input_file))
    xyz = _xyz_points(ds)
    values = np.asarray(ds[_RHO], dtype=float)
    queries = [_xy_plane_queries(xyz, resolution=n) for n in _PLANE_RAMP]

    oct_query_times: list[float] = []
    oct_results: list[np.ndarray] = []
    t0 = perf_counter()
    oct_interp = OctreeInterpolator(Octree.from_ds(ds, tree_coord=tree_coord), values)
    for q in queries:
        t1 = perf_counter()
        out = np.asarray(
            oct_interp(q, query_coord="xyz", log_outside_domain=False),
            dtype=float,
        )
        oct_query_times.append(float(perf_counter() - t1))
        oct_results.append(out)
    oct_total_s = float(perf_counter() - t0)

    scipy_query_times: list[float] = []
    overlap_total = 0
    scipy_finite_total = 0
    oct_finite_total = 0
    t0 = perf_counter()
    tri = Delaunay(xyz)
    scipy_interp = LinearNDInterpolator(tri, values, fill_value=np.nan)
    for q, oct_vals in zip(queries, oct_results):
        t1 = perf_counter()
        scipy_vals = np.asarray(scipy_interp(q), dtype=float)
        scipy_query_times.append(float(perf_counter() - t1))
        oct_finite = np.isfinite(oct_vals)
        scipy_finite = np.isfinite(scipy_vals)
        oct_finite_total += int(np.count_nonzero(oct_finite))
        scipy_finite_total += int(np.count_nonzero(scipy_finite))
        overlap_total += int(np.count_nonzero(oct_finite & scipy_finite))
    scipy_total_s = float(perf_counter() - t0)

    assert oct_finite_total > 0, f"octree produced no finite values for {file_name}"
    assert scipy_finite_total > 0, f"SciPy produced no finite values for {file_name}"
    assert overlap_total > 0, f"No finite overlap between octree and SciPy for {file_name}"
    assert oct_total_s < 0.8 * scipy_total_s, (
        f"{file_name} end-to-end ramp too slow: "
        f"octree={oct_total_s:.3f}s vs scipy={scipy_total_s:.3f}s; "
        f"oct_query={oct_query_times}; scipy_query={scipy_query_times}; "
        f"overlap={overlap_total}"
    )
