from __future__ import annotations

import numpy as np
import pytest
from starwinds_readplt.dataset import Dataset

from batcamp import OctreeInterpolator
from batcamp import OctreeRayInterpolator
from batcamp import OctreeRayTracer
from sample_data_helper import data_file


def _notebook_ray_setup() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Private test helper: reproduce the diagnostic ray used in `examples/ray.ipynb`."""
    r_shell = 4.0
    p_start = r_shell * np.array([1.0, 0.0, 0.0], dtype=float)
    p_end = r_shell * np.array([-0.25, 0.96, 0.11], dtype=float)
    p_end = r_shell * (p_end / np.linalg.norm(p_end))
    origin = p_start
    direction = p_end - p_start
    t0 = 0.0
    t1 = float(np.linalg.norm(direction))
    return origin, direction, t0, t1


@pytest.mark.pooch
def test_ray_notebook_step3_trace_and_sample_contract_sc() -> None:
    """Notebook step-3 contract: traced segments and sampled values are valid."""
    ds = Dataset.from_file(str(data_file("3d__var_4_n00044000.plt")))
    interp = OctreeInterpolator(ds, ["Rho [g/cm^3]"])
    origin, direction, t0, t1 = _notebook_ray_setup()

    segments = OctreeRayTracer(interp.tree).trace(origin, direction, t0, t1)
    assert len(segments) > 0
    assert all(int(seg.cell_id) >= 0 for seg in segments)
    assert float(segments[0].t_enter) >= 0.0
    assert float(segments[-1].t_exit) <= t1 + 1e-12

    n_samples = 1200
    t_values, ray_values, cell_ids, _ = OctreeRayInterpolator(interp).sample(
        origin,
        direction,
        t0,
        t1,
        n_samples,
    )
    vals = np.asarray(ray_values, dtype=float).reshape(-1)
    cids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
    assert np.asarray(t_values, dtype=float).shape == (n_samples,)
    assert vals.shape == (n_samples,)
    assert cids.shape == (n_samples,)
    assert np.all(cids >= 0)
    assert np.all(np.isfinite(vals))


@pytest.mark.pooch
def test_ray_notebook_steps4_to6_piecewise_contract_sc() -> None:
    """Notebook step-4..6 contract: piecewise model matches samples and integral."""
    ds = Dataset.from_file(str(data_file("3d__var_4_n00044000.plt")))
    interp = OctreeInterpolator(ds, ["Rho [g/cm^3]"])
    ray = OctreeRayInterpolator(interp)
    origin, direction, t0, t1 = _notebook_ray_setup()

    n_samples = 1200
    t_values, ray_values, _ray_cell_ids, _ = ray.sample(origin, direction, t0, t1, n_samples)
    vals = np.asarray(ray_values, dtype=float).reshape(-1)
    pieces = ray.linear_pieces(origin, direction, t0, t1)
    assert len(pieces) > 0

    piece_vals = np.full_like(vals, np.nan)
    idx = 0
    for i, t in enumerate(t_values):
        while idx + 1 < len(pieces) and t > pieces[idx].t_end:
            idx += 1
        if idx < len(pieces):
            p = pieces[idx]
            if p.t_start - 1e-8 <= t <= p.t_end + 1e-8:
                piece_vals[i] = p.slope * t + p.intercept
    mask = np.isfinite(piece_vals) & np.isfinite(vals)
    assert np.any(mask)
    np.testing.assert_allclose(piece_vals[mask], vals[mask], atol=1e-18, rtol=1e-9)

    integral_exact = np.asarray(0.0, dtype=float)
    for p in pieces:
        a = float(p.t_start)
        b = float(p.t_end)
        integral_exact = integral_exact + 0.5 * p.slope * (b * b - a * a) + p.intercept * (b - a)
    integral_exact_f = float(np.asarray(integral_exact))
    integral_trap = float(np.trapezoid(vals, t_values))
    assert np.isfinite(integral_exact_f)
    assert np.isfinite(integral_trap)
    assert np.isclose(integral_exact_f, integral_trap, atol=1e-20, rtol=2e-2)
