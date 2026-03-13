from __future__ import annotations

import numpy as np
import pytest
from starwinds_readplt.dataset import Dataset

from batcamp import OctreeInterpolator
from batcamp import OctreeRayInterpolator
from batcamp import OctreeRayTracer
from sample_data_helper import data_file


def _diagnostic_ray_setup() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Private test helper: one deterministic SC diagnostic ray."""
    r_shell = 4.0
    p_start = r_shell * np.array([1.0, 0.0, 0.0], dtype=float)
    p_end = r_shell * np.array([-0.25, 0.96, 0.11], dtype=float)
    p_end = r_shell * (p_end / np.linalg.norm(p_end))
    origin = p_start
    direction = p_end - p_start
    t0 = 0.0
    t1 = float(np.linalg.norm(direction))
    return origin, direction, t0, t1


@pytest.fixture(scope="module")
def _sc_interp() -> OctreeInterpolator:
    """Private test helper: shared SC interpolator for ray contract tests."""
    ds = Dataset.from_file(str(data_file("3d__var_4_n00044000.plt")))
    return OctreeInterpolator(ds, ["Rho [g/cm^3]"])


def _integrate_rho2_with_rays(
    ray: OctreeRayInterpolator,
    interp: OctreeInterpolator,
    origins_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    t_end: float,
) -> np.ndarray:
    """Private test helper: integrate `rho^2` along rays using midpoint segments."""
    mids, weights, offsets = ray.adaptive_midpoint_rule(
        origins_xyz,
        direction_xyz,
        0.0,
        float(t_end),
        chunk_size=4096,
    )
    out = np.full(origins_xyz.shape[0], np.nan, dtype=float)
    if mids.shape[0] == 0:
        return out
    rho_mid = np.asarray(
        interp(mids, query_coord="xyz", log_outside_domain=False),
        dtype=float,
    ).reshape(-1)
    for i in range(out.shape[0]):
        s = int(offsets[i])
        e = int(offsets[i + 1])
        if e <= s:
            continue
        seg_rho = rho_mid[s:e]
        seg_w = weights[s:e]
        finite = np.isfinite(seg_rho)
        if not np.any(finite):
            continue
        out[i] = float(np.sum((seg_rho[finite] * seg_rho[finite]) * seg_w[finite]))
    return out


def _integrate_rho2_resample_baseline(
    interp: OctreeInterpolator,
    origins_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    t_end: float,
    *,
    n_steps: int = 256,
) -> np.ndarray:
    """Private test helper: baseline `rho^2` line integral from uniform resampling."""
    t = np.linspace(0.0, float(t_end), int(n_steps), dtype=float)
    dt = float(t[1] - t[0]) if t.size > 1 else float(t_end)
    d = np.asarray(direction_xyz, dtype=float).reshape(3)
    d = d / np.linalg.norm(d)
    q = origins_xyz[:, None, :] + t[None, :, None] * d[None, None, :]
    rho = np.asarray(
        interp(q.reshape(-1, 3), query_coord="xyz", log_outside_domain=False),
        dtype=float,
    ).reshape(origins_xyz.shape[0], t.size)
    rho2 = np.where(np.isfinite(rho), rho * rho, 0.0)
    return np.sum(rho2, axis=1) * dt


@pytest.mark.pooch
def test_trace_and_sample_sc(_sc_interp: OctreeInterpolator) -> None:
    """Ray contract: traced segments and sampled values are valid."""
    interp = _sc_interp
    origin, direction, t0, t1 = _diagnostic_ray_setup()

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
def test_piecewise_matches_samples_sc(_sc_interp: OctreeInterpolator) -> None:
    """Ray contract: piecewise model matches sampled values and integral."""
    interp = _sc_interp
    ray = OctreeRayInterpolator(interp)
    origin, direction, t0, t1 = _diagnostic_ray_setup()

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


@pytest.mark.parametrize(
    "file_name",
    [
        pytest.param("3d__var_4_n00044000.plt", id="sc", marks=pytest.mark.pooch),
        pytest.param("3d__var_4_n00005000.plt", id="ih", marks=pytest.mark.pooch),
    ],
)
@pytest.mark.slow
def test_rho2_matches_resample_baseline(file_name: str) -> None:
    """Ray contract: outside-start rays that hit data must integrate correctly."""
    ds = Dataset.from_file(str(data_file(file_name)))
    interp = OctreeInterpolator(ds, ["Rho [g/cm^3]"])
    ray = OctreeRayInterpolator(interp)

    dmin, dmax = interp.tree.domain_bounds(coord="xyz")
    xmin, xmax = float(dmin[0]), float(dmax[0])
    ymin, ymax = float(dmin[1]), float(dmax[1])
    zmin, zmax = float(dmin[2]), float(dmax[2])

    ny, nz = 3, 3
    yg, zg = np.meshgrid(np.linspace(ymin, ymax, ny), np.linspace(zmin, zmax, nz), indexing="xy")
    x0 = xmin - 0.05 * (xmax - xmin)
    origins = np.column_stack((np.full(yg.size, x0, dtype=float), yg.ravel(), zg.ravel()))
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    t_end = (xmax - x0) * 0.999999

    base_vals = _integrate_rho2_resample_baseline(interp, origins, direction, t_end, n_steps=96)
    candidates = np.where(np.isfinite(base_vals) & (base_vals > 0.0))[0]
    assert candidates.size > 0
    idx = int(candidates[0])
    ray_vals = _integrate_rho2_with_rays(ray, interp, origins[idx : idx + 1], direction, t_end)
    ray_val = float(ray_vals[0])
    base_val = float(base_vals[idx])

    assert np.isfinite(ray_val)
    assert ray_val > 0.0
    rel = abs(ray_val - base_val) / max(abs(base_val), 1e-30)
    assert rel < 1e-8
