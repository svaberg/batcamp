from __future__ import annotations

import math

import numpy as np
import pytest

from batcamp import Octree
from batcamp import OctreeBuilder
from batcamp import OctreeInterpolator
from batcamp import OctreeRayInterpolator
from batcamp import OctreeRayTracer
from fake_dataset import FakeDataset as _FakeDataset
from fake_dataset import build_cartesian_hex_mesh as _build_cartesian_hex_mesh
from fake_dataset import build_spherical_hex_mesh as _build_spherical_hex_mesh


# ==============================================================================
# ALGORITHMS (NON-TEST HELPERS)
# ==============================================================================


def _slab_entry_xyz(
    origin: np.ndarray,
    direction: np.ndarray,
    bmin: np.ndarray,
    bmax: np.ndarray,
) -> float | None:
    """Return first forward slab-entry time into an XYZ box, or None on miss."""
    t_enter = -np.inf
    t_exit = np.inf
    for i in range(3):
        oi = float(origin[i])
        di = float(direction[i])
        lo = float(bmin[i])
        hi = float(bmax[i])
        if abs(di) <= 1.0e-15:
            if oi < lo or oi > hi:
                return None
            continue
        t1 = (lo - oi) / di
        t2 = (hi - oi) / di
        t_axis_min = min(t1, t2)
        t_axis_max = max(t1, t2)
        t_enter = max(t_enter, t_axis_min)
        t_exit = min(t_exit, t_axis_max)
        if t_exit < t_enter:
            return None
    if t_exit <= 0.0:
        return None
    return float(max(t_enter, 0.0))


def _slab_entry_rpa(origin: np.ndarray, direction: np.ndarray, radius: float) -> float | None:
    """Return first forward intersection time with spherical outer radius."""
    b = float(np.dot(origin, direction))
    c = float(np.dot(origin, origin) - radius * radius)
    disc = b * b - c
    if disc < 0.0:
        return None
    s = math.sqrt(max(0.0, disc))
    t1 = -b - s
    t2 = -b + s
    candidates = [t for t in (t1, t2) if t >= 0.0]
    if not candidates:
        return None
    return float(min(candidates))


def _containing_cell_via_contains(tree: Octree, point_xyz: np.ndarray) -> int | None:
    """Find the first cell that contains `point_xyz`, or None."""
    for cid in range(int(tree.cell_count)):
        if tree.contains_cell(int(cid), point_xyz, coord="xyz", tol=1.0e-10):
            return int(cid)
    return None


def _first_cell_intersection_algorithm(tree: Octree, origin: np.ndarray, direction: np.ndarray) -> tuple[int, float] | None:
    """Find the first ray-hit cell id and its entry parameter t."""
    # If origin is already in a leaf, first intersection is at t=0.
    cid0 = _containing_cell_via_contains(tree, origin)
    if cid0 is not None:
        return int(cid0), 0.0

    # Otherwise find first forward slab entry time.
    if str(tree.tree_coord) == "xyz":
        bmin, bmax = tree.domain_bounds(coord="xyz")
        t_enter = _slab_entry_xyz(
            origin,
            direction,
            np.asarray(bmin, dtype=float),
            np.asarray(bmax, dtype=float),
        )
    elif str(tree.tree_coord) == "rpa":
        _r_lo, r_hi = tree.domain_bounds(coord="rpa")
        t_enter = _slab_entry_rpa(origin, direction, float(r_hi[0]))
    else:
        return None

    if t_enter is None:
        return None
    p_inside = origin + (float(t_enter) + 1.0e-8) * direction
    cid = _containing_cell_via_contains(tree, np.asarray(p_inside, dtype=float))
    if cid is None:
        return None
    return int(cid), float(t_enter)


def _exit_point_from_cell_algorithm(
    tree: Octree,
    cell_id: int,
    origin: np.ndarray,
    direction: np.ndarray,
    t_in: float,
    *,
    t_cap: float,
    tol: float = 1.0e-10,
    max_expand: int = 64,
    max_bisect: int = 64,
) -> tuple[np.ndarray, float] | None:
    """Find the ray exit point/time for one cell using contains+bisect."""
    p_in = origin + float(t_in) * direction
    if not tree.contains_cell(int(cell_id), p_in, coord="xyz", tol=tol):
        return None

    bmin, bmax = tree.cell_bounds(int(cell_id), coord="xyz")
    span = float(np.linalg.norm(np.asarray(bmax, dtype=float) - np.asarray(bmin, dtype=float)))
    dt = span if span > 0.0 else 1.0e-6

    t_lo = float(t_in)
    t_hi = min(float(t_cap), t_lo + dt)
    p_hi = origin + t_hi * direction
    inside_hi = tree.contains_cell(int(cell_id), p_hi, coord="xyz", tol=tol)

    n_expand = 0
    while inside_hi and t_hi < float(t_cap) and n_expand < int(max_expand):
        t_lo = t_hi
        dt *= 2.0
        t_hi = min(float(t_cap), float(t_in) + dt)
        p_hi = origin + t_hi * direction
        inside_hi = tree.contains_cell(int(cell_id), p_hi, coord="xyz", tol=tol)
        n_expand += 1

    if inside_hi:
        return None

    lo = t_lo
    hi = t_hi
    for _ in range(int(max_bisect)):
        mid = 0.5 * (lo + hi)
        p_mid = origin + mid * direction
        if tree.contains_cell(int(cell_id), p_mid, coord="xyz", tol=tol):
            lo = mid
        else:
            hi = mid
        if (hi - lo) <= 1.0e-12:
            break

    t_out = float(lo)
    p_out = origin + t_out * direction
    return np.asarray(p_out, dtype=float), t_out


def _next_cell_from_exit_algorithm(
    tree: Octree,
    current_cell_id: int,
    origin: np.ndarray,
    direction: np.ndarray,
    t_exit: float,
    *,
    eps: float = 1.0e-8,
) -> int | None:
    """Epsilon-step past exit and return next lookup cell id, if any."""
    p_next = origin + (float(t_exit) + float(eps)) * direction
    hit = tree.lookup_point(np.asarray(p_next, dtype=float), coord="xyz")
    if hit is None:
        return None
    cid = int(hit.cell_id)
    if cid == int(current_cell_id):
        return None
    return cid


# ==============================================================================
# TEST SETUP AND TESTS
# ==============================================================================

# ------------------------------------------------------------------------------
# PRIVATE TEST HELPERS
# ------------------------------------------------------------------------------
def _build_cartesian_tree() -> tuple[Octree, np.ndarray, np.ndarray, float]:
    """Private test helper: return a tiny Cartesian tree and a forward ray."""
    points, corners = _build_cartesian_hex_mesh(
        x_edges=np.array([0.0, 1.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        z_edges=np.array([0.0, 1.0], dtype=float),
    )
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={
            Octree.X_VAR: points[:, 0],
            Octree.Y_VAR: points[:, 1],
            Octree.Z_VAR: points[:, 2],
        },
    )
    tree = OctreeBuilder().build(ds, tree_coord="xyz")
    origin = np.array([-1.0, 0.25, 0.25], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    return tree, origin, direction, 4.0


def _build_spherical_tree() -> tuple[Octree, np.ndarray, np.ndarray, float]:
    """Private test helper: return a tiny spherical tree and an inward ray."""
    points, corners = _build_spherical_hex_mesh(
        nr=1,
        ntheta=2,
        nphi=4,
        r_min=1.0,
        r_max=2.0,
    )
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={
            Octree.X_VAR: points[:, 0],
            Octree.Y_VAR: points[:, 1],
            Octree.Z_VAR: points[:, 2],
        },
    )
    tree = OctreeBuilder().build(ds, tree_coord="rpa")
    origin = np.array([3.0, 0.4, 0.2], dtype=float)
    direction = -origin / np.linalg.norm(origin)
    return tree, origin, direction, 6.0


@pytest.fixture(params=["xyz", "rpa"], ids=["cartesian_tree", "spherical_tree"])
def tree_case(request):
    """Provide both Cartesian and spherical tree/ray cases."""
    if request.param == "xyz":
        return _build_cartesian_tree()
    return _build_spherical_tree()


def test_first_slab_hit_cell_for_xyz_and_rpa(tree_case) -> None:
    tree, origin, direction, _t_end = tree_case
    expected = _first_cell_intersection_algorithm(tree, origin, direction)
    assert expected is not None
    cid, t_enter = expected
    assert cid >= 0
    assert t_enter >= 0.0

    # Point just after first entry must be in the reported first cell.
    p_after = origin + (t_enter + 1.0e-8) * direction
    assert tree.contains_cell(int(cid), p_after, coord="xyz", tol=1.0e-10)

    # If there is room before entry, point just before must not be in that cell.
    if t_enter > 1.0e-8:
        p_before = origin + (t_enter - 1.0e-8) * direction
        assert not tree.contains_cell(int(cid), p_before, coord="xyz", tol=1.0e-10)


def test_exit_point_from_cell_for_xyz_and_rpa(tree_case) -> None:
    tree, origin, direction, t_end = tree_case
    first = _first_cell_intersection_algorithm(tree, origin, direction)
    assert first is not None
    cid, t_enter = first

    out = _exit_point_from_cell_algorithm(
        tree,
        int(cid),
        origin,
        direction,
        float(t_enter),
        t_cap=float(t_end),
    )
    assert out is not None
    p_exit, t_exit = out

    assert float(t_exit) >= float(t_enter)
    assert tree.contains_cell(int(cid), np.asarray(p_exit, dtype=float), coord="xyz", tol=1.0e-8)

    # Small forward step from exit should no longer be in the same cell.
    p_after_exit = origin + (float(t_exit) + 1.0e-8) * direction
    assert not tree.contains_cell(int(cid), np.asarray(p_after_exit, dtype=float), coord="xyz", tol=1.0e-10)


def test_next_cell_from_exit_for_xyz_and_rpa(tree_case) -> None:
    tree, origin, direction, t_end = tree_case
    first = _first_cell_intersection_algorithm(tree, origin, direction)
    assert first is not None
    cid, t_enter = first

    out = _exit_point_from_cell_algorithm(
        tree,
        int(cid),
        origin,
        direction,
        float(t_enter),
        t_cap=float(t_end),
    )
    assert out is not None
    _p_exit, t_exit = out

    got_next = _next_cell_from_exit_algorithm(
        tree,
        int(cid),
        origin,
        direction,
        float(t_exit),
        eps=1.0e-8,
    )

    p_next = origin + (float(t_exit) + 1.0e-8) * direction
    hit = tree.lookup_point(np.asarray(p_next, dtype=float), coord="xyz")
    expected_next = None
    if hit is not None and int(hit.cell_id) != int(cid):
        expected_next = int(hit.cell_id)

    assert got_next == expected_next


def test_integral_finds_shell_after_outside_start() -> None:
    """Ray regression: spherical shell rays must not return empty from inside hole."""
    points, corners = _build_spherical_hex_mesh(
        nr=1,
        ntheta=2,
        nphi=4,
        r_min=1.0,
        r_max=2.0,
    )
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    scalar = 2.0 * x - 0.5 * y + 0.25 * z + 1.0
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={
            Octree.X_VAR: x,
            Octree.Y_VAR: y,
            Octree.Z_VAR: z,
            "Scalar": scalar,
        },
    )
    tree = OctreeBuilder().build(ds, tree_coord="rpa")
    interp = OctreeInterpolator(tree, ["Scalar"])
    ray = OctreeRayInterpolator(interp)

    origin = np.array([0.0, 0.0, 0.0], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    t0, t1 = 0.0, 3.0

    cell_ids, _t_enter, _t_exit = OctreeRayTracer(tree).trace(origin, direction, t0, t1)
    assert cell_ids.size > 0

    ray_int = float(ray.integrate_field_along_rays(origin[None, :], direction, t0, t1)[0])
    t = np.linspace(t0, t1, 512, dtype=float)
    q = origin[None, :] + t[:, None] * direction[None, :]
    vals = np.asarray(interp(q, query_coord="xyz", log_outside_domain=False), dtype=float).reshape(-1)
    baseline = float(np.trapezoid(np.where(np.isfinite(vals), vals, 0.0), t))

    assert np.isfinite(ray_int)
    assert np.isfinite(baseline)
    assert baseline > 0.0
    rel = abs(ray_int - baseline) / max(abs(baseline), 1e-30)
    assert rel < 5.0e-2
