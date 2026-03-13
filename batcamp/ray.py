#!/usr/bin/env python3
"""Ray traversal and ray-based integration helpers for octrees."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import NamedTuple

from numba import njit
from numba import prange
import numpy as np

from .cartesian import CartesianLookupKernelState
from .cartesian import _lookup_xyz_cell_id_kernel
from .interpolator import CartesianInterpKernelState
from .interpolator import SphericalInterpKernelState
from .interpolator import _trilinear_from_cell
from .interpolator import _trilinear_from_cell_rpa
from .octree import Octree
from .spherical import SphericalLookupKernelState
from .spherical import _lookup_rpa_cell_id_kernel
from .spherical import _xyz_to_rpa_components
from .topological import TopologicalKernelState
from .topological import build_topological_neighborhood

if TYPE_CHECKING:
    from .interpolator import OctreeInterpolator


_TRACE_CONTAIN_TOL = 1e-8
_DEFAULT_TRACE_BOUNDARY_TOL = 1e-9
_DEFAULT_TRACE_MAX_STEPS = 100000
_MAX_TOPOLOGY_CANDIDATES = 128
_RPA_FACE_NODE_IDS = (
    (0, 2, 6, 4),
    (1, 3, 7, 5),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (0, 1, 3, 2),
    (4, 5, 7, 6),
)
_FACE_TRIPLES = (
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 3),
    (1, 2, 3),
)


class RpaHexaKernelState(NamedTuple):
    """Plane representation of spherical-tree cells as Cartesian hexahedra."""

    face_normals: np.ndarray
    face_offsets: np.ndarray
    face_valid: np.ndarray


def _build_rpa_hexa_kernel_state(tree: Octree) -> RpaHexaKernelState:
    """Approximate spherical cells as convex Cartesian hexahedra with planar faces."""
    tree._require_lookup()
    corners = np.asarray(getattr(tree, "_corners"), dtype=np.int64)
    points = np.asarray(getattr(tree, "_points"), dtype=float)
    centers = np.asarray(getattr(tree, "_cell_centers"), dtype=float)
    n_cells = int(corners.shape[0])

    face_normals = np.zeros((n_cells, 6, 3), dtype=np.float64)
    face_offsets = np.zeros((n_cells, 6), dtype=np.float64)
    face_valid = np.zeros((n_cells, 6), dtype=np.bool_)
    tiny = 1.0e-24

    for cid in range(n_cells):
        cell_pts = np.asarray(points[corners[cid]], dtype=float)
        center = np.asarray(centers[cid], dtype=float)
        for face, node_ids in enumerate(_RPA_FACE_NODE_IDS):
            face_pts = np.asarray(cell_pts[list(node_ids)], dtype=float)
            best_norm2 = -1.0
            best_normal = np.zeros(3, dtype=float)
            best_anchor = np.zeros(3, dtype=float)
            for ia, ib, ic in _FACE_TRIPLES:
                a = face_pts[ia]
                b = face_pts[ib]
                c = face_pts[ic]
                normal = np.cross(b - a, c - a)
                norm2 = float(np.dot(normal, normal))
                if norm2 > best_norm2:
                    best_norm2 = norm2
                    best_normal = normal
                    best_anchor = a
            if best_norm2 <= tiny:
                continue
            best_normal = best_normal / math.sqrt(best_norm2)
            if float(np.dot(best_normal, center - best_anchor)) > 0.0:
                best_normal = -best_normal
            face_normals[cid, face, :] = best_normal
            face_offsets[cid, face] = float(np.dot(best_normal, best_anchor))
            face_valid[cid, face] = True

    return RpaHexaKernelState(
        face_normals=face_normals,
        face_offsets=face_offsets,
        face_valid=face_valid,
    )


@njit(cache=True)
def _contains_xyz_from_state(
    cid: int,
    x: float,
    y: float,
    z: float,
    lookup_state: CartesianLookupKernelState,
    tol: float = _TRACE_CONTAIN_TOL,
) -> bool:
    """Return whether one Cartesian point lies inside one cell."""
    if x < (lookup_state.cell_x_min[cid] - tol) or x > (lookup_state.cell_x_max[cid] + tol):
        return False
    if y < (lookup_state.cell_y_min[cid] - tol) or y > (lookup_state.cell_y_max[cid] + tol):
        return False
    if z < (lookup_state.cell_z_min[cid] - tol) or z > (lookup_state.cell_z_max[cid] + tol):
        return False
    return True


@njit(cache=True)
def _contains_rpa_hexa_from_xyz(
    cid: int,
    x: float,
    y: float,
    z: float,
    hexa_state: RpaHexaKernelState,
    tol: float = _TRACE_CONTAIN_TOL,
) -> bool:
    """Return whether one Cartesian point lies inside the planar hexahedron model."""
    for face in range(6):
        if not hexa_state.face_valid[cid, face]:
            continue
        nx = hexa_state.face_normals[cid, face, 0]
        ny = hexa_state.face_normals[cid, face, 1]
        nz = hexa_state.face_normals[cid, face, 2]
        offset = hexa_state.face_offsets[cid, face]
        signed = nx * x + ny * y + nz * z - offset
        if signed > tol:
            return False
    return True


@njit(cache=True)
def _rpa_hexa_exit_dt_and_mask(
    cid: int,
    x: float,
    y: float,
    z: float,
    d0: float,
    d1: float,
    d2: float,
    hexa_state: RpaHexaKernelState,
    abs_eps: float,
) -> tuple[float, int]:
    """Return forward exit distance and exited-face mask for the hexahedron model."""
    best_dt = np.inf
    face_mask = 0
    dir_eps = 1.0e-15
    tie_tol = max(4.0 * abs_eps, 1.0e-12)
    for face in range(6):
        if not hexa_state.face_valid[cid, face]:
            continue
        nx = hexa_state.face_normals[cid, face, 0]
        ny = hexa_state.face_normals[cid, face, 1]
        nz = hexa_state.face_normals[cid, face, 2]
        offset = hexa_state.face_offsets[cid, face]
        signed = nx * x + ny * y + nz * z - offset
        if signed > tie_tol:
            return np.inf, 0
        if signed > 0.0:
            signed = 0.0
        denom = nx * d0 + ny * d1 + nz * d2
        if denom <= dir_eps:
            continue
        dt = -signed / denom
        if dt <= abs_eps:
            continue
        if dt < (best_dt - tie_tol):
            best_dt = dt
            face_mask = 1 << face
        elif abs(dt - best_dt) <= tie_tol:
            face_mask |= 1 << face
    return best_dt, face_mask


@njit(cache=True)
def _append_unique_node(nodes: np.ndarray, n_nodes: int, node_id: int) -> int:
    """Append `node_id` if absent; return updated count."""
    for i in range(n_nodes):
        if int(nodes[i]) == int(node_id):
            return n_nodes
    if n_nodes < int(nodes.shape[0]):
        nodes[n_nodes] = int(node_id)
        return n_nodes + 1
    return n_nodes


@njit(cache=True)
def _expand_topology_nodes_for_face(
    src_nodes: np.ndarray,
    n_src: int,
    face: int,
    topo_state: TopologicalKernelState,
    dst_nodes: np.ndarray,
) -> int:
    """Expand one candidate node set across one face."""
    n_dst = 0
    for i in range(n_src):
        node_id = int(src_nodes[i])
        slot = node_id * 6 + int(face)
        start = int(topo_state.face_offsets[slot])
        end = int(topo_state.face_offsets[slot + 1])
        for pos in range(start, end):
            n_dst = _append_unique_node(dst_nodes, n_dst, int(topo_state.face_neighbors[pos]))
    return n_dst


@njit(cache=True)
def _candidate_nodes_after_exit(
    current_node_id: int,
    face_mask: int,
    topo_state: TopologicalKernelState,
    work0: np.ndarray,
    work1: np.ndarray,
) -> tuple[int, int]:
    """Follow all exited faces and return `(n_candidates, active_buffer)`."""
    work0[0] = int(current_node_id)
    n_active = 1
    active = 0
    for face in range(6):
        if (int(face_mask) & (1 << face)) == 0:
            continue
        if active == 0:
            n_next = _expand_topology_nodes_for_face(work0, n_active, face, topo_state, work1)
            active = 1
        else:
            n_next = _expand_topology_nodes_for_face(work1, n_active, face, topo_state, work0)
            active = 0
        n_active = int(n_next)
        if n_active <= 0:
            return 0, active
    return n_active, active


@njit(cache=True)
def _select_next_xyz_cell_from_topology(
    current_node_id: int,
    face_mask: int,
    x_next: float,
    y_next: float,
    z_next: float,
    topo_state: TopologicalKernelState,
    lookup_state: CartesianLookupKernelState,
    work0: np.ndarray,
    work1: np.ndarray,
) -> tuple[int, int]:
    """Choose next Cartesian `(node_id, cell_id)` from topology candidates."""
    n_candidates, active = _candidate_nodes_after_exit(current_node_id, face_mask, topo_state, work0, work1)
    if n_candidates <= 0:
        return -1, -1
    nodes = work0 if active == 0 else work1
    for i in range(n_candidates):
        node_id = int(nodes[i])
        cid = int(topo_state.node_cell_ids[node_id])
        if _contains_xyz_from_state(cid, x_next, y_next, z_next, lookup_state):
            return node_id, cid
    return -1, -1


@njit(cache=True)
def _select_next_rpa_cell_from_topology(
    current_node_id: int,
    face_mask: int,
    x_next: float,
    y_next: float,
    z_next: float,
    topo_state: TopologicalKernelState,
    hexa_state: RpaHexaKernelState,
    work0: np.ndarray,
    work1: np.ndarray,
) -> tuple[int, int]:
    """Choose next spherical `(node_id, cell_id)` from topology candidates."""
    n_candidates, active = _candidate_nodes_after_exit(current_node_id, face_mask, topo_state, work0, work1)
    if n_candidates <= 0:
        return -1, -1
    nodes = work0 if active == 0 else work1
    for i in range(n_candidates):
        node_id = int(nodes[i])
        cid = int(topo_state.node_cell_ids[node_id])
        if _contains_rpa_hexa_from_xyz(cid, x_next, y_next, z_next, hexa_state):
            return node_id, cid
    return -1, -1


@njit(cache=True)
def _xyz_exit_face_mask(
    dt_exit: float,
    tx: float,
    ty: float,
    tz: float,
    d0: float,
    d1: float,
    d2: float,
    abs_eps: float,
) -> int:
    """Return Cartesian exited-face bitmask for the current segment."""
    face_mask = 0
    tol = max(4.0 * abs_eps, 1.0e-12)
    if math.isfinite(tx) and abs(tx - dt_exit) <= tol:
        face_mask |= 1 << (1 if d0 > 0.0 else 0)
    if math.isfinite(ty) and abs(ty - dt_exit) <= tol:
        face_mask |= 1 << (3 if d1 > 0.0 else 2)
    if math.isfinite(tz) and abs(tz - dt_exit) <= tol:
        face_mask |= 1 << (5 if d2 > 0.0 else 4)
    return face_mask


@njit(cache=True)
def _forward_face_exit_dt(
    coord: float,
    direction_component: float,
    coord_min: float,
    coord_max: float,
    abs_eps: float,
    dir_eps: float,
) -> float:
    """Return forward distance to next Cartesian face along one axis."""
    if abs(direction_component) <= dir_eps:
        return np.inf
    if direction_component > 0.0:
        dt = (coord_max - coord) / direction_component
    else:
        dt = (coord_min - coord) / direction_component
    if dt > abs_eps:
        return dt
    return 0.0


@njit(cache=True)
def _clip_ray_interval_to_slab(
    origin_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    slab_min_xyz: np.ndarray,
    slab_max_xyz: np.ndarray,
    tol: float,
) -> tuple[bool, float, float]:
    """Clip ray interval `[t_start, t_end]` against an axis-aligned slab."""
    t_near = t_start
    t_far = t_end
    dir_eps = 1e-15
    for axis in range(3):
        o = origin_xyz[axis]
        d = direction_xyz_unit[axis]
        bmin = slab_min_xyz[axis] - tol
        bmax = slab_max_xyz[axis] + tol
        if abs(d) <= dir_eps:
            if o < bmin or o > bmax:
                return False, t_start, t_start
            continue
        t0 = (bmin - o) / d
        t1 = (bmax - o) / d
        if t0 > t1:
            tmp = t0
            t0 = t1
            t1 = tmp
        if t0 > t_near:
            t_near = t0
        if t1 < t_far:
            t_far = t1
        if t_far <= t_near:
            return False, t_near, t_far
    return True, t_near, t_far


@njit(cache=True)
def _clip_ray_interval_to_sphere(
    origin_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    radius: float,
    tol: float,
) -> tuple[bool, float, float]:
    """Clip ray interval `[t_start, t_end]` against sphere at origin."""
    if radius <= 0.0:
        return False, t_start, t_start

    ox = origin_xyz[0]
    oy = origin_xyz[1]
    oz = origin_xyz[2]
    dx = direction_xyz_unit[0]
    dy = direction_xyz_unit[1]
    dz = direction_xyz_unit[2]

    b = ox * dx + oy * dy + oz * dz
    c = (ox * ox + oy * oy + oz * oz) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return False, t_start, t_start

    s = math.sqrt(max(0.0, disc))
    ta = -b - s
    tb = -b + s
    if ta > tb:
        tmp = ta
        ta = tb
        tb = tmp

    t_near = t_start
    t_far = t_end
    if (ta - tol) > t_near:
        t_near = ta - tol
    if (tb + tol) < t_far:
        t_far = tb + tol
    if t_far <= t_near:
        return False, t_near, t_far
    return True, t_near, t_far


@njit(cache=True)
def _seek_first_cell_xyz(
    origin_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    abs_eps: float,
    lookup_state: CartesianLookupKernelState,
) -> tuple[bool, float, int, float, float, float]:
    """Seek first `t` where ray enters a valid Cartesian cell."""
    t = t_start
    x = origin_xyz[0] + t * direction_xyz_unit[0]
    y = origin_xyz[1] + t * direction_xyz_unit[1]
    z = origin_xyz[2] + t * direction_xyz_unit[2]
    cid = _lookup_xyz_cell_id_kernel(x, y, z, lookup_state, -1)
    if cid >= 0:
        return True, t, cid, x, y, z

    lo = t
    hi = t
    dt = max(abs_eps, 1e-6 * (1.0 + abs(t_end - t_start)))
    cid_hi = -1
    while hi < t_end:
        hi = hi + dt
        if hi > t_end:
            hi = t_end
        xh = origin_xyz[0] + hi * direction_xyz_unit[0]
        yh = origin_xyz[1] + hi * direction_xyz_unit[1]
        zh = origin_xyz[2] + hi * direction_xyz_unit[2]
        cid_hi = _lookup_xyz_cell_id_kernel(xh, yh, zh, lookup_state, -1)
        if cid_hi >= 0:
            break
        if hi >= t_end:
            return False, t_start, -1, x, y, z
        dt = dt * 2.0
    if cid_hi < 0:
        return False, t_start, -1, x, y, z

    lo_out = lo
    hi_in = hi
    cid_in = cid_hi
    for _ in range(48):
        mid = 0.5 * (lo_out + hi_in)
        xm = origin_xyz[0] + mid * direction_xyz_unit[0]
        ym = origin_xyz[1] + mid * direction_xyz_unit[1]
        zm = origin_xyz[2] + mid * direction_xyz_unit[2]
        cid_mid = _lookup_xyz_cell_id_kernel(xm, ym, zm, lookup_state, cid_in)
        if cid_mid >= 0:
            hi_in = mid
            cid_in = cid_mid
        else:
            lo_out = mid
        if (hi_in - lo_out) <= abs_eps:
            break

    t = hi_in
    if t < t_end:
        t = t + abs_eps
        if t > t_end:
            t = t_end
    x = origin_xyz[0] + t * direction_xyz_unit[0]
    y = origin_xyz[1] + t * direction_xyz_unit[1]
    z = origin_xyz[2] + t * direction_xyz_unit[2]
    cid = _lookup_xyz_cell_id_kernel(x, y, z, lookup_state, cid_in)
    if cid < 0:
        t = hi_in
        x = origin_xyz[0] + t * direction_xyz_unit[0]
        y = origin_xyz[1] + t * direction_xyz_unit[1]
        z = origin_xyz[2] + t * direction_xyz_unit[2]
        cid = _lookup_xyz_cell_id_kernel(x, y, z, lookup_state, cid_in)
    if cid < 0:
        return False, t_start, -1, x, y, z
    return True, t, cid, x, y, z


@njit(cache=True)
def _seek_first_cell_rpa(
    origin_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    abs_eps: float,
    lookup_state: SphericalLookupKernelState,
    hexa_state: RpaHexaKernelState,
) -> tuple[bool, float, int, float, float, float]:
    """Seek first `t` where ray enters a valid spherical hexahedron cell."""
    t = t_start
    x = origin_xyz[0] + t * direction_xyz_unit[0]
    y = origin_xyz[1] + t * direction_xyz_unit[1]
    z = origin_xyz[2] + t * direction_xyz_unit[2]
    r, polar, azimuth = _xyz_to_rpa_components(x, y, z)
    cid = _lookup_rpa_cell_id_kernel(r, polar, azimuth, lookup_state, -1)
    if cid >= 0 and _contains_rpa_hexa_from_xyz(cid, x, y, z, hexa_state):
        return True, t, cid, x, y, z

    lo = t
    hi = t
    dt = max(abs_eps, 1e-6 * (1.0 + abs(t_end - t_start)))
    cid_hi = -1
    while hi < t_end:
        hi = hi + dt
        if hi > t_end:
            hi = t_end
        xh = origin_xyz[0] + hi * direction_xyz_unit[0]
        yh = origin_xyz[1] + hi * direction_xyz_unit[1]
        zh = origin_xyz[2] + hi * direction_xyz_unit[2]
        r_hi, polar_hi, azimuth_hi = _xyz_to_rpa_components(xh, yh, zh)
        cid_hi = _lookup_rpa_cell_id_kernel(r_hi, polar_hi, azimuth_hi, lookup_state, -1)
        if cid_hi >= 0 and _contains_rpa_hexa_from_xyz(cid_hi, xh, yh, zh, hexa_state):
            break
        if hi >= t_end:
            return False, t_start, -1, x, y, z
        dt = dt * 2.0
    if cid_hi < 0:
        return False, t_start, -1, x, y, z

    lo_out = lo
    hi_in = hi
    cid_in = cid_hi
    for _ in range(48):
        mid = 0.5 * (lo_out + hi_in)
        xm = origin_xyz[0] + mid * direction_xyz_unit[0]
        ym = origin_xyz[1] + mid * direction_xyz_unit[1]
        zm = origin_xyz[2] + mid * direction_xyz_unit[2]
        r_mid, polar_mid, azimuth_mid = _xyz_to_rpa_components(xm, ym, zm)
        cid_mid = _lookup_rpa_cell_id_kernel(r_mid, polar_mid, azimuth_mid, lookup_state, cid_in)
        if cid_mid >= 0 and _contains_rpa_hexa_from_xyz(cid_mid, xm, ym, zm, hexa_state):
            hi_in = mid
            cid_in = cid_mid
        else:
            lo_out = mid
        if (hi_in - lo_out) <= abs_eps:
            break

    t = hi_in
    if t < t_end:
        t = t + abs_eps
        if t > t_end:
            t = t_end
    x = origin_xyz[0] + t * direction_xyz_unit[0]
    y = origin_xyz[1] + t * direction_xyz_unit[1]
    z = origin_xyz[2] + t * direction_xyz_unit[2]
    r, polar, azimuth = _xyz_to_rpa_components(x, y, z)
    cid = _lookup_rpa_cell_id_kernel(r, polar, azimuth, lookup_state, cid_in)
    if cid < 0 or not _contains_rpa_hexa_from_xyz(cid, x, y, z, hexa_state):
        t = hi_in
        x = origin_xyz[0] + t * direction_xyz_unit[0]
        y = origin_xyz[1] + t * direction_xyz_unit[1]
        z = origin_xyz[2] + t * direction_xyz_unit[2]
        r, polar, azimuth = _xyz_to_rpa_components(x, y, z)
        cid = _lookup_rpa_cell_id_kernel(r, polar, azimuth, lookup_state, cid_in)
    if cid < 0 or not _contains_rpa_hexa_from_xyz(cid, x, y, z, hexa_state):
        return False, t_start, -1, x, y, z
    return True, t, cid, x, y, z


@njit(cache=True)
def _trace_segments_xyz_kernel(
    origin_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    boundary_tol: float,
    lookup_state: CartesianLookupKernelState,
    topo_state: TopologicalKernelState,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Trace one Cartesian ray; return `(n_seg, cell_ids, t_enter, t_exit)`."""
    cell_ids = np.empty(max_steps, dtype=np.int64)
    enters = np.empty(max_steps, dtype=np.float64)
    exits = np.empty(max_steps, dtype=np.float64)

    if t_end <= t_start or max_steps <= 0:
        return 0, cell_ids, enters, exits

    clipped, t0_clip, t1_clip = _clip_ray_interval_to_slab(
        origin_xyz,
        direction_xyz_unit,
        t_start,
        t_end,
        lookup_state.xyz_min,
        lookup_state.xyz_max,
        boundary_tol,
    )
    if not clipped or t1_clip <= t0_clip:
        return 0, cell_ids, enters, exits

    abs_eps = max(boundary_tol * (1.0 + abs(t1_clip - t0_clip)), 1e-12)
    t = t0_clip
    if t0_clip > t_start:
        t = min(t1_clip, t0_clip + abs_eps)
    if t >= t1_clip:
        return 0, cell_ids, enters, exits

    t_end = t1_clip
    found, t, cid, x, y, z = _seek_first_cell_xyz(
        origin_xyz,
        direction_xyz_unit,
        t,
        t_end,
        abs_eps,
        lookup_state,
    )
    if not found or cid < 0:
        return 0, cell_ids, enters, exits

    current_node = int(topo_state.cell_to_node_id[cid])
    if current_node < 0:
        return 0, cell_ids, enters, exits

    d0 = direction_xyz_unit[0]
    d1 = direction_xyz_unit[1]
    d2 = direction_xyz_unit[2]
    dir_eps = 1e-15
    candidate_nodes0 = np.empty(_MAX_TOPOLOGY_CANDIDATES, dtype=np.int64)
    candidate_nodes1 = np.empty(_MAX_TOPOLOGY_CANDIDATES, dtype=np.int64)
    n_seg = 0
    for _ in range(max_steps):
        if t >= (t_end - abs_eps):
            break

        if not _contains_xyz_from_state(cid, x, y, z, lookup_state):
            break

        tx = _forward_face_exit_dt(
            x,
            d0,
            lookup_state.cell_x_min[cid],
            lookup_state.cell_x_max[cid],
            abs_eps,
            dir_eps,
        )
        ty = _forward_face_exit_dt(
            y,
            d1,
            lookup_state.cell_y_min[cid],
            lookup_state.cell_y_max[cid],
            abs_eps,
            dir_eps,
        )
        tz = _forward_face_exit_dt(
            z,
            d2,
            lookup_state.cell_z_min[cid],
            lookup_state.cell_z_max[cid],
            abs_eps,
            dir_eps,
        )

        dt_exit = tx
        if ty < dt_exit:
            dt_exit = ty
        if tz < dt_exit:
            dt_exit = tz
        if not math.isfinite(dt_exit):
            dt_exit = t_end - t
        if dt_exit <= abs_eps:
            dt_exit = abs_eps
        t_exit = t + dt_exit
        if t_exit > t_end:
            t_exit = t_end
        if t_exit < t:
            t_exit = t

        cell_ids[n_seg] = cid
        enters[n_seg] = t
        exits[n_seg] = t_exit
        n_seg += 1
        if n_seg >= max_steps:
            break
        if t_exit >= (t_end - abs_eps):
            break

        face_mask = _xyz_exit_face_mask(dt_exit, tx, ty, tz, d0, d1, d2, abs_eps)
        if face_mask == 0:
            break

        t_next = t_exit + abs_eps
        if t_next > t_end:
            t_next = t_end
        if t_next <= t + abs_eps * 0.25:
            t_next = t + abs_eps
            if t_next > t_end:
                t_next = t_end

        x_next = origin_xyz[0] + t_next * d0
        y_next = origin_xyz[1] + t_next * d1
        z_next = origin_xyz[2] + t_next * d2
        next_node, cid_next = _select_next_xyz_cell_from_topology(
            current_node,
            face_mask,
            x_next,
            y_next,
            z_next,
            topo_state,
            lookup_state,
            candidate_nodes0,
            candidate_nodes1,
        )
        if cid_next < 0 and t_next < t_end:
            t_next = t_next + 4.0 * abs_eps
            if t_next > t_end:
                t_next = t_end
            x_next = origin_xyz[0] + t_next * d0
            y_next = origin_xyz[1] + t_next * d1
            z_next = origin_xyz[2] + t_next * d2
            next_node, cid_next = _select_next_xyz_cell_from_topology(
                current_node,
                face_mask,
                x_next,
                y_next,
                z_next,
                topo_state,
                lookup_state,
                candidate_nodes0,
                candidate_nodes1,
            )
            if cid_next < 0:
                break

        t = t_next
        x = x_next
        y = y_next
        z = z_next
        cid = cid_next
        current_node = next_node

    return n_seg, cell_ids, enters, exits


@njit(cache=True)
def _trace_segments_rpa_kernel(
    origin_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    boundary_tol: float,
    lookup_state: SphericalLookupKernelState,
    topo_state: TopologicalKernelState,
    hexa_state: RpaHexaKernelState,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Trace one Cartesian ray on spherical tree; return segment arrays."""
    cell_ids = np.empty(max_steps, dtype=np.int64)
    enters = np.empty(max_steps, dtype=np.float64)
    exits = np.empty(max_steps, dtype=np.float64)

    if t_end <= t_start or max_steps <= 0:
        return 0, cell_ids, enters, exits

    abs_eps = max(boundary_tol * (1.0 + abs(t_end - t_start)), 1e-12)
    found, t, cid, x, y, z = _seek_first_cell_rpa(
        origin_xyz,
        direction_xyz_unit,
        t_start,
        t_end,
        abs_eps,
        lookup_state,
        hexa_state,
    )
    if not found or cid < 0:
        return 0, cell_ids, enters, exits

    current_node = int(topo_state.cell_to_node_id[cid])
    if current_node < 0:
        return 0, cell_ids, enters, exits

    candidate_nodes0 = np.empty(_MAX_TOPOLOGY_CANDIDATES, dtype=np.int64)
    candidate_nodes1 = np.empty(_MAX_TOPOLOGY_CANDIDATES, dtype=np.int64)
    d0 = direction_xyz_unit[0]
    d1 = direction_xyz_unit[1]
    d2 = direction_xyz_unit[2]
    n_seg = 0
    for _ in range(max_steps):
        if t >= (t_end - abs_eps):
            break

        if not _contains_rpa_hexa_from_xyz(cid, x, y, z, hexa_state):
            break

        dt_exit, face_mask = _rpa_hexa_exit_dt_and_mask(
            cid,
            x,
            y,
            z,
            d0,
            d1,
            d2,
            hexa_state,
            abs_eps,
        )
        if not math.isfinite(dt_exit):
            dt_exit = t_end - t
        if dt_exit <= abs_eps:
            dt_exit = abs_eps
        t_exit = t + dt_exit
        if t_exit > t_end:
            t_exit = t_end
        if t_exit < t:
            t_exit = t
        cell_ids[n_seg] = cid
        enters[n_seg] = t
        exits[n_seg] = t_exit
        n_seg += 1
        if n_seg >= max_steps:
            break
        if t_exit >= (t_end - abs_eps):
            break

        if face_mask == 0:
            break

        t_next = t_exit + abs_eps
        if t_next > t_end:
            t_next = t_end
        if t_next <= t + abs_eps * 0.25:
            t_next = t + abs_eps
            if t_next > t_end:
                t_next = t_end

        x_next = origin_xyz[0] + t_next * d0
        y_next = origin_xyz[1] + t_next * d1
        z_next = origin_xyz[2] + t_next * d2
        next_node, cid_next = _select_next_rpa_cell_from_topology(
            current_node,
            face_mask,
            x_next,
            y_next,
            z_next,
            topo_state,
            hexa_state,
            candidate_nodes0,
            candidate_nodes1,
        )
        if cid_next < 0 and t_next < t_end:
            t_next = t_next + 4.0 * abs_eps
            if t_next > t_end:
                t_next = t_end
            x_next = origin_xyz[0] + t_next * d0
            y_next = origin_xyz[1] + t_next * d1
            z_next = origin_xyz[2] + t_next * d2
            next_node, cid_next = _select_next_rpa_cell_from_topology(
                current_node,
                face_mask,
                x_next,
                y_next,
                z_next,
                topo_state,
                hexa_state,
                candidate_nodes0,
                candidate_nodes1,
            )
            if cid_next < 0:
                break

        t = t_next
        x = x_next
        y = y_next
        z = z_next
        cid = cid_next
        current_node = next_node

    return n_seg, cell_ids, enters, exits


@njit(cache=True, parallel=True)
def _segment_counts_xyz_kernel(
    origins_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    boundary_tol: float,
    lookup_state: CartesianLookupKernelState,
    topo_state: TopologicalKernelState,
) -> np.ndarray:
    """Count traced segments for each ray on Cartesian trees."""
    n_rays = int(origins_xyz.shape[0])
    counts = np.zeros(n_rays, dtype=np.int64)
    for i in prange(n_rays):
        n_seg, _cids, _enters, _exits = _trace_segments_xyz_kernel(
            origins_xyz[i],
            direction_xyz_unit,
            t_start,
            t_end,
            max_steps,
            boundary_tol,
            lookup_state,
            topo_state,
        )
        counts[i] = int(n_seg)
    return counts


@njit(cache=True, parallel=True)
def _segment_counts_rpa_kernel(
    origins_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    boundary_tol: float,
    lookup_state: SphericalLookupKernelState,
    topo_state: TopologicalKernelState,
    hexa_state: RpaHexaKernelState,
) -> np.ndarray:
    """Count traced segments for each ray on spherical trees."""
    n_rays = int(origins_xyz.shape[0])
    counts = np.zeros(n_rays, dtype=np.int64)
    for i in prange(n_rays):
        n_seg, _cids, _enters, _exits = _trace_segments_rpa_kernel(
            origins_xyz[i],
            direction_xyz_unit,
            t_start,
            t_end,
            max_steps,
            boundary_tol,
            lookup_state,
            topo_state,
            hexa_state,
        )
        counts[i] = int(n_seg)
    return counts


@njit(cache=True)
def _fill_traces_xyz_kernel(
    origins_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    boundary_tol: float,
    lookup_state: CartesianLookupKernelState,
    topo_state: TopologicalKernelState,
    ray_offsets: np.ndarray,
    out_cell_ids: np.ndarray,
    out_t_enter: np.ndarray,
    out_t_exit: np.ndarray,
) -> None:
    """Fill flattened trace arrays from per-ray offsets on Cartesian trees."""
    n_rays = int(origins_xyz.shape[0])
    for i in range(n_rays):
        n_seg, cids, enters, exits = _trace_segments_xyz_kernel(
            origins_xyz[i],
            direction_xyz_unit,
            t_start,
            t_end,
            max_steps,
            boundary_tol,
            lookup_state,
            topo_state,
        )
        base = int(ray_offsets[i])
        for j in range(int(n_seg)):
            out_cell_ids[base + j] = int(cids[j])
            out_t_enter[base + j] = float(enters[j])
            out_t_exit[base + j] = float(exits[j])


@njit(cache=True)
def _fill_traces_rpa_kernel(
    origins_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    boundary_tol: float,
    lookup_state: SphericalLookupKernelState,
    topo_state: TopologicalKernelState,
    hexa_state: RpaHexaKernelState,
    ray_offsets: np.ndarray,
    out_cell_ids: np.ndarray,
    out_t_enter: np.ndarray,
    out_t_exit: np.ndarray,
) -> None:
    """Fill flattened trace arrays from per-ray offsets on spherical trees."""
    n_rays = int(origins_xyz.shape[0])
    for i in range(n_rays):
        n_seg, cids, enters, exits = _trace_segments_rpa_kernel(
            origins_xyz[i],
            direction_xyz_unit,
            t_start,
            t_end,
            max_steps,
            boundary_tol,
            lookup_state,
            topo_state,
            hexa_state,
        )
        base = int(ray_offsets[i])
        for j in range(int(n_seg)):
            out_cell_ids[base + j] = int(cids[j])
            out_t_enter[base + j] = float(enters[j])
            out_t_exit[base + j] = float(exits[j])


@njit(cache=True, parallel=True)
def _midpoints_from_segments_kernel(
    origins_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_enter: np.ndarray,
    t_exit: np.ndarray,
    ray_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert flattened segment arrays to midpoint points and weights."""
    n_seg_total = int(t_enter.shape[0])
    mids = np.empty((n_seg_total, 3), dtype=np.float64)
    weights = np.empty(n_seg_total, dtype=np.float64)
    d0 = float(direction_xyz_unit[0])
    d1 = float(direction_xyz_unit[1])
    d2 = float(direction_xyz_unit[2])
    n_rays = int(origins_xyz.shape[0])
    for i in prange(n_rays):
        base = int(ray_offsets[i])
        end = int(ray_offsets[i + 1])
        ox = float(origins_xyz[i, 0])
        oy = float(origins_xyz[i, 1])
        oz = float(origins_xyz[i, 2])
        for k in range(base, end):
            ta = float(t_enter[k])
            tb = float(t_exit[k])
            tm = 0.5 * (ta + tb)
            mids[k, 0] = ox + tm * d0
            mids[k, 1] = oy + tm * d1
            mids[k, 2] = oz + tm * d2
            weights[k] = max(0.0, tb - ta)
    return mids, weights


@njit(cache=True, parallel=True)
def _integrate_rays_xyz_kernel(
    origins_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    boundary_tol: float,
    scale: float,
    trace_lookup_state: CartesianLookupKernelState,
    topo_state: TopologicalKernelState,
    interp_state: CartesianInterpKernelState,
) -> np.ndarray:
    """Integrate fields along rays on Cartesian trees using midpoint segments."""
    n_rays = int(origins_xyz.shape[0])
    ncomp = int(interp_state.point_values_2d.shape[1])
    out = np.full((n_rays, ncomp), np.nan, dtype=np.float64)
    d0 = float(direction_xyz_unit[0])
    d1 = float(direction_xyz_unit[1])
    d2 = float(direction_xyz_unit[2])
    for i in prange(n_rays):
        n_seg, cids, enters, exits = _trace_segments_xyz_kernel(
            origins_xyz[i],
            direction_xyz_unit,
            t_start,
            t_end,
            max_steps,
            boundary_tol,
            trace_lookup_state,
            topo_state,
        )
        if n_seg <= 0:
            if ncomp == 1:
                out[i, 0] = 0.0
            continue
        acc = np.zeros(ncomp, dtype=np.float64)
        row = np.empty(ncomp, dtype=np.float64)
        ox = float(origins_xyz[i, 0])
        oy = float(origins_xyz[i, 1])
        oz = float(origins_xyz[i, 2])
        used = False
        for j in range(int(n_seg)):
            ta = float(enters[j])
            tb = float(exits[j])
            dt = tb - ta
            if dt <= 0.0:
                continue
            tm = 0.5 * (ta + tb)
            x = ox + tm * d0
            y = oy + tm * d1
            z = oz + tm * d2
            _trilinear_from_cell(
                row,
                int(cids[j]),
                x,
                y,
                z,
                interp_state,
            )
            acc += row * dt
            used = True
        if used:
            out[i, :] = scale * acc
        elif ncomp == 1:
            out[i, 0] = 0.0
    return out


@njit(cache=True, parallel=True)
def _integrate_rays_rpa_kernel(
    origins_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    boundary_tol: float,
    scale: float,
    trace_lookup_state: SphericalLookupKernelState,
    topo_state: TopologicalKernelState,
    hexa_state: RpaHexaKernelState,
    interp_state: SphericalInterpKernelState,
) -> np.ndarray:
    """Integrate fields along rays on spherical trees using midpoint segments."""
    n_rays = int(origins_xyz.shape[0])
    ncomp = int(interp_state.point_values_2d.shape[1])
    out = np.full((n_rays, ncomp), np.nan, dtype=np.float64)
    d0 = float(direction_xyz_unit[0])
    d1 = float(direction_xyz_unit[1])
    d2 = float(direction_xyz_unit[2])
    for i in prange(n_rays):
        n_seg, cids, enters, exits = _trace_segments_rpa_kernel(
            origins_xyz[i],
            direction_xyz_unit,
            t_start,
            t_end,
            max_steps,
            boundary_tol,
            trace_lookup_state,
            topo_state,
            hexa_state,
        )
        if n_seg <= 0:
            if ncomp == 1:
                out[i, 0] = 0.0
            continue
        acc = np.zeros(ncomp, dtype=np.float64)
        row = np.empty(ncomp, dtype=np.float64)
        ox = float(origins_xyz[i, 0])
        oy = float(origins_xyz[i, 1])
        oz = float(origins_xyz[i, 2])
        used = False
        for j in range(int(n_seg)):
            ta = float(enters[j])
            tb = float(exits[j])
            dt = tb - ta
            if dt <= 0.0:
                continue
            tm = 0.5 * (ta + tb)
            x = ox + tm * d0
            y = oy + tm * d1
            z = oz + tm * d2
            r, polar, azimuth = _xyz_to_rpa_components(x, y, z)
            _trilinear_from_cell_rpa(
                row,
                int(cids[j]),
                r,
                polar,
                azimuth,
                interp_state,
            )
            acc += row * dt
            used = True
        if used:
            out[i, :] = scale * acc
        elif ncomp == 1:
            out[i, 0] = 0.0
    return out


def _normalize_direction(direction_xyz: np.ndarray) -> np.ndarray:
    """Normalize one Cartesian ray direction."""
    d = np.asarray(direction_xyz, dtype=float).reshape(3)
    norm = float(np.linalg.norm(d))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("direction_xyz must be finite and non-zero.")
    return d / norm


def _as_xyz(point_xyz: np.ndarray) -> np.ndarray:
    """Coerce one Cartesian point to shape `(3,)`."""
    return np.asarray(point_xyz, dtype=float).reshape(3)


def _coerce_origins_xyz(origins_xyz: np.ndarray) -> np.ndarray:
    """Validate and normalize ray origins to shape `(n_rays, 3)`."""
    origins = np.asarray(origins_xyz, dtype=float)
    if origins.ndim == 1:
        if origins.size != 3:
            raise ValueError("origins_xyz must have shape (n_rays, 3) or (3,).")
        origins = origins.reshape(1, 3)
    if origins.ndim != 2 or origins.shape[1] != 3:
        raise ValueError("origins_xyz must have shape (n_rays, 3) or (3,).")
    return origins


def _coerce_ray_interval(t_start: float, t_end: float) -> tuple[float, float]:
    """Validate and normalize ray interval."""
    t0 = float(t_start)
    t1 = float(t_end)
    if t1 <= t0:
        raise ValueError("t_end must be greater than t_start.")
    return t0, t1


def _coerce_positive_chunk_size(chunk_size: int) -> int:
    """Validate chunk size."""
    chunk = int(chunk_size)
    if chunk <= 0:
        raise ValueError("chunk_size must be positive.")
    return chunk


def _coerce_positive_int(name: str, value: int) -> int:
    """Validate positive integer parameter."""
    iv = int(value)
    if iv <= 0:
        raise ValueError(f"{name} must be positive.")
    return iv


class OctreeRayTracer:
    """Thin convenience wrapper around compiled ray tracing kernels."""

    def __init__(self, tree: Octree) -> None:
        """Bind one built octree."""
        self.tree = tree
        self._tree_coord = str(tree.tree_coord)
        dmin, dmax = tree.domain_bounds(coord="xyz")
        self._domain_xyz_min = np.asarray(dmin, dtype=float).reshape(3)
        self._domain_xyz_max = np.asarray(dmax, dtype=float).reshape(3)
        _r_lo, r_hi = tree.domain_bounds(coord="rpa")
        self._domain_r_max = float(np.asarray(r_hi, dtype=float).reshape(3)[0])
        self._lookup_state = tree.lookup.lookup_state
        topology = getattr(tree, "_ray_topology_full", None)
        if topology is None or int(topology.max_level) != int(tree.max_level):
            topology = build_topological_neighborhood(tree, max_level=int(tree.max_level))
            tree._ray_topology_full = topology
        self._topology = topology
        self._topology_state = topology.kernel_state
        self._rpa_hexa_state = None
        if self._tree_coord == "rpa":
            hexa_state = getattr(tree, "_ray_rpa_hexa_state", None)
            if hexa_state is None:
                hexa_state = _build_rpa_hexa_kernel_state(tree)
                tree._ray_rpa_hexa_state = hexa_state
            self._rpa_hexa_state = hexa_state

    def trace(
        self,
        origin_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
        *,
        max_steps: int = _DEFAULT_TRACE_MAX_STEPS,
        boundary_tol: float = _DEFAULT_TRACE_BOUNDARY_TOL,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Trace one ray and return `(cell_ids, t_enter, t_exit)`."""
        o = _as_xyz(origin_xyz)
        d = _normalize_direction(direction_xyz)
        return self.trace_prepared(
            o,
            d,
            float(t_start),
            float(t_end),
            max_steps=max_steps,
            boundary_tol=boundary_tol,
        )

    def trace_prepared(
        self,
        origin_xyz: np.ndarray,
        direction_xyz_unit: np.ndarray,
        t_start: float,
        t_end: float,
        *,
        max_steps: int = _DEFAULT_TRACE_MAX_STEPS,
        boundary_tol: float = _DEFAULT_TRACE_BOUNDARY_TOL,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Trace one ray for normalized inputs and return segment arrays."""
        max_steps = _coerce_positive_int("max_steps", max_steps)
        t0 = float(t_start)
        t1 = float(t_end)
        if t1 <= t0:
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=float),
            )

        o = np.asarray(origin_xyz, dtype=float).reshape(3)
        d = np.asarray(direction_xyz_unit, dtype=float).reshape(3)

        clipped, t0_clip, t1_clip = _clip_ray_interval_to_slab(
            o,
            d,
            t0,
            t1,
            self._domain_xyz_min,
            self._domain_xyz_max,
            float(boundary_tol),
        )
        if not clipped or t1_clip <= t0_clip:
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=float),
            )

        if self._tree_coord == "rpa":
            clipped_sphere, t0_sphere, t1_sphere = _clip_ray_interval_to_sphere(
                o,
                d,
                t0_clip,
                t1_clip,
                float(self._domain_r_max),
                float(boundary_tol),
            )
            if not clipped_sphere or t1_sphere <= t0_sphere:
                return (
                    np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=float),
                    np.empty((0,), dtype=float),
                )
            t0_clip = t0_sphere
            t1_clip = t1_sphere

        abs_eps = max(float(boundary_tol) * (1.0 + abs(t1_clip - t0_clip)), 1e-12)
        t0 = t0_clip
        if t0_clip > float(t_start):
            t0 = min(t1_clip, t0_clip + abs_eps)
        t1 = t1_clip
        if t1 <= t0:
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=float),
            )

        lookup_state = self._lookup_state
        topo_state = self._topology_state
        if self._tree_coord == "xyz":
            if not isinstance(lookup_state, CartesianLookupKernelState):
                raise TypeError(
                    "Cartesian ray tracing requires CartesianLookupKernelState; "
                    f"got {type(lookup_state).__name__}."
                )
            n_seg, cids, enters, exits = _trace_segments_xyz_kernel(
                o,
                d,
                t0,
                t1,
                int(max_steps),
                float(boundary_tol),
                lookup_state,
                topo_state,
            )
        elif self._tree_coord == "rpa":
            if not isinstance(lookup_state, SphericalLookupKernelState):
                raise TypeError(
                    "Spherical ray tracing requires SphericalLookupKernelState; "
                    f"got {type(lookup_state).__name__}."
                )
            n_seg, cids, enters, exits = _trace_segments_rpa_kernel(
                o,
                d,
                t0,
                t1,
                int(max_steps),
                float(boundary_tol),
                lookup_state,
                topo_state,
                self._rpa_hexa_state,
            )
        else:
            raise ValueError(f"Unsupported tree_coord '{self._tree_coord}'.")

        n = int(n_seg)
        return (
            np.array(cids[:n], dtype=np.int64, copy=True),
            np.array(enters[:n], dtype=float, copy=True),
            np.array(exits[:n], dtype=float, copy=True),
        )

    def segment_counts(
        self,
        origins_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
        *,
        chunk_size: int = 2048,
        max_steps: int = _DEFAULT_TRACE_MAX_STEPS,
        boundary_tol: float = _DEFAULT_TRACE_BOUNDARY_TOL,
    ) -> np.ndarray:
        """Return traced segment count for each ray."""
        origins = _coerce_origins_xyz(origins_xyz)
        d = _normalize_direction(direction_xyz)
        t0, t1 = _coerce_ray_interval(t_start, t_end)
        chunk = _coerce_positive_chunk_size(chunk_size)
        max_steps = _coerce_positive_int("max_steps", max_steps)

        counts = np.zeros(origins.shape[0], dtype=np.int64)
        for start in range(0, int(origins.shape[0]), chunk):
            stop = min(int(origins.shape[0]), start + chunk)
            sub = origins[start:stop]
            topo_state = self._topology_state
            if self._tree_coord == "xyz":
                lookup_state = self._lookup_state
                if not isinstance(lookup_state, CartesianLookupKernelState):
                    raise TypeError(
                        "Cartesian ray tracing requires CartesianLookupKernelState; "
                        f"got {type(lookup_state).__name__}."
                    )
                counts[start:stop] = _segment_counts_xyz_kernel(
                    sub,
                    d,
                    t0,
                    t1,
                    int(max_steps),
                    float(boundary_tol),
                    lookup_state,
                    topo_state,
                )
            elif self._tree_coord == "rpa":
                lookup_state = self._lookup_state
                if not isinstance(lookup_state, SphericalLookupKernelState):
                    raise TypeError(
                        "Spherical ray tracing requires SphericalLookupKernelState; "
                        f"got {type(lookup_state).__name__}."
                    )
                counts[start:stop] = _segment_counts_rpa_kernel(
                    sub,
                    d,
                    t0,
                    t1,
                    int(max_steps),
                    float(boundary_tol),
                    lookup_state,
                    topo_state,
                    self._rpa_hexa_state,
                )
            else:
                raise ValueError(f"Unsupported tree_coord '{self._tree_coord}'.")
        return counts

    def trace_many(
        self,
        origins_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
        *,
        max_steps: int = _DEFAULT_TRACE_MAX_STEPS,
        boundary_tol: float = _DEFAULT_TRACE_BOUNDARY_TOL,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Trace many rays and return flattened arrays plus `ray_offsets`."""
        origins = _coerce_origins_xyz(origins_xyz)
        d = _normalize_direction(direction_xyz)
        t0, t1 = _coerce_ray_interval(t_start, t_end)
        max_steps = _coerce_positive_int("max_steps", max_steps)

        if self._tree_coord == "xyz":
            lookup_state = self._lookup_state
            if not isinstance(lookup_state, CartesianLookupKernelState):
                raise TypeError(
                    "Cartesian ray tracing requires CartesianLookupKernelState; "
                    f"got {type(lookup_state).__name__}."
                )
            counts = _segment_counts_xyz_kernel(
                origins,
                d,
                t0,
                t1,
                int(max_steps),
                float(boundary_tol),
                lookup_state,
                self._topology_state,
            )
        elif self._tree_coord == "rpa":
            lookup_state = self._lookup_state
            if not isinstance(lookup_state, SphericalLookupKernelState):
                raise TypeError(
                    "Spherical ray tracing requires SphericalLookupKernelState; "
                    f"got {type(lookup_state).__name__}."
                )
            counts = _segment_counts_rpa_kernel(
                origins,
                d,
                t0,
                t1,
                int(max_steps),
                float(boundary_tol),
                lookup_state,
                self._topology_state,
                self._rpa_hexa_state,
            )
        else:
            raise ValueError(f"Unsupported tree_coord '{self._tree_coord}'.")

        ray_offsets = np.zeros(int(origins.shape[0]) + 1, dtype=np.int64)
        ray_offsets[1:] = np.cumsum(counts, dtype=np.int64)
        n_seg_total = int(ray_offsets[-1])
        cell_ids = np.empty(n_seg_total, dtype=np.int64)
        t_enter = np.empty(n_seg_total, dtype=np.float64)
        t_exit = np.empty(n_seg_total, dtype=np.float64)

        if self._tree_coord == "xyz":
            _fill_traces_xyz_kernel(
                origins,
                d,
                t0,
                t1,
                int(max_steps),
                float(boundary_tol),
                self._lookup_state,
                self._topology_state,
                ray_offsets,
                cell_ids,
                t_enter,
                t_exit,
            )
        else:
            _fill_traces_rpa_kernel(
                origins,
                d,
                t0,
                t1,
                int(max_steps),
                float(boundary_tol),
                self._lookup_state,
                self._topology_state,
                self._rpa_hexa_state,
                ray_offsets,
                cell_ids,
                t_enter,
                t_exit,
            )

        return cell_ids, t_enter, t_exit, ray_offsets


class OctreeRayInterpolator:
    """Thin convenience wrapper around compiled ray integration kernels."""

    def __init__(self, interpolator: "OctreeInterpolator") -> None:
        """Bind one interpolator and its tree."""
        self.interpolator = interpolator
        self.tree = interpolator.tree
        self.ray_tracer = OctreeRayTracer(self.tree)

    def sample(
        self,
        origin_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
        n_samples: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Sample interpolated values at uniform `t` points on one ray."""
        if int(n_samples) <= 0:
            raise ValueError("n_samples must be positive.")
        o = _as_xyz(origin_xyz)
        d = _normalize_direction(direction_xyz)
        t_values = np.linspace(float(t_start), float(t_end), int(n_samples))
        query_xyz = o[None, :] + t_values[:, None] * d[None, :]
        values, cell_ids = self.interpolator(
            query_xyz,
            query_coord="xyz",
            return_cell_ids=True,
            log_outside_domain=False,
        )
        segments = self.ray_tracer.trace_prepared(o, d, float(t_start), float(t_end))
        return t_values, values, np.asarray(cell_ids, dtype=np.int64), segments

    def segment_counts(
        self,
        origins_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
        *,
        chunk_size: int = 2048,
        max_steps: int = _DEFAULT_TRACE_MAX_STEPS,
        boundary_tol: float = _DEFAULT_TRACE_BOUNDARY_TOL,
    ) -> np.ndarray:
        """Return per-ray segment counts."""
        return self.ray_tracer.segment_counts(
            origins_xyz,
            direction_xyz,
            t_start,
            t_end,
            chunk_size=chunk_size,
            max_steps=max_steps,
            boundary_tol=boundary_tol,
        )

    def adaptive_midpoint_rule(
        self,
        origins_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
        *,
        chunk_size: int = 2048,
        max_steps: int = _DEFAULT_TRACE_MAX_STEPS,
        boundary_tol: float = _DEFAULT_TRACE_BOUNDARY_TOL,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build flattened midpoint quadrature arrays for many rays."""
        origins = _coerce_origins_xyz(origins_xyz)
        d = _normalize_direction(direction_xyz)
        t0, t1 = _coerce_ray_interval(t_start, t_end)
        chunk = _coerce_positive_chunk_size(chunk_size)
        max_steps = _coerce_positive_int("max_steps", max_steps)

        mids_chunks: list[np.ndarray] = []
        weights_chunks: list[np.ndarray] = []
        offsets = np.zeros(int(origins.shape[0]) + 1, dtype=np.int64)

        written = 0
        for start in range(0, int(origins.shape[0]), chunk):
            stop = min(int(origins.shape[0]), start + chunk)
            sub = origins[start:stop]
            cids, t_enter, t_exit, sub_offsets = self.ray_tracer.trace_many(
                sub,
                d,
                t0,
                t1,
                max_steps=max_steps,
                boundary_tol=boundary_tol,
            )
            mids_sub, weights_sub = _midpoints_from_segments_kernel(sub, d, t_enter, t_exit, sub_offsets)
            mids_chunks.append(mids_sub)
            weights_chunks.append(weights_sub)

            local_counts = np.diff(sub_offsets)
            offsets[start + 1 : stop + 1] = written + np.cumsum(local_counts, dtype=np.int64)
            written += int(sub_offsets[-1])

        if written == 0:
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0,), dtype=float),
                offsets,
            )
        return np.vstack(mids_chunks), np.concatenate(weights_chunks), offsets

    def integrate_midpoint_rule(
        self,
        midpoints_xyz: np.ndarray,
        weights: np.ndarray,
        ray_offsets: np.ndarray,
        *,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Integrate field values from precomputed midpoint quadrature arrays."""
        mids = np.asarray(midpoints_xyz, dtype=float)
        w = np.asarray(weights, dtype=float).reshape(-1)
        offsets = np.asarray(ray_offsets, dtype=np.int64).reshape(-1)

        if mids.ndim != 2 or mids.shape[1] != 3:
            raise ValueError("midpoints_xyz must have shape (n_samples, 3).")
        if mids.shape[0] != w.shape[0]:
            raise ValueError("weights length must equal number of midpoint samples.")
        if offsets.size < 1:
            raise ValueError("ray_offsets must have length >= 1.")
        if offsets[0] != 0:
            raise ValueError("ray_offsets must start at 0.")
        if offsets[-1] != mids.shape[0]:
            raise ValueError("ray_offsets must end at n_samples.")
        if np.any(np.diff(offsets) < 0):
            raise ValueError("ray_offsets must be non-decreasing.")

        n_rays = int(offsets.size - 1)
        ncomp = int(self.interpolator.n_value_components)
        out = np.full((n_rays, ncomp), np.nan, dtype=float)
        if mids.shape[0] == 0:
            if ncomp == 1:
                out[:, 0] = 0.0
                return out[:, 0]
            return out

        values, cell_ids = self.interpolator(
            mids,
            query_coord="xyz",
            return_cell_ids=True,
            log_outside_domain=False,
        )
        value_arr = np.asarray(values, dtype=float)
        if value_arr.ndim == 1:
            value_arr = value_arr.reshape(-1, 1)
        cids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)

        for i in range(n_rays):
            s = int(offsets[i])
            e = int(offsets[i + 1])
            if e <= s:
                if ncomp == 1:
                    out[i, 0] = 0.0
                continue
            valid = cids[s:e] >= 0
            if not np.any(valid):
                if ncomp == 1:
                    out[i, 0] = 0.0
                continue
            seg_vals = value_arr[s:e][valid]
            seg_w = w[s:e][valid].reshape(-1, 1)
            out[i] = float(scale) * np.sum(seg_vals * seg_w, axis=0)

        if ncomp == 1:
            return out[:, 0]
        return out

    def integrate_field_along_rays_midpoint(
        self,
        origins_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
        *,
        chunk_size: int = 2048,
        scale: float = 1.0,
        max_steps: int = _DEFAULT_TRACE_MAX_STEPS,
        boundary_tol: float = _DEFAULT_TRACE_BOUNDARY_TOL,
    ) -> np.ndarray:
        """One-shot midpoint integration (trace->midpoints->interpolate->sum)."""
        mids, weights, offsets = self.adaptive_midpoint_rule(
            origins_xyz,
            direction_xyz,
            t_start,
            t_end,
            chunk_size=chunk_size,
            max_steps=max_steps,
            boundary_tol=boundary_tol,
        )
        return self.integrate_midpoint_rule(mids, weights, offsets, scale=scale)

    def integrate_field_along_rays(
        self,
        origins_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
        *,
        chunk_size: int = 2048,
        scale: float = 1.0,
        max_steps: int = _DEFAULT_TRACE_MAX_STEPS,
        boundary_tol: float = _DEFAULT_TRACE_BOUNDARY_TOL,
    ) -> np.ndarray:
        """Integrate interpolated fields along many rays using compiled kernels."""
        origins = _coerce_origins_xyz(origins_xyz)
        d = _normalize_direction(direction_xyz)
        t0, t1 = _coerce_ray_interval(t_start, t_end)
        chunk = _coerce_positive_chunk_size(chunk_size)
        max_steps = _coerce_positive_int("max_steps", max_steps)

        n_rays = int(origins.shape[0])
        ncomp = int(self.interpolator.n_value_components)
        out = np.full((n_rays, ncomp), np.nan, dtype=float)

        tree_coord = str(self.tree.tree_coord)
        topo_state = self.ray_tracer._topology_state
        for start in range(0, n_rays, chunk):
            stop = min(n_rays, start + chunk)
            sub = origins[start:stop]
            if tree_coord == "xyz":
                lookup_state = self.tree.lookup.lookup_state
                if not isinstance(lookup_state, CartesianLookupKernelState):
                    raise TypeError(
                        "Cartesian ray tracing requires CartesianLookupKernelState; "
                        f"got {type(lookup_state).__name__}."
                    )
                interp_state = self.interpolator._interp_state_xyz
                if not isinstance(interp_state, CartesianInterpKernelState):
                    raise TypeError(
                        "Cartesian ray interpolation requires CartesianInterpKernelState; "
                        f"got {type(interp_state).__name__}."
                    )
                out[start:stop] = _integrate_rays_xyz_kernel(
                    sub,
                    d,
                    t0,
                    t1,
                    int(max_steps),
                    float(boundary_tol),
                    float(scale),
                    lookup_state,
                    topo_state,
                    interp_state,
                )
            elif tree_coord == "rpa":
                lookup_state = self.tree.lookup.lookup_state
                if not isinstance(lookup_state, SphericalLookupKernelState):
                    raise TypeError(
                        "Spherical ray tracing requires SphericalLookupKernelState; "
                        f"got {type(lookup_state).__name__}."
                    )
                interp_state = self.interpolator._interp_state_rpa
                if not isinstance(interp_state, SphericalInterpKernelState):
                    raise TypeError(
                        "Spherical ray interpolation requires SphericalInterpKernelState; "
                        f"got {type(interp_state).__name__}."
                    )
                out[start:stop] = _integrate_rays_rpa_kernel(
                    sub,
                    d,
                    t0,
                    t1,
                    int(max_steps),
                    float(boundary_tol),
                    float(scale),
                    lookup_state,
                    topo_state,
                    self.ray_tracer._rpa_hexa_state,
                    interp_state,
                )
            else:
                raise ValueError(f"Unsupported tree_coord '{tree_coord}'.")

        if ncomp == 1:
            return out[:, 0]
        return out
