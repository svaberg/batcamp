#!/usr/bin/env python3
"""Ray traversal and ray-based integration helpers for octrees."""

from __future__ import annotations

from dataclasses import dataclass
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
from .topological import TopologicalNeighborhood
from .topological import _cell_local_indices_from_bounds
from .topological import build_topological_neighborhood_kernel
from .topological import build_topological_neighborhood

if TYPE_CHECKING:
    from .interpolator import OctreeInterpolator


_TRACE_CONTAIN_TOL = 1e-8
_DEFAULT_TRACE_BOUNDARY_TOL = 1e-9
_DEFAULT_TRACE_MAX_STEPS = 100000
_MAX_TOPOLOGY_CANDIDATES = 128
_GAUSS_LEGENDRE_2_ABSCISSA = 0.5773502691896257
_FACE_TRIPLES = (
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 3),
    (1, 2, 3),
)
_FACE_TRIPLES_ARRAY = np.asarray(_FACE_TRIPLES, dtype=np.int64)


class CellPlaneKernelState(NamedTuple):
    """Plane representation of cell faces in Cartesian coordinates."""

    face_normals: np.ndarray
    face_offsets: np.ndarray
    face_valid: np.ndarray


class CartesianRayCellGeometry(NamedTuple):
    """Truncated Cartesian ray-cell geometry at one depth cutoff."""

    topology_state: TopologicalKernelState
    lookup_state: CartesianLookupKernelState
    corners: np.ndarray
    bin_to_corner: np.ndarray
    cell_x0: np.ndarray
    cell_xden: np.ndarray
    cell_y0: np.ndarray
    cell_yden: np.ndarray
    cell_z0: np.ndarray
    cell_zden: np.ndarray


class SphericalRayCellGeometry(NamedTuple):
    """Truncated spherical ray-cell geometry at one depth cutoff."""

    topology_state: TopologicalKernelState
    plane_state: CellPlaneKernelState
    cell_centers: np.ndarray
    corners: np.ndarray
    bin_to_corner: np.ndarray
    cell_r0: np.ndarray
    cell_rden: np.ndarray
    cell_t0: np.ndarray
    cell_tden: np.ndarray
    cell_p_start: np.ndarray
    cell_p_width: np.ndarray
    cell_pden: np.ndarray
    cell_phi_full: np.ndarray
    cell_phi_tiny: np.ndarray


def _camera_xyz(value: np.ndarray, *, name: str) -> np.ndarray:
    """Return one finite Cartesian 3-vector for camera setup."""
    xyz = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(xyz)):
        raise ValueError(f"{name} must be finite.")
    return xyz


def _normalize_camera_xyz(value: np.ndarray, *, name: str) -> np.ndarray:
    """Return one normalized finite Cartesian 3-vector for camera setup."""
    xyz = _camera_xyz(value, name=name)
    norm = float(np.linalg.norm(xyz))
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero.")
    return xyz / norm


def _camera_basis(forward_xyz: np.ndarray, up_hint_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orthonormal `(forward, right, up)` camera axes."""
    forward = _normalize_camera_xyz(forward_xyz, name="forward_xyz")
    up_hint = _normalize_camera_xyz(up_hint_xyz, name="up_hint_xyz")
    right = np.cross(up_hint, forward)
    right_norm = float(np.linalg.norm(right))
    if right_norm == 0.0:
        raise ValueError("up_hint_xyz must not be parallel to forward_xyz.")
    right = right / right_norm
    up = np.cross(forward, right)
    up = up / float(np.linalg.norm(up))
    return forward, right, up


@dataclass(frozen=True)
class FlatCamera:
    """Orthographic camera with one rectangular launch plane and shared ray direction."""

    plane_center_xyz: np.ndarray
    forward_xyz: np.ndarray
    up_hint_xyz: np.ndarray
    width: float
    height: float
    t_end: float

    @classmethod
    def from_domain_x(
        cls,
        bounds: tuple[float, float, float, float, float, float],
        *,
        pad_fraction: float = 1.0e-6,
        t_scale: float = 0.999999,
    ) -> "FlatCamera":
        """Build the current compare-style `+x` flat camera from XYZ bounds."""
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        x_span = float(xmax - xmin)
        x0 = float(xmin - float(pad_fraction) * max(1.0, x_span))
        return cls(
            plane_center_xyz=np.array([x0, 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)], dtype=float),
            forward_xyz=np.array([1.0, 0.0, 0.0], dtype=float),
            up_hint_xyz=np.array([0.0, 0.0, 1.0], dtype=float),
            width=float(ymax - ymin),
            height=float(zmax - zmin),
            t_end=float((xmax - x0) * float(t_scale)),
        )

    def rays(self, *, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int]]:
        """Return `(origins, direction, t_end, image_shape)` for one `(z, y)` image."""
        ny = int(ny)
        nz = int(nz)
        if ny <= 0 or nz <= 0:
            raise ValueError("ny and nz must be positive.")
        if float(self.width) <= 0.0 or float(self.height) <= 0.0:
            raise ValueError("Flat camera width and height must be positive.")
        if float(self.t_end) <= 0.0:
            raise ValueError("Flat camera t_end must be positive.")

        center = _camera_xyz(self.plane_center_xyz, name="plane_center_xyz")
        forward, right, up = _camera_basis(self.forward_xyz, self.up_hint_xyz)
        y = np.linspace(-0.5 * float(self.width), 0.5 * float(self.width), ny, dtype=float)
        z = np.linspace(-0.5 * float(self.height), 0.5 * float(self.height), nz, dtype=float)
        yg, zg = np.meshgrid(y, z, indexing="xy")
        origins = (
            center[None, None, :]
            + yg[:, :, None] * right[None, None, :]
            + zg[:, :, None] * up[None, None, :]
        ).reshape(-1, 3)
        return origins, forward, float(self.t_end), (nz, ny)


@dataclass(frozen=True)
class FovCamera:
    """Perspective camera with one pinhole eye and per-pixel ray directions."""

    eye_xyz: np.ndarray
    target_xyz: np.ndarray
    up_hint_xyz: np.ndarray
    vertical_fov_degrees: float
    t_end: float

    def rays(self, *, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int]]:
        """Return `(origins, directions, t_end, image_shape)` for one `(z, y)` image."""
        ny = int(ny)
        nz = int(nz)
        if ny <= 0 or nz <= 0:
            raise ValueError("ny and nz must be positive.")
        vfov = float(self.vertical_fov_degrees)
        if vfov <= 0.0 or vfov >= 180.0:
            raise ValueError("vertical_fov_degrees must be in (0, 180).")
        if float(self.t_end) <= 0.0:
            raise ValueError("FOV camera t_end must be positive.")

        eye = _camera_xyz(self.eye_xyz, name="eye_xyz")
        target = _camera_xyz(self.target_xyz, name="target_xyz")
        forward, right, up = _camera_basis(target - eye, self.up_hint_xyz)

        half_height = math.tan(0.5 * math.radians(vfov))
        half_width = (float(ny) / float(nz)) * half_height
        y = np.linspace(-half_width, half_width, ny, dtype=float)
        z = np.linspace(-half_height, half_height, nz, dtype=float)
        yg, zg = np.meshgrid(y, z, indexing="xy")
        directions = (
            forward[None, None, :]
            + yg[:, :, None] * right[None, None, :]
            + zg[:, :, None] * up[None, None, :]
        )
        directions = directions / np.linalg.norm(directions, axis=2, keepdims=True)
        origins = np.broadcast_to(eye.reshape(1, 1, 3), directions.shape).reshape(-1, 3).copy()
        return origins, directions.reshape(-1, 3), float(self.t_end), (nz, ny)


_TARGET_UNIT_CORNERS = np.array(
    [[k & 1, (k >> 1) & 1, (k >> 2) & 1] for k in range(8)],
    dtype=np.float64,
)
_ORDERED_FACE_CORNERS = (
    (0, 2, 4, 6),
    (1, 3, 5, 7),
    (0, 1, 4, 5),
    (2, 3, 6, 7),
    (0, 1, 2, 3),
    (4, 5, 6, 7),
)
_ORDERED_FACE_CORNERS_ARRAY = np.asarray(_ORDERED_FACE_CORNERS, dtype=np.int64)


def _resolve_ray_maxdepth(tree: Octree, maxdepth: int | None) -> int:
    """Return requested ray depth cutoff, defaulting to the full tree depth."""
    if maxdepth is None:
        return int(tree.depth)
    depth = int(maxdepth)
    if depth < 0 or depth > int(tree.depth):
        raise ValueError(f"maxdepth={depth} is outside [0, {int(tree.depth)}] for this tree.")
    return depth


def _shape_for_ray_depth(tree: Octree, depth: int) -> tuple[int, int, int]:
    """Return `(n0, n1, n2)` cell counts at one root-relative depth."""
    if depth < 0 or depth > int(tree.depth):
        raise ValueError(f"depth={depth} is outside [0, {int(tree.depth)}] for this tree.")
    scale = 1 << int(depth)
    return (
        int(tree.root_shape[0]) * scale,
        int(tree.root_shape[1]) * scale,
        int(tree.root_shape[2]) * scale,
    )


def _depth_shapes_for_cutoff(tree: Octree, min_depth: int, max_depth: int) -> np.ndarray:
    """Return `(n0, n1, n2)` for every ray depth in `[min_depth, max_depth]`."""
    if max_depth < min_depth:
        raise ValueError(f"Invalid depth bounds: min_depth={min_depth}, max_depth={max_depth}.")
    out = np.empty((int(max_depth - min_depth + 1), 3), dtype=np.int64)
    for depth in range(int(min_depth), int(max_depth) + 1):
        out[int(depth - min_depth), :] = _shape_for_ray_depth(tree, depth)
    return out


def _topology_for_ray_maxdepth(tree: Octree, maxdepth: int) -> TopologicalNeighborhood:
    """Build face-neighbor topology at one root-relative ray depth cutoff."""
    cache = getattr(tree, "_ray_topology_by_maxdepth", None)
    if cache is None:
        cache = {}
        tree._ray_topology_by_maxdepth = cache
    key = int(maxdepth)
    if key in cache:
        return cache[key]

    tree._require_lookup()
    if tree.cell_levels is None:
        raise ValueError("Octree has no cell_levels; cannot build ray topology.")

    levels_all = np.asarray(tree.cell_levels, dtype=np.int64)
    valid = levels_all >= 0
    if not np.any(valid):
        raise ValueError("Octree contains no valid cells (all levels are < 0).")

    cell_ids = np.flatnonzero(valid).astype(np.int64)
    cell_levels = levels_all[valid]
    cell_depths = np.asarray(int(tree.depth) - (int(tree.max_level) - cell_levels), dtype=np.int64)
    active_depths = np.minimum(cell_depths, key)

    if str(tree.tree_coord) == "xyz":
        i0_valid, i1_valid, i2_valid = _cell_local_indices_from_bounds(tree, cell_ids, cell_levels)
    else:
        if not hasattr(tree, "_i0") or not hasattr(tree, "_i1") or not hasattr(tree, "_i2"):
            raise ValueError("Lookup indices (_i0/_i1/_i2) are unavailable; build lookup before ray topology.")
        i0_valid = np.asarray(getattr(tree, "_i0"), dtype=np.int64)[valid]
        i1_valid = np.asarray(getattr(tree, "_i1"), dtype=np.int64)[valid]
        i2_valid = np.asarray(getattr(tree, "_i2"), dtype=np.int64)[valid]

    shift = cell_depths - active_depths
    active_i0 = np.right_shift(i0_valid, shift)
    active_i1 = np.right_shift(i1_valid, shift)
    active_i2 = np.right_shift(i2_valid, shift)

    keys = np.column_stack((active_depths, active_i0, active_i1, active_i2)).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

    node_cell_ids = np.full(int(unique_keys.shape[0]), -1, dtype=np.int64)
    cell_to_node_id = np.full(levels_all.shape[0], -1, dtype=np.int64)
    for row, node_id in enumerate(inverse):
        nid = int(node_id)
        if node_cell_ids[nid] < 0:
            node_cell_ids[nid] = int(cell_ids[row])
        cell_to_node_id[int(cell_ids[row])] = nid

    depths = np.asarray(unique_keys[:, 0], dtype=np.int64)
    i0 = np.asarray(unique_keys[:, 1], dtype=np.int64)
    i1 = np.asarray(unique_keys[:, 2], dtype=np.int64)
    i2 = np.asarray(unique_keys[:, 3], dtype=np.int64)
    min_depth = int(np.min(depths))
    level_shapes = _depth_shapes_for_cutoff(tree, min_depth, key)
    periodic_i2 = str(tree.tree_coord) == "rpa"
    face_counts, face_offsets, face_neighbors = build_topological_neighborhood_kernel(
        depths,
        i0,
        i1,
        i2,
        min_depth,
        key,
        level_shapes,
        periodic_i2,
    )
    topology = TopologicalNeighborhood(
        levels=depths,
        i0=i0,
        i1=i1,
        i2=i2,
        face_counts=face_counts,
        face_offsets=face_offsets,
        face_neighbors=face_neighbors,
        node_cell_ids=node_cell_ids,
        cell_to_node_id=cell_to_node_id,
        min_level=min_depth,
        max_level=key,
        periodic_i2=periodic_i2,
    )
    cache[key] = topology
    return topology


def _topology_state_for_node_cells(topology) -> TopologicalKernelState:
    """Return one topology state whose active cell ids are the frontier node ids."""
    node_ids = np.arange(int(topology.node_count), dtype=np.int64)
    return TopologicalKernelState(
        face_offsets=np.asarray(topology.face_offsets, dtype=np.int64),
        face_neighbors=np.asarray(topology.face_neighbors, dtype=np.int64),
        node_cell_ids=node_ids,
        cell_to_node_id=np.asarray(topology.cell_to_node_id, dtype=np.int64),
    )


def _node_point_candidates(tree: Octree, topology) -> list[np.ndarray]:
    """Return unique descendant-corner point ids for every frontier node."""
    tree._require_lookup()
    if tree.cell_levels is None:
        raise ValueError("Octree has no cell levels; cannot build truncated ray cells.")
    corners = np.asarray(getattr(tree, "_corners"), dtype=np.int64)
    cell_levels = np.asarray(tree.cell_levels, dtype=np.int64)
    valid_cells = np.flatnonzero(cell_levels >= 0).astype(np.int64)
    node_ids = np.asarray(topology.cell_to_node_id[valid_cells], dtype=np.int64)
    order = np.argsort(node_ids, kind="stable")
    node_ids = node_ids[order]
    valid_cells = valid_cells[order]
    groups: list[np.ndarray] = [np.empty((0,), dtype=np.int64) for _ in range(int(topology.node_count))]
    start = 0
    while start < int(node_ids.size):
        stop = start + 1
        node_id = int(node_ids[start])
        while stop < int(node_ids.size) and int(node_ids[stop]) == node_id:
            stop += 1
        groups[node_id] = np.unique(corners[valid_cells[start:stop]].reshape(-1)).astype(np.int64)
        start = stop
    return groups


def _ordered_corners_from_unit_coords(point_ids: np.ndarray, unit_coords: np.ndarray) -> np.ndarray:
    """Pick 8 logical corner point ids by nearest unit-cube corners."""
    d2 = np.sum((unit_coords[:, None, :] - _TARGET_UNIT_CORNERS[None, :, :]) ** 2, axis=2)
    picks = np.argmin(d2, axis=0).astype(np.int64)
    return np.asarray(point_ids[picks], dtype=np.int64)


def _ordered_spherical_corners_from_targets(
    point_ids: np.ndarray,
    point_xyz: np.ndarray,
    r0: float,
    r1: float,
    theta0: float,
    theta1: float,
    phi0: float,
    phi1: float,
) -> np.ndarray:
    """Pick 8 spherical logical corners by nearest exact coarse-corner target in `xyz`."""
    out = np.empty(8, dtype=np.int64)
    r_targets = (float(r0), float(r1))
    theta_targets = (float(theta0), float(theta1))
    phi_targets = (float(phi0), float(phi1))

    for k in range(8):
        r = r_targets[k & 1]
        theta = theta_targets[(k >> 1) & 1]
        phi = phi_targets[(k >> 2) & 1]
        s = math.sin(theta)
        target = np.array(
            [
                r * s * math.cos(phi),
                r * s * math.sin(phi),
                r * math.cos(theta),
            ],
            dtype=np.float64,
        )
        d2 = np.sum((point_xyz - target[None, :]) ** 2, axis=1)
        out[k] = int(point_ids[int(np.argmin(d2))])
    return out


@njit(cache=True, parallel=True)
def _ordered_spherical_corners_from_targets_kernel(
    points: np.ndarray,
    corners: np.ndarray,
    cell_r_min: np.ndarray,
    cell_r_max: np.ndarray,
    cell_theta_min: np.ndarray,
    cell_theta_max: np.ndarray,
    cell_phi_start: np.ndarray,
    cell_phi_width: np.ndarray,
) -> np.ndarray:
    """Return ordered spherical corners for every cell from exact xyz corner targets."""
    n_cells = int(corners.shape[0])
    ordered = np.empty_like(corners)

    for cid in prange(n_cells):
        r0 = float(cell_r_min[cid])
        r1 = float(cell_r_max[cid])
        theta0 = float(cell_theta_min[cid])
        theta1 = float(cell_theta_max[cid])
        phi0 = float(cell_phi_start[cid])
        phi1 = float(phi0 + cell_phi_width[cid])

        for k in range(8):
            r = r0 if (k & 1) == 0 else r1
            theta = theta0 if ((k >> 1) & 1) == 0 else theta1
            phi = phi0 if ((k >> 2) & 1) == 0 else phi1
            s = math.sin(theta)
            tx = r * s * math.cos(phi)
            ty = r * s * math.sin(phi)
            tz = r * math.cos(theta)

            best_j = 0
            best_d2 = math.inf
            for j in range(8):
                pid = int(corners[cid, j])
                dx = float(points[pid, 0] - tx)
                dy = float(points[pid, 1] - ty)
                dz = float(points[pid, 2] - tz)
                d2 = dx * dx + dy * dy + dz * dz
                if d2 < best_d2:
                    best_d2 = d2
                    best_j = j
            ordered[cid, k] = corners[cid, best_j]
    return ordered


@njit(cache=True, parallel=True)
def _build_plane_state_from_ordered_corners_kernel(
    points: np.ndarray,
    corners: np.ndarray,
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build face planes from ordered hexahedron corner ids."""
    n_cells = int(corners.shape[0])
    face_normals = np.zeros((n_cells, 6, 3), dtype=np.float64)
    face_offsets = np.zeros((n_cells, 6), dtype=np.float64)
    face_valid = np.zeros((n_cells, 6), dtype=np.bool_)
    tiny = 1.0e-24

    for cid in prange(n_cells):
        center0 = float(centers[cid, 0])
        center1 = float(centers[cid, 1])
        center2 = float(centers[cid, 2])

        for face in range(6):
            local0 = int(_ORDERED_FACE_CORNERS_ARRAY[face, 0])
            local1 = int(_ORDERED_FACE_CORNERS_ARRAY[face, 1])
            local2 = int(_ORDERED_FACE_CORNERS_ARRAY[face, 2])
            local3 = int(_ORDERED_FACE_CORNERS_ARRAY[face, 3])

            pid0 = int(corners[cid, local0])
            pid1 = int(corners[cid, local1])
            pid2 = int(corners[cid, local2])
            pid3 = int(corners[cid, local3])

            face_pts = np.empty((4, 3), dtype=np.float64)
            face_pts[0, 0] = points[pid0, 0]
            face_pts[0, 1] = points[pid0, 1]
            face_pts[0, 2] = points[pid0, 2]
            face_pts[1, 0] = points[pid1, 0]
            face_pts[1, 1] = points[pid1, 1]
            face_pts[1, 2] = points[pid1, 2]
            face_pts[2, 0] = points[pid2, 0]
            face_pts[2, 1] = points[pid2, 1]
            face_pts[2, 2] = points[pid2, 2]
            face_pts[3, 0] = points[pid3, 0]
            face_pts[3, 1] = points[pid3, 1]
            face_pts[3, 2] = points[pid3, 2]

            best_norm2 = -1.0
            best_nx = 0.0
            best_ny = 0.0
            best_nz = 0.0
            best_ax = 0.0
            best_ay = 0.0
            best_az = 0.0

            for triple in range(4):
                ia = int(_FACE_TRIPLES_ARRAY[triple, 0])
                ib = int(_FACE_TRIPLES_ARRAY[triple, 1])
                ic = int(_FACE_TRIPLES_ARRAY[triple, 2])

                ax = float(face_pts[ia, 0])
                ay = float(face_pts[ia, 1])
                az = float(face_pts[ia, 2])
                bx = float(face_pts[ib, 0])
                by = float(face_pts[ib, 1])
                bz = float(face_pts[ib, 2])
                cx = float(face_pts[ic, 0])
                cy = float(face_pts[ic, 1])
                cz = float(face_pts[ic, 2])

                abx = bx - ax
                aby = by - ay
                abz = bz - az
                acx = cx - ax
                acy = cy - ay
                acz = cz - az

                nx = aby * acz - abz * acy
                ny = abz * acx - abx * acz
                nz = abx * acy - aby * acx
                norm2 = nx * nx + ny * ny + nz * nz
                if norm2 > best_norm2:
                    best_norm2 = norm2
                    best_nx = nx
                    best_ny = ny
                    best_nz = nz
                    best_ax = ax
                    best_ay = ay
                    best_az = az

            if best_norm2 <= tiny:
                continue

            inv_norm = 1.0 / math.sqrt(best_norm2)
            best_nx = best_nx * inv_norm
            best_ny = best_ny * inv_norm
            best_nz = best_nz * inv_norm

            orient = best_nx * (center0 - best_ax) + best_ny * (center1 - best_ay) + best_nz * (center2 - best_az)
            if orient > 0.0:
                best_nx = -best_nx
                best_ny = -best_ny
                best_nz = -best_nz

            face_normals[cid, face, 0] = best_nx
            face_normals[cid, face, 1] = best_ny
            face_normals[cid, face, 2] = best_nz
            face_offsets[cid, face] = best_nx * best_ax + best_ny * best_ay + best_nz * best_az
            face_valid[cid, face] = True

    return face_normals, face_offsets, face_valid


def _build_plane_state_from_ordered_corners(
    points: np.ndarray,
    corners: np.ndarray,
    centers: np.ndarray,
) -> CellPlaneKernelState:
    """Build plane faces from ordered hexahedron corner ids."""
    face_normals, face_offsets, face_valid = _build_plane_state_from_ordered_corners_kernel(points, corners, centers)
    return CellPlaneKernelState(
        face_normals=face_normals,
        face_offsets=face_offsets,
        face_valid=face_valid,
    )


def _build_cell_plane_kernel_state(tree: Octree) -> CellPlaneKernelState:
    """Build one Cartesian plane model from topology-aligned cell faces."""
    tree._require_lookup()
    corners = np.asarray(getattr(tree, "_corners"), dtype=np.int64)
    points = np.asarray(getattr(tree, "_points"), dtype=float)
    centers = np.asarray(getattr(tree, "_cell_centers"), dtype=float)
    lookup_state = tree.lookup.lookup_state
    n_cells = int(corners.shape[0])

    face_normals = np.zeros((n_cells, 6, 3), dtype=np.float64)
    face_offsets = np.zeros((n_cells, 6), dtype=np.float64)
    face_valid = np.zeros((n_cells, 6), dtype=np.bool_)
    tiny = 1.0e-24
    tree_coord = str(tree.tree_coord)

    if tree_coord == "rpa":
        if not isinstance(lookup_state, SphericalLookupKernelState):
            raise TypeError(
                "Spherical cell-plane construction requires SphericalLookupKernelState; "
                f"got {type(lookup_state).__name__}."
            )
        ordered_corners = _ordered_spherical_corners_from_targets_kernel(
            points,
            corners,
            np.asarray(lookup_state.cell_r_min, dtype=np.float64),
            np.asarray(lookup_state.cell_r_max, dtype=np.float64),
            np.asarray(lookup_state.cell_theta_min, dtype=np.float64),
            np.asarray(lookup_state.cell_theta_max, dtype=np.float64),
            np.asarray(lookup_state.cell_phi_start, dtype=np.float64),
            np.asarray(lookup_state.cell_phi_width, dtype=np.float64),
        )
        return _build_plane_state_from_ordered_corners(points, ordered_corners, centers)

    def _closest_four(values: np.ndarray, target: float, tol: float) -> np.ndarray:
        mask = np.abs(values - target) <= tol
        idx = np.flatnonzero(mask)
        if idx.size == 4:
            return idx
        order = np.argsort(np.abs(values - target))
        return np.asarray(order[:4], dtype=np.int64)

    for cid in range(n_cells):
        cell_pts = np.asarray(points[corners[cid]], dtype=float)
        center = np.asarray(centers[cid], dtype=float)
        if tree_coord == "xyz":
            if not isinstance(lookup_state, CartesianLookupKernelState):
                raise TypeError(
                    "Cartesian cell-plane construction requires CartesianLookupKernelState; "
                    f"got {type(lookup_state).__name__}."
                )
            cx = cell_pts[:, 0]
            cy = cell_pts[:, 1]
            cz = cell_pts[:, 2]
            tx = max(1e-10, 1e-8 * float(lookup_state.cell_x_max[cid] - lookup_state.cell_x_min[cid]))
            ty = max(1e-10, 1e-8 * float(lookup_state.cell_y_max[cid] - lookup_state.cell_y_min[cid]))
            tz = max(1e-10, 1e-8 * float(lookup_state.cell_z_max[cid] - lookup_state.cell_z_min[cid]))
            face_corner_sets = (
                _closest_four(cx, float(lookup_state.cell_x_min[cid]), tx),
                _closest_four(cx, float(lookup_state.cell_x_max[cid]), tx),
                _closest_four(cy, float(lookup_state.cell_y_min[cid]), ty),
                _closest_four(cy, float(lookup_state.cell_y_max[cid]), ty),
                _closest_four(cz, float(lookup_state.cell_z_min[cid]), tz),
                _closest_four(cz, float(lookup_state.cell_z_max[cid]), tz),
            )
        else:
            raise ValueError(f"Unsupported tree_coord '{tree_coord}'.")

        for face, node_ids in enumerate(face_corner_sets):
            face_pts = np.asarray(cell_pts[np.asarray(node_ids, dtype=np.int64)], dtype=float)
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

    return CellPlaneKernelState(
        face_normals=face_normals,
        face_offsets=face_offsets,
        face_valid=face_valid,
    )


def _build_cartesian_ray_cell_geometry(tree: Octree, topology) -> CartesianRayCellGeometry:
    """Build truncated Cartesian ray-cell geometry from one frontier topology."""
    points = np.asarray(tree.lookup.points, dtype=float)
    point_groups = _node_point_candidates(tree, topology)
    n_cells = int(topology.node_count)

    corners = np.empty((n_cells, 8), dtype=np.int64)
    bin_to_corner = np.broadcast_to(np.arange(8, dtype=np.int64), (n_cells, 8)).copy()
    cell_x0 = np.empty(n_cells, dtype=np.float64)
    cell_xden = np.empty(n_cells, dtype=np.float64)
    cell_y0 = np.empty(n_cells, dtype=np.float64)
    cell_yden = np.empty(n_cells, dtype=np.float64)
    cell_z0 = np.empty(n_cells, dtype=np.float64)
    cell_zden = np.empty(n_cells, dtype=np.float64)
    cell_x1 = np.empty(n_cells, dtype=np.float64)
    cell_y1 = np.empty(n_cells, dtype=np.float64)
    cell_z1 = np.empty(n_cells, dtype=np.float64)
    centers = np.empty((n_cells, 3), dtype=np.float64)
    tiny = np.finfo(float).tiny

    for node_id in range(n_cells):
        point_ids = np.asarray(point_groups[node_id], dtype=np.int64)
        cell_pts = np.asarray(points[point_ids], dtype=float)
        x0 = float(np.min(cell_pts[:, 0]))
        x1 = float(np.max(cell_pts[:, 0]))
        y0 = float(np.min(cell_pts[:, 1]))
        y1 = float(np.max(cell_pts[:, 1]))
        z0 = float(np.min(cell_pts[:, 2]))
        z1 = float(np.max(cell_pts[:, 2]))
        dx = max(x1 - x0, tiny)
        dy = max(y1 - y0, tiny)
        dz = max(z1 - z0, tiny)
        unit = np.empty((int(point_ids.size), 3), dtype=np.float64)
        unit[:, 0] = np.clip((cell_pts[:, 0] - x0) / dx, 0.0, 1.0)
        unit[:, 1] = np.clip((cell_pts[:, 1] - y0) / dy, 0.0, 1.0)
        unit[:, 2] = np.clip((cell_pts[:, 2] - z0) / dz, 0.0, 1.0)
        corners[node_id, :] = _ordered_corners_from_unit_coords(point_ids, unit)
        centers[node_id, :] = np.mean(points[corners[node_id]], axis=0)
        cell_x0[node_id] = x0
        cell_x1[node_id] = x1
        cell_y0[node_id] = y0
        cell_y1[node_id] = y1
        cell_z0[node_id] = z0
        cell_z1[node_id] = z1
        cell_xden[node_id] = dx
        cell_yden[node_id] = dy
        cell_zden[node_id] = dz

    seed_lookup = tree.lookup.lookup_state
    if not isinstance(seed_lookup, CartesianLookupKernelState):
        raise TypeError(
            "Cartesian truncated ray geometry requires CartesianLookupKernelState; "
            f"got {type(seed_lookup).__name__}."
        )
    lookup_state = CartesianLookupKernelState(
        cell_centers=centers,
        cell_x_min=cell_x0,
        cell_x_max=cell_x1,
        cell_y_min=cell_y0,
        cell_y_max=cell_y1,
        cell_z_min=cell_z0,
        cell_z_max=cell_z1,
        cell_valid=np.ones(n_cells, dtype=np.bool_),
        xyz_min=np.asarray(seed_lookup.xyz_min, dtype=np.float64),
        xyz_max=np.asarray(seed_lookup.xyz_max, dtype=np.float64),
        xyz_span=np.asarray(seed_lookup.xyz_span, dtype=np.float64),
        bin_shape=np.array([1, 1, 1], dtype=np.int64),
        bin_offsets=np.zeros(2, dtype=np.int64),
        bin_cell_ids=np.empty((0,), dtype=np.int64),
        max_radius=0,
    )
    return CartesianRayCellGeometry(
        topology_state=_topology_state_for_node_cells(topology),
        lookup_state=lookup_state,
        corners=corners,
        bin_to_corner=bin_to_corner,
        cell_x0=cell_x0,
        cell_xden=cell_xden,
        cell_y0=cell_y0,
        cell_yden=cell_yden,
        cell_z0=cell_z0,
        cell_zden=cell_zden,
    )


def _build_spherical_ray_cell_geometry(tree: Octree, topology) -> SphericalRayCellGeometry:
    """Build truncated spherical ray-cell geometry from one frontier topology."""
    points = np.asarray(tree.lookup.points, dtype=float)
    point_groups = _node_point_candidates(tree, topology)
    point_r = np.linalg.norm(points, axis=1)
    point_phi = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * math.pi)

    n_cells = int(topology.node_count)
    corners = np.empty((n_cells, 8), dtype=np.int64)
    bin_to_corner = np.broadcast_to(np.arange(8, dtype=np.int64), (n_cells, 8)).copy()
    centers = np.empty((n_cells, 3), dtype=np.float64)
    cell_r0 = np.empty(n_cells, dtype=np.float64)
    cell_rden = np.empty(n_cells, dtype=np.float64)
    cell_t0 = np.empty(n_cells, dtype=np.float64)
    cell_tden = np.empty(n_cells, dtype=np.float64)
    cell_p_start = np.empty(n_cells, dtype=np.float64)
    cell_p_width = np.empty(n_cells, dtype=np.float64)
    cell_pden = np.empty(n_cells, dtype=np.float64)
    cell_phi_full = np.zeros(n_cells, dtype=np.bool_)
    cell_phi_tiny = np.zeros(n_cells, dtype=np.bool_)
    tiny = np.finfo(float).tiny

    for node_id in range(n_cells):
        point_ids = np.asarray(point_groups[node_id], dtype=np.int64)
        cell_pts = np.asarray(points[point_ids], dtype=float)
        depth = int(topology.levels[node_id])
        _nr, ntheta, nphi = _shape_for_ray_depth(tree, depth)
        dtheta = math.pi / float(ntheta)
        dphi = 2.0 * math.pi / float(nphi)
        theta0 = float(int(topology.i1[node_id]) * dtheta)
        theta1 = theta0 + dtheta
        phi_start = float(int(topology.i2[node_id]) * dphi)
        phi_width = float(dphi)

        cr = point_r[point_ids]
        cp = np.array(point_phi[point_ids], dtype=float, copy=True)
        center_phi = phi_start + 0.5 * phi_width
        cp[cp < (center_phi - math.pi)] += 2.0 * math.pi
        cp[cp > (center_phi + math.pi)] -= 2.0 * math.pi

        r0 = float(np.min(cr))
        r1 = float(np.max(cr))
        dr = max(r1 - r0, tiny)
        dt = max(theta1 - theta0, tiny)
        dp = max(phi_width, tiny)
        p_rel = np.clip(cp - phi_start, 0.0, phi_width)

        corners[node_id, :] = _ordered_spherical_corners_from_targets(
            point_ids,
            cell_pts,
            r0,
            r1,
            theta0,
            theta1,
            phi_start,
            phi_start + phi_width,
        )
        centers[node_id, :] = np.mean(points[corners[node_id]], axis=0)

        cell_r0[node_id] = r0
        cell_rden[node_id] = dr
        cell_t0[node_id] = theta0
        cell_tden[node_id] = dt
        cell_p_start[node_id] = phi_start
        cell_p_width[node_id] = phi_width
        cell_pden[node_id] = dp
        cell_phi_full[node_id] = phi_width >= (2.0 * math.pi - 1e-10)
        cell_phi_tiny[node_id] = phi_width <= tiny

    plane_state = _build_plane_state_from_ordered_corners(points, corners, centers)
    return SphericalRayCellGeometry(
        topology_state=_topology_state_for_node_cells(topology),
        plane_state=plane_state,
        cell_centers=centers,
        corners=corners,
        bin_to_corner=bin_to_corner,
        cell_r0=cell_r0,
        cell_rden=cell_rden,
        cell_t0=cell_t0,
        cell_tden=cell_tden,
        cell_p_start=cell_p_start,
        cell_p_width=cell_p_width,
        cell_pden=cell_pden,
        cell_phi_full=cell_phi_full,
        cell_phi_tiny=cell_phi_tiny,
    )


def _build_sparse_seed_plane_state(
    rep_cell_ids: np.ndarray,
    n_leaf_cells: int,
    dense_plane_state: CellPlaneKernelState,
) -> CellPlaneKernelState:
    """Expand dense node-indexed plane faces onto representative leaf ids for seed lookup."""
    face_normals = np.zeros((int(n_leaf_cells), 6, 3), dtype=np.float64)
    face_offsets = np.zeros((int(n_leaf_cells), 6), dtype=np.float64)
    face_valid = np.zeros((int(n_leaf_cells), 6), dtype=np.bool_)
    for node_id, rep_cid in enumerate(np.asarray(rep_cell_ids, dtype=np.int64)):
        cid = int(rep_cid)
        face_normals[cid, :, :] = dense_plane_state.face_normals[int(node_id), :, :]
        face_offsets[cid, :] = dense_plane_state.face_offsets[int(node_id), :]
        face_valid[cid, :] = dense_plane_state.face_valid[int(node_id), :]
    return CellPlaneKernelState(
        face_normals=face_normals,
        face_offsets=face_offsets,
        face_valid=face_valid,
    )


def _build_sparse_spherical_seed_lookup_state(
    tree: Octree,
    topology,
    geometry: SphericalRayCellGeometry,
) -> SphericalLookupKernelState:
    """Build one sparse coarse spherical lookup state keyed by representative leaf ids."""
    n_leaf_cells = int(np.asarray(tree.cell_levels, dtype=np.int64).shape[0])
    rep_cell_ids = np.asarray(topology.node_cell_ids, dtype=np.int64)
    depths = np.asarray(topology.levels, dtype=np.int64)
    depth_levels = np.array(sorted(set(int(v) for v in depths.tolist())), dtype=np.int64)
    levels_desc = depth_levels[::-1].copy()
    depth_cap = int(np.max(depth_levels)) + 1

    shape_table = np.full((depth_cap, 3), -1, dtype=np.int64)
    dtheta_table = np.full(depth_cap, np.nan, dtype=np.float64)
    dphi_table = np.full(depth_cap, np.nan, dtype=np.float64)
    bin_level_offset = np.full(depth_cap, -1, dtype=np.int64)
    running_offset = 0
    for depth in depth_levels:
        d = int(depth)
        nr, ntheta, nphi = _shape_for_ray_depth(tree, d)
        shape_table[d, 0] = int(nr)
        shape_table[d, 1] = int(ntheta)
        shape_table[d, 2] = int(nphi)
        dtheta_table[d] = math.pi / float(ntheta)
        dphi_table[d] = 2.0 * math.pi / float(nphi)
        bin_level_offset[d] = running_offset
        running_offset += int(ntheta) * int(nphi)

    bin_lists: list[list[int]] = [[] for _ in range(int(running_offset))]
    cell_r_center = np.linalg.norm(np.asarray(geometry.cell_centers, dtype=np.float64), axis=1)
    for node_id, rep_cid in enumerate(rep_cell_ids):
        depth = int(depths[node_id])
        nphi = int(shape_table[depth, 2])
        key = int(bin_level_offset[depth] + int(topology.i1[node_id]) * nphi + int(topology.i2[node_id]))
        bin_lists[key].append(int(rep_cid))

    bin_counts = np.zeros(int(running_offset), dtype=np.int64)
    for key, ids in enumerate(bin_lists):
        if not ids:
            continue
        arr = np.array(ids, dtype=np.int64)
        node_ids = np.array([int(topology.cell_to_node_id[cid]) for cid in arr], dtype=np.int64)
        order = np.argsort(cell_r_center[node_ids])
        sorted_ids = arr[order]
        bin_lists[key] = sorted_ids.tolist()
        bin_counts[key] = int(sorted_ids.size)

    bin_offsets = np.zeros(int(running_offset) + 1, dtype=np.int64)
    if int(running_offset) > 0:
        np.cumsum(bin_counts, out=bin_offsets[1:])
    bin_cell_ids = np.empty(int(bin_offsets[-1]), dtype=np.int64)
    for key in range(int(running_offset)):
        start = int(bin_offsets[key])
        end = int(bin_offsets[key + 1])
        if end <= start:
            continue
        bin_cell_ids[start:end] = np.array(bin_lists[key], dtype=np.int64)

    cell_r_min = np.zeros(n_leaf_cells, dtype=np.float64)
    cell_r_max = np.zeros(n_leaf_cells, dtype=np.float64)
    cell_theta_min = np.zeros(n_leaf_cells, dtype=np.float64)
    cell_theta_max = np.zeros(n_leaf_cells, dtype=np.float64)
    cell_phi_start = np.zeros(n_leaf_cells, dtype=np.float64)
    cell_phi_width = np.zeros(n_leaf_cells, dtype=np.float64)
    cell_valid = np.zeros(n_leaf_cells, dtype=np.bool_)
    cell_centers = np.zeros((n_leaf_cells, 3), dtype=np.float64)
    for node_id, rep_cid in enumerate(rep_cell_ids):
        cid = int(rep_cid)
        cell_r_min[cid] = float(geometry.cell_r0[node_id])
        cell_r_max[cid] = float(geometry.cell_r0[node_id] + geometry.cell_rden[node_id])
        cell_theta_min[cid] = float(geometry.cell_t0[node_id])
        cell_theta_max[cid] = float(geometry.cell_t0[node_id] + geometry.cell_tden[node_id])
        cell_phi_start[cid] = float(geometry.cell_p_start[node_id])
        cell_phi_width[cid] = float(geometry.cell_p_width[node_id])
        cell_valid[cid] = True
        cell_centers[cid, :] = geometry.cell_centers[node_id, :]

    seed_lookup = tree.lookup.lookup_state
    if not isinstance(seed_lookup, SphericalLookupKernelState):
        raise TypeError(
            "Spherical truncated ray geometry requires SphericalLookupKernelState; "
            f"got {type(seed_lookup).__name__}."
        )
    return SphericalLookupKernelState(
        levels_desc=levels_desc,
        shape_table=shape_table,
        dtheta_table=dtheta_table,
        dphi_table=dphi_table,
        bin_level_offset=bin_level_offset,
        bin_offsets=bin_offsets,
        bin_cell_ids=bin_cell_ids,
        cell_r_min=cell_r_min,
        cell_r_max=cell_r_max,
        cell_theta_min=cell_theta_min,
        cell_theta_max=cell_theta_max,
        cell_phi_start=cell_phi_start,
        cell_phi_width=cell_phi_width,
        cell_valid=cell_valid,
        cell_centers=cell_centers,
        r_min=float(np.min(geometry.cell_r0)),
        r_max=float(np.max(geometry.cell_r0 + geometry.cell_rden)),
        max_radius=int(seed_lookup.max_radius),
    )


def _cartesian_interp_state_from_geometry(
    point_values_2d: np.ndarray,
    geometry: CartesianRayCellGeometry,
) -> CartesianInterpKernelState:
    """Build one Cartesian interpolation state from truncated ray-cell geometry."""
    return CartesianInterpKernelState(
        point_values_2d=point_values_2d,
        corners=geometry.corners,
        bin_to_corner=geometry.bin_to_corner,
        cell_x0=geometry.cell_x0,
        cell_xden=geometry.cell_xden,
        cell_y0=geometry.cell_y0,
        cell_yden=geometry.cell_yden,
        cell_z0=geometry.cell_z0,
        cell_zden=geometry.cell_zden,
    )


def _spherical_interp_state_from_geometry(
    point_values_2d: np.ndarray,
    geometry: SphericalRayCellGeometry,
) -> SphericalInterpKernelState:
    """Build one spherical interpolation state from truncated ray-cell geometry."""
    return SphericalInterpKernelState(
        point_values_2d=point_values_2d,
        corners=geometry.corners,
        bin_to_corner=geometry.bin_to_corner,
        cell_r0=geometry.cell_r0,
        cell_rden=geometry.cell_rden,
        cell_t0=geometry.cell_t0,
        cell_tden=geometry.cell_tden,
        cell_p_start=geometry.cell_p_start,
        cell_p_width=geometry.cell_p_width,
        cell_pden=geometry.cell_pden,
        cell_phi_full=geometry.cell_phi_full,
        cell_phi_tiny=geometry.cell_phi_tiny,
    )


def _ray_cell_geometry_for_maxdepth(tree: Octree, maxdepth: int) -> CartesianRayCellGeometry | SphericalRayCellGeometry:
    """Return cached truncated ray-cell geometry for one depth cutoff."""
    cache = getattr(tree, "_ray_cell_geometry_by_maxdepth", None)
    if cache is None:
        cache = {}
        tree._ray_cell_geometry_by_maxdepth = cache
    key = int(maxdepth)
    if key in cache:
        return cache[key]

    topology = _topology_for_ray_maxdepth(tree, key)
    if str(tree.tree_coord) == "xyz":
        geometry = _build_cartesian_ray_cell_geometry(tree, topology)
    elif str(tree.tree_coord) == "rpa":
        geometry = _build_spherical_ray_cell_geometry(tree, topology)
    else:
        raise ValueError(f"Unsupported tree_coord '{tree.tree_coord}'.")
    cache[key] = geometry
    return geometry


def _ray_interp_state_for_maxdepth(
    interpolator: "OctreeInterpolator",
    maxdepth: int,
) -> CartesianInterpKernelState | SphericalInterpKernelState:
    """Return cached truncated interpolation state for one ray depth cutoff."""
    cache = getattr(interpolator, "_ray_interp_state_by_maxdepth", None)
    if cache is None:
        cache = {}
        interpolator._ray_interp_state_by_maxdepth = cache
    key = int(maxdepth)
    if key in cache:
        return cache[key]

    geometry = _ray_cell_geometry_for_maxdepth(interpolator.tree, key)
    point_values_2d = np.asarray(interpolator._point_values_2d, dtype=np.float64)
    if isinstance(geometry, CartesianRayCellGeometry):
        state = _cartesian_interp_state_from_geometry(point_values_2d, geometry)
    elif isinstance(geometry, SphericalRayCellGeometry):
        state = _spherical_interp_state_from_geometry(point_values_2d, geometry)
    else:
        raise TypeError(f"Unsupported truncated ray geometry {type(geometry).__name__}.")
    cache[key] = state
    return state


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
def _contains_cell_from_xyz(
    cid: int,
    x: float,
    y: float,
    z: float,
    plane_state: CellPlaneKernelState,
    tol: float = _TRACE_CONTAIN_TOL,
) -> bool:
    """Return whether one Cartesian point lies inside one plane-bounded cell."""
    for face in range(6):
        if not plane_state.face_valid[cid, face]:
            continue
        nx = plane_state.face_normals[cid, face, 0]
        ny = plane_state.face_normals[cid, face, 1]
        nz = plane_state.face_normals[cid, face, 2]
        offset = plane_state.face_offsets[cid, face]
        signed = nx * x + ny * y + nz * z - offset
        if signed > tol:
            return False
    return True


@njit(cache=True)
def _cell_exit_dt_and_mask(
    cid: int,
    x: float,
    y: float,
    z: float,
    d0: float,
    d1: float,
    d2: float,
    plane_state: CellPlaneKernelState,
    abs_eps: float,
) -> tuple[float, int]:
    """Return forward exit distance and exited-face mask for one plane-bounded cell."""
    best_dt = np.inf
    face_mask = 0
    boundary_mask = 0
    dir_eps = 1.0e-15
    tie_tol = max(4.0 * abs_eps, 1.0e-12)
    for face in range(6):
        if not plane_state.face_valid[cid, face]:
            continue
        nx = plane_state.face_normals[cid, face, 0]
        ny = plane_state.face_normals[cid, face, 1]
        nz = plane_state.face_normals[cid, face, 2]
        offset = plane_state.face_offsets[cid, face]
        signed = nx * x + ny * y + nz * z - offset
        denom = nx * d0 + ny * d1 + nz * d2
        if signed > tie_tol:
            return np.inf, 0
        if signed >= -tie_tol and denom > dir_eps:
            boundary_mask |= 1 << face
        if signed > 0.0:
            signed = 0.0
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
    if boundary_mask != 0:
        face_mask |= boundary_mask
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
    """Return next nodes reachable across any non-empty subset of exited faces."""
    work0[0] = int(current_node_id)
    n_reachable = 1
    for face in range(6):
        if (int(face_mask) & (1 << face)) == 0:
            continue
        n_new = _expand_topology_nodes_for_face(work0, n_reachable, face, topo_state, work1)
        for i in range(n_new):
            n_reachable = _append_unique_node(work0, n_reachable, int(work1[i]))
    n_candidates = 0
    for i in range(n_reachable):
        node_id = int(work0[i])
        if node_id == int(current_node_id):
            continue
        work0[n_candidates] = node_id
        n_candidates += 1
    return n_candidates, 0


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
def _select_next_cell_from_topology(
    current_node_id: int,
    face_mask: int,
    x_next: float,
    y_next: float,
    z_next: float,
    topo_state: TopologicalKernelState,
    plane_state: CellPlaneKernelState,
    work0: np.ndarray,
    work1: np.ndarray,
) -> tuple[int, int]:
    """Choose next `(node_id, cell_id)` from topology candidates."""
    n_candidates, active = _candidate_nodes_after_exit(current_node_id, face_mask, topo_state, work0, work1)
    if n_candidates <= 0:
        return -1, -1
    nodes = work0 if active == 0 else work1
    for i in range(n_candidates):
        node_id = int(nodes[i])
        cid = int(topo_state.node_cell_ids[node_id])
        if _contains_cell_from_xyz(cid, x_next, y_next, z_next, plane_state):
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
    plane_state: CellPlaneKernelState,
) -> tuple[bool, float, int, float, float, float, float, float, float]:
    """Seek first `t` where ray enters a valid plane-bounded cell."""
    t = t_start
    x = origin_xyz[0] + t * direction_xyz_unit[0]
    y = origin_xyz[1] + t * direction_xyz_unit[1]
    z = origin_xyz[2] + t * direction_xyz_unit[2]
    r, polar, azimuth = _xyz_to_rpa_components(x, y, z)
    cid = _lookup_rpa_cell_id_kernel(r, polar, azimuth, lookup_state, -1)
    if cid >= 0 and _contains_cell_from_xyz(cid, x, y, z, plane_state):
        return True, t, cid, x, y, z, r, polar, azimuth

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
        if cid_hi >= 0 and _contains_cell_from_xyz(cid_hi, xh, yh, zh, plane_state):
            break
        if hi >= t_end:
            return False, t_start, -1, x, y, z, r, polar, azimuth
        dt = dt * 2.0
    if cid_hi < 0:
        return False, t_start, -1, x, y, z, r, polar, azimuth

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
        if cid_mid >= 0 and _contains_cell_from_xyz(cid_mid, xm, ym, zm, plane_state):
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
    if cid < 0 or not _contains_cell_from_xyz(cid, x, y, z, plane_state):
        t = hi_in
        x = origin_xyz[0] + t * direction_xyz_unit[0]
        y = origin_xyz[1] + t * direction_xyz_unit[1]
        z = origin_xyz[2] + t * direction_xyz_unit[2]
        r, polar, azimuth = _xyz_to_rpa_components(x, y, z)
        cid = _lookup_rpa_cell_id_kernel(r, polar, azimuth, lookup_state, cid_in)
    if cid < 0 or not _contains_cell_from_xyz(cid, x, y, z, plane_state):
        return False, t_start, -1, x, y, z, r, polar, azimuth
    return True, t, cid, x, y, z, r, polar, azimuth


@njit(cache=True)
def _ray_sphere_interval(
    origin_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    radius: float,
    t_start: float,
    t_end: float,
) -> tuple[bool, float, float]:
    """Return clipped ray/sphere interval for one radius."""
    if radius <= 0.0 or t_end <= t_start:
        return False, 0.0, 0.0

    ox = float(origin_xyz[0])
    oy = float(origin_xyz[1])
    oz = float(origin_xyz[2])
    dx = float(direction_xyz_unit[0])
    dy = float(direction_xyz_unit[1])
    dz = float(direction_xyz_unit[2])
    b = ox * dx + oy * dy + oz * dz
    c = ox * ox + oy * oy + oz * oz - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return False, 0.0, 0.0

    root = math.sqrt(max(disc, 0.0))
    t0 = -b - root
    t1 = -b + root
    if t1 <= t_start or t0 >= t_end:
        return False, 0.0, 0.0
    if t0 < t_start:
        t0 = t_start
    if t1 > t_end:
        t1 = t_end
    if t1 <= t0:
        return False, 0.0, 0.0
    return True, t0, t1


@njit(cache=True)
def _front_material_interval_rpa(
    origin_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    r_min: float,
    r_max: float,
) -> tuple[bool, float, float]:
    """Return the first connected shell interval along the ray."""
    outer_hit, outer_t0, outer_t1 = _ray_sphere_interval(origin_xyz, direction_xyz_unit, r_max, t_start, t_end)
    if not outer_hit:
        return False, 0.0, 0.0
    if r_min <= 0.0:
        return True, outer_t0, outer_t1

    inner_hit, inner_t0, inner_t1 = _ray_sphere_interval(origin_xyz, direction_xyz_unit, r_min, t_start, t_end)
    if not inner_hit or inner_t1 <= outer_t0 or inner_t0 >= outer_t1:
        return True, outer_t0, outer_t1
    if outer_t0 < inner_t0:
        return True, outer_t0, inner_t0
    if inner_t1 < outer_t1:
        return True, inner_t1, outer_t1
    return False, 0.0, 0.0


@njit(cache=True)
def _approx_front_boundary_segments_rpa(
    origin_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    lookup_state: SphericalLookupKernelState,
    max_steps: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Approximate a missed front shell interval with one or two lookup-cell segments."""
    cell_ids = np.empty(max_steps, dtype=np.int64)
    enters = np.empty(max_steps, dtype=np.float64)
    exits = np.empty(max_steps, dtype=np.float64)
    if max_steps <= 0:
        return 0, cell_ids, enters, exits

    hit, t0, t1 = _front_material_interval_rpa(
        origin_xyz,
        direction_xyz_unit,
        t_start,
        t_end,
        float(lookup_state.r_min),
        float(lookup_state.r_max),
    )
    if not hit or t1 <= t0:
        return 0, cell_ids, enters, exits

    tm = 0.5 * (t0 + t1)
    tq0 = t0 + 0.25 * (t1 - t0)
    tq1 = t0 + 0.75 * (t1 - t0)

    x = origin_xyz[0] + tq0 * direction_xyz_unit[0]
    y = origin_xyz[1] + tq0 * direction_xyz_unit[1]
    z = origin_xyz[2] + tq0 * direction_xyz_unit[2]
    r, polar, azimuth = _xyz_to_rpa_components(x, y, z)
    cid0 = _lookup_rpa_cell_id_kernel(r, polar, azimuth, lookup_state, -1)

    x = origin_xyz[0] + tm * direction_xyz_unit[0]
    y = origin_xyz[1] + tm * direction_xyz_unit[1]
    z = origin_xyz[2] + tm * direction_xyz_unit[2]
    r, polar, azimuth = _xyz_to_rpa_components(x, y, z)
    cidm = _lookup_rpa_cell_id_kernel(r, polar, azimuth, lookup_state, -1)

    x = origin_xyz[0] + tq1 * direction_xyz_unit[0]
    y = origin_xyz[1] + tq1 * direction_xyz_unit[1]
    z = origin_xyz[2] + tq1 * direction_xyz_unit[2]
    r, polar, azimuth = _xyz_to_rpa_components(x, y, z)
    cid1 = _lookup_rpa_cell_id_kernel(r, polar, azimuth, lookup_state, -1)

    left_cid = cid0 if cid0 >= 0 else cidm
    right_cid = cid1 if cid1 >= 0 else cidm
    if left_cid < 0 and right_cid < 0:
        return 0, cell_ids, enters, exits

    if left_cid >= 0 and right_cid >= 0 and left_cid != right_cid and max_steps >= 2:
        cell_ids[0] = left_cid
        enters[0] = t0
        exits[0] = tm
        cell_ids[1] = right_cid
        enters[1] = tm
        exits[1] = t1
        return 2, cell_ids, enters, exits

    cid = left_cid if left_cid >= 0 else right_cid
    cell_ids[0] = cid
    enters[0] = t0
    exits[0] = t1
    return 1, cell_ids, enters, exits


@njit(cache=True)
def _trace_segments_xyz_kernel(
    origin_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    boundary_tol: float,
    seed_lookup_state: CartesianLookupKernelState,
    cell_lookup_state: CartesianLookupKernelState,
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
        seed_lookup_state.xyz_min,
        seed_lookup_state.xyz_max,
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
        seed_lookup_state,
    )
    if not found or cid < 0:
        return 0, cell_ids, enters, exits

    current_node = int(topo_state.cell_to_node_id[cid])
    if current_node < 0:
        return 0, cell_ids, enters, exits
    cid = int(topo_state.node_cell_ids[current_node])
    if cid < 0:
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

        if not _contains_xyz_from_state(cid, x, y, z, cell_lookup_state):
            break

        tx = _forward_face_exit_dt(
            x,
            d0,
            cell_lookup_state.cell_x_min[cid],
            cell_lookup_state.cell_x_max[cid],
            abs_eps,
            dir_eps,
        )
        ty = _forward_face_exit_dt(
            y,
            d1,
            cell_lookup_state.cell_y_min[cid],
            cell_lookup_state.cell_y_max[cid],
            abs_eps,
            dir_eps,
        )
        tz = _forward_face_exit_dt(
            z,
            d2,
            cell_lookup_state.cell_z_min[cid],
            cell_lookup_state.cell_z_max[cid],
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
            cell_lookup_state,
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
                cell_lookup_state,
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
    seed_lookup_state: SphericalLookupKernelState,
    topo_state: TopologicalKernelState,
    seed_plane_state: CellPlaneKernelState,
    plane_state: CellPlaneKernelState,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Trace one Cartesian ray on a plane-bounded cell tree; return segment arrays."""
    cell_ids = np.empty(max_steps, dtype=np.int64)
    enters = np.empty(max_steps, dtype=np.float64)
    exits = np.empty(max_steps, dtype=np.float64)

    if t_end <= t_start or max_steps <= 0:
        return 0, cell_ids, enters, exits

    abs_eps = max(boundary_tol * (1.0 + abs(t_end - t_start)), 1e-12)
    found, t, cid, x, y, z, r, polar, azimuth = _seek_first_cell_rpa(
        origin_xyz,
        direction_xyz_unit,
        t_start,
        t_end,
        abs_eps,
        seed_lookup_state,
        seed_plane_state,
    )
    if not found or cid < 0:
        return _approx_front_boundary_segments_rpa(
            origin_xyz,
            direction_xyz_unit,
            t_start,
            t_end,
            seed_lookup_state,
            max_steps,
        )

    current_node = int(topo_state.cell_to_node_id[cid])
    if current_node < 0:
        return 0, cell_ids, enters, exits
    cid = int(topo_state.node_cell_ids[current_node])
    if cid < 0:
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

        if not _contains_cell_from_xyz(cid, x, y, z, plane_state):
            break

        dt_exit, face_mask = _cell_exit_dt_and_mask(
            cid,
            x,
            y,
            z,
            d0,
            d1,
            d2,
            plane_state,
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
        next_node, cid_next = _select_next_cell_from_topology(
            current_node,
            face_mask,
            x_next,
            y_next,
            z_next,
            topo_state,
            plane_state,
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
            next_node, cid_next = _select_next_cell_from_topology(
                current_node,
                face_mask,
                x_next,
                y_next,
                z_next,
                topo_state,
                plane_state,
                candidate_nodes0,
                candidate_nodes1,
            )
            if cid_next < 0:
                break

        t = t_next
        x = x_next
        y = y_next
        z = z_next
        r, polar, azimuth = _xyz_to_rpa_components(x, y, z)
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
    seed_lookup_state: CartesianLookupKernelState,
    cell_lookup_state: CartesianLookupKernelState,
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
            seed_lookup_state,
            cell_lookup_state,
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
    seed_lookup_state: SphericalLookupKernelState,
    topo_state: TopologicalKernelState,
    seed_plane_state: CellPlaneKernelState,
    plane_state: CellPlaneKernelState,
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
            seed_lookup_state,
            topo_state,
            seed_plane_state,
            plane_state,
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
    seed_lookup_state: CartesianLookupKernelState,
    cell_lookup_state: CartesianLookupKernelState,
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
            seed_lookup_state,
            cell_lookup_state,
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
    seed_lookup_state: SphericalLookupKernelState,
    topo_state: TopologicalKernelState,
    seed_plane_state: CellPlaneKernelState,
    plane_state: CellPlaneKernelState,
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
            seed_lookup_state,
            topo_state,
            seed_plane_state,
            plane_state,
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
    seed_lookup_state: CartesianLookupKernelState,
    cell_lookup_state: CartesianLookupKernelState,
    topo_state: TopologicalKernelState,
    interp_state: CartesianInterpKernelState,
) -> np.ndarray:
    """Integrate fields along rays on Cartesian trees with 2-point Gauss segments."""
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
            seed_lookup_state,
            cell_lookup_state,
            topo_state,
        )
        if n_seg <= 0:
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
            th = 0.5 * dt
            tg0 = tm - th * _GAUSS_LEGENDRE_2_ABSCISSA
            tg1 = tm + th * _GAUSS_LEGENDRE_2_ABSCISSA
            x = ox + tg0 * d0
            y = oy + tg0 * d1
            z = oz + tg0 * d2
            _trilinear_from_cell(
                row,
                int(cids[j]),
                x,
                y,
                z,
                interp_state,
            )
            acc += row * th
            x = ox + tg1 * d0
            y = oy + tg1 * d1
            z = oz + tg1 * d2
            _trilinear_from_cell(
                row,
                int(cids[j]),
                x,
                y,
                z,
                interp_state,
            )
            acc += row * th
            used = True
        if used:
            out[i, :] = scale * acc
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
    seed_lookup_state: SphericalLookupKernelState,
    topo_state: TopologicalKernelState,
    seed_plane_state: CellPlaneKernelState,
    plane_state: CellPlaneKernelState,
    interp_state: SphericalInterpKernelState,
) -> np.ndarray:
    """Integrate fields along rays on spherical trees with 2-point Gauss segments."""
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
            seed_lookup_state,
            topo_state,
            seed_plane_state,
            plane_state,
        )
        if n_seg <= 0:
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
            th = 0.5 * dt
            tg0 = tm - th * _GAUSS_LEGENDRE_2_ABSCISSA
            tg1 = tm + th * _GAUSS_LEGENDRE_2_ABSCISSA
            x = ox + tg0 * d0
            y = oy + tg0 * d1
            z = oz + tg0 * d2
            r, polar, azimuth = _xyz_to_rpa_components(x, y, z)
            _trilinear_from_cell_rpa(
                row,
                int(cids[j]),
                r,
                polar,
                azimuth,
                interp_state,
            )
            acc += row * th
            x = ox + tg1 * d0
            y = oy + tg1 * d1
            z = oz + tg1 * d2
            r, polar, azimuth = _xyz_to_rpa_components(x, y, z)
            _trilinear_from_cell_rpa(
                row,
                int(cids[j]),
                r,
                polar,
                azimuth,
                interp_state,
            )
            acc += row * th
            used = True
        if used:
            out[i, :] = scale * acc
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


def _coerce_directions_xyz(direction_xyz: np.ndarray, *, n_rays: int) -> tuple[np.ndarray, bool]:
    """Validate directions and return `(directions, shared)`."""
    direction = np.asarray(direction_xyz, dtype=float)
    if direction.ndim == 1:
        return _normalize_direction(direction), True
    if direction.ndim != 2 or direction.shape[1] != 3:
        raise ValueError("direction_xyz must have shape (3,) or (n_rays, 3).")
    if int(direction.shape[0]) != int(n_rays):
        raise ValueError("direction_xyz with shape (n_rays, 3) must match origins_xyz.")
    norms = np.linalg.norm(direction, axis=1)
    if not np.all(np.isfinite(direction)) or np.any(norms == 0.0):
        raise ValueError("direction_xyz must be finite and non-zero.")
    return direction / norms[:, None], False


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

    def __init__(self, tree: Octree, *, maxdepth: int | None = None) -> None:
        """Bind one built octree."""
        self.tree = tree
        self.maxdepth = _resolve_ray_maxdepth(tree, maxdepth)
        self._tree_coord = str(tree.tree_coord)
        dmin, dmax = tree.domain_bounds(coord="xyz")
        self._domain_xyz_min = np.asarray(dmin, dtype=float).reshape(3)
        self._domain_xyz_max = np.asarray(dmax, dtype=float).reshape(3)
        _r_lo, r_hi = tree.domain_bounds(coord="rpa")
        self._domain_r_max = float(np.asarray(r_hi, dtype=float).reshape(3)[0])
        self._seed_lookup_state = tree.lookup.lookup_state
        self._seed_cell_plane_state = None

        if int(self.maxdepth) >= int(tree.depth):
            topology = getattr(tree, "_ray_topology_full", None)
            if topology is None or int(topology.max_level) != int(tree.max_level):
                topology = build_topological_neighborhood(tree, max_level=int(tree.max_level))
                tree._ray_topology_full = topology
            self._topology = topology
            self._topology_state = topology.kernel_state
            if self._tree_coord == "xyz":
                if not isinstance(self._seed_lookup_state, CartesianLookupKernelState):
                    raise TypeError(
                        "Cartesian ray tracing requires CartesianLookupKernelState; "
                        f"got {type(self._seed_lookup_state).__name__}."
                    )
                self._cell_lookup_state = self._seed_lookup_state
                self._cell_plane_state = None
            elif self._tree_coord == "rpa":
                plane_state = getattr(tree, "_ray_cell_plane_state", None)
                if plane_state is None:
                    plane_state = _build_cell_plane_kernel_state(tree)
                    tree._ray_cell_plane_state = plane_state
                self._seed_cell_plane_state = plane_state
                self._cell_lookup_state = None
                self._cell_plane_state = self._seed_cell_plane_state
            else:
                raise ValueError(f"Unsupported tree_coord '{self._tree_coord}'.")
            self._ray_cell_geometry = None
            return

        geometry = _ray_cell_geometry_for_maxdepth(tree, int(self.maxdepth))
        self._ray_cell_geometry = geometry
        self._topology = None
        self._topology_state = geometry.topology_state
        if self._tree_coord == "xyz":
            if not isinstance(geometry, CartesianRayCellGeometry):
                raise TypeError(
                    "Cartesian ray tracing requires CartesianRayCellGeometry; "
                    f"got {type(geometry).__name__}."
                )
            self._cell_lookup_state = geometry.lookup_state
            self._cell_plane_state = None
        elif self._tree_coord == "rpa":
            if not isinstance(geometry, SphericalRayCellGeometry):
                raise TypeError(
                    "Spherical ray tracing requires SphericalRayCellGeometry; "
                    f"got {type(geometry).__name__}."
                )
            topology = _topology_for_ray_maxdepth(tree, int(self.maxdepth))
            self._seed_lookup_state = _build_sparse_spherical_seed_lookup_state(tree, topology, geometry)
            self._seed_cell_plane_state = _build_sparse_seed_plane_state(
                topology.node_cell_ids,
                int(np.asarray(tree.cell_levels, dtype=np.int64).shape[0]),
                geometry.plane_state,
            )
            self._cell_lookup_state = None
            self._cell_plane_state = geometry.plane_state
        else:
            raise ValueError(f"Unsupported tree_coord '{self._tree_coord}'.")

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

        topo_state = self._topology_state
        if self._tree_coord == "xyz":
            seed_lookup_state = self._seed_lookup_state
            cell_lookup_state = self._cell_lookup_state
            if not isinstance(seed_lookup_state, CartesianLookupKernelState):
                raise TypeError(
                    "Cartesian ray tracing requires CartesianLookupKernelState; "
                    f"got {type(seed_lookup_state).__name__}."
                )
            if not isinstance(cell_lookup_state, CartesianLookupKernelState):
                raise TypeError(
                    "Cartesian ray tracing requires CartesianLookupKernelState for active cells; "
                    f"got {type(cell_lookup_state).__name__}."
                )
            n_seg, cids, enters, exits = _trace_segments_xyz_kernel(
                o,
                d,
                t0,
                t1,
                int(max_steps),
                float(boundary_tol),
                seed_lookup_state,
                cell_lookup_state,
                topo_state,
            )
        elif self._tree_coord == "rpa":
            seed_lookup_state = self._seed_lookup_state
            seed_plane_state = self._seed_cell_plane_state
            plane_state = self._cell_plane_state
            if not isinstance(seed_lookup_state, SphericalLookupKernelState):
                raise TypeError(
                    "Spherical ray tracing requires SphericalLookupKernelState; "
                    f"got {type(seed_lookup_state).__name__}."
                )
            if not isinstance(seed_plane_state, CellPlaneKernelState):
                raise TypeError(
                    "Spherical ray tracing requires CellPlaneKernelState for seed cells; "
                    f"got {type(seed_plane_state).__name__}."
                )
            if not isinstance(plane_state, CellPlaneKernelState):
                raise TypeError(
                    "Spherical ray tracing requires CellPlaneKernelState for active cells; "
                    f"got {type(plane_state).__name__}."
                )
            n_seg, cids, enters, exits = _trace_segments_rpa_kernel(
                o,
                d,
                t0,
                t1,
                int(max_steps),
                float(boundary_tol),
                seed_lookup_state,
                topo_state,
                seed_plane_state,
                plane_state,
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
                seed_lookup_state = self._seed_lookup_state
                cell_lookup_state = self._cell_lookup_state
                if not isinstance(seed_lookup_state, CartesianLookupKernelState):
                    raise TypeError(
                        "Cartesian ray tracing requires CartesianLookupKernelState; "
                        f"got {type(seed_lookup_state).__name__}."
                    )
                if not isinstance(cell_lookup_state, CartesianLookupKernelState):
                    raise TypeError(
                        "Cartesian ray tracing requires CartesianLookupKernelState for active cells; "
                        f"got {type(cell_lookup_state).__name__}."
                    )
                counts[start:stop] = _segment_counts_xyz_kernel(
                    sub,
                    d,
                    t0,
                    t1,
                    int(max_steps),
                    float(boundary_tol),
                    seed_lookup_state,
                    cell_lookup_state,
                    topo_state,
                )
            elif self._tree_coord == "rpa":
                seed_lookup_state = self._seed_lookup_state
                if not isinstance(seed_lookup_state, SphericalLookupKernelState):
                    raise TypeError(
                        "Spherical ray tracing requires SphericalLookupKernelState; "
                        f"got {type(seed_lookup_state).__name__}."
                    )
                if not isinstance(self._seed_cell_plane_state, CellPlaneKernelState):
                    raise TypeError(
                        "Spherical ray tracing requires CellPlaneKernelState for seed cells; "
                        f"got {type(self._seed_cell_plane_state).__name__}."
                    )
                if not isinstance(self._cell_plane_state, CellPlaneKernelState):
                    raise TypeError(
                        "Spherical ray tracing requires CellPlaneKernelState for active cells; "
                        f"got {type(self._cell_plane_state).__name__}."
                    )
                counts[start:stop] = _segment_counts_rpa_kernel(
                    sub,
                    d,
                    t0,
                    t1,
                    int(max_steps),
                    float(boundary_tol),
                    seed_lookup_state,
                    topo_state,
                    self._seed_cell_plane_state,
                    self._cell_plane_state,
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
            seed_lookup_state = self._seed_lookup_state
            cell_lookup_state = self._cell_lookup_state
            if not isinstance(seed_lookup_state, CartesianLookupKernelState):
                raise TypeError(
                    "Cartesian ray tracing requires CartesianLookupKernelState; "
                    f"got {type(seed_lookup_state).__name__}."
                )
            if not isinstance(cell_lookup_state, CartesianLookupKernelState):
                raise TypeError(
                    "Cartesian ray tracing requires CartesianLookupKernelState for active cells; "
                    f"got {type(cell_lookup_state).__name__}."
                )
            counts = _segment_counts_xyz_kernel(
                origins,
                d,
                t0,
                t1,
                int(max_steps),
                float(boundary_tol),
                seed_lookup_state,
                cell_lookup_state,
                self._topology_state,
            )
        elif self._tree_coord == "rpa":
            seed_lookup_state = self._seed_lookup_state
            if not isinstance(seed_lookup_state, SphericalLookupKernelState):
                raise TypeError(
                    "Spherical ray tracing requires SphericalLookupKernelState; "
                    f"got {type(seed_lookup_state).__name__}."
                )
            if not isinstance(self._seed_cell_plane_state, CellPlaneKernelState):
                raise TypeError(
                    "Spherical ray tracing requires CellPlaneKernelState for seed cells; "
                    f"got {type(self._seed_cell_plane_state).__name__}."
                )
            if not isinstance(self._cell_plane_state, CellPlaneKernelState):
                raise TypeError(
                    "Spherical ray tracing requires CellPlaneKernelState for active cells; "
                    f"got {type(self._cell_plane_state).__name__}."
                )
            counts = _segment_counts_rpa_kernel(
                origins,
                d,
                t0,
                t1,
                int(max_steps),
                float(boundary_tol),
                seed_lookup_state,
                self._topology_state,
                self._seed_cell_plane_state,
                self._cell_plane_state,
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
                self._seed_lookup_state,
                self._cell_lookup_state,
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
                self._seed_lookup_state,
                self._topology_state,
                self._seed_cell_plane_state,
                self._cell_plane_state,
                ray_offsets,
                cell_ids,
                t_enter,
                t_exit,
            )

        return cell_ids, t_enter, t_exit, ray_offsets


class OctreeRayInterpolator:
    """Thin convenience wrapper around compiled ray integration kernels."""

    def __init__(self, interpolator: "OctreeInterpolator", *, maxdepth: int | None = None) -> None:
        """Bind one interpolator and its tree."""
        self.interpolator = interpolator
        self.tree = interpolator.tree
        self.maxdepth = _resolve_ray_maxdepth(self.tree, maxdepth)
        self.ray_tracer = OctreeRayTracer(self.tree, maxdepth=maxdepth)
        if int(self.maxdepth) >= int(self.tree.depth):
            self._interp_state_xyz = getattr(interpolator, "_interp_state_xyz", None)
            self._interp_state_rpa = getattr(interpolator, "_interp_state_rpa", None)
        else:
            interp_state = _ray_interp_state_for_maxdepth(interpolator, int(self.maxdepth))
            if isinstance(interp_state, CartesianInterpKernelState):
                self._interp_state_xyz = interp_state
                self._interp_state_rpa = None
            elif isinstance(interp_state, SphericalInterpKernelState):
                self._interp_state_xyz = None
                self._interp_state_rpa = interp_state
            else:
                raise TypeError(f"Unsupported truncated interpolation state {type(interp_state).__name__}.")

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
        origins = _coerce_origins_xyz(origins_xyz)
        directions, shared = _coerce_directions_xyz(direction_xyz, n_rays=int(origins.shape[0]))
        if shared:
            return self.ray_tracer.segment_counts(
                origins,
                directions,
                t_start,
                t_end,
                chunk_size=chunk_size,
                max_steps=max_steps,
                boundary_tol=boundary_tol,
            )

        t0, t1 = _coerce_ray_interval(t_start, t_end)
        max_steps = _coerce_positive_int("max_steps", max_steps)
        counts = np.zeros(int(origins.shape[0]), dtype=np.int64)
        for i in range(int(origins.shape[0])):
            cell_ids, _t_enter, _t_exit = self.ray_tracer.trace_prepared(
                origins[i],
                directions[i],
                t0,
                t1,
                max_steps=max_steps,
                boundary_tol=boundary_tol,
            )
            counts[i] = int(cell_ids.size)
        return counts

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
        directions, shared = _coerce_directions_xyz(direction_xyz, n_rays=int(origins.shape[0]))
        t0, t1 = _coerce_ray_interval(t_start, t_end)
        chunk = _coerce_positive_chunk_size(chunk_size)
        max_steps = _coerce_positive_int("max_steps", max_steps)

        mids_chunks: list[np.ndarray] = []
        weights_chunks: list[np.ndarray] = []
        offsets = np.zeros(int(origins.shape[0]) + 1, dtype=np.int64)

        if not shared:
            written = 0
            for i in range(int(origins.shape[0])):
                _cell_ids, t_enter, t_exit = self.ray_tracer.trace_prepared(
                    origins[i],
                    directions[i],
                    t0,
                    t1,
                    max_steps=max_steps,
                    boundary_tol=boundary_tol,
                )
                n_seg = int(t_enter.size)
                if n_seg > 0:
                    mids_sub = origins[i][None, :] + (0.5 * (t_enter + t_exit))[:, None] * directions[i][None, :]
                    weights_sub = np.maximum(0.0, t_exit - t_enter)
                    mids_chunks.append(np.asarray(mids_sub, dtype=float))
                    weights_chunks.append(np.asarray(weights_sub, dtype=float))
                written += n_seg
                offsets[i + 1] = written
            if written == 0:
                return (
                    np.empty((0, 3), dtype=float),
                    np.empty((0,), dtype=float),
                    offsets,
                )
            return np.vstack(mids_chunks), np.concatenate(weights_chunks), offsets

        written = 0
        for start in range(0, int(origins.shape[0]), chunk):
            stop = min(int(origins.shape[0]), start + chunk)
            sub = origins[start:stop]
            cids, t_enter, t_exit, sub_offsets = self.ray_tracer.trace_many(
                sub,
                directions,
                t0,
                t1,
                max_steps=max_steps,
                boundary_tol=boundary_tol,
            )
            mids_sub, weights_sub = _midpoints_from_segments_kernel(sub, directions, t_enter, t_exit, sub_offsets)
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
                continue
            valid = cids[s:e] >= 0
            if not np.any(valid):
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
        directions, shared = _coerce_directions_xyz(direction_xyz, n_rays=int(origins.shape[0]))
        t0, t1 = _coerce_ray_interval(t_start, t_end)
        chunk = _coerce_positive_chunk_size(chunk_size)
        max_steps = _coerce_positive_int("max_steps", max_steps)

        n_rays = int(origins.shape[0])
        ncomp = int(self.interpolator.n_value_components)
        out = np.full((n_rays, ncomp), np.nan, dtype=float)

        if not shared:
            for i in range(n_rays):
                one = np.asarray(
                    self.integrate_field_along_rays(
                        origins[i : i + 1],
                        directions[i],
                        t0,
                        t1,
                        chunk_size=1,
                        scale=scale,
                        max_steps=max_steps,
                        boundary_tol=boundary_tol,
                    ),
                    dtype=float,
                )
                out[i] = one.reshape(1, -1)[0]
            if ncomp == 1:
                return out[:, 0]
            return out

        tree_coord = str(self.tree.tree_coord)
        topo_state = self.ray_tracer._topology_state
        for start in range(0, n_rays, chunk):
            stop = min(n_rays, start + chunk)
            sub = origins[start:stop]
            if tree_coord == "xyz":
                seed_lookup_state = self.ray_tracer._seed_lookup_state
                cell_lookup_state = self.ray_tracer._cell_lookup_state
                if not isinstance(seed_lookup_state, CartesianLookupKernelState):
                    raise TypeError(
                        "Cartesian ray tracing requires CartesianLookupKernelState; "
                        f"got {type(seed_lookup_state).__name__}."
                    )
                if not isinstance(cell_lookup_state, CartesianLookupKernelState):
                    raise TypeError(
                        "Cartesian ray tracing requires CartesianLookupKernelState for active cells; "
                        f"got {type(cell_lookup_state).__name__}."
                    )
                interp_state = self._interp_state_xyz
                if not isinstance(interp_state, CartesianInterpKernelState):
                    raise TypeError(
                        "Cartesian ray interpolation requires CartesianInterpKernelState; "
                        f"got {type(interp_state).__name__}."
                    )
                out[start:stop] = _integrate_rays_xyz_kernel(
                    sub,
                    directions,
                    t0,
                    t1,
                    int(max_steps),
                    float(boundary_tol),
                    float(scale),
                    seed_lookup_state,
                    cell_lookup_state,
                    topo_state,
                    interp_state,
                )
            elif tree_coord == "rpa":
                seed_lookup_state = self.ray_tracer._seed_lookup_state
                seed_plane_state = self.ray_tracer._seed_cell_plane_state
                plane_state = self.ray_tracer._cell_plane_state
                if not isinstance(seed_lookup_state, SphericalLookupKernelState):
                    raise TypeError(
                        "Spherical ray tracing requires SphericalLookupKernelState; "
                        f"got {type(seed_lookup_state).__name__}."
                    )
                if not isinstance(seed_plane_state, CellPlaneKernelState):
                    raise TypeError(
                        "Spherical ray tracing requires CellPlaneKernelState for seed cells; "
                        f"got {type(seed_plane_state).__name__}."
                    )
                if not isinstance(plane_state, CellPlaneKernelState):
                    raise TypeError(
                        "Spherical ray tracing requires CellPlaneKernelState for active cells; "
                        f"got {type(plane_state).__name__}."
                    )
                interp_state = self._interp_state_rpa
                if not isinstance(interp_state, SphericalInterpKernelState):
                    raise TypeError(
                        "Spherical ray interpolation requires SphericalInterpKernelState; "
                        f"got {type(interp_state).__name__}."
                    )
                out[start:stop] = _integrate_rays_rpa_kernel(
                    sub,
                    directions,
                    t0,
                    t1,
                    int(max_steps),
                    float(boundary_tol),
                    float(scale),
                    seed_lookup_state,
                    topo_state,
                    seed_plane_state,
                    plane_state,
                    interp_state,
                )
            else:
                raise ValueError(f"Unsupported tree_coord '{tree_coord}'.")

        if ncomp == 1:
            return out[:, 0]
        return out
