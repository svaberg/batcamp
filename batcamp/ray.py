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

from .interpolator import CartesianInterpKernelState
from .interpolator import SphericalInterpKernelState
from .interpolator import _trilinear_from_cell
from .interpolator import _trilinear_from_cell_rpa
from .octree import _lookup_cell_id_kernel
from .octree import LookupKernelState
from .octree import Octree
from .spherical import _xyz_to_rpa_components
from .face_neighbors import FaceNeighborKernelState

if TYPE_CHECKING:
    from .interpolator import OctreeInterpolator

CartesianLookupKernelState = LookupKernelState
SphericalLookupKernelState = LookupKernelState


_TRACE_CONTAIN_TOL = 1e-8
_DEFAULT_TRACE_BOUNDARY_TOL = 1e-9
_DEFAULT_TRACE_MAX_STEPS = 100000
_MAX_FACE_NEIGHBOR_CANDIDATES = 128
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

    face_neighbor_state: FaceNeighborKernelState
    lookup_state: LookupKernelState
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

    face_neighbor_state: FaceNeighborKernelState
    plane_state: CellPlaneKernelState
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


def _resolve_ray_max_level(tree: Octree, max_level: int | None) -> int:
    """Return requested ray level cutoff, defaulting to the full tree level."""
    if max_level is None:
        return int(tree.max_level)
    level = int(max_level)
    if level < 0 or level > int(tree.max_level):
        raise ValueError(f"max_level={level} is outside [0, {int(tree.max_level)}] for this tree.")
    return level


def _face_neighbor_state_for_node_cells(face_neighbors) -> FaceNeighborKernelState:
    """Return one face-neighbor state whose active cell ids are the frontier nodes."""
    node_ids = np.arange(int(face_neighbors.node_count), dtype=np.int64)
    return FaceNeighborKernelState(
        face_offsets=np.asarray(face_neighbors.face_offsets, dtype=np.int64),
        face_neighbors=np.asarray(face_neighbors.face_neighbors, dtype=np.int64),
        node_cell_ids=node_ids,
        cell_to_node_id=np.asarray(face_neighbors.cell_to_node_id, dtype=np.int64),
    )


def _node_point_candidates(tree: Octree, face_neighbors) -> list[np.ndarray]:
    """Return unique descendant-corner point ids for every frontier node."""
    lookup_geometry = tree.lookup_geometry()
    if tree.cell_levels is None:
        raise ValueError("Octree has no cell levels; cannot build truncated ray cells.")
    corners = lookup_geometry.corners
    cell_levels = np.asarray(tree.cell_levels, dtype=np.int64)
    valid_cells = np.flatnonzero(cell_levels >= 0).astype(np.int64)
    node_ids = np.asarray(face_neighbors.cell_to_node_id[valid_cells], dtype=np.int64)
    order = np.argsort(node_ids, kind="stable")
    node_ids = node_ids[order]
    valid_cells = valid_cells[order]
    groups: list[np.ndarray] = [np.empty((0,), dtype=np.int64) for _ in range(int(face_neighbors.node_count))]
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
    """Build one Cartesian plane model from face_neighbors-aligned cell faces."""
    lookup_geometry = tree.lookup_geometry()
    corners = lookup_geometry.corners
    points = lookup_geometry.points
    centers = np.mean(points[corners], axis=1)
    lookup_state = lookup_geometry.coord_state
    n_cells = int(corners.shape[0])

    face_normals = np.zeros((n_cells, 6, 3), dtype=np.float64)
    face_offsets = np.zeros((n_cells, 6), dtype=np.float64)
    face_valid = np.zeros((n_cells, 6), dtype=np.bool_)
    tiny = 1.0e-24
    tree_coord = str(tree.tree_coord)

    if tree_coord == "rpa":
        ordered_corners = _ordered_spherical_corners_from_targets_kernel(
            points,
            corners,
            np.asarray(lookup_state.cell_axis0_start, dtype=np.float64),
            np.asarray(lookup_state.cell_axis0_start + lookup_state.cell_axis0_width, dtype=np.float64),
            np.asarray(lookup_state.cell_axis1_start, dtype=np.float64),
            np.asarray(lookup_state.cell_axis1_start + lookup_state.cell_axis1_width, dtype=np.float64),
            np.asarray(lookup_state.cell_axis2_start, dtype=np.float64),
            np.asarray(lookup_state.cell_axis2_width, dtype=np.float64),
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
            cx = cell_pts[:, 0]
            cy = cell_pts[:, 1]
            cz = cell_pts[:, 2]
            x0 = float(lookup_state.cell_axis0_start[cid])
            x1 = float(lookup_state.cell_axis0_start[cid] + lookup_state.cell_axis0_width[cid])
            y0 = float(lookup_state.cell_axis1_start[cid])
            y1 = float(lookup_state.cell_axis1_start[cid] + lookup_state.cell_axis1_width[cid])
            z0 = float(lookup_state.cell_axis2_start[cid])
            z1 = float(lookup_state.cell_axis2_start[cid] + lookup_state.cell_axis2_width[cid])
            tx = max(1e-10, 1e-8 * (x1 - x0))
            ty = max(1e-10, 1e-8 * (y1 - y0))
            tz = max(1e-10, 1e-8 * (z1 - z0))
            face_corner_sets = (
                _closest_four(cx, x0, tx),
                _closest_four(cx, x1, tx),
                _closest_four(cy, y0, ty),
                _closest_four(cy, y1, ty),
                _closest_four(cz, z0, tz),
                _closest_four(cz, z1, tz),
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


def _build_cartesian_ray_cell_geometry(tree: Octree, face_neighbors) -> CartesianRayCellGeometry:
    """Build truncated Cartesian ray-cell geometry from one frontier face_neighbors."""
    lookup_geometry = tree.lookup_geometry()
    points = lookup_geometry.points
    point_groups = _node_point_candidates(tree, face_neighbors)
    n_cells = int(face_neighbors.node_count)

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

    seed_lookup = lookup_geometry.coord_state
    lookup_state = LookupKernelState(
        cell_axis0_start=cell_x0,
        cell_axis0_width=cell_xden,
        cell_axis1_start=cell_y0,
        cell_axis1_width=cell_yden,
        cell_axis2_start=cell_z0,
        cell_axis2_width=cell_zden,
        cell_valid=np.ones(n_cells, dtype=np.bool_),
        domain_axis0_start=float(seed_lookup.domain_axis0_start),
        domain_axis0_width=float(seed_lookup.domain_axis0_width),
        domain_axis1_start=float(seed_lookup.domain_axis1_start),
        domain_axis1_width=float(seed_lookup.domain_axis1_width),
        domain_axis2_start=float(seed_lookup.domain_axis2_start),
        domain_axis2_width=float(seed_lookup.domain_axis2_width),
        axis2_period=float(seed_lookup.axis2_period),
        axis2_periodic=bool(seed_lookup.axis2_periodic),
        node_value=np.arange(n_cells, dtype=np.int64),
        node_child=np.full((n_cells, 8), -1, dtype=np.int64),
        root_node_ids=np.arange(n_cells, dtype=np.int64),
        node_parent=np.full(n_cells, -1, dtype=np.int64),
        cell_node_id=np.arange(n_cells, dtype=np.int64),
        node_axis0_start=cell_x0,
        node_axis0_width=cell_xden,
        node_axis1_start=cell_y0,
        node_axis1_width=cell_yden,
        node_axis2_start=cell_z0,
        node_axis2_width=cell_zden,
    )
    return CartesianRayCellGeometry(
        face_neighbor_state=_face_neighbor_state_for_node_cells(face_neighbors),
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


def _build_spherical_ray_cell_geometry(tree: Octree, face_neighbors) -> SphericalRayCellGeometry:
    """Build truncated spherical ray-cell geometry from one frontier face_neighbors."""
    lookup_geometry = tree.lookup_geometry()
    points = lookup_geometry.points
    point_groups = _node_point_candidates(tree, face_neighbors)
    point_r = np.linalg.norm(points, axis=1)
    point_phi = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * math.pi)

    n_cells = int(face_neighbors.node_count)
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
        depth = int(face_neighbors.levels[node_id])
        scale = 1 << depth
        _nr = int(tree.root_shape[0]) * scale
        ntheta = int(tree.root_shape[1]) * scale
        nphi = int(tree.root_shape[2]) * scale
        dtheta = math.pi / float(ntheta)
        dphi = 2.0 * math.pi / float(nphi)
        theta0 = float(int(face_neighbors.i1[node_id]) * dtheta)
        theta1 = theta0 + dtheta
        phi_start = float(int(face_neighbors.i2[node_id]) * dphi)
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
        face_neighbor_state=_face_neighbor_state_for_node_cells(face_neighbors),
        plane_state=plane_state,
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
    face_neighbors,
    geometry: SphericalRayCellGeometry,
) -> LookupKernelState:
    """Build one sparse coarse spherical lookup state keyed by representative leaf ids."""
    lookup_geometry = tree.lookup_geometry()
    n_leaf_cells = int(np.asarray(tree.cell_levels, dtype=np.int64).shape[0])
    rep_cell_ids = np.asarray(face_neighbors.node_cell_ids, dtype=np.int64)
    n_nodes = int(rep_cell_ids.shape[0])
    cell_r_min = np.zeros(n_leaf_cells, dtype=np.float64)
    cell_r_max = np.zeros(n_leaf_cells, dtype=np.float64)
    cell_theta_min = np.zeros(n_leaf_cells, dtype=np.float64)
    cell_theta_max = np.zeros(n_leaf_cells, dtype=np.float64)
    cell_phi_start = np.zeros(n_leaf_cells, dtype=np.float64)
    cell_phi_width = np.zeros(n_leaf_cells, dtype=np.float64)
    cell_valid = np.zeros(n_leaf_cells, dtype=np.bool_)
    for node_id, rep_cid in enumerate(rep_cell_ids):
        cid = int(rep_cid)
        cell_r_min[cid] = float(geometry.cell_r0[node_id])
        cell_r_max[cid] = float(geometry.cell_r0[node_id] + geometry.cell_rden[node_id])
        cell_theta_min[cid] = float(geometry.cell_t0[node_id])
        cell_theta_max[cid] = float(geometry.cell_t0[node_id] + geometry.cell_tden[node_id])
        cell_phi_start[cid] = float(geometry.cell_p_start[node_id])
        cell_phi_width[cid] = float(geometry.cell_p_width[node_id])
        cell_valid[cid] = True

    cell_node_id = np.full(n_leaf_cells, -1, dtype=np.int64)
    for node_id, rep_cid in enumerate(rep_cell_ids):
        cell_node_id[int(rep_cid)] = int(node_id)

    return LookupKernelState(
        cell_axis0_start=cell_r_min,
        cell_axis0_width=cell_r_max - cell_r_min,
        cell_axis1_start=cell_theta_min,
        cell_axis1_width=cell_theta_max - cell_theta_min,
        cell_axis2_start=cell_phi_start,
        cell_axis2_width=cell_phi_width,
        cell_valid=cell_valid,
        domain_axis0_start=float(np.min(geometry.cell_r0)),
        domain_axis0_width=float(np.max(geometry.cell_r0 + geometry.cell_rden) - np.min(geometry.cell_r0)),
        domain_axis1_start=0.0,
        domain_axis1_width=float(math.pi),
        domain_axis2_start=0.0,
        domain_axis2_width=float(2.0 * math.pi),
        axis2_period=float(2.0 * math.pi),
        axis2_periodic=True,
        node_value=np.asarray(rep_cell_ids, dtype=np.int64),
        node_child=np.full((n_nodes, 8), -1, dtype=np.int64),
        root_node_ids=np.arange(n_nodes, dtype=np.int64),
        node_parent=np.full(n_nodes, -1, dtype=np.int64),
        cell_node_id=cell_node_id,
        node_axis0_start=np.asarray(geometry.cell_r0, dtype=np.float64),
        node_axis0_width=np.asarray(geometry.cell_rden, dtype=np.float64),
        node_axis1_start=np.asarray(geometry.cell_t0, dtype=np.float64),
        node_axis1_width=np.asarray(geometry.cell_tden, dtype=np.float64),
        node_axis2_start=np.asarray(geometry.cell_p_start, dtype=np.float64),
        node_axis2_width=np.asarray(geometry.cell_p_width, dtype=np.float64),
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


def _ray_cell_geometry_for_max_level(tree: Octree, max_level: int) -> CartesianRayCellGeometry | SphericalRayCellGeometry:
    """Build truncated ray-cell geometry for one level cutoff."""
    face_neighbors = tree.face_neighbors(max_level=int(max_level))
    if str(tree.tree_coord) == "xyz":
        return _build_cartesian_ray_cell_geometry(tree, face_neighbors)
    if str(tree.tree_coord) == "rpa":
        return _build_spherical_ray_cell_geometry(tree, face_neighbors)
    raise ValueError(f"Unsupported tree_coord '{tree.tree_coord}'.")


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
    x0 = lookup_state.cell_axis0_start[cid]
    x1 = lookup_state.cell_axis0_start[cid] + lookup_state.cell_axis0_width[cid]
    y0 = lookup_state.cell_axis1_start[cid]
    y1 = lookup_state.cell_axis1_start[cid] + lookup_state.cell_axis1_width[cid]
    z0 = lookup_state.cell_axis2_start[cid]
    z1 = lookup_state.cell_axis2_start[cid] + lookup_state.cell_axis2_width[cid]
    if x < (x0 - tol) or x > (x1 + tol):
        return False
    if y < (y0 - tol) or y > (y1 + tol):
        return False
    if z < (z0 - tol) or z > (z1 + tol):
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
def _expand_face_neighbor_nodes_for_face(
    src_nodes: np.ndarray,
    n_src: int,
    face: int,
    face_state: FaceNeighborKernelState,
    dst_nodes: np.ndarray,
) -> int:
    """Expand one candidate node set across one face."""
    n_dst = 0
    for i in range(n_src):
        node_id = int(src_nodes[i])
        slot = node_id * 6 + int(face)
        start = int(face_state.face_offsets[slot])
        end = int(face_state.face_offsets[slot + 1])
        for pos in range(start, end):
            n_dst = _append_unique_node(dst_nodes, n_dst, int(face_state.face_neighbors[pos]))
    return n_dst


@njit(cache=True)
def _candidate_face_neighbor_nodes_after_exit(
    current_node_id: int,
    face_mask: int,
    face_state: FaceNeighborKernelState,
    work0: np.ndarray,
    work1: np.ndarray,
) -> tuple[int, int]:
    """Return next nodes reachable across any non-empty subset of exited faces."""
    work0[0] = int(current_node_id)
    n_reachable = 1
    for face in range(6):
        if (int(face_mask) & (1 << face)) == 0:
            continue
        n_new = _expand_face_neighbor_nodes_for_face(work0, n_reachable, face, face_state, work1)
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
def _select_next_xyz_cell_from_face_neighbors(
    current_node_id: int,
    face_mask: int,
    x_next: float,
    y_next: float,
    z_next: float,
    face_state: FaceNeighborKernelState,
    lookup_state: CartesianLookupKernelState,
    work0: np.ndarray,
    work1: np.ndarray,
) -> tuple[int, int]:
    """Choose next Cartesian `(node_id, cell_id)` from face_neighbors candidates."""
    n_candidates, active = _candidate_face_neighbor_nodes_after_exit(current_node_id, face_mask, face_state, work0, work1)
    if n_candidates <= 0:
        return -1, -1
    nodes = work0 if active == 0 else work1
    for i in range(n_candidates):
        node_id = int(nodes[i])
        cid = int(face_state.node_cell_ids[node_id])
        if _contains_xyz_from_state(cid, x_next, y_next, z_next, lookup_state):
            return node_id, cid
    return -1, -1


@njit(cache=True)
def _select_next_cell_from_face_neighbors(
    current_node_id: int,
    face_mask: int,
    x_next: float,
    y_next: float,
    z_next: float,
    face_state: FaceNeighborKernelState,
    plane_state: CellPlaneKernelState,
    work0: np.ndarray,
    work1: np.ndarray,
) -> tuple[int, int]:
    """Choose next `(node_id, cell_id)` from face_neighbors candidates."""
    n_candidates, active = _candidate_face_neighbor_nodes_after_exit(current_node_id, face_mask, face_state, work0, work1)
    if n_candidates <= 0:
        return -1, -1
    nodes = work0 if active == 0 else work1
    for i in range(n_candidates):
        node_id = int(nodes[i])
        cid = int(face_state.node_cell_ids[node_id])
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
    cid = _lookup_cell_id_kernel(x, y, z, lookup_state, -1)
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
        cid_hi = _lookup_cell_id_kernel(xh, yh, zh, lookup_state, -1)
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
        cid_mid = _lookup_cell_id_kernel(xm, ym, zm, lookup_state, cid_in)
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
    cid = _lookup_cell_id_kernel(x, y, z, lookup_state, cid_in)
    if cid < 0:
        t = hi_in
        x = origin_xyz[0] + t * direction_xyz_unit[0]
        y = origin_xyz[1] + t * direction_xyz_unit[1]
        z = origin_xyz[2] + t * direction_xyz_unit[2]
        cid = _lookup_cell_id_kernel(x, y, z, lookup_state, cid_in)
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
    cid = _lookup_cell_id_kernel(r, polar, azimuth, lookup_state, -1)
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
        cid_hi = _lookup_cell_id_kernel(r_hi, polar_hi, azimuth_hi, lookup_state, -1)
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
        cid_mid = _lookup_cell_id_kernel(r_mid, polar_mid, azimuth_mid, lookup_state, cid_in)
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
    cid = _lookup_cell_id_kernel(r, polar, azimuth, lookup_state, cid_in)
    if cid < 0 or not _contains_cell_from_xyz(cid, x, y, z, plane_state):
        t = hi_in
        x = origin_xyz[0] + t * direction_xyz_unit[0]
        y = origin_xyz[1] + t * direction_xyz_unit[1]
        z = origin_xyz[2] + t * direction_xyz_unit[2]
        r, polar, azimuth = _xyz_to_rpa_components(x, y, z)
        cid = _lookup_cell_id_kernel(r, polar, azimuth, lookup_state, cid_in)
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
        float(lookup_state.domain_axis0_start),
        float(lookup_state.domain_axis0_start + lookup_state.domain_axis0_width),
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
    cid0 = _lookup_cell_id_kernel(r, polar, azimuth, lookup_state, -1)

    x = origin_xyz[0] + tm * direction_xyz_unit[0]
    y = origin_xyz[1] + tm * direction_xyz_unit[1]
    z = origin_xyz[2] + tm * direction_xyz_unit[2]
    r, polar, azimuth = _xyz_to_rpa_components(x, y, z)
    cidm = _lookup_cell_id_kernel(r, polar, azimuth, lookup_state, -1)

    x = origin_xyz[0] + tq1 * direction_xyz_unit[0]
    y = origin_xyz[1] + tq1 * direction_xyz_unit[1]
    z = origin_xyz[2] + tq1 * direction_xyz_unit[2]
    r, polar, azimuth = _xyz_to_rpa_components(x, y, z)
    cid1 = _lookup_cell_id_kernel(r, polar, azimuth, lookup_state, -1)

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
def _trace_segments_xyz_neighbors_kernel(
    origin_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    boundary_tol: float,
    seed_lookup_state: CartesianLookupKernelState,
    cell_lookup_state: CartesianLookupKernelState,
    face_state: FaceNeighborKernelState,
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
        np.array(
            [
                seed_lookup_state.domain_axis0_start,
                seed_lookup_state.domain_axis1_start,
                seed_lookup_state.domain_axis2_start,
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                seed_lookup_state.domain_axis0_start + seed_lookup_state.domain_axis0_width,
                seed_lookup_state.domain_axis1_start + seed_lookup_state.domain_axis1_width,
                seed_lookup_state.domain_axis2_start + seed_lookup_state.domain_axis2_width,
            ],
            dtype=np.float64,
        ),
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

    current_node = int(face_state.cell_to_node_id[cid])
    if current_node < 0:
        return 0, cell_ids, enters, exits
    cid = int(face_state.node_cell_ids[current_node])
    if cid < 0:
        return 0, cell_ids, enters, exits

    d0 = direction_xyz_unit[0]
    d1 = direction_xyz_unit[1]
    d2 = direction_xyz_unit[2]
    dir_eps = 1e-15
    candidate_nodes0 = np.empty(_MAX_FACE_NEIGHBOR_CANDIDATES, dtype=np.int64)
    candidate_nodes1 = np.empty(_MAX_FACE_NEIGHBOR_CANDIDATES, dtype=np.int64)
    n_seg = 0
    for _ in range(max_steps):
        if t >= (t_end - abs_eps):
            break

        if not _contains_xyz_from_state(cid, x, y, z, cell_lookup_state):
            break

        tx = _forward_face_exit_dt(
            x,
            d0,
            cell_lookup_state.cell_axis0_start[cid],
            cell_lookup_state.cell_axis0_start[cid] + cell_lookup_state.cell_axis0_width[cid],
            abs_eps,
            dir_eps,
        )
        ty = _forward_face_exit_dt(
            y,
            d1,
            cell_lookup_state.cell_axis1_start[cid],
            cell_lookup_state.cell_axis1_start[cid] + cell_lookup_state.cell_axis1_width[cid],
            abs_eps,
            dir_eps,
        )
        tz = _forward_face_exit_dt(
            z,
            d2,
            cell_lookup_state.cell_axis2_start[cid],
            cell_lookup_state.cell_axis2_start[cid] + cell_lookup_state.cell_axis2_width[cid],
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
        next_node, cid_next = _select_next_xyz_cell_from_face_neighbors(
            current_node,
            face_mask,
            x_next,
            y_next,
            z_next,
            face_state,
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
            next_node, cid_next = _select_next_xyz_cell_from_face_neighbors(
                current_node,
                face_mask,
                x_next,
                y_next,
                z_next,
                face_state,
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
def _trace_segments_rpa_neighbors_kernel(
    origin_xyz: np.ndarray,
    direction_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    boundary_tol: float,
    seed_lookup_state: SphericalLookupKernelState,
    face_state: FaceNeighborKernelState,
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

    current_node = int(face_state.cell_to_node_id[cid])
    if current_node < 0:
        return 0, cell_ids, enters, exits
    cid = int(face_state.node_cell_ids[current_node])
    if cid < 0:
        return 0, cell_ids, enters, exits

    candidate_nodes0 = np.empty(_MAX_FACE_NEIGHBOR_CANDIDATES, dtype=np.int64)
    candidate_nodes1 = np.empty(_MAX_FACE_NEIGHBOR_CANDIDATES, dtype=np.int64)
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
        next_node, cid_next = _select_next_cell_from_face_neighbors(
            current_node,
            face_mask,
            x_next,
            y_next,
            z_next,
            face_state,
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
            next_node, cid_next = _select_next_cell_from_face_neighbors(
                current_node,
                face_mask,
                x_next,
                y_next,
                z_next,
                face_state,
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
def _segment_counts_xyz_neighbors_kernel(
    origins_xyz: np.ndarray,
    directions_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    boundary_tol: float,
    seed_lookup_state: CartesianLookupKernelState,
    cell_lookup_state: CartesianLookupKernelState,
    face_state: FaceNeighborKernelState,
) -> np.ndarray:
    """Count traced segments for each ray on Cartesian trees."""
    n_rays = int(origins_xyz.shape[0])
    counts = np.zeros(n_rays, dtype=np.int64)
    for i in prange(n_rays):
        n_seg, _cids, _enters, _exits = _trace_segments_xyz_neighbors_kernel(
            origins_xyz[i],
            directions_xyz_unit[i],
            t_start,
            t_end,
            max_steps,
            boundary_tol,
            seed_lookup_state,
            cell_lookup_state,
            face_state,
        )
        counts[i] = int(n_seg)
    return counts


@njit(cache=True, parallel=True)
def _segment_counts_rpa_neighbors_kernel(
    origins_xyz: np.ndarray,
    directions_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    boundary_tol: float,
    seed_lookup_state: SphericalLookupKernelState,
    face_state: FaceNeighborKernelState,
    seed_plane_state: CellPlaneKernelState,
    plane_state: CellPlaneKernelState,
) -> np.ndarray:
    """Count traced segments for each ray on spherical trees."""
    n_rays = int(origins_xyz.shape[0])
    counts = np.zeros(n_rays, dtype=np.int64)
    for i in prange(n_rays):
        n_seg, _cids, _enters, _exits = _trace_segments_rpa_neighbors_kernel(
            origins_xyz[i],
            directions_xyz_unit[i],
            t_start,
            t_end,
            max_steps,
            boundary_tol,
            seed_lookup_state,
            face_state,
            seed_plane_state,
            plane_state,
        )
        counts[i] = int(n_seg)
    return counts


@njit(cache=True)
def _fill_traces_xyz_neighbors_kernel(
    origins_xyz: np.ndarray,
    directions_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    boundary_tol: float,
    seed_lookup_state: CartesianLookupKernelState,
    cell_lookup_state: CartesianLookupKernelState,
    face_state: FaceNeighborKernelState,
    ray_offsets: np.ndarray,
    out_cell_ids: np.ndarray,
    out_t_enter: np.ndarray,
    out_t_exit: np.ndarray,
) -> None:
    """Fill flattened trace arrays from per-ray offsets on Cartesian trees."""
    n_rays = int(origins_xyz.shape[0])
    for i in range(n_rays):
        n_seg, cids, enters, exits = _trace_segments_xyz_neighbors_kernel(
            origins_xyz[i],
            directions_xyz_unit[i],
            t_start,
            t_end,
            max_steps,
            boundary_tol,
            seed_lookup_state,
            cell_lookup_state,
            face_state,
        )
        base = int(ray_offsets[i])
        for j in range(int(n_seg)):
            out_cell_ids[base + j] = int(cids[j])
            out_t_enter[base + j] = float(enters[j])
            out_t_exit[base + j] = float(exits[j])


@njit(cache=True)
def _fill_traces_rpa_neighbors_kernel(
    origins_xyz: np.ndarray,
    directions_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    boundary_tol: float,
    seed_lookup_state: SphericalLookupKernelState,
    face_state: FaceNeighborKernelState,
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
        n_seg, cids, enters, exits = _trace_segments_rpa_neighbors_kernel(
            origins_xyz[i],
            directions_xyz_unit[i],
            t_start,
            t_end,
            max_steps,
            boundary_tol,
            seed_lookup_state,
            face_state,
            seed_plane_state,
            plane_state,
        )
        base = int(ray_offsets[i])
        for j in range(int(n_seg)):
            out_cell_ids[base + j] = int(cids[j])
            out_t_enter[base + j] = float(enters[j])
            out_t_exit[base + j] = float(exits[j])


@njit(cache=True, parallel=True)
def _integrate_rays_xyz_neighbors_kernel(
    origins_xyz: np.ndarray,
    directions_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    boundary_tol: float,
    scale: float,
    seed_lookup_state: CartesianLookupKernelState,
    cell_lookup_state: CartesianLookupKernelState,
    face_state: FaceNeighborKernelState,
    interp_state: CartesianInterpKernelState,
) -> np.ndarray:
    """Integrate fields along rays on Cartesian trees with 2-point Gauss segments."""
    n_rays = int(origins_xyz.shape[0])
    ncomp = int(interp_state.point_values_2d.shape[1])
    out = np.full((n_rays, ncomp), np.nan, dtype=np.float64)
    for i in prange(n_rays):
        d0 = float(directions_xyz_unit[i, 0])
        d1 = float(directions_xyz_unit[i, 1])
        d2 = float(directions_xyz_unit[i, 2])
        n_seg, cids, enters, exits = _trace_segments_xyz_neighbors_kernel(
            origins_xyz[i],
            directions_xyz_unit[i],
            t_start,
            t_end,
            max_steps,
            boundary_tol,
            seed_lookup_state,
            cell_lookup_state,
            face_state,
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
            _trilinear_from_cell(row, int(cids[j]), x, y, z, interp_state)
            acc += row * th
            x = ox + tg1 * d0
            y = oy + tg1 * d1
            z = oz + tg1 * d2
            _trilinear_from_cell(row, int(cids[j]), x, y, z, interp_state)
            acc += row * th
            used = True
        if used:
            out[i, :] = scale * acc
    return out


@njit(cache=True, parallel=True)
def _integrate_rays_rpa_neighbors_kernel(
    origins_xyz: np.ndarray,
    directions_xyz_unit: np.ndarray,
    t_start: float,
    t_end: float,
    max_steps: int,
    boundary_tol: float,
    scale: float,
    seed_lookup_state: SphericalLookupKernelState,
    face_state: FaceNeighborKernelState,
    seed_plane_state: CellPlaneKernelState,
    plane_state: CellPlaneKernelState,
    interp_state: SphericalInterpKernelState,
) -> np.ndarray:
    """Integrate fields along rays on spherical trees with 2-point Gauss segments."""
    n_rays = int(origins_xyz.shape[0])
    ncomp = int(interp_state.point_values_2d.shape[1])
    out = np.full((n_rays, ncomp), np.nan, dtype=np.float64)
    for i in prange(n_rays):
        d0 = float(directions_xyz_unit[i, 0])
        d1 = float(directions_xyz_unit[i, 1])
        d2 = float(directions_xyz_unit[i, 2])
        n_seg, cids, enters, exits = _trace_segments_rpa_neighbors_kernel(
            origins_xyz[i],
            directions_xyz_unit[i],
            t_start,
            t_end,
            max_steps,
            boundary_tol,
            seed_lookup_state,
            face_state,
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
            _trilinear_from_cell_rpa(row, int(cids[j]), r, polar, azimuth, interp_state)
            acc += row * th
            x = ox + tg1 * d0
            y = oy + tg1 * d1
            z = oz + tg1 * d2
            r, polar, azimuth = _xyz_to_rpa_components(x, y, z)
            _trilinear_from_cell_rpa(row, int(cids[j]), r, polar, azimuth, interp_state)
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


def _coerce_directions_xyz(direction_xyz: np.ndarray, *, n_rays: int) -> np.ndarray:
    """Validate directions and return shape `(n_rays, 3)`."""
    direction = np.asarray(direction_xyz, dtype=float)
    if direction.ndim == 1:
        return np.tile(_normalize_direction(direction).reshape(1, 3), (int(n_rays), 1))
    if direction.ndim != 2 or direction.shape[1] != 3:
        raise ValueError("direction_xyz must have shape (3,) or (n_rays, 3).")
    if int(direction.shape[0]) == 1:
        return np.tile(_normalize_direction(direction[0]).reshape(1, 3), (int(n_rays), 1))
    if int(direction.shape[0]) != int(n_rays):
        raise ValueError("direction_xyz with shape (n_rays, 3) must match origins_xyz.")
    norms = np.linalg.norm(direction, axis=1)
    if not np.all(np.isfinite(direction)) or np.any(norms == 0.0):
        raise ValueError("direction_xyz must be finite and non-zero.")
    return direction / norms[:, None]


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

    def __init__(self, tree: Octree, *, max_level: int | None = None) -> None:
        """Bind one built octree."""
        self.tree = tree
        lookup_geometry = tree.lookup_geometry()
        self.max_level = _resolve_ray_max_level(tree, max_level)
        self._tree_coord = str(tree.tree_coord)
        dmin, dmax = tree.domain_bounds(coord="xyz")
        self._domain_xyz_min = np.asarray(dmin, dtype=float).reshape(3)
        self._domain_xyz_max = np.asarray(dmax, dtype=float).reshape(3)
        _r_lo, r_hi = tree.domain_bounds(coord="rpa")
        self._domain_r_max = float(np.asarray(r_hi, dtype=float).reshape(3)[0])
        self._seed_lookup_state = lookup_geometry.coord_state
        self._seed_cell_plane_state = None

        if int(self.max_level) >= int(tree.max_level):
            face_neighbors = tree.face_neighbors(max_level=int(tree.max_level))
            self._face_neighbor_state = face_neighbors.kernel_state
            if self._tree_coord == "xyz":
                if not isinstance(self._seed_lookup_state, CartesianLookupKernelState):
                    raise TypeError(
                        "Cartesian ray tracing requires CartesianLookupKernelState; "
                        f"got {type(self._seed_lookup_state).__name__}."
                    )
                self._cell_lookup_state = self._seed_lookup_state
                self._cell_plane_state = None
            elif self._tree_coord == "rpa":
                if not isinstance(self._seed_lookup_state, SphericalLookupKernelState):
                    raise TypeError(
                        "Spherical ray tracing requires SphericalLookupKernelState; "
                        f"got {type(self._seed_lookup_state).__name__}."
                    )
                self._seed_cell_plane_state = _build_cell_plane_kernel_state(tree)
                self._cell_lookup_state = None
                self._cell_plane_state = self._seed_cell_plane_state
            else:
                raise ValueError(f"Unsupported tree_coord '{self._tree_coord}'.")
            return

        geometry = _ray_cell_geometry_for_max_level(tree, int(self.max_level))
        self._face_neighbor_state = geometry.face_neighbor_state
        if self._tree_coord == "xyz":
            if not isinstance(geometry, CartesianRayCellGeometry):
                raise TypeError(
                    "Cartesian ray tracing requires CartesianRayCellGeometry; "
                    f"got {type(geometry).__name__}."
                )
            self._cell_lookup_state = geometry.coord_state
            self._cell_plane_state = None
        elif self._tree_coord == "rpa":
            if not isinstance(geometry, SphericalRayCellGeometry):
                raise TypeError(
                    "Spherical ray tracing requires SphericalRayCellGeometry; "
                    f"got {type(geometry).__name__}."
                )
            face_neighbors = tree.face_neighbors(max_level=int(self.max_level))
            self._seed_lookup_state = _build_sparse_spherical_seed_lookup_state(tree, face_neighbors, geometry)
            self._seed_cell_plane_state = _build_sparse_seed_plane_state(
                face_neighbors.node_cell_ids,
                int(np.asarray(tree.cell_levels, dtype=np.int64).shape[0]),
                geometry.plane_state,
            )
            self._cell_lookup_state = None
            self._cell_plane_state = geometry.plane_state
        else:
            raise ValueError(f"Unsupported tree_coord '{self._tree_coord}'.")

        if self._tree_coord == "xyz":
            if not isinstance(self._seed_lookup_state, CartesianLookupKernelState):
                raise TypeError(
                    "Cartesian ray tracing requires CartesianLookupKernelState; "
                    f"got {type(self._seed_lookup_state).__name__}."
                )
            if not isinstance(self._cell_lookup_state, CartesianLookupKernelState):
                raise TypeError(
                    "Cartesian ray tracing requires CartesianLookupKernelState for active cells; "
                    f"got {type(self._cell_lookup_state).__name__}."
                )
            return
        if not isinstance(self._seed_lookup_state, SphericalLookupKernelState):
            raise TypeError(
                "Spherical ray tracing requires SphericalLookupKernelState; "
                f"got {type(self._seed_lookup_state).__name__}."
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

        face_state = self._face_neighbor_state
        if self._tree_coord == "xyz":
            n_seg, cids, enters, exits = _trace_segments_xyz_neighbors_kernel(
                o,
                d,
                t0,
                t1,
                int(max_steps),
                float(boundary_tol),
                self._seed_lookup_state,
                self._cell_lookup_state,
                face_state,
            )
        elif self._tree_coord == "rpa":
            n_seg, cids, enters, exits = _trace_segments_rpa_neighbors_kernel(
                o,
                d,
                t0,
                t1,
                int(max_steps),
                float(boundary_tol),
                self._seed_lookup_state,
                face_state,
                self._seed_cell_plane_state,
                self._cell_plane_state,
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
        directions = _coerce_directions_xyz(direction_xyz, n_rays=int(origins.shape[0]))
        t0, t1 = _coerce_ray_interval(t_start, t_end)
        chunk = _coerce_positive_chunk_size(chunk_size)
        max_steps = _coerce_positive_int("max_steps", max_steps)

        counts = np.zeros(origins.shape[0], dtype=np.int64)
        face_state = self._face_neighbor_state
        for start in range(0, int(origins.shape[0]), chunk):
            stop = min(int(origins.shape[0]), start + chunk)
            sub = origins[start:stop]
            if self._tree_coord == "xyz":
                counts[start:stop] = _segment_counts_xyz_neighbors_kernel(
                    sub,
                    directions[start:stop],
                    t0,
                    t1,
                    int(max_steps),
                    float(boundary_tol),
                    self._seed_lookup_state,
                    self._cell_lookup_state,
                    face_state,
                )
            elif self._tree_coord == "rpa":
                counts[start:stop] = _segment_counts_rpa_neighbors_kernel(
                    sub,
                    directions[start:stop],
                    t0,
                    t1,
                    int(max_steps),
                    float(boundary_tol),
                    self._seed_lookup_state,
                    face_state,
                    self._seed_cell_plane_state,
                    self._cell_plane_state,
                )
            else:
                raise ValueError(f"Unsupported tree_coord '{self._tree_coord}'.")
        return counts

class OctreeRayInterpolator:
    """Thin wrapper around compiled ray integration kernels for one interpolator."""

    def __init__(
        self,
        interpolator: "OctreeInterpolator",
        *,
        max_level: int | None = None,
    ) -> None:
        """Bind one interpolator."""
        from .interpolator import OctreeInterpolator

        if not isinstance(interpolator, OctreeInterpolator):
            raise TypeError(
                "OctreeRayInterpolator requires an OctreeInterpolator."
            )

        self.interpolator = interpolator
        self.tree = interpolator.tree
        self.max_level = _resolve_ray_max_level(self.tree, max_level)
        self.ray_tracer = OctreeRayTracer(self.tree, max_level=max_level)
        self._face_neighbor_state = self.ray_tracer._face_neighbor_state
        self._seed_lookup_state = self.ray_tracer._seed_lookup_state
        self._cell_lookup_state = self.ray_tracer._cell_lookup_state
        self._seed_cell_plane_state = self.ray_tracer._seed_cell_plane_state
        self._cell_plane_state = self.ray_tracer._cell_plane_state
        tree_coord = str(self.tree.tree_coord)
        if int(self.max_level) >= int(self.tree.max_level):
            if tree_coord == "xyz":
                self._interp_state = interpolator.xyz_interp_state
            elif tree_coord == "rpa":
                self._interp_state = getattr(interpolator, "_interp_state_rpa", None)
            else:
                raise ValueError(f"Unsupported tree_coord '{tree_coord}'.")
        else:
            geometry = _ray_cell_geometry_for_max_level(self.tree, int(self.max_level))
            point_values = np.asarray(interpolator.point_values, dtype=np.float64)
            point_values_2d = np.array(
                point_values.reshape(int(point_values.shape[0]), -1),
                dtype=np.float64,
                order="C",
            )
            if isinstance(geometry, CartesianRayCellGeometry):
                self._interp_state = _cartesian_interp_state_from_geometry(point_values_2d, geometry)
            elif isinstance(geometry, SphericalRayCellGeometry):
                self._interp_state = _spherical_interp_state_from_geometry(point_values_2d, geometry)
            else:
                raise TypeError(f"Unsupported truncated ray geometry {type(geometry).__name__}.")

        if tree_coord == "xyz":
            if not isinstance(self._seed_lookup_state, CartesianLookupKernelState):
                raise TypeError(
                    "Cartesian ray tracing requires CartesianLookupKernelState; "
                    f"got {type(self._seed_lookup_state).__name__}."
                )
            if not isinstance(self._cell_lookup_state, CartesianLookupKernelState):
                raise TypeError(
                    "Cartesian ray tracing requires CartesianLookupKernelState for active cells; "
                    f"got {type(self._cell_lookup_state).__name__}."
                )
            if not isinstance(self._interp_state, CartesianInterpKernelState):
                raise TypeError(
                    "Cartesian ray interpolation requires CartesianInterpKernelState; "
                    f"got {type(self._interp_state).__name__}."
                )
            return
        if not isinstance(self._seed_lookup_state, SphericalLookupKernelState):
            raise TypeError(
                "Spherical ray tracing requires SphericalLookupKernelState; "
                f"got {type(self._seed_lookup_state).__name__}."
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
        if not isinstance(self._interp_state, SphericalInterpKernelState):
            raise TypeError(
                "Spherical ray interpolation requires SphericalInterpKernelState; "
                f"got {type(self._interp_state).__name__}."
            )

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
        directions = _coerce_directions_xyz(direction_xyz, n_rays=int(origins.shape[0]))
        t0, t1 = _coerce_ray_interval(t_start, t_end)
        chunk = _coerce_positive_chunk_size(chunk_size)
        max_steps = _coerce_positive_int("max_steps", max_steps)

        n_rays = int(origins.shape[0])
        ncomp = int(self.interpolator.n_value_components)
        out = np.full((n_rays, ncomp), np.nan, dtype=float)

        tree_coord = str(self.tree.tree_coord)
        for start in range(0, n_rays, chunk):
            stop = min(n_rays, start + chunk)
            sub = origins[start:stop]
            if tree_coord == "xyz":
                out[start:stop] = _integrate_rays_xyz_neighbors_kernel(
                    sub,
                    directions[start:stop],
                    t0,
                    t1,
                    int(max_steps),
                    float(boundary_tol),
                    float(scale),
                    self._seed_lookup_state,
                    self._cell_lookup_state,
                    self._face_neighbor_state,
                    self._interp_state,
                )
            elif tree_coord == "rpa":
                out[start:stop] = _integrate_rays_rpa_neighbors_kernel(
                    sub,
                    directions[start:stop],
                    t0,
                    t1,
                    int(max_steps),
                    float(boundary_tol),
                    float(scale),
                    self._seed_lookup_state,
                    self._face_neighbor_state,
                    self._seed_cell_plane_state,
                    self._cell_plane_state,
                    self._interp_state,
                )
            else:
                raise ValueError(f"Unsupported tree_coord '{tree_coord}'.")

        if ncomp == 1:
            return out[:, 0]
        return out
