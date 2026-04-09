#!/usr/bin/env python3
"""Ray-domain seeding helpers for octrees."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .octree import Octree

__all__ = ["OctreeRayTracer", "RaySegments", "render_midpoint_image"]

_FACE_AXIS = np.array([0, 0, 1, 1, 2, 2], dtype=np.int8)
_FACE_SIDE = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
_FACE_TANGENTIAL_AXES = np.array(
    [
        [1, 2],
        [1, 2],
        [0, 2],
        [0, 2],
        [0, 1],
        [0, 1],
    ],
    dtype=np.int8,
)
_XYZ_CORNER_BITS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.int8,
)
_RPA_CORNER_BITS = np.array(
    [
        [0, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [0, 0, 1],
    ],
    dtype=np.int8,
)
_UNKNOWN_TRANSITION = np.int32(-2)
_NO_TRANSITION = np.int32(-1)
_EXIT_TOL = 1.0e-12
_SUBFACE_TOL = 1.0e-12
_BOUNDARY_SHIFT_FACTOR = 1.0e-6


def _dot3(a: np.ndarray, b: np.ndarray) -> float:
    """Return one 3D dot product as a Python float."""
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return one 3D cross product."""
    return np.array(
        (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ),
        dtype=np.float64,
    )


def _pack_leaf_key(depth: int, ijk: np.ndarray, axis_bases: np.ndarray) -> np.uint64:
    """Pack one `(depth, axis0, axis1, axis2)` leaf address into one sortable key."""
    key = np.uint64(depth)
    key = key * np.uint64(axis_bases[0]) + np.uint64(ijk[0])
    key = key * np.uint64(axis_bases[1]) + np.uint64(ijk[1])
    key = key * np.uint64(axis_bases[2]) + np.uint64(ijk[2])
    return key


def _build_face_corner_order(corner_bits: np.ndarray) -> np.ndarray:
    """Return one `(6, 4)` face-corner table in `[00, 01, 11, 10]` tangential order."""
    face_corners = np.full((6, 4), -1, dtype=np.int8)
    for face_id in range(6):
        axis = int(_FACE_AXIS[face_id])
        side = int(_FACE_SIDE[face_id])
        tangential_axes = _FACE_TANGENTIAL_AXES[face_id]
        targets = (
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 0),
        )
        for pos, (bit_first, bit_second) in enumerate(targets):
            target = np.full(3, -1, dtype=np.int8)
            target[axis] = side
            target[int(tangential_axes[0])] = bit_first
            target[int(tangential_axes[1])] = bit_second
            matches = np.flatnonzero(np.all(corner_bits == target, axis=1))
            if matches.size != 1:
                raise ValueError(f"Could not resolve one unique face corner for face_id={face_id}, target={target}.")
            face_corners[face_id, pos] = np.int8(matches[0])
    return face_corners


@dataclass(frozen=True)
class RaySegments:
    """Packed per-ray cell segments with CSR-like ray indexing."""

    ray_offsets: np.ndarray
    cell_ids: np.ndarray
    t_enter: np.ndarray
    t_exit: np.ndarray
    ray_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        ray_offsets = np.asarray(self.ray_offsets, dtype=np.int64)
        cell_ids = np.asarray(self.cell_ids, dtype=np.int64)
        t_enter = np.asarray(self.t_enter, dtype=np.float64)
        t_exit = np.asarray(self.t_exit, dtype=np.float64)
        if ray_offsets.ndim != 1:
            raise ValueError("ray_offsets must be one 1D array.")
        if cell_ids.ndim != 1 or t_enter.ndim != 1 or t_exit.ndim != 1:
            raise ValueError("cell_ids, t_enter, and t_exit must be 1D arrays.")
        if cell_ids.shape != t_enter.shape or cell_ids.shape != t_exit.shape:
            raise ValueError("cell_ids, t_enter, and t_exit must have the same shape.")
        if ray_offsets.size == 0 or ray_offsets[0] != 0:
            raise ValueError("ray_offsets must start at 0.")
        if np.any(np.diff(ray_offsets) < 0):
            raise ValueError("ray_offsets must be nondecreasing.")
        if int(ray_offsets[-1]) != int(cell_ids.size):
            raise ValueError("ray_offsets[-1] must equal the segment count.")
        if np.any(t_exit <= t_enter):
            raise ValueError("Each segment must satisfy t_exit > t_enter.")
        object.__setattr__(self, "ray_offsets", ray_offsets)
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "t_enter", t_enter)
        object.__setattr__(self, "t_exit", t_exit)
        object.__setattr__(self, "ray_shape", tuple(int(v) for v in self.ray_shape))

    @property
    def n_rays(self) -> int:
        """Return the number of rays packed into this segment bundle."""
        return int(self.ray_offsets.size - 1)

    @property
    def segment_length(self) -> np.ndarray:
        """Return one 1D array of per-segment lengths."""
        return self.t_exit - self.t_enter


class OctreeRayTracer:
    """Trace rays against one octree geometry.

    The current implementation only seeds rays inside the global octree domain.
    """

    def __init__(self, tree: Octree) -> None:
        """Bind one tracer to one built octree."""
        if not isinstance(tree, Octree):
            raise TypeError("OctreeRayTracer requires a built Octree as its first argument.")
        self.tree = tree
        if str(tree.tree_coord) == "xyz":
            self._face_corners = _build_face_corner_order(_XYZ_CORNER_BITS)
        elif str(tree.tree_coord) == "rpa":
            self._face_corners = _build_face_corner_order(_RPA_CORNER_BITS)
        else:
            raise NotImplementedError(f"Unsupported tree_coord '{tree.tree_coord}' for OctreeRayTracer.")
        self._leaf_slot_count = int(tree.corners.shape[0])
        self._leaf_valid = tree.cell_levels >= 0
        self._axis_bases = np.asarray(tree.leaf_shape, dtype=np.uint64) + np.uint64(1)
        self._max_level = int(tree.max_level)
        valid_leaf_ids = np.flatnonzero(self._leaf_valid).astype(np.int64)
        valid_depth = tree.cell_depth[valid_leaf_ids]
        valid_ijk = tree.cell_ijk[valid_leaf_ids]
        valid_keys = np.array(
            [_pack_leaf_key(int(valid_depth[i]), valid_ijk[i], self._axis_bases) for i in range(valid_leaf_ids.size)],
            dtype=np.uint64,
        )
        order = np.argsort(valid_keys)
        self._leaf_keys_sorted = valid_keys[order]
        self._leaf_ids_sorted = valid_leaf_ids[order].astype(np.int32, copy=False)
        self._next_leaf = np.full((self._leaf_slot_count, 6, 4), _UNKNOWN_TRANSITION, dtype=np.int32)

    @staticmethod
    def _normalize_rays(origins: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
        """Return flat finite ray arrays plus the leading broadcast shape."""
        o = np.array(origins, dtype=np.float64, order="C")
        d = np.array(directions, dtype=np.float64, order="C")
        if o.ndim == 0 or o.shape[-1] != 3:
            raise ValueError("origins must have shape (..., 3).")
        if d.shape != o.shape:
            raise ValueError("directions must have the same shape as origins.")
        if not np.all(np.isfinite(o)):
            raise ValueError("origins must contain only finite values.")
        if not np.all(np.isfinite(d)):
            raise ValueError("directions must contain only finite values.")
        d_flat = d.reshape(-1, 3)
        if np.any(np.linalg.norm(d_flat, axis=1) <= 0.0):
            raise ValueError("directions must be nonzero.")
        shape = (1,) if o.ndim == 1 else o.shape[:-1]
        return o.reshape(-1, 3), d_flat, shape

    @staticmethod
    def _sphere_interval(origin_xyz: np.ndarray, direction_xyz: np.ndarray, radius: float) -> tuple[bool, float, float]:
        """Return one line-parameter interval inside one sphere centered at the origin."""
        ox = float(origin_xyz[0])
        oy = float(origin_xyz[1])
        oz = float(origin_xyz[2])
        dx = float(direction_xyz[0])
        dy = float(direction_xyz[1])
        dz = float(direction_xyz[2])
        rr = float(radius) * float(radius)
        a = dx * dx + dy * dy + dz * dz
        b = ox * dx + oy * dy + oz * dz
        c = ox * ox + oy * oy + oz * oz - rr
        disc = b * b - a * c
        if disc < 0.0:
            return False, np.nan, np.nan
        root = math.sqrt(max(0.0, disc))
        t0 = (-b - root) / a
        t1 = (-b + root) / a
        if t0 <= t1:
            return True, t0, t1
        return True, t1, t0

    def _lookup_leaf(self, depth: int, ijk: np.ndarray) -> int:
        """Return one exact leaf-slot id for one `(depth, ijk)` address, or `-1`."""
        if depth < 0 or depth > self._max_level:
            return -1
        if np.any(ijk < 0):
            return -1
        key = _pack_leaf_key(int(depth), ijk, self._axis_bases)
        pos = int(np.searchsorted(self._leaf_keys_sorted, key))
        if pos >= int(self._leaf_keys_sorted.size) or self._leaf_keys_sorted[pos] != key:
            return -1
        return int(self._leaf_ids_sorted[pos])

    def _resolve_transition(self, leaf_id: int, face_id: int, subface_id: int) -> int:
        """Return the neighboring leaf across one face/subface, or `-1` at the domain boundary.

        Face order is `(xmin, xmax, ymin, ymax, zmin, zmax)`.

        Subfaces always split the crossed face into a `2 x 2` grid. Their bits
        are ordered along the two tangential axes of that face:

        - `x*` faces: `(y_bit, z_bit)`
        - `y*` faces: `(x_bit, z_bit)`
        - `z*` faces: `(x_bit, y_bit)`

        with `subface_id = 2 * first_bit + second_bit`.
        """
        leaf = int(leaf_id)
        face = int(face_id)
        subface = int(subface_id)
        if leaf < 0 or leaf >= self._leaf_slot_count or not bool(self._leaf_valid[leaf]):
            raise ValueError("leaf_id must reference one valid leaf slot.")
        if face < 0 or face >= 6:
            raise ValueError("face_id must be in the range [0, 5].")
        if subface < 0 or subface >= 4:
            raise ValueError("subface_id must be in the range [0, 3].")

        cached = int(self._next_leaf[leaf, face, subface])
        if cached != int(_UNKNOWN_TRANSITION):
            return cached

        axis = int(_FACE_AXIS[face])
        side = int(_FACE_SIDE[face])
        tangential_axes = _FACE_TANGENTIAL_AXES[face]
        delta = -1 if side == 0 else 1
        depth = int(self.tree.cell_depth[leaf])
        ijk = self.tree.cell_ijk[leaf]

        next_leaf = _NO_TRANSITION

        same_ijk = np.array(ijk, copy=True)
        same_ijk[axis] += delta
        next_leaf = np.int32(self._lookup_leaf(depth, same_ijk))

        if next_leaf < 0 and depth > 0:
            coarse_ijk = np.right_shift(ijk, 1).astype(np.int64, copy=False)
            coarse_ijk = np.array(coarse_ijk, copy=True)
            coarse_ijk[axis] = (int(ijk[axis]) + delta) >> 1
            next_leaf = np.int32(self._lookup_leaf(depth - 1, coarse_ijk))

        if next_leaf < 0 and depth < self._max_level:
            fine_ijk = np.left_shift(ijk, 1).astype(np.int64, copy=False)
            fine_ijk = np.array(fine_ijk, copy=True)
            fine_ijk[int(tangential_axes[0])] += (subface >> 1) & 1
            fine_ijk[int(tangential_axes[1])] += subface & 1
            fine_ijk[axis] = 2 * int(ijk[axis]) + (-1 if side == 0 else 2)
            next_leaf = np.int32(self._lookup_leaf(depth + 1, fine_ijk))

        self._next_leaf[leaf, face, subface] = np.int32(next_leaf)
        return int(next_leaf)

    def _cell_faces(self, leaf_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return one leaf's explicit corners and geometric center in physical `xyz`."""
        cell_xyz = np.asarray(self.tree.cell_points(int(leaf_id)), dtype=np.float64)
        return cell_xyz, np.mean(cell_xyz, axis=0)

    @staticmethod
    def _face_normal(face_xyz: np.ndarray, cell_center_xyz: np.ndarray) -> np.ndarray:
        """Return one outward face normal for one planar face quad."""
        normal = _cross3(face_xyz[1] - face_xyz[0], face_xyz[2] - face_xyz[1])
        if _dot3(normal, np.mean(face_xyz, axis=0) - cell_center_xyz) < 0.0:
            normal = -normal
        return normal

    @staticmethod
    def _point_inside_face(face_xyz: np.ndarray, normal_xyz: np.ndarray, point_xyz: np.ndarray, tol: float) -> bool:
        """Return whether one point lies inside one convex planar face quad."""
        signs = np.empty(4, dtype=np.float64)
        for edge_id in range(4):
            p0 = face_xyz[edge_id]
            p1 = face_xyz[(edge_id + 1) % 4]
            signs[edge_id] = _dot3(_cross3(p1 - p0, point_xyz - p0), normal_xyz)
        if bool(np.all(signs >= -tol) or np.all(signs <= tol)):
            return True

        edge_scale = float(
            np.max(np.linalg.norm(np.roll(face_xyz, -1, axis=0) - face_xyz, axis=1))
        )
        edge_tol = max(float(tol), (2.0 * _BOUNDARY_SHIFT_FACTOR) * max(1.0, edge_scale))
        for edge_id in range(4):
            p0 = face_xyz[edge_id]
            p1 = face_xyz[(edge_id + 1) % 4]
            edge = p1 - p0
            edge_sq = _dot3(edge, edge)
            if edge_sq <= 0.0:
                distance = float(np.linalg.norm(point_xyz - p0))
            else:
                weight = np.clip(_dot3(point_xyz - p0, edge) / edge_sq, 0.0, 1.0)
                closest_xyz = p0 + weight * edge
                distance = float(np.linalg.norm(point_xyz - closest_xyz))
            if distance <= edge_tol:
                return True
        return False

    def _point_inside_cell(self, leaf_id: int, point_xyz: np.ndarray, tol: float) -> bool:
        """Return whether one point lies inside one convex planar-face cell."""
        cell_xyz, cell_center = self._cell_faces(leaf_id)
        for face_id in range(6):
            face_xyz = cell_xyz[self._face_corners[face_id]]
            normal_xyz = self._face_normal(face_xyz, cell_center)
            if _dot3(normal_xyz, point_xyz - face_xyz[0]) > tol:
                return False
        return True

    def _face_exit_subface(self, leaf_id: int, exit_xyz: np.ndarray, face_id: int, normal_xyz: np.ndarray) -> int:
        """Return the crossed face quadrant using the face's two native tangential axes."""
        cell_xyz, _ = self._cell_faces(leaf_id)
        c00, c01, c11, c10 = cell_xyz[self._face_corners[int(face_id)]]

        first_split0 = 0.5 * (c00 + c10)
        first_split1 = 0.5 * (c01 + c11)
        second_split0 = 0.5 * (c00 + c01)
        second_split1 = 0.5 * (c10 + c11)

        def classify(split0: np.ndarray, split1: np.ndarray, low_ref: np.ndarray, high_ref: np.ndarray) -> int:
            split_direction = split1 - split0
            low_sign = _dot3(_cross3(split_direction, low_ref - split0), normal_xyz)
            high_sign = _dot3(_cross3(split_direction, high_ref - split0), normal_xyz)
            hit_sign = _dot3(_cross3(split_direction, exit_xyz - split0), normal_xyz)
            if abs(hit_sign) <= _SUBFACE_TOL:
                raise ValueError("Degenerate face exit on a subface boundary is not supported yet.")
            if low_sign == 0.0 or high_sign == 0.0 or low_sign * high_sign >= 0.0:
                raise ValueError("Failed to construct a valid face subface split.")
            return 0 if hit_sign * low_sign > 0.0 else 1

        first_bit = classify(first_split0, first_split1, c00, c10)
        second_bit = classify(second_split0, second_split1, c00, c01)
        return 2 * first_bit + second_bit

    def _touching_neighbor_leaves(self, leaf_id: int, point_xyz: np.ndarray) -> list[int]:
        """Return unique neighboring leaves whose cells touch one boundary point of one leaf."""
        cell_xyz, cell_center = self._cell_faces(leaf_id)
        point_tol = _EXIT_TOL * max(1.0, float(np.max(np.linalg.norm(cell_xyz - cell_center, axis=1))))
        neighbors: list[int] = []
        for face_id in range(6):
            face_xyz = cell_xyz[self._face_corners[face_id]]
            normal_xyz = self._face_normal(face_xyz, cell_center)
            face_distance = _dot3(normal_xyz, point_xyz - face_xyz[0])
            if abs(face_distance) > point_tol:
                continue
            for subface_id in range(4):
                next_leaf = int(self._resolve_transition(leaf_id, face_id, subface_id))
                if next_leaf >= 0 and next_leaf not in neighbors:
                    neighbors.append(next_leaf)
        return neighbors

    def _boundary_continuation_leaf(
        self,
        initial_leaf_ids: list[int],
        point_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        *,
        exclude_leaf_ids: tuple[int, ...] = (),
    ) -> int:
        """Return one deterministic continuation leaf from one boundary point, or `-1`."""
        queue = [int(v) for v in initial_leaf_ids]
        visited: set[int] = set()
        valid: list[tuple[float, float, int, float]] = []

        while queue:
            leaf_id = int(queue.pop(0))
            if leaf_id < 0 or leaf_id in visited:
                continue
            visited.add(leaf_id)
            try:
                segment_enter, segment_exit, _, _ = self._cell_segment(
                    leaf_id,
                    point_xyz,
                    direction_xyz,
                    current_t=0.0,
                    t_min=0.0,
                )
            except ValueError as exc:
                message = str(exc)
                if message not in {
                    "The requested leaf does not intersect the ray.",
                    "Failed to resolve a forward cell segment from the current leaf.",
                    "Degenerate cell exit across an edge or corner is not supported yet.",
                    "Degenerate face exit on a subface boundary is not supported yet.",
                }:
                    raise
            else:
                if leaf_id not in exclude_leaf_ids:
                    valid.append(
                        (
                            float(segment_enter),
                            float(segment_exit - segment_enter),
                            int(leaf_id),
                            float(segment_exit),
                        )
                    )
                    continue
            for next_leaf in self._touching_neighbor_leaves(leaf_id, point_xyz):
                if next_leaf not in visited:
                    queue.append(int(next_leaf))

        if str(self.tree.tree_coord) == "rpa":
            for candidate_leaf in self._rpa_axis_candidate_leaves(point_xyz, direction_xyz):
                if candidate_leaf in visited or candidate_leaf in exclude_leaf_ids:
                    continue
                try:
                    segment_enter, segment_exit, _, _ = self._cell_segment(
                        candidate_leaf,
                        point_xyz,
                        direction_xyz,
                        current_t=0.0,
                        t_min=0.0,
                    )
                except ValueError as exc:
                    message = str(exc)
                    if message not in {
                        "The requested leaf does not intersect the ray.",
                        "Failed to resolve a forward cell segment from the current leaf.",
                        "Degenerate cell exit across an edge or corner is not supported yet.",
                        "Degenerate face exit on a subface boundary is not supported yet.",
                    }:
                        raise
                else:
                    valid.append(
                        (
                            float(segment_enter),
                            float(segment_exit - segment_enter),
                            int(candidate_leaf),
                            float(segment_exit),
                        )
                    )
        if not valid:
            return -1
        valid.sort(key=lambda item: (item[0], -item[1], -item[3], item[2]))
        return int(valid[0][2])

    def _rpa_axis_candidate_leaves(self, point_xyz: np.ndarray, direction_xyz: np.ndarray | None = None) -> list[int]:
        """Return one finite set of native-sector candidate leaves for one spherical axis point."""
        from .spherical import xyz_to_rpa_components

        x = float(point_xyz[0])
        y = float(point_xyz[1])
        z = float(point_xyz[2])
        radial_xy = math.hypot(x, y)
        radial_scale = max(float(np.linalg.norm(point_xyz)), 1.0)
        axis_tol = _BOUNDARY_SHIFT_FACTOR * radial_scale
        if radial_xy > axis_tol:
            return []

        unique_leaf: list[int] = []
        if direction_xyz is not None:
            direction_norm = float(np.linalg.norm(direction_xyz))
            if direction_norm <= 0.0:
                raise ValueError("directions must be nonzero.")
            probe_xyz = np.asarray(point_xyz, dtype=np.float64) + (
                (_BOUNDARY_SHIFT_FACTOR * radial_scale) / direction_norm
            ) * np.asarray(direction_xyz, dtype=np.float64)
            forward_leaf = int(self.tree.lookup_points(probe_xyz.reshape(1, 3), coord="xyz")[0])
            if forward_leaf >= 0:
                unique_leaf.append(forward_leaf)

        r_value, _, _ = xyz_to_rpa_components(x, y, z)
        n_azimuth = int(self.tree.leaf_shape[2])
        n_polar = int(self.tree.leaf_shape[1])
        polar_center = 0.5 * (math.pi / float(n_polar))
        if z < 0.0:
            polar_center = math.pi - polar_center
        azimuth_centers = (np.arange(n_azimuth, dtype=np.float64) + 0.5) * ((2.0 * math.pi) / float(n_azimuth))
        query_rpa = np.column_stack(
            (
                np.full(n_azimuth, r_value, dtype=np.float64),
                np.full(n_azimuth, polar_center, dtype=np.float64),
                np.mod(azimuth_centers, 2.0 * math.pi),
            )
        )
        candidate_leaf = self.tree.lookup_points(query_rpa, coord="rpa").reshape(-1)
        for leaf_id in candidate_leaf:
            resolved = int(leaf_id)
            if resolved >= 0 and resolved not in unique_leaf:
                unique_leaf.append(resolved)
        return unique_leaf

    def _seed_interval_candidates(
        self,
        seed_leaf_id: int,
        seed_xyz: np.ndarray,
        direction_xyz: np.ndarray,
    ) -> list[tuple[float, float, int]]:
        """Return candidate ray intervals near one approximate seed point."""
        candidate_leaf_ids: list[int] = [int(seed_leaf_id)]
        for leaf_id in self._touching_neighbor_leaves(int(seed_leaf_id), seed_xyz):
            if leaf_id not in candidate_leaf_ids:
                candidate_leaf_ids.append(int(leaf_id))
        if str(self.tree.tree_coord) == "rpa":
            for leaf_id in self._rpa_axis_candidate_leaves(seed_xyz, direction_xyz):
                if leaf_id not in candidate_leaf_ids:
                    candidate_leaf_ids.append(int(leaf_id))

        intervals: list[tuple[float, float, int]] = []
        for leaf_id in candidate_leaf_ids:
            try:
                enter, exit_, _, _ = self._cell_segment(
                    leaf_id,
                    seed_xyz,
                    direction_xyz,
                    current_t=0.0,
                    t_min=0.0,
                )
            except ValueError as exc:
                if str(exc) not in {
                    "The requested leaf does not intersect the ray.",
                    "Failed to resolve a forward cell segment from the current leaf.",
                    "Degenerate cell exit across an edge or corner is not supported yet.",
                    "Degenerate face exit on a subface boundary is not supported yet.",
                }:
                    raise
            else:
                intervals.append((float(enter), float(exit_), int(leaf_id)))

            try:
                back_enter, back_exit, _, _ = self._cell_segment(
                    leaf_id,
                    seed_xyz,
                    -direction_xyz,
                    current_t=0.0,
                    t_min=0.0,
                )
            except ValueError as exc:
                if str(exc) not in {
                    "The requested leaf does not intersect the ray.",
                    "Failed to resolve a forward cell segment from the current leaf.",
                    "Degenerate cell exit across an edge or corner is not supported yet.",
                    "Degenerate face exit on a subface boundary is not supported yet.",
                }:
                    raise
            else:
                intervals.append((-float(back_exit), -float(back_enter), int(leaf_id)))
        return intervals

    def _canonicalize_seed(
        self,
        seed_leaf_id: int,
        seed_xyz: np.ndarray,
        direction_xyz: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Return one seed point guaranteed to lie inside one traced cell interval."""
        intervals = self._seed_intervals(int(seed_leaf_id), seed_xyz, direction_xyz)
        if not intervals:
            raise ValueError("Failed to place the seed point inside any traced cell interval.")

        def sort_key(item: tuple[float, float, int]) -> tuple[int, float, float, int]:
            enter, exit_, leaf_id = item
            contains_zero = 0 if (enter <= 0.0 <= exit_) else 1
            midpoint = 0.5 * (enter + exit_)
            length = exit_ - enter
            return contains_zero, -length, abs(midpoint), int(leaf_id)

        best_enter, best_exit, best_leaf = min(intervals, key=sort_key)
        seed_offset = 0.5 * (best_enter + best_exit)
        canonical_seed = np.asarray(seed_xyz, dtype=np.float64) + seed_offset * np.asarray(direction_xyz, dtype=np.float64)
        return canonical_seed, int(best_leaf)

    @staticmethod
    def _combine_seed_intervals(
        intervals: list[tuple[float, float, int]],
    ) -> list[tuple[float, float, int]]:
        """Combine same-leaf local ray intervals into one convex cell interval."""
        by_leaf: dict[int, tuple[float, float]] = {}
        for enter, exit_, leaf_id in intervals:
            if leaf_id in by_leaf:
                old_enter, old_exit = by_leaf[leaf_id]
                by_leaf[leaf_id] = (min(old_enter, enter), max(old_exit, exit_))
            else:
                by_leaf[leaf_id] = (float(enter), float(exit_))
        return [(enter, exit_, leaf_id) for leaf_id, (enter, exit_) in by_leaf.items()]

    @staticmethod
    def _usable_seed_intervals(
        intervals: list[tuple[float, float, int]],
    ) -> list[tuple[float, float, int]]:
        """Drop numerically collapsed seed intervals when a real local branch exists."""
        intervals = OctreeRayTracer._combine_seed_intervals(intervals)
        if not intervals:
            return []
        max_length = max(exit_ - enter for enter, exit_, _ in intervals)
        min_length = 1.0e-6 * max_length
        usable = [item for item in intervals if (item[1] - item[0]) > min_length]
        if usable:
            return usable
        return intervals

    def _seed_intervals(
        self,
        seed_leaf_id: int,
        seed_xyz: np.ndarray,
        direction_xyz: np.ndarray,
    ) -> list[tuple[float, float, int]]:
        """Return usable local seed intervals for one approximate seed point."""
        return self._usable_seed_intervals(
            self._seed_interval_candidates(int(seed_leaf_id), seed_xyz, direction_xyz)
        )

    def _select_common_seed(
        self,
        seed_leaf_id: int,
        seed_xyz: np.ndarray,
        direction_xyz: np.ndarray,
    ) -> tuple[np.ndarray, int] | None:
        """Return one shared interior seed when one leaf spans both ray directions."""
        intervals = self._seed_intervals(int(seed_leaf_id), seed_xyz, direction_xyz)
        shared = [item for item in intervals if item[0] < 0.0 and item[1] > 0.0]
        if not shared:
            return None

        def sort_key(item: tuple[float, float, int]) -> tuple[float, float, int]:
            enter, exit_, leaf_id = item
            midpoint = 0.5 * (enter + exit_)
            length = exit_ - enter
            return abs(midpoint), -length, int(leaf_id)

        best_enter, best_exit, best_leaf = min(shared, key=sort_key)
        seed_offset = 0.5 * (best_enter + best_exit)
        common_seed = np.asarray(seed_xyz, dtype=np.float64) + seed_offset * np.asarray(direction_xyz, dtype=np.float64)
        return common_seed, int(best_leaf)

    def _select_seed_branch(
        self,
        seed_leaf_id: int,
        seed_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        *,
        branch: str,
    ) -> tuple[np.ndarray, int] | None:
        """Return one interior seed point and leaf for one branch off the approximate seed."""
        intervals = self._seed_intervals(int(seed_leaf_id), seed_xyz, direction_xyz)
        if branch == "forward":
            branch_intervals = [item for item in intervals if item[1] > 0.0]
            if not branch_intervals:
                return None

            def sort_key(item: tuple[float, float, int]) -> tuple[float, float, float, int]:
                enter, exit_, leaf_id = item
                start_distance = max(0.0, enter)
                midpoint = 0.5 * (enter + exit_)
                length = exit_ - enter
                return start_distance, -length, abs(midpoint), int(leaf_id)

        elif branch == "backward":
            branch_intervals = [item for item in intervals if item[0] < 0.0]
            if not branch_intervals:
                return None

            def sort_key(item: tuple[float, float, int]) -> tuple[float, float, float, int]:
                enter, exit_, leaf_id = item
                end_distance = max(0.0, -exit_)
                midpoint = 0.5 * (enter + exit_)
                length = exit_ - enter
                return end_distance, -length, abs(midpoint), int(leaf_id)

        else:
            raise ValueError("branch must be 'forward' or 'backward'.")

        best_enter, best_exit, best_leaf = min(branch_intervals, key=sort_key)
        seed_offset = 0.5 * (best_enter + best_exit)
        branch_seed = np.asarray(seed_xyz, dtype=np.float64) + seed_offset * np.asarray(direction_xyz, dtype=np.float64)
        return branch_seed, int(best_leaf)

    def _cell_segment(
        self,
        leaf_id: int,
        origin_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        *,
        current_t: float,
        t_min: float,
    ) -> tuple[float, float, int, int]:
        """Return one exact forward cell segment and its exit face/subface."""
        cell_xyz, cell_center = self._cell_faces(leaf_id)
        point_tol = _EXIT_TOL * max(1.0, float(np.max(np.linalg.norm(cell_xyz - cell_center, axis=1))))
        def solve(local_origin_xyz: np.ndarray) -> tuple[float, float, int, int]:
            inside_now = self._point_inside_cell(int(leaf_id), local_origin_xyz + float(current_t) * direction_xyz, point_tol)

            face_hits: list[tuple[float, int, np.ndarray, np.ndarray]] = []
            for face_id in range(6):
                face_xyz = cell_xyz[self._face_corners[face_id]]
                normal_xyz = self._face_normal(face_xyz, cell_center)
                denom = _dot3(normal_xyz, direction_xyz)
                if abs(denom) <= _EXIT_TOL:
                    continue
                t_hit = _dot3(normal_xyz, face_xyz[0] - local_origin_xyz) / denom
                hit_xyz = local_origin_xyz + t_hit * direction_xyz
                if self._point_inside_face(face_xyz, normal_xyz, hit_xyz, point_tol):
                    face_hits.append((t_hit, int(face_id), hit_xyz, normal_xyz))

            if not face_hits:
                raise ValueError("The requested leaf does not intersect the ray.")

            face_hits.sort(key=lambda item: item[0])
            segment_enter = float(current_t) if inside_now else np.nan
            exit_t = np.nan
            exit_face = -1
            exit_xyz = np.full(3, np.nan, dtype=np.float64)
            exit_normal = np.full(3, np.nan, dtype=np.float64)
            enter_tol = _EXIT_TOL * max(1.0, abs(float(current_t)))

            for t_hit, face_id, hit_xyz, normal_xyz in face_hits:
                if t_hit < (float(t_min) - enter_tol):
                    continue
                if not np.isfinite(segment_enter):
                    segment_enter = float(t_hit)
                    continue
                if t_hit <= (float(segment_enter) + enter_tol):
                    continue
                exit_t = float(t_hit)
                exit_face = int(face_id)
                exit_xyz = hit_xyz
                exit_normal = normal_xyz
                break

            if not np.isfinite(segment_enter) or not np.isfinite(exit_t):
                raise ValueError("Failed to resolve a forward cell segment from the current leaf.")
            if exit_t <= segment_enter:
                raise ValueError("Degenerate zero-length cell interval is not supported yet.")

            exit_tol = _EXIT_TOL * max(1.0, abs(float(exit_t)))
            degenerate_hits = [face_id for t_hit, face_id, _, _ in face_hits if abs(t_hit - exit_t) <= exit_tol]
            degenerate_face_exit = len(degenerate_hits) > 1
            if len(degenerate_hits) > 1:
                exit_face = int(min(degenerate_hits))

            if degenerate_face_exit:
                subface_id = -1
            else:
                try:
                    subface_id = self._face_exit_subface(leaf_id, exit_xyz, exit_face, exit_normal)
                except ValueError as exc:
                    if str(exc) != "Degenerate face exit on a subface boundary is not supported yet.":
                        raise
                    subface_id = -1
            return float(segment_enter), float(exit_t), int(exit_face), int(subface_id)

        try:
            return solve(origin_xyz)
        except ValueError as exc:
            if str(exc) != "Failed to resolve a forward cell segment from the current leaf.":
                raise

        direction_norm = float(np.linalg.norm(direction_xyz))
        if direction_norm <= 0.0:
            raise ValueError("directions must be nonzero.")
        boundary_shift = (_BOUNDARY_SHIFT_FACTOR * max(1.0, float(np.max(np.linalg.norm(cell_xyz, axis=1))))) / direction_norm
        segment_enter, exit_t, exit_face, subface_id = solve(origin_xyz + boundary_shift * direction_xyz)
        return (
            float(segment_enter + boundary_shift),
            float(exit_t + boundary_shift),
            int(exit_face),
            int(subface_id),
        )

    def _trace_one_ray(
        self,
        start_leaf_id: int,
        origin_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        *,
        t_min: float = 0.0,
        t_max: float = np.inf,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Walk one ray across neighboring planar-face hex cells and return exact cell intervals.

        This is an internal first-pass walker used to validate the topology cache.
        It assumes a non-degenerate ray that crosses one face at a time.
        """
        clip_lo = float(t_min)
        clip_hi = float(t_max)
        if not math.isfinite(clip_lo):
            raise ValueError("t_min must be finite.")
        if math.isnan(clip_hi):
            raise ValueError("t_max must not be NaN.")
        if clip_hi < clip_lo:
            raise ValueError("t_max must be greater than or equal to t_min.")
        o_flat, d_flat, shape = self._normalize_rays(origin_xyz, direction_xyz)
        if shape != (1,):
            raise ValueError("_trace_one_ray requires one origin and one direction vector.")

        current_leaf = int(start_leaf_id)
        if current_leaf < 0 or current_leaf >= self._leaf_slot_count or not bool(self._leaf_valid[current_leaf]):
            raise ValueError("start_leaf_id must reference one valid leaf slot.")

        origin = o_flat[0]
        direction = d_flat[0]
        leaf_ids: list[int] = []
        t_enter_list: list[float] = []
        t_exit_list: list[float] = []
        current_t = clip_lo
        max_steps = int(np.count_nonzero(self._leaf_valid)) + 1

        for _ in range(max_steps):
            if current_t >= clip_hi:
                break
            segment_enter, t_exit, exit_face, subface_id = self._cell_segment(
                current_leaf,
                origin,
                direction,
                current_t=float(current_t),
                t_min=float(clip_lo),
            )
            if segment_enter >= clip_hi:
                break
            segment_exit = min(float(t_exit), clip_hi)

            leaf_ids.append(int(current_leaf))
            t_enter_list.append(float(segment_enter))
            t_exit_list.append(float(segment_exit))

            if t_exit >= clip_hi:
                break
            exit_xyz = origin + float(t_exit) * direction
            next_leaf = -1
            if subface_id >= 0:
                next_leaf = int(self._resolve_transition(current_leaf, exit_face, subface_id))
                if next_leaf >= 0:
                    next_leaf = self._boundary_continuation_leaf(
                        [next_leaf],
                        exit_xyz,
                        direction,
                        exclude_leaf_ids=(int(current_leaf),),
                    )
            if next_leaf < 0:
                next_leaf = self._boundary_continuation_leaf(
                    self._touching_neighbor_leaves(current_leaf, exit_xyz),
                    exit_xyz,
                    direction,
                    exclude_leaf_ids=(int(current_leaf),),
                )
            if next_leaf < 0:
                break
            current_leaf = int(next_leaf)
            current_t = float(t_exit)

        return (
            np.asarray(leaf_ids, dtype=np.int64),
            np.asarray(t_enter_list, dtype=np.float64),
            np.asarray(t_exit_list, dtype=np.float64),
        )

    @staticmethod
    def _normalize_seed(seed_xyz: np.ndarray, ray_shape: tuple[int, ...]) -> np.ndarray:
        """Return one flat seed array matching one normalized ray shape."""
        seed = np.array(seed_xyz, dtype=np.float64, order="C")
        if seed.shape == (3,) and ray_shape == (1,):
            seed = seed.reshape(1, 3)
        if seed.shape != ray_shape + (3,):
            raise ValueError("seed_xyz must have the same shape as the ray origins.")
        if np.any(~(np.isnan(seed) | np.isfinite(seed))):
            raise ValueError("seed_xyz must contain only finite values or NaNs.")
        return seed.reshape(-1, 3)

    @staticmethod
    def _merge_seed_branches(
        backward_leaf_ids: np.ndarray,
        backward_t_enter: np.ndarray,
        backward_t_exit: np.ndarray,
        forward_leaf_ids: np.ndarray,
        forward_t_enter: np.ndarray,
        forward_t_exit: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Merge one backward and one forward seed trace into ascending original-ray order."""
        if backward_leaf_ids.size == 0:
            return forward_leaf_ids, forward_t_enter, forward_t_exit
        if forward_leaf_ids.size == 0:
            return backward_leaf_ids, backward_t_enter, backward_t_exit

        if backward_leaf_ids[-1] != forward_leaf_ids[0]:
            return (
                np.concatenate((backward_leaf_ids, forward_leaf_ids)),
                np.concatenate((backward_t_enter, forward_t_enter)),
                np.concatenate((backward_t_exit, forward_t_exit)),
            )

        join_tol = _EXIT_TOL * max(1.0, abs(float(backward_t_exit[-1])), abs(float(forward_t_enter[0])))
        if abs(float(backward_t_exit[-1]) - float(forward_t_enter[0])) > join_tol:
            return (
                np.concatenate((backward_leaf_ids, forward_leaf_ids)),
                np.concatenate((backward_t_enter, forward_t_enter)),
                np.concatenate((backward_t_exit, forward_t_exit)),
            )

        merged_leaf = np.concatenate((backward_leaf_ids[:-1], forward_leaf_ids))
        merged_enter = np.concatenate((backward_t_enter[:-1], np.array([backward_t_enter[-1]]), forward_t_enter[1:]))
        merged_exit = np.concatenate((backward_t_exit[:-1], np.array([forward_t_exit[0]]), forward_t_exit[1:]))
        return merged_leaf, merged_enter, merged_exit

    def trace(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        *,
        seed_xyz: np.ndarray | None = None,
        t_min: float = 0.0,
        t_max: float = np.inf,
    ) -> RaySegments:
        """Trace one batch of seeded rays and return packed per-cell intervals."""
        t_lo = float(t_min)
        t_hi = float(t_max)
        if not math.isfinite(t_lo):
            raise ValueError("t_min must be finite.")
        if math.isnan(t_hi):
            raise ValueError("t_max must not be NaN.")
        if t_hi < t_lo:
            raise ValueError("t_max must be greater than or equal to t_min.")

        o_flat, d_flat, ray_shape = self._normalize_rays(origins, directions)
        seed_flat = (
            self.seed_domain(origins, directions, t_min=t_lo, t_max=t_hi).reshape(-1, 3)
            if seed_xyz is None
            else self._normalize_seed(seed_xyz, ray_shape)
        )
        trace_origin = np.array(o_flat, copy=True)
        trace_seed = np.array(seed_flat, copy=True)
        if str(self.tree.tree_coord) == "rpa":
            seam_mask = (
                np.isfinite(trace_seed[:, 1])
                & (np.abs(trace_origin[:, 1]) <= _BOUNDARY_SHIFT_FACTOR)
                & (np.abs(d_flat[:, 1]) <= _BOUNDARY_SHIFT_FACTOR)
                & (np.abs(trace_seed[:, 1]) <= _BOUNDARY_SHIFT_FACTOR)
            )
            trace_origin[seam_mask, 1] = _BOUNDARY_SHIFT_FACTOR
            trace_seed[seam_mask, 1] = _BOUNDARY_SHIFT_FACTOR
        clip_lo = max(0.0, t_lo)
        valid_seed = np.all(np.isfinite(seed_flat), axis=1)
        start_leaf = np.full(o_flat.shape[0], -1, dtype=np.int64)
        if np.any(valid_seed):
            start_leaf[valid_seed] = self.tree.lookup_points(trace_seed[valid_seed], coord="xyz").reshape(-1)

        def launch_branch(
            start_leaf_id: int,
            start_xyz: np.ndarray,
            launch_direction: np.ndarray,
            limit: float,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if limit <= 0.0:
                return (
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.float64),
                    np.empty(0, dtype=np.float64),
                )
            try:
                return self._trace_one_ray(
                    start_leaf_id,
                    start_xyz,
                    launch_direction,
                    t_min=0.0,
                    t_max=limit,
                )
            except ValueError as exc:
                if str(exc) != "Failed to resolve a forward cell segment from the current leaf.":
                    raise
                return (
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.float64),
                    np.empty(0, dtype=np.float64),
                )

        ray_offsets = np.zeros(o_flat.shape[0] + 1, dtype=np.int64)
        cell_ids: list[int] = []
        t_enter: list[float] = []
        t_exit: list[float] = []

        for ray_id in range(o_flat.shape[0]):
            ray_offsets[ray_id] = len(cell_ids)
            if not valid_seed[ray_id]:
                continue
            leaf_id = int(start_leaf[ray_id])
            if leaf_id < 0:
                continue

            origin = trace_origin[ray_id]
            direction = d_flat[ray_id]
            seed = trace_seed[ray_id]
            common_start = self._select_common_seed(leaf_id, seed, direction)
            backward_choice = self._select_seed_branch(leaf_id, seed, direction, branch="backward")
            forward_choice = self._select_seed_branch(leaf_id, seed, direction, branch="forward")

            if common_start is None and backward_choice is None and forward_choice is None:
                continue

            if common_start is not None or backward_choice is None or forward_choice is None:
                if common_start is not None:
                    seed, leaf_id = common_start
                else:
                    seed, leaf_id = self._canonicalize_seed(leaf_id, seed, direction)
                t_seed = float(np.dot(seed - origin, direction) / np.dot(direction, direction))

                backward_limit = max(0.0, t_seed - clip_lo)
                back_leaf_ids_local, back_enter_local, back_exit_local = launch_branch(
                    leaf_id,
                    seed,
                    -direction,
                    backward_limit,
                )
                if back_leaf_ids_local.size > 0:
                    back_leaf_ids = back_leaf_ids_local[::-1]
                    back_t_enter = t_seed - back_exit_local[::-1]
                    back_t_exit = t_seed - back_enter_local[::-1]
                else:
                    back_leaf_ids = np.empty(0, dtype=np.int64)
                    back_t_enter = np.empty(0, dtype=np.float64)
                    back_t_exit = np.empty(0, dtype=np.float64)

                forward_limit = max(0.0, t_hi - t_seed)
                forward_leaf_ids_local, forward_enter_local, forward_exit_local = launch_branch(
                    leaf_id,
                    seed,
                    direction,
                    forward_limit,
                )
                if forward_leaf_ids_local.size > 0:
                    forward_leaf_ids = forward_leaf_ids_local
                    forward_t_enter = t_seed + forward_enter_local
                    forward_t_exit = t_seed + forward_exit_local
                else:
                    forward_leaf_ids = np.empty(0, dtype=np.int64)
                    forward_t_enter = np.empty(0, dtype=np.float64)
                    forward_t_exit = np.empty(0, dtype=np.float64)
            else:
                t_seed = float(np.dot(seed - origin, direction) / np.dot(direction, direction))

                _, backward_leaf = backward_choice
                backward_limit = max(0.0, t_seed - clip_lo)
                back_leaf_ids_local, back_enter_local, back_exit_local = launch_branch(
                    backward_leaf,
                    seed,
                    -direction,
                    backward_limit,
                )
                if back_leaf_ids_local.size > 0:
                    back_leaf_ids = back_leaf_ids_local[::-1]
                    back_t_enter = t_seed - back_exit_local[::-1]
                    back_t_exit = t_seed - back_enter_local[::-1]
                else:
                    back_leaf_ids = np.empty(0, dtype=np.int64)
                    back_t_enter = np.empty(0, dtype=np.float64)
                    back_t_exit = np.empty(0, dtype=np.float64)

                _, forward_leaf = forward_choice
                forward_limit = max(0.0, t_hi - t_seed)
                forward_leaf_ids_local, forward_enter_local, forward_exit_local = launch_branch(
                    forward_leaf,
                    seed,
                    direction,
                    forward_limit,
                )
                if forward_leaf_ids_local.size > 0:
                    forward_leaf_ids = forward_leaf_ids_local
                    forward_t_enter = t_seed + forward_enter_local
                    forward_t_exit = t_seed + forward_exit_local
                else:
                    forward_leaf_ids = np.empty(0, dtype=np.int64)
                    forward_t_enter = np.empty(0, dtype=np.float64)
                    forward_t_exit = np.empty(0, dtype=np.float64)

            merged_leaf_ids, merged_t_enter, merged_t_exit = self._merge_seed_branches(
                back_leaf_ids,
                back_t_enter,
                back_t_exit,
                forward_leaf_ids,
                forward_t_enter,
                forward_t_exit,
            )

            cell_ids.extend(int(v) for v in merged_leaf_ids)
            t_enter.extend(float(v) for v in merged_t_enter)
            t_exit.extend(float(v) for v in merged_t_exit)

        ray_offsets[-1] = len(cell_ids)
        return RaySegments(
            ray_offsets=ray_offsets,
            cell_ids=np.asarray(cell_ids, dtype=np.int64),
            t_enter=np.asarray(t_enter, dtype=np.float64),
            t_exit=np.asarray(t_exit, dtype=np.float64),
            ray_shape=ray_shape,
        )

    def seed_domain(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        *,
        t_min: float = 0.0,
        t_max: float = np.inf,
    ) -> np.ndarray:
        """Return one in-domain seed point per ray, or `NaN` for misses.

        Returned arrays have the same leading shape as the input ray arrays,
        with one trailing Cartesian axis of length 3.

        For `tree_coord="xyz"`, the octree domain is treated as one axis-aligned
        Cartesian box, and the seed is the midpoint of the clipped domain
        segment.

        For `tree_coord="rpa"`, this first pass assumes one full spherical shell
        with an opaque inner boundary. The seed lies on the front visible shell
        segment only, not a backside interval after crossing the central hole.
        """
        t_lo = float(t_min)
        t_hi = float(t_max)
        if not math.isfinite(t_lo):
            raise ValueError("t_min must be finite.")
        if math.isnan(t_hi):
            raise ValueError("t_max must not be NaN.")
        if t_hi < t_lo:
            raise ValueError("t_max must be greater than or equal to t_min.")

        o_flat, d_flat, shape = self._normalize_rays(origins, directions)
        clip_lo = max(0.0, t_lo)
        n_rays = int(o_flat.shape[0])
        seed_xyz = np.full((n_rays, 3), np.nan, dtype=np.float64)

        if str(self.tree.tree_coord) == "xyz":
            domain_lo, domain_hi = self.tree.domain_bounds(coord="xyz")
            enter = np.full(n_rays, clip_lo, dtype=np.float64)
            exit_ = np.full(n_rays, t_hi, dtype=np.float64)
            hit = np.ones(n_rays, dtype=bool)

            for axis in range(3):
                oa = o_flat[:, axis]
                da = d_flat[:, axis]
                slab_lo = float(domain_lo[axis])
                slab_hi = float(domain_hi[axis])

                parallel = da == 0.0
                hit[parallel & ((oa < slab_lo) | (oa > slab_hi))] = False

                active = hit & (~parallel)
                if not np.any(active):
                    continue
                t0 = (slab_lo - oa[active]) / da[active]
                t1 = (slab_hi - oa[active]) / da[active]
                enter[active] = np.maximum(enter[active], np.minimum(t0, t1))
                exit_[active] = np.minimum(exit_[active], np.maximum(t0, t1))

            hit &= exit_ >= enter
            if np.any(hit):
                seed_t = 0.5 * (enter[hit] + exit_[hit])
                seed_xyz[hit] = o_flat[hit] + seed_t[:, None] * d_flat[hit]
            return seed_xyz.reshape(shape + (3,))

        if str(self.tree.tree_coord) != "rpa":
            raise NotImplementedError(f"Unsupported tree_coord '{self.tree.tree_coord}' for seed_domain.")

        domain_lo, domain_hi = self.tree.domain_bounds(coord="rpa")
        if not np.isclose(float(domain_lo[1]), 0.0, atol=1e-12, rtol=0.0):
            raise NotImplementedError("seed_domain for rpa currently requires polar_min == 0.")
        if not np.isclose(float(domain_hi[1]), math.pi, atol=1e-12, rtol=0.0):
            raise NotImplementedError("seed_domain for rpa currently requires polar_max == pi.")
        if not np.isclose(float(domain_lo[2]), 0.0, atol=1e-12, rtol=0.0):
            raise NotImplementedError("seed_domain for rpa currently requires azimuth_start == 0.")
        if not np.isclose(float(domain_hi[2] - domain_lo[2]), 2.0 * math.pi, atol=1e-12, rtol=0.0):
            raise NotImplementedError("seed_domain for rpa currently requires full 2pi azimuth coverage.")

        r_min = float(domain_lo[0])
        r_max = float(domain_hi[0])
        if r_min < 0.0 or r_max <= r_min:
            raise ValueError(f"Invalid spherical domain radii r_min={r_min}, r_max={r_max}.")

        for i in range(n_rays):
            hit_outer, t_outer0, t_outer1 = self._sphere_interval(o_flat[i], d_flat[i], r_max)
            if not hit_outer:
                continue
            visible_start = max(0.0, float(t_outer0))
            visible_end = float(t_outer1)
            if visible_end < visible_start:
                continue
            if r_min > 0.0:
                hit_inner, t_inner0, t_inner1 = self._sphere_interval(o_flat[i], d_flat[i], r_min)
                if hit_inner and t_inner1 >= visible_start:
                    if t_inner0 > visible_start:
                        visible_end = min(visible_end, float(t_inner0))
                    else:
                        # Rays that start inside the opaque inner sphere, or that
                        # have already crossed into it before the requested clip
                        # window, do not expose a front-side visible interval.
                        continue
            visible_start = max(visible_start, clip_lo)
            visible_end = min(visible_end, t_hi)
            if visible_end < visible_start:
                continue
            tol = 1e-12 * max(r_max, 1.0)
            seed_t = np.nan
            direction_norm_sq = float(np.dot(d_flat[i], d_flat[i]))
            t_closest = -float(np.dot(o_flat[i], d_flat[i])) / direction_norm_sq
            closest_xyz = o_flat[i] + t_closest * d_flat[i]
            closest_radius = float(np.linalg.norm(closest_xyz))
            closest_radial_xy = math.hypot(float(closest_xyz[0]), float(closest_xyz[1]))
            axis_tol = _BOUNDARY_SHIFT_FACTOR * max(closest_radius, 1.0)
            if (
                t_closest >= (visible_start - tol)
                and t_closest <= (visible_end + tol)
                and closest_radius >= (r_min - tol)
                and closest_radius <= (r_max + tol)
                and closest_radial_xy > axis_tol
            ):
                seed_t = float(t_closest)
            r_seed = 0.5 * (r_min + r_max)
            if not np.isfinite(seed_t):
                hit_seed, t_seed0, t_seed1 = self._sphere_interval(o_flat[i], d_flat[i], r_seed)
                if hit_seed:
                    candidates = np.array([t_seed0, t_seed1], dtype=np.float64)
                    mask = (candidates >= (visible_start - tol)) & (candidates <= (visible_end + tol))
                    if np.any(mask):
                        seed_t = float(np.min(candidates[mask]))
            if not np.isfinite(seed_t):
                seed_t = 0.5 * (visible_start + visible_end)
            seed_xyz[i] = o_flat[i] + seed_t * d_flat[i]
        return seed_xyz.reshape(shape + (3,))

    def __str__(self) -> str:
        """Return a compact human-readable tracer summary."""
        return f"OctreeRayTracer(tree_coord={self.tree.tree_coord})"


def render_midpoint_image(
    interpolator,
    origins: np.ndarray,
    directions: np.ndarray,
    segments: RaySegments,
) -> np.ndarray:
    """Render one midpoint-sampled line integral over packed traced segments."""
    from .interpolator import OctreeInterpolator

    if not isinstance(interpolator, OctreeInterpolator):
        raise TypeError("render_midpoint_image requires one OctreeInterpolator.")

    o_flat, d_flat, ray_shape = OctreeRayTracer._normalize_rays(origins, directions)
    if tuple(ray_shape) != tuple(segments.ray_shape):
        raise ValueError("segments.ray_shape must match the ray origin/direction shape.")

    n_rays = int(o_flat.shape[0])
    n_components = int(interpolator._point_values_2d.shape[1])
    accum = np.zeros((n_rays, n_components), dtype=np.float64)
    if segments.cell_ids.size == 0:
        out = accum.reshape(tuple(ray_shape) + interpolator._value_shape_tail)
        if interpolator._value_shape_tail:
            return out
        return out.reshape(tuple(ray_shape))

    counts = np.diff(segments.ray_offsets)
    ray_ids = np.repeat(np.arange(n_rays, dtype=np.int64), counts)
    mid_t = 0.5 * (segments.t_enter + segments.t_exit)
    mid_xyz = o_flat[ray_ids] + mid_t[:, None] * d_flat[ray_ids]
    samples = np.asarray(interpolator(mid_xyz, query_coord="xyz", log_outside_domain=False), dtype=np.float64)
    samples_2d = samples.reshape(samples.shape[0], -1)
    np.add.at(accum, ray_ids, samples_2d * segments.segment_length[:, None])

    out = accum.reshape(tuple(ray_shape) + interpolator._value_shape_tail)
    if interpolator._value_shape_tail:
        return out
    return out.reshape(tuple(ray_shape))
