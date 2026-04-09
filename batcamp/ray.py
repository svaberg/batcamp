#!/usr/bin/env python3
"""Ray-domain seeding helpers for octrees."""

from __future__ import annotations

import math

import numpy as np

from .octree import Octree

__all__ = ["OctreeRayTracer"]

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
        normal = np.cross(face_xyz[1] - face_xyz[0], face_xyz[2] - face_xyz[1])
        if np.dot(normal, np.mean(face_xyz, axis=0) - cell_center_xyz) < 0.0:
            normal = -normal
        return normal

    @staticmethod
    def _point_inside_face(face_xyz: np.ndarray, normal_xyz: np.ndarray, point_xyz: np.ndarray, tol: float) -> bool:
        """Return whether one point lies inside one convex planar face quad."""
        signs = np.empty(4, dtype=np.float64)
        for edge_id in range(4):
            p0 = face_xyz[edge_id]
            p1 = face_xyz[(edge_id + 1) % 4]
            signs[edge_id] = float(np.dot(np.cross(p1 - p0, point_xyz - p0), normal_xyz))
        return bool(np.all(signs >= -tol) or np.all(signs <= tol))

    def _point_inside_cell(self, leaf_id: int, point_xyz: np.ndarray, tol: float) -> bool:
        """Return whether one point lies inside one convex planar-face cell."""
        cell_xyz, cell_center = self._cell_faces(leaf_id)
        for face_id in range(6):
            face_xyz = cell_xyz[self._face_corners[face_id]]
            normal_xyz = self._face_normal(face_xyz, cell_center)
            if float(np.dot(normal_xyz, point_xyz - face_xyz[0])) > tol:
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
            low_sign = float(np.dot(np.cross(split1 - split0, low_ref - split0), normal_xyz))
            high_sign = float(np.dot(np.cross(split1 - split0, high_ref - split0), normal_xyz))
            hit_sign = float(np.dot(np.cross(split1 - split0, exit_xyz - split0), normal_xyz))
            if abs(hit_sign) <= _SUBFACE_TOL:
                raise ValueError("Degenerate face exit on a subface boundary is not supported yet.")
            if low_sign == 0.0 or high_sign == 0.0 or low_sign * high_sign >= 0.0:
                raise ValueError("Failed to construct a valid face subface split.")
            return 0 if hit_sign * low_sign > 0.0 else 1

        first_bit = classify(first_split0, first_split1, c00, c10)
        second_bit = classify(second_split0, second_split1, c00, c01)
        return 2 * first_bit + second_bit

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
        inside_now = self._point_inside_cell(int(leaf_id), origin_xyz + float(current_t) * direction_xyz, point_tol)

        face_hits: list[tuple[float, int, np.ndarray, np.ndarray]] = []
        for face_id in range(6):
            face_xyz = cell_xyz[self._face_corners[face_id]]
            normal_xyz = self._face_normal(face_xyz, cell_center)
            denom = float(np.dot(normal_xyz, direction_xyz))
            if abs(denom) <= _EXIT_TOL:
                continue
            t_hit = float(np.dot(normal_xyz, face_xyz[0] - origin_xyz) / denom)
            hit_xyz = origin_xyz + t_hit * direction_xyz
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
        if len(degenerate_hits) != 1:
            raise ValueError("Degenerate cell exit across an edge or corner is not supported yet.")

        subface_id = self._face_exit_subface(leaf_id, exit_xyz, exit_face, exit_normal)
        return float(segment_enter), float(exit_t), int(exit_face), int(subface_id)

    def _trace_one_ray(
        self,
        start_leaf_id: int,
        origin_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        *,
        t_min: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Walk one ray across neighboring planar-face hex cells and return exact cell intervals.

        This is an internal first-pass walker used to validate the topology cache.
        It assumes a non-degenerate ray that crosses one face at a time.
        """
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
        current_t = float(t_min)
        max_steps = int(np.count_nonzero(self._leaf_valid)) + 1

        for _ in range(max_steps):
            segment_enter, t_exit, exit_face, subface_id = self._cell_segment(
                current_leaf,
                origin,
                direction,
                current_t=float(current_t),
                t_min=float(t_min),
            )

            leaf_ids.append(int(current_leaf))
            t_enter_list.append(float(segment_enter))
            t_exit_list.append(float(t_exit))

            next_leaf = self._resolve_transition(current_leaf, exit_face, subface_id)
            if next_leaf < 0:
                break
            current_leaf = int(next_leaf)
            current_t = float(t_exit)

        return (
            np.asarray(leaf_ids, dtype=np.int64),
            np.asarray(t_enter_list, dtype=np.float64),
            np.asarray(t_exit_list, dtype=np.float64),
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
            a = float(np.dot(d_flat[i], d_flat[i]))
            t_closest = -float(np.dot(o_flat[i], d_flat[i])) / a
            p_closest = o_flat[i] + t_closest * d_flat[i]
            rho_closest = float(np.linalg.norm(p_closest))
            tol = 1e-12 * max(r_max, 1.0)
            seed_t = np.nan
            if (
                visible_start - tol <= t_closest <= visible_end + tol
                and (r_min - tol) <= rho_closest <= (r_max + tol)
            ):
                seed_t = t_closest
            else:
                r_seed = 0.5 * (r_min + r_max)
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
