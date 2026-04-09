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


class OctreeRayTracer:
    """Trace rays against one octree geometry.

    The current implementation only seeds rays inside the global octree domain.
    """

    def __init__(self, tree: Octree) -> None:
        """Bind one tracer to one built octree."""
        if not isinstance(tree, Octree):
            raise TypeError("OctreeRayTracer requires a built Octree as its first argument.")
        self.tree = tree
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

    def _xyz_cell_interval(self, leaf_id: int, origin_xyz: np.ndarray, direction_xyz: np.ndarray) -> tuple[float, float]:
        """Return one exact Cartesian box interval for one leaf and one line."""
        bounds = self.tree.cell_bounds[int(leaf_id)]
        t_enter = -np.inf
        t_exit = np.inf
        for axis in range(3):
            start = float(bounds[axis, 0])
            stop = start + float(bounds[axis, 1])
            origin_axis = float(origin_xyz[axis])
            direction_axis = float(direction_xyz[axis])
            if direction_axis == 0.0:
                if origin_axis < start or origin_axis > stop:
                    return np.inf, -np.inf
                continue
            t0 = (start - origin_axis) / direction_axis
            t1 = (stop - origin_axis) / direction_axis
            axis_enter = min(t0, t1)
            axis_exit = max(t0, t1)
            t_enter = max(t_enter, axis_enter)
            t_exit = min(t_exit, axis_exit)
        return float(t_enter), float(t_exit)

    def _xyz_exit_face(self, leaf_id: int, origin_xyz: np.ndarray, direction_xyz: np.ndarray, t_exit: float) -> int:
        """Return the unique exit face for one non-degenerate Cartesian cell crossing."""
        bounds = self.tree.cell_bounds[int(leaf_id)]
        face_hits: list[int] = []
        tol = _EXIT_TOL * max(1.0, abs(float(t_exit)))
        for axis in range(3):
            direction_axis = float(direction_xyz[axis])
            if direction_axis == 0.0:
                continue
            start = float(bounds[axis, 0])
            stop = start + float(bounds[axis, 1])
            if direction_axis > 0.0:
                face_t = (stop - float(origin_xyz[axis])) / direction_axis
                face_id = 2 * axis + 1
            else:
                face_t = (start - float(origin_xyz[axis])) / direction_axis
                face_id = 2 * axis
            if abs(face_t - float(t_exit)) <= tol:
                face_hits.append(int(face_id))
        if len(face_hits) != 1:
            raise ValueError("Degenerate Cartesian cell exit across an edge or corner is not supported yet.")
        return int(face_hits[0])

    def _xyz_exit_subface(self, leaf_id: int, exit_xyz: np.ndarray, face_id: int) -> int:
        """Return the crossed face quadrant for one non-degenerate Cartesian face exit."""
        bounds = self.tree.cell_bounds[int(leaf_id)]
        tangential_axes = _FACE_TANGENTIAL_AXES[int(face_id)]
        first_axis = int(tangential_axes[0])
        second_axis = int(tangential_axes[1])

        bits = [0, 0]
        for bit_pos, axis in enumerate((first_axis, second_axis)):
            start = float(bounds[axis, 0])
            width = float(bounds[axis, 1])
            local = (float(exit_xyz[axis]) - start) / width
            if (
                abs(local) <= _SUBFACE_TOL
                or abs(local - 0.5) <= _SUBFACE_TOL
                or abs(local - 1.0) <= _SUBFACE_TOL
            ):
                raise ValueError("Degenerate Cartesian face exit on a subface boundary is not supported yet.")
            if local < 0.0 or local > 1.0:
                raise ValueError("Cartesian exit point fell outside the crossed face bounds.")
            bits[bit_pos] = 0 if local < 0.5 else 1
        return 2 * bits[0] + bits[1]

    def _trace_one_xyz_ray(
        self,
        start_leaf_id: int,
        origin_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        *,
        t_min: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Walk one Cartesian ray across neighboring leaves and return exact cell intervals.

        This is an internal first-pass walker used to validate the topology cache.
        It assumes a non-degenerate ray that crosses one face at a time.
        """
        if str(self.tree.tree_coord) != "xyz":
            raise NotImplementedError("_trace_one_xyz_ray currently supports only tree_coord='xyz'.")
        o_flat, d_flat, shape = self._normalize_rays(origin_xyz, direction_xyz)
        if shape != (1,):
            raise ValueError("_trace_one_xyz_ray requires one origin and one direction vector.")

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
            t_enter, t_exit = self._xyz_cell_interval(current_leaf, origin, direction)
            if not np.isfinite(t_exit) or t_exit < max(float(t_min), t_enter):
                raise ValueError("The requested start leaf does not intersect the forward ray.")
            segment_enter = max(float(t_min), current_t, float(t_enter))
            if t_exit <= segment_enter:
                raise ValueError("Degenerate zero-length Cartesian cell interval is not supported yet.")

            exit_face = self._xyz_exit_face(current_leaf, origin, direction, t_exit)
            exit_xyz = origin + t_exit * direction
            subface_id = self._xyz_exit_subface(current_leaf, exit_xyz, exit_face)

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
