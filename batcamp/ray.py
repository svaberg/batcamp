#!/usr/bin/env python3
"""Ray seeding and seed-neighborhood traversal helpers for octrees."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .octree import Octree

_HEX_FACE_TRIANGLES = np.array(
    [
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 2, 6],
        [3, 6, 7],
        [0, 3, 7],
        [0, 7, 4],
        [1, 2, 6],
        [1, 6, 5],
    ],
    dtype=np.int64,
)

__all__ = ["OctreeRayTracer", "RaySegments"]


@dataclass(frozen=True)
class RaySegments:
    """Packed per-ray segment arrays in CSR-like layout."""

    ray_offsets: np.ndarray
    cell_ids: np.ndarray
    t_enter: np.ndarray
    t_exit: np.ndarray

    def __post_init__(self) -> None:
        offsets = np.asarray(self.ray_offsets, dtype=np.int64)
        cell_ids = np.asarray(self.cell_ids, dtype=np.int64)
        t_enter = np.asarray(self.t_enter, dtype=np.float64)
        t_exit = np.asarray(self.t_exit, dtype=np.float64)

        if offsets.ndim != 1:
            raise ValueError("ray_offsets must have shape (n_rays + 1,).")
        if offsets.size == 0:
            raise ValueError("ray_offsets must contain at least one element.")
        if offsets[0] != 0:
            raise ValueError("ray_offsets must start at 0.")
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("ray_offsets must be nondecreasing.")
        if cell_ids.ndim != 1 or t_enter.ndim != 1 or t_exit.ndim != 1:
            raise ValueError("cell_ids, t_enter, and t_exit must be 1D arrays.")
        if not (cell_ids.shape == t_enter.shape == t_exit.shape):
            raise ValueError("cell_ids, t_enter, and t_exit must have matching shapes.")
        if offsets[-1] != cell_ids.size:
            raise ValueError("ray_offsets[-1] must equal the packed segment count.")
        if np.any(t_exit < t_enter):
            raise ValueError("Each segment must satisfy t_exit >= t_enter.")

        object.__setattr__(self, "ray_offsets", offsets)
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "t_enter", t_enter)
        object.__setattr__(self, "t_exit", t_exit)


class OctreeRayTracer:
    """Trace rays against one octree geometry.

    The current implementation supports:
    - `seed_domain(...)` for one visible in-domain seed point per ray
    - `trace_seed_segments(...)` for the first traced cell segment(s) around that seed
    """

    def __init__(self, tree: Octree) -> None:
        """Bind one tracer to one built octree."""
        if not isinstance(tree, Octree):
            raise TypeError("OctreeRayTracer requires a built Octree as its first argument.")
        self.tree = tree
        point_span = np.ptp(self.tree.points, axis=0)
        self._probe_scale = max(float(np.max(point_span)), 1.0)

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

    @staticmethod
    def _triangle_hit_distance(
        origin_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        triangle_xyz: np.ndarray,
        *,
        eps: float,
    ) -> float:
        """Return the positive ray distance to one triangle, or `NaN` for no hit."""
        v0 = triangle_xyz[0]
        v1 = triangle_xyz[1]
        v2 = triangle_xyz[2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        pvec = np.cross(direction_xyz, edge2)
        det = float(np.dot(edge1, pvec))
        if abs(det) <= eps:
            return np.nan
        inv_det = 1.0 / det
        tvec = origin_xyz - v0
        u = float(np.dot(tvec, pvec)) * inv_det
        if u < -eps or u > (1.0 + eps):
            return np.nan
        qvec = np.cross(tvec, edge1)
        v = float(np.dot(direction_xyz, qvec)) * inv_det
        if v < -eps or (u + v) > (1.0 + eps):
            return np.nan
        distance = float(np.dot(edge2, qvec)) * inv_det
        if distance <= eps:
            return np.nan
        return distance

    def _cell_interval_on_ray(
        self,
        cell_id: int,
        origin_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        probe_t: float,
        *,
        contain_tol: float,
    ) -> tuple[float, float]:
        """Return one exact-in-xyz hexahedral interval that contains `probe_t`."""
        corner_ids = self.tree.corners[int(cell_id)]
        vertices = self.tree.points[corner_ids]
        eps = 1.0e-12 * max(self._probe_scale, float(np.linalg.norm(origin_xyz)), 1.0)
        hit_t: list[float] = []
        for tri_ord in range(int(_HEX_FACE_TRIANGLES.shape[0])):
            triangle = vertices[_HEX_FACE_TRIANGLES[tri_ord]]
            hit_forward = self._triangle_hit_distance(origin_xyz, direction_xyz, triangle, eps=eps)
            if np.isfinite(hit_forward):
                hit_t.append(float(hit_forward))
            hit_backward = self._triangle_hit_distance(origin_xyz, -direction_xyz, triangle, eps=eps)
            if np.isfinite(hit_backward):
                hit_t.append(float(-hit_backward))
        if len(hit_t) < 2:
            return np.nan, np.nan
        hit_t = sorted(hit_t)
        unique_hit_t: list[float] = []
        merge_tol = 1.0e-10 * max(self._probe_scale, abs(probe_t), 1.0)
        for t_value in hit_t:
            if unique_hit_t and abs(t_value - unique_hit_t[-1]) <= merge_tol:
                continue
            unique_hit_t.append(float(t_value))
        if len(unique_hit_t) < 2:
            return np.nan, np.nan
        for idx in range(len(unique_hit_t) - 1):
            t_enter = unique_hit_t[idx]
            t_exit = unique_hit_t[idx + 1]
            if (t_enter - contain_tol) <= probe_t <= (t_exit + contain_tol):
                return float(t_enter), float(t_exit)
        return np.nan, np.nan

    def _trace_seed_side(
        self,
        seed_xyz: np.ndarray,
        origin_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        *,
        sign: float,
    ) -> tuple[int, float, float]:
        """Return one first traced segment on one side of a seed, or `(-1, nan, nan)`."""
        direction_norm = float(np.linalg.norm(direction_xyz))
        if direction_norm <= 0.0:
            return -1, np.nan, np.nan
        direction_norm_sq = float(np.dot(direction_xyz, direction_xyz))
        direction_unit = direction_xyz / direction_norm
        for exponent in range(-12, -7):
            probe_distance = (10.0 ** exponent) * self._probe_scale
            probe_xyz = seed_xyz + float(sign) * probe_distance * direction_unit
            cell_id = int(self.tree.lookup_points(probe_xyz, coord="xyz").reshape(-1)[0])
            if cell_id < 0:
                continue
            probe_t = float(np.dot(probe_xyz - origin_xyz, direction_xyz)) / direction_norm_sq
            contain_tol = 0.25 * probe_distance / direction_norm
            t_enter, t_exit = self._cell_interval_on_ray(
                cell_id,
                origin_xyz,
                direction_xyz,
                probe_t,
                contain_tol=contain_tol,
            )
            if np.isfinite(t_enter) and np.isfinite(t_exit):
                return cell_id, t_enter, t_exit
        return -1, np.nan, np.nan

    def trace_seed_segments(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        *,
        seed_xyz: np.ndarray | None = None,
        t_min: float = 0.0,
        t_max: float = np.inf,
    ) -> RaySegments:
        """Return the first traced segment(s) on each side of the visible domain seed.

        This is the first real traced seed neighborhood:
        - find one visible in-domain seed point per ray
        - probe slightly along `+direction` and `-direction`
        - resolve the first leaf cell on each side of the seed
        - compute exact per-cell `t_enter` / `t_exit` in physical `xyz`

        Rays can contribute:
        - 0 segments for misses
        - 1 segment when both probes land in the same seed cell
        - 2 segments when the seed lies on a cell boundary and the two sides differ
        """
        o_flat, d_flat, shape = self._normalize_rays(origins, directions)
        n_rays = int(o_flat.shape[0])
        if seed_xyz is None:
            seed_flat = self.seed_domain(origins, directions, t_min=t_min, t_max=t_max).reshape(-1, 3)
        else:
            seed_arr = np.asarray(seed_xyz, dtype=np.float64)
            if seed_arr.shape != shape + (3,):
                raise ValueError(
                    "seed_xyz must have the same leading shape as the input rays with one trailing axis of length 3."
                )
            if not np.all(np.isfinite(seed_arr) | np.isnan(seed_arr)):
                raise ValueError("seed_xyz must contain only finite values or NaN.")
            seed_flat = seed_arr.reshape(-1, 3)

        ray_offsets = np.zeros(n_rays + 1, dtype=np.int64)
        packed_cell_ids: list[int] = []
        packed_t_enter: list[float] = []
        packed_t_exit: list[float] = []

        t_lo = float(t_min)
        t_hi = float(t_max)
        if t_hi < t_lo:
            raise ValueError("t_max must be greater than or equal to t_min.")

        for i in range(n_rays):
            seed_i = seed_flat[i]
            origin_i = o_flat[i]
            direction_i = d_flat[i]
            ray_segments: list[tuple[float, float, int]] = []
            if np.all(np.isfinite(seed_i)):
                for sign in (-1.0, 1.0):
                    cell_id, t_enter, t_exit = self._trace_seed_side(seed_i, origin_i, direction_i, sign=sign)
                    if cell_id < 0:
                        continue
                    t_enter = max(t_enter, t_lo)
                    t_exit = min(t_exit, t_hi)
                    if t_exit < t_enter:
                        continue
                    ray_segments.append((t_enter, t_exit, int(cell_id)))

            if len(ray_segments) == 2 and ray_segments[0][2] == ray_segments[1][2]:
                merged_enter = min(ray_segments[0][0], ray_segments[1][0])
                merged_exit = max(ray_segments[0][1], ray_segments[1][1])
                ray_segments = [(merged_enter, merged_exit, ray_segments[0][2])]

            ray_segments.sort(key=lambda item: (item[0], item[1], item[2]))
            if len(ray_segments) == 2:
                same_interval = (
                    ray_segments[0][2] == ray_segments[1][2]
                    and math.isclose(ray_segments[0][0], ray_segments[1][0], rel_tol=0.0, abs_tol=1.0e-10)
                    and math.isclose(ray_segments[0][1], ray_segments[1][1], rel_tol=0.0, abs_tol=1.0e-10)
                )
                if same_interval:
                    ray_segments = [ray_segments[0]]

            for t_enter, t_exit, cell_id in ray_segments:
                packed_cell_ids.append(int(cell_id))
                packed_t_enter.append(float(t_enter))
                packed_t_exit.append(float(t_exit))
            ray_offsets[i + 1] = len(packed_cell_ids)

        return RaySegments(
            ray_offsets=ray_offsets,
            cell_ids=np.asarray(packed_cell_ids, dtype=np.int64),
            t_enter=np.asarray(packed_t_enter, dtype=np.float64),
            t_exit=np.asarray(packed_t_exit, dtype=np.float64),
        )

    def __str__(self) -> str:
        """Return a compact human-readable tracer summary."""
        return f"OctreeRayTracer(tree_coord={self.tree.tree_coord})"
