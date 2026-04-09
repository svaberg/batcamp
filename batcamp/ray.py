#!/usr/bin/env python3
"""Ray-domain seeding helpers for octrees."""

from __future__ import annotations

import math

import numpy as np

from .octree import Octree

__all__ = ["OctreeRayTracer"]


class OctreeRayTracer:
    """Trace rays against one octree geometry.

    The current implementation only seeds rays inside the global octree domain.
    """

    def __init__(self, tree: Octree) -> None:
        """Bind one tracer to one built octree."""
        if not isinstance(tree, Octree):
            raise TypeError("OctreeRayTracer requires a built Octree as its first argument.")
        self.tree = tree

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

    def __str__(self) -> str:
        """Return a compact human-readable tracer summary."""
        return f"OctreeRayTracer(tree_coord={self.tree.tree_coord})"
