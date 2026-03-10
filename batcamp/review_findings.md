# Octree Review Findings

Date: 2026-03-09

## Scope note
The Cartesian-path findings assume your guarantee that Cartesian cells are grid-aligned.

## Findings

### 1) Lookup fallback can return false positives (nearest-cell snap on miss)
Both lookup backends can return a nearest cell even when no cell containment test passes.

Why this is a problem:
- API semantics become ambiguous: "lookup" behaves like "nearest" for misses.
- In disconnected domains or meshes with gaps, points outside valid cells can still receive a cell id.
- Downstream interpolation/ray logic can then operate on the wrong cell instead of reporting miss.

Relevant code:
- `/Users/dagfev/Documents/starwinds/starwinds-analysis/starwinds_analysis/octree/cartesian.py:284`
- `/Users/dagfev/Documents/starwinds/starwinds-analysis/starwinds_analysis/octree/cartesian.py:305`
- `/Users/dagfev/Documents/starwinds/starwinds-analysis/starwinds_analysis/octree/spherical.py:180`
- `/Users/dagfev/Documents/starwinds/starwinds-analysis/starwinds_analysis/octree/spherical.py:216`
- `/Users/dagfev/Documents/starwinds/starwinds-analysis/starwinds_analysis/octree/spherical.py:228`

Suggested fix:
- Return `-1` when containment fails across candidates.
- If nearest behavior is desirable, expose it via a separate explicit method/flag.

### 2) Package import has a hard `numba` dependency via `OctreeInterpolator`
Importing `starwinds_analysis.octree` pulls in `OctreeInterpolator`, which imports `numba` at import time.

Why this is a problem:
- Users cannot import basic octree functionality (`Octree`, builder, persistence) without `numba` installed.
- This couples optional performance/interpolation functionality to core data structures.

Relevant code:
- `/Users/dagfev/Documents/starwinds/starwinds-analysis/starwinds_analysis/octree/__init__.py:19`
- `/Users/dagfev/Documents/starwinds/starwinds-analysis/starwinds_analysis/octree/interpolator.py:13`

Suggested fix:
- Make `OctreeInterpolator` import lazy/optional in `__init__.py`.
- Or guard `numba` import and provide a clear runtime error only when interpolation kernels are used.

## Not flagged due to project assumption
Given your clarification that Cartesian cells are grid-aligned, I am not flagging:
- AABB containment in Cartesian lookup.
- Cartesian trilinear mapping based on axis-aligned local coordinates.
