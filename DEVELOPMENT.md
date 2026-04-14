# Development Notes

## Current direction
The current development order is:

1. get the octree fully correct and well-specified
2. lock the octree invariants that downstream code may rely on
3. design a minimal ray and imaging API against those invariants
4. implement the new traversal and image-formation code

The immediate goal is not to restore the old ray code. The goal is to make the octree trustworthy enough that a new octree-aware ray traversal layer can be built on top of it without carrying temporary assumptions or cleanup debt.

## Why this matters
Ray traversal is unusually sensitive to geometric ambiguity. If the octree contract is vague, the traversal layer will absorb that vagueness as:

- ad hoc boundary handling
- special cases for misses near faces, edges, and corners
- duplicated coordinate logic
- inconsistent behavior between lookup, interpolation, and traversal

The octree should therefore expose a small number of exact, stable rules that the future tracer is allowed to assume.

## Ray traversal geometry target
The future ray tracer should be geometry-first.

Its primary output should be exact per-cell ray intervals:

- `t_enter`
- `t_exit`

for each crossed cell, in physical `xyz` ray coordinates.

In other words, traversal should determine where the parametric ray

- `x(t) = origin + t * direction`

enters and exits each supported cell geometry. Those intervals should not come from:

- ray marching
- midpoint approximations
- adaptive stepping
- post hoc estimates of path length from sample placement
- native-coordinate path lengths that are later converted to `xyz`

Downstream sampling and integration may choose midpoint, quadrature, or other rules, but those sit on top of exact segment geometry rather than replacing it.

## Explicit non-goals for the first ray pass
The first traversal implementation should work directly on the adaptive octree geometry.

It should not rely on proxy-volume tricks such as:

- resampling the entire octree onto a finest-level uniform grid
- materializing a whole-domain maximum-resolution Cartesian volume
- tracing through a dense voxelized surrogate instead of the adaptive cells
- approximating adaptive traversal by first expanding all cells to the finest refinement level

Those approaches are out of scope for the intended design because they:

- discard the main structural benefit of the octree
- inflate memory and preprocessing cost
- blur the geometric contract the tracer is supposed to honor
- make it easier for path-length calculations to drift away from exact per-cell geometry

For this first pass, the tracer should intersect the supported cell geometry directly and produce exact per-cell intervals from that geometry.

### Cartesian-tree traversal geometry
For `tree_coord="xyz"`, the intended traversal geometry is:

- axis-aligned Cartesian leaf cells

For these cells, "exact interval length" means the ray enters and exits the cell's actual Cartesian box, and the resulting `t_enter` and `t_exit` are exact for that supported geometry model.

### Spherical-tree traversal geometry
For `tree_coord="rpa"`, the ray tracer is still conceptually and numerically in physical `xyz`.

That means:

- rays are represented in `xyz`
- interval endpoints are `xyz` ray parameters
- segment lengths are defined in `xyz`, not in native `rpa` coordinates

The intended traversal geometry is not "native-coordinate interval length along the ray". Instead, each spherical leaf cell should be treated for traversal as its corresponding squashed hexahedral cell in Cartesian space.

For future work, the intended contract is:

- `t_enter` and `t_exit` are based on intersection with the cell's supported squashed-hexahedron geometry in `xyz`
- they are not based on separately intersecting `r`, `polar`, and `azimuth` intervals and then treating the result as the physical path length

This distinction is important. Even when the tree is built and indexed in `rpa`, the traversal result should be a physical `xyz` segment through the represented cell geometry.

For the first astronomy-oriented shell-tracing pass, the inner radial boundary should be treated as opaque:

- the visible interval is the front shell segment only
- when a forward ray reaches `r = rmin`, the ray is finished
- the tracer does not continue through the central hole to a backside shell interval

It is worth being explicit about why this is the preferred direction. Intersections against native `rpa` boundary surfaces are not impossible in principle:

- `r = const` gives spherical surfaces
- `polar = const` gives conical surfaces
- `azimuth = const` gives half-planes through the axis

But turning those into a robust per-cell ray interval algorithm is still materially nontrivial in practice because it requires consistent handling of:

- periodic azimuth wrapping
- pole-adjacent behavior
- tangent and grazing hits
- face, edge, and corner coincidences
- interval ordering along a Cartesian ray
- agreement with the octree's own containment and boundary rules

So the issue is not that native-`rpa` intersections are mathematically forbidden. The issue is that they are substantially harder to make exact, robust, and contract-consistent, and they are not the primary geometry target if the tracer's truth is defined in physical `xyz`.

## Current public octree surface
The current octree-facing public entrypoints are:

- `Octree(...)`
- `Octree.from_ds(...)`
- `Octree.lookup_points(points, coord=...)`
- `Octree.domain_bounds(coord=...)`
- `OctreeInterpolator(tree, values)`

The future traversal layer should be designed to rely on stable octree behavior, not on hidden implementation details beyond what is explicitly documented here.

## Octree invariants required before ray work

### 1. Tree coordinate system is explicit and stable
Every `Octree` has one fixed `tree_coord`.

Supported meanings:

- `tree_coord="xyz"` means the tree geometry is interpreted in Cartesian coordinates
- `tree_coord="rpa"` means the tree geometry is interpreted in spherical `(r, polar, azimuth)` coordinates

Required invariant:

- all octree geometry state is internally consistent with `tree_coord`
- coordinate conversion is explicit, never inferred implicitly by downstream ray code

For future ray work this means:

- rays themselves should still be described in physical `xyz`
- any conversion into tree-local coordinates belongs to octree-owned geometry logic, not to public imaging code

### 2. Leaf ownership is unambiguous for interior points
For any point strictly inside the represented domain and not lying on a cell boundary, `lookup_points` must resolve exactly one containing leaf cell.

Required invariant:

- interior points do not map to multiple leaves
- interior points do not miss because of numerical ambiguity
- the returned cell id is a leaf id, not an internal runtime node id

This is the minimum correctness condition for any later ray segment integration.

### 3. Boundary behavior is specified, not accidental
Boundary semantics matter for traversal even when point interpolation appears to work.

Required invariant:

- domain boundary behavior is documented and deterministic
- cell-face boundary behavior is documented and deterministic
- edge and corner behavior is documented and deterministic

At minimum, the project needs one clear rule for each of these cases:

- query exactly on a shared face between two leaf cells
- query exactly on a shared edge
- query exactly on a shared corner
- query exactly on the outer domain boundary

The future tracer must not contain its own independent tie-breaking policy. It should be able to rely on the octree contract.

### 4. Domain bounds are exact enough to be a clipping primitive
`Octree.domain_bounds(...)` should be a reliable representation of the global represented domain in the tree's own coordinate system.

Required invariant:

- the returned bounds are the intended traversal domain, not just a loose envelope
- the same domain interpretation is used by lookup, interpolation, and future traversal
- points clearly outside the domain miss
- points clearly inside the domain are not rejected by domain clipping

For `tree_coord="xyz"` this means the domain is an axis-aligned Cartesian box.

For `tree_coord="rpa"` this means the domain description must be good enough for:

- deciding whether a ray can intersect the represented spherical region
- clipping to the radial extent
- treating azimuth periodicity consistently

### 5. Cell bounds have a stable mathematical meaning
The packed `cell_bounds` array is central enough that its contract must be explicit.

Current storage shape:

- `(n_cells, 3, 2)`
- slot `[..., axis, 0]` is axis start
- slot `[..., axis, 1]` is axis width

Required invariant:

- every runtime cell has one exact local box in the tree coordinate system
- start and width semantics are the same everywhere in the codebase
- width is non-negative
- periodic-axis handling is explicit, not guessed from context

For `tree_coord="xyz"` the important contract is:

- the ray code may treat leaf cells as axis-aligned Cartesian boxes if and only if this is part of the supported geometry model

If skewed or non-axis-aligned Cartesian cells are out of scope, that should remain an explicit limitation rather than an accidental approximation.

For `tree_coord="rpa"` the important contract is:

- radial, polar, and azimuth intervals must be meaningful enough to support exact cell entry and exit calculations in spherical-tree traversal logic

However, if the future ray tracer defines its intervals in physical `xyz` against squashed hexahedral cell geometry, then native `rpa` bounds alone are not the whole traversal geometry contract. In that case the octree must expose or support the corresponding `xyz` cell geometry clearly enough for exact `t_enter` and `t_exit` calculations.

### 6. Runtime topology is correct and complete
The rebuilt runtime tree arrays must represent one valid octree topology.

Relevant arrays include:

- `cell_depth`
- `cell_ijk`
- `cell_child`
- `cell_parent`
- `root_cell_ids`

Required invariant:

- each non-root runtime cell has exactly one parent
- child links and parent links agree
- root cells have no parent
- leaves are exactly the cells with no children
- internal runtime cells do not overlap illegally
- the runtime topology is sufficient to navigate containment and adjacency-related logic later

The tracer will not necessarily need all topology arrays directly, but it must be able to trust that the tree structure is exact.

### 7. Neighbor refinement is locally balanced
The octree should satisfy the intended local refinement rule for neighboring cells.

Required invariant:

- neighboring cells differ by at most one refinement level

This is the usual 2:1 balance condition. It matters for traversal because it constrains:

- which boundary transitions are possible
- how many candidate neighbor configurations exist at a face, edge, or corner
- whether traversal logic can rely on local bounded refinement jumps

If this condition is part of the supported octree model, it should be treated as a real invariant and tested as such rather than left implicit.

### 8. Leaf ids remain stable data-bearing ids
Downstream code needs a stable relationship between leaf cell ids and leaf data rows.

Required invariant:

- the leaf ids returned by lookup correspond to the data-carrying cells
- corner ids for a leaf correspond to the same leaf that lookup returns
- interpolation, resampling, and later traversal all agree on the meaning of a returned leaf id

This matters because image formation will eventually combine:

- ray segment geometry from traversal
- corner-based interpolation or cellwise accumulation
- field arrays indexed by leaf or point ids

If the leaf-id contract drifts, every downstream layer becomes fragile.

### 9. Cross-coordinate lookup is consistent where supported
For spherical trees, point queries expressed in `xyz` and `rpa` should resolve to the same leaf whenever both coordinates describe the same physical point away from boundaries.

Required invariant:

- coordinate conversion does not change cell ownership for interior points
- periodic azimuth handling does not create lookup disagreement
- lookup behavior near the poles is deliberate and tested

This consistency is especially important because the future imaging API should remain physically `xyz`-oriented even when the underlying tree is spherical.

### 10. Domain occupancy is described in native coordinates
The represented domain should be understood first in the tree's native coordinates.

Required invariant:

- there are no holes inside the represented domain in native coordinates
- for `tree_coord="xyz"`, this means there are no internal voids in the supported Cartesian domain model
- for `tree_coord="rpa"`, this means the native spherical domain is continuous in its own coordinates even if its image in `xyz` has an excluded inner ball

Important consequence:

- an `rpa` tree may have `rmin > 0`
- in that case, the native spherical domain still has no hole according to its own contract
- but in physical `xyz` space there is a real central excluded region `r < rmin`

This distinction should remain explicit throughout lookup, clipping, traversal, and imaging code. The central excluded region for `rpa` trees is not an accidental missing-data hole. It is part of the represented physical domain.

### 11. Missing regions are a supported part of the model
The tree may or may not represent every point in its global bounding box with occupied leaf data, depending on which coordinate system and domain description are being used.

Required invariant:

- misses are real information, not undefined behavior
- outside-domain misses and inside-envelope-but-unrepresented misses are handled deliberately
- the project distinguishes between true unsupported gaps and intentionally excluded regions implied by the native domain definition
- the project should know whether traversal is allowed to pass through empty regions, stop at them, or treat them as gaps between represented cells

This affects image formation directly. A synthetic image pipeline needs to know whether a missed segment means:

- no data contribution
- invalid image contribution
- traversal should terminate

That policy can live above the octree, but the octree behavior must make the distinction possible.

### 12. Lookup, interpolation, and traversal must share one geometry model
The future tracer should not introduce a second geometry model.

Required invariant:

- point containment, interpolation fractions, and ray entry-exit computations all refer to the same cell geometry
- tolerances used in one layer do not contradict another layer's notion of inside versus outside
- the same coordinate conventions and periodic rules are used everywhere

If a future traversal method needs geometry that the current octree does not expose cleanly, the octree should be extended first rather than patched around downstream.

## Readiness checklist for new ray and imaging work
Before adding the new traversal subsystem, the octree should be in a state where the following are true:

- `lookup_points` has stable documented boundary behavior
- `domain_bounds` is trusted as the represented traversal domain
- `cell_bounds` semantics are explicitly documented and tested
- local neighbor refinement balance is explicit and tested
- Cartesian-tree geometry limitations are explicit
- spherical periodic-axis behavior is explicit
- native-domain occupancy rules are explicit, including the `rmin > 0` spherical case
- the traversal geometry model is explicit for both `xyz` and `rpa` trees
- exact segment intervals are defined in physical `xyz`
- the first traversal path does not depend on densifying the tree to maximum resolution
- cross-coordinate lookup consistency is tested for interior points
- the meaning of misses is clear enough for downstream integration
- leaf ids are a stable contract for downstream data access

## What the future ray layer should be allowed to assume
Once the above is true, the new ray layer may assume:

- it can trace in physical `xyz`
- the octree owns coordinate conversion details
- the octree exposes one consistent notion of domain membership
- neighboring cells differ by at most one refinement level
- `t_enter` and `t_exit` are exact for the supported cell geometry model
- for `rpa` trees, those exact intervals come from the supported squashed-hexahedron geometry in `xyz`
- traversal operates directly on adaptive cells rather than on a densified whole-domain proxy
- leaf ids returned by traversal can be used for downstream sampling and integration
- exact segment ordering can be built from exact cell geometry rather than from heuristic stepping

## What the future ray layer should not assume
The new ray layer should not assume, unless the octree explicitly promises it:

- arbitrary non-axis-aligned Cartesian cell geometry
- fallback behavior for ambiguous boundary hits
- hidden coercions between `xyz` and `rpa`
- that every global bounding-box point has represented data
- that old ray API shapes or names should be preserved

## Immediate development implication
The next ray milestone should not be "restore ray code".

The next ray-adjacent milestone should be:

- write down and test the octree invariants that traversal depends on

Only after that should the project define the minimal public surface for the new synthetic imaging code.
