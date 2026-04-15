# Development Notes

## Current direction
The current development order is:

1. get the octree fully correct and well-specified
2. lock the octree invariants that downstream code may rely on
3. design a minimal ray and imaging API against those invariants
4. implement the new traversal and image-formation code

The immediate goal is not to restore old ray behavior. The goal is to keep the
octree trustworthy enough that the octree-aware ray traversal layer can be
extended without carrying temporary assumptions or cleanup debt.

## Why this matters
Ray traversal is unusually sensitive to geometric ambiguity. If the octree contract is vague, the traversal layer will absorb that vagueness as:

- ad hoc boundary handling
- special cases for misses near faces, edges, and corners
- duplicated coordinate logic
- inconsistent behavior between lookup, interpolation, and traversal

The octree should therefore expose a small number of exact, stable rules that the tracer is allowed to assume.

## Ray traversal geometry target
The ray tracer should be geometry-first.

Its primary output should be exact per-cell ray-parameter intervals, represented
as packed crossing events:

- `cell_ids[i]`
- `times[i]`
- `times[i + 1]`

for each crossed cell, in physical `xyz` ray coordinates. This is equivalent to
one `t_enter` and `t_exit` pair per crossed cell, but the shared trace product is
the packed `cell_ids` and `times` representation.

In other words, traversal should determine where the parametric ray

- `x(t) = origin + t * direction`

enters and exits each supported coordinate-slab cell. Those intervals should not
come from:

- ray marching
- midpoint approximations
- adaptive stepping
- post hoc estimates of path length from sample placement
- native-coordinate path lengths that are later converted to `xyz`

Downstream sampling and integration may choose midpoint, quadrature, or other
rules, but those sit on top of exact time intervals rather than replacing them.

## Ray traversal time-event contract
The ray tracer should not become separate Cartesian and spherical systems. The
shared contract is crossing events in ray-parameter time.

For every supported coordinate system, tracing should produce only:

- `cell_ids[i]`
- `times[i]`
- `times[i + 1]`

meaning the physical ray occupies `cell_ids[i]` over `[times[i], times[i + 1]]`.

Coordinate-specific code should be limited to finding and resolving the next
time event.

For `tree_coord="xyz"`, cells are slabs in Cartesian coordinate space and the
physical ray is linear in that space:

- `x(t)`
- `y(t)`
- `z(t)`

For `tree_coord="rpa"`, cells are slabs in spherical coordinate space, while the
same physical straight ray becomes a curved coordinate trajectory. Here `p`
means polar angle:

- `r(t)`
- `p(t)`
- `a(t)`

The two branches should differ only in:

- how candidate crossing times are computed
- how active faces at one time event are identified
- how the destination-side owning cell is resolved

After that, both branches should rejoin the same traversal shape:

1. current cell
2. current time
3. next crossing time
4. append current cell and crossing time
5. resolve destination cell
6. repeat

Spherical details such as event coordinates, azimuth seams, polar-axis transfer,
face ids, probe points, and half-open RPA ownership are local machinery for
resolving one time event. They should not leak into the trace product.

The layering rule is:

- geometry and traversal find cell ids and time intervals
- interpolation and accumulation consume cell ids and time intervals
- image formation sits above both

## Ray traversal refactor tasks
Move the current ray code toward the time-event contract with the following
tasks, in this order.

1. Keep the packed trace output as the only cross-branch product.
   Status: done.

   The shared contract is `cell_ids`, `times`, `ray_offsets`, and
   `time_offsets`. Do not expose coordinate-specific event state above the
   traversal layer. Packing scratch rows into flat trace arrays is shared by
   both coordinate branches.

2. Make the shared trace loop shape explicit.
   Status: done.

   Both coordinate branches should follow the same structure: current cell,
   current time, next event time, append segment, resolve destination cell,
   repeat.

3. Split each coordinate branch into two local responsibilities.
   Status: done.

   One part computes candidate crossing times for the current cell. The other
   resolves the destination-side owner at the chosen time event.

4. Keep `xyz` as the structural reference.
   Status: done.

   The Cartesian branch already has fixed scratch buffers, packed output, and
   direct accumulation. Shared non-geometric mechanics should not live in the
   coordinate branches.

5. Rework RPA event solving around fixed scratch arrays.
   Status: pending.

   Replace hot-path Python lists, small per-event allocations, and
   exception-driven invalid states with fixed buffers and explicit return codes.
   This is now the main remaining place where RPA keeps private traversal
   machinery that is not forced by the event equations.

6. Keep RPA seam and pole handling local to owner resolution.
   Status: done.

   Azimuth wrapping, polar-axis transfer, polar caps, probe points, and
   half-open RPA ownership should resolve one time event; they should not
   change the packed trace contract.

7. Make active-face handling time-based in both branches.
   Status: done.

   A multi-face event is one crossing time with several active faces, not
   several independent trace events. Zero-length hops may exist only to preserve
   correct ownership across the event.

8. Keep accumulation branch-independent above traced segments.
   Status: done.

   Midpoint and exact accumulation should consume known cell ids and time
   intervals. Coordinate-specific interpolation details belong below that
   boundary.

9. Do not add exact RPA accumulation until RPA traversal is structurally stable.
   Status: done.

   First make RPA produce the same kind of reliable time intervals as XYZ; only
   then design exact integration for spherical-coordinate cells.

10. Add or preserve tests around the shared invariants.
    Status: done.

    Tests should check clipped interval coverage, strictly increasing
    positive-length times, midpoint ownership, multi-face events, zero-hop
    ownership transitions, RPA azimuth seams, and polar-axis transfer.

11. Only optimize RPA after the structure is fixed.
    Status: pending.

    Once RPA no longer depends on Python lists, dynamic allocations, or
    exceptions in the trace loop, make the batch path Numba-compatible and
    compare it against the same workloads used for XYZ.

## Explicit non-goals for ray traversal
Traversal should work directly on the adaptive octree geometry.

It should not rely on proxy-volume tricks such as:

- resampling the entire octree onto a finest-level uniform grid
- materializing a whole-domain maximum-resolution Cartesian volume
- tracing through a dense voxelized surrogate instead of the adaptive cells
- approximating adaptive traversal by first expanding all cells to the finest refinement level

Those approaches are out of scope for the intended design because they:

- discard the main structural benefit of the octree
- inflate memory and preprocessing cost
- blur the geometric contract the tracer is supposed to honor
- make it easier for path-length calculations to drift away from exact time intervals

For this pass, the tracer should intersect the supported coordinate-slab cells
directly and produce exact per-cell time intervals from that geometry.

### Cartesian-tree traversal geometry
For `tree_coord="xyz"`, the intended traversal geometry is:

- axis-aligned Cartesian leaf cells

For these cells, "exact interval length" means the ray enters and exits the
cell's actual Cartesian box, and the resulting time interval is exact for that
supported geometry model.

### Spherical-tree traversal geometry
For `tree_coord="rpa"`, the ray tracer is still conceptually and numerically in physical `xyz`.

That means:

- rays are represented in `xyz`
- interval endpoints are `xyz` ray parameters
- segment lengths are defined in `xyz`, not in native `rpa` coordinates

The intended traversal model is slabs in `rpa` space with a curved coordinate
trajectory. A straight physical ray

- `x(t) = origin + t * direction`

induces coordinate functions:

- `r(t)`
- `polar(t)`
- `azimuth(t)`

Each spherical leaf is treated as an axis-aligned interval box in those
coordinates. Crossing events occur when the curved coordinate trajectory reaches
one or more cell-boundary slabs:

- `r = const`
- `polar = const`
- `azimuth = const`

The resulting segment intervals are still physical ray-parameter intervals in
`t`. The tracer should not compute a native-coordinate path length and then
convert it into a physical length.

For astronomy-oriented shell tracing, the inner radial boundary should be treated as opaque:

- the visible interval is the front shell segment only
- when a forward ray reaches `r = rmin`, the ray is finished
- the tracer does not continue through the central hole to a backside shell interval

The native `rpa` boundary surfaces are:

- `r = const` gives spherical surfaces
- `polar = const` gives conical surfaces
- `azimuth = const` gives half-planes through the axis

Turning those into a robust per-cell time-event algorithm requires consistent
handling of:

- periodic azimuth wrapping
- pole-adjacent behavior
- tangent and grazing hits
- face, edge, and corner coincidences
- interval ordering along a Cartesian ray
- agreement with the octree's own containment and boundary rules

## Current public octree surface
The current octree-facing public entrypoints are:

- `Octree(...)`
- `Octree.from_ds(...)`
- `Octree.lookup_points(points, coord=...)`
- `Octree.domain_bounds(coord=...)`
- `OctreeInterpolator(tree, values)`

The traversal layer should be designed to rely on stable octree behavior, not on hidden implementation details beyond what is explicitly documented here.

## Octree invariants required before ray work

### 1. Tree coordinate system is explicit and stable
Every `Octree` has one fixed `tree_coord`.

Supported meanings:

- `tree_coord="xyz"` means the tree geometry is interpreted in Cartesian coordinates
- `tree_coord="rpa"` means the tree geometry is interpreted in spherical `(r, polar, azimuth)` coordinates

Required invariant:

- all octree geometry state is internally consistent with `tree_coord`
- coordinate conversion is explicit, never inferred implicitly by public imaging code

For ray work this means:

- rays themselves should still be described in physical `xyz`
- any conversion into tree-local coordinates belongs to geometry and traversal logic, not to public imaging code

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

The tracer must not contain its own independent tie-breaking policy. It should
be able to rely on the octree contract.

### 4. Domain bounds are exact enough to be a clipping primitive
`Octree.domain_bounds(...)` should be a reliable representation of the global represented domain in the tree's own coordinate system.

Required invariant:

- the returned bounds are the intended traversal domain, not just a loose envelope
- the same domain interpretation is used by lookup, interpolation, and traversal
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

- radial, polar, and azimuth intervals define the supported traversal slabs
- periodic azimuth behavior must be explicit enough for exact time-event ownership
- pole-adjacent ownership must be explicit enough for exact time-event ownership

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

This matters because image formation combines:

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

This consistency is especially important because the imaging API should remain
physically `xyz`-oriented even when the underlying tree is spherical.

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
The tracer should not introduce a second geometry model.

Required invariant:

- point containment, interpolation fractions, and ray time-event computations all refer to the same coordinate-slab geometry
- tolerances used in one layer do not contradict another layer's notion of inside versus outside
- the same coordinate conventions and periodic rules are used everywhere

If traversal needs geometry that the current octree does not expose cleanly, the
octree should be extended first rather than patched around downstream.

## Readiness checklist for ray and imaging work
Before extending the traversal subsystem, the octree should be in a state where
the following are true:

- `lookup_points` has stable documented boundary behavior
- `domain_bounds` is trusted as the represented traversal domain
- `cell_bounds` semantics are explicitly documented and tested
- local neighbor refinement balance is explicit and tested
- Cartesian-tree geometry limitations are explicit
- spherical periodic-axis behavior is explicit
- native-domain occupancy rules are explicit, including the `rmin > 0` spherical case
- the traversal geometry model is explicit for both `xyz` and `rpa` trees
- exact segment time intervals are defined in physical `xyz` ray parameters
- the traversal path does not depend on densifying the tree to maximum resolution
- cross-coordinate lookup consistency is tested for interior points
- the meaning of misses is clear enough for downstream integration
- leaf ids are a stable contract for downstream data access

## What the ray layer should be allowed to assume
Once the above is true, the ray layer may assume:

- it can trace in physical `xyz`
- geometry and traversal code own coordinate conversion details
- the octree exposes one consistent notion of domain membership
- neighboring cells differ by at most one refinement level
- packed times define exact enter and exit parameters for the supported coordinate-slab geometry model
- for `rpa` trees, those exact intervals come from crossing native `rpa` slabs along a physical `xyz` ray
- traversal operates directly on adaptive cells rather than on a densified whole-domain proxy
- leaf ids returned by traversal can be used for downstream sampling and integration
- exact segment ordering can be built from exact time events rather than from heuristic stepping

## What the ray layer should not assume
The ray layer should not assume, unless the octree explicitly promises it:

- arbitrary non-axis-aligned Cartesian cell geometry
- fallback behavior for ambiguous boundary hits
- hidden coercions between `xyz` and `rpa`
- that every global bounding-box point has represented data
- that old ray API shapes or names should be preserved

## Immediate development implication
The next ray milestone should be to make the current traversal code match the
time-event contract while preserving the octree invariants that traversal
depends on.
