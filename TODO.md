# TODO

- [ ] Enforce ownership: spherical-specific methods/logic must live only in spherical classes/modules (no spherical helpers on coord-agnostic facades).
- [ ] Seek and destroy wrapper layering: remove pass-through APIs that mostly forward to another method without adding meaningful behavior.
- [ ] Expand `[project.optional-dependencies].tests` to include non-pytest test imports (at least `pooch`) and verify tests run in a clean `. [tests]` environment.
- [ ] Investigate and suppress/resolve intermittent runtime warning: `OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.`
- [ ] Investigate and suppress/resolve notebook warning from `tqdm.auto`: `TqdmWarning: IProgress not found. Please update jupyter and ipywidgets.`
- [ ] Fix `examples/octree.ipynb` for current builder API: replace removed `OctreeBuilder.compute_phi_levels(...)` usage with `SphericalOctreeBuilder.compute_delta_phi_and_levels(...)` (or equivalent current public API) so executed notebook tests pass.
- [ ] Audit polar/azimuth plotting across examples: use angular grids/ticks that divide 180 cleanly and orient polar-angle axes so minimum polar is not shown at the top.
- [ ] Reconcile ownership/debt policy with implementation: for any spherical logic left outside `batcamp/spherical.py`, either move it or record the blocker/rationale explicitly.
- [ ] Replace `OctreeLookup` private-attribute probing (`getattr(..., \"_lookup_state\"/\"_points\"/\"_cell_phi_*\")`) with explicit backend contracts.
- [ ] Collapse `OctreeLookup` pass-through wrappers where they only forward to `Octree` without adding behavior.
- [ ] Remove thin pass-through docstrings by removing the wrapper method or moving the useful behavior to one canonical method.
- [ ] Ensure every remaining public docstring states plain behavior: required inputs, returned value shape/type, and failure conditions.
- [ ] Post-release API pass: make `OctreeInterpolator` feel more like `scipy.interpolate.LinearNDInterpolator` where that fits the octree model. Target the call contract first: predictable query-shape handling, predictable return-shape handling for scalar/vector fields, and `NaN` for misses/outside-domain. Keep `tree=` as an advanced knob rather than part of the main mental model.
