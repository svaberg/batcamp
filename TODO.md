# TODO

- [x] Remove or justify `OctreeInterpolator.rpa_to_xyz`, which is currently unused in the repo.
- [x] Remove or use dead argument `bisect_iters` in `_trace_segments_xyz_kernel` (currently ignored).
- [x] Align backend evaluator signatures to remove dead placeholder args (for example `_evaluate_xyz(..., _qs, ...)`).
- [x] Reclassify long-running tests that are not unit-fast behind explicit markers/flags so default runs stay fast.
- [x] De-duplicate expensive test setup by sharing one cached dataset/tree fixture across modules that use `difflevels-3d__var_1_n00000000.dat`.
- [x] Reduce test lock-in on private internals (`_cell_*`, `_lookup_state`, `_bin_to_corner`) unless there is no public API alternative.
- [x] Rename `batcamp/base.py` to `batcamp/octree.py` and update imports.
- [x] Consolidate duplicated XYZ<->RPA conversion helpers shared across `interpolator.py`, `spherical.py`, and `ray.py`.
- [x] Keep interpolation backend type/cache naming parallel (`SphericalInterpKernelState` vs `CartesianInterpKernelState`, with matching `_interp_state_*` and `_lookup_state_*` attributes).
- [x] Unify tree auto-build fallback logic so there is one source of truth instead of split behavior paths.
- [x] Reassess and eventually remove stale-module numba cache cleanup paths once legacy cache compatibility is no longer needed.
- [x] Reduce complexity in oversized modules (`interpolator.py`, `ray.py`) in-place (no new modules for now).
- [ ] Apply default-`xyz` naming consistently in interpolation internals: default path keeps base name, non-default path gets explicit suffix; rename `_trilinear_from_cell` / `_trilinear_from_cell_xyz` accordingly.
- [ ] Align lookup-state type names between backends: `spherical.LookupKernelState` should mirror `cartesian.CartesianLookupKernelState` naming (for example `SphericalLookupKernelState`).
- [ ] Align per-cell index field naming between backends (`_i0/_i1/_i2` in cartesian vs `_ir/_itheta/_iphi` in spherical) to one consistent convention.
- [ ] Align `_path(...)` parameter naming between backends (`i0/i1/i2` vs `ir/itheta/iphi`) with the same convention used for stored index fields.
- [ ] Align `hit_from_chosen` local index variable names between backends (`cell_i0/i1/i2` vs `cell_ir/cell_ipolar/cell_iazimuth`).
- [ ] Align lookup tuning constant naming between backends (cartesian uses inline literals where spherical uses named constants like `_LOOKUP_CONTAIN_TOL` / `_DEFAULT_LOOKUP_MAX_RADIUS`).
