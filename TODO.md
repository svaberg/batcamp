# TODO

- [ ] Do not create additional modules for now; reduce complexity in existing modules first.
- [x] Remove or justify `OctreeInterpolator.rpa_to_xyz`, which is currently unused in the repo.
- [x] Remove or use dead argument `bisect_iters` in `_trace_segments_xyz_kernel` (currently ignored).
- [x] Align backend evaluator signatures to remove dead placeholder args (for example `_evaluate_xyz(..., _qs, ...)`).
- [x] Reclassify long-running tests that are not unit-fast behind explicit markers/flags so default runs stay fast.
- [x] De-duplicate expensive test setup by sharing one cached dataset/tree fixture across modules that use `difflevels-3d__var_1_n00000000.dat`.
- [ ] Reduce test lock-in on private internals (`_cell_*`, `_lookup_state`, `_bin_to_corner`) unless there is no public API alternative.
- [x] Rename `batcamp/base.py` to `batcamp/octree.py` and update imports.
- [x] Consolidate duplicated XYZ<->RPA conversion helpers shared across `interpolator.py`, `spherical.py`, and `ray.py`.
- [x] Unify tree auto-build fallback logic so there is one source of truth instead of split behavior paths.
- [x] Reassess and eventually remove stale-module numba cache cleanup paths once legacy cache compatibility is no longer needed.
- [ ] Reduce complexity in oversized modules (`interpolator.py`, `ray.py`) in-place (no new modules for now).
