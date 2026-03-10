# TODO

- [ ] Rename `batcamp/base.py` to `batcamp/octree.py` and update imports.
- [ ] Keep ray-related code in `batcamp/ray.py` whenever possible.
- [ ] If ray-related code cannot be moved to `batcamp/ray.py`, document the blocker and rationale in `DEBT.md`.
- [ ] Keep spherical-related code in `batcamp/spherical.py` whenever possible.
- [ ] If spherical-related code cannot be moved to `batcamp/spherical.py`, document the blocker and rationale in `DEBT.md`.
- [ ] Beware of dead paths and dead code; remove them when possible.
- [ ] Critically evaluate test usefulness: remove redundant checks and keep behavior-contract tests as the default.
- [ ] Reduce test lock-in on private internals (`_cell_*`, `_lookup_state`, `_bin_to_corner`) unless there is no public API alternative.
- [ ] De-duplicate expensive test setup by sharing one cached dataset/tree fixture across modules that use `difflevels-3d__var_1_n00000000.dat`.
- [ ] Reclassify long-running tests that are not unit-fast behind explicit markers/flags so default runs stay fast.
- [ ] Remove or justify `OctreeInterpolator.rpa_to_xyz`, which is currently unused in the repo.
- [ ] Remove or use dead argument `bisect_iters` in `_trace_segments_xyz_kernel` (currently ignored).
- [ ] Align backend evaluator signatures to remove dead placeholder args (for example `_evaluate_xyz(..., _qs, ...)`).
- [ ] Reassess and eventually remove stale-module numba cache cleanup paths once legacy cache compatibility is no longer needed.
- [ ] Consolidate duplicated XYZ<->RPA conversion helpers shared across `interpolator.py`, `spherical.py`, and `ray.py`.
- [ ] Split oversized modules (`interpolator.py`, `ray.py`) while removing dead/duplicate branches during extraction.
- [ ] Unify tree auto-build fallback logic so there is one source of truth instead of split behavior paths.
