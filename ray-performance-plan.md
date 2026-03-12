# Ray Performance Plan

## Current state (measured)

Using `examples/ray_fast_two_pooch.ipynb` style workload at `3x3` rays:

- SC (`tree_coord=rpa`)
  - `OctreeInterpolator` build: ~7.9s
  - `integrate_field_along_rays` (`exact`): ~2.6s
  - Baseline (`interp` on `nx=256` grid + sum): ~0.001s
  - Ray is ~3440x slower than baseline
- IH (`tree_coord=xyz`)
  - `OctreeInterpolator` build: ~14.6s
  - `integrate_field_along_rays` (`exact`): ~4.8s
  - Baseline (`interp` on `nx=256` grid + sum): ~3.7s
  - Ray is ~1.3x slower than baseline

Additional checks:

- IH fast-path eligibility is true (`can_xyz_scalar_kernel=True`), so this is not just “wrong path selection.”
- One-ray tracer hot path is fast (IH: ~0.0004s after warm-up), so the bottleneck is in integration path, not plain tracing.
- Tests are mixed:
  - many unit/edge ray tests are fast (`<1s` range)
  - notebook-contract ray test is slow (`~14s`)
  - slow-marked ray tracer module run is `~14s`

## Goal

Ray integration must be faster than the grid-sample-then-sum baseline for the same output task.

Target bands:

1. Immediate: 10x faster for SC at tiny grids (`3x3`) and clearly faster than baseline.
2. Mid-term: 10x-100x at practical image sizes (`32x32`, `64x64`).
3. Long-term: 100x-1000x for large ray batches where kernels should dominate.

## Step-by-step plan

1. Lock down a reproducible benchmark harness
- Add one dedicated perf script (not notebook) that reports:
  - setup time (`Dataset.from_file`, tree/interpolator build)
  - hot-call integration time (after warm-up)
  - baseline grid-sum time
- Evaluate both files (`SC`, `IH`) at `3x3`, `32x32`, `64x64`.
- Output CSV/markdown table for before/after comparisons.

2. Separate setup cost from algorithm cost everywhere
- Keep notebook setup cached and never rebuild for benchmark/render cells.
- Ensure benchmark cells never include `Dataset.from_file` or `OctreeInterpolator(...)`.
- Keep method selection explicit (no default 2x2 benchmark loops in notebook).

3. Add explicit runtime-path diagnostics (once per run)
- Print selected path for each run:
  - `xyz axis-aligned kernel`
  - `xyz general kernel`
  - `rpa kernel`
  - fallback path
- This removes guesswork about whether numba kernels were used.

4. Fix tiny-ray performance for xyz kernels
- Add serial (`parallel=False`) versions of xyz scalar kernels.
- Dispatch rule:
  - small `n_rays` -> serial kernel
  - large `n_rays` -> parallel kernel
- Rationale: current `prange` path is expensive for tiny workloads.
- Acceptance: IH `3x3` should drop from multi-second to sub-second.

5. Add true bulk integration kernel for rpa path
- Current rpa integration still behaves like generic tracing/integration flow.
- Implement dedicated numba kernel for spherical tree integration (scalar first):
  - trace + accumulate in one compiled pass
  - avoid Python segment materialization and per-ray Python loops
- Acceptance: SC `3x3` and `32x32` move from orders-of-magnitude behind baseline to at least parity, then faster.

6. Remove avoidable overhead in integration kernels
- Eliminate duplicate per-step work in hot loops:
  - repeated conversions
  - repeated lookup fallback checks
- Tighten stepping logic with clear convergence guards to avoid pathological tiny-step loops.
- Replace large fixed `max_steps` constants with data-driven bounds where safe.

7. Improve data-locality for value evaluation
- For xyz scalar path, precompute cell-local interpolation coefficients if cheaper than repeated corner gathers.
- Avoid repeatedly rebuilding index maps inside tight loops.

8. Add fast miss culling before tracing
- For both coord systems, skip rays that cannot intersect domain with cheap tests.
- For spherical domain workloads, use radial intersection prefilter before heavy traversal.

9. Rework notebook defaults for speed-first behavior
- Default method map fixed to known-fast choices.
- Benchmark cell explicitly optional and single-label by default.
- Keep low-res debug mode (`3x3`) but ensure it is actually fast.

10. Performance guardrails in tests/CI
- Add opt-in perf tests (`--run-perf`) with wall-time budgets:
  - IH `3x3` hot call budget
  - SC `3x3` hot call budget
  - `32x32` throughput budget
- Keep them separate from unit-fast default runs.

## Execution order (highest ROI first)

1. Step 4 (serial dispatch for tiny xyz workloads)
2. Step 3 (path diagnostics)
3. Step 2 + Step 9 (notebook/setup hygiene)
4. Step 5 (dedicated rpa bulk kernel)
5. Step 6 + Step 7 + Step 8 (kernel-level optimization pass)
6. Step 10 (perf CI guardrails)

## Definition of done

- For both SC and IH tasks, ray integration beats grid-sample+sum baseline at matched output.
- Notebook `3x3` runs in seconds, not minute scale.
- Performance path is explicit and auditable.
- No hidden fallback behavior in “fast” notebook flow.
