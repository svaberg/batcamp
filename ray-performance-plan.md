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

## Step-by-step plan (execution order)

1. Build reproducible benchmark harness (30 min)
- Deliverable:
  - one perf script (not notebook) with machine-readable output table
  - runs `SC` and `IH` for `3x3`, `32x32`, `64x64`
- Measures:
  - setup time (`Dataset.from_file`, tree/interpolator build)
  - hot-call ray integration time
  - baseline grid-sample+sum time
- Exit criteria:
  - baseline table checked in and rerunnable in `batcamp` env.

2. Add runtime path diagnostics (20 min)
- Deliverable:
  - one explicit per-run path summary (`xyz axis`, `xyz general`, `rpa`, fallback)
- Exit criteria:
  - no ambiguity about whether numba kernel paths were used.

3. Keep setup and benchmarking separated everywhere (20 min)
- Deliverable:
  - notebook and perf script never rebuild setup inside benchmark loops
  - method selection explicit by default; benchmark loops opt-in only
- Exit criteria:
  - `3x3` debug flow does not include hidden repeated setup/timing loops.

4. Fix tiny-ray xyz path with serial kernel dispatch (40 min)
- Deliverable:
  - serial (`parallel=False`) xyz scalar integration kernels
  - runtime dispatch rule: small `n_rays` -> serial, otherwise parallel
- Why:
  - current `prange` overhead is too large for tiny workloads.
- Exit criteria:
  - IH `3x3` hot call drops from multi-second toward sub-second.

5. Optimize xyz hot-loop overhead (60 min)
- Deliverable:
  - reduced per-step overhead in xyz kernels:
    - remove duplicated checks/conversions
    - tighten boundary stepping and loop guards
- Exit criteria:
  - measurable throughput gain at `32x32` and `64x64`.

6. Add dedicated rpa bulk integration kernel (120+ min, likely multi-session)
- Deliverable:
  - scalar rpa kernel that traces and accumulates in one compiled pass
  - avoids Python segment object/materialization path
- Exit criteria:
  - SC `3x3` and `32x32` move from orders-of-magnitude behind baseline toward parity/faster.

7. Add miss-culling and cheap prefilters (30 min)
- Deliverable:
  - cheap domain/radial culling before expensive tracing for both coord families
- Exit criteria:
  - rays that cannot hit are rejected early with measurable time reduction.

8. Improve interpolation data locality (45 min)
- Deliverable:
  - reduced repeated corner/index map work in tight loops
  - optional precomputed local coefficients where beneficial
- Exit criteria:
  - additional speedup visible in perf harness for both files.

9. Add perf guardrails (45 min)
- Deliverable:
  - opt-in perf tests (`--run-perf`) with budgets:
    - IH `3x3` hot call budget
    - SC `3x3` hot call budget
    - `32x32` throughput budget
- Exit criteria:
  - regressions are caught automatically without slowing default unit test runs.

10. Final target validation (20 min)
- Deliverable:
  - before/after comparison against baseline grid-sum approach
- Exit criteria:
  - ray path faster than baseline for target cases, with table in repo.

## Definition of done

- For both SC and IH tasks, ray integration beats grid-sample+sum baseline at matched output.
- Notebook `3x3` runs in seconds, not minute scale.
- Performance path is explicit and auditable.
- No hidden fallback behavior in “fast” notebook flow.
