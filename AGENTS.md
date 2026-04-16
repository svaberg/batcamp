# AGENTS.md

## Scope
These instructions apply to the entire repository.

## Project character
This is a numerics-oriented codebase.

Priorities, in order:

1. correctness
2. clarity
3. directness
4. performance
5. minimal surface area

This is not a framework, not a plugin system, and not a compatibility layer.

Current ray work should be treated as a synthetic imaging subsystem built on octree-aware traversal. Keep geometry and traversal separate from interpolation, accumulation, and image formation.

## General coding bias
Prefer the simplest implementation that satisfies the current requirements.

Do not add:
- shims
- adapters
- wrappers for their own sake
- compatibility layers
- deprecation paths
- fallback implementations
- defensive handling for hypothetical future cases
- defensive numpy casting
- abstraction for unneeded extensibility
- version-dependent branching unless explicitly required
- optional parameters added only for future flexibility
- base classes, registries, factories, or plugin mechanisms unless explicitly requested

Do not "future-proof" the code unless the task explicitly requires it.

Do not preserve old APIs, old argument names, old data formats, or old behaviors unless the task explicitly requires backward compatibility.

## Objects and arrays
Prefer direct NumPy-style array operations and explicit data flow.

Use objects sparingly when they improve naming, validation, or packaging of flat arrays.

Good objects:
- thin wrappers over flat or contiguous arrays
- small data containers with clear shape, unit, or indexing invariants
- small entrypoint objects that own one numerical state bundle

Avoid:
- class hierarchies
- stateful orchestration objects
- method-heavy wrappers around array operations
- hidden caches unless they are required for current performance
- configuration objects that mostly forward arguments

## Executing tests, code and notebooks
- If you see an `environment.yml` file, assume that the environment exists and use it when running tests, code, and notebooks.
- On branches other than `main`, commit notebooks unrun: no outputs and no execution counts.
- On `main`, commit run notebooks only after running them exactly once from a clean kernel.
- When notebooks generate plots you should check whether they look right.
- Before committing, run the same flake8 command as CI and do not commit if it fails: `flake8 batcamp tests examples --count --statistics`.

## Execution discipline
- Verify before claiming. If a statement depends on code, tests, profiling, timings, or artifacts, check it first or say that it has not been checked.
- If something is not verified, say so plainly and use tentative language. Prefer `I think ...`, `it looks like ...`, or `could it be that ...` over stating guesses as facts.
- Treat performance claims as experimental results, not impressions. Use controlled before/after comparisons with the same command, the same workload, and no overlapping runs.
- Prefer the authoritative benchmark or script for the subsystem over ad hoc side measurements. Side profiling may explain a result, but it does not replace the real benchmark.
- Default to the smallest relevant check:
  - `py_compile` for syntax
  - the narrowest relevant `pytest -k ...` selection for behavior
  - a full suite only when it is truly needed or explicitly requested
- Do not launch long-running tests or benchmarks when a targeted check is sufficient.
- When a result is contaminated, noisy, or obviously inconsistent, rerun it properly before drawing a conclusion.
- Once a change is accepted and the relevant checks are green, commit it promptly. Do not leave accepted work hanging uncommitted.
- Do not stop halfway through obvious cleanup. If a naming, API, or wording rule is accepted, apply it consistently through the touched code.
- Do not narrate process when the finished result is available. Report the concrete change, the concrete check, or the concrete blocker.
- Before tagging, releasing, or talking about branch state, verify the current branch, target commit, and whether the work belongs on that branch at all.

## Notebook hygiene
- Notebooks should demonstrate basic use cases, and showcase the speed, precision, and elegance of the library code.
- Prefer library code over notebook-local logic.
- Keep notebook narrative concise and concrete.
- When a notebook exists mainly as an example, optimize for clarity of use rather than exploratory clutter.

## Style
Prefer:
- plain functions and thin array-wrapping objects over class hierarchies
- explicit data flow over implicit state
- direct NumPy-style array operations
- small, composable helpers only when they reduce real duplication
- concrete names tied to the mathematics or algorithm
- straightforward control flow

Avoid:
- one-use indirection
- pass-through helper layers
- thin wrappers around existing functions
- "helper" functions that merely rename an operation
- object-oriented structure where stateless functions are sufficient
- configuration plumbing unless it is required by the task

## Numerics-specific guidance
Assume the code is written for known scientific and numerical use cases, not hostile or arbitrary inputs, unless stated otherwise.

Do not add guards for:
- impossible states that cannot occur under current invariants
- unsupported dtypes not mentioned in the task
- unsupported shapes not mentioned in the task
- hypothetical malformed data unless malformed input handling is part of the task
- hypothetical cross-version behavior differences unless explicitly required

Validate only what is necessary for correctness of the present task.

When checks are needed, prefer:
- clear preconditions
- explicit shape, unit, and domain checks
- failures that happen early and noisily

Do not add silent correction, silent fallback, or silent coercion.

## Error handling
Do not use broad exception handling.

Avoid:
- `except Exception`
- retry logic
- fallback branches
- "best effort" behavior
- warning-based recovery for core numerical logic

Prefer:
- specific exceptions
- explicit precondition checks
- immediate failure on unsupported cases

Errors should expose unsupported usage, not hide it.

## Compatibility
Assume the current supported environment only.

Do not add:
- legacy paths
- compatibility aliases
- conditional imports for old environments
- old and new API bridging
- polyfills
- dual implementations for historical reasons

If compatibility is genuinely required, it must be explicitly requested by the task.

## API discipline
Keep the public API small.

When adding new surface area, design the public API first and keep the first version intentionally minimal.

For ray and imaging work:
- start from the smallest public contract that can produce a correct image
- prefer exact traversal primitives and thin data containers
- layer interpolation and integration above traversal instead of conflating them
- expose array shapes, units, and indexing conventions clearly

Do not:
- add new public functions unless they are needed
- add convenience overloads
- add optional flags for speculative use cases
- expose internal plumbing

Prefer changing internals over expanding the public surface.

## Refactoring bias
When editing code, prefer reducing indirection.

Good refactors:
- remove dead branches
- remove unused helpers
- inline one-use wrappers
- collapse unnecessary abstraction
- simplify argument passing
- make data flow more explicit
- keep algorithmic structure close to the mathematics

Bad refactors:
- introducing a layer to "keep options open"
- adding abstractions before a second real use case exists
- extracting helpers that make the code harder to read
- replacing direct numerical code with framework-like structure

## Performance
Do not pessimize numerical code for the sake of abstraction.

Prefer:
- contiguous array-oriented operations where appropriate
- avoiding unnecessary allocations
- avoiding unnecessary conversions
- keeping hot paths obvious

Do not add extra layers in hot paths unless they provide a clear benefit.

## Documentation and comments
Comments should explain:
- the mathematical idea
- the algorithmic choice
- units, conventions, indexing, or invariants

Comments should not restate obvious code.

Do not add verbose defensive commentary about hypothetical future adaptations.

## Tests
Tests should target:
- current required behavior
- numerical correctness
- important invariants
- edge cases that are actually relevant to the supported domain
- public API contracts when new surface area is introduced

Do not add tests for speculative compatibility behavior that the project does not claim to support.

Tests should use pytest. In general modules should be tested in test modules with the same name.

## When uncertain
Use this decision rule:

If a layer, option, guard, wrapper, compatibility path, or abstraction is not required by the current task, do not add it.

Before finishing, check for:
- unnecessary wrappers
- compatibility code that was not requested
- speculative defensive logic
- optional parameters that are not needed
- abstraction without a present use case
- broad exception handling
- silent fallback behavior
