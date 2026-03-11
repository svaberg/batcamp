# Good Practices

A short, practical guide for writing maintainable scientific/technical code.

## 1. Keep code short and explicit
- Prefer direct expressions when logic is simple.
- Add helpers only when they reduce real duplication or complexity.
- Avoid boilerplate wrappers that hide 1-3 lines of real work.

## 2. Use one clear code path by default
- Keep branching minimal and intentional.
- Do not add defensive fallbacks unless they are part of the real API.
- If behavior is optional, make it explicit via parameters.

## 3. Preserve data semantics
- Keep raw data in its native meaning (for example node data vs cell data).
- Perform conversions explicitly and close to where they are needed.
- Do not silently reinterpret geometry or units for convenience.

## 4. Separate orchestration from implementation
- High-level workflows should coordinate steps, not contain core math.
- Put reusable algorithms in reusable modules.
- Keep entrypoints readable: load -> compute -> visualize/save -> report.

## 5. Prefer generic APIs over quantity-specific APIs
- Parameterize by field/metric names instead of duplicating near-identical functions.
- Reuse the same interface for similar operations.
- Keep API surface small and consistent.

## 6. Centralize formulas, units, and constants
- Define each core quantity/formula once and reuse it.
- Keep conversions and constants in one authoritative place.
- Avoid scattered hard-coded units and repeated numeric constants.

## 7. Keep dependencies and imports clean
- Avoid circular imports and layer inversions.
- Keep package facades small; import from owning modules when possible.
- Make optional dependencies truly optional (or clearly required).

## 8. Log at boundaries, not in hot loops
- Log start/end of workflows and major steps.
- Use debug logs for diagnostics, info logs for progress.
- Avoid print-based logging in library code.

## 9. Design return values for clarity
- Prefer explicit return shapes over large ad hoc dictionaries.
- Reuse shared data structures instead of inventing one-off containers.
- If a map/dict is returned, document why key-value output is the right abstraction.

## 10. Test behavior, not accidental structure
- Focus tests on correctness and meaningful edge cases.
- Keep detailed tests on reusable logic; keep workflow tests lightweight.
- Do not preserve poor design only to satisfy lock-in tests.

## 11. Catch exceptions narrowly
- Do not catch broad exceptions (for example `except Exception`) in core paths.
- Catch only expected failure types and define explicit fallback behavior.
- Let unexpected errors surface so defects are visible and fixable.

## Project-specific ownership and debt logging
- Keep ray-related implementation in `batcamp/ray.py` whenever possible.
- Keep spherical-related implementation in `batcamp/spherical.py` whenever possible.
- If code cannot be placed in its owning module, record blocker and rationale in `DEBT.md`.
- Remove dead paths and dead code whenever practical.
- Do not create additional modules at this stage; simplify existing modules first.
- Continuously evaluate test usefulness and prune redundant tests that do not improve behavior confidence.

## Quick self-check
- Is this code easier to read than before?
- Does it remove duplication rather than move it?
- Are assumptions explicit (units, shapes, fallbacks)?
- Would another engineer find ownership and data flow quickly?
