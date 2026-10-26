# Versioned geometry and common legalization

The API used by the main runner is:

```python
from chiplet_layout_experiment.geometry_versions import apply_geometry_version
from chiplet_layout_experiment.legalization import legalize

problem, metadata = apply_geometry_version(problem, metadata, "original")
result = legalize(problem, positions)  # optional keyword: max_attempts=6
```

Both functions own their outputs and leave input arrays and metadata unchanged.
Neither module changes `trajectory_data.py` or `trajectory_problem.py`.

## Geometry versions and provenance

| Requested version | Recorded `metadata["geometry_version"]` | Meaning |
| --- | --- | --- |
| `original` | `original` | Retain loaded geometry, including Case4 at 37 × 37 mm. |
| `case4-maskplace-59mm-alternative-v1` | `case4-maskplace-59mm-alternative-v1` | Explicit Case4 alternative at 59 × 59 mm. |
| `case4-59mm-alternative` | `case4-maskplace-59mm-alternative-v1` | Runner-compatible alias for the same alternative. |

The alternative is **not an official correction**. Only its board dimensions
come from MaskPlace; chiplets, powers, nets and pin definitions remain those of
the loaded problem. It is rejected for cases other than Case4. Unknown version
names raise `ValueError`. `geometry_version_requested` preserves the caller's
spelling. Original and alternative Case4 results must retain separate version
labels; a larger board changes the placement problem.

Local provenance was verified on 2026-10-25:

- Source: `F:/GAME/2027-SRTP-面向2.5D系统集成的芯粒布局方法研究/4-开源源码/MaskPlace2.5D/cases/Case4/Case4.intpsize`.
- First line: `59000.00000`, in micrometres.
- SHA-256 of the source bytes: `87397dc96a29ed206e446b76ac58d8a7ef95815e78d0a52c97509cbd9549e9cf`.
- Interpretation: `MaskPlace2.5D/process_thermal.py`, function `parse_intpsize_file`, lines 9–15, reads the first line and assigns `width = height = float(line)`.
- SHA-256 of that parser file: `27175aa825d9a996574411780e9ab03ab18f3883d2fc3628bce957f9ded78c6f`.

This audited source path, content, hash and interpretation are frozen in
`geometry_provenance`; execution elsewhere does not depend on that path existing.
The original source-file hash dictionary is preserved without inserting the
alternative board into it. The optional local-source test rechecks the actual
`.intpsize` hash when the file is available.

For each axis, the conversion is
`new_normalized = old_normalized * old_board_mm / new_board_mm`.
It applies to chiplet sizes, pin offsets and initial lower-left positions.
Consequently chiplet dimensions, pin offsets, pin absolute positions and initial
chiplet positions in millimetres are invariant. Powers, pin indices and names
are unchanged. Every call checks the physical array invariants with
`rtol=atol=1e-13`; metadata records `physical_invariants_verified`.

When the board changes, `TracedProblem.__post_init__` rebuilds its derived
geometry, thermal variances and initial reference scales. This computes reference
HPWL/thermal quantities through the existing constructor logic, without invoking
`evaluate` or `gradient`. The spatial thermal proxy uses normalized board axes,
so its values need not remain invariant under a board change. Reapplying the
same version is idempotent. Applying `original` to an already-versioned problem
restores the saved `geometry_origin` board and source description.

The real original Case4 has normalized total chiplet area approximately
`1.507669832` at 37 × 37 mm and remains area-impossible. The alternative has
utilization approximately `0.5929`. The adapter updates `board_um`,
`area_utilization` and `proven_infeasible_by_area` for the selected board.

## Legality and result contract

`legalize(problem, positions, *, max_attempts=6)` uses normalized lower-left
coordinates with fixed rectangle dimensions and orientations. It requires
finite `(problem.n, 2)` positions, positive finite sizes and positive finite
board axes; malformed input raises `ValueError`.

Its geometric verdict uses the same arithmetic and thresholds as
`TracedProblem.evaluate`:

```text
boundary_valid = all(positions >= -1e-10)
                 and all(positions + sizes <= 1 + 1e-10)
overlap_fraction = sum(pairwise exact rectangle intersection areas)
                   / sum(chiplet areas)
valid = boundary_valid and overlap_fraction < 1e-10
```

The intersection calculation detects complete containment as well as partial
overlap. Touching edges are allowed. It never substitutes the smooth overlap
penalty or raster occupancy for the exact intersection calculation. Every
returned valid layout has passed this geometric verdict. Legalization calls
neither objective nor gradient functions and leaves their counters unchanged.

`LegalizationResult` exposes these attributes:

| Attribute | Meaning |
| --- | --- |
| `positions` | Owned output array; original coordinates on search failure. |
| `valid` | Boolean geometric verdict on the returned positions. |
| `status` | Status from the table below. |
| `method` | `identity`, `clip`, successful named packing strategy, `proof`, or `bounded_packing` on exhaustion. |
| `wall_seconds` | Elapsed legalization time measured with `perf_counter`. |
| `geometry_checks` | Count of complete-layout geometric validations, including input and successful output checks; free-space bookkeeping is not a complete-layout check. |
| `displacement` | Frobenius norm of output minus input in normalized axes. |
| `displacement_mm` | Frobenius norm after scaling displacement by physical board axes. |
| `attempts` | Ordered tuple of dictionaries with `method`, `placed`, `complete`, `valid`; includes unsuccessful partial packings. |
| `overlap_fraction` | Exact pairwise overlap fraction of returned coordinates. |
| `boundary_valid` | Boundary verdict on returned coordinates. |

| Status | Interpretation |
| --- | --- |
| `already_valid` | Input is preserved exactly, with one geometry check and zero displacement. |
| `legalized` | Clipping or a named packing strategy produced a validated layout. |
| `proven_infeasible_by_area` | A necessary area condition proves no evaluator-valid layout exists. |
| `proven_infeasible_by_dimensions` | A rectangle cannot fit even with the evaluator's boundary tolerance. |
| `search_exhausted` | The configured heuristic search failed; feasibility is unknown. |

The area short circuit conservatively allows the evaluator's tolerances:
`area > (1 + 2e-10)**2 / (1 - 1e-10)` is sufficient for the proof.
The dimension proof requires an axis length above `1 + 2e-10`.
Valid input is checked first, so tolerance-valid existing states are preserved.

## Deterministic bounded work

After the initial check and necessary-condition proofs, boundary clipping is
checked only if it changes the input. Packing then tries this fixed sequence,
stopping on the first validated success:

1. MaxRects free-space packing in descending area order, choosing positions near the input.
2. MaxRects using shortest-side fit in descending area order.
3. The same corner strategy in descending height order.
4. The same corner strategy in descending width order.
5. Shelf packing in descending height order.
6. Transposed shelf packing in descending width order.

Chiplet index resolves sorting ties. There is no randomness, objective-based
selection, continuous optimization, rotation, solver or restart loop.
`max_attempts` is a nonnegative integer selecting a prefix of these strategies;
values above six still run at most six. Zero disables packing but retains the
input, proof and clipping checks. There are at most `2 + min(max_attempts, 6)`
complete-layout geometry checks, often fewer because incomplete packings are
reported without validating an incomplete array. Timing is measured, not used
as a cutoff, so repeated calls select the same coordinates and strategy.

Packing can move many chiplets and does not minimize displacement. Its
displacement and named method must be reported as common postprocessing,
separately from an optimizer's gains and objective-evaluation budget.

## Verification

```powershell
python -m pytest tests/experiment/test_geometry_versions.py tests/experiment/test_legalization.py -q
python -m pytest tests/experiment -q
```

The focused suite passed 23 tests. It covers physical invariance, independent
outputs, original 37 mm geometry, aliases and version round trips, provenance,
containment, boundaries, area/dimension proofs, honest heuristic exhaustion,
zero budget, malformed input, evaluator tolerances, no hidden objective calls,
and deterministic 61-, 64- and 100-chiplet packing. The experiment directory
passed 307 tests in the shared checkout at verification time.

A local audit also loaded all ten real cases with seed `20261024`. Case1,
Case3, alternative Case4 and Case5–10 legalized successfully and every returned
verdict matched `evaluate`. Original Case4 stopped after one check with the
area proof. Case2 (utilization approximately `0.9837`) exhausted six strategies
and remained `search_exhausted`; this observation is not a feasibility proof.
Observed per-case times in that single audit were approximately 1–15 ms,
including about 14 ms for the 61-chiplet Case10. These are local smoke timings,
not a performance guarantee or an equal-time algorithm comparison.
