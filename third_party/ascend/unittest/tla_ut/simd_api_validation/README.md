# SIMD API validation harness

Portable validation material for the elementwise SIMD APIs exposed by
`triton.language.extra.tla.simd` (`vdiv` / `vmax` / `vmin` / `vmaxs` / `vexp` /
`vdup` / `vdups` / `vdupv`), with mirrored TileLang cases for cross-side
comparison.  TileLang has no `vdups`; its scalar `vdup` is the mirror of both
TLA `vdup` and TLA `vdups`.

The directory contains only case definitions, CPU references, device kernels
and host tests.  It owns no profiler, no interpreter selection, no environment
variables and no run artifacts; the caller provides the Python environment and
an output directory.

## Layout

```
simd_harness/            # device-agnostic core
  spec.py                # CaseSpec / OutputSpec / comparison rules
  cases.py               # case registry, deterministic inputs, CPU reference
  inputs.py              # f32/f16/bf16/uint bit patterns (NaN payload from bits)
  reference.py           # CPU reference + comparator (first lane, zeroing, NaN)
  encoding.py            # raw-bit JSON (base64) / npz (no pickle)
  capabilities.py        # frontend + compiler capability probes
  runner.py              # --list / --check / --case / --output / --execute
  kernels_tla.py         # TLA device kernels only
  kernels_tilelang.py    # TileLang device kernels only
  backends/              # thin import/launch adapters (imported on execute only)
tests/test_harness.py    # host tests: inputs/comparator/JSON/CLI exit codes
compare_results.py       # standalone cross-side comparison (raw bits)
conftest.py              # keeps this harness out of tla_ut's "collect every *.py"
```

## Case semantics

A case is a pure-Python description: deterministic inputs, a CPU reference, a
comparison rule per output, and a kernel name per side.  Nothing in the case
definitions imports torch, triton or tilelang.

- `tier=implemented` — inputs, reference and kernels for the listed `sides`
  exist; the case can be evaluated on the host or executed on a backend.
- `tier=suggested_not_implemented` — evaluated idea without inputs/reference/
  kernels yet. `--list` reports it; `--case` refuses it with a clear error.

Comparison rules are explicit per output:

| field | meaning |
|---|---|
| `mode=exact` | raw-bit equality (after the NaN policy) |
| `mode=close` | `rtol`/`atol` on finite lanes, `inf` compared exactly |
| `nan_policy` | `forbid` / `position` / `propagate` / `canonical_bits` |
| `signed_zero=ieee` | a zero result must keep the reference sign |
| `zero_inactive` | masked-out lanes must be bitwise `+0` |

The FTZ (`vexp`) reference classifies lanes against the f32 normal/subnormal
boundary (`ln(FLT_MIN) = -87.3365`), not against any measured device result:
FTZ replaces subnormal results with `+0`, so the case checks that deeply
underflowing lanes are bitwise zero while normal lanes stay inside the
declared `rtol` budget.

`f32_dupv_mask` checks `vdups` and `vdupv` under all/first-half/all-false masks.
Lane 0 and the last lane are deliberately different, so a half-register
broadcast cannot pass as `POS_HIGHEST`.  `f32_mask_pattern` extends the
arbitrary/all-false matrix to `vdups` and both `vdupv` positions, so all eight
public APIs are exercised under a `vcmp`-built non-prefix mask and an
all-false mask.

`i64_u64_dupv_mask` does the same for real 64-lane i64/u64 registers
(`vector<64xi64>` / `vector<64xi1>`, `NORM_B64`).  Its arbitrary mask is built
from signed `int64` lane indices: `idx == lane`, `rem == lane + (lane % 3 != 0)`
and the predicate is `rem > idx`, i.e. exactly 42 active lanes.  No u64 payload
is compared, so the API case has no dependency on unsigned `vcmp` semantics.
This case is TLA-only because the local TileLang package cannot store
`NORM_B64`.  `diagnostic_u64_vcmp_mask` keeps the unsigned-compare question as
an isolated probe: it materialises `vcmp(u64, u64, "gt")` as u64 0/1 lanes and
never calls `vdupv`.  It is a vcmp finding, not an API acceptance row and not
evidence that `vdupv` is broken; do not include it in API acceptance.

`div_signed_wide_masks` covers signed i16/i32 and i64/u64 division under
prefix-half, `vcmp(a>b)` (i16/i32 only) and all-false masks.  The i64/u64
arbitrary row uses the same signed-index mask as `i64_u64_dupv_mask` (`rem >
idx`, 42 active lanes) instead of a u64 data compare, so the wide rows are
independent of unsigned `vcmp` semantics.  It keeps the strict integer oracle
and bitwise inactive-zero check without extending the unsigned 16/32-bit
`vsel(result, 0)` compensation to types that have not shown the defect.

`f32_div_exact` runs `precision='exact'` and `'vdiv_0ulp_ftz_true'` under
full/prefix/all-false masks and requires bit-for-bit equality with
correctly-rounded f32 division (plus bitwise inactive zeros).  It checks the
frontend maps these tiers to `ave.hir.vdivfhp`, not the FTZ `ave.hir.vdiv`.

`u16_div_zero` / `u32_div_zero` store masked unsigned division with the ALL
store mask and require every inactive lane to be bitwise `+0`, for full, prefix
half and all-false masks.  TLA runs both `MODE_ZEROING` and `MODE_MERGING`;
the MERGING rows are TLA-only because their merge source is a `+0` SSA vector
(builder-side `vsel`), not TileLang's dst-preserving merge.  For
`MODE_ZEROING`, a caller-supplied mask on unsigned 16/32-bit division appends
an explicit `vsel(result, 0)` in the TLA frontend; `MODE_MERGING` uses only the
builder merge and never stacks a second select.

`f32_mode_mask` runs the six APIs twice under one arbitrary partial mask,
once with `MODE.ZEROING` and once with `MODE.MERGING`.  Both TLA outputs must
equal the same masked reference with bitwise `+0` in inactive lanes.  The mode
is consumed by `tla_builder.cc::applySimdMode`: ZEROING keeps the bare masked
AVE op, MERGING adds an `arith.constant` `+0` -> `ave.hir.scalar_broadcast`
merge source and an `ave.hir.vsel` for the inactive lanes.  That is still not
TileLang's dst-preserving merge, so the TileLang side of this case is compiled
with `MODE_ZEROING` and is never compared against a preserving-dst MERGING
result.

## Long / tail matrix

`f32_long_tail` and `f16_long_tail` are TLA cases that sweep a runtime `N`
over long, padded buffers (one kernel launch per `N`).  They are functional
matrix tests, not simulator or profiling artifacts.

| case | register | chunk | capacity | runtime N |
|---|---|---|---|---|
| `f32_long_tail` | 64-lane f32 | 64 | 320 (5 chunks) | 1, 31, 32, 63, 64, 65, 127, 128, 129, 257 |
| `f16_long_tail` | 128-lane f16 | 128 | 512 (4 chunks) | 1, 63, 127, 128, 129, 255, 256, 257, 385 |

Every buffer is guarded on both sides: inputs are `8 + CAP + 8` (guard lanes
`123.5`) and the output rows are `8 + CAP + 8`, pre-filled with the non-zero
sentinel `17.0`.  `vlds` has no mask, so full chunks are only read inside the
legally padded `CAP` buffer; these cases do not claim masked loads without
padding.  The tail predicate is a single signed `vcmp <LT>` between a runtime
lane index and a runtime `N` broadcast (`vci` on the 64-lane side, a padded
`int16` index load on the 128-lane side).  `pge` is used only for the ALL store
mask, never to encode a 31/63/... tail prefix.

Per `N`, nine arithmetic rows cover all eight public APIs (`vdupv` has a
lowest-lane and a highest-lane row) and store with the ALL mask: inactive data
lanes must be bitwise `+0`.  Four representative rows (`vmax`, `vdiv`,
`vdupv_low`, `vdupv_high`) use an actual masked `vsts`, so every lane after
`N` keeps the non-zero sentinel.  Guards stay sentinel in both families.
`vdupv` references are per chunk: the lowest row repeats lane `chunk_start`
and the highest row repeats lane `chunk_start + CHUNK - 1`, even when `N` ends
inside that chunk.  `vdiv` uses a separate constant divisor so the active
results stay position-unique at f16 precision; `vexp` active results are
monotonic and unique over the chosen range.  A `CORES=2` program loop walks
all chunks, so each program executes more than one chunk per launch.

Host coverage checks the N matrix, guard/sentinel contract, per-chunk `vdupv`
reference, unique active patterns, and that the oracle rejects both a missing
ALL store (sentinel left in the arithmetic tail) and a masked store that
wrongly writes zeros after `N`.  The wheel-side IR test
`test_tail_mask_ir` in `vector/simd_elementwise.py` checks that the runtime `N`
and `vci` feed one signed `vcmp <LT>`, that four stores use that predicate as
an actual tail mask while nine use the independent all-true mask, and that the
chunk loop is present.

## Requirements

- Python with `numpy` (and `pytest` for the host tests).
- For device execution: a Triton/TileLang frontend that contains these APIs and
  a `bishengir-compile` that advertises `--enable-tla-mode` (found via `PATH`,
  `TRITON_NPU_COMPILER_PATH` or the installed Triton backend package).  Use a
  build or wheel that contains the APIs; nothing here is pinned to a specific
  installation.  `--check` reports what the current environment provides.
  MERGING's zero-init + `vsel` lowering lives in
  `language/tla/cpp/tla_builder.cc::applySimdMode`, so a device/IR run of the
  MERGING rows requires a TLA extension rebuilt from this tree; a stale `.so`
  still emits the old bare-op MERGING IR.

## Running

```bash
HARNESS=<repo>/third_party/ascend/unittest/tla_ut/simd_api_validation

# describe cases / probe the environment
PYTHONPATH=$HARNESS python -m simd_harness.runner --list
PYTHONPATH=$HARNESS python -m simd_harness.runner --check --side tla

# host reference only (inputs + CPU reference + serialised result)
PYTHONPATH=$HARNESS python -m simd_harness.runner \
  --case smoke --side tla --output <out>

# execute the device backend for the same case
PYTHONPATH=$HARNESS python -m simd_harness.runner \
  --case smoke --side tla --output <out> --execute

# compare two result trees lane by lane (raw bits)
PYTHONPATH=$HARNESS python $HARNESS/compare_results.py \
  --baseline <out>/tla --candidate <out>/tilelang

# host tests (no device, no compiler)
python -m pytest -q $HARNESS/tests/test_harness.py
```

`--executor reference|zero|teardown` are test-only executors used by the host
tests to exercise the exit codes (0 / 1 / 3) without importing a backend.
Each run writes `result.json`, `arrays.npz` and `inputs.npz` under
`<out>/<side>/<case>/`; payloads are raw bytes, never Python floats.

## pytest integration

`tla_ut/conftest.py` collects **every** `*.py` as a test module because the op
cases are named by op rather than `test_*`.  That rule does not fit a harness
whose modules import `torch`/`triton` and the compiled TLA bindings at import
time, so the local `conftest.py` here:

* ignores the harness and the standalone comparator (`collect_ignore_glob`),
  so no device module is imported during collection;
* drops the duplicate `tests/test_harness.py` node id that the `tla_ut` hook
  and pytest's own python plugin both produce.

Running `pytest third_party/ascend/unittest/tla_ut` therefore still collects
the existing cases exactly as before, plus the host tests here.
