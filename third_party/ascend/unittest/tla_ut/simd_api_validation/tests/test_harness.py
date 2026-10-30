"""Host tests: deterministic inputs, comparison rules, JSON, CLI exits.

These tests never import torch/triton/tilelang and never launch a device or a
profiler.  Run with any Python that has numpy/pytest:

    PYTHONPATH=<this directory> python -m pytest -q tests
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SIM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM_DIR))

from simd_harness import cases  # noqa: E402
from simd_harness.encoding import (array_digest, load_npz, save_npz,  # noqa: E402
                                   tensor_from_json, tensor_to_json)
from simd_harness.inputs import f32_from_bits  # noqa: E402
from simd_harness.reference import compare_output  # noqa: E402
from simd_harness.spec import OutputSpec  # noqa: E402


def _run(*args, cwd=None, env=None):
    full_env = os.environ.copy()
    full_env["PYTHONPATH"] = str(SIM_DIR) + (
        os.pathsep + full_env["PYTHONPATH"] if full_env.get("PYTHONPATH") else "")
    if env:
        full_env.update(env)
    return subprocess.run([sys.executable, "-m", "simd_harness.runner", *args],
                          cwd=str(cwd or SIM_DIR), env=full_env,
                          capture_output=True, text=True)


# ---------------------------------------------------------------------------
# deterministic inputs / lossless storage
# ---------------------------------------------------------------------------


def test_inputs_are_deterministic_and_lossless(tmp_path):
    for spec in cases.implemented_cases():
        arrays = spec.inputs_fn()
        again = spec.inputs_fn()
        assert set(arrays) == set(again), spec.name
        for name in arrays:
            a, b = arrays[name], again[name]
            assert a.dtype == b.dtype and a.shape == b.shape, (spec.name, name)
            assert array_digest(a) == array_digest(b), (spec.name, name)
            np.testing.assert_array_equal(
                np.ascontiguousarray(a).view(np.uint8),
                np.ascontiguousarray(b).view(np.uint8),
                err_msg=f"{spec.name}/{name} not bit-identical")
        if arrays:
            path = tmp_path / f"{spec.name}.npz"
            save_npz(path, arrays)
            loaded = load_npz(path)
            for name in arrays:
                key = name.replace("/", "__")
                assert loaded[key].dtype == arrays[name].dtype
                assert loaded[key].shape == arrays[name].shape
                assert array_digest(loaded[key]) == array_digest(arrays[name])


def test_nan_payload_survives_json_and_npz(tmp_path):
    arr = f32_from_bits([0x7FC00001, 0xFFC00002, 0x7F800000, 0x80000000])
    obj = tensor_to_json(arr)
    back = tensor_from_json(obj)
    assert array_digest(arr) == array_digest(back)
    assert back.view(np.uint32).tolist() == [0x7FC00001, 0xFFC00002, 0x7F800000, 0x80000000]
    path = tmp_path / "nan.npz"
    save_npz(path, {"x": arr})
    loaded = load_npz(path)["x"]
    assert loaded.view(np.uint32).tolist() == arr.view(np.uint32).tolist()


# ---------------------------------------------------------------------------
# comparator rules
# ---------------------------------------------------------------------------


def test_exact_comparator_reports_first_lane_and_bits():
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    got = ref.copy()
    got[1] = np.float32(2.5)
    r = compare_output(got, ref, OutputSpec("x", "<f4"))
    assert not r["pass"]
    assert r["first_fail"]["index"] == [1]
    assert r["first_fail"]["got"] == "2.5"
    assert r["first_fail"]["got_bits"] != r["first_fail"]["ref_bits"]
    assert r["max_abs_error"] is None or r["max_abs_error"] >= 0.5


def test_close_comparator_tolerance_and_signed_zero():
    ref = np.array([1.0, 0.0, -0.0, np.inf], dtype=np.float32)
    got = np.array([1.0 + 1e-7, 0.0, 0.0, np.inf], dtype=np.float32)
    ok = compare_output(got, ref, OutputSpec("x", "<f4", "close", 1e-6, 1e-6))
    assert ok["pass"], ok
    strict = compare_output(got, ref, OutputSpec("x", "<f4", "close", 1e-6, 1e-6,
                                                 signed_zero="ieee"))
    assert not strict["pass"]
    assert strict["first_fail"]["index"] == [2]


def test_zeroing_requires_bitwise_positive_zero():
    ref = np.zeros(8, dtype=np.float32)
    got = np.zeros(8, dtype=np.float32)
    got[5] = np.float32(-0.0)
    active = np.ones(8, dtype=bool)
    active[5] = False
    r = compare_output(got, ref, OutputSpec("x", "<f4", zero_inactive=True), active_mask=active)
    assert not r["pass"]
    assert r["zeroing"]["non_zero_inactive"] == 1


def test_nan_policy_position_vs_canonical_bits():
    ref = f32_from_bits([0x7FC00000, 0x3F800000])
    got = f32_from_bits([0x7FC00001, 0x3F800000])
    pos = compare_output(got, ref, OutputSpec("x", "<f4", nan_policy="position"))
    assert pos["pass"], pos
    strict = compare_output(got, ref, OutputSpec("x", "<f4", nan_policy="canonical_bits"))
    assert not strict["pass"]
    forbid = compare_output(got, ref, OutputSpec("x", "<f4", nan_policy="forbid"))
    assert not forbid["pass"]


def test_mode_case_merging_merge_source_is_zero_reference():
    spec = cases.get_case("f32_mode_mask")
    ref = spec.reference_fn(spec.inputs_fn())
    for suffix in ("D", "U", "E", "X", "S", "N"):
        z, m = ref.outputs[f"Z{suffix}"], ref.outputs[f"M{suffix}"]
        np.testing.assert_array_equal(z.view(np.uint32), m.view(np.uint32),
                                      err_msg=f"mode mismatch on {suffix}")
        assert ref.active_masks[f"M{suffix}"].any()
        assert not ref.active_masks[f"M{suffix}"].all()
    # MERGING is TLA-only: its builder-side merge source is +0, so the
    # numeric reference matches ZEROING without being TileLang's dst merge.
    assert spec.output("MD").sides == ("tla",)
    assert "tilelang" in spec.output("ZD").sides
    assert spec.supports("tilelang")


def test_unsigned_div_zero_cases_cover_masks_and_modes():
    for name, dtype in (("u16_div_zero", "<u2"), ("u32_div_zero", "<u4")):
        spec = cases.get_case(name)
        ref = spec.reference_fn(spec.inputs_fn())
        for prefix in ("Z", "M"):
            full = ref.outputs[f"{prefix}full"]
            half = ref.outputs[f"{prefix}half"]
            none = ref.outputs[f"{prefix}none"]
            assert full.dtype == np.dtype(dtype)
            # inactive lanes are bitwise zero; the full row is not masked out
            assert np.all(half[~ref.active_masks[f"{prefix}half"]] == 0)
            assert np.all(none == 0)
            assert np.array_equal(half[ref.active_masks[f"{prefix}half"]],
                                  full[ref.active_masks[f"{prefix}half"]])
        assert spec.output("Zhalf").zero_inactive
        assert spec.output("Mhalf").zero_inactive
        assert spec.output("Mfull").sides == ("tla",)
        assert spec.output("Zfull").sides == ("tla", "tilelang")
        assert spec.supports("tilelang")


def test_dupv_case_uses_lane_zero_and_last_lane_not_half_register():
    spec = cases.get_case("f32_dupv_mask")
    inp = spec.inputs_fn()
    ref = spec.reference_fn(inp)
    x = inp["x"]
    assert x[0] != x[-1] and x[0] != x[len(x) // 2] and x[-1] != x[len(x) // 2]
    np.testing.assert_array_equal(ref.outputs["low_full"], np.full_like(x, x[0]))
    np.testing.assert_array_equal(ref.outputs["high_full"], np.full_like(x, x[-1]))
    np.testing.assert_array_equal(ref.outputs["low_half"][32:], np.zeros(32, dtype=x.dtype))
    np.testing.assert_array_equal(ref.outputs["high_half"][32:], np.zeros(32, dtype=x.dtype))
    assert ref.active_masks["low_half"].any() and not ref.active_masks["low_half"].all()
    assert not ref.active_masks["low_none"].any()
    assert spec.output("high_half").zero_inactive
    assert spec.supports("tilelang")


def test_f32_mask_pattern_covers_all_eight_apis_for_arb_and_none():
    spec = cases.get_case("f32_mask_pattern")
    ref = spec.reference_fn(spec.inputs_fn())
    base = ("vdiv", "vdup", "vexp", "vmax", "vmaxs", "vmin",
            "vdups", "vdupv_low", "vdupv_high")
    for prefix in ("arb", "none"):
        for name in base:
            value = ref.outputs[f"{prefix}_{name}"]
            active = ref.active_masks[f"{prefix}_{name}"]
            assert np.all(value[~active] == 0), (prefix, name)
        assert not ref.active_masks[f"{prefix}_vdupv_high"].all() or prefix == "arb"
    # lane 0 and the last lane must differ so LOWEST/HIGHEST are distinguishable
    a = spec.inputs_fn()["a"]
    assert a[0] != a[-1]


def test_wide_dupv_reference_uses_lane_zero_last_and_index_mask():
    spec = cases.get_case("i64_u64_dupv_mask")
    inp = spec.inputs_fn()
    ref = spec.reference_fn(inp)
    x, idx, rem = inp["x"], inp["idx"], inp["rem"]
    assert x[0] not in (x[-1], x[len(x) // 2])
    # The arbitrary mask is signed-index built: rem > idx <=> lane % 3 != 0.
    expected_arb = (np.arange(len(x)) % 3) != 0
    assert int(expected_arb.sum()) == 42
    assert np.array_equal(rem > idx, expected_arb)
    assert "thr" not in inp
    for tag in ("i64", "u64"):
        for kind, active in (("prefix", np.arange(len(x)) < 32),
                             ("arb", expected_arb),
                             ("none", np.zeros(len(x), dtype=bool))):
            assert np.array_equal(ref.active_masks[f"{tag}_{kind}_low"], active)
            low = ref.outputs[f"{tag}_{kind}_low"]
            high = ref.outputs[f"{tag}_{kind}_high"]
            assert np.all(low[~active] == 0) and np.all(high[~active] == 0)
            if active.any():
                assert np.all(low[active] == low[active][0])
                assert np.all(high[active] == high[active][0])


def test_u64_output_contract_detects_unwritten_sentinels():
    # The launcher must seed u64 outputs with a non-zero sentinel, otherwise a
    # missed store would look like the expected inactive zero.
    sentinel = cases.U64_OUTPUT_SENTINEL
    assert sentinel != 0
    source = (SIM_DIR / "simd_harness" / "kernels_tla.py").read_text()
    u64_allocs = [ln for ln in source.splitlines()
                  if "torch.uint64" in ln and ("torch.full" in ln or "_dev(" in ln)]
    assert u64_allocs, "expected explicit torch.uint64 output allocations"
    for line in u64_allocs:
        assert "U64_OUTPUT_SENTINEL" in line or "17" in line, line

    for case_name in ("i64_u64_dupv_mask", "div_signed_wide_masks"):
        spec = cases.get_case(case_name)
        ref = spec.reference_fn(spec.inputs_fn())
        for name, value in ref.outputs.items():
            if not name.startswith("u64_") or value.dtype != np.uint64:
                continue
            active = ref.active_masks[name]
            if not active.any():
                continue
            unwritten = np.full_like(value, sentinel)
            cmp = compare_output(unwritten, value, spec.output(name), active_mask=active)
            assert not cmp["pass"], f"{case_name}/{name}: all-sentinel must fail"
            # The old zero-initialised buffer would also fail on active lanes,
            # but only after the active comparison is actually checked.
            zero_seeded = np.zeros_like(value)
            cmp_zero = compare_output(zero_seeded, value, spec.output(name),
                                      active_mask=active)
            assert not cmp_zero["pass"], f"{case_name}/{name}: zero-seeded missing writes must fail"
            written = value.copy()
            written[active] = value[active]
            cmp_ok = compare_output(written, value, spec.output(name),
                                    active_mask=active)
            assert cmp_ok["pass"], f"{case_name}/{name}: correctly written buffer must pass"


def test_diagnostic_u64_vcmp_mask_is_isolated_from_api_cases():
    spec = cases.get_case("diagnostic_u64_vcmp_mask")
    inp = spec.inputs_fn()
    ref = spec.reference_fn(inp)
    active = ref.active_masks["mask"]
    assert int(active.sum()) == 42
    assert np.array_equal(active, (np.arange(len(active)) % 3) != 0)
    assert np.array_equal(ref.outputs["mask"], active.astype(np.uint64))
    # It is not an API acceptance case and must not be used as vdupv evidence.
    assert spec.sides == ("tla",)
    assert "vdupv" not in spec.kernels.values()
    assert "not part of API acceptance" in spec.description
    assert "not" in spec.notes and "vdupv" in spec.notes


def _tail_case(dtype_name):
    name = "f32_long_tail" if dtype_name == "f32" else "f16_long_tail"
    return cases.get_case(name)


@pytest.mark.parametrize("dtype_name", ["f32", "f16"])
def test_tail_matrix_dimensions_guards_and_zeroing(dtype_name):
    spec = _tail_case(dtype_name)
    cap = cases.TAIL_F32_CAP if dtype_name == "f32" else cases.TAIL_F16_CAP
    chunk = cases.TAIL_F32_CHUNK if dtype_name == "f32" else cases.TAIL_F16_CHUNK
    ns = cases.TAIL_F32_NS if dtype_name == "f32" else cases.TAIL_F16_NS
    assert cap % chunk == 0
    assert max(ns) <= cap
    assert min(ns) == 1
    inp = spec.inputs_fn()
    ref = spec.reference_fn(inp)
    guard = cases.TAIL_GUARD
    sentinel = (np.float32(cases.TAIL_SENTINEL) if dtype_name == "f32"
                else np.float16(cases.TAIL_SENTINEL))
    for key in ("a", "b", "c"):
        arr = inp[key]
        assert arr.size == guard + cap + guard
        assert np.all(arr[:guard] == arr.dtype.type(123.5))
        assert np.all(arr[-guard:] == arr.dtype.type(123.5))
    if dtype_name == "f16":
        assert inp["idx"].dtype == np.int16
        assert inp["idx"].size == guard + cap + guard

    for n in ns:
        for op in cases.TAIL_OPS:
            row = ref.outputs[f"n{n}_{op}"]
            assert row.size == guard + cap + guard
            assert np.all(row[:guard] == sentinel), (n, op)
            assert np.all(row[-guard:] == sentinel), (n, op)
            inactive = row[guard + n:guard + cap]
            assert np.all(inactive == row.dtype.type(0.0)), (n, op)
        for op in cases.TAIL_MASKED_OPS:
            row = ref.outputs[f"n{n}_masked_{op}"]
            assert np.all(row[:guard] == sentinel)
            assert np.all(row[-guard:] == sentinel)
            assert np.all(row[guard + n:guard + cap] == sentinel), (n, op)


@pytest.mark.parametrize("dtype_name", ["f32", "f16"])
def test_tail_active_values_are_position_unique_and_vdupv_is_per_chunk(dtype_name):
    spec = _tail_case(dtype_name)
    cap = cases.TAIL_F32_CAP if dtype_name == "f32" else cases.TAIL_F16_CAP
    chunk = cases.TAIL_F32_CHUNK if dtype_name == "f32" else cases.TAIL_F16_CHUNK
    ns = cases.TAIL_F32_NS if dtype_name == "f32" else cases.TAIL_F16_NS
    guard = cases.TAIL_GUARD
    inp = spec.inputs_fn()
    data_a = inp["a"][guard:guard + cap]
    ref = spec.reference_fn(inp)
    for n in ns:
        active = slice(guard, guard + n)
        for op in ("vmax", "vmin", "vmaxs", "vdiv", "vexp"):
            got = ref.outputs[f"n{n}_{op}"][active]
            if op == "vexp" and dtype_name == "f16":
                got = got[np.isfinite(got)]
            assert len(np.unique(got)) == len(got), (n, op)
        low = ref.outputs[f"n{n}_vdupv_low"][guard:guard + cap]
        high = ref.outputs[f"n{n}_vdupv_high"][guard:guard + cap]
        for start in range(0, cap, chunk):
            stop = min(start + chunk, n)
            if stop <= start:
                continue
            assert np.all(low[start:stop] == low[start])
            assert np.all(high[start:stop] == high[start])
        # For N inside the first chunk the highest-lane reference is the
        # chunk's last lane, not lane N-1 and not the whole-buffer last lane.
        if n < chunk:
            assert high[0] == data_a[chunk - 1]
            assert high[0] != data_a[max(n - 1, 0)]


def test_tail_oracle_catches_masked_store_and_arithmetic_tail_bugs():
    spec = cases.get_case("f32_long_tail")
    ref = spec.reference_fn(spec.inputs_fn())
    guard = cases.TAIL_GUARD
    n = 31
    # Arithmetic ALL store: an unwritten tail keeps the non-zero sentinel and
    # must fail the explicit +0 tail reference.
    name = f"n{n}_vmax"
    got = ref.outputs[name].copy()
    got[guard + n:guard + cases.TAIL_F32_CAP] = np.float32(cases.TAIL_SENTINEL)
    assert not compare_output(got, ref.outputs[name], spec.output(name))["pass"]
    # Masked store: an ALL-store bug writes zeros after N and must fail.
    mname = f"n{n}_masked_vmax"
    got = ref.outputs[mname].copy()
    got[guard + n:guard + cases.TAIL_F32_CAP] = np.float32(0.0)
    assert not compare_output(got, ref.outputs[mname], spec.output(mname))["pass"]
    # A missing masked store leaves the sentinel in the active lanes and fails.
    got = np.full_like(ref.outputs[mname], np.float32(cases.TAIL_SENTINEL))
    assert not compare_output(got, ref.outputs[mname], spec.output(mname))["pass"]


def test_tail_runtime_n_and_mask_construction_are_not_constexpr_prefixes():
    source = (SIM_DIR / "simd_harness" / "kernels_tla.py").read_text()
    assert "def tla_tail_f32(A_ptr, B_ptr, C_ptr, O_ptr, N, ROW0," in source
    assert "def tla_tail_f16(A_ptr, B_ptr, C_ptr, I_ptr, O_ptr, N, ROW0," in source
    assert "tla.simd.vdup(N, tl.int32)" in source
    assert 'tla.simd.vcmp(idx, nv, "lt")' in source
    assert 'tla.simd.vcmp(iv, nv, "lt")' in source
    # The tail cases must not express the dynamic tail with a power-of-two pge.
    tail_block = source[source.index("def tla_tail_f32"):source.index("# numpy <-> torch adapters")]
    assert "pge(" not in tail_block
    assert "vsel(" not in tail_block
    for spec_name in ("f32_long_tail", "f16_long_tail"):
        spec = cases.get_case(spec_name)
        assert "runtime N" in spec.mask
        assert "vcmp" in spec.mask
        assert "pge" in spec.mask  # states that pge prefixes are not used


def test_signed_wide_div_reference_truncates_and_zeroes_inactive():
    spec = cases.get_case("div_signed_wide_masks")
    inp = spec.inputs_fn()
    ref = spec.reference_fn(inp)
    wide_arb = (np.arange(len(inp["a64"])) % 3) != 0
    assert int(wide_arb.sum()) == 42
    for tag in ("i16", "i32", "i64", "u64"):
        for kind in ("prefix", "arb", "none"):
            name = f"{tag}_{kind}"
            value = ref.outputs[name]
            active = ref.active_masks[name]
            assert np.all(value[~active] == 0), name
    # i64/u64 keep the same signed-index mask; no u64 data compare is involved.
    for tag in ("i64", "u64"):
        assert np.array_equal(ref.active_masks[f"{tag}_arb"], wide_arb)
    assert np.array_equal(inp["rem64"] > inp["idx64"], wide_arb)
    # signed truncation: -7 / 3 is -2, not numpy's floor -3
    a, b = inp["a32"], inp["b32"]
    hit = np.nonzero((a < 0) & (b > 0))[0][0]
    expected = -((-int(a[hit])) // int(b[hit]))
    assert int(ref.outputs["i32_prefix"][hit]) == expected or not ref.active_masks["i32_prefix"][hit]


def test_nan_inf_reference_policy_is_explicit():
    arrays = cases.get_case("f32_nan_inf").inputs_fn()
    ref = cases.get_case("f32_nan_inf").reference_fn(arrays).outputs
    # max of +NaN and 1.0 is a NaN under the declared propagate policy
    assert np.isnan(ref["vmax"][0])
    # max(+0, -0) is +0, min(+0, -0) is -0
    assert ref["vmaxs_0"][5].view(np.uint32) == 0x00000000
    assert ref["vmin"][4].view(np.uint32) == 0x80000000


# ---------------------------------------------------------------------------
# CLI: list, invalid arguments, exit codes, JSON
# ---------------------------------------------------------------------------


def test_cli_list_json_contains_status_and_rules():
    proc = _run("--list")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    names = {c["name"]: c for c in payload["cases"]}
    assert {"smoke", "f32_combo", "f32_nan_inf", "f32_multi_block"} <= set(names)
    assert names["f32_combo"]["tier"] == "implemented"
    assert names["f32_nan_inf"]["tier"] == "implemented"
    assert names["f32_nan_inf"]["implemented"] is True
    assert names["f32_nan_inf"]["sides"] == ["tla"]
    assert names["fp8_vdup_device"]["tier"] == "suggested_not_implemented"
    assert names["fp8_vdup_device"]["implemented"] is False
    combo = names["f32_combo"]
    assert combo["outputs"][0]["mode"] in ("exact", "close")
    assert any(o["zero_inactive"] for o in combo["outputs"])


def test_cli_invalid_arguments_exit_2():
    assert _run("--case", "does_not_exist", "--side", "tla", "--output", "/tmp/x").returncode == 2
    assert _run("--case", "f32_combo", "--side", "nope", "--output", "/tmp/x").returncode == 2
    assert _run("--case", "fp8_vdup_device", "--side", "tla", "--output", "/tmp/x").returncode == 2
    assert _run("--case", "f32_nan_inf", "--side", "tilelang", "--output", "/tmp/x").returncode == 2
    assert _run("--case", "smoke").returncode == 2


def test_cli_self_test_serialisation_and_failure_exits(tmp_path):
    ok = _run("--case", "smoke", "--side", "tla", "--output", str(tmp_path),
              "--executor", "reference")
    assert ok.returncode == 0, ok.stderr
    result_path = tmp_path / "tla" / "smoke" / "result.json"
    result = json.loads(result_path.read_text())
    assert result["status"] == "SELF_TEST_PASS"
    assert result["numerical_pass"] is True
    assert result["target"] == "npu"
    assert "capability" in result["versions"]
    assert result["outputs"]["out"]["rule"]["mode"] == "exact"
    got = tensor_from_json(result["outputs"]["out"]["got"])
    ref = tensor_from_json(result["outputs"]["out"]["reference"])
    assert array_digest(got) == array_digest(ref)

    bad = _run("--case", "int_bounds", "--side", "tla", "--output", str(tmp_path),
               "--executor", "zero")
    assert bad.returncode == 1, bad.stderr
    bad_result = json.loads((tmp_path / "tla" / "int_bounds" / "result.json").read_text())
    assert bad_result["status"] == "NUMERICAL_FAIL"
    assert bad_result["failure_type"] == "numerical"
    assert bad_result["outputs"]["i_max"]["result"]["first_fail"] is not None

    crash = _run("--case", "smoke", "--side", "tla", "--output", str(tmp_path),
                 "--executor", "teardown")
    assert crash.returncode == 3, crash.stderr
    crash_result = json.loads((tmp_path / "tla" / "smoke" / "result.json").read_text())
    assert crash_result["status"] == "TEARDOWN_CRASH"
    assert crash_result["numerical_pass"] is True
    assert crash_result["teardown_ok"] is False


def test_result_json_preserves_nan_payload(tmp_path):
    proc = _run("--case", "f32_nan_inf", "--side", "tla", "--output", str(tmp_path),
                "--executor", "reference")
    assert proc.returncode == 0, proc.stderr
    result = json.loads((tmp_path / "tla" / "f32_nan_inf" / "result.json").read_text())
    # input a starts with 0x7fc00001, a payload that a float JSON round-trip would lose
    from simd_harness.encoding import load_npz
    inputs = load_npz(tmp_path / "tla" / "f32_nan_inf" / "inputs.npz")
    assert inputs["in__a"].view(np.uint32)[0] == 0x7FC00001
    assert inputs["in__b"].view(np.uint32)[6] == 0x7FC00003
    arr = tensor_from_json(result["outputs"]["vmax"]["reference"])
    assert arr.view(np.uint32)[0] in (0x7FC00000, 0x7FC00001, 0x7FC00003)


def test_standalone_comparator(tmp_path):
    for side in ("tla", "tilelang"):
        proc = _run("--case", "smoke", "--side", side, "--output", str(tmp_path),
                    "--executor", "reference")
        assert proc.returncode == 0, proc.stderr
    comp = subprocess.run(
        [sys.executable, str(SIM_DIR / "compare_results.py"),
         "--baseline", str(tmp_path / "tla"), "--candidate", str(tmp_path / "tilelang"),
         "--json", str(tmp_path / "cmp.json")],
        cwd=str(SIM_DIR), capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(SIM_DIR)})
    assert comp.returncode == 0, comp.stderr + comp.stdout
    report = json.loads((tmp_path / "cmp.json").read_text())
    assert report["pass"] is True

    zero = _run("--case", "smoke", "--side", "tla", "--output", str(tmp_path),
                "--executor", "zero")
    assert zero.returncode == 1
    comp2 = subprocess.run(
        [sys.executable, str(SIM_DIR / "compare_results.py"),
         "--baseline", str(tmp_path / "tla"), "--candidate", str(tmp_path / "tilelang")],
        cwd=str(SIM_DIR), capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(SIM_DIR)})
    assert comp2.returncode == 1


def test_harness_has_no_local_env_or_removed_entry_point_references():
    offenders = []
    for relative in ("simd_harness/__init__.py", "simd_harness/spec.py",
                     "simd_harness/cases.py", "simd_harness/runner.py",
                     "simd_harness/capabilities.py", "simd_harness/reference.py",
                     "simd_harness/inputs.py", "simd_harness/encoding.py",
                     "simd_harness/kernels_tla.py", "simd_harness/kernels_tilelang.py",
                     "simd_harness/backends/tla_backend.py",
                     "simd_harness/backends/tilelang_backend.py",
                     "compare_results.py"):
        text = (SIM_DIR / relative).read_text()
        for needle in ("/home/", "site-packages", "tl312-venv",
                       "import msprof", "from msprof",
                       "run_all.sh", "run_hardware.py",
                       "simulated_sides", "legacy_module",
                       "simulator_passed", "new_executable_not_on_device",
                       "RESULTS.md", "HARDWARE_VALIDATION.md"):
            if needle in text:
                offenders.append((relative, needle))
    assert not offenders, offenders
