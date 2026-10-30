"""CPU references and comparison rules (no device, no profiler).

The comparison result keeps enough diagnostics for a hardware operator:

* the first failing flat index and its unravelled index,
* got/reference values rendered without relying on JSON floats,
* raw bit patterns for float lanes (NaN payload preserved),
* the maximum absolute error over finite lanes,
* a separate bitwise zeroing result for inactive mask lanes.

NaN policy is explicit per output:

``forbid``         no NaN is allowed in either side
``position``       NaN positions must match; payload is recorded but not required
``propagate``      the reference is computed with NaN propagation (same check as position)
``canonical_bits`` NaN positions *and* raw bits must match

Signed zero is explicit too: ``signed_zero="ieee"`` requires the sign bit of a
zero result to match the reference (``any`` only compares numeric equality).
"""

from __future__ import annotations

import math

import numpy as np

from .spec import OutputSpec


def _is_float(dtype: np.dtype) -> bool:
    return dtype.kind == "f"


def _value_repr(value) -> str:
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    return str(value)


def _bits_int(value) -> int:
    arr = np.asarray([value])
    if arr.dtype.kind in "fiu":
        return int(arr.view(np.dtype(f"u{arr.dtype.itemsize}"))[0])
    return int(value)


def _zero_bits_ok(array: np.ndarray) -> np.ndarray:
    """Elementwise: True when the element is a bitwise all-zero pattern."""
    arr = np.ascontiguousarray(array)
    raw = arr.view(np.uint8).reshape(arr.shape + (arr.dtype.itemsize,))
    return (raw == 0).all(axis=-1)


def _nan_bits(array: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(array)
    return arr.view(np.dtype(f"u{arr.dtype.itemsize}"))


def _first_bad_index(mask: np.ndarray):
    if not mask.any():
        return None
    flat = int(np.argmax(mask.reshape(-1)))
    return flat, tuple(int(i) for i in np.unravel_index(flat, mask.shape))


def _json_float(value):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return _value_repr(value)
    return value


def ref_maximum(a, b):
    """IEEE-754 maximum with explicit NaN propagation and +0 for max(+-0)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        res = np.maximum(a, b)
    nan = np.isnan(a) | np.isnan(b)
    if nan.any():
        res = np.where(nan, np.float32(np.nan), res)
    both_zero = (a == 0) & (b == 0)
    if both_zero.any():
        res = np.where(both_zero, np.float32(0.0), res)
    return res.astype(np.float32, copy=False)


def ref_minimum(a, b):
    """IEEE-754 minimum with explicit NaN propagation and -0 for min(+-0)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        res = np.minimum(a, b)
    nan = np.isnan(a) | np.isnan(b)
    if nan.any():
        res = np.where(nan, np.float32(np.nan), res)
    both_zero = (a == 0) & (b == 0)
    if both_zero.any():
        res = np.where(both_zero, np.float32(-0.0), res)
    return res.astype(np.float32, copy=False)


def ref_maximum_scalar(a, scalar):
    return ref_maximum(a, np.full_like(np.asarray(a, dtype=np.float32), np.float32(scalar)))


def ref_minimum_scalar(a, scalar):
    return ref_minimum(a, np.full_like(np.asarray(a, dtype=np.float32), np.float32(scalar)))


def _word_equal(got, ref):
    g = np.ascontiguousarray(got).view(np.uint8).reshape(got.shape + (-1,))
    r = np.ascontiguousarray(ref).view(np.uint8).reshape(ref.shape + (-1,))
    return (g == r).all(axis=-1)


def _compare_exact(got, ref, spec: OutputSpec) -> dict:
    ref_dtype = ref.dtype
    is_float = _is_float(ref_dtype)
    nan_ref = np.isnan(ref) if is_float else np.zeros(ref.shape, dtype=bool)
    nan_got = np.isnan(got) if is_float else np.zeros(got.shape, dtype=bool)
    nan_mismatch = nan_ref != nan_got
    words_equal = _word_equal(got, ref)

    if not is_float:
        bad = ~words_equal
    elif spec.nan_policy == "forbid":
        bad = (~words_equal) | nan_ref | nan_got
    elif spec.nan_policy == "canonical_bits":
        bad = ~words_equal
    else:  # position / propagate: payload is recorded but not required
        non_nan = ~(nan_ref | nan_got)
        bad = nan_mismatch | (non_nan & ~words_equal)

    return {
        "bad": bad,
        "nan_mismatch": int(nan_mismatch.sum()) if is_float else 0,
        "max_abs_error": None,
    }


def _compare_close(got, ref, spec: OutputSpec) -> dict:
    ref_dtype = ref.dtype
    nan_ref = np.isnan(ref)
    nan_got = np.isnan(got)
    if spec.nan_policy == "forbid":
        bad = nan_ref | nan_got
    elif spec.nan_policy in ("position", "propagate"):
        bad = nan_ref != nan_got
    else:  # canonical_bits
        bad = (nan_ref != nan_got) | (nan_ref & (_nan_bits(got) != _nan_bits(ref)))

    finite = np.isfinite(ref) & ~nan_ref
    if finite.any():
        close = np.isclose(got, ref, rtol=spec.rtol, atol=spec.atol, equal_nan=False)
        bad = bad | (finite & ~close)
    # inf == inf is not "finite"; compare it explicitly.
    inf_ref = np.isinf(ref)
    if inf_ref.any():
        bad = bad | (inf_ref & (got != ref))

    if spec.signed_zero == "ieee":
        zero = (ref == 0) & ~nan_ref
        if zero.any():
            bad = bad | (zero & (np.signbit(got) != np.signbit(ref)))

    max_abs = None
    both_finite = np.isfinite(got) & np.isfinite(ref)
    if both_finite.any():
        diff = np.abs(got[both_finite].astype(np.float64) - ref[both_finite].astype(np.float64))
        max_abs = float(np.max(diff)) if diff.size else 0.0
    return {
        "bad": bad,
        "nan_mismatch": int((nan_ref != nan_got).sum()),
        "max_abs_error": max_abs,
    }


def compare_output(got, ref, spec: OutputSpec, active_mask=None) -> dict:
    got = np.asarray(got)
    ref = np.asarray(ref)
    problems = []
    if got.shape != ref.shape:
        return {
            "pass": False,
            "problems": [f"shape mismatch got={got.shape} ref={ref.shape}"],
            "first_fail": None,
            "nan_mismatch": None,
            "max_abs_error": None,
            "zeroing": None,
        }
    if got.dtype != ref.dtype:
        problems.append(f"dtype mismatch got={got.dtype} ref={ref.dtype}")
        return {
            "pass": False,
            "problems": problems,
            "first_fail": None,
            "nan_mismatch": None,
            "max_abs_error": None,
            "zeroing": None,
        }

    if spec.mode == "exact":
        cmp = _compare_exact(got, ref, spec)
    else:
        cmp = _compare_close(got, ref, spec)
    bad = cmp["bad"].copy()

    zeroing = None
    if spec.zero_inactive:
        if active_mask is None:
            problems.append("zero_inactive requested without an active mask")
            bad[...] = True
        else:
            active = np.asarray(active_mask, dtype=bool)
            if active.shape != got.shape:
                problems.append(f"active mask shape {active.shape} != output {got.shape}")
                bad[...] = True
            else:
                inactive = ~active
                inactive_bad = inactive & ~_zero_bits_ok(got)
                zeroing = {
                    "inactive_lanes": int(inactive.sum()),
                    "non_zero_inactive": int(inactive_bad.sum()),
                    "pass": bool(not inactive_bad.any()),
                }
                if inactive_bad.any():
                    bad = bad | inactive_bad

    first = _first_bad_index(bad)
    first_fail = None
    if first is not None:
        flat, index = first
        g = got[index]
        r = ref[index]
        first_fail = {
            "flat_index": flat,
            "index": list(index),
            "got": _value_repr(g),
            "ref": _value_repr(r),
            "got_bits": hex(_bits_int(g)),
            "ref_bits": hex(_bits_int(r)),
        }
    ok = not bad.any() and not problems
    return {
        "pass": bool(ok),
        "problems": problems,
        "first_fail": first_fail,
        "nan_mismatch": cmp["nan_mismatch"],
        "max_abs_error": _json_float(cmp["max_abs_error"]),
        "zeroing": zeroing,
    }
