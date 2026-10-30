"""Registry of SIMD API cases.

Each case carries its own implementation status, which describes what the
harness can do with it -- not what some machine happened to report:

``implemented``
    Deterministic inputs, a CPU reference and kernels for the listed sides are
    present.  The case can run against the host reference (inputs/reference
    only) or against a real TLA/TileLang backend.

``suggested_not_implemented``
    Evaluated idea without inputs/reference/kernels yet (usually because the
    AVE lowering or the comparison policy is unresolved).  ``--list`` reports
    it; ``--case`` refuses it with a clear error instead of silently skipping.
"""

from __future__ import annotations

import numpy as np

from . import inputs as I
from .reference import (ref_maximum, ref_maximum_scalar, ref_minimum,
                        ref_minimum_scalar)
from .spec import (TIER_IMPLEMENTED, TIER_SUGGESTED, CaseSpec, OutputSpec,
                   RefResult)

BLK = 64
F16_BLK = 128
U_HIGH = 0x80000000

# Non-zero sentinel used to initialise 64-bit output buffers.  A missing store
# (or a masked store that forgot an active lane) then differs from the
# reference instead of silently matching an initial zero.
U64_OUTPUT_SENTINEL = 17

# Long / tail matrix.  CAP is a whole number of register-sized chunks, so the
# unmasked vlds always reads within the legally padded input buffer.  TAIL_NS
# covers below / equal / just above one chunk and multi-chunk tails.
TAIL_SENTINEL = 17.0
TAIL_GUARD = 8
TAIL_F32_CHUNK = 64
TAIL_F32_CAP = 320          # 5 * 64, >= max(TAIL_F32_NS)
TAIL_F32_NS = (1, 31, 32, 63, 64, 65, 127, 128, 129, 257)
TAIL_F16_CHUNK = 128
TAIL_F16_CAP = 512          # 4 * 128, >= max(TAIL_F16_NS)
TAIL_F16_NS = (1, 63, 127, 128, 129, 255, 256, 257, 385)
# 8 public APIs; vdupv contributes two rows (lowest / highest lane).
TAIL_OPS = ("vdiv", "vmax", "vmin", "vmaxs", "vexp",
            "vdup", "vdups", "vdupv_low", "vdupv_high")
# Representative outputs that use an actual masked vsts, so every lane after
# the runtime N keeps the non-zero sentinel instead of being overwritten.
TAIL_MASKED_OPS = ("vmax", "vdiv", "vdupv_low", "vdupv_high")
TAIL_ROWS = len(TAIL_OPS) + len(TAIL_MASKED_OPS)


def _f32_outputs(names, mode="close", rtol=1e-6, atol=1e-6, **kw):
    return tuple(OutputSpec(name=n, dtype="<f4", mode=mode, rtol=rtol, atol=atol, **kw)
                 for n in names)


# ---------------------------------------------------------------------------
# 1. f32 six-API matrix, full + prefix mask
# ---------------------------------------------------------------------------


def _combo_inputs():
    a, b = I.exact_f32_grid(BLK)
    return {"a": a, "b": b}


def _combo_reference(inp):
    a, b = inp["a"], inp["b"]
    keep = np.arange(BLK) < 32

    def masked(value):
        return np.where(keep, value, np.float32(0.0)).astype(np.float32)

    full = {
        "OD": (a / b).astype(np.float32),
        "OU": np.full_like(a, np.float32(1.5)),
        "OE": np.exp(a).astype(np.float32),
        "OX": ref_maximum(a, b),
        "OS": ref_maximum_scalar(a, 0.0),
        "ON": ref_minimum(a, b),
    }
    part = {
        "PD": masked(a / b),
        "PU": masked(np.full_like(a, np.float32(1.5))),
        "PE": masked(np.exp(a).astype(np.float32)),
        "PX": masked(ref_maximum(a, b)),
        "PS": masked(ref_maximum_scalar(a, 0.0)),
        "PN": masked(ref_minimum(a, b)),
    }
    masks = {name: keep for name in part}
    return RefResult({**full, **part}, masks)


COMBO_OUTPUTS = (
    OutputSpec("OD", "<f4", "close", 1e-6, 1e-6),
    OutputSpec("OU", "<f4", "close", 1e-6, 1e-6),
    OutputSpec("OE", "<f4", "close", 1e-6, 1e-6),
    OutputSpec("OX", "<f4", "close", 1e-6, 1e-6),
    OutputSpec("OS", "<f4", "close", 1e-6, 1e-6, signed_zero="ieee"),
    OutputSpec("ON", "<f4", "close", 1e-6, 1e-6),
    OutputSpec("PD", "<f4", "close", 1e-6, 1e-6, zero_inactive=True),
    OutputSpec("PU", "<f4", "close", 1e-6, 1e-6, zero_inactive=True),
    OutputSpec("PE", "<f4", "close", 1e-6, 1e-6, zero_inactive=True),
    OutputSpec("PX", "<f4", "close", 1e-6, 1e-6, zero_inactive=True),
    OutputSpec("PS", "<f4", "close", 1e-6, 1e-6, signed_zero="ieee", zero_inactive=True),
    OutputSpec("PN", "<f4", "close", 1e-6, 1e-6, zero_inactive=True),
)


# ---------------------------------------------------------------------------
# 2. div special values + exp FTZ
# ---------------------------------------------------------------------------

DIV_A = [1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 3.0, 1e38, 1.0, -0.0, 5.0, -5.0, 0.0, 2.0, -3.0, 1.0]
DIV_B = [0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 2.0, 1e-38, 1.0, 1.0, -2.0, 2.0, 1.0, 0.0, 0.0, -1.0]
EXP_X = [0.0, 1.0, -1.0, -1.5, 2.0, -10.0, -80.0, -87.0, -87.3, -88.0, -90.0,
         -100.0, -103.0, -104.0, -110.0, -0.0, 5.0, -5.0]
# FTZ (flush-to-zero) semantics: every finite result whose magnitude is below
# the smallest normal f32 (FLT_MIN = 2**-126) is replaced by +0.  For exp(x)
# that boundary sits at x = ln(FLT_MIN) = -87.3365, so a result is normal iff
# x > -87.3365.  EXP_UNDERFLOW is a rounding of that boundary, chosen to fall
# between two probed lanes (normal at x=-87.3, subnormal at x=-88.0) so the
# reference does not depend on the exact underflow point of one implementation.
# The test intent is a policy check -- deeply underflowing lanes must be
# bitwise +0 while normal lanes keep the SFU accuracy budget used by
# ``exp_default``/``exp_ftz_true`` (rtol from the OutputSpec) -- not a measured
# device boundary.
EXP_UNDERFLOW = -87.4


def _special_inputs():
    a = I.f32(I.pad(DIV_A, BLK, 6.0))
    b = I.f32(I.pad(DIV_B, BLK, 2.0))
    x = I.f32(I.pad(EXP_X, BLK, 0.5))
    return {"a": a, "b": b, "x": x}


def _exp_ftz_true(x):
    with np.errstate(under="ignore"):
        e = np.exp(x).astype(np.float32)
    return np.where(x > np.float32(EXP_UNDERFLOW), e, np.float32(0.0)).astype(np.float32)


def _special_reference(inp):
    a, b, x = inp["a"], inp["b"], inp["x"]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        div = (a / b).astype(np.float32)
    with np.errstate(under="ignore", over="ignore"):
        exp_false = np.exp(x).astype(np.float32)
    out = {
        "div": div,
        "exp_default": _exp_ftz_true(x),
        "exp_ftz_true": _exp_ftz_true(x),
        "exp_ftz_false": exp_false,
    }
    return RefResult(out)


SPECIAL_OUTPUTS = (
    OutputSpec("div", "<f4", "exact", nan_policy="position", signed_zero="ieee",
               note="0 ulp; +-0/+-inf signs exact; NaN position required, payload recorded"),
    OutputSpec("exp_default", "<f4", "close", 1e-5, 0.0,
               sides=("tla",), note="TLA default precision keeps the FTZ (SFU) path"),
    OutputSpec("exp_ftz_true", "<f4", "close", 1e-5, 0.0,
               sides=("tla", "tilelang"), note="FTZ mode; subnormal results flush to +0"),
    OutputSpec("exp_ftz_false", "<f4", "close", 1e-4, 0.0,
               sides=("tilelang",), note="TileLang software path keeps subnormals (1 ulp budget); the TLA API exposes only the FTZ path"),
)


# ---------------------------------------------------------------------------
# 3. int32 / uint32 boundary
# ---------------------------------------------------------------------------

INT_MIN = -2147483648
INT_MAX = 2147483647
U_MAX = 0xFFFFFFFF
I_VALS = [0, -1, INT_MIN, INT_MAX, 1, -2, 12345, -12345, 7, -7, INT_MAX, INT_MIN, 100, -100, 2, -3]
J_VALS = [1, 0, INT_MAX, INT_MIN, -1, 1, -12345, 12345, -7, 7, INT_MIN, INT_MAX, -100, 100, 3, -2]
U_VALS = [U_HIGH, U_MAX, 0, 1, 0x7FFFFFFF, U_HIGH + 1, 12345, U_MAX - 1, 7, 0x7FFFFFFF,
          0xFFFFFFFE, U_HIGH, 100, 0, 2, U_MAX]
V_VALS = [1, 0, U_MAX, U_HIGH, 0x80000001, 0x7FFFFFFF, U_MAX, 12345, 0xFFFFFFFF,
          U_HIGH, 1, U_HIGH + 1, 0, 100, 3, U_HIGH]


def _int_inputs():
    return {
        "i": np.asarray(I.pad(I_VALS, BLK, 5), dtype=np.int32),
        "j": np.asarray(I.pad(J_VALS, BLK, 6), dtype=np.int32),
        "u": np.asarray(I.pad(U_VALS, BLK, 5), dtype=np.uint32),
        "v": np.asarray(I.pad(V_VALS, BLK, 6), dtype=np.uint32),
    }


def _int_reference(inp):
    i, j, u, v = inp["i"], inp["j"], inp["u"], inp["v"]
    out = {
        "i_max": np.maximum(i, j),
        "i_min": np.minimum(i, j),
        "i_maxs": np.maximum(i, np.int32(-1)),
        "u_max": np.maximum(u, v),
        "u_min": np.minimum(u, v),
        "u_maxs": np.maximum(u, np.uint32(U_HIGH)),
    }
    return RefResult(out)


INT_OUTPUTS = (
    OutputSpec("i_max", "<i4"),
    OutputSpec("i_min", "<i4"),
    OutputSpec("i_maxs", "<i4"),
    OutputSpec("u_max", "<u4"),
    OutputSpec("u_min", "<u4"),
    OutputSpec("u_maxs", "<u4"),
)


# ---------------------------------------------------------------------------
# 4. vdup int32 / bf16
# ---------------------------------------------------------------------------

BF16_ONE_AND_HALF = 0x3FC0


def _vdup_inputs():
    return {}


def _bf16_array(bits):
    return np.asarray(bits, dtype=np.uint16)


def _vdup_reference(_inp):
    full_i = np.full((BLK,), 7, dtype=np.int32)
    part_i = np.concatenate([np.full(32, 7, dtype=np.int32), np.zeros(BLK - 32, dtype=np.int32)])
    full_b = np.full((F16_BLK,), BF16_ONE_AND_HALF, dtype=np.uint16)
    part_b = np.concatenate([np.full(32, BF16_ONE_AND_HALF, dtype=np.uint16),
                             np.zeros(F16_BLK - 32, dtype=np.uint16)])
    active32 = np.arange(BLK) < 32
    active16 = np.arange(F16_BLK) < 32
    return RefResult(
        {"OI": full_i, "OP": part_i, "OB": full_b, "OQ": part_b},
        {"OP": active32, "OQ": active16},
    )


VDUP_OUTPUTS = (
    OutputSpec("OI", "<i4"),
    OutputSpec("OP", "<i4", zero_inactive=True),
    OutputSpec("OB", "<u2", note="bf16 raw bits"),
    OutputSpec("OQ", "<u2", zero_inactive=True, note="bf16 raw bits"),
)


# ---------------------------------------------------------------------------
# 4b. unsigned 16/32-bit vdiv: full / prefix half / all-false masks
# ---------------------------------------------------------------------------

U16_DIV_BLK = 128


def _u_div_inputs(blk, dtype):
    lane = np.arange(blk, dtype=np.uint64)
    a = ((lane * 7 + 3) % 251 + 1).astype(dtype)
    b = ((lane * 11 + 5) % 17 + 1).astype(dtype)
    return {"a": a, "b": b}


def _u_div_reference(inp):
    a, b = inp["a"], inp["b"]
    q = np.floor_divide(a, b).astype(a.dtype)
    half = np.arange(a.size) < (a.size // 2)
    none = np.zeros(a.size, dtype=bool)
    half_out = np.where(half, q, 0).astype(a.dtype)
    none_out = np.zeros_like(q)
    out = {
        "Zfull": q.copy(), "Zhalf": half_out.copy(), "Znone": none_out.copy(),
        "Mfull": q.copy(), "Mhalf": half_out.copy(), "Mnone": none_out.copy(),
    }
    masks = {
        "Zhalf": half, "Znone": none,
        "Mhalf": half, "Mnone": none,
    }
    return RefResult(out, masks)


def _u_div_outputs(dtype):
    specs = []
    for prefix, sides in (("Z", ("tla", "tilelang")), ("M", ("tla", ))):
        for mask, zero_inactive in (("full", False), ("half", True), ("none", True)):
            specs.append(OutputSpec(f"{prefix}{mask}", dtype, "exact",
                                    zero_inactive=zero_inactive,
                                    sides=sides,
                                    note="stored with the ALL store mask"))
    return tuple(specs)


# ---------------------------------------------------------------------------
# 4c. vdups / vdupv lowest/highest lane, full / prefix half / all-false
# ---------------------------------------------------------------------------

def _dupv_inputs():
    return {"x": (np.arange(BLK, dtype=np.float32) + np.float32(1.0))}


def _dupv_reference(inp):
    x = inp["x"]
    active_half = np.arange(BLK) < 32
    active_none = np.zeros(BLK, dtype=bool)

    def full(value):
        return np.full_like(x, np.float32(value))

    def masked(value, active):
        return np.where(active, np.float32(value), np.float32(0.0)).astype(np.float32)

    low, high, scalar = np.float32(x[0]), np.float32(x[-1]), np.float32(1.5)
    out = {
        "scalar_full": full(scalar),
        "low_full": full(low),
        "high_full": full(high),
        "scalar_half": masked(scalar, active_half),
        "low_half": masked(low, active_half),
        "high_half": masked(high, active_half),
        "scalar_none": masked(scalar, active_none),
        "low_none": masked(low, active_none),
        "high_none": masked(high, active_none),
    }
    masks = {
        "scalar_half": active_half,
        "low_half": active_half,
        "high_half": active_half,
        "scalar_none": active_none,
        "low_none": active_none,
        "high_none": active_none,
    }
    return RefResult(out, masks)


DUPV_OUTPUTS = (
    OutputSpec("scalar_full", "<f4", "exact"),
    OutputSpec("low_full", "<f4", "exact"),
    OutputSpec("high_full", "<f4", "exact"),
    OutputSpec("scalar_half", "<f4", "exact", zero_inactive=True),
    OutputSpec("low_half", "<f4", "exact", zero_inactive=True),
    OutputSpec("high_half", "<f4", "exact", zero_inactive=True),
    OutputSpec("scalar_none", "<f4", "exact", zero_inactive=True),
    OutputSpec("low_none", "<f4", "exact", zero_inactive=True),
    OutputSpec("high_none", "<f4", "exact", zero_inactive=True),
)


# ---------------------------------------------------------------------------
# 5. minimal smoke
# ---------------------------------------------------------------------------


def _smoke_inputs():
    a, b = I.exact_f32_grid(BLK, mul_a=13, mul_b=7)
    return {"a": a, "b": b}


def _smoke_reference(inp):
    return RefResult({"out": ref_maximum(inp["a"], inp["b"])})


# ---------------------------------------------------------------------------
# 6. arbitrary non-prefix mask + all-false mask
# ---------------------------------------------------------------------------

ARB_LANES = [0, 1, 3, 4, 7, 8, 15, 16, 23, 31, 32, 33, 47, 48, 55, 62, 63]


def _mask_inputs():
    a, b = I.exact_f32_grid(BLK)
    m = np.zeros(BLK, dtype=np.float32)
    m[ARB_LANES] = 1.0
    return {"a": a, "b": b, "m_arb": m, "m_none": np.zeros(BLK, dtype=np.float32)}


def _apply_mask(value, active):
    return np.where(active, value, np.float32(0.0)).astype(np.float32)


def _mask_reference(inp):
    a, b, m = inp["a"], inp["b"], inp["m_arb"]
    active = m != 0
    active_none = inp["m_none"] != 0
    out, masks = {}, {}
    ops = {
        "vdiv": (a / b).astype(np.float32),
        "vdup": np.full_like(a, np.float32(1.5)),
        "vexp": np.exp(a).astype(np.float32),
        "vmax": ref_maximum(a, b),
        "vmaxs": ref_maximum_scalar(a, 0.0),
        "vmin": ref_minimum(a, b),
        # arbitrary/all-false coverage for the remaining two public APIs
        "vdups": np.full_like(a, np.float32(1.5)),
        "vdupv_low": np.full_like(a, np.float32(a[0])),
        "vdupv_high": np.full_like(a, np.float32(a[-1])),
    }
    for name, value in ops.items():
        out[f"arb_{name}"] = _apply_mask(value, active)
        out[f"none_{name}"] = _apply_mask(value, active_none)
        masks[f"arb_{name}"] = active
        masks[f"none_{name}"] = active_none
    return RefResult(out, masks)


def _mask_outputs():
    specs = []
    for prefix, active_note in (("arb", "arbitrary non-prefix mask"),
                                ("none", "all-false mask")):
        specs.extend([
            OutputSpec(f"{prefix}_vdiv", "<f4", "close", 1e-6, 1e-6, zero_inactive=True, note=active_note),
            OutputSpec(f"{prefix}_vdup", "<f4", "exact", zero_inactive=True, note=active_note),
            OutputSpec(f"{prefix}_vexp", "<f4", "close", 1e-5, 0.0, zero_inactive=True, note=active_note),
            OutputSpec(f"{prefix}_vmax", "<f4", "exact", zero_inactive=True, note=active_note),
            OutputSpec(f"{prefix}_vmaxs", "<f4", "exact", signed_zero="ieee",
                       zero_inactive=True, note=active_note),
            OutputSpec(f"{prefix}_vmin", "<f4", "exact", zero_inactive=True, note=active_note),
            OutputSpec(f"{prefix}_vdups", "<f4", "exact", zero_inactive=True, note=active_note),
            OutputSpec(f"{prefix}_vdupv_low", "<f4", "exact", zero_inactive=True, note=active_note),
            OutputSpec(f"{prefix}_vdupv_high", "<f4", "exact", zero_inactive=True, note=active_note),
        ])
    return tuple(specs)


# ---------------------------------------------------------------------------
# 5b. signed i16/i32 + wide i64/u64 division under prefix/arbitrary/all-false
# ---------------------------------------------------------------------------

I16_DIV_BLK = 128


def _trunc_div(a, b):
    """Integer division truncated toward zero (numpy // floors for negatives)."""
    q = np.abs(a) // np.abs(b)
    signed = np.issubdtype(a.dtype, np.signedinteger)
    if signed:
        return np.where((a < 0) ^ (b < 0), -q, q).astype(a.dtype)
    return q.astype(a.dtype)


def _div_sw_inputs():
    lane16 = np.arange(I16_DIV_BLK, dtype=np.int64)
    a16 = ((lane16 * 37) % 401 - 200).astype(np.int16)
    b16 = ((lane16 * 11) % 29 + 1).astype(np.int16)
    lane32 = np.arange(BLK, dtype=np.int64)
    a32 = ((lane32 * 7919) % 2001 - 1000).astype(np.int32)
    b32 = ((lane32 * 13) % 37 + 1).astype(np.int32)
    a64 = ((lane32 * 104729) % 200000 - 100000).astype(np.int64)
    b64 = ((lane32 * 17) % 97 + 1).astype(np.int64)
    au64 = ((lane32 * 65537) % 1000000 + 1).astype(np.uint64)
    bu64 = ((lane32 * 19) % 127 + 1).astype(np.uint64)
    idx64 = lane32.copy()
    rem64 = lane32 + ((lane32 % 3) != 0).astype(np.int64)
    return {"a16": a16, "b16": b16, "a32": a32, "b32": b32,
            "a64": a64, "b64": b64, "au64": au64, "bu64": bu64,
            "idx64": idx64, "rem64": rem64}


def _div_masks_for(a, b, prefix_n, arb=None):
    lanes = np.arange(a.size)
    return {
        "prefix": lanes < prefix_n,
        "arb": a > b if arb is None else arb,
        "none": np.zeros(a.size, dtype=bool),
    }


def _div_sw_reference(inp):
    out, masks = {}, {}
    wide_arb = inp["rem64"] > inp["idx64"]  # signed index-only mask, 42 lanes
    for tag, akey, bkey, prefix, arb in (
            ("i16", "a16", "b16", I16_DIV_BLK // 2, None),
            ("i32", "a32", "b32", BLK // 2, None),
            # i64/u64 share the signed-index mask; no u64 data compare.
            ("i64", "a64", "b64", BLK // 2, wide_arb),
            ("u64", "au64", "bu64", BLK // 2, wide_arb)):
        a, b = inp[akey], inp[bkey]
        q = _trunc_div(a, b)
        for kind, active in _div_masks_for(a, b, prefix, arb).items():
            name = f"{tag}_{kind}"
            out[name] = np.where(active, q, np.zeros_like(q)).astype(q.dtype)
            masks[name] = active
    return RefResult(out, masks)


def _div_sw_outputs():
    specs = []
    for tag, dtype in (("i16", "<i2"), ("i32", "<i4"), ("i64", "<i8"), ("u64", "<u8")):
        for kind, note in (("prefix", "prefix half mask"),
                           ("arb", "signed-index arbitrary mask" if tag in ("i64", "u64")
                            else "arbitrary vcmp(a>b) signed mask"),
                           ("none", "all-false mask")):
            specs.append(OutputSpec(f"{tag}_{kind}", dtype, "exact",
                                    zero_inactive=True, sides=("tla",), note=note))
    return tuple(specs)


# ---------------------------------------------------------------------------
# 5c. i64/u64 vdupv lowest/highest under prefix/arbitrary/all-false
# ---------------------------------------------------------------------------

def _wide_dupv_inputs():
    lanes = np.arange(BLK, dtype=np.int64)
    x = (lanes * 0x010101010101 + 0x123456789ABC).astype(np.int64)
    x[-1] = 0x7FFFFFFFFFFFFFF
    # The arbitrary mask is built from a signed 64-bit index, never from the
    # u64 payload: idx == lane and rem == lane + 1 for lane % 3 != 0, so
    # ``rem > idx`` is a signed compare with exactly 42 active lanes.  This
    # keeps the API case independent of the unverified unsigned vcmp path.
    idx = lanes.copy()
    rem = lanes + ((lanes % 3) != 0).astype(np.int64)
    return {"x": x, "idx": idx, "rem": rem}


def _wide_dupv_reference(inp):
    x, idx, rem = inp["x"], inp["idx"], inp["rem"]
    xu = x.view(np.uint64)
    lanes = np.arange(BLK)
    # Signed i64 GT on index-derived vectors (rem > idx <=> lane % 3 != 0).
    arb = rem > idx
    masks = {
        "prefix": lanes < 32,
        "arb": arb,
        "none": np.zeros(BLK, dtype=bool),
    }
    out, active_masks = {}, {}
    for kind, active in masks.items():
        for tag, value, dtype in (("i64", np.int64(x[0]), np.int64),
                                  ("u64", np.uint64(xu[0]), np.uint64)):
            high = np.int64(x[-1]) if tag == "i64" else np.uint64(xu[-1])
            out[f"{tag}_{kind}_low"] = np.where(active, np.full(BLK, value, dtype=dtype),
                                                np.zeros(BLK, dtype=dtype))
            out[f"{tag}_{kind}_high"] = np.where(active, np.full(BLK, high, dtype=dtype),
                                                 np.zeros(BLK, dtype=dtype))
            active_masks[f"{tag}_{kind}_low"] = active
            active_masks[f"{tag}_{kind}_high"] = active
    return RefResult(out, active_masks)


# ---------------------------------------------------------------------------
# 5c-bis. isolated unsigned-vcmp diagnostic (NOT an API acceptance case)
# ---------------------------------------------------------------------------

def _u64_vcmp_diag_inputs():
    lanes = np.arange(BLK, dtype=np.int64)
    u = ((np.uint64(1) << np.uint64(63)) + lanes.astype(np.uint64)).astype(np.uint64)
    active = (lanes % 3) != 0
    # Unsigned: u > 0 is true, u > UINT64_MAX is false.  Signed: both are false
    # because every u64 payload has the top bit set.
    thr = np.where(active, np.uint64(0), np.uint64(0xFFFFFFFFFFFFFFFF)).astype(np.uint64)
    return {"u": u, "thr": thr,
            "ones": np.ones(BLK, dtype=np.uint64),
            "zeros": np.zeros(BLK, dtype=np.uint64)}


def _u64_vcmp_diag_reference(inp):
    u, thr = inp["u"], inp["thr"]
    active = u > thr  # uint64 comparison on the host
    value = np.where(active, np.uint64(1), np.uint64(0)).astype(np.uint64)
    return RefResult({"mask": value}, {"mask": active})


U64_VCMP_DIAG_OUTPUTS = (
    OutputSpec("mask", "<u8", "exact", zero_inactive=True, sides=("tla",),
               note="u64 mask probe: 1 where the unsigned compare is true"),
)


def _wide_dupv_outputs():
    specs = []
    for tag, dtype in (("i64", "<i8"), ("u64", "<u8")):
        for kind in ("prefix", "arb", "none"):
            for pos in ("low", "high"):
                specs.append(OutputSpec(f"{tag}_{kind}_{pos}", dtype, "exact",
                                        zero_inactive=True, sides=("tla",),
                                        note="lowest/last lane of a real 64-lane register"))
    return tuple(specs)


# ---------------------------------------------------------------------------
# 5e. long / tail matrix: runtime N, padded capacity, per-chunk vdupv reference
# ---------------------------------------------------------------------------

def _tail_inputs(cap, dtype, with_index):
    """Guarded, legally padded inputs; every active lane value is unique."""
    data = np.arange(cap, dtype=np.float64) / 64.0
    a = data.astype(dtype)
    b = (data + 0.5).astype(dtype)
    c = np.full(data.shape, 1.0 / 16.0, dtype=dtype)
    guard = np.full(TAIL_GUARD, 123.5, dtype=dtype)
    arrays = {
        "a": np.concatenate([guard, a, guard]),
        "b": np.concatenate([guard, b, guard]),
        # vdiv uses a separate exact divisor so a/(1/16) stays position
        # unique even at f16 precision (a/b would quantise adjacent lanes).
        "c": np.concatenate([guard, c, guard]),
    }
    if with_index:
        idx_guard = np.zeros(TAIL_GUARD, dtype=np.int16)
        idx = np.concatenate([idx_guard,
                              np.arange(cap, dtype=np.int16),
                              idx_guard])
        arrays["idx"] = idx
    return arrays


def _tail_values(a, b, c, dtype, chunk):
    af, cf = a.astype(np.float32), c.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        div = (af / cf).astype(dtype)
        expv = np.exp(af).astype(dtype)
    zeros = np.zeros(a.size, dtype=dtype)
    low = np.empty_like(a)
    high = np.empty_like(a)
    for start in range(0, a.size, chunk):
        low[start:start + chunk] = a[start]
        high[start:start + chunk] = a[start + chunk - 1]
    return {
        "vdiv": div,
        "vmax": np.maximum(a, b),
        "vmin": np.minimum(a, b),
        "vmaxs": np.maximum(a, zeros),
        "vexp": expv,
        "vdup": np.full(a.size, dtype(1.5), dtype=dtype),
        "vdups": np.full(a.size, dtype(1.5), dtype=dtype),
        "vdupv_low": low,
        "vdupv_high": high,
    }


def _tail_reference(inp, cap, chunk, ns, dtype):
    a = inp["a"][TAIL_GUARD:TAIL_GUARD + cap]
    b = inp["b"][TAIL_GUARD:TAIL_GUARD + cap]
    c = inp["c"][TAIL_GUARD:TAIL_GUARD + cap]
    values = _tail_values(a, b, c, dtype, chunk)
    sentinel = dtype(TAIL_SENTINEL)
    zero = dtype(0.0)
    full_len = TAIL_GUARD + cap + TAIL_GUARD
    out = {}
    for n in ns:
        for op in TAIL_OPS:
            row = np.full(full_len, sentinel, dtype=dtype)
            row[TAIL_GUARD:TAIL_GUARD + cap] = zero
            row[TAIL_GUARD:TAIL_GUARD + n] = values[op][:n]
            out[f"n{n}_{op}"] = row
        for op in TAIL_MASKED_OPS:
            row = np.full(full_len, sentinel, dtype=dtype)
            row[TAIL_GUARD:TAIL_GUARD + n] = values[op][:n]
            out[f"n{n}_masked_{op}"] = row
    return RefResult(out)


def _tail_outputs(ns, dtype):
    dtype_str = "<f2" if dtype == np.float16 else "<f4"
    div_rtol = 2e-3 if dtype == np.float16 else 1e-6
    exp_rtol = 5e-3 if dtype == np.float16 else 1e-5
    specs = []
    for n in ns:
        for op in TAIL_OPS:
            if op == "vdiv":
                rule = dict(mode="close", rtol=div_rtol, atol=0.0)
            elif op == "vexp":
                rule = dict(mode="close", rtol=exp_rtol, atol=0.0)
            else:
                rule = dict(mode="exact")
            specs.append(OutputSpec(
                f"n{n}_{op}", dtype_str, sides=("tla",),
                note="guarded row: guard/arithmetic-tail sentinel, inactive data +0",
                **rule))
        for op in TAIL_MASKED_OPS:
            if op == "vdiv":
                rule = dict(mode="close", rtol=div_rtol, atol=0.0)
            else:
                rule = dict(mode="exact")
            specs.append(OutputSpec(
                f"n{n}_masked_{op}", dtype_str, sides=("tla",),
                note="masked vsts: every lane >= runtime N keeps the non-zero sentinel",
                **rule))
    return tuple(specs)


# ---------------------------------------------------------------------------
# 5d. f32 precision='exact' / 'vdiv_0ulp_ftz_true' -> ave.hir.vdivfhp
# ---------------------------------------------------------------------------

DIV_EXACT_A = [1.0, 3.0, 1e-30, 1.0, 7.0, -7.0, 1e20, 1e-20, 2.0, 5.0, 11.0, -3.0]
DIV_EXACT_B = [3.0, 7.0, 3e-30, 0.1, 3.0, 3.0, 3.0, 7e-21, 3.0, 7.0, 13.0, 7.0]


def _div_exact_inputs():
    a = I.pad(DIV_EXACT_A, BLK, 1.25)
    b = I.pad(DIV_EXACT_B, BLK, 1.75)
    return {"a": I.f32(a), "b": I.f32(b)}


def _div_exact_reference(inp):
    a, b = inp["a"], inp["b"]
    q = (a / b).astype(np.float32)
    prefix = np.arange(BLK) < 32
    none = np.zeros(BLK, dtype=bool)
    return RefResult(
        {"full": q.copy(), "prefix": _apply_mask(q, prefix), "none": _apply_mask(q, none),
         "alias_prefix": _apply_mask(q, prefix)},
        {"prefix": prefix, "none": none, "alias_prefix": prefix})


DIV_EXACT_OUTPUTS = (
    OutputSpec("full", "<f4", "exact", sides=("tla",),
               note="0ULP f32 division, compared bit-for-bit"),
    OutputSpec("prefix", "<f4", "exact", zero_inactive=True, sides=("tla",),
               note="prefix mask; inactive lanes bitwise +0"),
    OutputSpec("none", "<f4", "exact", zero_inactive=True, sides=("tla",),
               note="all-false mask"),
    OutputSpec("alias_prefix", "<f4", "exact", zero_inactive=True, sides=("tla",),
               note="'vdiv_0ulp_ftz_true' alias uses the same vdivfhp op"),
)


# ---------------------------------------------------------------------------
# 6b. MODE.ZEROING vs MODE.MERGING under one partial mask
# ---------------------------------------------------------------------------

def _mode_reference(inp):
    """Both modes zero inactive lanes; MERGING's builder merge source is +0."""
    a, b, m = inp["a"], inp["b"], inp["m_arb"]
    active = m != 0
    values = {
        "D": (a / b).astype(np.float32),
        "U": np.full_like(a, np.float32(1.5)),
        "E": np.exp(a).astype(np.float32),
        "X": ref_maximum(a, b),
        "S": ref_maximum_scalar(a, 0.0),
        "N": ref_minimum(a, b),
    }
    out, masks = {}, {}
    for suffix, value in values.items():
        zeroed = _apply_mask(value, active)
        for prefix in ("Z", "M"):
            name = f"{prefix}{suffix}"
            out[name] = zeroed
            masks[name] = active
    return RefResult(out, masks)


def _mode_outputs():
    specs = []
    rules = {
        "D": dict(mode="close", rtol=1e-6, atol=1e-6),
        "U": dict(mode="exact"),
        "E": dict(mode="close", rtol=1e-5, atol=0.0),
        "X": dict(mode="exact"),
        "S": dict(mode="exact", signed_zero="ieee"),
        "N": dict(mode="exact"),
    }
    for prefix, sides in (("Z", ("tla", "tilelang")), ("M", ("tla", ))):
        for suffix, rule in rules.items():
            specs.append(OutputSpec(f"{prefix}{suffix}", "<f4", **rule,
                                    zero_inactive=True, sides=sides,
                                    note="inactive lanes must be bitwise +0"))
    return tuple(specs)


# ---------------------------------------------------------------------------
# 7. NaN / Inf max, min, maxs
# ---------------------------------------------------------------------------

def _bits_of(values):
    return I.f32(values).view(np.uint32).tolist()


def _nan_inf_inputs():
    a_special = [
        I.F32_POS_QNAN_1, I.F32_NEG_QNAN_2, I.F32_POS_INF, I.F32_NEG_INF,
        I.F32_POS_ZERO, I.F32_NEG_ZERO,
    ] + _bits_of([1.0, -1.0, 3.5, -2.5, 0.0, -0.0])
    b_special = [
        _bits_of([1.0])[0], _bits_of([2.0])[0], _bits_of([3.0])[0], _bits_of([-3.0])[0],
        I.F32_NEG_ZERO, I.F32_POS_ZERO, I.F32_POS_QNAN_3, I.F32_NEG_QNAN_4,
        I.F32_NEG_INF, I.F32_POS_INF,
    ] + _bits_of([0.5, -0.5, 1.25, -1.25, 4.0, -4.0])
    a_bits = (a_special + _bits_of([2.0, -3.0, 7.0, -8.0]) * (BLK // 4 + 1))[:BLK]
    b_bits = (b_special + _bits_of([1.5, -2.5, 6.0, -7.0]) * (BLK // 4 + 1))[:BLK]
    # make sure the padding has no zero divisor etc. -- this case has no div
    return {"a": I.f32_from_bits(a_bits), "b": I.f32_from_bits(b_bits)}


def _nan_inf_reference(inp):
    a, b = inp["a"], inp["b"]
    return RefResult({
        "vmax": ref_maximum(a, b),
        "vmin": ref_minimum(a, b),
        "vmaxs_0": ref_maximum_scalar(a, 0.0),
        "vmaxs_neginf": ref_maximum_scalar(a, -np.inf),
    })


NAN_INF_OUTPUTS = (
    OutputSpec("vmax", "<f4", "close", 0.0, 0.0, nan_policy="propagate",
               signed_zero="ieee", note="declared policy: NaN propagates; max(+-0)=+0"),
    OutputSpec("vmin", "<f4", "close", 0.0, 0.0, nan_policy="propagate",
               signed_zero="ieee", note="declared policy: NaN propagates; min(+-0)=-0"),
    OutputSpec("vmaxs_0", "<f4", "close", 0.0, 0.0, nan_policy="propagate",
               signed_zero="ieee"),
    OutputSpec("vmaxs_neginf", "<f4", "close", 0.0, 0.0, nan_policy="propagate",
               signed_zero="ieee"),
)


# ---------------------------------------------------------------------------
# 8. uint32 vmaxs at / above 2**31
# ---------------------------------------------------------------------------

U32_HIGH_VALS = [0, 1, 2, U_HIGH - 1, U_HIGH, U_HIGH + 1, U_HIGH + 2,
                 U_HIGH + 0x1000, 0xFFFFFFFE, U_MAX, 12345, 0x40000000]


def _u32_high_inputs():
    return {"u": np.asarray(I.pad(U32_HIGH_VALS, BLK, 0x7FFFFFFF), dtype=np.uint32)}


def _u32_high_reference(inp):
    u = inp["u"]
    return RefResult({
        "s_80000000": np.maximum(u, np.uint32(U_HIGH)),
        "s_80000001": np.maximum(u, np.uint32(U_HIGH + 1)),
        "s_ffffffff": np.maximum(u, np.uint32(U_MAX)),
    })


# ---------------------------------------------------------------------------
# 9. i64 / u64 vmaxs exact scalars
# ---------------------------------------------------------------------------

I64_VALS = [0, 1, -1, 2**53 - 1, 2**53, 2**53 + 1, 2**53 + 2, 2**62, 2**62 + 12345,
            2**63 - 1, -(2**63), -(2**62), 12345, -12345]
U64_VALS = [0, 1, 2**31, 2**53 - 1, 2**53, 2**53 + 1, 2**63 - 1, 2**63,
            2**63 + 1, 2**64 - 2, 2**64 - 1, 12345]


def _i64_inputs():
    return {
        "i": np.asarray(I.pad(I64_VALS, BLK, 7), dtype=np.int64),
        "u": np.asarray(I.pad(U64_VALS, BLK, 9), dtype=np.uint64),
    }


def _i64_reference(inp):
    i, u = inp["i"], inp["u"]
    return RefResult({
        "i_s53": np.maximum(i, np.int64(2**53)),
        "i_s53p1": np.maximum(i, np.int64(2**53 + 1)),
        "i_s63m1": np.maximum(i, np.int64(2**63 - 1)),
        "u_s63": np.maximum(u, np.uint64(2**63)),
        "u_s63p1": np.maximum(u, np.uint64(2**63 + 1)),
        "u_smax": np.maximum(u, np.uint64(2**64 - 1)),
    })


# ---------------------------------------------------------------------------
# 10. f16 ops
# ---------------------------------------------------------------------------


def _f16_inputs():
    a = [((i * 11) % 23 - 11) / 4.0 for i in range(F16_BLK)]
    b = [((i * 7) % 19 + 1) / 8.0 for i in range(F16_BLK)]
    return {"a": I.f16(a), "b": I.f16(b)}


def _f16_reference(inp):
    a, b = inp["a"], inp["b"]
    af, bf = a.astype(np.float32), b.astype(np.float32)
    keep = np.arange(F16_BLK) < 64
    dup_part = np.where(keep, np.float16(1.5), np.float16(0.0))
    return RefResult(
        {
            "div": (af / bf).astype(np.float16),
            "exp": np.exp(af).astype(np.float16),
            "max": np.maximum(a, b),
            "min": np.minimum(a, b),
            "dup_full": np.full((F16_BLK,), np.float16(1.5), dtype=np.float16),
            "dup_part": dup_part.astype(np.float16),
        },
        {"dup_part": keep},
    )


F16_OUTPUTS = (
    OutputSpec("div", "<f2", "close", 2e-3, 0.0),
    OutputSpec("exp", "<f2", "close", 5e-3, 0.0),
    OutputSpec("max", "<f2"),
    OutputSpec("min", "<f2"),
    OutputSpec("dup_full", "<f2"),
    OutputSpec("dup_part", "<f2", zero_inactive=True),
)


# ---------------------------------------------------------------------------
# 11. multi-block / repeated dynamic views
# ---------------------------------------------------------------------------


def _multi_inputs():
    rows = []
    for r in range(8):
        a = [((i * 13 + r * 3) % 29 - 14) / 4.0 for i in range(BLK)]
        b = [((i * 7 + r * 5) % 31 + 1) / 5.0 for i in range(BLK)]
        rows.append(I.f32(a))
        rows.append(I.f32(b))
    return {"x": np.stack(rows).astype(np.float32)}


def _multi_reference(inp):
    x = inp["x"]
    out = np.empty_like(x)
    for r in range(0, 16, 2):
        out[r] = ref_maximum(x[r], x[r + 1])
        out[r + 1] = (x[r] / x[r + 1]).astype(np.float32)
    return RefResult({"out": out})


MULTI_OUTPUTS = (
    OutputSpec("out", "<f4", "close", 1e-6, 1e-6,
               note="even rows max (exact in practice), odd rows div"),
)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

CASES = {}


def _add(spec):
    if spec.name in CASES:
        raise ValueError(f"duplicate case {spec.name}")
    CASES[spec.name] = spec


_add(CaseSpec(
    name="smoke", title="minimal f32 vmax smoke", dtype="f32", mask="ALL", precision="n/a",
    description="One 64-lane f32 vmax(a,b) launch; the smallest device smoke test.",
    tier=TIER_IMPLEMENTED, sides=("tla", "tilelang"),
    kernels={"tla": "tla_smoke", "tilelang": "kernel"},
    outputs=(OutputSpec("out", "<f4", note="vmax(a,b)"),),
    inputs_fn=_smoke_inputs, reference_fn=_smoke_reference,
))

_add(CaseSpec(
    name="f32_combo", title="six APIs, full + prefix mask", dtype="f32",
    mask="ALL + pge(32)/PAT_VL32", precision="div/exp hardware (TLA) vs ftz_true (TileLang)",
    description="The 12-row matrix: vdiv/vdup/vexp/vmax/vmaxs/vmin in full and first-32 masks.",
    tier=TIER_IMPLEMENTED, sides=("tla", "tilelang"),
    kernels={"tla": "tla_ops_f32", "tilelang": "kernel"},
    outputs=COMBO_OUTPUTS, inputs_fn=_combo_inputs, reference_fn=_combo_reference,
))

_add(CaseSpec(
    name="f32_special", title="vdiv specials + vexp FTZ", dtype="f32",
    mask="ALL", precision="div ftz_true; exp default + ftz_true (+ TileLang ftz_false)",
    description="Zero divisor, signed zero, inf/overflow, subnormal exp boundary, FTZ policy.",
    tier=TIER_IMPLEMENTED, sides=("tla", "tilelang"),
    kernels={"tla": "tla_f32_special", "tilelang": "kernel"},
    outputs=SPECIAL_OUTPUTS, inputs_fn=_special_inputs, reference_fn=_special_reference,
))

_add(CaseSpec(
    name="int_bounds", title="int32/uint32 sign boundary", dtype="i32/u32",
    mask="ALL", precision="n/a",
    description="vmax/vmin/vmaxs across INT_MIN/INT_MAX/0x80000000/0xffffffff.",
    tier=TIER_IMPLEMENTED, sides=("tla", "tilelang"),
    kernels={"tla": "tla_int_bounds", "tilelang": "kernel"},
    outputs=INT_OUTPUTS, inputs_fn=_int_inputs, reference_fn=_int_reference,
))

_add(CaseSpec(
    name="vdup_int_bf16", title="vdup int32 / bf16, full + prefix mask", dtype="i32/bf16",
    mask="ALL + pge(32)/PAT_VL32", precision="n/a",
    description="Masked broadcast for int32 and bf16, including bitwise inactive-zero checks.",
    tier=TIER_IMPLEMENTED, sides=("tla", "tilelang"),
    kernels={"tla": "tla_vdup", "tilelang": "kernel"},
    outputs=VDUP_OUTPUTS, inputs_fn=_vdup_inputs, reference_fn=_vdup_reference,
))

_add(CaseSpec(
    name="u16_div_zero", title="u16 vdiv zeroing, full/half/all-false, both modes", dtype="u16",
    mask="ALL + prefix 64/128 + all-false; TLA ZEROING and MERGING", precision="integer truncation toward zero",
    description="Masked unsigned 16-bit division stored with the ALL store mask; every inactive lane must be bitwise +0. "
                "TLA MERGING rows are TLA-only because their merge source is a +0 SSA vector (builder-side vsel), not TileLang's dst-preserving merge.",
    tier=TIER_IMPLEMENTED, sides=("tla", "tilelang"),
    kernels={"tla": "tla_u16_div_zero", "tilelang": "u16_div_zero"},
    outputs=_u_div_outputs("<u2"),
    inputs_fn=lambda: _u_div_inputs(U16_DIV_BLK, np.uint16), reference_fn=_u_div_reference,
    notes="Covers the common downstream unsigned software-division zeroing defect without weakening the oracle.",
))

_add(CaseSpec(
    name="u32_div_zero", title="u32 vdiv zeroing, full/half/all-false, both modes", dtype="u32",
    mask="ALL + prefix 32/64 + all-false; TLA ZEROING and MERGING", precision="integer truncation toward zero",
    description="Masked unsigned 32-bit division stored with the ALL store mask; every inactive lane must be bitwise +0. "
                "TLA MERGING rows are TLA-only because their merge source is a +0 SSA vector (builder-side vsel), not TileLang's dst-preserving merge.",
    tier=TIER_IMPLEMENTED, sides=("tla", "tilelang"),
    kernels={"tla": "tla_u32_div_zero", "tilelang": "u32_div_zero"},
    outputs=_u_div_outputs("<u4"),
    inputs_fn=lambda: _u_div_inputs(BLK, np.uint32), reference_fn=_u_div_reference,
    notes="Covers the common downstream unsigned software-division zeroing defect without weakening the oracle.",
))

_add(CaseSpec(
    name="f32_dupv_mask", title="vdups / vdupv lowest & highest lane masks", dtype="f32",
    mask="ALL + prefix 32/64 + all-false", precision="n/a",
    description="vdups is the masked scalar vdup; vdupv duplicates lane 0 or the last lane (not the highest half-register). "
                "Lane 0 and the last lane are deliberately different so a half-register broadcast cannot pass.",
    tier=TIER_IMPLEMENTED, sides=("tla", "tilelang"),
    kernels={"tla": "tla_dupv_mask", "tilelang": "f32_dupv_mask"},
    outputs=DUPV_OUTPUTS, inputs_fn=_dupv_inputs, reference_fn=_dupv_reference,
))

_add(CaseSpec(
    name="div_signed_wide_masks", title="signed i16/i32 + i64/u64 vdiv under three masks", dtype="i16/i32/i64/u64",
    mask="prefix half + signed-index arbitrary (i64/u64) / signed-data compare (i16/i32) + all-false",
    precision="integer truncation toward zero",
    description="Signed i16/i32 and i64/u64 division with caller masks, stored with the ALL store mask. "
                "Every inactive lane must stay bitwise zero; the unsigned 16/32-bit zeroing select is not "
                "extended to these types without device evidence. i64/u64 share a signed i64 index mask "
                "(lane % 3 != 0) so the wide rows do not depend on u64 vcmp semantics.",
    tier=TIER_IMPLEMENTED, sides=("tla",),
    kernels={"tla": "tla_div_signed_wide_masks"},
    outputs=_div_sw_outputs(), inputs_fn=_div_sw_inputs, reference_fn=_div_sw_reference,
    notes="TLA-only: the TileLang package used locally cannot store NORM_B64 (CANN mask families stop at "
          "8/16/32), so the 64-bit side is not mirrored here. The reference is still the strict integer oracle.",
))

_add(CaseSpec(
    name="i64_u64_dupv_mask", title="i64/u64 vdupv lowest/highest, three masks", dtype="i64/u64",
    mask="prefix half + signed-index arbitrary (rem > idx) + all-false", precision="n/a",
    description="Real 64-lane i64/u64 vload into vdupv (vector_broadcast) at POS_LOWEST and POS_HIGHEST. "
                "Lane 0 and lane 63 differ, so a half-register copy cannot pass; inactive lanes must be zero. "
                "The arbitrary mask is built from signed i64 lane indices (lane % 3 != 0, 42 active lanes), "
                "not from a u64 data compare.",
    tier=TIER_IMPLEMENTED, sides=("tla",),
    kernels={"tla": "tla_wide_dupv_mask"},
    outputs=_wide_dupv_outputs(), inputs_fn=_wide_dupv_inputs, reference_fn=_wide_dupv_reference,
    notes="TLA-only because the local TileLang package does not support NORM_B64 storage. "
          "i64 and u64 share the signless 64-lane ABI; u64 is checked by bit pattern. "
          "This API case deliberately does not exercise unsigned vcmp: that predicate is "
          "unverified on the current wheel and is isolated in diagnostic_u64_vcmp_mask.",
))

_add(CaseSpec(
    name="diagnostic_u64_vcmp_mask", title="isolated u64 unsigned-vcmp mask probe (diagnostic)", dtype="u64",
    mask="u64 payload > u64 threshold, both above 2**63", precision="n/a",
    description="Diagnostic only: stores the predicate produced by vcmp on real 64-lane u64 registers "
                "as u64 0/1 lanes. It never calls vdupv and is not part of API acceptance. On a wheel "
                "without an unsigned predicate the compare is signed and this case reports the mask "
                "as all false; that is a vcmp finding, not a vdupv failure.",
    tier=TIER_IMPLEMENTED, sides=("tla",),
    kernels={"tla": "tla_u64_vcmp_diag"},
    outputs=U64_VCMP_DIAG_OUTPUTS,
    inputs_fn=_u64_vcmp_diag_inputs, reference_fn=_u64_vcmp_diag_reference,
    notes="Kept separate from i64_u64_dupv_mask so the API case has no dependency on unsigned "
          "vcmp semantics. Do not include this row in API acceptance and do not report it as a "
          "vdupv defect; the unsigned-predicate limitation is a separate, still-open issue.",
))

_add(CaseSpec(
    name="f32_long_tail", title="f32 runtime-N long/tail matrix, 10 N values", dtype="f32",
    mask="runtime vci(index) < runtime N -> vcmp <LT>; no pge prefix", precision="div/exp hardware",
    description="One guarded row per op and N. Runtime N drives a vcmp <LT> tail predicate; "
                "CAP=320 (5 chunks) is read in full only because it is legally padded. Every op has an "
                "ALL-store arithmetic row (inactive data lanes +0), and vmax/vdiv/vdupv-low/high also "
                "have a masked-store row that keeps the non-zero sentinel after N. Guards stay sentinel. "
                "vdupv references are per chunk (lane 0 / lane CHUNK-1), not the whole-input last lane.",
    tier=TIER_IMPLEMENTED, sides=("tla",),
    kernels={"tla": "tla_tail_f32"},
    outputs=_tail_outputs(TAIL_F32_NS, np.float32),
    inputs_fn=lambda: _tail_inputs(TAIL_F32_CAP, np.float32, with_index=False),
    reference_fn=lambda inp: _tail_reference(inp, TAIL_F32_CAP, TAIL_F32_CHUNK,
                                             TAIL_F32_NS, np.float32),
    notes="TLA-only. vlds has no mask, so the legal-padding contract is explicit: CAP is a whole "
          "number of 64-lane chunks and N <= CAP. This does not claim support for masked loads "
          "without padding. The per-program loop walks 5 chunks with CORES=2, so a program executes "
          "more than one chunk.",
))

_add(CaseSpec(
    name="f16_long_tail", title="f16 runtime-N long/tail matrix, 9 N values", dtype="f16",
    mask="runtime i16 index < runtime N -> vcmp <LT>; no pge prefix", precision="hardware",
    description="Same guarded matrix as f32_long_tail with 128-lane f16 registers, CAP=512 (4 chunks), "
                "and an int16 lane-index input because vci is 64-lane i32 only. Runtime N is broadcast "
                "with a runtime vdup scalar and compared by a signed vcmp <LT>.",
    tier=TIER_IMPLEMENTED, sides=("tla",),
    kernels={"tla": "tla_tail_f16"},
    outputs=_tail_outputs(TAIL_F16_NS, np.float16),
    inputs_fn=lambda: _tail_inputs(TAIL_F16_CAP, np.float16, with_index=True),
    reference_fn=lambda inp: _tail_reference(inp, TAIL_F16_CAP, TAIL_F16_CHUNK,
                                             TAIL_F16_NS, np.float16),
    notes="TLA-only. The 128-lane register means the mask is built from a padded int16 index vector, "
          "not vci; vlds still reads whole 128-lane blocks inside the legal CAP=512 padding.",
))

_add(CaseSpec(
    name="f32_mask_pattern", title="arbitrary non-prefix + all-false masks", dtype="f32",
    mask="vcmp-built arbitrary lane pattern; all-false", precision="div/exp hardware",
    description="Builds masks with vcmp (not pge), checks active results and bitwise zeroing of inactive lanes.",
    tier=TIER_IMPLEMENTED, sides=("tla", "tilelang"),
    kernels={"tla": "tla_mask_pattern", "tilelang": "kernel"},
    outputs=_mask_outputs(), inputs_fn=_mask_inputs, reference_fn=_mask_reference,
))

_add(CaseSpec(
    name="f32_div_exact", title="f32 vdiv precision='exact' -> ave.hir.vdivfhp", dtype="f32",
    mask="ALL + prefix half + all-false", precision="0ULP exact (vdiv_0ulp_ftz_true)",
    description="Both precise aliases must emit ave.hir.vdivfhp (not the FTZ vdiv), keep the caller mask, "
                "and match correctly-rounded f32 division bit-for-bit; inactive lanes stay zero.",
    tier=TIER_IMPLEMENTED, sides=("tla",),
    kernels={"tla": "tla_div_exact"},
    outputs=DIV_EXACT_OUTPUTS, inputs_fn=_div_exact_inputs, reference_fn=_div_exact_reference,
    notes="TLA-only in this harness; TileLang has the same fp32-only precise path but is not installed "
          "in the local simulator environment.",
))

_add(CaseSpec(
    name="f32_mode_mask", title="MODE.ZEROING vs MODE.MERGING, one partial mask", dtype="f32",
    mask="vcmp-built arbitrary lane pattern", precision="div/exp hardware",
    description="Both TLA modes must return masked results with bitwise +0 inactive lanes. "
                "The TileLang side is explicitly MODE_ZEROING and is never compared with a dst-preserving merge.",
    tier=TIER_IMPLEMENTED, sides=("tla", "tilelang"),
    kernels={"tla": "tla_mode_mask", "tilelang": "mode_zeroing"},
    outputs=_mode_outputs(), inputs_fn=_mask_inputs, reference_fn=_mode_reference,
    notes="TLA MERGING is lowered by the C++ builder to the same masked op plus a same-typed "
          "+0 merge source and ave.hir.vsel, so inactive lanes are +0. It does not preserve "
          "the old dst/src0 and is not CANN's native MODE_MERGING.",
))

_add(CaseSpec(
    name="f32_nan_inf", title="NaN / Inf max, min, maxs", dtype="f32",
    mask="ALL", precision="n/a",
    description="Quiet NaNs with distinct payloads, +-Inf, +-0; declared policy: NaN propagates, IEEE signed zero.",
    tier=TIER_IMPLEMENTED, sides=("tla",),
    kernels={"tla": "tla_nan_inf"},
    outputs=NAN_INF_OUTPUTS, inputs_fn=_nan_inf_inputs, reference_fn=_nan_inf_reference,
))

_add(CaseSpec(
    name="u32_vmaxs_high", title="uint32 vmaxs at/above 2**31", dtype="u32",
    mask="ALL", precision="n/a",
    description="Unsigned scalar maximum with 0x80000000, 0x80000001 and 0xffffffff.",
    tier=TIER_IMPLEMENTED, sides=("tla",),
    kernels={"tla": "tla_u32_vmaxs_high"},
    outputs=(
        OutputSpec("s_80000000", "<u4"),
        OutputSpec("s_80000001", "<u4"),
        OutputSpec("s_ffffffff", "<u4"),
    ),
    inputs_fn=_u32_high_inputs, reference_fn=_u32_high_reference,
))

_add(CaseSpec(
    name="i64_u64_vmaxs", title="i64/u64 exact vmaxs scalars", dtype="i64/u64",
    mask="ALL", precision="n/a",
    description="Scalars above 2**53 and 2**63; proves no double round-trip and two's-complement u64 bits.",
    tier=TIER_IMPLEMENTED, sides=("tla",),
    kernels={"tla": "tla_i64_u64_vmaxs"},
    outputs=(
        OutputSpec("i_s53", "<i8"),
        OutputSpec("i_s53p1", "<i8"),
        OutputSpec("i_s63m1", "<i8"),
        OutputSpec("u_s63", "<u8"),
        OutputSpec("u_s63p1", "<u8"),
        OutputSpec("u_smax", "<u8"),
    ),
    inputs_fn=_i64_inputs, reference_fn=_i64_reference,
))

_add(CaseSpec(
    name="f16_ops", title="f16 vdiv/vexp/max/min/vdup", dtype="f16",
    mask="ALL + prefix 64/128", precision="hardware",
    description="f16 dtype coverage with exact mask-zeroing on the partial vdup.",
    tier=TIER_IMPLEMENTED, sides=("tla", "tilelang"),
    kernels={"tla": "tla_f16_ops", "tilelang": "kernel"},
    outputs=F16_OUTPUTS, inputs_fn=_f16_inputs, reference_fn=_f16_reference,
))

_add(CaseSpec(
    name="f32_multi_block", title="4 blocks x repeated dynamic view offsets", dtype="f32",
    mask="ALL", precision="div/exp hardware",
    description="grid=(4,), 8 token pairs; each block walks two pairs through dynamic tla.view offsets.",
    tier=TIER_IMPLEMENTED, sides=("tla",),
    kernels={"tla": "tla_multi_block"},
    outputs=MULTI_OUTPUTS, inputs_fn=_multi_inputs, reference_fn=_multi_reference,
))

# -- evaluated but intentionally not implemented ----------------------------

_add(CaseSpec(
    name="fp8_vdup_device", title="fp8e4nv vdup device value check", dtype="fp8e4nv",
    mask="ALL + prefix", precision="n/a", tier=TIER_SUGGESTED,
    description="vdup supports fp8e4nv in the AVE lowering; storage/compare for fp8 needs a bit-level adapter before a device case is trustworthy.",
    sides=("tla",), kernels={}, outputs=(),
    notes="Implement with uint8 raw-bit transport; compare exact E4M3FN bit patterns.",
))

_add(CaseSpec(
    name="u8_vmax_vmin_device", title="u8/i8 vmax/vmin device", dtype="u8/i8",
    mask="ALL", precision="n/a", tier=TIER_SUGGESTED,
    description="TLA advertises i8/u8 for vmax/vmin; the 2048-bit register has 256 lanes, which needs a new input/layout contract.",
    sides=("tla",), kernels={}, outputs=(),
))

_add(CaseSpec(
    name="f16_mask_pattern", title="f16 arbitrary mask", dtype="f16",
    mask="vcmp-built arbitrary pattern", precision="hardware", tier=TIER_SUGGESTED,
    description="Extends the non-prefix mask case to f16; wait for f16 vcmp/pset width semantics to be confirmed on both sides.",
    sides=("tla", "tilelang"), kernels={}, outputs=(),
))

_add(CaseSpec(
    name="multi_block_sync", title="cross-block sync / repeated launch stress", dtype="f32",
    mask="ALL", precision="hardware", tier=TIER_SUGGESTED,
    description="Large repeated multi-block schedule to stress sync/hoist passes; separate from the single-launch correctness matrix.",
    sides=("tla",), kernels={}, outputs=(),
    notes="Do not fold into the correctness gate until the repeated dynamic-view launch issue is resolved.",
))


def get_case(name: str) -> CaseSpec:
    if name not in CASES:
        raise KeyError(f"unknown case {name!r}; use --list")
    return CASES[name]


def implemented_cases():
    return [c for c in CASES.values() if c.implemented]


def list_payload():
    from .spec import public_case_metadata
    return [public_case_metadata(c) for c in CASES.values()]
