"""TLA 3-in-1 test: elementwise SIMD ops (vdiv / vmax / vmin / vmaxs / vexp / vdup).

Semantics for one 64-lane f32 tile (row) of ``X``:

    one = vdup(1.0)                         masked scalar broadcast, ALL mask
    h   = vdup(0.5)                         masked scalar broadcast, ALL mask
    q   = vdiv(x, one)                      ave.hir.vdiv
    mx  = vmax(x, zero)                     ave.hir.vmax
    mn  = vmin(x, one)                      ave.hir.vmin
    ms  = vmaxs(x, 0.0)                     ave.hir.vmaxs (scalar == f32)
    ex  = vexp(vmin(x, zero))               ave.hir.vexp (hardware SFU)
    acc = q + mx + mn + ms + ex
    o   = vsel(pge(32, 64), acc, h)         lanes >= 32 -> 0.5

The unsigned / i64 / u64 paths are IR-only (``simd_elementwise_dtype_ir``):
they check that uint32 selects VMaxUIOp/VMinUIOp/vdiv-cast-unsigned, that i64
and u64 come from real 64-lane ``vlds`` registers, that an i64 scalar above
2**53 stays exact, and that u64 constexprs are materialized as exact
two's-complement i64 constants (including 2**63 and 2**64-1).  Runtime scalar
operands are covered as well.

``simd_vdup_alias_ir`` covers the vdup dtype spellings against the real
builder (``i8``/``u8``/``i32``/``u32``/``f16``/``f32``/``bf16``/``fp8e4nv`` and
their canonical ``int*``/``uint*``/``fp*``/``float*`` aliases).

``simd_mode_ir`` compiles the six APIs once with ``MODE_ZEROING`` and once with
``MODE_MERGING`` under the same partial mask.  ZEROING keeps the bare masked
AVE ops; MERGING keeps those same ops and adds a real zero-init + merge:
``arith.constant`` ``+0`` -> ``ave.hir.scalar_broadcast`` (the same-typed merge
source) -> ``ave.hir.vsel`` for each result.  The mask value is unchanged, so
both modes read inactive lanes as ``+0``, but MERGING is not TileLang's
dst-preserving merge and is not CANN's native ``MODE_MERGING``.

``simd_unsigned_div_zero_ir`` covers the ZEROING-only exception: the 16/32-bit
unsigned integer division software sequence does not guarantee zero inactive
lanes, so a caller-supplied mask appends ``vsel(result, 0)`` in the frontend.
The default all-true mask stays bare, and MERGING relies solely on the
builder-side merge select (no stacked second select).
``simd_dupv_f32_ir`` / ``simd_dupv_typed_ir``
check ``vdups`` -> ``ave.hir.broadcast`` and ``vdupv`` -> lowest/highest-lane
``ave.hir.vector_broadcast`` for f32, int32 and bf16; ``simd_dupv_wide_ir``
does the same for real 64-lane i64/u64 registers.

``simd_precision_exact_ir`` maps the f32 ``precision='exact'`` /
``'vdiv_0ulp_ftz_true'`` tiers onto the real ``ave.hir.vdivfhp`` op (not the
FTZ ``ave.hir.vdiv``) with the caller mask connected.  ``simd_mask_ssa_ir``
connects one prefix, one arbitrary and one all-false mask to all eight public
APIs (including ``vdups`` and both ``vdupv`` positions) and checks that each op
consumes the exact mask SSA value.  ``simd_div_masks_ir`` checks that signed
i32 and i64/u64 division under prefix/arbitrary/all-false masks keeps the bare
divide (the unsigned 16/32-bit zeroing select is deliberately not extended).
``simd_tail_mask_ir`` covers the dynamic long/tail shape: runtime ``N`` is
broadcast with ``vdup``, a runtime ``vci`` lane index is compared with a single
signed ``vcmp <LT>``, and a 5-chunk / step-2 loop feeds both ALL-store
arithmetic rows and four rows that use an actual masked store.  It also checks
that ``pge`` appears only as the ALL mask, never as a 31/63/... tail prefix.

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline HIVM IR assert (ave.hir.vdiv/vmax/vmin/vmaxs/vexp/broadcast)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy (SIMD_ELEMENTWISE OK)
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tla_ut/
import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir,
                                  check_compile_artifacts)

# default shapes: rows (tmax), lane block, cores
SHAPE = dict(tmax=8, blk=64, cores=4)
I64_BIG = 2**62 + 12345  # not representable as a double -> proves exact materialization
U64_HIGH = 2**63         # lowest value that overflows an int64_t binding
U64_MAX = 2**64 - 1      # -1 as the exact i64 two's-complement bit pattern

SIGNATURE = {"X_ptr": "*fp32", "O_ptr": "*fp32", "num_tokens": "i32"}
DTYPE_SIGNATURE = {
    "U_ptr": "*u32", "I_ptr": "*i64", "W_ptr": "*u64",
    "UO_ptr": "*u32", "IO_ptr": "*i64", "WO_ptr": "*u64",
    "n": "u32", "n64": "u64",
}


@triton.jit
def simd_elementwise(X_ptr, O_ptr, num_tokens, TMAX: tl.constexpr, BLK: tl.constexpr, CORES: tl.constexpr):
    tl.static_assert(BLK == 64, "one full f32 register per tile")
    tl.static_assert(TMAX % 2 == 0, "one pair of rows per output row")
    xy_ub = tla.alloc_buffer((2, BLK), tl.float32, space="ub", versions=2)
    o_ub = tla.alloc_buffer((BLK, ), tl.float32, space="ub", versions=2)
    for p in range(tl.program_id(0), num_tokens // 2, CORES):
        xy = tla.view(X_ptr, (2, BLK), strides=(BLK, 1), offsets=(2 * p, 0))
        o = tla.view(O_ptr, (BLK, ), strides=(1, ), offsets=(2 * p * BLK, ))
        tla.copy(xy, xy_ub)
        with tla.scope.simd():
            a = tla.simd.vlds(xy_ub[0, 0], dist="NORM")
            b = tla.simd.vlds(xy_ub[1, 0], dist="NORM")
            zero = tla.simd.vdup(0.0, tl.float32)
            half = tla.simd.vdup(0.5, tl.float32)
            q = tla.simd.vdiv(a, b)
            mx = tla.simd.vmax(a, b)
            mn = tla.simd.vmin(a, b)
            ms = tla.simd.vmaxs(a, 0.0)
            ex = tla.simd.vexp(tla.simd.vmin(a, zero))
            acc = tla.simd.vadd(tla.simd.vadd(q, mx), tla.simd.vadd(mn, ms))
            acc = tla.simd.vadd(acc, ex)
            res = tla.simd.vsel(tla.simd.pge(32, BLK), acc, half)
            tla.simd.vsts(o_ub[0], res, dist="NORM_B32")
        tla.copy(o_ub, o)


@triton.jit
def simd_elementwise_dtype_ir(U_ptr, I_ptr, W_ptr, UO_ptr, IO_ptr, WO_ptr, n, n64,
                              BLK: tl.constexpr):
    """IR-only: unsigned ops, runtime scalars, and full 64-lane i64/u64 registers."""
    u_ub = tla.alloc_buffer((BLK, ), tl.uint32, space="ub")
    uo_ub = tla.alloc_buffer((BLK, ), tl.uint32, space="ub")
    i_ub = tla.alloc_buffer((BLK, ), tl.int64, space="ub")
    io_ub = tla.alloc_buffer((BLK, ), tl.int64, space="ub")
    w_ub = tla.alloc_buffer((BLK, ), tl.uint64, space="ub")
    wo_ub = tla.alloc_buffer((BLK, ), tl.uint64, space="ub")
    u = tla.view(U_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    uo = tla.view(UO_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    i = tla.view(I_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    io = tla.view(IO_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    w = tla.view(W_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    wo = tla.view(WO_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    tla.copy(u, u_ub)
    tla.copy(i, i_ub)
    tla.copy(w, w_ub)
    with tla.scope.simd():
        uv = tla.simd.vlds(u_ub[0], dist="NORM")
        u1 = tla.simd.vmax(uv, uv)  # ave.hir.vumax
        u2 = tla.simd.vmin(u1, uv)  # ave.hir.vumin
        u3 = tla.simd.vdiv(u2, uv)  # ave.hir.vdiv {cast = cast_unsigned}
        u4 = tla.simd.vmaxs(u3, 7)  # ave.hir.vumaxs, constexpr
        u5 = tla.simd.vmaxs(u4, n)  # ave.hir.vumaxs, runtime u32 scalar
        tla.simd.vsts(uo_ub[0], u5, dist="NORM_B32")

        iv = tla.simd.vlds(i_ub[0], dist="NORM")  # vector<64xi64>
        i1 = tla.simd.vmaxs(iv, 2**62 + 12345)  # exact i64 constexpr
        tla.simd.vsts(io_ub[0], i1, dist="NORM_B64")

        wv = tla.simd.vlds(w_ub[0], dist="NORM")  # vector<64xi64>, uint64
        w1 = tla.simd.vmaxs(wv, 2**63)     # two's-complement i64 min
        w2 = tla.simd.vmaxs(w1, 2**64 - 1)  # two's-complement -1
        w3 = tla.simd.vmaxs(w2, n64)       # runtime u64 scalar, full range
        tla.simd.vsts(wo_ub[0], w3, dist="NORM_B64")
    tla.copy(uo_ub, uo)
    tla.copy(io_ub, io)
    tla.copy(wo_ub, wo)


@triton.jit
def simd_vdup_alias_ir(O_ptr, DT: tl.constexpr):
    """IR-only: vdup dtype aliases resolve to the canonical element types."""
    o_ub = tla.alloc_buffer((64, ), tl.uint32, space="ub")
    o = tla.view(O_ptr, (64, ), strides=(1, ), offsets=(0, ))
    with tla.scope.simd():
        d = tla.simd.vdup(1, DT)
        bc = tla.simd.vbitcast(d, "uint32")
        tla.simd.vsts(o_ub[0], bc, dist="NORM_B32")
    tla.copy(o_ub, o)


@triton.jit
def simd_mode_ir(X_ptr, O_ptr, BLK: tl.constexpr, MODE_STR: tl.constexpr):
    """IR-only: ZEROING is the bare op, MERGING adds the +0 merge select.

    Both modes read inactive lanes as zero, but MERGING must be consumed by
    the C++ builder: the IR contains a same-typed zero broadcast and a
    ``vsel`` over the result.  It must never select ``src0`` / the old dst.
    """
    x_ub = tla.alloc_buffer((2, BLK), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((6, BLK), tl.float32, space="ub")
    x = tla.view(X_ptr, (2, BLK), strides=(BLK, 1), offsets=(0, 0))
    o = tla.view(O_ptr, (6, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(x, x_ub)
    with tla.scope.simd():
        a = tla.simd.vlds(x_ub[0, 0], dist="NORM")
        b = tla.simd.vlds(x_ub[1, 0], dist="NORM")
        p = tla.simd.pge(32, BLK)
        tla.simd.vsts(o_ub[0, 0], tla.simd.vdiv(a, b, mask=p, mode=MODE_STR))
        tla.simd.vsts(o_ub[1, 0], tla.simd.vdup(1.5, tl.float32, mask=p, mode=MODE_STR))
        tla.simd.vsts(o_ub[2, 0], tla.simd.vexp(a, mask=p, mode=MODE_STR))
        tla.simd.vsts(o_ub[3, 0], tla.simd.vmax(a, b, mask=p, mode=MODE_STR))
        tla.simd.vsts(o_ub[4, 0], tla.simd.vmaxs(a, 0.0, mask=p, mode=MODE_STR))
        tla.simd.vsts(o_ub[5, 0], tla.simd.vmin(a, b, mask=p, mode=MODE_STR))
    tla.copy(o_ub, o)


@triton.jit
def simd_unsigned_div_zero_ir(U_ptr, O_ptr, BLK: tl.constexpr, MODE_STR: tl.constexpr):
    """IR-only: masked unsigned vdiv is followed by an explicit zeroing select.

    The 16/32-bit unsigned division software sequence does not guarantee zero
    inactive lanes by itself, so a caller-supplied mask must produce a masked
    ``vsel(result, 0)``.  The default all-true mask must not add that select.
    """
    u_ub = tla.alloc_buffer((BLK, ), tl.uint32, space="ub")
    o_ub = tla.alloc_buffer((3, BLK), tl.uint32, space="ub")
    u = tla.view(U_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (3, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(u, u_ub)
    with tla.scope.simd():
        uv = tla.simd.vlds(u_ub[0], dist="NORM")
        p_half = tla.simd.pge(32, BLK)
        p_none = tla.simd.vcmp(uv, uv, "ne")  # all-false predicate
        tla.simd.vsts(o_ub[0, 0], tla.simd.vdiv(uv, uv, mask=p_half, mode=MODE_STR),
                      dist="NORM_B32")
        tla.simd.vsts(o_ub[1, 0], tla.simd.vdiv(uv, uv, mode=MODE_STR),
                      dist="NORM_B32")
        tla.simd.vsts(o_ub[2, 0], tla.simd.vdiv(uv, uv, mask=p_none, mode=MODE_STR),
                      dist="NORM_B32")
    tla.copy(o_ub, o)


@triton.jit
def simd_dupv_f32_ir(X_ptr, O_ptr, BLK: tl.constexpr):
    """IR-only: vdups alias plus vdupv lowest/highest lane under one mask."""
    x_ub = tla.alloc_buffer((BLK, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((3, BLK), tl.float32, space="ub")
    x = tla.view(X_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (3, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(x, x_ub)
    with tla.scope.simd():
        xv = tla.simd.vlds(x_ub[0], dist="NORM")
        p = tla.simd.pge(32, BLK)
        tla.simd.vsts(o_ub[0, 0], tla.simd.vdups(1.5, tl.float32, mask=p),
                      dist="NORM_B32")
        tla.simd.vsts(o_ub[1, 0], tla.simd.vdupv(xv, mask=p, pos=tla.simd.POS.LOWEST),
                      dist="NORM_B32")
        tla.simd.vsts(o_ub[2, 0], tla.simd.vdupv(xv, mask=p, pos=tla.simd.POS.HIGHEST),
                      dist="NORM_B32")
    tla.copy(o_ub, o)


@triton.jit
def simd_dupv_typed_ir(I_ptr, B_ptr, IO_ptr, BO_ptr, BLK: tl.constexpr):
    """IR-only: vdups/vdupv for int32 and bf16 (no fake i64 support)."""
    i_ub = tla.alloc_buffer((BLK, ), tl.int32, space="ub")
    b_ub = tla.alloc_buffer((BLK, ), tl.bfloat16, space="ub")
    io_ub = tla.alloc_buffer((3, BLK), tl.int32, space="ub")
    bo_ub = tla.alloc_buffer((3, BLK), tl.bfloat16, space="ub")
    i = tla.view(I_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    b = tla.view(B_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    io = tla.view(IO_ptr, (3, BLK), strides=(BLK, 1), offsets=(0, 0))
    bo = tla.view(BO_ptr, (3, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(i, i_ub)
    tla.copy(b, b_ub)
    with tla.scope.simd():
        iv = tla.simd.vlds(i_ub[0], dist="NORM")
        bv = tla.simd.vlds(b_ub[0], dist="NORM")
        p32 = tla.simd.pge(32, BLK)
        p16 = tla.simd.pge(32, 2 * BLK)
        tla.simd.vsts(io_ub[0, 0], tla.simd.vdups(7, tl.int32, mask=p32),
                      dist="NORM_B32")
        tla.simd.vsts(io_ub[1, 0], tla.simd.vdupv(iv, mask=p32, pos=tla.simd.POS.LOWEST),
                      dist="NORM_B32")
        tla.simd.vsts(io_ub[2, 0], tla.simd.vdupv(iv, mask=p32, pos=tla.simd.POS.HIGHEST),
                      dist="NORM_B32")
        tla.simd.vsts(bo_ub[0, 0], tla.simd.vdups(1.5, tl.bfloat16, mask=p16),
                      dist="NORM_B16")
        tla.simd.vsts(bo_ub[1, 0], tla.simd.vdupv(bv, mask=p16, pos=tla.simd.POS.LOWEST),
                      dist="NORM_B16")
        tla.simd.vsts(bo_ub[2, 0], tla.simd.vdupv(bv, mask=p16, pos=tla.simd.POS.HIGHEST),
                      dist="NORM_B16")
    tla.copy(io_ub, io)
    tla.copy(bo_ub, bo)


@triton.jit
def simd_precision_exact_ir(X_ptr, O_ptr, BLK: tl.constexpr):
    """IR-only: f32 precise tiers emit ave.hir.vdivfhp with the same mask."""
    x_ub = tla.alloc_buffer((2, BLK), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((2, BLK), tl.float32, space="ub")
    x = tla.view(X_ptr, (2, BLK), strides=(BLK, 1), offsets=(0, 0))
    o = tla.view(O_ptr, (2, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(x, x_ub)
    with tla.scope.simd():
        a = tla.simd.vlds(x_ub[0, 0], dist="NORM")
        b = tla.simd.vlds(x_ub[1, 0], dist="NORM")
        p32 = tla.simd.pge(32, BLK)
        p16 = tla.simd.pge(16, BLK)
        tla.simd.vsts(o_ub[0, 0], tla.simd.vdiv(a, b, mask=p32, precision="exact"))
        tla.simd.vsts(o_ub[1, 0],
                      tla.simd.vdiv(a, b, mask=p16, precision="vdiv_0ulp_ftz_true"))
    tla.copy(o_ub, o)


@triton.jit
def simd_dupv_wide_ir(I_ptr, U_ptr, OI_ptr, OU_ptr, BLK: tl.constexpr):
    """IR-only: real 64-lane i64/u64 vload -> vector_broadcast -> NORM_B64."""
    i_ub = tla.alloc_buffer((BLK, ), tl.int64, space="ub")
    u_ub = tla.alloc_buffer((BLK, ), tl.uint64, space="ub")
    io_ub = tla.alloc_buffer((4, BLK), tl.int64, space="ub")
    uo_ub = tla.alloc_buffer((4, BLK), tl.uint64, space="ub")
    i = tla.view(I_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    u = tla.view(U_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    io = tla.view(OI_ptr, (4, BLK), strides=(BLK, 1), offsets=(0, 0))
    uo = tla.view(OU_ptr, (4, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(i, i_ub)
    tla.copy(u, u_ub)
    with tla.scope.simd():
        iv = tla.simd.vlds(i_ub[0], dist="NORM")
        uv = tla.simd.vlds(u_ub[0], dist="NORM")
        p_prefix = tla.simd.pge(32, BLK)
        p_none = tla.simd.vcmp(iv, iv, "ne")
        tla.simd.vsts(io_ub[0, 0], tla.simd.vdupv(iv, mask=p_prefix, pos=tla.simd.POS.LOWEST),
                      dist="NORM_B64")
        tla.simd.vsts(io_ub[1, 0], tla.simd.vdupv(iv, mask=p_prefix, pos=tla.simd.POS.HIGHEST),
                      dist="NORM_B64")
        tla.simd.vsts(io_ub[2, 0], tla.simd.vdupv(iv, mask=p_none, pos=tla.simd.POS.LOWEST),
                      dist="NORM_B64")
        tla.simd.vsts(io_ub[3, 0], tla.simd.vdupv(iv, mask=p_none, pos=tla.simd.POS.HIGHEST),
                      dist="NORM_B64")
        tla.simd.vsts(uo_ub[0, 0], tla.simd.vdupv(uv, mask=p_prefix, pos=tla.simd.POS.LOWEST),
                      dist="NORM_B64")
        tla.simd.vsts(uo_ub[1, 0], tla.simd.vdupv(uv, mask=p_prefix, pos=tla.simd.POS.HIGHEST),
                      dist="NORM_B64")
        tla.simd.vsts(uo_ub[2, 0], tla.simd.vdupv(uv, mask=p_none, pos=tla.simd.POS.LOWEST),
                      dist="NORM_B64")
        tla.simd.vsts(uo_ub[3, 0], tla.simd.vdupv(uv, mask=p_none, pos=tla.simd.POS.HIGHEST),
                      dist="NORM_B64")
    tla.copy(io_ub, io)
    tla.copy(uo_ub, uo)


@triton.jit
def simd_mask_ssa_ir(X_ptr, M_ptr, O_ptr, BLK: tl.constexpr):
    """IR-only: one prefix / arbitrary / all-false mask feeds all eight APIs."""
    x_ub = tla.alloc_buffer((2, BLK), tl.float32, space="ub")
    m_ub = tla.alloc_buffer((BLK, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((27, BLK), tl.float32, space="ub")
    x = tla.view(X_ptr, (2, BLK), strides=(BLK, 1), offsets=(0, 0))
    m = tla.view(M_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (27, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(x, x_ub)
    tla.copy(m, m_ub)
    with tla.scope.simd():
        a = tla.simd.vlds(x_ub[0, 0], dist="NORM")
        b = tla.simd.vlds(x_ub[1, 0], dist="NORM")
        mv = tla.simd.vlds(m_ub[0], dist="NORM")
        m_prefix = tla.simd.pge(32, BLK)
        zero = tla.simd.vdup(0.0, tl.float32, mask=m_prefix)
        m_arb = tla.simd.vcmp(mv, zero, "ne")
        m_none = tla.simd.vcmp(mv, mv, "ne")
        # prefix mask
        tla.simd.vsts(o_ub[0, 0], tla.simd.vdiv(a, b, mask=m_prefix))
        tla.simd.vsts(o_ub[1, 0], tla.simd.vdup(1.5, tl.float32, mask=m_prefix))
        tla.simd.vsts(o_ub[2, 0], tla.simd.vexp(a, mask=m_prefix))
        tla.simd.vsts(o_ub[3, 0], tla.simd.vmax(a, b, mask=m_prefix))
        tla.simd.vsts(o_ub[4, 0], tla.simd.vmaxs(a, 0.0, mask=m_prefix))
        tla.simd.vsts(o_ub[5, 0], tla.simd.vmin(a, b, mask=m_prefix))
        tla.simd.vsts(o_ub[6, 0], tla.simd.vdups(1.5, tl.float32, mask=m_prefix))
        tla.simd.vsts(o_ub[7, 0], tla.simd.vdupv(a, mask=m_prefix, pos=tla.simd.POS.LOWEST))
        tla.simd.vsts(o_ub[8, 0], tla.simd.vdupv(a, mask=m_prefix, pos=tla.simd.POS.HIGHEST))
        # arbitrary mask
        tla.simd.vsts(o_ub[9, 0], tla.simd.vdiv(a, b, mask=m_arb))
        tla.simd.vsts(o_ub[10, 0], tla.simd.vdup(1.5, tl.float32, mask=m_arb))
        tla.simd.vsts(o_ub[11, 0], tla.simd.vexp(a, mask=m_arb))
        tla.simd.vsts(o_ub[12, 0], tla.simd.vmax(a, b, mask=m_arb))
        tla.simd.vsts(o_ub[13, 0], tla.simd.vmaxs(a, 0.0, mask=m_arb))
        tla.simd.vsts(o_ub[14, 0], tla.simd.vmin(a, b, mask=m_arb))
        tla.simd.vsts(o_ub[15, 0], tla.simd.vdups(1.5, tl.float32, mask=m_arb))
        tla.simd.vsts(o_ub[16, 0], tla.simd.vdupv(a, mask=m_arb, pos=tla.simd.POS.LOWEST))
        tla.simd.vsts(o_ub[17, 0], tla.simd.vdupv(a, mask=m_arb, pos=tla.simd.POS.HIGHEST))
        # all-false mask
        tla.simd.vsts(o_ub[18, 0], tla.simd.vdiv(a, b, mask=m_none))
        tla.simd.vsts(o_ub[19, 0], tla.simd.vdup(1.5, tl.float32, mask=m_none))
        tla.simd.vsts(o_ub[20, 0], tla.simd.vexp(a, mask=m_none))
        tla.simd.vsts(o_ub[21, 0], tla.simd.vmax(a, b, mask=m_none))
        tla.simd.vsts(o_ub[22, 0], tla.simd.vmaxs(a, 0.0, mask=m_none))
        tla.simd.vsts(o_ub[23, 0], tla.simd.vmin(a, b, mask=m_none))
        tla.simd.vsts(o_ub[24, 0], tla.simd.vdups(1.5, tl.float32, mask=m_none))
        tla.simd.vsts(o_ub[25, 0], tla.simd.vdupv(a, mask=m_none, pos=tla.simd.POS.LOWEST))
        tla.simd.vsts(o_ub[26, 0], tla.simd.vdupv(a, mask=m_none, pos=tla.simd.POS.HIGHEST))
    tla.copy(o_ub, o)


@triton.jit
def simd_div_masks_ir(I32_ptr, I64_ptr, U64_ptr, O32_ptr, O64_ptr, OU64_ptr,
                      BLK: tl.constexpr):
    """IR-only: signed i32 and i64/u64 keep the bare divide under every mask."""
    i32_ub = tla.alloc_buffer((2, BLK), tl.int32, space="ub")
    i64_ub = tla.alloc_buffer((2, BLK), tl.int64, space="ub")
    u64_ub = tla.alloc_buffer((2, BLK), tl.uint64, space="ub")
    o32_ub = tla.alloc_buffer((3, BLK), tl.int32, space="ub")
    o64_ub = tla.alloc_buffer((3, BLK), tl.int64, space="ub")
    ou64_ub = tla.alloc_buffer((3, BLK), tl.uint64, space="ub")
    i32 = tla.view(I32_ptr, (2, BLK), strides=(BLK, 1), offsets=(0, 0))
    i64 = tla.view(I64_ptr, (2, BLK), strides=(BLK, 1), offsets=(0, 0))
    u64 = tla.view(U64_ptr, (2, BLK), strides=(BLK, 1), offsets=(0, 0))
    o32 = tla.view(O32_ptr, (3, BLK), strides=(BLK, 1), offsets=(0, 0))
    o64 = tla.view(O64_ptr, (3, BLK), strides=(BLK, 1), offsets=(0, 0))
    ou64 = tla.view(OU64_ptr, (3, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(i32, i32_ub)
    tla.copy(i64, i64_ub)
    tla.copy(u64, u64_ub)
    with tla.scope.simd():
        a32 = tla.simd.vlds(i32_ub[0, 0], dist="NORM")
        b32 = tla.simd.vlds(i32_ub[1, 0], dist="NORM")
        a64 = tla.simd.vlds(i64_ub[0, 0], dist="NORM")
        b64 = tla.simd.vlds(i64_ub[1, 0], dist="NORM")
        au64 = tla.simd.vlds(u64_ub[0, 0], dist="NORM")
        bu64 = tla.simd.vlds(u64_ub[1, 0], dist="NORM")
        p_prefix = tla.simd.pge(32, BLK)
        p_none = tla.simd.vcmp(a32, a32, "ne")
        p_arb = tla.simd.vcmp(a32, b32, "gt")
        tla.simd.vsts(o32_ub[0, 0], tla.simd.vdiv(a32, b32, mask=p_prefix))
        tla.simd.vsts(o32_ub[1, 0], tla.simd.vdiv(a32, b32, mask=p_arb))
        tla.simd.vsts(o32_ub[2, 0], tla.simd.vdiv(a32, b32, mask=p_none))
        tla.simd.vsts(o64_ub[0, 0], tla.simd.vdiv(a64, b64, mask=p_prefix), dist="NORM_B64")
        tla.simd.vsts(o64_ub[1, 0], tla.simd.vdiv(a64, b64, mask=p_arb), dist="NORM_B64")
        tla.simd.vsts(o64_ub[2, 0], tla.simd.vdiv(a64, b64, mask=p_none), dist="NORM_B64")
        tla.simd.vsts(ou64_ub[0, 0], tla.simd.vdiv(au64, bu64, mask=p_prefix), dist="NORM_B64")
        tla.simd.vsts(ou64_ub[1, 0], tla.simd.vdiv(au64, bu64, mask=p_arb), dist="NORM_B64")
        tla.simd.vsts(ou64_ub[2, 0], tla.simd.vdiv(au64, bu64, mask=p_none), dist="NORM_B64")
    tla.copy(o32_ub, o32)
    tla.copy(o64_ub, o64)
    tla.copy(ou64_ub, ou64)


@triton.jit
def simd_tail_mask_ir(A_ptr, B_ptr, C_ptr, O_ptr, N, ROW0,
                      CAP: tl.constexpr, CHUNK: tl.constexpr, CORES: tl.constexpr,
                      GIN: tl.constexpr, GOUT: tl.constexpr):
    """IR-only: runtime-N tail predicate, ALL vs masked stores, multi-chunk loop."""
    a_ub = tla.alloc_buffer((CHUNK, ), tl.float32, space="ub")
    b_ub = tla.alloc_buffer((CHUNK, ), tl.float32, space="ub")
    c_ub = tla.alloc_buffer((CHUNK, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((13, CHUNK), tl.float32, space="ub")
    NBLK: tl.constexpr = (CAP + CHUNK - 1) // CHUNK
    for p in range(tl.program_id(0), NBLK, CORES):
        off = p * CHUNK
        a = tla.view(A_ptr, (CHUNK, ), strides=(1, ), offsets=(GIN + off, ))
        b = tla.view(B_ptr, (CHUNK, ), strides=(1, ), offsets=(GIN + off, ))
        c = tla.view(C_ptr, (CHUNK, ), strides=(1, ), offsets=(GIN + off, ))
        o = tla.view(O_ptr, (13, CHUNK), strides=(CHUNK, 1), offsets=(ROW0, GIN + off))
        tla.copy(a, a_ub)
        tla.copy(b, b_ub)
        tla.copy(c, c_ub)
        tla.copy(o, o_ub)
        with tla.scope.simd():
            av = tla.simd.vlds(a_ub[0], dist="NORM")
            bv = tla.simd.vlds(b_ub[0], dist="NORM")
            cv = tla.simd.vlds(c_ub[0], dist="NORM")
            idx = tla.simd.vci(off)
            nv = tla.simd.vdup(N, tl.int32)
            tail = tla.simd.vcmp(idx, nv, "lt")
            tla.simd.vsts(o_ub[0, 0], tla.simd.vdiv(av, cv, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[1, 0], tla.simd.vmax(av, bv, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[2, 0], tla.simd.vmin(av, bv, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[3, 0], tla.simd.vmaxs(av, 0.0, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[4, 0], tla.simd.vexp(av, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[5, 0], tla.simd.vdup(1.5, tl.float32, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[6, 0], tla.simd.vdups(1.5, tl.float32, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[7, 0], tla.simd.vdupv(av, mask=tail, pos=tla.simd.POS.LOWEST), dist="NORM_B32")
            tla.simd.vsts(o_ub[8, 0], tla.simd.vdupv(av, mask=tail, pos=tla.simd.POS.HIGHEST), dist="NORM_B32")
            tla.simd.vsts(o_ub[9, 0], tla.simd.vmax(av, bv, mask=tail), dist="NORM_B32", mask=tail)
            tla.simd.vsts(o_ub[10, 0], tla.simd.vdiv(av, cv, mask=tail), dist="NORM_B32", mask=tail)
            tla.simd.vsts(o_ub[11, 0], tla.simd.vdupv(av, mask=tail, pos=tla.simd.POS.LOWEST),
                          dist="NORM_B32", mask=tail)
            tla.simd.vsts(o_ub[12, 0], tla.simd.vdupv(av, mask=tail, pos=tla.simd.POS.HIGHEST),
                          dist="NORM_B32", mask=tail)
        tla.copy(o_ub, o)


def _strip_locs(text):
    """Drop MLIR location ids so two builds can be compared structurally."""
    text = re.sub(r" loc\(#loc\d+\)", "", text)
    return "\n".join(ln for ln in text.splitlines() if not re.match(r"\s*#loc\d* = loc\(", ln))


def constexprs():
    s = SHAPE
    return {"TMAX": s["tmax"], "BLK": s["blk"], "CORES": s["cores"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(simd_elementwise, SIGNATURE, constexprs(), output_dir=output_dir)
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    check_lowered_ir(text, loads=1, stores=1, buffers=2)
    for probe in ("ave.hir.vdiv", "ave.hir.vmax", "ave.hir.vmin", "ave.hir.vmaxs",
                  "ave.hir.vexp", "ave.hir.broadcast"):
        assert probe in text, f"missing {probe} in lowered IR"
    # vdup must be the masked broadcast, never the unmasked scalar_broadcast
    assert "ave.hir.scalar_broadcast" not in text, "vdup must not degrade to vsplat"

    # unsigned op selection, runtime scalars, and full 64-lane i64/u64 registers
    module, ctx = compile_offline(simd_elementwise_dtype_ir, DTYPE_SIGNATURE, {"BLK": SHAPE["blk"]})
    module = lower_offline(module, ctx)
    dtext = str(module)
    check_lowered_ir(dtext, loads=3, stores=3, buffers=6)
    for probe in ("ave.hir.vumax", "ave.hir.vumin", "ave.hir.vumaxs", "ave.hir.vmaxs",
                  "cast_unsigned"):
        assert probe in dtext, f"missing {probe} in dtype-IR"
    assert str(I64_BIG) in dtext, "i64 scalar must be an exact integer constant"
    assert str(2**62) not in dtext.replace(str(I64_BIG), ""), \
        "i64 scalar appears to have been rounded through a double"
    # i64/u64 must come from a real 64-lane vlds, not a 32-lane bitcast hack
    assert "vbitcast" not in dtext, "i64 path must use vlds on an int64 buffer"
    assert "vector<64xi64>" in dtext, "missing full 64-lane i64/u64 registers"
    # u64 constexprs are exact two's-complement i64 bits (no float round-trip)
    assert "arith.constant -9223372036854775808 : i64" in dtext, "u64 2**63 bit pattern"
    assert "arith.constant -1 : i64" in dtext, "u64 2**64-1 bit pattern"
    # runtime scalars stay as operands rather than being folded into constants
    assert any("%n," in ln and "vumaxs" in ln for ln in dtext.splitlines()), \
        "runtime u32 scalar not forwarded to vumaxs"
    assert any("%n64," in ln and "vumaxs" in ln for ln in dtext.splitlines()), \
        "runtime u64 scalar not forwarded to vumaxs"
    print("IR PASS: simd_elementwise op selection ok "
          "(vdiv/vmax/vmin/vmaxs/vexp/broadcast, vumax/vumin/vumaxs, "
          "exact i64/u64 scalars, runtime scalars, 64-lane i64/u64)")


def test_vdup_alias_ir():
    """Real IR: every documented vdup dtype spelling maps to its canonical type."""
    cases = (
        ("i8", "vector<256xi8>"),
        ("u8", "vector<256xi8>"),
        ("i32", "vector<64xi32>"),
        ("u32", "vector<64xi32>"),
        ("f16", "vector<128xf16>"),
        ("f32", "vector<64xf32>"),
        ("bf16", "vector<128xbf16>"),
        ("fp8e4nv", "vector<256xf8E4M3FN>"),
        ("int32", "vector<64xi32>"),
        ("uint8", "vector<256xi8>"),
        ("fp16", "vector<128xf16>"),
        ("float32", "vector<64xf32>"),
    )
    for name, vec_ty in cases:
        module, ctx = compile_offline(simd_vdup_alias_ir, {"O_ptr": "*u32"}, {"DT": name})
        module = lower_offline(module, ctx)
        text = str(module)
        assert "ave.hir.broadcast" in text, name
        assert "ave.hir.scalar_broadcast" not in text, name
        assert vec_ty in text, (name, vec_ty)
    print("IR PASS: vdup dtype aliases -> ave.hir.broadcast with canonical element types")


def test_mode_ir():
    """Real IR: ZEROING is bare; MERGING adds a +0 merge source and a vsel.

    Requires a TLA extension rebuilt with tla_builder.cc::applySimdMode; a
    stale .so will still show the old bare-op MERGING IR.
    """
    texts = {}
    for mode in ("MODE_ZEROING", "MODE_MERGING"):
        module, ctx = compile_offline(simd_mode_ir, {"X_ptr": "*fp32", "O_ptr": "*fp32"},
                                      {"BLK": SHAPE["blk"], "MODE_STR": mode})
        module = lower_offline(module, ctx)
        text = str(module)
        check_lowered_ir(text, loads=1, stores=1, buffers=2)
        # the custom partial mask is kept, and each op consumes that same value
        assert text.count("ave.hir.pge <VL32>") == 1, mode
        for probe in ("ave.hir.vdiv ", "ave.hir.broadcast ", "ave.hir.vexp ",
                      "ave.hir.vmax ", "ave.hir.vmaxs ", "ave.hir.vmin "):
            assert text.count(probe) == 1, (mode, probe)
            line = next(ln for ln in text.splitlines() if probe in ln)
            assert ", %p :" in line, (mode, probe, line)
        if mode == "MODE_ZEROING":
            # the bare-op path must not pay for a merge select
            assert "ave.hir.vsel " not in text, text
            assert "ave.hir.scalar_broadcast " not in text, text
            assert "arith.select" not in text and " vector.select " not in text, text
        else:
            # one zero-init merge source (+0 broadcast) and one zero-merge
            # select per result; the zero broadcast may be CSE'd by lowering.
            assert text.count("ave.hir.scalar_broadcast ") >= 1, text
            assert text.count("ave.hir.vsel ") == 6, text
            assert "vector<64xf32>" in text, text
            for line in (ln for ln in text.splitlines() if "ave.hir.vsel " in ln):
                assert "%p" in line, line
            assert "arith.select" not in text and " vector.select " not in text, text
        texts[mode] = _strip_locs(text)
    assert texts["MODE_ZEROING"] != texts["MODE_MERGING"], \
        "MODE_MERGING must be consumed by the builder, not flattened to ZEROING"
    assert texts["MODE_MERGING"].count("ave.hir.vsel ") == 6
    print("IR PASS: ZEROING bare op; MERGING adds +0 merge source + vsel")


def test_dupv_ir():
    """Real IR: vdups is ave.hir.broadcast; vdupv low/high are vector_broadcast."""
    module, ctx = compile_offline(simd_dupv_f32_ir, {"X_ptr": "*fp32", "O_ptr": "*fp32"},
                                  {"BLK": SHAPE["blk"]})
    module = lower_offline(module, ctx)
    text = str(module)
    check_lowered_ir(text, loads=1, stores=1, buffers=2)
    assert text.count("ave.hir.broadcast ") == 1, text
    assert text.count("ave.hir.vector_broadcast ") == 2, text
    assert "ave.hir.scalar_broadcast" not in text
    low_lines = [ln for ln in text.splitlines() if "ave.hir.vector_broadcast " in ln]
    assert any("true" in ln for ln in low_lines), low_lines
    assert any("false" in ln for ln in low_lines), low_lines
    print("IR PASS: vdups -> broadcast, vdupv -> vector_broadcast low/high")

    module, ctx = compile_offline(
        simd_dupv_typed_ir,
        {"I_ptr": "*i32", "B_ptr": "*bf16", "IO_ptr": "*i32", "BO_ptr": "*bf16"},
        {"BLK": SHAPE["blk"]})
    module = lower_offline(module, ctx)
    text = str(module)
    check_lowered_ir(text, loads=2, stores=2, buffers=4)
    assert text.count("ave.hir.broadcast ") == 2, text
    assert text.count("ave.hir.vector_broadcast ") == 4, text
    assert "vector<64xi32>" in text and "vector<128xbf16>" in text
    print("IR PASS: vdups/vdupv int32 + bf16")


def test_unsigned_div_zero_ir():
    """Real IR: ZEROING keeps the unsigned fix; MERGING uses only its own select.

    Requires a rebuilt extension; see test_mode_ir.
    """
    texts = {}
    for mode in ("MODE_ZEROING", "MODE_MERGING"):
        module, ctx = compile_offline(
            simd_unsigned_div_zero_ir, {"U_ptr": "*u32", "O_ptr": "*u32"},
            {"BLK": SHAPE["blk"], "MODE_STR": mode})
        module = lower_offline(module, ctx)
        text = str(module)
        check_lowered_ir(text, loads=1, stores=1, buffers=2)
        assert text.count("ave.hir.vdiv ") == 3, (mode, text)
        assert "cast_unsigned" in text, mode
        if mode == "MODE_ZEROING":
            # prefix + all-false masks need the frontend zeroing fix; the
            # all-true row stays a bare divide.
            assert text.count("ave.hir.vsel ") == 2, (mode, text)
        else:
            # the builder merge applies to all three rows, and the frontend
            # must not stack its ZEROING select on top (that would be 5).
            assert text.count("ave.hir.vsel ") == 3, (mode, text)
        # the zero vector is a single CSE'd scalar_broadcast shared by selects
        assert text.count("ave.hir.scalar_broadcast ") == 1, (mode, text)
        texts[mode] = _strip_locs(text)
    assert texts["MODE_ZEROING"] != texts["MODE_MERGING"], \
        "MERGING must use the builder merge, not the ZEROING frontend fix"
    print("IR PASS: unsigned vdiv ZEROING fix vs MERGING builder select, no stacking")


def test_precision_exact_ir():
    """Real IR: f32 precise tiers are ave.hir.vdivfhp with the caller mask."""
    module, ctx = compile_offline(simd_precision_exact_ir,
                                  {"X_ptr": "*fp32", "O_ptr": "*fp32"},
                                  {"BLK": SHAPE["blk"]})
    module = lower_offline(module, ctx)
    text = str(module)
    check_lowered_ir(text, loads=1, stores=1, buffers=2)
    assert text.count("ave.hir.vdivfhp ") == 2, text
    assert "ave.hir.vdiv " not in text, text
    assert "vsel" not in text and "arith.select" not in text
    assert text.count("ave.hir.pge <VL32>") == 1, text
    assert text.count("ave.hir.pge <VL16>") == 1, text
    def ssa(marker):
        line = next(ln for ln in text.splitlines() if marker in ln)
        return line.split("=")[0].strip()
    masks = {ssa("ave.hir.pge <VL32>"), ssa("ave.hir.pge <VL16>")}
    for line in (ln for ln in text.splitlines() if "ave.hir.vdivfhp " in ln):
        assert set(re.findall(r"%[A-Za-z_][\w.$]*", line)) & masks, line
    # f16 precise is rejected at trace time (TileLang's wrappers are f32-only)
    # the frontend ValueError is wrapped by the Triton traceback machinery
    with pytest.raises(Exception, match="requires float32"):
        compile_offline(_exact_f16_kernel, {"X_ptr": "*fp16", "O_ptr": "*fp16"},
                        {"BLK": SHAPE["blk"]})
    print("IR PASS: f32 exact -> ave.hir.vdivfhp (mask forwarded, no VFDivOp)")


@triton.jit
def _exact_f16_kernel(X_ptr, O_ptr, BLK: tl.constexpr):
    x_ub = tla.alloc_buffer((2, BLK), tl.float16, space="ub")
    o_ub = tla.alloc_buffer((BLK, ), tl.float16, space="ub")
    x = tla.view(X_ptr, (2, BLK), strides=(BLK, 1), offsets=(0, 0))
    o = tla.view(O_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    tla.copy(x, x_ub)
    with tla.scope.simd():
        a = tla.simd.vlds(x_ub[0, 0], dist="NORM")
        b = tla.simd.vlds(x_ub[1, 0], dist="NORM")
        tla.simd.vsts(o_ub[0], tla.simd.vdiv(a, b, precision="exact"), dist="NORM_B16")
    tla.copy(o_ub, o)


def test_dupv_wide_ir():
    """Real IR: i64/u64 use a 64-lane vload and 64-lane vector_broadcast."""
    module, ctx = compile_offline(
        simd_dupv_wide_ir,
        {"I_ptr": "*i64", "U_ptr": "*u64", "OI_ptr": "*i64", "OU_ptr": "*u64"},
        {"BLK": SHAPE["blk"]})
    module = lower_offline(module, ctx)
    text = str(module)
    check_lowered_ir(text, loads=2, stores=2, buffers=4)
    assert text.count("ave.hir.vector_broadcast ") == 8, text
    assert "ave.hir.scalar_broadcast" not in text
    assert text.count("vector<64xi64>") >= 8, text
    assert "vector<64xi1>" in text
    broadcast_lines = [ln for ln in text.splitlines() if "ave.hir.vector_broadcast " in ln]
    assert sum("true" in ln for ln in broadcast_lines) == 4, broadcast_lines
    assert sum("false" in ln for ln in broadcast_lines) == 4, broadcast_lines
    print("IR PASS: i64/u64 vdupv low/high on real 64-lane vload registers")


def test_mask_ssa_connection_ir():
    """Real IR: all eight APIs consume the exact prefix/arbitrary/none mask SSA."""
    module, ctx = compile_offline(
        simd_mask_ssa_ir,
        {"X_ptr": "*fp32", "M_ptr": "*fp32", "O_ptr": "*fp32"},
        {"BLK": SHAPE["blk"]})
    module = lower_offline(module, ctx)
    text = str(module)
    check_lowered_ir(text, loads=2, stores=1, buffers=3)
    for op in ("vdiv", "vmax ", "vmin ", "vmaxs ", "vexp "):
        assert text.count(f"ave.hir.{op}") == 3, (op, text.count(f"ave.hir.{op}"))
    # zero + three distinct vdup/vdups values; vdups is an alias of vdup, so
    # identical (scalar, mask) pairs are CSE'd in the real IR
    assert text.count("ave.hir.broadcast ") == 4, text
    assert text.count("ave.hir.vector_broadcast ") == 6, text
    assert "vsel" not in text

    def ssa(marker):
        line = next(ln for ln in text.splitlines() if marker in ln)
        return line.split("=")[0].strip()

    masks = {
        ssa("ave.hir.pge <VL32>"),
        ssa("ave.hir.vcmp <NE> %mv, %zero"),
        ssa("ave.hir.vcmp <NE> %mv, %mv"),
    }
    assert len(masks) == 3, masks
    probes = ("ave.hir.vdiv ", "ave.hir.broadcast ", "ave.hir.vexp ",
              "ave.hir.vmax ", "ave.hir.vmaxs ", "ave.hir.vmin ",
              "ave.hir.vector_broadcast ")
    for probe in probes:
        lines = [ln for ln in text.splitlines() if probe in ln]
        for line in lines:
            operands = set(re.findall(r"%[A-Za-z_][\w.$]*", line))
            assert operands & masks, (probe, line)
    print("IR PASS: eight APIs connect the exact prefix/arbitrary/none mask SSA")


def test_signed_and_wide_div_masks_ir():
    """Real IR: signed i32 and i64/u64 have no unsigned-only zeroing vsel."""
    module, ctx = compile_offline(
        simd_div_masks_ir,
        {"I32_ptr": "*i32", "I64_ptr": "*i64", "U64_ptr": "*u64",
         "O32_ptr": "*i32", "O64_ptr": "*i64", "OU64_ptr": "*u64"},
        {"BLK": SHAPE["blk"]})
    module = lower_offline(module, ctx)
    text = str(module)
    check_lowered_ir(text, loads=3, stores=3, buffers=6)
    assert text.count("ave.hir.vdiv ") == 9, text
    # only the three u64 rows carry the unsigned cast; no compensation select
    assert text.count("cast_unsigned") == 3, text
    assert "ave.hir.vsel " not in text, text
    assert "ave.hir.scalar_broadcast " not in text, text
    print("IR PASS: signed i32 and i64/u64 keep the bare masked divide")


def test_tail_mask_ir():
    """Real IR: runtime-N tail predicate, ALL vs masked stores, multi-chunk loop."""
    module, ctx = compile_offline(
        simd_tail_mask_ir,
        {"A_ptr": "*fp32", "B_ptr": "*fp32", "C_ptr": "*fp32", "O_ptr": "*fp32",
         "N": "i32", "ROW0": "i32"},
        {"CAP": 320, "CHUNK": 64, "CORES": 2, "GIN": 8, "GOUT": 8})
    module = lower_offline(module, ctx)
    text = str(module)
    check_lowered_ir(text, loads=4, stores=1, buffers=4)

    # Runtime N is broadcast into a vector, the lane index comes from vci, and
    # a single signed vcmp <LT> builds the tail predicate.  No boolean combine.
    assert text.count("ave.hir.vcmp <LT>") == 1, text
    assert "ave.hir.broadcast %N," in text, text
    assert "ave.hir.vci" in text, text
    assert "ave.hir.vsel " not in text, text

    def ssa(marker):
        line = next(ln for ln in text.splitlines() if marker in ln)
        return line.split("=")[0].strip()

    tail_ssa = ssa("ave.hir.vcmp <LT>")
    all_ssa = ssa("ave.hir.pge <ALL>")
    stores = [ln for ln in text.splitlines() if "ave.hir.masked_store" in ln]
    assert len(stores) == 13, stores
    # 4 tail-masked stores use the vcmp SSA; 9 arithmetic ALL stores use the
    # independent all-true observation mask.
    assert sum(f", {tail_ssa}," in ln for ln in stores) == 4, stores
    assert sum(f", {all_ssa}," in ln for ln in stores) == 9, stores
    for op in ("ave.hir.vdiv ", "ave.hir.vmax ", "ave.hir.vmin ",
               "ave.hir.vmaxs ", "ave.hir.vexp "):
        assert op in text, op
    assert text.count("ave.hir.vector_broadcast ") == 2, text
    # pge is only the ALL mask; the dynamic 31/63/... tail is never a pge prefix.
    patterns = set(re.findall(r"ave\.hir\.pge <([^>]+)>", text))
    assert patterns == {"ALL"}, patterns
    # 5 chunks per launch, step 2: each program walks more than one chunk.
    assert re.search(r"scf\.for .* to %c5_i32 step %c2_i32", text), \
        "expected a 5-iteration / step-2 chunk loop"
    print("IR PASS: runtime-N vcmp tail predicate feeds masked stores; ALL rows use all-true")


def _launch():
    s = SHAPE
    # strictly positive inputs: b is the divisor, so no inf/NaN in the reference
    X = torch.rand(s["tmax"], s["blk"], dtype=torch.float32, device="npu") + 0.5
    O = torch.full((s["tmax"] // 2, s["blk"]), -7.0, dtype=torch.float32, device="npu")
    simd_elementwise[(s["cores"], )](X, O, s["tmax"], enable_tla_mode=True, **constexprs())
    torch.npu.synchronize()
    x = X.cpu()
    a, b = x[0::2], x[1::2]
    zero = torch.zeros_like(a)
    acc = (a / b) + torch.maximum(a, b) + torch.minimum(a, b)
    acc = acc + torch.maximum(a, zero) + torch.exp(torch.minimum(a, zero))
    keep = (torch.arange(s["blk"]) < 32).to(torch.float32)
    ref = acc * keep + 0.5 * (1.0 - keep)
    return O.cpu(), ref


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: simd_elementwise generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    out, ref = _launch()
    ok = torch.allclose(out, ref, rtol=1e-5, atol=1e-5)
    if not ok:
        bad = (out - ref).abs() > 1e-5
        print("num_mismatch:", int(bad.sum()), "first:", bad.nonzero()[:8].tolist())
    print("SIMD_ELEMENTWISE", "OK" if ok else "MISMATCH", "tmax", SHAPE["tmax"])
    assert ok, "simd_elementwise e2e mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
        test_vdup_alias_ir()
        test_mode_ir()
        test_dupv_ir()
        test_dupv_wide_ir()
        test_unsigned_div_zero_ir()
        test_precision_exact_ir()
        test_mask_ssa_connection_ir()
        test_signed_and_wide_div_masks_ir()
        test_tail_mask_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: simd_elementwise", "ir" if do_ir else "", "compile" if do_compile else "",
          "run" if do_run else "")


if __name__ == "__main__":
    main()
