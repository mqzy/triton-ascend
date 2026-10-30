"""TLA device kernels and their numpy<->torch launch adapters.

This module contains no profiler call, no environment-variable path and no
process/exit handling.  It is imported only when the TLA backend actually
executes.  The case inputs/references live in ``cases.py``.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla

from .cases import (TAIL_F16_CAP, TAIL_F16_CHUNK, TAIL_F16_NS, TAIL_F32_CAP,
                    TAIL_F32_CHUNK, TAIL_F32_NS, TAIL_GUARD, TAIL_MASKED_OPS,
                    TAIL_OPS, TAIL_ROWS, TAIL_SENTINEL, U64_OUTPUT_SENTINEL)

BLK = 64
F16_BLK = 128
U16_DIV_BLK = 128
I16_DIV_BLK = 128
U_HIGH = 0x80000000
TAIL_CORES = 2


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def tla_smoke(X_ptr, O_ptr, BLK: tl.constexpr, CORES: tl.constexpr):
    x_ub = tla.alloc_buffer((2, BLK), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((BLK, ), tl.float32, space="ub")
    for p in range(tl.program_id(0), 1, CORES):
        xy = tla.view(X_ptr, (2, BLK), strides=(BLK, 1), offsets=(0, 0))
        o = tla.view(O_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
        tla.copy(xy, x_ub)
        with tla.scope.simd():
            a = tla.simd.vlds(x_ub[0, 0], dist="NORM")
            b = tla.simd.vlds(x_ub[1, 0], dist="NORM")
            r = tla.simd.vmax(a, b)
            tla.simd.vsts(o_ub[0], r, dist="NORM_B32")
        tla.copy(o_ub, o)


@triton.jit
def tla_ops_f32(X_ptr, O_ptr, BLK: tl.constexpr):
    a_ub = tla.alloc_buffer((2, BLK), tl.float32, space="ub")
    outs = tla.alloc_buffer((12, BLK), tl.float32, space="ub")
    x = tla.view(X_ptr, (2, BLK), strides=(BLK, 1), offsets=(0, 0))
    o = tla.view(O_ptr, (12, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(x, a_ub)
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0, 0], dist="NORM")
        b = tla.simd.vlds(a_ub[1, 0], dist="NORM")
        m = tla.simd.pge(32, BLK)
        tla.simd.vsts(outs[0, 0], tla.simd.vdiv(a, b))
        tla.simd.vsts(outs[1, 0], tla.simd.vdup(1.5, tl.float32))
        tla.simd.vsts(outs[2, 0], tla.simd.vexp(a))
        tla.simd.vsts(outs[3, 0], tla.simd.vmax(a, b))
        tla.simd.vsts(outs[4, 0], tla.simd.vmaxs(a, 0.0))
        tla.simd.vsts(outs[5, 0], tla.simd.vmin(a, b))
        tla.simd.vsts(outs[6, 0], tla.simd.vdiv(a, b, mask=m))
        tla.simd.vsts(outs[7, 0], tla.simd.vdup(1.5, tl.float32, mask=m))
        tla.simd.vsts(outs[8, 0], tla.simd.vexp(a, mask=m))
        tla.simd.vsts(outs[9, 0], tla.simd.vmax(a, b, mask=m))
        tla.simd.vsts(outs[10, 0], tla.simd.vmaxs(a, 0.0, mask=m))
        tla.simd.vsts(outs[11, 0], tla.simd.vmin(a, b, mask=m))
    tla.copy(outs, o)


@triton.jit
def tla_f32_special(IN_ptr, O_ptr, BLK: tl.constexpr):
    in_ub = tla.alloc_buffer((3, BLK), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((3, BLK), tl.float32, space="ub")
    src = tla.view(IN_ptr, (3, BLK), strides=(BLK, 1), offsets=(0, 0))
    dst = tla.view(O_ptr, (3, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(src, in_ub)
    with tla.scope.simd():
        a = tla.simd.vlds(in_ub[0, 0], dist="NORM")
        b = tla.simd.vlds(in_ub[1, 0], dist="NORM")
        x = tla.simd.vlds(in_ub[2, 0], dist="NORM")
        tla.simd.vsts(o_ub[0, 0], tla.simd.vdiv(a, b, precision="ftz_true"))
        tla.simd.vsts(o_ub[1, 0], tla.simd.vexp(x))
        tla.simd.vsts(o_ub[2, 0], tla.simd.vexp(x, precision="ftz_true"))
    tla.copy(o_ub, dst)


@triton.jit
def tla_int_bounds(I_ptr, J_ptr, U_ptr, V_ptr, OI_ptr, OU_ptr, BLK: tl.constexpr):
    i_ub = tla.alloc_buffer((BLK, ), tl.int32, space="ub")
    j_ub = tla.alloc_buffer((BLK, ), tl.int32, space="ub")
    u_ub = tla.alloc_buffer((BLK, ), tl.uint32, space="ub")
    v_ub = tla.alloc_buffer((BLK, ), tl.uint32, space="ub")
    oi_ub = tla.alloc_buffer((3, BLK), tl.int32, space="ub")
    ou_ub = tla.alloc_buffer((3, BLK), tl.uint32, space="ub")
    i = tla.view(I_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    j = tla.view(J_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    u = tla.view(U_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    v = tla.view(V_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    oi = tla.view(OI_ptr, (3, BLK), strides=(BLK, 1), offsets=(0, 0))
    ou = tla.view(OU_ptr, (3, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(i, i_ub)
    tla.copy(j, j_ub)
    tla.copy(u, u_ub)
    tla.copy(v, v_ub)
    with tla.scope.simd():
        iv = tla.simd.vlds(i_ub[0], dist="NORM")
        jv = tla.simd.vlds(j_ub[0], dist="NORM")
        uv = tla.simd.vlds(u_ub[0], dist="NORM")
        vv = tla.simd.vlds(v_ub[0], dist="NORM")
        tla.simd.vsts(oi_ub[0, 0], tla.simd.vmax(iv, jv))
        tla.simd.vsts(oi_ub[1, 0], tla.simd.vmin(iv, jv))
        tla.simd.vsts(oi_ub[2, 0], tla.simd.vmaxs(iv, -1))
        tla.simd.vsts(ou_ub[0, 0], tla.simd.vmax(uv, vv))
        tla.simd.vsts(ou_ub[1, 0], tla.simd.vmin(uv, vv))
        tla.simd.vsts(ou_ub[2, 0], tla.simd.vmaxs(uv, 0x80000000))
    tla.copy(oi_ub, oi)
    tla.copy(ou_ub, ou)


@triton.jit
def tla_vdup(OI_ptr, OP_ptr, OB_ptr, OQ_ptr, L32: tl.constexpr, L16: tl.constexpr):
    oi_ub = tla.alloc_buffer((L32, ), tl.int32, space="ub")
    op_ub = tla.alloc_buffer((L32, ), tl.int32, space="ub")
    ob_ub = tla.alloc_buffer((L16, ), tl.bfloat16, space="ub")
    oq_ub = tla.alloc_buffer((L16, ), tl.bfloat16, space="ub")
    oi = tla.view(OI_ptr, (L32, ), strides=(1, ), offsets=(0, ))
    op = tla.view(OP_ptr, (L32, ), strides=(1, ), offsets=(0, ))
    ob = tla.view(OB_ptr, (L16, ), strides=(1, ), offsets=(0, ))
    oq = tla.view(OQ_ptr, (L16, ), strides=(1, ), offsets=(0, ))
    with tla.scope.simd():
        p32 = tla.simd.pge(32, L32)
        p16 = tla.simd.pge(32, L16)
        tla.simd.vsts(oi_ub[0], tla.simd.vdup(7, tl.int32), dist="NORM_B32")
        tla.simd.vsts(op_ub[0], tla.simd.vdup(7, tl.int32, mask=p32), dist="NORM_B32")
        tla.simd.vsts(ob_ub[0], tla.simd.vdup(1.5, tl.bfloat16), dist="NORM_B16")
        tla.simd.vsts(oq_ub[0], tla.simd.vdup(1.5, tl.bfloat16, mask=p16), dist="NORM_B16")
    tla.copy(oi_ub, oi)
    tla.copy(op_ub, op)
    tla.copy(ob_ub, ob)
    tla.copy(oq_ub, oq)


@triton.jit
def tla_mask_pattern(AB_ptr, M_ptr, O_ptr, BLK: tl.constexpr):
    a_ub = tla.alloc_buffer((2, BLK), tl.float32, space="ub")
    m_ub = tla.alloc_buffer((BLK, ), tl.float32, space="ub")
    outs = tla.alloc_buffer((18, BLK), tl.float32, space="ub")
    m = tla.view(M_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (18, BLK), strides=(BLK, 1), offsets=(0, 0))
    ab = tla.view(AB_ptr, (2, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(ab, a_ub)
    tla.copy(m, m_ub)
    with tla.scope.simd():
        av = tla.simd.vlds(a_ub[0, 0], dist="NORM")
        bv = tla.simd.vlds(a_ub[1, 0], dist="NORM")
        mv = tla.simd.vlds(m_ub[0], dist="NORM")
        zero = tla.simd.vdup(0.0, tl.float32)
        ma = tla.simd.vcmp(mv, zero, "ne")
        mf = tla.simd.vcmp(mv, mv, "ne")
        tla.simd.vsts(outs[0, 0], tla.simd.vdiv(av, bv, mask=ma))
        tla.simd.vsts(outs[1, 0], tla.simd.vdup(1.5, tl.float32, mask=ma))
        tla.simd.vsts(outs[2, 0], tla.simd.vexp(av, mask=ma))
        tla.simd.vsts(outs[3, 0], tla.simd.vmax(av, bv, mask=ma))
        tla.simd.vsts(outs[4, 0], tla.simd.vmaxs(av, 0.0, mask=ma))
        tla.simd.vsts(outs[5, 0], tla.simd.vmin(av, bv, mask=ma))
        tla.simd.vsts(outs[6, 0], tla.simd.vdiv(av, bv, mask=mf))
        tla.simd.vsts(outs[7, 0], tla.simd.vdup(1.5, tl.float32, mask=mf))
        tla.simd.vsts(outs[8, 0], tla.simd.vexp(av, mask=mf))
        tla.simd.vsts(outs[9, 0], tla.simd.vmax(av, bv, mask=mf))
        tla.simd.vsts(outs[10, 0], tla.simd.vmaxs(av, 0.0, mask=mf))
        tla.simd.vsts(outs[11, 0], tla.simd.vmin(av, bv, mask=mf))
        tla.simd.vsts(outs[12, 0], tla.simd.vdups(1.5, tl.float32, mask=ma))
        tla.simd.vsts(outs[13, 0], tla.simd.vdupv(av, mask=ma, pos=tla.simd.POS.LOWEST))
        tla.simd.vsts(outs[14, 0], tla.simd.vdupv(av, mask=ma, pos=tla.simd.POS.HIGHEST))
        tla.simd.vsts(outs[15, 0], tla.simd.vdups(1.5, tl.float32, mask=mf))
        tla.simd.vsts(outs[16, 0], tla.simd.vdupv(av, mask=mf, pos=tla.simd.POS.LOWEST))
        tla.simd.vsts(outs[17, 0], tla.simd.vdupv(av, mask=mf, pos=tla.simd.POS.HIGHEST))
    tla.copy(outs, o)


@triton.jit
def tla_mode_mask(AB_ptr, M_ptr, O_ptr, BLK: tl.constexpr):
    """Both modes read inactive lanes as +0; MERGING's merge source is +0.

    The TLA C++ builder lowers MERGING to the op plus a zero broadcast and a
    vsel, not to TileLang's dst-preserving merge.
    """
    a_ub = tla.alloc_buffer((2, BLK), tl.float32, space="ub")
    m_ub = tla.alloc_buffer((BLK, ), tl.float32, space="ub")
    outs = tla.alloc_buffer((12, BLK), tl.float32, space="ub")
    m = tla.view(M_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (12, BLK), strides=(BLK, 1), offsets=(0, 0))
    ab = tla.view(AB_ptr, (2, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(ab, a_ub)
    tla.copy(m, m_ub)
    with tla.scope.simd():
        av = tla.simd.vlds(a_ub[0, 0], dist="NORM")
        bv = tla.simd.vlds(a_ub[1, 0], dist="NORM")
        mv = tla.simd.vlds(m_ub[0], dist="NORM")
        zero = tla.simd.vdup(0.0, tl.float32)
        ma = tla.simd.vcmp(mv, zero, "ne")
        tla.simd.vsts(outs[0, 0], tla.simd.vdiv(av, bv, mask=ma, mode="MODE_ZEROING"))
        tla.simd.vsts(outs[1, 0], tla.simd.vdup(1.5, tl.float32, mask=ma, mode="MODE_ZEROING"))
        tla.simd.vsts(outs[2, 0], tla.simd.vexp(av, mask=ma, mode="MODE_ZEROING"))
        tla.simd.vsts(outs[3, 0], tla.simd.vmax(av, bv, mask=ma, mode="MODE_ZEROING"))
        tla.simd.vsts(outs[4, 0], tla.simd.vmaxs(av, 0.0, mask=ma, mode="MODE_ZEROING"))
        tla.simd.vsts(outs[5, 0], tla.simd.vmin(av, bv, mask=ma, mode="MODE_ZEROING"))
        tla.simd.vsts(outs[6, 0], tla.simd.vdiv(av, bv, mask=ma, mode="MODE_MERGING"))
        tla.simd.vsts(outs[7, 0], tla.simd.vdup(1.5, tl.float32, mask=ma, mode="MODE_MERGING"))
        tla.simd.vsts(outs[8, 0], tla.simd.vexp(av, mask=ma, mode="MODE_MERGING"))
        tla.simd.vsts(outs[9, 0], tla.simd.vmax(av, bv, mask=ma, mode="MODE_MERGING"))
        tla.simd.vsts(outs[10, 0], tla.simd.vmaxs(av, 0.0, mask=ma, mode="MODE_MERGING"))
        tla.simd.vsts(outs[11, 0], tla.simd.vmin(av, bv, mask=ma, mode="MODE_MERGING"))
    tla.copy(outs, o)


@triton.jit
def tla_nan_inf(AB_ptr, O_ptr, BLK: tl.constexpr):
    a_ub = tla.alloc_buffer((2, BLK), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((4, BLK), tl.float32, space="ub")
    ab = tla.view(AB_ptr, (2, BLK), strides=(BLK, 1), offsets=(0, 0))
    o = tla.view(O_ptr, (4, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(ab, a_ub)
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0, 0], dist="NORM")
        b = tla.simd.vlds(a_ub[1, 0], dist="NORM")
        tla.simd.vsts(o_ub[0, 0], tla.simd.vmax(a, b))
        tla.simd.vsts(o_ub[1, 0], tla.simd.vmin(a, b))
        tla.simd.vsts(o_ub[2, 0], tla.simd.vmaxs(a, 0.0))
        tla.simd.vsts(o_ub[3, 0], tla.simd.vmaxs(a, float("-inf")))
    tla.copy(o_ub, o)


@triton.jit
def tla_u32_vmaxs_high(U_ptr, O_ptr, BLK: tl.constexpr):
    u_ub = tla.alloc_buffer((BLK, ), tl.uint32, space="ub")
    o_ub = tla.alloc_buffer((3, BLK), tl.uint32, space="ub")
    u = tla.view(U_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (3, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(u, u_ub)
    with tla.scope.simd():
        uv = tla.simd.vlds(u_ub[0], dist="NORM")
        tla.simd.vsts(o_ub[0, 0], tla.simd.vmaxs(uv, 0x80000000))
        tla.simd.vsts(o_ub[1, 0], tla.simd.vmaxs(uv, 0x80000001))
        tla.simd.vsts(o_ub[2, 0], tla.simd.vmaxs(uv, 0xFFFFFFFF))
    tla.copy(o_ub, o)


@triton.jit
def tla_i64_u64_vmaxs(I_ptr, U_ptr, OI_ptr, OU_ptr, BLK: tl.constexpr):
    i_ub = tla.alloc_buffer((BLK, ), tl.int64, space="ub")
    u_ub = tla.alloc_buffer((BLK, ), tl.uint64, space="ub")
    oi_ub = tla.alloc_buffer((3, BLK), tl.int64, space="ub")
    ou_ub = tla.alloc_buffer((3, BLK), tl.uint64, space="ub")
    i = tla.view(I_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    u = tla.view(U_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    oi = tla.view(OI_ptr, (3, BLK), strides=(BLK, 1), offsets=(0, 0))
    ou = tla.view(OU_ptr, (3, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(i, i_ub)
    tla.copy(u, u_ub)
    with tla.scope.simd():
        iv = tla.simd.vlds(i_ub[0], dist="NORM")
        uv = tla.simd.vlds(u_ub[0], dist="NORM")
        tla.simd.vsts(oi_ub[0, 0], tla.simd.vmaxs(iv, 2**53), dist="NORM_B64")
        tla.simd.vsts(oi_ub[1, 0], tla.simd.vmaxs(iv, 2**53 + 1), dist="NORM_B64")
        tla.simd.vsts(oi_ub[2, 0], tla.simd.vmaxs(iv, 2**63 - 1), dist="NORM_B64")
        tla.simd.vsts(ou_ub[0, 0], tla.simd.vmaxs(uv, 2**63), dist="NORM_B64")
        tla.simd.vsts(ou_ub[1, 0], tla.simd.vmaxs(uv, 2**63 + 1), dist="NORM_B64")
        tla.simd.vsts(ou_ub[2, 0], tla.simd.vmaxs(uv, 2**64 - 1), dist="NORM_B64")
    tla.copy(oi_ub, oi)
    tla.copy(ou_ub, ou)


@triton.jit
def tla_u16_div_zero(AB_ptr, O_ptr, BLK: tl.constexpr):
    a_ub = tla.alloc_buffer((2, BLK), tl.uint16, space="ub")
    o_ub = tla.alloc_buffer((6, BLK), tl.uint16, space="ub")
    ab = tla.view(AB_ptr, (2, BLK), strides=(BLK, 1), offsets=(0, 0))
    o = tla.view(O_ptr, (6, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(ab, a_ub)
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0, 0], dist="NORM")
        b = tla.simd.vlds(a_ub[1, 0], dist="NORM")
        p_full = tla.simd.pge(BLK, BLK)
        p_half = tla.simd.pge(BLK // 2, BLK)
        p_none = tla.simd.vcmp(a, a, "ne")
        tla.simd.vsts(o_ub[0, 0], tla.simd.vdiv(a, b, mask=p_full, mode="MODE_ZEROING"), dist="NORM_B16")
        tla.simd.vsts(o_ub[1, 0], tla.simd.vdiv(a, b, mask=p_half, mode="MODE_ZEROING"), dist="NORM_B16")
        tla.simd.vsts(o_ub[2, 0], tla.simd.vdiv(a, b, mask=p_none, mode="MODE_ZEROING"), dist="NORM_B16")
        tla.simd.vsts(o_ub[3, 0], tla.simd.vdiv(a, b, mask=p_full, mode="MODE_MERGING"), dist="NORM_B16")
        tla.simd.vsts(o_ub[4, 0], tla.simd.vdiv(a, b, mask=p_half, mode="MODE_MERGING"), dist="NORM_B16")
        tla.simd.vsts(o_ub[5, 0], tla.simd.vdiv(a, b, mask=p_none, mode="MODE_MERGING"), dist="NORM_B16")
    tla.copy(o_ub, o)


@triton.jit
def tla_u32_div_zero(AB_ptr, O_ptr, BLK: tl.constexpr):
    a_ub = tla.alloc_buffer((2, BLK), tl.uint32, space="ub")
    o_ub = tla.alloc_buffer((6, BLK), tl.uint32, space="ub")
    ab = tla.view(AB_ptr, (2, BLK), strides=(BLK, 1), offsets=(0, 0))
    o = tla.view(O_ptr, (6, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(ab, a_ub)
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0, 0], dist="NORM")
        b = tla.simd.vlds(a_ub[1, 0], dist="NORM")
        p_full = tla.simd.pge(BLK, BLK)
        p_half = tla.simd.pge(BLK // 2, BLK)
        p_none = tla.simd.vcmp(a, a, "ne")
        tla.simd.vsts(o_ub[0, 0], tla.simd.vdiv(a, b, mask=p_full, mode="MODE_ZEROING"))
        tla.simd.vsts(o_ub[1, 0], tla.simd.vdiv(a, b, mask=p_half, mode="MODE_ZEROING"))
        tla.simd.vsts(o_ub[2, 0], tla.simd.vdiv(a, b, mask=p_none, mode="MODE_ZEROING"))
        tla.simd.vsts(o_ub[3, 0], tla.simd.vdiv(a, b, mask=p_full, mode="MODE_MERGING"))
        tla.simd.vsts(o_ub[4, 0], tla.simd.vdiv(a, b, mask=p_half, mode="MODE_MERGING"))
        tla.simd.vsts(o_ub[5, 0], tla.simd.vdiv(a, b, mask=p_none, mode="MODE_MERGING"))
    tla.copy(o_ub, o)


@triton.jit
def tla_dupv_mask(X_ptr, O_ptr, BLK: tl.constexpr):
    x_ub = tla.alloc_buffer((BLK, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((9, BLK), tl.float32, space="ub")
    x = tla.view(X_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (9, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(x, x_ub)
    with tla.scope.simd():
        xv = tla.simd.vlds(x_ub[0], dist="NORM")
        p_half = tla.simd.pge(BLK // 2, BLK)
        p_none = tla.simd.vcmp(xv, xv, "ne")
        tla.simd.vsts(o_ub[0, 0], tla.simd.vdups(1.5, tl.float32), dist="NORM_B32")
        tla.simd.vsts(o_ub[1, 0], tla.simd.vdupv(xv, pos=tla.simd.POS.LOWEST), dist="NORM_B32")
        tla.simd.vsts(o_ub[2, 0], tla.simd.vdupv(xv, pos=tla.simd.POS.HIGHEST), dist="NORM_B32")
        tla.simd.vsts(o_ub[3, 0], tla.simd.vdups(1.5, tl.float32, mask=p_half), dist="NORM_B32")
        tla.simd.vsts(o_ub[4, 0], tla.simd.vdupv(xv, mask=p_half, pos=tla.simd.POS.LOWEST), dist="NORM_B32")
        tla.simd.vsts(o_ub[5, 0], tla.simd.vdupv(xv, mask=p_half, pos=tla.simd.POS.HIGHEST), dist="NORM_B32")
        tla.simd.vsts(o_ub[6, 0], tla.simd.vdups(1.5, tl.float32, mask=p_none), dist="NORM_B32")
        tla.simd.vsts(o_ub[7, 0], tla.simd.vdupv(xv, mask=p_none, pos=tla.simd.POS.LOWEST), dist="NORM_B32")
        tla.simd.vsts(o_ub[8, 0], tla.simd.vdupv(xv, mask=p_none, pos=tla.simd.POS.HIGHEST), dist="NORM_B32")
    tla.copy(o_ub, o)


@triton.jit
def tla_div_exact(AB_ptr, O_ptr, BLK: tl.constexpr):
    ab_ub = tla.alloc_buffer((2, BLK), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((4, BLK), tl.float32, space="ub")
    ab = tla.view(AB_ptr, (2, BLK), strides=(BLK, 1), offsets=(0, 0))
    o = tla.view(O_ptr, (4, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(ab, ab_ub)
    with tla.scope.simd():
        a = tla.simd.vlds(ab_ub[0, 0], dist="NORM")
        b = tla.simd.vlds(ab_ub[1, 0], dist="NORM")
        p = tla.simd.pge(BLK // 2, BLK)
        n = tla.simd.vcmp(a, a, "ne")
        tla.simd.vsts(o_ub[0, 0], tla.simd.vdiv(a, b, precision="exact"), dist="NORM_B32")
        tla.simd.vsts(o_ub[1, 0], tla.simd.vdiv(a, b, mask=p, precision="exact"), dist="NORM_B32")
        tla.simd.vsts(o_ub[2, 0], tla.simd.vdiv(a, b, mask=n, precision="exact"), dist="NORM_B32")
        tla.simd.vsts(o_ub[3, 0],
                      tla.simd.vdiv(a, b, mask=p, precision="vdiv_0ulp_ftz_true"),
                      dist="NORM_B32")
    tla.copy(o_ub, o)


@triton.jit
def tla_div_signed_wide_masks(A16_ptr, A32_ptr, A64_ptr, AU64_ptr, IDX_ptr, REM_ptr,
                              O16_ptr, O32_ptr, O64_ptr, OU64_ptr,
                              BLK: tl.constexpr, BLK16: tl.constexpr):
    a16_ub = tla.alloc_buffer((2, BLK16), tl.int16, space="ub")
    a32_ub = tla.alloc_buffer((2, BLK), tl.int32, space="ub")
    a64_ub = tla.alloc_buffer((2, BLK), tl.int64, space="ub")
    au64_ub = tla.alloc_buffer((2, BLK), tl.uint64, space="ub")
    idx_ub = tla.alloc_buffer((BLK, ), tl.int64, space="ub")
    rem_ub = tla.alloc_buffer((BLK, ), tl.int64, space="ub")
    o16_ub = tla.alloc_buffer((3, BLK16), tl.int16, space="ub")
    o32_ub = tla.alloc_buffer((3, BLK), tl.int32, space="ub")
    o64_ub = tla.alloc_buffer((3, BLK), tl.int64, space="ub")
    ou64_ub = tla.alloc_buffer((3, BLK), tl.uint64, space="ub")
    a16 = tla.view(A16_ptr, (2, BLK16), strides=(BLK16, 1), offsets=(0, 0))
    a32 = tla.view(A32_ptr, (2, BLK), strides=(BLK, 1), offsets=(0, 0))
    a64 = tla.view(A64_ptr, (2, BLK), strides=(BLK, 1), offsets=(0, 0))
    au64 = tla.view(AU64_ptr, (2, BLK), strides=(BLK, 1), offsets=(0, 0))
    idx = tla.view(IDX_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    rem = tla.view(REM_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    o16 = tla.view(O16_ptr, (3, BLK16), strides=(BLK16, 1), offsets=(0, 0))
    o32 = tla.view(O32_ptr, (3, BLK), strides=(BLK, 1), offsets=(0, 0))
    o64 = tla.view(O64_ptr, (3, BLK), strides=(BLK, 1), offsets=(0, 0))
    ou64 = tla.view(OU64_ptr, (3, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(a16, a16_ub)
    tla.copy(a32, a32_ub)
    tla.copy(a64, a64_ub)
    tla.copy(au64, au64_ub)
    tla.copy(idx, idx_ub)
    tla.copy(rem, rem_ub)
    with tla.scope.simd():
        a16v = tla.simd.vlds(a16_ub[0, 0], dist="NORM")
        b16v = tla.simd.vlds(a16_ub[1, 0], dist="NORM")
        a32v = tla.simd.vlds(a32_ub[0, 0], dist="NORM")
        b32v = tla.simd.vlds(a32_ub[1, 0], dist="NORM")
        a64v = tla.simd.vlds(a64_ub[0, 0], dist="NORM")
        b64v = tla.simd.vlds(a64_ub[1, 0], dist="NORM")
        au64v = tla.simd.vlds(au64_ub[0, 0], dist="NORM")
        bu64v = tla.simd.vlds(au64_ub[1, 0], dist="NORM")
        iv = tla.simd.vlds(idx_ub[0], dist="NORM")
        rv = tla.simd.vlds(rem_ub[0], dist="NORM")
        p16p = tla.simd.pge(BLK16 // 2, BLK16)
        p16a = tla.simd.vcmp(a16v, b16v, "gt")
        p16n = tla.simd.vcmp(a16v, a16v, "ne")
        p32p = tla.simd.pge(BLK // 2, BLK)
        p32a = tla.simd.vcmp(a32v, b32v, "gt")
        p32n = tla.simd.vcmp(a32v, a32v, "ne")
        p64p = tla.simd.pge(BLK // 2, BLK)
        # i64/u64 share the signed-index mask; no u64 data compare.
        p64a = tla.simd.vcmp(rv, iv, "gt")
        p64n = tla.simd.vcmp(a64v, a64v, "ne")
        pu64p = tla.simd.pge(BLK // 2, BLK)
        pu64n = tla.simd.vcmp(au64v, au64v, "ne")
        tla.simd.vsts(o16_ub[0, 0], tla.simd.vdiv(a16v, b16v, mask=p16p), dist="NORM_B16")
        tla.simd.vsts(o16_ub[1, 0], tla.simd.vdiv(a16v, b16v, mask=p16a), dist="NORM_B16")
        tla.simd.vsts(o16_ub[2, 0], tla.simd.vdiv(a16v, b16v, mask=p16n), dist="NORM_B16")
        tla.simd.vsts(o32_ub[0, 0], tla.simd.vdiv(a32v, b32v, mask=p32p), dist="NORM_B32")
        tla.simd.vsts(o32_ub[1, 0], tla.simd.vdiv(a32v, b32v, mask=p32a), dist="NORM_B32")
        tla.simd.vsts(o32_ub[2, 0], tla.simd.vdiv(a32v, b32v, mask=p32n), dist="NORM_B32")
        tla.simd.vsts(o64_ub[0, 0], tla.simd.vdiv(a64v, b64v, mask=p64p), dist="NORM_B64")
        tla.simd.vsts(o64_ub[1, 0], tla.simd.vdiv(a64v, b64v, mask=p64a), dist="NORM_B64")
        tla.simd.vsts(o64_ub[2, 0], tla.simd.vdiv(a64v, b64v, mask=p64n), dist="NORM_B64")
        tla.simd.vsts(ou64_ub[0, 0], tla.simd.vdiv(au64v, bu64v, mask=pu64p), dist="NORM_B64")
        tla.simd.vsts(ou64_ub[1, 0], tla.simd.vdiv(au64v, bu64v, mask=p64a), dist="NORM_B64")
        tla.simd.vsts(ou64_ub[2, 0], tla.simd.vdiv(au64v, bu64v, mask=pu64n), dist="NORM_B64")
    tla.copy(o16_ub, o16)
    tla.copy(o32_ub, o32)
    tla.copy(o64_ub, o64)
    tla.copy(ou64_ub, ou64)


@triton.jit
def tla_wide_dupv_mask(X_ptr, U_ptr, IDX_ptr, REM_ptr, OI_ptr, OU_ptr,
                       BLK: tl.constexpr):
    x_ub = tla.alloc_buffer((BLK, ), tl.int64, space="ub")
    u_ub = tla.alloc_buffer((BLK, ), tl.uint64, space="ub")
    idx_ub = tla.alloc_buffer((BLK, ), tl.int64, space="ub")
    rem_ub = tla.alloc_buffer((BLK, ), tl.int64, space="ub")
    oi_ub = tla.alloc_buffer((6, BLK), tl.int64, space="ub")
    ou_ub = tla.alloc_buffer((6, BLK), tl.uint64, space="ub")
    x = tla.view(X_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    u = tla.view(U_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    idx = tla.view(IDX_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    rem = tla.view(REM_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    oi = tla.view(OI_ptr, (6, BLK), strides=(BLK, 1), offsets=(0, 0))
    ou = tla.view(OU_ptr, (6, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(x, x_ub)
    tla.copy(u, u_ub)
    tla.copy(idx, idx_ub)
    tla.copy(rem, rem_ub)
    with tla.scope.simd():
        xi = tla.simd.vlds(x_ub[0], dist="NORM")
        xu = tla.simd.vlds(u_ub[0], dist="NORM")
        iv = tla.simd.vlds(idx_ub[0], dist="NORM")
        rv = tla.simd.vlds(rem_ub[0], dist="NORM")
        pi = tla.simd.pge(BLK // 2, BLK)
        # Signed i64 index compare: rem > idx <=> lane % 3 != 0 (42 lanes).
        ai = tla.simd.vcmp(rv, iv, "gt")
        ni = tla.simd.vcmp(xi, xi, "ne")
        pu = tla.simd.pge(BLK // 2, BLK)
        nu = tla.simd.vcmp(xu, xu, "ne")
        tla.simd.vsts(oi_ub[0, 0], tla.simd.vdupv(xi, mask=pi, pos=tla.simd.POS.LOWEST), dist="NORM_B64")
        tla.simd.vsts(oi_ub[1, 0], tla.simd.vdupv(xi, mask=pi, pos=tla.simd.POS.HIGHEST), dist="NORM_B64")
        tla.simd.vsts(oi_ub[2, 0], tla.simd.vdupv(xi, mask=ai, pos=tla.simd.POS.LOWEST), dist="NORM_B64")
        tla.simd.vsts(oi_ub[3, 0], tla.simd.vdupv(xi, mask=ai, pos=tla.simd.POS.HIGHEST), dist="NORM_B64")
        tla.simd.vsts(oi_ub[4, 0], tla.simd.vdupv(xi, mask=ni, pos=tla.simd.POS.LOWEST), dist="NORM_B64")
        tla.simd.vsts(oi_ub[5, 0], tla.simd.vdupv(xi, mask=ni, pos=tla.simd.POS.HIGHEST), dist="NORM_B64")
        tla.simd.vsts(ou_ub[0, 0], tla.simd.vdupv(xu, mask=pu, pos=tla.simd.POS.LOWEST), dist="NORM_B64")
        tla.simd.vsts(ou_ub[1, 0], tla.simd.vdupv(xu, mask=pu, pos=tla.simd.POS.HIGHEST), dist="NORM_B64")
        # The same signed-index predicate drives both i64 and u64: the mask is
        # lane-shaped, so no u64 data compare (and no unsigned vcmp) is needed.
        tla.simd.vsts(ou_ub[2, 0], tla.simd.vdupv(xu, mask=ai, pos=tla.simd.POS.LOWEST), dist="NORM_B64")
        tla.simd.vsts(ou_ub[3, 0], tla.simd.vdupv(xu, mask=ai, pos=tla.simd.POS.HIGHEST), dist="NORM_B64")
        tla.simd.vsts(ou_ub[4, 0], tla.simd.vdupv(xu, mask=nu, pos=tla.simd.POS.LOWEST), dist="NORM_B64")
        tla.simd.vsts(ou_ub[5, 0], tla.simd.vdupv(xu, mask=nu, pos=tla.simd.POS.HIGHEST), dist="NORM_B64")
    tla.copy(oi_ub, oi)
    tla.copy(ou_ub, ou)


@triton.jit
def tla_u64_vcmp_diag(U_ptr, THR_ptr, ONE_ptr, ZERO_ptr, O_ptr, BLK: tl.constexpr):
    """Diagnostic only: materialise the u64 vcmp predicate as u64 0/1 lanes."""
    u_ub = tla.alloc_buffer((BLK, ), tl.uint64, space="ub")
    thr_ub = tla.alloc_buffer((BLK, ), tl.uint64, space="ub")
    one_ub = tla.alloc_buffer((BLK, ), tl.uint64, space="ub")
    zero_ub = tla.alloc_buffer((BLK, ), tl.uint64, space="ub")
    o_ub = tla.alloc_buffer((BLK, ), tl.uint64, space="ub")
    u = tla.view(U_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    thr = tla.view(THR_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    one = tla.view(ONE_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    zero = tla.view(ZERO_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (BLK, ), strides=(1, ), offsets=(0, ))
    tla.copy(u, u_ub)
    tla.copy(thr, thr_ub)
    tla.copy(one, one_ub)
    tla.copy(zero, zero_ub)
    with tla.scope.simd():
        uv = tla.simd.vlds(u_ub[0], dist="NORM")
        tv = tla.simd.vlds(thr_ub[0], dist="NORM")
        ov = tla.simd.vlds(one_ub[0], dist="NORM")
        zv = tla.simd.vlds(zero_ub[0], dist="NORM")
        m = tla.simd.vcmp(uv, tv, "gt")
        tla.simd.vsts(o_ub[0], tla.simd.vsel(m, ov, zv), dist="NORM_B64")
    tla.copy(o_ub, o)


@triton.jit
def tla_f16_ops(AB_ptr, O_ptr, BLK: tl.constexpr):
    a_ub = tla.alloc_buffer((2, BLK), tl.float16, space="ub")
    o_ub = tla.alloc_buffer((6, BLK), tl.float16, space="ub")
    ab = tla.view(AB_ptr, (2, BLK), strides=(BLK, 1), offsets=(0, 0))
    o = tla.view(O_ptr, (6, BLK), strides=(BLK, 1), offsets=(0, 0))
    tla.copy(ab, a_ub)
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0, 0], dist="NORM")
        b = tla.simd.vlds(a_ub[1, 0], dist="NORM")
        m = tla.simd.pge(64, BLK)
        tla.simd.vsts(o_ub[0, 0], tla.simd.vdiv(a, b), dist="NORM_B16")
        tla.simd.vsts(o_ub[1, 0], tla.simd.vexp(a), dist="NORM_B16")
        tla.simd.vsts(o_ub[2, 0], tla.simd.vmax(a, b), dist="NORM_B16")
        tla.simd.vsts(o_ub[3, 0], tla.simd.vmin(a, b), dist="NORM_B16")
        tla.simd.vsts(o_ub[4, 0], tla.simd.vdup(1.5, tl.float16), dist="NORM_B16")
        tla.simd.vsts(o_ub[5, 0], tla.simd.vdup(1.5, tl.float16, mask=m), dist="NORM_B16")
    tla.copy(o_ub, o)


@triton.jit
def tla_multi_block(X_ptr, O_ptr, ROWS: tl.constexpr, BLK: tl.constexpr, CORES: tl.constexpr):
    x_ub = tla.alloc_buffer((2, BLK), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((2, BLK), tl.float32, space="ub")
    for p in range(tl.program_id(0), ROWS // 2, CORES):
        x = tla.view(X_ptr, (2, BLK), strides=(BLK, 1), offsets=(2 * p, 0))
        o = tla.view(O_ptr, (2, BLK), strides=(BLK, 1), offsets=(2 * p, 0))
        tla.copy(x, x_ub)
        with tla.scope.simd():
            a = tla.simd.vlds(x_ub[0, 0], dist="NORM")
            b = tla.simd.vlds(x_ub[1, 0], dist="NORM")
            tla.simd.vsts(o_ub[0, 0], tla.simd.vmax(a, b))
            tla.simd.vsts(o_ub[1, 0], tla.simd.vdiv(a, b))
        tla.copy(o_ub, o)


# ---------------------------------------------------------------------------
# long / tail matrix: runtime N, padded capacity, per-program multi-chunk loop
# ---------------------------------------------------------------------------


@triton.jit
def tla_tail_f32(A_ptr, B_ptr, C_ptr, O_ptr, N, ROW0,
                 CAP: tl.constexpr, CHUNK: tl.constexpr, CORES: tl.constexpr,
                 GIN: tl.constexpr, GOUT: tl.constexpr):
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
        # Seed the UB rows from the current GM rows.  Arithmetic rows are then
        # fully overwritten by the ALL store, while masked-store rows keep the
        # non-zero sentinel in every lane >= N.
        tla.copy(o, o_ub)
        with tla.scope.simd():
            av = tla.simd.vlds(a_ub[0], dist="NORM")
            bv = tla.simd.vlds(b_ub[0], dist="NORM")
            cv = tla.simd.vlds(c_ub[0], dist="NORM")
            idx = tla.simd.vci(off)
            nv = tla.simd.vdup(N, tl.int32)
            tail = tla.simd.vcmp(idx, nv, "lt")
            # ALL-store arithmetic rows: inactive data lanes must be +0.
            tla.simd.vsts(o_ub[0, 0], tla.simd.vdiv(av, cv, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[1, 0], tla.simd.vmax(av, bv, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[2, 0], tla.simd.vmin(av, bv, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[3, 0], tla.simd.vmaxs(av, 0.0, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[4, 0], tla.simd.vexp(av, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[5, 0], tla.simd.vdup(1.5, tl.float32, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[6, 0], tla.simd.vdups(1.5, tl.float32, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[7, 0], tla.simd.vdupv(av, mask=tail, pos=tla.simd.POS.LOWEST), dist="NORM_B32")
            tla.simd.vsts(o_ub[8, 0], tla.simd.vdupv(av, mask=tail, pos=tla.simd.POS.HIGHEST), dist="NORM_B32")
            # Masked-store rows: only lanes < N are written.
            tla.simd.vsts(o_ub[9, 0], tla.simd.vmax(av, bv, mask=tail), dist="NORM_B32", mask=tail)
            tla.simd.vsts(o_ub[10, 0], tla.simd.vdiv(av, cv, mask=tail), dist="NORM_B32", mask=tail)
            tla.simd.vsts(o_ub[11, 0], tla.simd.vdupv(av, mask=tail, pos=tla.simd.POS.LOWEST),
                          dist="NORM_B32", mask=tail)
            tla.simd.vsts(o_ub[12, 0], tla.simd.vdupv(av, mask=tail, pos=tla.simd.POS.HIGHEST),
                          dist="NORM_B32", mask=tail)
        tla.copy(o_ub, o)


@triton.jit
def tla_tail_f16(A_ptr, B_ptr, C_ptr, I_ptr, O_ptr, N, ROW0,
                 CAP: tl.constexpr, CHUNK: tl.constexpr, CORES: tl.constexpr,
                 GIN: tl.constexpr, GOUT: tl.constexpr):
    a_ub = tla.alloc_buffer((CHUNK, ), tl.float16, space="ub")
    b_ub = tla.alloc_buffer((CHUNK, ), tl.float16, space="ub")
    c_ub = tla.alloc_buffer((CHUNK, ), tl.float16, space="ub")
    i_ub = tla.alloc_buffer((CHUNK, ), tl.int16, space="ub")
    o_ub = tla.alloc_buffer((13, CHUNK), tl.float16, space="ub")
    NBLK: tl.constexpr = (CAP + CHUNK - 1) // CHUNK
    for p in range(tl.program_id(0), NBLK, CORES):
        off = p * CHUNK
        a = tla.view(A_ptr, (CHUNK, ), strides=(1, ), offsets=(GIN + off, ))
        b = tla.view(B_ptr, (CHUNK, ), strides=(1, ), offsets=(GIN + off, ))
        c = tla.view(C_ptr, (CHUNK, ), strides=(1, ), offsets=(GIN + off, ))
        i = tla.view(I_ptr, (CHUNK, ), strides=(1, ), offsets=(GIN + off, ))
        o = tla.view(O_ptr, (13, CHUNK), strides=(CHUNK, 1), offsets=(ROW0, GIN + off))
        tla.copy(a, a_ub)
        tla.copy(b, b_ub)
        tla.copy(c, c_ub)
        tla.copy(i, i_ub)
        tla.copy(o, o_ub)
        with tla.scope.simd():
            av = tla.simd.vlds(a_ub[0], dist="NORM")
            bv = tla.simd.vlds(b_ub[0], dist="NORM")
            cv = tla.simd.vlds(c_ub[0], dist="NORM")
            iv = tla.simd.vlds(i_ub[0], dist="NORM")
            nv = tla.simd.vdup(N.to(tl.int16), tl.int16)
            tail = tla.simd.vcmp(iv, nv, "lt")
            tla.simd.vsts(o_ub[0, 0], tla.simd.vdiv(av, cv, mask=tail), dist="NORM_B16")
            tla.simd.vsts(o_ub[1, 0], tla.simd.vmax(av, bv, mask=tail), dist="NORM_B16")
            tla.simd.vsts(o_ub[2, 0], tla.simd.vmin(av, bv, mask=tail), dist="NORM_B16")
            tla.simd.vsts(o_ub[3, 0], tla.simd.vmaxs(av, 0.0, mask=tail), dist="NORM_B16")
            tla.simd.vsts(o_ub[4, 0], tla.simd.vexp(av, mask=tail), dist="NORM_B16")
            tla.simd.vsts(o_ub[5, 0], tla.simd.vdup(1.5, tl.float16, mask=tail), dist="NORM_B16")
            tla.simd.vsts(o_ub[6, 0], tla.simd.vdups(1.5, tl.float16, mask=tail), dist="NORM_B16")
            tla.simd.vsts(o_ub[7, 0], tla.simd.vdupv(av, mask=tail, pos=tla.simd.POS.LOWEST), dist="NORM_B16")
            tla.simd.vsts(o_ub[8, 0], tla.simd.vdupv(av, mask=tail, pos=tla.simd.POS.HIGHEST), dist="NORM_B16")
            tla.simd.vsts(o_ub[9, 0], tla.simd.vmax(av, bv, mask=tail), dist="NORM_B16", mask=tail)
            tla.simd.vsts(o_ub[10, 0], tla.simd.vdiv(av, cv, mask=tail), dist="NORM_B16", mask=tail)
            tla.simd.vsts(o_ub[11, 0], tla.simd.vdupv(av, mask=tail, pos=tla.simd.POS.LOWEST),
                          dist="NORM_B16", mask=tail)
            tla.simd.vsts(o_ub[12, 0], tla.simd.vdupv(av, mask=tail, pos=tla.simd.POS.HIGHEST),
                          dist="NORM_B16", mask=tail)
        tla.copy(o_ub, o)


# ---------------------------------------------------------------------------
# numpy <-> torch adapters
# ---------------------------------------------------------------------------


def _to_npu(array):
    arr = np.ascontiguousarray(array)
    if arr.dtype == np.uint16:
        # bf16 raw bits travel as int16; torch has no numpy bf16 dtype.
        t = torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    else:
        t = torch.from_numpy(arr)
    return t.npu()


def _from_npu(tensor, bf16_bits=False):
    arr = tensor.detach().cpu()
    if bf16_bits:
        return arr.view(torch.int16).numpy().view(np.uint16)
    return arr.numpy()


def _to_npu_u16(array):
    """uint16 has no numpy<->torch direct dtype bridge on every torch build."""
    arr = np.ascontiguousarray(array)
    return torch.from_numpy(arr.view(np.int16)).view(torch.uint16).npu()


def _from_npu_u16(tensor):
    return tensor.detach().cpu().view(torch.int16).numpy().view(np.uint16)


def _to_npu_u64(array):
    arr = np.ascontiguousarray(array)
    return torch.from_numpy(arr.view(np.int64)).view(torch.uint64).npu()


def _from_npu_u64(tensor):
    return tensor.detach().cpu().view(torch.int64).numpy().view(np.uint64)


def _dev(shape, dtype, fill):
    return torch.full(shape, fill, dtype=dtype).npu()


def _launch_smoke(arrays):
    X = torch.stack([_to_npu(arrays["a"]), _to_npu(arrays["b"])])
    O = _dev((BLK, ), torch.float32, -12345.0)
    tla_smoke[(1,)](X, O, BLK=BLK, CORES=1, enable_tla_mode=True)
    torch.npu.synchronize()
    return {"out": _from_npu(O)}


def _launch_combo(arrays):
    AB = torch.stack([_to_npu(arrays["a"]), _to_npu(arrays["b"])])
    O = _dev((12, BLK), torch.float32, -12345.0)
    tla_ops_f32[(1,)](AB, O, BLK=BLK, enable_tla_mode=True)
    torch.npu.synchronize()
    names = ("OD", "OU", "OE", "OX", "OS", "ON", "PD", "PU", "PE", "PX", "PS", "PN")
    got = _from_npu(O)
    return {name: got[i] for i, name in enumerate(names)}


def _launch_special(arrays):
    IN = torch.stack([_to_npu(arrays["a"]), _to_npu(arrays["b"]), _to_npu(arrays["x"])])
    O = _dev((3, BLK), torch.float32, -12345.0)
    tla_f32_special[(1,)](IN, O, BLK=BLK, enable_tla_mode=True)
    torch.npu.synchronize()
    got = _from_npu(O)
    return {"div": got[0], "exp_default": got[1], "exp_ftz_true": got[2]}


def _launch_int(arrays):
    I = _to_npu(arrays["i"])
    J = _to_npu(arrays["j"])
    U = _to_npu(arrays["u"])
    V = _to_npu(arrays["v"])
    OI = _dev((3, BLK), torch.int32, -7)
    OU = torch.zeros((3, BLK), dtype=torch.uint32).npu()
    tla_int_bounds[(1,)](I, J, U, V, OI, OU, BLK=BLK, enable_tla_mode=True)
    torch.npu.synchronize()
    oi, ou = _from_npu(OI), _from_npu(OU)
    return {
        "i_max": oi[0], "i_min": oi[1], "i_maxs": oi[2],
        "u_max": ou[0], "u_min": ou[1], "u_maxs": ou[2],
    }


def _launch_vdup(_arrays):
    OI = _dev((BLK, ), torch.int32, -7)
    OP = _dev((BLK, ), torch.int32, -7)
    OB = _dev((F16_BLK, ), torch.bfloat16, -7.0)
    OQ = _dev((F16_BLK, ), torch.bfloat16, -7.0)
    tla_vdup[(1,)](OI, OP, OB, OQ, L32=BLK, L16=F16_BLK, enable_tla_mode=True)
    torch.npu.synchronize()
    return {
        "OI": _from_npu(OI), "OP": _from_npu(OP),
        "OB": _from_npu(OB, bf16_bits=True), "OQ": _from_npu(OQ, bf16_bits=True),
    }


def _launch_mask(arrays):
    AB = torch.stack([_to_npu(arrays["a"]), _to_npu(arrays["b"])])
    M = _to_npu(arrays["m_arb"])
    O = _dev((18, BLK), torch.float32, -12345.0)
    # A/B are adjacent in one buffer; the kernel views them as (2, BLK).
    tla_mask_pattern[(1,)](AB, M, O, BLK=BLK, enable_tla_mode=True)
    torch.npu.synchronize()
    got = _from_npu(O)
    names = ("vdiv", "vdup", "vexp", "vmax", "vmaxs", "vmin")
    out = {}
    for i, name in enumerate(names):
        out[f"arb_{name}"] = got[i]
        out[f"none_{name}"] = got[i + 6]
    extra = ("vdups", "vdupv_low", "vdupv_high")
    for i, name in enumerate(extra):
        out[f"arb_{name}"] = got[12 + i]
        out[f"none_{name}"] = got[15 + i]
    return out


def _launch_mode(arrays):
    AB = torch.stack([_to_npu(arrays["a"]), _to_npu(arrays["b"])])
    M = _to_npu(arrays["m_arb"])
    O = _dev((12, BLK), torch.float32, -12345.0)
    tla_mode_mask[(1,)](AB, M, O, BLK=BLK, enable_tla_mode=True)
    torch.npu.synchronize()
    got = _from_npu(O)
    names = ("D", "U", "E", "X", "S", "N")
    out = {}
    for i, name in enumerate(names):
        out[f"Z{name}"] = got[i]
        out[f"M{name}"] = got[i + 6]
    return out


def _launch_nan_inf(arrays):
    AB = torch.stack([_to_npu(arrays["a"]), _to_npu(arrays["b"])])
    O = _dev((4, BLK), torch.float32, -12345.0)
    tla_nan_inf[(1,)](AB, O, BLK=BLK, enable_tla_mode=True)
    torch.npu.synchronize()
    got = _from_npu(O)
    return {"vmax": got[0], "vmin": got[1], "vmaxs_0": got[2], "vmaxs_neginf": got[3]}


def _launch_u32_high(arrays):
    U = _to_npu(arrays["u"])
    O = torch.zeros((3, BLK), dtype=torch.uint32).npu()
    tla_u32_vmaxs_high[(1,)](U, O, BLK=BLK, enable_tla_mode=True)
    torch.npu.synchronize()
    got = _from_npu(O)
    return {"s_80000000": got[0], "s_80000001": got[1], "s_ffffffff": got[2]}


def _launch_i64(arrays):
    I = _to_npu(arrays["i"])
    U = _to_npu(arrays["u"])
    OI = _dev((3, BLK), torch.int64, -7)
    OU = _dev((3, BLK), torch.uint64, U64_OUTPUT_SENTINEL)
    tla_i64_u64_vmaxs[(1,)](I, U, OI, OU, BLK=BLK, enable_tla_mode=True)
    torch.npu.synchronize()
    oi, ou = _from_npu(OI), _from_npu(OU)
    return {
        "i_s53": oi[0], "i_s53p1": oi[1], "i_s63m1": oi[2],
        "u_s63": ou[0], "u_s63p1": ou[1], "u_smax": ou[2],
    }


def _launch_u16_div(arrays):
    AB = torch.stack([_to_npu_u16(arrays["a"]), _to_npu_u16(arrays["b"])])
    O = torch.full((6, U16_DIV_BLK), 0x5A5A, dtype=torch.uint16).npu()
    tla_u16_div_zero[(1,)](AB, O, BLK=U16_DIV_BLK, enable_tla_mode=True)
    torch.npu.synchronize()
    got = _from_npu_u16(O)
    names = ("Zfull", "Zhalf", "Znone", "Mfull", "Mhalf", "Mnone")
    return {name: got[i] for i, name in enumerate(names)}


def _launch_u32_div(arrays):
    AB = torch.stack([_to_npu(arrays["a"]), _to_npu(arrays["b"])])
    O = torch.full((6, BLK), 0x5A5A5A5A, dtype=torch.uint32).npu()
    tla_u32_div_zero[(1,)](AB, O, BLK=BLK, enable_tla_mode=True)
    torch.npu.synchronize()
    got = _from_npu(O)
    names = ("Zfull", "Zhalf", "Znone", "Mfull", "Mhalf", "Mnone")
    return {name: got[i] for i, name in enumerate(names)}


def _launch_dupv(arrays):
    X = _to_npu(arrays["x"])
    O = _dev((9, BLK), torch.float32, -12345.0)
    tla_dupv_mask[(1,)](X, O, BLK=BLK, enable_tla_mode=True)
    torch.npu.synchronize()
    got = _from_npu(O)
    names = ("scalar_full", "low_full", "high_full",
             "scalar_half", "low_half", "high_half",
             "scalar_none", "low_none", "high_none")
    return {name: got[i] for i, name in enumerate(names)}


def _launch_div_exact(arrays):
    AB = torch.stack([_to_npu(arrays["a"]), _to_npu(arrays["b"])])
    O = _dev((4, BLK), torch.float32, -12345.0)
    tla_div_exact[(1,)](AB, O, BLK=BLK, enable_tla_mode=True)
    torch.npu.synchronize()
    got = _from_npu(O)
    names = ("full", "prefix", "none", "alias_prefix")
    return {name: got[i] for i, name in enumerate(names)}


def _launch_div_signed_wide(arrays):
    A16 = torch.stack([_to_npu(arrays["a16"]), _to_npu(arrays["b16"])])
    A32 = torch.stack([_to_npu(arrays["a32"]), _to_npu(arrays["b32"])])
    A64 = torch.stack([_to_npu(arrays["a64"]), _to_npu(arrays["b64"])])
    AU64 = torch.stack([_to_npu_u64(arrays["au64"]), _to_npu_u64(arrays["bu64"])])
    IDX = _to_npu(arrays["idx64"])
    REM = _to_npu(arrays["rem64"])
    O16 = torch.full((3, I16_DIV_BLK), -7, dtype=torch.int16).npu()
    O32 = torch.full((3, BLK), -7, dtype=torch.int32).npu()
    O64 = torch.full((3, BLK), -7, dtype=torch.int64).npu()
    OU64 = torch.full((3, BLK), U64_OUTPUT_SENTINEL, dtype=torch.uint64).npu()
    tla_div_signed_wide_masks[(1,)](A16, A32, A64, AU64, IDX, REM, O16, O32, O64, OU64,
                                    BLK=BLK, BLK16=I16_DIV_BLK, enable_tla_mode=True)
    torch.npu.synchronize()
    g16, g32, g64 = _from_npu(O16), _from_npu(O32), _from_npu(O64)
    gu64 = _from_npu_u64(OU64)
    out = {}
    for tag, got in (("i16", g16), ("i32", g32), ("i64", g64), ("u64", gu64)):
        for i, kind in enumerate(("prefix", "arb", "none")):
            out[f"{tag}_{kind}"] = got[i]
    return out


def _launch_wide_dupv(arrays):
    x = arrays["x"]
    X = _to_npu(x)
    U = _to_npu_u64(x.view(np.uint64))
    IDX = _to_npu(arrays["idx"])
    REM = _to_npu(arrays["rem"])
    OI = torch.full((6, BLK), -7, dtype=torch.int64).npu()
    OU = torch.full((6, BLK), U64_OUTPUT_SENTINEL, dtype=torch.uint64).npu()
    tla_wide_dupv_mask[(1,)](X, U, IDX, REM, OI, OU, BLK=BLK, enable_tla_mode=True)
    torch.npu.synchronize()
    gi, gu = _from_npu(OI), _from_npu_u64(OU)
    out = {}
    names = ("prefix_low", "prefix_high", "arb_low", "arb_high", "none_low", "none_high")
    for i, name in enumerate(names):
        out[f"i64_{name}"] = gi[i]
        out[f"u64_{name}"] = gu[i]
    return out


def _launch_u64_vcmp_diag(arrays):
    U = _to_npu_u64(arrays["u"])
    THR = _to_npu_u64(arrays["thr"])
    ONE = _to_npu_u64(arrays["ones"])
    ZERO = _to_npu_u64(arrays["zeros"])
    O = torch.full((BLK, ), U64_OUTPUT_SENTINEL, dtype=torch.uint64).npu()
    tla_u64_vcmp_diag[(1,)](U, THR, ONE, ZERO, O, BLK=BLK, enable_tla_mode=True)
    torch.npu.synchronize()
    return {"mask": _from_npu_u64(O)}


def _tail_launch_common(arrays, kernel, ns, cap, chunk, with_index):
    A = _to_npu(arrays["a"])
    B = _to_npu(arrays["b"])
    C = _to_npu(arrays["c"])
    I = _to_npu(arrays["idx"]) if with_index else None
    dtype = torch.float16 if with_index else torch.float32
    width = TAIL_GUARD + cap + TAIL_GUARD
    O = torch.full((len(ns) * TAIL_ROWS, width), TAIL_SENTINEL, dtype=dtype).npu()
    for n_index, n in enumerate(ns):
        args = [A, B, C] if not with_index else [A, B, C, I]
        kernel[(TAIL_CORES,)](
            *args, O, n, n_index * TAIL_ROWS,
            CAP=cap, CHUNK=chunk, CORES=TAIL_CORES,
            GIN=TAIL_GUARD, GOUT=TAIL_GUARD, enable_tla_mode=True)
    torch.npu.synchronize()
    got = _from_npu(O)
    out = {}
    for n_index, n in enumerate(ns):
        base = n_index * TAIL_ROWS
        for i, op in enumerate(TAIL_OPS):
            out[f"n{n}_{op}"] = got[base + i]
        for i, op in enumerate(TAIL_MASKED_OPS):
            out[f"n{n}_masked_{op}"] = got[base + len(TAIL_OPS) + i]
    return out


def _launch_tail_f32(arrays):
    return _tail_launch_common(arrays, tla_tail_f32, TAIL_F32_NS,
                               TAIL_F32_CAP, TAIL_F32_CHUNK, with_index=False)


def _launch_tail_f16(arrays):
    return _tail_launch_common(arrays, tla_tail_f16, TAIL_F16_NS,
                               TAIL_F16_CAP, TAIL_F16_CHUNK, with_index=True)


def _launch_f16(arrays):
    AB = torch.stack([_to_npu(arrays["a"]), _to_npu(arrays["b"])])
    O = _dev((6, F16_BLK), torch.float16, -7.0)
    tla_f16_ops[(1,)](AB, O, BLK=F16_BLK, enable_tla_mode=True)
    torch.npu.synchronize()
    got = _from_npu(O)
    names = ("div", "exp", "max", "min", "dup_full", "dup_part")
    return {name: got[i] for i, name in enumerate(names)}


def _launch_multi(arrays):
    X = _to_npu(arrays["x"])
    O = _dev((16, BLK), torch.float32, -12345.0)
    tla_multi_block[(4,)](X, O, ROWS=16, BLK=BLK, CORES=4, enable_tla_mode=True)
    torch.npu.synchronize()
    return {"out": _from_npu(O)}


LAUNCHERS = {
    "smoke": _launch_smoke,
    "f32_combo": _launch_combo,
    "f32_special": _launch_special,
    "int_bounds": _launch_int,
    "vdup_int_bf16": _launch_vdup,
    "f32_mask_pattern": _launch_mask,
    "f32_mode_mask": _launch_mode,
    "f32_nan_inf": _launch_nan_inf,
    "u32_vmaxs_high": _launch_u32_high,
    "i64_u64_vmaxs": _launch_i64,
    "f16_ops": _launch_f16,
    "u16_div_zero": _launch_u16_div,
    "u32_div_zero": _launch_u32_div,
    "f32_dupv_mask": _launch_dupv,
    "f32_div_exact": _launch_div_exact,
    "div_signed_wide_masks": _launch_div_signed_wide,
    "i64_u64_dupv_mask": _launch_wide_dupv,
    "diagnostic_u64_vcmp_mask": _launch_u64_vcmp_diag,
    "f32_long_tail": _launch_tail_f32,
    "f16_long_tail": _launch_tail_f16,
    "f32_multi_block": _launch_multi,
}
