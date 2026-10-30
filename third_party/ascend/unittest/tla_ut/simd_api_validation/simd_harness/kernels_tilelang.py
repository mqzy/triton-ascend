"""TileLang device kernels and numpy<->torch launch adapters.

Mirrors the TLA cases but uses the TileLang Ascend API.  No profiler, no
environment paths, no exit handling.  The TileLang Python environment is
selected by the caller; this module does not install or pin anything.
"""

from __future__ import annotations

import numpy as np
import torch
import tilelang
import tilelang.ascend.language as T

BLK = 64
F16_BLK = 128
U16_DIV_BLK = 128


def _to_npu(array):
    arr = np.ascontiguousarray(array)
    if arr.dtype == np.uint16:
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
    arr = np.ascontiguousarray(array)
    return torch.from_numpy(arr.view(np.int16)).view(torch.uint16).npu()


def _from_npu_u16(tensor):
    return tensor.detach().cpu().view(torch.int16).numpy().view(np.uint16)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def _build_smoke():
    @T.prim_func
    def kernel(A: T.Tensor((BLK,), "float32"),
               B: T.Tensor((BLK,), "float32"),
               O: T.Tensor((BLK,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((BLK,), "float32")
            b_ub = T.alloc_shared((BLK,), "float32")
            o_ub = T.alloc_shared((BLK,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                a = T.simd.vld(a_ub[0])
                b = T.simd.vld(b_ub[0])
                T.simd.vsts(o_ub[0], T.simd.vmax(a, b, full), full)
            T.copy(o_ub, O)

    return kernel


def _build_combo():
    @T.prim_func
    def kernel(A: T.Tensor((BLK,), "float32"),
               B: T.Tensor((BLK,), "float32"),
               O: T.Tensor((12, BLK), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((BLK,), "float32")
            b_ub = T.alloc_shared((BLK,), "float32")
            o_ub = T.alloc_shared((12, BLK), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                part = T.simd.pset(32, "PAT_VL32")
                a = T.simd.vld(a_ub[0])
                b = T.simd.vld(b_ub[0])
                half = T.cast(1.5, "float32")
                zero = T.cast(0.0, "float32")
                T.simd.vsts(o_ub[0, 0], T.simd.vdiv(a, b, full, precision="ftz_true"), full)
                T.simd.vsts(o_ub[1, 0], T.simd.vdup(half, "float32", full), full)
                T.simd.vsts(o_ub[2, 0], T.simd.vexp(a, full, precision="ftz_true"), full)
                T.simd.vsts(o_ub[3, 0], T.simd.vmax(a, b, full), full)
                T.simd.vsts(o_ub[4, 0], T.simd.vmaxs(a, zero, full), full)
                T.simd.vsts(o_ub[5, 0], T.simd.vmin(a, b, full), full)
                T.simd.vsts(o_ub[6, 0], T.simd.vdiv(a, b, part, precision="ftz_true"), full)
                T.simd.vsts(o_ub[7, 0], T.simd.vdup(half, "float32", part), full)
                T.simd.vsts(o_ub[8, 0], T.simd.vexp(a, part, precision="ftz_true"), full)
                T.simd.vsts(o_ub[9, 0], T.simd.vmax(a, b, part), full)
                T.simd.vsts(o_ub[10, 0], T.simd.vmaxs(a, zero, part), full)
                T.simd.vsts(o_ub[11, 0], T.simd.vmin(a, b, part), full)
            T.copy(o_ub, O)

    return kernel


def _build_special():
    @T.prim_func
    def kernel(A: T.Tensor((BLK,), "float32"),
               B: T.Tensor((BLK,), "float32"),
               X: T.Tensor((BLK,), "float32"),
               O: T.Tensor((3, BLK), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((BLK,), "float32")
            b_ub = T.alloc_shared((BLK,), "float32")
            x_ub = T.alloc_shared((BLK,), "float32")
            o_ub = T.alloc_shared((3, BLK), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.copy(X, x_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                a = T.simd.vld(a_ub[0])
                b = T.simd.vld(b_ub[0])
                x = T.simd.vld(x_ub[0])
                T.simd.vsts(o_ub[0, 0], T.simd.vdiv(a, b, full, precision="ftz_true"), full)
                T.simd.vsts(o_ub[1, 0], T.simd.vexp(x, full, precision="ftz_true"), full)
                T.simd.vsts(o_ub[2, 0], T.simd.vexp(x, full, precision="ftz_false"), full)
            T.copy(o_ub, O)

    return kernel


def _build_int():
    @T.prim_func
    def kernel(Ip: T.Tensor((BLK,), "int32"),
               Jp: T.Tensor((BLK,), "int32"),
               Up: T.Tensor((BLK,), "uint32"),
               Vp: T.Tensor((BLK,), "uint32"),
               OI: T.Tensor((3, BLK), "int32"),
               OU: T.Tensor((3, BLK), "uint32")):
        with T.Kernel(1):
            i_ub = T.alloc_shared((BLK,), "int32")
            j_ub = T.alloc_shared((BLK,), "int32")
            u_ub = T.alloc_shared((BLK,), "uint32")
            v_ub = T.alloc_shared((BLK,), "uint32")
            oi_ub = T.alloc_shared((3, BLK), "int32")
            ou_ub = T.alloc_shared((3, BLK), "uint32")
            T.copy(Ip, i_ub)
            T.copy(Jp, j_ub)
            T.copy(Up, u_ub)
            T.copy(Vp, v_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                i = T.simd.vld(i_ub[0])
                j = T.simd.vld(j_ub[0])
                u = T.simd.vld(u_ub[0])
                v = T.simd.vld(v_ub[0])
                T.simd.vsts(oi_ub[0, 0], T.simd.vmax(i, j, full), full)
                T.simd.vsts(oi_ub[1, 0], T.simd.vmin(i, j, full), full)
                T.simd.vsts(oi_ub[2, 0], T.simd.vmaxs(i, T.cast(-1, "int32"), full), full)
                T.simd.vsts(ou_ub[0, 0], T.simd.vmax(u, v, full), full)
                T.simd.vsts(ou_ub[1, 0], T.simd.vmin(u, v, full), full)
                T.simd.vsts(ou_ub[2, 0], T.simd.vmaxs(u, T.cast(0x80000000, "uint32"), full), full)
            T.copy(oi_ub, OI)
            T.copy(ou_ub, OU)

    return kernel


def _build_vdup():
    @T.prim_func
    def kernel(OI: T.Tensor((BLK,), "int32"),
               OP: T.Tensor((BLK,), "int32"),
               OB: T.Tensor((F16_BLK,), "bfloat16"),
               OQ: T.Tensor((F16_BLK,), "bfloat16")):
        with T.Kernel(1):
            oi_ub = T.alloc_shared((BLK,), "int32")
            op_ub = T.alloc_shared((BLK,), "int32")
            ob_ub = T.alloc_shared((F16_BLK,), "bfloat16")
            oq_ub = T.alloc_shared((F16_BLK,), "bfloat16")
            with T.SimdVF():
                full32 = T.simd.pset(32)
                part32 = T.simd.pset(32, "PAT_VL32")
                full16 = T.simd.pset(16)
                part16 = T.simd.pset(16, "PAT_VL32")
                T.simd.vsts(oi_ub[0], T.simd.vdup(T.cast(7, "int32"), "int32", full32), full32)
                T.simd.vsts(op_ub[0], T.simd.vdup(T.cast(7, "int32"), "int32", part32), full32)
                T.simd.vsts(ob_ub[0], T.simd.vdup(T.cast(1.5, "bfloat16"), "bfloat16", full16),
                            full16, dist="NORM_B16")
                T.simd.vsts(oq_ub[0], T.simd.vdup(T.cast(1.5, "bfloat16"), "bfloat16", part16),
                            full16, dist="NORM_B16")
            T.copy(oi_ub, OI)
            T.copy(op_ub, OP)
            T.copy(ob_ub, OB)
            T.copy(oq_ub, OQ)

    return kernel


def _build_mask():
    @T.prim_func
    def kernel(A: T.Tensor((BLK,), "float32"),
               B: T.Tensor((BLK,), "float32"),
               M: T.Tensor((BLK,), "float32"),
               O: T.Tensor((18, BLK), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((BLK,), "float32")
            b_ub = T.alloc_shared((BLK,), "float32")
            m_ub = T.alloc_shared((BLK,), "float32")
            o_ub = T.alloc_shared((18, BLK), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.copy(M, m_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                a = T.simd.vld(a_ub[0])
                b = T.simd.vld(b_ub[0])
                mv = T.simd.vld(m_ub[0])
                zero = T.simd.vdup(T.cast(0.0, "float32"), "float32", full)
                ma = T.simd.vcmp(mv, zero, full, op="ne")
                mf = T.simd.vcmp(mv, mv, full, op="ne")
                half = T.cast(1.5, "float32")
                T.simd.vsts(o_ub[0, 0], T.simd.vdiv(a, b, ma, precision="ftz_true"), full)
                T.simd.vsts(o_ub[1, 0], T.simd.vdup(half, "float32", ma), full)
                T.simd.vsts(o_ub[2, 0], T.simd.vexp(a, ma, precision="ftz_true"), full)
                T.simd.vsts(o_ub[3, 0], T.simd.vmax(a, b, ma), full)
                T.simd.vsts(o_ub[4, 0], T.simd.vmaxs(a, T.cast(0.0, "float32"), ma), full)
                T.simd.vsts(o_ub[5, 0], T.simd.vmin(a, b, ma), full)
                T.simd.vsts(o_ub[6, 0], T.simd.vdiv(a, b, mf, precision="ftz_true"), full)
                T.simd.vsts(o_ub[7, 0], T.simd.vdup(half, "float32", mf), full)
                T.simd.vsts(o_ub[8, 0], T.simd.vexp(a, mf, precision="ftz_true"), full)
                T.simd.vsts(o_ub[9, 0], T.simd.vmax(a, b, mf), full)
                T.simd.vsts(o_ub[10, 0], T.simd.vmaxs(a, T.cast(0.0, "float32"), mf), full)
                T.simd.vsts(o_ub[11, 0], T.simd.vmin(a, b, mf), full)
                T.simd.vsts(o_ub[12, 0], T.simd.vdup(half, "float32", ma), full)
                T.simd.vsts(o_ub[13, 0], T.simd.vdupv(a, ma, pos="POS_LOWEST"), full)
                T.simd.vsts(o_ub[14, 0], T.simd.vdupv(a, ma, pos="POS_HIGHEST"), full)
                T.simd.vsts(o_ub[15, 0], T.simd.vdup(half, "float32", mf), full)
                T.simd.vsts(o_ub[16, 0], T.simd.vdupv(a, mf, pos="POS_LOWEST"), full)
                T.simd.vsts(o_ub[17, 0], T.simd.vdupv(a, mf, pos="POS_HIGHEST"), full)
            T.copy(o_ub, O)

    return kernel


def _build_mode():
    """Explicit MODE_ZEROING only: never mirror TLA MERGING with dst preservation."""
    @T.prim_func
    def kernel(A: T.Tensor((BLK,), "float32"),
               B: T.Tensor((BLK,), "float32"),
               M: T.Tensor((BLK,), "float32"),
               O: T.Tensor((6, BLK), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((BLK,), "float32")
            b_ub = T.alloc_shared((BLK,), "float32")
            m_ub = T.alloc_shared((BLK,), "float32")
            o_ub = T.alloc_shared((6, BLK), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.copy(M, m_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                a = T.simd.vld(a_ub[0])
                b = T.simd.vld(b_ub[0])
                mv = T.simd.vld(m_ub[0])
                zero = T.simd.vdup(T.cast(0.0, "float32"), "float32", full)
                ma = T.simd.vcmp(mv, zero, full, op="ne")
                half = T.cast(1.5, "float32")
                T.simd.vsts(o_ub[0, 0], T.simd.vdiv(a, b, ma, mode="MODE_ZEROING",
                                                     precision="ftz_true"), full)
                T.simd.vsts(o_ub[1, 0], T.simd.vdup(half, "float32", ma,
                                                    mode="MODE_ZEROING"), full)
                T.simd.vsts(o_ub[2, 0], T.simd.vexp(a, ma, mode="MODE_ZEROING",
                                                    precision="ftz_true"), full)
                T.simd.vsts(o_ub[3, 0], T.simd.vmax(a, b, ma, mode="MODE_ZEROING"), full)
                T.simd.vsts(o_ub[4, 0], T.simd.vmaxs(a, T.cast(0.0, "float32"), ma,
                                                     mode="MODE_ZEROING"), full)
                T.simd.vsts(o_ub[5, 0], T.simd.vmin(a, b, ma, mode="MODE_ZEROING"), full)
            T.copy(o_ub, O)

    return kernel


def _build_u16_div():
    @T.prim_func
    def kernel(A: T.Tensor((U16_DIV_BLK,), "uint16"),
               B: T.Tensor((U16_DIV_BLK,), "uint16"),
               O: T.Tensor((3, U16_DIV_BLK), "uint16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((U16_DIV_BLK,), "uint16")
            b_ub = T.alloc_shared((U16_DIV_BLK,), "uint16")
            o_ub = T.alloc_shared((3, U16_DIV_BLK), "uint16")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                full = T.simd.pset(16)
                half = T.simd.pset(16, "PAT_VL64")
                a = T.simd.vld(a_ub[0])
                b = T.simd.vld(b_ub[0])
                none = T.simd.vcmp(a, a, full, op="ne")
                T.simd.vsts(o_ub[0, 0], T.simd.vdiv(a, b, full, mode="MODE_ZEROING"),
                            full, dist="NORM_B16")
                T.simd.vsts(o_ub[1, 0], T.simd.vdiv(a, b, half, mode="MODE_ZEROING"),
                            full, dist="NORM_B16")
                T.simd.vsts(o_ub[2, 0], T.simd.vdiv(a, b, none, mode="MODE_ZEROING"),
                            full, dist="NORM_B16")
            T.copy(o_ub, O)

    return kernel


def _build_u32_div():
    @T.prim_func
    def kernel(A: T.Tensor((BLK,), "uint32"),
               B: T.Tensor((BLK,), "uint32"),
               O: T.Tensor((3, BLK), "uint32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((BLK,), "uint32")
            b_ub = T.alloc_shared((BLK,), "uint32")
            o_ub = T.alloc_shared((3, BLK), "uint32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                half = T.simd.pset(32, "PAT_VL32")
                a = T.simd.vld(a_ub[0])
                b = T.simd.vld(b_ub[0])
                none = T.simd.vcmp(a, a, full, op="ne")
                T.simd.vsts(o_ub[0, 0], T.simd.vdiv(a, b, full, mode="MODE_ZEROING"), full)
                T.simd.vsts(o_ub[1, 0], T.simd.vdiv(a, b, half, mode="MODE_ZEROING"), full)
                T.simd.vsts(o_ub[2, 0], T.simd.vdiv(a, b, none, mode="MODE_ZEROING"), full)
            T.copy(o_ub, O)

    return kernel


def _build_dupv():
    @T.prim_func
    def kernel(X: T.Tensor((BLK,), "float32"),
               O: T.Tensor((9, BLK), "float32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((BLK,), "float32")
            o_ub = T.alloc_shared((9, BLK), "float32")
            T.copy(X, x_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                half = T.simd.pset(32, "PAT_VL32")
                x = T.simd.vld(x_ub[0])
                none = T.simd.vcmp(x, x, full, op="ne")
                one_half = T.cast(1.5, "float32")
                T.simd.vsts(o_ub[0, 0], T.simd.vdup(one_half, "float32", full), full)
                T.simd.vsts(o_ub[1, 0], T.simd.vdupv(x, full, pos="POS_LOWEST"), full)
                T.simd.vsts(o_ub[2, 0], T.simd.vdupv(x, full, pos="POS_HIGHEST"), full)
                T.simd.vsts(o_ub[3, 0], T.simd.vdup(one_half, "float32", half), full)
                T.simd.vsts(o_ub[4, 0], T.simd.vdupv(x, half, pos="POS_LOWEST"), full)
                T.simd.vsts(o_ub[5, 0], T.simd.vdupv(x, half, pos="POS_HIGHEST"), full)
                T.simd.vsts(o_ub[6, 0], T.simd.vdup(one_half, "float32", none), full)
                T.simd.vsts(o_ub[7, 0], T.simd.vdupv(x, none, pos="POS_LOWEST"), full)
                T.simd.vsts(o_ub[8, 0], T.simd.vdupv(x, none, pos="POS_HIGHEST"), full)
            T.copy(o_ub, O)

    return kernel


def _build_f16():
    @T.prim_func
    def kernel(A: T.Tensor((F16_BLK,), "float16"),
               B: T.Tensor((F16_BLK,), "float16"),
               O: T.Tensor((6, F16_BLK), "float16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((F16_BLK,), "float16")
            b_ub = T.alloc_shared((F16_BLK,), "float16")
            o_ub = T.alloc_shared((6, F16_BLK), "float16")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                full = T.simd.pset(16)
                part = T.simd.pset(16, "PAT_VL64")
                a = T.simd.vld(a_ub[0])
                b = T.simd.vld(b_ub[0])
                half = T.cast(1.5, "float16")
                T.simd.vsts(o_ub[0, 0], T.simd.vdiv(a, b, full), full, dist="NORM_B16")
                T.simd.vsts(o_ub[1, 0], T.simd.vexp(a, full), full, dist="NORM_B16")
                T.simd.vsts(o_ub[2, 0], T.simd.vmax(a, b, full), full, dist="NORM_B16")
                T.simd.vsts(o_ub[3, 0], T.simd.vmin(a, b, full), full, dist="NORM_B16")
                T.simd.vsts(o_ub[4, 0], T.simd.vdup(half, "float16", full), full, dist="NORM_B16")
                T.simd.vsts(o_ub[5, 0], T.simd.vdup(half, "float16", part), full, dist="NORM_B16")
            T.copy(o_ub, O)

    return kernel


# ---------------------------------------------------------------------------
# cached compiled kernels + launchers
# ---------------------------------------------------------------------------

_BUILDERS = {
    "smoke": _build_smoke,
    "f32_combo": _build_combo,
    "f32_special": _build_special,
    "int_bounds": _build_int,
    "vdup_int_bf16": _build_vdup,
    "f32_mask_pattern": _build_mask,
    "f32_mode_mask": _build_mode,
    "f16_ops": _build_f16,
    "u16_div_zero": _build_u16_div,
    "u32_div_zero": _build_u32_div,
    "f32_dupv_mask": _build_dupv,
}

_CACHE = {}


def _kernel(case):
    if case not in _CACHE:
        _CACHE[case] = tilelang.compile(_BUILDERS[case](), target="ascend")
    return _CACHE[case]


def _dev(shape, dtype, fill):
    return torch.full(shape, fill, dtype=dtype).npu()


def _launch_smoke(arrays):
    k = _kernel("smoke")
    A, B = _to_npu(arrays["a"]), _to_npu(arrays["b"])
    O = _dev((BLK,), torch.float32, -12345.0)
    k(A, B, O)
    torch.npu.synchronize()
    return {"out": _from_npu(O)}


def _launch_combo(arrays):
    k = _kernel("f32_combo")
    A, B = _to_npu(arrays["a"]), _to_npu(arrays["b"])
    O = _dev((12, BLK), torch.float32, -12345.0)
    k(A, B, O)
    torch.npu.synchronize()
    names = ("OD", "OU", "OE", "OX", "OS", "ON", "PD", "PU", "PE", "PX", "PS", "PN")
    got = _from_npu(O)
    return {name: got[i] for i, name in enumerate(names)}


def _launch_special(arrays):
    k = _kernel("f32_special")
    A, B, X = _to_npu(arrays["a"]), _to_npu(arrays["b"]), _to_npu(arrays["x"])
    O = _dev((3, BLK), torch.float32, -12345.0)
    k(A, B, X, O)
    torch.npu.synchronize()
    got = _from_npu(O)
    return {"div": got[0], "exp_ftz_true": got[1], "exp_ftz_false": got[2]}


def _launch_int(arrays):
    k = _kernel("int_bounds")
    I, J = _to_npu(arrays["i"]), _to_npu(arrays["j"])
    U, V = _to_npu(arrays["u"]), _to_npu(arrays["v"])
    OI = _dev((3, BLK), torch.int32, -7)
    OU = torch.zeros((3, BLK), dtype=torch.uint32).npu()
    k(I, J, U, V, OI, OU)
    torch.npu.synchronize()
    oi, ou = _from_npu(OI), _from_npu(OU)
    return {
        "i_max": oi[0], "i_min": oi[1], "i_maxs": oi[2],
        "u_max": ou[0], "u_min": ou[1], "u_maxs": ou[2],
    }


def _launch_vdup(_arrays):
    k = _kernel("vdup_int_bf16")
    OI = _dev((BLK,), torch.int32, -7)
    OP = _dev((BLK,), torch.int32, -7)
    OB = _dev((F16_BLK,), torch.bfloat16, -7.0)
    OQ = _dev((F16_BLK,), torch.bfloat16, -7.0)
    k(OI, OP, OB, OQ)
    torch.npu.synchronize()
    return {
        "OI": _from_npu(OI), "OP": _from_npu(OP),
        "OB": _from_npu(OB, bf16_bits=True), "OQ": _from_npu(OQ, bf16_bits=True),
    }


def _launch_mask(arrays):
    k = _kernel("f32_mask_pattern")
    A, B, M = _to_npu(arrays["a"]), _to_npu(arrays["b"]), _to_npu(arrays["m_arb"])
    O = _dev((18, BLK), torch.float32, -12345.0)
    k(A, B, M, O)
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
    k = _kernel("f32_mode_mask")
    A, B, M = _to_npu(arrays["a"]), _to_npu(arrays["b"]), _to_npu(arrays["m_arb"])
    O = _dev((6, BLK), torch.float32, -12345.0)
    k(A, B, M, O)
    torch.npu.synchronize()
    got = _from_npu(O)
    names = ("D", "U", "E", "X", "S", "N")
    return {f"Z{name}": got[i] for i, name in enumerate(names)}


def _launch_u16_div(arrays):
    k = _kernel("u16_div_zero")
    A, B = _to_npu_u16(arrays["a"]), _to_npu_u16(arrays["b"])
    O = torch.full((3, U16_DIV_BLK), 0x5A5A, dtype=torch.uint16).npu()
    k(A, B, O)
    torch.npu.synchronize()
    got = _from_npu_u16(O)
    return {"Zfull": got[0], "Zhalf": got[1], "Znone": got[2]}


def _launch_u32_div(arrays):
    k = _kernel("u32_div_zero")
    A, B = _to_npu(arrays["a"]), _to_npu(arrays["b"])
    O = torch.full((3, BLK), 0x5A5A5A5A, dtype=torch.uint32).npu()
    k(A, B, O)
    torch.npu.synchronize()
    got = _from_npu(O)
    return {"Zfull": got[0], "Zhalf": got[1], "Znone": got[2]}


def _launch_dupv(arrays):
    k = _kernel("f32_dupv_mask")
    X = _to_npu(arrays["x"])
    O = _dev((9, BLK), torch.float32, -12345.0)
    k(X, O)
    torch.npu.synchronize()
    got = _from_npu(O)
    names = ("scalar_full", "low_full", "high_full",
             "scalar_half", "low_half", "high_half",
             "scalar_none", "low_none", "high_none")
    return {name: got[i] for i, name in enumerate(names)}


def _launch_f16(arrays):
    k = _kernel("f16_ops")
    A, B = _to_npu(arrays["a"]), _to_npu(arrays["b"])
    O = _dev((6, F16_BLK), torch.float16, -7.0)
    k(A, B, O)
    torch.npu.synchronize()
    got = _from_npu(O)
    names = ("div", "exp", "max", "min", "dup_full", "dup_part")
    return {name: got[i] for i, name in enumerate(names)}


LAUNCHERS = {
    "smoke": _launch_smoke,
    "f32_combo": _launch_combo,
    "f32_special": _launch_special,
    "int_bounds": _launch_int,
    "vdup_int_bf16": _launch_vdup,
    "f32_mask_pattern": _launch_mask,
    "f32_mode_mask": _launch_mode,
    "f16_ops": _launch_f16,
    "u16_div_zero": _launch_u16_div,
    "u32_div_zero": _launch_u32_div,
    "f32_dupv_mask": _launch_dupv,
}
