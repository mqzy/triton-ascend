"""Version and capability probes with clear, actionable failures.

No path is hard-coded.  The BishengIR compiler is looked up through
``PATH`` / ``TRITON_NPU_COMPILER_PATH`` / the installed Triton backend package,
exactly like the Triton backend itself does.  A device run refuses to start
when the compiler does not advertise ``--enable-tla-mode``; it never treats a
bare process exit code as evidence that the case ran.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

TLA_APIS = ("vdiv", "vdup", "vdups", "vdupv", "vexp", "vmax", "vmaxs", "vmin")


def _run_version(path: str) -> dict:
    out = {"path": path, "version": None, "supports_enable_tla_mode": False,
           "error": None}
    try:
        ver = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=20)
        text = (ver.stdout or "") + (ver.stderr or "")
        out["version"] = text.strip().splitlines()[0] if text.strip() else None
        help_ = subprocess.run([path, "--help"], capture_output=True, text=True, timeout=20)
        htext = (help_.stdout or "") + (help_.stderr or "")
        out["supports_enable_tla_mode"] = "--enable-tla-mode" in htext
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def find_bishengir_compile() -> list:
    """Return candidate compiler paths, in the same order as the backend."""
    candidates = []
    which = shutil.which("bishengir-compile")
    if which:
        candidates.append(which)
    root = os.environ.get("TRITON_NPU_COMPILER_PATH")
    if root:
        for name in ("bishengir-compile", "bin/bishengir-compile", "compiler/bishengir-compile"):
            p = Path(root) / name
            if p.exists():
                candidates.append(str(p))
    try:
        import triton.backends.ascend as ascend_backend  # type: ignore

        bundled = Path(ascend_backend.__file__).resolve().parent / "bishengir" / "bin" / "bishengir-compile"
        if bundled.exists():
            candidates.append(str(bundled))
    except Exception:
        pass
    # stably de-duplicate
    seen, out = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def probe_bishengir(require_tla: bool = True) -> dict:
    candidates = find_bishengir_compile()
    reports = [_run_version(p) for p in candidates]
    chosen = None
    for report in reports:
        if not require_tla or report["supports_enable_tla_mode"]:
            chosen = report
            break
    if chosen is None:
        if reports:
            reason = ("found bishengir-compile but none advertises --enable-tla-mode; "
                      "build AscendNPU-IR from the TLA-capable source tree or put that "
                      "tool directory first in PATH/TRITON_NPU_COMPILER_PATH")
        else:
            reason = ("bishengir-compile not found; put a TLA-capable build in PATH "
                      "or set TRITON_NPU_COMPILER_PATH (do not point at an old CANN compiler)")
        return {"ready": False, "reason": reason, "candidates": reports, "chosen": None}
    return {"ready": True, "reason": None, "candidates": reports, "chosen": chosen}


def _base() -> dict:
    return {
        "python": os.sys.version.split()[0],
        "executable": os.sys.executable,
        "numpy": np.__version__,
        "soc_version": os.environ.get("SOC_VERSION"),
        "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
    }


def _torch_versions(info: dict) -> None:
    try:
        import torch

        info["torch"] = getattr(torch, "__version__", None)
    except Exception as exc:  # noqa: BLE001
        info["torch"] = None
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import torch_npu

        info["torch_npu"] = getattr(torch_npu, "__version__", None)
    except Exception as exc:  # noqa: BLE001
        info["torch_npu"] = None
        info["torch_npu_error"] = f"{type(exc).__name__}: {exc}"


def describe_tla(execute: bool = False) -> dict:
    info = _base()
    info["side"] = "tla"
    info["target"] = "npu"
    try:
        import triton
        import triton.language.extra.tla as tla
    except Exception as exc:  # noqa: BLE001
        info.update({"ready": False,
                     "reason": f"cannot import Triton TLA frontend: {type(exc).__name__}: {exc}. "
                               "Use a build tree or wheel built from a source tree that contains "
                               "these APIs.",
                     "triton": None})
        return info
    info["triton"] = getattr(triton, "__version__", None)
    if execute:
        _torch_versions(info)
    missing = [name for name in TLA_APIS if not hasattr(getattr(tla, "simd", object()), name)]
    info["missing_apis"] = missing
    compiler = probe_bishengir(require_tla=execute)
    info["compiler"] = compiler
    if missing:
        info["ready"] = False
        info["reason"] = f"TLA frontend is missing APIs: {missing} (stale wheel or stale build tree)"
        return info
    if execute and not compiler["ready"]:
        info["ready"] = False
        info["reason"] = compiler["reason"]
        return info
    info["ready"] = True
    info["reason"] = None
    return info


def describe_tilelang(execute: bool = False) -> dict:
    info = _base()
    info["side"] = "tilelang"
    info["target"] = "ascend"
    try:
        import tilelang
        from tilelang.ascend import language as T  # noqa: F401
        import tilelang.ascend.language.simd as simd
    except Exception as exc:  # noqa: BLE001
        info.update({"ready": False,
                     "reason": f"cannot import TileLang Ascend frontend: {type(exc).__name__}: {exc}. "
                               "Use the Python environment where TileLang is installed.",
                     "tilelang": None})
        return info
    info["tilelang"] = getattr(tilelang, "__version__", None)
    if execute:
        _torch_versions(info)
    needed = ("pset", "vld", "vsts", "vdiv", "vexp", "vmax", "vmin", "vmaxs", "vdup")
    missing = [name for name in needed if not hasattr(simd, name)]
    info["missing_apis"] = missing
    if missing:
        info["ready"] = False
        info["reason"] = f"TileLang SIMD frontend is missing functions: {missing}"
    else:
        info["ready"] = True
        info["reason"] = None
    return info


def describe(side: str, execute: bool = False) -> dict:
    if side == "tla":
        return describe_tla(execute=execute)
    if side == "tilelang":
        return describe_tilelang(execute=execute)
    raise ValueError(f"unknown side {side!r}")
