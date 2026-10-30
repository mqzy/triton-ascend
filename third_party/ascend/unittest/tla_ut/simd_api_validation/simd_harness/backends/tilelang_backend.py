"""TileLang execution adapter.

TileLang lives in its own Python environment (selected by the caller).  This
module does not install, pin or copy wheels; a missing import is a clear
capability failure.
"""

from __future__ import annotations

from .. import capabilities
from . import BackendError, ExecutionError

import traceback


def execute(case: str, arrays: dict) -> dict:
    info = capabilities.describe_tilelang(execute=True)
    if not info.get("ready"):
        raise BackendError(info.get("reason") or "TileLang capability check failed")
    try:
        from .. import kernels_tilelang
    except Exception as exc:  # noqa: BLE001
        raise BackendError(f"cannot import TileLang kernels: {type(exc).__name__}: {exc}") from exc
    launcher = kernels_tilelang.LAUNCHERS.get(case)
    if launcher is None:
        raise BackendError(f"TileLang has no launcher for case {case!r}")
    try:
        return launcher(arrays)
    except Exception as exc:  # noqa: BLE001
        raise ExecutionError(
            f"TileLang launch failed for {case}: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}") from exc
