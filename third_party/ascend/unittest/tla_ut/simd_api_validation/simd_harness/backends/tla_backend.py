"""TLA execution adapter.

The caller supplies the Python environment (build tree or wheel).  This
module only checks that the frontend/compiler are usable and then calls the
case launcher.  It never calls a profiler, never sets PYTHONPATH and never
writes a cache directory.
"""

from __future__ import annotations

from .. import capabilities
from . import BackendError, ExecutionError

import traceback


def execute(case: str, arrays: dict) -> dict:
    info = capabilities.describe_tla(execute=True)
    if not info.get("ready"):
        raise BackendError(info.get("reason") or "TLA capability check failed")
    try:
        from .. import kernels_tla
    except Exception as exc:  # noqa: BLE001
        raise BackendError(f"cannot import TLA kernels: {type(exc).__name__}: {exc}") from exc
    launcher = kernels_tla.LAUNCHERS.get(case)
    if launcher is None:
        raise BackendError(f"TLA has no launcher for case {case!r}")
    try:
        return launcher(arrays)
    except Exception as exc:  # noqa: BLE001
        raise ExecutionError(
            f"TLA launch failed for {case}: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}") from exc
