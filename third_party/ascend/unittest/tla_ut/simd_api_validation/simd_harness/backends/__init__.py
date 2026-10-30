"""Backend dispatch.  Import a backend only when it is actually executed."""

from __future__ import annotations


class BackendError(RuntimeError):
    """Capability/config failure: the case was not executed."""


class ExecutionError(RuntimeError):
    """The backend started but the device launch failed."""


def execute(side: str, case: str, arrays: dict) -> dict:
    if side == "tla":
        from . import tla_backend

        return tla_backend.execute(case, arrays)
    if side == "tilelang":
        from . import tilelang_backend

        return tilelang_backend.execute(case, arrays)
    raise BackendError(f"unknown side {side!r}")


def capability(side: str, execute: bool) -> dict:
    from ..capabilities import describe

    return describe(side, execute=execute)
