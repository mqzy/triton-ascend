"""Case metadata and comparison-rule types.

A case is a pure-Python description: deterministic input builder, CPU
reference, output rules, and the kernel name per side.  Nothing in this file
imports torch, triton, tilelang or a profiler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

TIER_IMPLEMENTED = "implemented"
TIER_SUGGESTED = "suggested_not_implemented"

VALID_TIERS = (TIER_IMPLEMENTED, TIER_SUGGESTED)
VALID_SIDES = ("tla", "tilelang")
VALID_MODES = ("exact", "close")
VALID_NAN_POLICIES = ("forbid", "position", "propagate", "canonical_bits")
VALID_SIGNED_ZERO = ("any", "ieee")


@dataclass(frozen=True)
class OutputSpec:
    """One output buffer and the rule used to compare it with the CPU reference."""

    name: str
    dtype: str
    mode: str = "exact"  # exact = raw-bit equality, close = rtol/atol
    rtol: float = 0.0
    atol: float = 0.0
    nan_policy: str = "forbid"  # forbid | position | propagate | canonical_bits
    signed_zero: str = "any"  # any | ieee (zero results must keep the reference sign)
    zero_inactive: bool = False  # inactive lanes must be *bitwise* +0
    sides: Tuple[str, ...] = VALID_SIDES
    note: str = ""

    def __post_init__(self):
        if self.mode not in VALID_MODES:
            raise ValueError(f"{self.name}: invalid mode {self.mode!r}")
        if self.nan_policy not in VALID_NAN_POLICIES:
            raise ValueError(f"{self.name}: invalid nan_policy {self.nan_policy!r}")
        if self.signed_zero not in VALID_SIGNED_ZERO:
            raise ValueError(f"{self.name}: invalid signed_zero {self.signed_zero!r}")
        for side in self.sides:
            if side not in VALID_SIDES:
                raise ValueError(f"{self.name}: invalid side {side!r}")

    def supports(self, side: str) -> bool:
        return side in self.sides


@dataclass
class RefResult:
    """CPU reference outputs and optional active-lane masks for zeroing checks."""

    outputs: Dict[str, np.ndarray]
    active_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseSpec:
    """One case, described independently of any machine or run.

    ``tier`` is the implementation status of the case itself:

    ``implemented``
        Inputs, CPU reference and kernels exist for the listed ``sides``; the
        case can be run against a host reference or a real backend.
    ``suggested_not_implemented``
        Evaluated idea that has no inputs/reference/kernel yet; ``--list``
        reports it, ``--case`` refuses it with a clear error.
    """

    name: str
    title: str
    description: str
    dtype: str
    mask: str
    precision: str
    tier: str
    sides: Tuple[str, ...]
    kernels: Dict[str, str]
    outputs: Tuple[OutputSpec, ...]
    inputs_fn: Optional[Callable[[], Dict[str, np.ndarray]]] = None
    reference_fn: Optional[Callable[[Dict[str, np.ndarray]], RefResult]] = None
    notes: str = ""

    def __post_init__(self):
        if self.tier not in VALID_TIERS:
            raise ValueError(f"{self.name}: invalid tier {self.tier!r}")
        for side in self.sides:
            if side not in VALID_SIDES:
                raise ValueError(f"{self.name}: invalid side {side!r}")
        if self.implemented:
            if self.inputs_fn is None or self.reference_fn is None:
                raise ValueError(f"{self.name}: implemented case needs inputs/reference")
            names = {o.name for o in self.outputs}
            if len(names) != len(self.outputs):
                raise ValueError(f"{self.name}: duplicate output name")

    @property
    def implemented(self) -> bool:
        return self.tier == TIER_IMPLEMENTED

    def supports(self, side: str) -> bool:
        return side in self.sides

    def output(self, name: str) -> OutputSpec:
        for spec in self.outputs:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def output_names(self, side: str) -> Tuple[str, ...]:
        return tuple(o.name for o in self.outputs if o.supports(side))


def public_case_metadata(spec: CaseSpec) -> dict:
    """JSON-safe description used by ``--list`` and by ``result.json``."""
    return {
        "name": spec.name,
        "title": spec.title,
        "description": spec.description,
        "dtype": spec.dtype,
        "mask": spec.mask,
        "precision": spec.precision,
        "tier": spec.tier,
        "implemented": spec.implemented,
        "sides": list(spec.sides),
        "kernels": dict(spec.kernels),
        "outputs": [
            {
                "name": o.name,
                "dtype": o.dtype,
                "mode": o.mode,
                "rtol": o.rtol,
                "atol": o.atol,
                "nan_policy": o.nan_policy,
                "signed_zero": o.signed_zero,
                "zero_inactive": o.zero_inactive,
                "sides": list(o.sides),
                "note": o.note,
            }
            for o in spec.outputs
        ],
        "notes": spec.notes,
    }
