"""Deterministic CPU-input helpers.

Rules for every case:

* no RNG and no environment-dependent values;
* f32/f16 values are constructed from exactly representable rationals unless a
  test explicitly needs a special bit pattern;
* NaN/Inf/negative-zero lanes are constructed from raw uint bits, so the JSON
  and npz payloads keep the exact NaN payload;
* bfloat16 is represented as ``uint16`` raw bits because numpy has no bf16.
"""

from __future__ import annotations

import struct

import numpy as np


def f32(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def f16(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float16)


def f32_from_bits(bits) -> np.ndarray:
    return np.asarray(bits, dtype=np.uint32).view(np.float32).copy()


def f16_from_bits(bits) -> np.ndarray:
    return np.asarray(bits, dtype=np.uint16).view(np.float16).copy()


def _round_f32_to_bf16_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    # round-to-nearest-even on the discarded 16 bits
    lsb = (bits >> 16) & 1
    rounded = (bits + 0x7FFF + lsb) >> 16
    return int(rounded & 0xFFFF)


def bf16_bits(values) -> np.ndarray:
    if np.isscalar(values):
        values = [values]
    return np.asarray([_round_f32_to_bf16_bits(v) for v in values], dtype=np.uint16)


def pad(values, lanes: int, fill):
    out = list(values) + [fill] * lanes
    return out[:lanes]


def exact_f32_grid(lanes: int, mul_a: int = 13, mul_b: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """The historical f32_combo input formula (exact in f32, b never zero)."""
    a = [((i * mul_a) % 29 - 14) / 4.0 for i in range(lanes)]
    b = [((i * mul_b) % 31 + 1) / 5.0 for i in range(lanes)]
    return f32(a), f32(b)


# Raw f32 bit patterns used by the NaN/Inf case.  Quiet NaNs with distinct
# payloads prove that the harness never goes through a Python float.
F32_POS_QNAN_1 = 0x7FC00001
F32_NEG_QNAN_2 = 0xFFC00002
F32_POS_QNAN_3 = 0x7FC00003
F32_NEG_QNAN_4 = 0xFFC00004
F32_POS_INF = 0x7F800000
F32_NEG_INF = 0xFF800000
F32_POS_ZERO = 0x00000000
F32_NEG_ZERO = 0x80000000
