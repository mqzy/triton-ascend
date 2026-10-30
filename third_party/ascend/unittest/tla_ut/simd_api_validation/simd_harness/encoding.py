"""Lossless, JSON-safe array encoding and small file helpers.

Floats are never serialised as Python floats: a float JSON round-trip
canonicalises NaN payloads and can turn -0.0 into +0.0.  Every array is stored
as ``dtype + shape + base64(raw bytes)``.  ``npz`` is used for the larger
side files (``numpy.savez`` writes a zip of raw arrays, no pickle).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def tensor_to_json(array) -> dict:
    arr = np.ascontiguousarray(array)
    if arr.dtype.hasobject:
        raise TypeError("object arrays are not allowed in harness results")
    return {
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
        "bits_b64": base64.b64encode(arr.tobytes(order="C")).decode("ascii"),
    }


def tensor_from_json(obj: dict) -> np.ndarray:
    dtype = np.dtype(obj["dtype"])
    raw = base64.b64decode(obj["bits_b64"])
    arr = np.frombuffer(raw, dtype=dtype)
    expected = int(np.prod(obj["shape"], dtype=np.int64)) if obj["shape"] else 1
    if arr.size != expected:
        raise ValueError(f"tensor payload has {arr.size} elements, shape {obj['shape']} needs {expected}")
    return arr.reshape(tuple(obj["shape"])).copy()


def array_digest(array) -> str:
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(arr.dtype.str.encode("ascii"))
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_npz(path, arrays: dict) -> None:
    """Save arrays losslessly.  Keys are sanitised because npz keys are paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = {}
    for key, value in arrays.items():
        safe = str(key).replace("/", "__")
        if safe in sanitized:
            raise ValueError(f"duplicate npz key after sanitising: {key}")
        sanitized[safe] = np.ascontiguousarray(value)
    np.savez(path, **sanitized)


def load_npz(path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def json_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, allow_nan=False).encode("utf-8")
