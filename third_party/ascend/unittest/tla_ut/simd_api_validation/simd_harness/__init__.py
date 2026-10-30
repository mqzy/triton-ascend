"""Device-agnostic harness for the TLA/TileLang SIMD APIs.

The package is deliberately split so that no profiler and no environment setup
is part of the case definition, the input generation, the reference, or the
comparison:

* ``spec``      -- immutable case metadata (dtype / mask / precision / rules)
* ``inputs``    -- deterministic CPU inputs (integer/bit-pattern based)
* ``reference`` -- CPU reference plus the comparison rules
* ``runner``    -- ``--list / --check / --case / --output`` entry point
* ``kernels_tla`` / ``kernels_tilelang`` -- device kernels only
* ``backends``  -- thin import/launch adapters (no profiler, no env paths)
* ``capabilities`` -- version and compiler-capability probes

The caller owns the Python environment: it must provide the frontend package
and a compiler that supports the APIs, as checked by ``--check``.  The harness
does not select an interpreter, set ``PYTHONPATH``, write a build tree or pick
a cache directory.

All array payloads written to JSON are raw little-endian bytes (base64), never
Python floats.  This keeps NaN payloads, signed zeros and bit-exact integers
intact.  ``npz`` side files are produced with ``numpy.savez`` (no pickle).
"""

__version__ = 1
