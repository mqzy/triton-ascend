"""Keep this directory's harness out of tla_ut's "collect every *.py" rule.

``tla_ut/conftest.py`` collects *every* ``*.py`` as a test module because the
op cases are named by op, not ``test_*``.  This directory instead holds the
device-validation harness (``simd_harness``) and the standalone comparator:
``simd_harness`` imports ``torch``/``triton`` (and, for the kernels, the
compiled TLA bindings) at module level, so importing it during collection
would fail on a host without a TLA-capable build/device.

Only ``tests/test_harness.py`` is a host test.  ``collect_ignore_glob`` is
consulted by pytest's ``pytest_ignore_collect`` hook before any collector
runs, so these files are neither imported nor reported as collection errors,
and every other directory under ``tla_ut`` keeps its existing discovery.

``pytest_collect_file`` is *not* a first-result hook: for a ``test_*.py`` file
both the ``tla_ut`` hook (which collects every ``*.py``) and pytest's own
python plugin return a module, so ``tests/test_harness.py`` would be collected
twice.  The duplicate items have the same node id, so drop the repeats here;
this leaves all other ``tla_ut`` files untouched (they are collected once).
"""

collect_ignore_glob = [
    "compare_*.py",
    "simd_harness/*",
]


def pytest_collection_modifyitems(session, config, items):
    """Run each node id once (parent hook + python plugin double-collect)."""
    seen = set()
    unique = []
    for item in items:
        if item.nodeid in seen:
            continue
        seen.add(item.nodeid)
        unique.append(item)
    items[:] = unique
