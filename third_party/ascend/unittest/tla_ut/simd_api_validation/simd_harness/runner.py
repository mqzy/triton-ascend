"""``--list / --check / --case / --output`` entry point.

The caller supplies the Python environment (frontend package and compiler); the
runner never selects an interpreter, sets environment variables or writes into
a build tree.

Useful modes:

* ``--list`` -- describe every case and its implementation status,
* ``--check`` -- probe what the current environment can execute,
* ``--case ... --output ...`` -- evaluate inputs/reference on the host,
* ``--case ... --output ... --execute`` -- launch the device backend,
* ``--executor reference|zero|teardown`` -- test-only executors that never
  import a device backend.

Exit codes
----------
0  PASS / SELF_TEST_PASS / HOST_REFERENCE_ONLY
1  NUMERICAL_FAIL
2  CONFIG_ERROR (bad case/side/args, missing compiler/package)
3  EXEC_ERROR or TEARDOWN_CRASH
4  SKIP (only when explicitly requested)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import __version__, backends, capabilities
from .cases import get_case, list_payload
from .encoding import (array_digest, read_json, save_npz, tensor_to_json,
                       write_json)
from .reference import compare_output
from .spec import TIER_SUGGESTED

STATUS_PASS = "PASS"
STATUS_SELF_TEST_PASS = "SELF_TEST_PASS"
STATUS_HOST_REFERENCE_ONLY = "HOST_REFERENCE_ONLY"
STATUS_NUMERICAL_FAIL = "NUMERICAL_FAIL"
STATUS_CONFIG_ERROR = "CONFIG_ERROR"
STATUS_EXEC_ERROR = "EXEC_ERROR"
STATUS_TEARDOWN_CRASH = "TEARDOWN_CRASH"
STATUS_SKIP = "SKIP"

EXIT_BY_STATUS = {
    STATUS_PASS: 0,
    STATUS_SELF_TEST_PASS: 0,
    STATUS_HOST_REFERENCE_ONLY: 0,
    STATUS_NUMERICAL_FAIL: 1,
    STATUS_CONFIG_ERROR: 2,
    STATUS_EXEC_ERROR: 3,
    STATUS_TEARDOWN_CRASH: 3,
    STATUS_SKIP: 4,
}


class CaseConfigError(RuntimeError):
    pass


def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _case_dir(output_root: Path, side: str, case: str) -> Path:
    return Path(output_root) / side / case


def _empty_like_reference(ref: dict) -> dict:
    return {name: np.zeros_like(value) for name, value in ref.items()}


def run_case(case_name: str, side: str, output_root, execute=False, executor="real",
             allow_skip=False) -> tuple[dict, int]:
    started = time.time()
    output_root = Path(output_root)
    try:
        spec = get_case(case_name)
    except KeyError as exc:
        raise CaseConfigError(str(exc)) from exc
    if not spec.implemented:
        raise CaseConfigError(
            f"case {case_name!r} is tier={TIER_SUGGESTED}; it is documented but not executable. "
            "Use --list to see every case and its implementation status.")
    if side not in ("tla", "tilelang"):
        raise CaseConfigError(f"invalid side {side!r}")
    if not spec.supports(side):
        raise CaseConfigError(f"case {case_name!r} does not implement side {side!r}")

    case_dir = _case_dir(output_root, side, case_name)
    case_dir.mkdir(parents=True, exist_ok=True)

    arrays = spec.inputs_fn() if spec.inputs_fn else {}
    ref_result = spec.reference_fn(arrays)
    refs = ref_result.outputs
    npz = {}
    for name, value in arrays.items():
        npz[f"in__{name}"] = value
    for name, value in refs.items():
        npz[f"ref__{name}"] = value

    result = {
        "schema": 1,
        "harness_version": __version__,
        "case": case_name,
        "title": spec.title,
        "side": side,
        "tier": spec.tier,
        "dtype": spec.dtype,
        "mask": spec.mask,
        "precision": spec.precision,
        "target": None,
        "executed": False,
        "executor": executor,
        "status": STATUS_CONFIG_ERROR,
        "failure_type": "config",
        "numerical_pass": False,
        "teardown_ok": True,
        "teardown_error": None,
        "exit_code": 2,
        "started_at": _now(),
        "duration_sec": None,
        "versions": {},
        "inputs": {},
        "notes": spec.notes,
        "reference_notes": ref_result.notes,
        "outputs": {},
        "error": None,
    }

    for name, value in arrays.items():
        result["inputs"][name] = {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": array_digest(value),
        }

    info = capabilities.describe(side, execute=False)
    result["target"] = info.get("target")
    result["versions"] = {"capability": info}

    outputs = None
    try:
        if executor == "real":
            if not execute:
                # Host-only validation of inputs/reference; never a device PASS.
                outputs = refs
                result["status"] = STATUS_HOST_REFERENCE_ONLY
                result["failure_type"] = None
                result["numerical_pass"] = True
            else:
                result["executed"] = True
                info_exec = capabilities.describe(side, execute=True)
                result["versions"] = {"capability": info_exec}
                if not info_exec.get("ready"):
                    result["error"] = info_exec.get("reason")
                    result["status"] = STATUS_CONFIG_ERROR
                    result["failure_type"] = "config"
                    outputs = None
                else:
                    outputs = backends.execute(side, case_name, arrays)
                    # Tentative PASS; the comparison below can still turn this
                    # into NUMERICAL_FAIL.
                    result["status"] = STATUS_PASS
        elif executor == "reference":
            outputs = refs
            result["status"] = STATUS_SELF_TEST_PASS
            result["failure_type"] = None
            result["executed"] = False
            result["numerical_pass"] = True
        elif executor == "zero":
            outputs = _empty_like_reference(refs)
            result["status"] = STATUS_NUMERICAL_FAIL
            result["failure_type"] = "numerical"
            result["executed"] = False
        elif executor == "teardown":
            outputs = refs
            result["status"] = STATUS_TEARDOWN_CRASH
            result["failure_type"] = "teardown"
            result["executed"] = False
            result["teardown_ok"] = False
            result["teardown_error"] = "simulated teardown crash (test executor)"
        else:  # pragma: no cover - argparse restricts this
            raise CaseConfigError(f"unknown executor {executor!r}")
    except backends.BackendError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["status"] = STATUS_CONFIG_ERROR
        result["failure_type"] = "config"
        result["executed"] = False
    except backends.ExecutionError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["status"] = STATUS_EXEC_ERROR
        result["failure_type"] = "execution"
        result["executed"] = True
        result["teardown_ok"] = False
    except Exception as exc:  # noqa: BLE001 - execution failures must be reported
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["status"] = STATUS_EXEC_ERROR
        result["failure_type"] = "execution"
        result["executed"] = True
        result["teardown_ok"] = False

    numerical_ok = True
    if outputs is not None:
        supported = spec.output_names(side)
        if isinstance(outputs, dict):
            extra = sorted(set(outputs) - set(supported))
            if extra:
                result["unexpected_outputs"] = extra
            for name in supported:
                out_spec = spec.output(name)
                if name not in outputs:
                    numerical_ok = False
                    result["outputs"][name] = {
                        "dtype": out_spec.dtype,
                        "shape": list(refs[name].shape),
                        "sha256": None,
                        "rule": out_spec.__dict__,
                        "result": {
                            "pass": False,
                            "problems": ["output missing from backend"],
                            "first_fail": None,
                            "nan_mismatch": None,
                            "max_abs_error": None,
                            "zeroing": None,
                        },
                        "got": None,
                        "reference": tensor_to_json(refs[name]),
                    }
                    continue
                got = np.asarray(outputs[name])
                active = ref_result.active_masks.get(name)
                cmp = compare_output(got, refs[name], out_spec, active_mask=active)
                numerical_ok = numerical_ok and cmp["pass"]
                result["outputs"][name] = {
                    "dtype": out_spec.dtype,
                    "shape": list(refs[name].shape),
                    "sha256": array_digest(got) if got.dtype == refs[name].dtype else None,
                    "rule": {
                        "mode": out_spec.mode,
                        "rtol": out_spec.rtol,
                        "atol": out_spec.atol,
                        "nan_policy": out_spec.nan_policy,
                        "signed_zero": out_spec.signed_zero,
                        "zero_inactive": out_spec.zero_inactive,
                        "note": out_spec.note,
                    },
                    "result": cmp,
                    "got": tensor_to_json(got),
                    "reference": tensor_to_json(refs[name]),
                }
                npz[f"got__{name}"] = got
        else:
            numerical_ok = False
            result["error"] = f"backend returned {type(outputs).__name__}, expected dict"

    result["numerical_pass"] = bool(numerical_ok)
    if result["status"] not in (STATUS_CONFIG_ERROR, STATUS_EXEC_ERROR, STATUS_TEARDOWN_CRASH):
        if not numerical_ok:
            result["status"] = STATUS_NUMERICAL_FAIL
            result["failure_type"] = "numerical"
        elif executor == "real" and execute:
            result["status"] = STATUS_PASS
            result["failure_type"] = None
        elif executor == "real":
            result["status"] = STATUS_HOST_REFERENCE_ONLY
            result["failure_type"] = None
        elif executor == "reference":
            result["status"] = STATUS_SELF_TEST_PASS
            result["failure_type"] = None

    result["teardown_ok"] = bool(result["status"] != STATUS_TEARDOWN_CRASH and
                                 result["failure_type"] not in ("teardown", "execution"))
    result["exit_code"] = EXIT_BY_STATUS[result["status"]]
    result["duration_sec"] = round(time.time() - started, 6)

    save_npz(case_dir / "arrays.npz", npz)
    save_npz(case_dir / "inputs.npz", {k: v for k, v in npz.items() if k.startswith("in__")})
    write_json(case_dir / "result.json", result)

    print(f"RESULT_JSON {case_dir / 'result.json'}")
    print(f"STATUS {result['status']} case={case_name} side={side} "
          f"exit={result['exit_code']} numerical={result['numerical_pass']}")
    if result.get("error"):
        print(f"ERROR {result['error']}", file=sys.stderr)
    return result, result["exit_code"]


def _print_list(args):
    if args.side:
        if args.side not in ("tla", "tilelang"):
            print(f"invalid side {args.side!r}", file=sys.stderr)
            return 2
    cases = []
    for meta in list_payload():
        if args.side and args.side not in meta["sides"]:
            continue
        cases.append(meta)
    if args.format == "text":
        for meta in cases:
            print(f"{meta['name']:<22} {meta['tier']:<32} "
                  f"sides={','.join(meta['sides'])} dtype={meta['dtype']} mask={meta['mask']}")
    else:
        print(json.dumps({"schema": 1, "harness_version": __version__, "cases": cases},
                         indent=2, sort_keys=True))
    return 0


def _print_check(args):
    sides = [args.side] if args.side else ["tla", "tilelang"]
    payload = {}
    ready = True
    for side in sides:
        try:
            info = capabilities.describe(side, execute=args.execute)
        except Exception as exc:  # noqa: BLE001
            info = {"side": side, "ready": False, "reason": f"{type(exc).__name__}: {exc}"}
        payload[side] = info
        ready = ready and bool(info.get("ready"))
    print(json.dumps({"schema": 1, "execute": bool(args.execute), "ready": ready,
                      "sides": payload}, indent=2, sort_keys=True))
    return 0 if ready else 2


def build_parser():
    parser = argparse.ArgumentParser(
        description="Device-agnostic SIMD API case runner (no profiler inside).")
    parser.add_argument("--list", action="store_true", help="list cases as JSON")
    parser.add_argument("--check", action="store_true", help="probe side capabilities")
    parser.add_argument("--side", choices=("tla", "tilelang"), help="side to run/list/check")
    parser.add_argument("--case", help="case name (see --list)")
    parser.add_argument("--output", help="output root; result at OUTPUT/<side>/<case>/result.json")
    parser.add_argument("--execute", action="store_true",
                        help="actually launch the device backend (otherwise host-reference only)")
    parser.add_argument("--executor", default="real",
                        choices=("real", "reference", "zero", "teardown"),
                        help="test-only executors; real is the device path")
    parser.add_argument("--format", default="json", choices=("json", "text"))
    parser.add_argument("--allow-skip", action="store_true",
                        help="do not turn an unsupported side into a hard failure")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        return _print_list(args)
    if args.check:
        return _print_check(args)
    if not args.case or not args.side or not args.output:
        print("--case, --side and --output are required (or use --list/--check)",
              file=sys.stderr)
        return 2
    try:
        _result, code = run_case(args.case, args.side, args.output, execute=args.execute,
                                 executor=args.executor, allow_skip=args.allow_skip)
    except CaseConfigError as exc:
        print(f"CONFIG_ERROR {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
