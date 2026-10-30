#!/usr/bin/env python3
"""Compare two harness result trees without a profiler.

Typical use: compare the TLA result tree with the TileLang result tree, lane by
lane, using the same case rules as the per-side reference comparison.

    python compare_results.py --baseline out/tla --candidate out/tilelang

Baseline/candidate may be:
  * a run root written by ``simd_harness.runner`` (``<side>/<case>/result.json``),
  * a side directory (``<case>/result.json``),
  * a single ``result.json``.

The script is device-agnostic: it imports no torch/triton/tilelang.  Arrays are
read from the raw-bit JSON payloads, so NaN payload checks are possible
(``--nan-payload strict``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from simd_harness.cases import CASES
from simd_harness.encoding import read_json, tensor_from_json
from simd_harness.reference import compare_output
from simd_harness.spec import OutputSpec


def _collect(root: Path) -> dict:
    root = Path(root)
    results = {}
    if root.is_file() and root.name == "result.json":
        candidates = [root]
    else:
        candidates = list(root.rglob("result.json"))
    for path in candidates:
        try:
            obj = read_json(path)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
            continue
        case = obj.get("case")
        side = obj.get("side")
        if not case or not side:
            continue
        results.setdefault(case, {})[side] = (path, obj)
    return results


def _pick(entry: dict, preferred: str | None):
    if not entry:
        return None
    if preferred and preferred in entry:
        return entry[preferred]
    if len(entry) == 1:
        return next(iter(entry.values()))
    return None


def build_parser():
    p = argparse.ArgumentParser(description="Compare two SIMD case result trees.")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--baseline-side")
    p.add_argument("--candidate-side")
    p.add_argument("--nan-payload", choices=("position", "strict"), default="position",
                   help="strict = compare NaN payload bits too")
    p.add_argument("--json", help="write the comparison report here")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    base = _collect(Path(args.baseline))
    cand = _collect(Path(args.candidate))
    common_cases = sorted(set(base) & set(cand))
    if not common_cases:
        print("no common cases between the two result trees", file=sys.stderr)
        return 2

    report = {"schema": 1, "baseline": args.baseline, "candidate": args.candidate,
              "nan_payload": args.nan_payload, "cases": {}}
    all_ok = True
    for case in common_cases:
        b = _pick(base[case], args.baseline_side)
        c = _pick(cand[case], args.candidate_side)
        if b is None or c is None:
            print(f"{case}: ambiguous side selection; use --baseline-side/--candidate-side",
                  file=sys.stderr)
            return 2
        bpath, bres = b
        cpath, cres = c
        spec = CASES.get(case)
        b_out = bres.get("outputs", {})
        c_out = cres.get("outputs", {})
        common_outputs = sorted(set(b_out) & set(c_out))
        case_report = {
            "baseline_result": str(bpath),
            "candidate_result": str(cpath),
            "baseline_side": bres.get("side"),
            "candidate_side": cres.get("side"),
            "baseline_status": bres.get("status"),
            "candidate_status": cres.get("status"),
            "outputs": {},
        }
        comparable_status = {"PASS", "HOST_REFERENCE_ONLY", "SELF_TEST_PASS"}
        if bres.get("status") not in comparable_status or cres.get("status") not in comparable_status:
            case_report["pass"] = False
            case_report["problem"] = (
                f"non-comparable status: baseline={bres.get('status')} candidate={cres.get('status')}")
            all_ok = False
            print(f"{case}: {case_report['problem']}")
        if not common_outputs:
            case_report["pass"] = False
            case_report["problem"] = "no common outputs"
            all_ok = False
            print(f"{case}: no common outputs")
        for name in common_outputs:
            if b_out[name].get("got") is None or c_out[name].get("got") is None:
                case_report["outputs"][name] = {"pass": False, "problem": "missing got payload"}
                all_ok = False
                continue
            got_b = tensor_from_json(b_out[name]["got"])
            got_c = tensor_from_json(c_out[name]["got"])
            if spec is not None and spec.implemented:
                try:
                    rule = spec.output(name)
                except KeyError:
                    rule = OutputSpec(name=name, dtype=got_b.dtype.str)
            else:
                rule = OutputSpec(name=name, dtype=got_b.dtype.str)
            # Cross-side comparison has no per-side active mask; zeroing is
            # already validated against the reference on each side.
            nan_policy = rule.nan_policy
            if args.nan_payload == "strict" and got_b.dtype.kind == "f":
                nan_policy = "canonical_bits"
            rule = OutputSpec(name=name, dtype=got_b.dtype.str, mode=rule.mode,
                              rtol=rule.rtol, atol=rule.atol,
                              nan_policy=nan_policy, signed_zero=rule.signed_zero,
                              zero_inactive=False)
            cmp = compare_output(got_c, got_b, rule)
            case_report["outputs"][name] = cmp
            all_ok = all_ok and cmp["pass"]
            print(f"{case}/{name}: {'OK' if cmp['pass'] else 'MISMATCH'}"
                  + ("" if cmp["pass"] else f" first={cmp['first_fail']}"))
        case_report["pass"] = all(c["pass"] for c in case_report["outputs"].values())
        report["cases"][case] = case_report

    report["pass"] = all_ok
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
            fh.write("\n")
    print("RESULT", "OK" if all_ok else "MISMATCH")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
