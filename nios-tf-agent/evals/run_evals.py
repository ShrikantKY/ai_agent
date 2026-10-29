#!/usr/bin/env python3
"""
Run the agent on every case in cases.json (plan only, never apply) and record metrics.
Compare against the previous run so every change to a prompt, tool, or model is measured.

  python evals/run_evals.py                 # all cases
  python evals/run_evals.py --only record   # cases whose resource name contains "record"
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_core.policy import deny_all  # noqa: E402
from config import load_settings  # noqa: E402
from nios.app import RunOptions, _make_llm, preflight, run_agent  # noqa: E402
from nios.terraform import TerraformRunner  # noqa: E402
from nios.toolset import as_expected  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="substring filter on resource type")
    args = ap.parse_args()

    s = load_settings()
    problems = preflight(s, need_nios=True)
    if problems:
        print("Cannot start:\n  - " + "\n  - ".join(problems))
        return 2

    llm, provider = _make_llm(s)
    print(f"Using {s.model} via {provider}")

    known = set(TerraformRunner(s).resource_types())
    cases = [c for c in json.loads((HERE / "cases.json").read_text()) if args.only in c["resource_type"]]
    rows = []
    for case in cases:
        rt = case["resource_type"]
        if rt not in known:
            print(f"SKIP {rt}: not in the provider schema")
            continue
        print(f"RUN  {rt} ...", flush=True)
        opts = RunOptions(rt, case.get("positive", 3), case.get("negative", 1), apply=False)
        report = run_agent(s, opts, llm=llm, approver=deny_all, echo=None)
        recs = list(report.tracker.records.values())
        rows.append({
            "resource_type": rt,
            "status": report.result.status,
            "steps": report.result.steps,
            "tool_calls": report.result.tool_calls,
            "seconds": report.result.seconds,
            "scenarios_written": len(recs),
            "scenarios_as_expected": sum(as_expected(r, False) for r in recs),
            "trace": report.trace_path,
        })
        print(f"     {rows[-1]['status']:<11} steps={rows[-1]['steps']:<3} {rows[-1]['seconds']}s")

    if not rows:
        print("No cases ran.")
        return 1

    solved = sum(r["status"] == "done" for r in rows)
    summary = {
        "model": s.model,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cases": len(rows),
        "solved": solved,
        "solve_rate": round(solved / len(rows), 3),
        "avg_steps": round(sum(r["steps"] for r in rows) / len(rows), 1),
        "avg_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 1),
        "rows": rows,
    }

    RESULTS.mkdir(exist_ok=True)
    previous = sorted(RESULTS.glob("*.json"))
    out = RESULTS / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"\nSolved {solved}/{len(rows)} ({summary['solve_rate']:.0%}), "
          f"avg {summary['avg_steps']} steps, {summary['avg_seconds']}s per case")
    if previous:
        last = json.loads(previous[-1].read_text())
        delta = summary["solve_rate"] - last["solve_rate"]
        print(f"Previous run ({last['model']}, {last['timestamp']}): {last['solve_rate']:.0%}  →  change {delta:+.0%}")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
