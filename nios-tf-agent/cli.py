#!/usr/bin/env python3
"""
Usage:
  python cli.py list [filter]                      list resource types in the provider schema
  python cli.py describe <resource_type>           show fields (no model involved)
  python cli.py plan <resource_type> <scenario>    plan one scenario file by hand (no model)
  python cli.py run <resource_type> [--positive 3] [--negative 1] [--apply]
  python cli.py index-docs <file.txt> [more files]  build the documentation search index
  python cli.py cleanup <resource_type> <scenario>  destroy leftovers after a failed cleanup
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_core.llm import OllamaClient
from config import load_settings
from nios.app import RunOptions, preflight, run_agent
from nios.docs import DocIndex
from nios.terraform import TerraformRunner, parse_resource_fields
from nios.toolset import as_expected


def cmd_list(args, s) -> int:
    names = [n for n in TerraformRunner(s).resource_types() if args.filter in n]
    print("\n".join(names) or "no matches")
    return 0


def cmd_describe(args, s) -> int:
    runner = TerraformRunner(s)
    rs = runner.resource_schema(args.resource_type)
    if rs is None:
        print(f"unknown resource type: {args.resource_type}")
        return 1
    for f in parse_resource_fields(rs):
        print(f"{'required' if f.required else 'optional'}  {f.name:<32} {f.kind:<8} {f.description[:70]}")
    return 0


def cmd_plan(args, s) -> int:
    outcome = TerraformRunner(s).plan(args.resource_type, args.scenario)
    print(outcome.for_model())
    return 0 if outcome.ok else 1


def cmd_cleanup(args, s) -> int:
    outcome = TerraformRunner(s).cleanup(args.resource_type, args.scenario)
    print(outcome.for_model())
    return 0 if outcome.ok else 1


def cmd_index_docs(args, s) -> int:
    llm = OllamaClient(s.ollama_url, s.model)
    if s.docs_index.exists():
        s.docs_index.unlink()  # rebuild from scratch
    index = DocIndex(s.docs_index, lambda texts: llm.embed(texts, s.embed_model), s.embed_model)
    count = index.build([Path(f) for f in args.files])
    print(f"Indexed {count} chunks into {s.docs_index}")
    return 0


def cmd_run(args, s) -> int:
    problems = preflight(s)
    if problems:
        print("Cannot start:\n  - " + "\n  - ".join(problems))
        return 2

    provider = "Claude" if s.anthropic_api_key else "Ollama"
    opts = RunOptions(args.resource_type, args.positive, args.negative, args.apply)
    print(f"Testing {opts.resource_type} with {s.model} via {provider} "
          f"({opts.positive} passing, {opts.negative} negative{', with apply' if opts.apply else ''})\n")
    report = run_agent(s, opts)
    r = report.result

    print(f"\n{'=' * 72}\nStatus: {r.status.upper()}  |  steps {r.steps}  |  tool calls {r.tool_calls}  |  {r.seconds}s")
    print(f"{'Scenario':<32} {'Expect':<7} {'Plan':<9} {'Apply':<7} OK")
    for rec in report.tracker.records.values():
        ok = "✓" if as_expected(rec, opts.apply) else "✗"
        print(f"{rec.name:<32} {rec.expect:<7} {rec.plan or '-':<9} {rec.apply or '-':<7} {ok}")
    print(f"\nSummary: {r.summary}")
    print(f"Trace:   {report.trace_path}")
    print(f"Results: {report.tracker.path}")
    return 0 if r.status == "done" else 1


def main() -> int:
    p = argparse.ArgumentParser(description="NIOS Terraform test agent")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("list")
    sp.add_argument("filter", nargs="?", default="")
    sp.set_defaults(fn=cmd_list)

    sp = sub.add_parser("describe")
    sp.add_argument("resource_type")
    sp.set_defaults(fn=cmd_describe)

    for name, fn in [("plan", cmd_plan), ("cleanup", cmd_cleanup)]:
        sp = sub.add_parser(name)
        sp.add_argument("resource_type")
        sp.add_argument("scenario")
        sp.set_defaults(fn=fn)

    sp = sub.add_parser("index-docs")
    sp.add_argument("files", nargs="+")
    sp.set_defaults(fn=cmd_index_docs)

    sp = sub.add_parser("run")
    sp.add_argument("resource_type")
    sp.add_argument("--positive", type=int, default=3)
    sp.add_argument("--negative", type=int, default=1)
    sp.add_argument("--apply", action="store_true", help="also create and destroy on NIOS (asks first)")
    sp.set_defaults(fn=cmd_run)

    args = p.parse_args()
    try:
        return args.fn(args, load_settings())
    except KeyError as e:
        print(f"Unknown resource type: {e}. See: python cli.py list")
    except RuntimeError as e:
        print(f"Error: {e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
