"""
The tools the agent gets for one run, bound to one resource type.

Binding the resource type in code (instead of making it a parameter) removes a whole
class of mistakes: the model can't test the wrong resource or mistype its name.
"""
from __future__ import annotations

import json
import re
import ssl
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent_core.memory import Memory
from agent_core.policy import safe_path
from agent_core.tools import Param, Risk, Tool, ToolResult

from .docs import DocIndex
from .terraform import TerraformRunner, coerce_values, nested_blocks, validate_values

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,39}$")


# ── Scenario tracking: the ground truth for "is the task done?" ──────────────
@dataclass
class ScenarioRecord:
    name: str
    expect: str                 # "pass" | "fail"
    purpose: str
    values: dict[str, Any]
    plan: str | None = None     # None | "passed" | "rejected" | "error"
    apply: str | None = None    # None | "passed" | "failed"
    last_error: str = ""


class ScenarioTracker:
    def __init__(self, results_path: Path):
        self.path = results_path
        self.records: dict[str, ScenarioRecord] = {}

    def upsert(self, name: str, expect: str, purpose: str, values: dict[str, Any]) -> None:
        # Rewriting a scenario resets its results: old verdicts don't apply to new values.
        self.records[name] = ScenarioRecord(name, expect, purpose, values)
        self.save()

    def update(self, name: str, **changes: Any) -> None:
        for key, value in changes.items():
            setattr(self.records[name], key, value)
        self.save()

    def duplicate_of(self, name: str, values: dict[str, Any]) -> str | None:
        for other in self.records.values():
            if other.name != name and other.values == values:
                return other.name
        return None

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(r) for r in self.records.values()], indent=2))


def as_expected(r: ScenarioRecord, require_apply: bool) -> bool:
    if r.expect == "fail":
        return r.plan == "rejected"
    return r.plan == "passed" and (not require_apply or r.apply == "passed")


def make_done_check(tracker: ScenarioTracker, positive: int, negative: int, require_apply: bool):
    def check() -> tuple[bool, str]:
        recs = list(tracker.records.values())
        pos = [r for r in recs if r.expect == "pass" and as_expected(r, require_apply)]
        neg = [r for r in recs if r.expect == "fail" and as_expected(r, require_apply)]
        if len(pos) >= positive and len(neg) >= negative:
            return True, ""
        parts = [f"{len(pos)} of {positive} passing scenario(s) verified"]
        if negative:
            parts.append(f"{len(neg)} of {negative} negative scenario(s) rejected as expected")
        if require_apply:
            waiting = [r.name for r in recs if r.expect == "pass" and r.plan == "passed" and r.apply is None]
            if waiting:
                parts.append(f"call apply_scenario for: {', '.join(waiting)}")
        failing = [f"{r.name}: {r.last_error.splitlines()[0][:100]}" for r in recs
                   if not as_expected(r, require_apply) and r.last_error]
        if failing:
            parts.append("still wrong: " + "; ".join(failing[:3]))
        return False, ". ".join(parts) + "."
    return check


# ── Tools ─────────────────────────────────────────────────────────────────────
def build_tools(resource_type: str, runner: TerraformRunner, memory: Memory,
                docs: DocIndex | None, tracker: ScenarioTracker, allow_apply: bool) -> list[Tool]:
    fields = runner.fields(resource_type)  # raises KeyError early if the resource doesn't exist
    resource_dir = runner.resource_dir(resource_type)
    state = {"cleanup_failed": False}

    def describe_resource() -> ToolResult:
        def line(f):
            desc = f.description[:150] + ("..." if len(f.description) > 150 else "")
            return f"  - {f.name} ({f.kind}{', sensitive' if f.sensitive else ''}): {desc or 'no description'}"
        required = [line(f) for f in fields if f.required]
        optional = [line(f) for f in fields if not f.required]
        out = [f"Resource: {resource_type}",
               "Required fields:", *(required or ["  (none)"]),
               "Optional fields:", *(optional or ["  (none)"])]
        blocks = nested_blocks(runner.resource_schema(resource_type) or {})
        if blocks:
            out.append(f"Nested blocks (not supported by this harness; don't set them): {', '.join(blocks)}")
        notes = memory.notes(resource_type)
        if notes:
            out += ["Rules learned in earlier runs:", *[f"  - {n}" for n in notes]]
        return ToolResult.success("\n".join(out))

    def search_docs(query: str) -> ToolResult:
        if docs is None or not docs.ready:
            return ToolResult.failure("no documentation index is available. "
                                      "Rely on describe_resource and Terraform's error messages.")
        hits = docs.search(query, k=4)
        return ToolResult.success("\n\n".join(
            f"[{h.source} | relevance {h.score:.2f}]\n{h.text[:900]}" for h in hits))

    def get_examples() -> ToolResult:
        examples = memory.examples(resource_type, limit=2)
        if not examples:
            return ToolResult.success("No verified examples yet for this resource.")
        return ToolResult.success("\n\n".join(
            f"Example '{e['name']}' (verified by terraform {e['verified_by']}):\n{e['content']}" for e in examples))

    def write_scenario(name: str, expect: str, purpose: str, values: dict[str, Any]) -> ToolResult:
        if not NAME_RE.match(name):
            return ToolResult.failure("name must be snake_case, start with a letter, 2-40 characters")
        values = coerce_values(fields, values)
        problems = validate_values(fields, values)
        if problems:
            return ToolResult.failure("scenario not written:\n  - " + "\n  - ".join(problems))
        duplicate = tracker.duplicate_of(name, values)
        if duplicate:
            return ToolResult.failure(f"these values are identical to scenario '{duplicate}'. "
                                      "Each scenario should test something different.")
        path = safe_path(resource_dir, "scenarios", f"{name}.tfvars.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
        tracker.upsert(name, expect, purpose, values)
        return ToolResult.success(f"Wrote scenario '{name}' (expect={expect}). Next: plan_scenario(name='{name}').")

    def plan_scenario(name: str) -> ToolResult:
        rec = tracker.records.get(name)
        if rec is None:
            return ToolResult.failure(f"no scenario named '{name}' in this run. Write it first with write_scenario.")
        outcome = runner.plan(resource_type, name)

        if outcome.stage == "init" and not outcome.ok:
            tracker.update(name, plan="error", last_error=outcome.for_model())
            return ToolResult.failure("environment problem, not a scenario problem. Stop and report this:\n"
                                      + outcome.for_model())
        if outcome.ok:
            tracker.update(name, plan="passed", last_error="")
            if rec.expect == "pass":
                content = runner.scenario_path(resource_type, name).read_text()
                memory.add_example(resource_type, name, content, verified_by="plan")
                return ToolResult.success(f"{outcome.for_model()} Scenario '{name}' passes as expected.")
            tracker.update(name, last_error="provider accepted a value that was expected to be rejected")
            return ToolResult.failure(f"plan passed, but '{name}' is expected to FAIL. The provider accepted it, "
                                      "so it doesn't test a rejection. Use a value the provider should reject.")
        # Plan-stage failure
        tracker.update(name, plan="rejected", last_error=outcome.for_model())
        if rec.expect == "fail":
            first = outcome.errors[0].splitlines()[0] if outcome.errors else "rejected"
            return ToolResult.success(f"Plan rejected '{name}' as expected: {first}")
        return ToolResult.failure(outcome.for_model() + "\n\nFix only what the error points to, then call "
                                  f"write_scenario again with name='{name}'.")

    def apply_scenario(name: str) -> ToolResult:
        if state["cleanup_failed"]:
            return ToolResult.failure("refusing to apply: an earlier cleanup failed. Stop and report it.")
        rec = tracker.records.get(name)
        if rec is None or rec.expect != "pass" or rec.plan != "passed":
            return ToolResult.failure(f"'{name}' must be a scenario with expect='pass' that has passed plan_scenario.")
        applied, destroyed = runner.apply_and_destroy(resource_type, name)
        if destroyed is not None and not destroyed.ok:
            state["cleanup_failed"] = True
            tracker.update(name, apply="failed", last_error=destroyed.for_model())
            return ToolResult.failure(destroyed.for_model() + "\nStop applying and report this to the user.")
        if not applied.ok:
            tracker.update(name, apply="failed", last_error=applied.for_model())
            return ToolResult.failure(applied.for_model() + f"\n\nFix the values and write_scenario '{name}' again.")
        tracker.update(name, apply="passed", last_error="")
        content = runner.scenario_path(resource_type, name).read_text()
        memory.add_example(resource_type, name, content, verified_by="apply")
        return ToolResult.success(f"APPLY OK: '{name}' was created on NIOS and then destroyed.")

    def remember(note: str) -> ToolResult:
        note = note.strip()
        if not 10 <= len(note) <= 300:
            return ToolResult.failure("a note must be 10-300 characters: one specific, reusable rule")
        is_new = memory.add_note(resource_type, note)
        return ToolResult.success("Saved." if is_new else "Already known.")

    def list_scenarios() -> ToolResult:
        if not tracker.records:
            return ToolResult.success("No scenarios written yet.")
        lines = [f"{'Name':<35} {'Expect':<7} {'Plan':<9} {'Apply':<8} Done"]
        lines.append("-" * 70)
        for r in tracker.records.values():
            done = "YES" if as_expected(r, False) else "no"
            lines.append(f"{r.name:<35} {r.expect:<7} {r.plan or '-':<9} {r.apply or '-':<8} {done}")
        pending_plan = [r.name for r in tracker.records.values() if r.plan is None]
        if pending_plan:
            lines.append(f"\nNeeds plan_scenario: {', '.join(pending_plan)}")
        return ToolResult.success("\n".join(lines))

    def fix_scenario(name: str, changes: dict[str, Any]) -> ToolResult:
        rec = tracker.records.get(name)
        if rec is None:
            return ToolResult.failure(f"no scenario named '{name}'. Use list_scenarios to see what exists.")
        path = safe_path(resource_dir, "scenarios", f"{name}.tfvars.json")
        if not path.exists():
            return ToolResult.failure(f"scenario file for '{name}' not found on disk.")
        current = json.loads(path.read_text())
        updated = {**current, **changes}
        updated = coerce_values(fields, updated)
        problems = validate_values(fields, updated)
        if problems:
            return ToolResult.failure("fix not applied:\n  - " + "\n  - ".join(problems))
        duplicate = tracker.duplicate_of(name, updated)
        if duplicate:
            return ToolResult.failure(f"after this fix the values would be identical to '{duplicate}'.")
        path.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
        tracker.upsert(name, rec.expect, rec.purpose, updated)
        changed = ", ".join(f"{k}={v!r}" for k, v in changes.items())
        return ToolResult.success(f"Fixed '{name}': {changed}. Plan status reset. Next: plan_scenario(name='{name}').")

    def wapi_lookup(object_type: str, fields: list[str] | None = None) -> ToolResult:
        host = runner.s.nios_host_url.rstrip("/")
        user = runner.s.nios_username
        pwd  = runner.s.nios_password
        if not host:
            return ToolResult.failure("NIOS_HOST_URL is not set — cannot query the grid.")
        field_list = ",".join(fields) if fields else "_basic"
        url = f"{host}/wapi/v2.1/{object_type}?_return_fields={field_list}&_max_results=3"
        req = urllib.request.Request(url)
        import base64
        token = base64.b64encode(f"{user}:{pwd}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")
        req.add_header("Content-Type", "application/json")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            return ToolResult.failure(f"WAPI {object_type} returned HTTP {e.code}: {body}")
        except Exception as e:
            return ToolResult.failure(f"WAPI request failed: {e}")
        if not data:
            return ToolResult.success(f"No {object_type} objects found on the grid (empty result).")
        lines = [f"Found {len(data)} {object_type} object(s) on the grid:"]
        for obj in data:
            ref = obj.pop("_ref", "")
            lines.append(f"\n  ref: {ref}")
            for k, v in obj.items():
                lines.append(f"    {k}: {v!r}")
        return ToolResult.success("\n".join(lines))

    def summarize_errors() -> ToolResult:
        recs = list(tracker.records.values())
        if not recs:
            return ToolResult.success("No scenarios written yet.")
        failing = [r for r in recs if r.last_error]
        if not failing:
            return ToolResult.success("No errors recorded — all scenarios are passing as expected.")
        # Group errors by first meaningful line
        from collections import Counter
        error_lines: list[str] = []
        for r in failing:
            first = next((ln.strip() for ln in r.last_error.splitlines() if ln.strip()), r.last_error[:100])
            error_lines.append(first)
        counts = Counter(error_lines)
        lines = ["Error patterns across all scenarios:", ""]
        for msg, cnt in counts.most_common():
            names = [r.name for r in failing if msg in r.last_error]
            lines.append(f"  [{cnt}x] {msg}")
            lines.append(f"        Affects: {', '.join(names)}")
        unsaved = [msg for msg, _ in counts.most_common() if not any(msg in n for n in memory.notes(resource_type))]
        if unsaved:
            lines.append("\nConsider calling remember() for patterns not yet saved to memory.")
        return ToolResult.success("\n".join(lines))

    name_param = Param("name", "string", "Scenario name, snake_case, e.g. required_fields_only")
    tools = [
        Tool("describe_resource",
             "List the fields of the resource under test: which are required, their types, and "
             "rules learned in earlier runs. Call this first.", [], describe_resource),
        Tool("list_scenarios",
             "Show all scenarios written so far with their plan/apply status and whether they are done. "
             "Call this to check progress and find scenarios that still need plan_scenario.",
             [], list_scenarios),
        Tool("search_docs",
             "Search the Infoblox WAPI reference for a field's format, allowed values, or dependencies "
             "between fields. Use when a field's rules are unclear or Terraform rejects a value.",
             [Param("query", "string", "What you want to know, e.g. 'allowed values for record ttl'")],
             search_docs),
        Tool("wapi_lookup",
             "Query the live NIOS grid via WAPI to see real objects and their actual field values. "
             "Use to discover valid enum values or field formats from existing grid objects.",
             [Param("object_type", "string", "WAPI object type, e.g. 'record:a', 'record:host', 'network'"),
              Param("fields", "array", "Field names to return, e.g. ['name','ipv4addr','creator']. "
                    "Omit to get basic fields.", required=False)],
             wapi_lookup),
        Tool("get_examples", "Get scenario files for this resource that Terraform already accepted.",
             [], get_examples),
        Tool("write_scenario",
             "Create or overwrite a test scenario. Values are checked against the schema before saving.",
             [name_param,
              Param("expect", "string", "'pass' if Terraform should accept it, 'fail' if it should be rejected",
                    enum=["pass", "fail"]),
              Param("purpose", "string", "One sentence: what this scenario tests"),
              Param("values", "object", "Field names mapped to values. Omit optional fields you aren't testing.")],
             write_scenario, Risk.WRITE),
        Tool("fix_scenario",
             "Patch specific fields in an existing scenario without rewriting it entirely. "
             "Use this for targeted corrections (e.g. wrong enum value) instead of a full write_scenario.",
             [name_param,
              Param("changes", "object", "Only the fields to change, e.g. {\"creator\": \"STATIC\"}")],
             fix_scenario, Risk.WRITE),
        Tool("plan_scenario", "Run terraform plan for a scenario and report whether it matched its expectation.",
             [name_param], plan_scenario),
        Tool("summarize_errors",
             "Group and count repeated errors across all scenarios to spot patterns. "
             "Call this when the same error is appearing on multiple scenarios.",
             [], summarize_errors),
        Tool("remember",
             "Save a rule you learned from an error so future runs avoid it, "
             "e.g. 'ttl must be between 0 and 2147483647'.",
             [Param("note", "string", "One specific rule")], remember, Risk.WRITE),
    ]
    if allow_apply:
        tools.append(Tool("apply_scenario",
                          "Create the scenario's object on the NIOS test grid, then destroy it. "
                          "Only for expect='pass' scenarios that passed plan. Requires user approval.",
                          [name_param], apply_scenario, Risk.DESTRUCTIVE))

        def apply_all_passing() -> ToolResult:
            if state["cleanup_failed"]:
                return ToolResult.failure("refusing to apply: an earlier cleanup failed. Stop and report it.")
            eligible = [r for r in tracker.records.values()
                        if r.expect == "pass" and r.plan == "passed" and r.apply is None]
            if not eligible:
                return ToolResult.success("No scenarios are ready to apply (need expect='pass' and plan=passed).")
            results = []
            for rec in eligible:
                result = apply_scenario(rec.name)
                results.append(f"{'OK' if result.success else 'FAIL'}  {rec.name}: {result.output[:120]}")
                if not result.success and state["cleanup_failed"]:
                    results.append("Stopping: cleanup failed.")
                    break
            return ToolResult(success=not state["cleanup_failed"], output="\n".join(results))

        tools.append(Tool("apply_all_passing",
                          "Apply and destroy all expect='pass' scenarios that passed plan in one batch. "
                          "Stops immediately if a cleanup fails.",
                          [], apply_all_passing, Risk.DESTRUCTIVE))
    return tools
