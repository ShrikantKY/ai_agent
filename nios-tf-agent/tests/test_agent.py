import json
import tempfile
import unittest
from pathlib import Path

from agent_core.llm import LLMError
from agent_core.loop import COMPACTED, Agent
from agent_core.memory import Memory
from agent_core.policy import Policy, Redactor, deny_all
from agent_core.tools import Param, Risk, Tool, ToolRegistry, ToolResult
from agent_core.tracing import NullTracer
from nios.app import RunOptions, run_agent
from nios.terraform import TerraformRunner
from nios.toolset import ScenarioTracker, build_tools
from tests.fakes import TF_ERROR_OUTPUT, FakeTerraform, ScriptedLLM, call, make_settings, turn


def _agent(llm, tools, done_check_policy=None, **kw):
    return Agent(llm, ToolRegistry(tools), done_check_policy or Policy(deny_all), NullTracer(),
                 "system", echo=None, **kw)


def _echo_tool():
    return Tool("echo", "echo text", [Param("text", "string", "text")],
                lambda text: ToolResult.success(f"echo: {text}"))


def _last_tool_output(llm):
    return [m for m in llm.seen[-1] if m["role"] == "tool"][-1]["content"]


class LoopTests(unittest.TestCase):
    def test_happy_path(self):
        llm = ScriptedLLM([turn(call("echo", text="hi")), turn(content="all done")])
        result = _agent(llm, [_echo_tool()]).run("task")
        self.assertEqual((result.status, result.summary, result.tool_calls), ("done", "all done", 1))
        self.assertEqual(_last_tool_output(llm), "echo: hi")

    def test_unknown_tool_and_bad_args_become_messages_not_crashes(self):
        llm = ScriptedLLM([turn(call("nope")), turn(call("echo", txt="x")), turn(content="ok")])
        result = _agent(llm, [_echo_tool()]).run("task")
        self.assertEqual(result.status, "done")
        tool_msgs = [m["content"] for m in llm.seen[-1] if m["role"] == "tool"]
        self.assertIn("unknown tool 'nope'", tool_msgs[0])
        self.assertIn("missing required argument 'text'", tool_msgs[1])

    def test_tool_exception_is_contained(self):
        def boom(text):
            raise RuntimeError("kaboom")
        tool = Tool("echo", "d", [Param("text", "string", "t")], boom)
        llm = ScriptedLLM([turn(call("echo", text="x")), turn(content="ok")])
        self.assertEqual(_agent(llm, [tool]).run("task").status, "done")
        self.assertIn("crashed: RuntimeError: kaboom", _last_tool_output(llm))

    def test_destructive_tool_denied_without_approval(self):
        ran = []
        tool = Tool("destroy", "d", [], lambda: ran.append(1) or ToolResult.success("gone"), Risk.DESTRUCTIVE)
        llm = ScriptedLLM([turn(call("destroy")), turn(content="ok")])
        _agent(llm, [tool]).run("task")
        self.assertEqual(ran, [])
        self.assertIn("DENIED", _last_tool_output(llm))

    def test_done_check_nudges_then_gives_up(self):
        state = {"done": False}
        llm = ScriptedLLM([turn(content="done!"), turn(call("echo", text="x")), turn(content="really done")])

        def check():
            return state["done"], "scenario X still failing."

        def echo(text):
            state["done"] = True
            return ToolResult.success("fixed")

        tool = Tool("echo", "d", [Param("text", "string", "t")], echo)
        result = _agent(llm, [tool]).run("task", done_check=check)
        self.assertEqual(result.status, "done")
        nudge = [m for m in llm.seen[1] if m["role"] == "user"][-1]["content"]
        self.assertIn("scenario X still failing", nudge)

        stubborn = ScriptedLLM([turn(content="done")] * 5)
        result = _agent(stubborn, [tool]).run("task", done_check=lambda: (False, "nope."), max_nudges=2)
        self.assertEqual(result.status, "incomplete")

    def test_step_limit_and_llm_errors(self):
        looping = ScriptedLLM([turn(call("echo", text="x"))] * 10)
        self.assertEqual(_agent(looping, [_echo_tool()], max_steps=3).run("t").status, "max_steps")
        broken = ScriptedLLM([LLMError("ollama down")])
        self.assertEqual(_agent(broken, [_echo_tool()]).run("t").status, "llm_error")

    def test_truncation_compaction_and_redaction(self):
        big = Tool("big", "d", [], lambda: ToolResult.success("SECRETVALUE " + "x" * 10_000))
        llm = ScriptedLLM([turn(call("big")) for _ in range(6)] + [turn(content="ok")])
        agent = _agent(llm, [big], max_tool_output=500, keep_recent_messages=4,
                       redact=Redactor(["SECRETVALUE"]))
        agent.run("t")
        tool_msgs = [m["content"] for m in llm.seen[-1] if m["role"] == "tool"]
        self.assertEqual(tool_msgs[0], COMPACTED)
        self.assertIn("characters truncated", tool_msgs[-1])
        self.assertLessEqual(len(tool_msgs[-1]), 600)
        self.assertNotIn("SECRETVALUE", tool_msgs[-1])


class EndToEndTests(unittest.TestCase):
    """The real toolset and loop, with a scripted model and a fake terraform binary."""

    def setUp(self):
        self.settings = make_settings(Path(tempfile.mkdtemp()))

    def _run(self, script, fake, positive=1, negative=1, apply=False, approve=False):
        runner = TerraformRunner(self.settings, executor=fake)
        llm = ScriptedLLM(script)
        report = run_agent(self.settings, RunOptions("nios_dns_record_a", positive, negative, apply),
                           llm=llm, runner=runner, approver=lambda n, a: approve, echo=None)
        return report, llm

    def test_positive_and_negative_scenarios_with_one_repair(self):
        fake = FakeTerraform(plan_results={
            "basic": [(1, TF_ERROR_OUTPUT), (0, "Plan: 1 to add, 0 to change, 0 to destroy.")],
            "negative_ttl": (1, TF_ERROR_OUTPUT),
        })
        script = [
            turn(call("describe_resource")),
            turn(call("write_scenario", name="basic", expect="pass", purpose="required only",
                      values={"name": "a.example.com", "ttl": -1})),
            turn(call("plan_scenario", name="basic")),
            turn(call("remember", note="ttl must be between 0 and 2147483647")),
            turn(call("write_scenario", name="basic", expect="pass", purpose="required only",
                      values={"name": "a.example.com", "ttl": "300"})),
            turn(call("plan_scenario", name="basic")),
            turn(call("write_scenario", name="negative_ttl", expect="fail", purpose="ttl below zero",
                      values={"name": "b.example.com", "ttl": -5})),
            turn(call("plan_scenario", name="negative_ttl")),
            turn(content="Both scenarios behave as expected."),
        ]
        report, llm = self._run(script, fake)
        self.assertEqual(report.result.status, "done")
        recs = report.tracker.records
        self.assertEqual((recs["basic"].plan, recs["negative_ttl"].plan), ("passed", "rejected"))

        written = json.loads(Path(self.settings.workspace, "terraform", "nios_dns_record_a",
                                  "scenarios", "basic.tfvars.json").read_text())
        self.assertEqual(written["ttl"], 300)  # the string "300" was coerced to a number

        memory = Memory(self.settings.memory_db)
        self.assertEqual([e["name"] for e in memory.examples("nios_dns_record_a")], ["basic"])  # negatives aren't examples
        self.assertEqual(memory.notes("nios_dns_record_a"), ["ttl must be between 0 and 2147483647"])

        trace = Path(report.trace_path).read_text()
        self.assertIn('"event": "run_end"', trace)
        self.assertNotIn("s3cret-password", trace)

    def test_model_cannot_claim_success_early(self):
        script = [turn(call("describe_resource")), turn(content="Done!"), turn(content="Done!"),
                  turn(content="Done!"), turn(content="Done!")]
        report, llm = self._run(script, FakeTerraform(), positive=1, negative=0)
        self.assertEqual(report.result.status, "incomplete")
        self.assertIn("0 of 1 passing", report.result.summary)

    def test_schema_violations_duplicates_and_accepted_negatives_are_reported(self):
        script = [
            turn(call("write_scenario", name="bad", expect="pass", purpose="x", values={"bogus": 1})),
            turn(call("write_scenario", name="one", expect="pass", purpose="x", values={"name": "a.example.com"})),
            turn(call("write_scenario", name="two", expect="pass", purpose="x", values={"name": "a.example.com"})),
            turn(call("write_scenario", name="neg", expect="fail", purpose="x", values={"name": "c.example.com"})),
            turn(call("plan_scenario", name="neg")),
            turn(content="done"),
        ]
        report, llm = self._run(script, FakeTerraform(), positive=0, negative=1)
        outputs = [m["content"] for m in llm.seen[-1] if m["role"] == "tool"]
        self.assertIn("unknown field(s) ['bogus']", outputs[0])
        self.assertIn("identical to scenario 'one'", outputs[2])
        self.assertIn("expected to FAIL", outputs[4])
        self.assertNotEqual(report.result.status, "done")

    def test_apply_requires_approval(self):
        script = [
            turn(call("write_scenario", name="basic", expect="pass", purpose="x", values={"name": "a.example.com"})),
            turn(call("plan_scenario", name="basic")),
            turn(call("apply_scenario", name="basic")),
            turn(content="done"),
        ]
        fake = FakeTerraform()
        denied, _ = self._run(list(script), fake, positive=1, negative=0, apply=True, approve=False)
        self.assertNotIn("apply", [c[0] for c in fake.calls])
        self.assertNotEqual(denied.result.status, "done")

        fake = FakeTerraform()
        self.settings = make_settings(Path(tempfile.mkdtemp()))
        approved, _ = self._run(list(script), fake, positive=1, negative=0, apply=True, approve=True)
        self.assertEqual(approved.result.status, "done")
        self.assertEqual([c[0] for c in fake.calls if c[0] in ("apply", "destroy")], ["apply", "destroy"])

    def test_apply_tool_hidden_unless_enabled(self):
        runner = TerraformRunner(self.settings, executor=FakeTerraform())
        tracker = ScenarioTracker(Path(tempfile.mkdtemp()) / "r.json")
        names = [t.name for t in build_tools("nios_dns_record_a", runner, Memory(":memory:"), None, tracker, False)]
        self.assertNotIn("apply_scenario", names)


if __name__ == "__main__":
    unittest.main()
