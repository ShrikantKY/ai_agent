import tempfile
import unittest
from pathlib import Path

from agent_core.memory import Memory
from agent_core.policy import Policy, Redactor, deny_all, safe_path
from agent_core.tools import Param, Risk, Tool, ToolArgumentError, ToolRegistry, ToolResult
from nios.docs import chunk_text
from nios.terraform import (TerraformRunner, coerce_values, extract_errors, parse_resource_fields,
                            render_main_tf, render_variables_tf, validate_values)
from tests.fakes import SAMPLE_SCHEMA, TF_ERROR_OUTPUT, FakeTerraform, make_settings

RECORD_A = SAMPLE_SCHEMA["provider_schemas"]["registry.terraform.io/infobloxopen/nios"]["resource_schemas"]["nios_dns_record_a"]


def _tool(risk=Risk.READ):
    return Tool("t", "test tool", [Param("n", "integer", "a number"),
                                   Param("mode", "string", "a mode", enum=["a", "b"]),
                                   Param("opt", "boolean", "optional flag", required=False)],
                lambda **kw: ToolResult.success(str(kw)), risk)


class ToolTests(unittest.TestCase):
    def test_coerces_common_model_mistakes(self):
        self.assertEqual(_tool().validate({"n": "5", "mode": "a", "opt": "true"}), {"n": 5, "mode": "a", "opt": True})

    def test_bool_is_not_accepted_as_integer(self):
        with self.assertRaises(ToolArgumentError):
            _tool().validate({"n": True, "mode": "a"})

    def test_reports_every_problem_at_once(self):
        with self.assertRaises(ToolArgumentError) as ctx:
            _tool().validate({"mode": "z", "extra": 1})
        msg = str(ctx.exception)
        for fragment in ("unknown argument", "missing required argument 'n'", "one of ['a', 'b']"):
            self.assertIn(fragment, msg)

    def test_schema_marks_required_params(self):
        params = _tool().schema()["function"]["parameters"]
        self.assertEqual(params["required"], ["n", "mode"])
        self.assertEqual(params["properties"]["mode"]["enum"], ["a", "b"])

    def test_registry_rejects_duplicate_names(self):
        with self.assertRaises(ValueError):
            ToolRegistry([_tool(), _tool()])


class PolicyTests(unittest.TestCase):
    def test_safe_path_blocks_escapes(self):
        root = Path(tempfile.mkdtemp())
        self.assertEqual(safe_path(root, "a", "b.txt"), (root / "a" / "b.txt").resolve())
        for bad in (("..", "x"), ("/etc/passwd",), ("a", "..", "..", "x")):
            with self.assertRaises(ValueError):
                safe_path(root, *bad)

    def test_destructive_needs_approval(self):
        destructive = _tool(Risk.DESTRUCTIVE)
        self.assertIsNone(Policy(deny_all).check(_tool(Risk.READ), {}))
        self.assertIn("did not approve", Policy(deny_all).check(destructive, {}))
        self.assertIsNone(Policy(lambda name, args: True).check(destructive, {}))

    def test_blocked_tools_are_refused_even_if_read_only(self):
        self.assertIn("disabled", Policy(blocked_tools={"t"}).check(_tool(), {}))

    def test_redactor(self):
        r = Redactor(["hunter2-long", "ab"])  # secrets shorter than 4 chars are ignored
        self.assertEqual(r("pw=hunter2-long ab"), "pw=*** ab")


class MemoryTests(unittest.TestCase):
    def test_notes_are_deduplicated_and_apply_examples_rank_first(self):
        m = Memory(":memory:")
        self.assertTrue(m.add_note("r", "ttl must be >= 0"))
        self.assertFalse(m.add_note("r", "ttl must be >= 0 "))
        m.add_example("r", "one", '{"a": 1}', "plan")
        m.add_example("r", "two", '{"a": 2}', "apply")
        self.assertEqual([e["name"] for e in m.examples("r")], ["two", "one"])
        m.add_example("r", "one", '{"a": 1}', "apply")  # same content upgraded to apply
        self.assertEqual({e["verified_by"] for e in m.examples("r")}, {"apply"})


class TerraformHelperTests(unittest.TestCase):
    def setUp(self):
        self.fields = parse_resource_fields(RECORD_A)

    def test_only_settable_fields_are_exposed(self):
        names = [f.name for f in self.fields]
        self.assertEqual(names, ["comment", "extattrs", "ipv4addr", "name", "ttl", "use_ttl", "view"])
        kinds = {f.name: f.kind for f in self.fields}
        self.assertEqual((kinds["ttl"], kinds["use_ttl"], kinds["extattrs"]), ("number", "bool", "complex"))

    def test_generated_files(self):
        main = render_main_tf("nios_dns_record_a", self.fields)
        self.assertIn('resource "nios_dns_record_a" "under_test"', main)
        self.assertIn("name     = var.name", main)
        self.assertIn('variable "ttl" {\n  type    = number\n  default = null\n}', render_variables_tf(self.fields))

    def test_value_validation(self):
        problems = validate_values(self.fields, {"bogus": 1, "ttl": "abc", "comment": "<your comment>"})
        text = " ".join(problems)
        for fragment in ("unknown field(s) ['bogus']", "required field 'name' is missing",
                         "'ttl' must be a number", "looks like a placeholder"):
            self.assertIn(fragment, text)
        self.assertEqual(validate_values(self.fields, {"name": "a.example.com", "ttl": 300}), [])

    def test_coercion(self):
        out = coerce_values(self.fields, {"ttl": "300", "use_ttl": "true", "extattrs": '{"Site": "HQ"}'})
        self.assertEqual(out, {"ttl": 300, "use_ttl": True, "extattrs": {"Site": "HQ"}})

    def test_extract_errors_strips_box_drawing(self):
        errors = extract_errors(TF_ERROR_OUTPUT)
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith("Error: Invalid Attribute Value"))
        self.assertIn("must be between 0 and 2147483647", errors[0])
        self.assertNotIn("│", errors[0])

    def test_failure_headline_shows_the_detail(self):
        from nios.terraform import TfOutcome
        headline = TfOutcome(False, "plan", "x", extract_errors(TF_ERROR_OUTPUT)).for_model().splitlines()[0]
        self.assertEqual(headline, "PLAN FAILED: Invalid Attribute Value: "
                                   "Attribute ttl value must be between 0 and 2147483647, got: -1")

    def test_chunking_keeps_order_and_size(self):
        text = "short para\n\n" + ("x" * 3000) + "\n\nend para"
        chunks = chunk_text(text, size=1000, overlap=100)
        self.assertEqual(chunks[0], "short para")
        self.assertTrue(chunks[-1].endswith("end para"))
        self.assertTrue(all(len(c) <= 1200 for c in chunks))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.settings = make_settings(Path(tempfile.mkdtemp()))

    def _write(self, runner, name, values='{"name": "a.example.com"}'):
        path = runner.scenario_path("nios_dns_record_a", name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(values)

    def test_plan_generates_files_inits_once_and_passes_secrets_via_env(self):
        seen_env = {}
        fake = FakeTerraform()

        def spy(argv, cwd, env, timeout):
            seen_env.update(env)
            return fake(argv, cwd, env, timeout)

        runner = TerraformRunner(self.settings, executor=spy)
        self._write(runner, "s1")
        self.assertTrue(runner.plan("nios_dns_record_a", "s1").ok)
        self.assertTrue(runner.plan("nios_dns_record_a", "s1").ok)
        commands = [c[0] for c in fake.calls]
        self.assertEqual(commands.count("init"), 2)  # once for the schema dir, once for the resource dir
        self.assertEqual(seen_env["TF_VAR_harness_password"], "s3cret-password")
        for f in runner.resource_dir("nios_dns_record_a").glob("*.tf"):
            self.assertNotIn("s3cret-password", f.read_text())

    def test_apply_refuses_dirty_state(self):
        runner = TerraformRunner(self.settings, executor=FakeTerraform(state="nios_dns_record_a.under_test"))
        self._write(runner, "s1")
        applied, destroyed = runner.apply_and_destroy("nios_dns_record_a", "s1")
        self.assertFalse(applied.ok)
        self.assertEqual(applied.stage, "precheck")
        self.assertIsNone(destroyed)

    def test_destroy_runs_even_when_apply_fails(self):
        fake = FakeTerraform(apply_rc=1)
        runner = TerraformRunner(self.settings, executor=fake)
        self._write(runner, "s1")
        applied, destroyed = runner.apply_and_destroy("nios_dns_record_a", "s1")
        self.assertFalse(applied.ok)
        self.assertTrue(destroyed.ok)
        self.assertIn("destroy", [c[0] for c in fake.calls])


if __name__ == "__main__":
    unittest.main()
