"""Stand-ins for the model and Terraform, so the whole agent can be tested offline."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from agent_core.llm import LLMResponse, OllamaClient, ToolCall
from config import Settings

SAMPLE_SCHEMA = {"provider_schemas": {"registry.terraform.io/infobloxopen/nios": {"resource_schemas": {
    "nios_dns_record_a": {"block": {"attributes": {
        "id": {"type": "string", "computed": True},
        "ref": {"type": "string", "computed": True},
        "name": {"type": "string", "required": True, "description": "FQDN of the record."},
        "ipv4addr": {"type": "string", "optional": True, "description": "IPv4 address."},
        "ttl": {"type": "number", "optional": True, "description": "Time to live."},
        "use_ttl": {"type": "bool", "optional": True},
        "view": {"type": "string", "optional": True, "computed": True},
        "comment": {"type": "string", "optional": True},
        "extattrs": {"type": ["map", "string"], "optional": True},
        "cloud_info": {"nested_type": {"attributes": {}, "nesting_mode": "single"}, "computed": True},
    }}},
}}}}

TF_ERROR_OUTPUT = """\
Planning failed. Terraform encountered an error while generating this plan.

╷
│ Error: Invalid Attribute Value
│
│   with nios_dns_record_a.under_test,
│   on main.tf line 4, in resource "nios_dns_record_a" "under_test":
│    4:   ttl      = var.ttl
│
│ Attribute ttl value must be between 0 and 2147483647, got: -1
╵
"""


def call(tool_name: str, /, **arguments) -> ToolCall:
    return ToolCall(tool_name, arguments)


def turn(*calls: ToolCall, content: str = "") -> LLMResponse:
    return LLMResponse(content, list(calls))


class ScriptedLLM:
    """Plays back a fixed list of responses and records everything the agent sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen: list[list[dict]] = []
        self._fmt = OllamaClient("http://unused", "unused")

    def chat(self, messages, tools):
        self.seen.append(copy.deepcopy(messages))
        if not self.responses:
            return LLMResponse("I am finished.")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def assistant_message(self, response):
        return self._fmt.assistant_message(response)

    def tool_message(self, tool_name, content):
        return self._fmt.tool_message(tool_name, content)


class FakeTerraform:
    """Pretends to be the terraform binary. plan_results maps scenario -> (rc, output) or a list of them."""

    def __init__(self, schema=SAMPLE_SCHEMA, plan_results=None, apply_rc=0, destroy_rc=0, state=""):
        self.schema_json = json.dumps(schema)
        self.plan_results = plan_results or {}
        self.apply_rc = apply_rc
        self.destroy_rc = destroy_rc
        self.state = state
        self.calls: list[list[str]] = []

    def __call__(self, argv, cwd: Path, env, timeout):
        self.calls.append(argv[1:])
        cmd = argv[1]
        if cmd == "init":
            (cwd / ".terraform").mkdir(exist_ok=True)
            return 0, "Terraform has been successfully initialized!"
        if cmd == "providers":
            return 0, self.schema_json
        if cmd == "state":
            return 0, self.state
        if cmd == "plan":
            var_file = next(a for a in argv if a.startswith("-var-file="))
            scenario = Path(var_file.split("=", 1)[1]).name.replace(".tfvars.json", "")
            result = self.plan_results.get(scenario, (0, "Plan: 1 to add, 0 to change, 0 to destroy."))
            if isinstance(result, list):
                result = result.pop(0) if len(result) > 1 else result[0]
            return result
        if cmd == "apply":
            return self.apply_rc, "Apply complete!" if self.apply_rc == 0 else "Error: apply broke\n\ndetails"
        if cmd == "destroy":
            return self.destroy_rc, "Destroy complete!" if self.destroy_rc == 0 else "Error: destroy broke\n\ndetails"
        return 1, f"Error: unexpected command {argv}"


def make_settings(workspace: Path, **overrides) -> Settings:
    base = dict(
        ollama_url="http://unused", model="fake", embed_model="fake-embed", num_ctx=4096,
        max_steps=20, auto_approve_destructive=False, workspace=workspace,
        terraform_bin="terraform", terraform_timeout=30, provider_source="infobloxopen/nios",
        provider_version="", nios_host_url="https://grid.test", nios_username="admin",
        nios_password="s3cret-password",
    )
    base.update(overrides)
    return Settings(**base)
