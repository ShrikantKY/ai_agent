"""
Composition root: the one place where concrete pieces are wired together.
The CLI and the eval runner both build agents through here, so they test the same thing.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass

from agent_core.llm import ClaudeClient, LLMClient, LLMError, OllamaClient
from agent_core.loop import Agent, RunResult
from agent_core.memory import Memory
from agent_core.policy import Approver, Policy, Redactor, console_approver
from agent_core.tools import Risk, ToolRegistry
from agent_core.tracing import Tracer
from config import Settings

from .docs import DocIndex
from .prompts import SYSTEM_PROMPT, build_task
from .terraform import TerraformRunner
from .toolset import ScenarioTracker, build_tools, make_done_check


@dataclass
class RunOptions:
    resource_type: str
    positive: int = 3
    negative: int = 1
    apply: bool = False


@dataclass
class RunReport:
    result: RunResult
    tracker: ScenarioTracker
    trace_path: str


def preflight(settings: Settings, need_nios: bool = True) -> list[str]:
    """Fail fast with clear messages instead of failing halfway through a run."""
    problems = []
    if not shutil.which(settings.terraform_bin):
        problems.append(f"Terraform not found ('{settings.terraform_bin}'). Install it or set TERRAFORM_BIN.")
    if settings.anthropic_api_key:
        # Claude — no local model check needed
        if not settings.anthropic_api_key.startswith("sk-ant-"):
            problems.append("ANTHROPIC_API_KEY looks invalid (should start with 'sk-ant-').")
    else:
        # Ollama — check server reachability and model availability
        try:
            ollama = OllamaClient(settings.ollama_url, settings.model, num_ctx=settings.num_ctx)
            models = ollama.available_models()
            if not any(m == settings.model or m.split(":")[0] == settings.model for m in models):
                problems.append(f"Model '{settings.model}' is not pulled. Run: ollama pull {settings.model}")
        except LLMError as e:
            problems.append(str(e))
    if need_nios:
        missing = [v for v, val in [("NIOS_HOST_URL", settings.nios_host_url),
                                    ("NIOS_USERNAME", settings.nios_username),
                                    ("NIOS_PASSWORD", settings.nios_password)] if not val]
        if missing:
            problems.append(f"Missing environment variables: {', '.join(missing)}")
    return problems


def open_docs(settings: Settings, llm: OllamaClient) -> DocIndex | None:
    if not settings.docs_index.exists():
        return None
    return DocIndex(settings.docs_index, lambda texts: llm.embed(texts, settings.embed_model), settings.embed_model)


def run_agent(settings: Settings, opts: RunOptions, *, llm: LLMClient | None = None,
              runner: TerraformRunner | None = None, docs: DocIndex | None = None,
              approver: Approver = console_approver, echo=print) -> RunReport:
    redact = Redactor(settings.secrets)
    tracer = Tracer(settings.trace_dir, redact=redact)
    memory = Memory(settings.memory_db)
    runner = runner or TerraformRunner(settings)
    if llm is None:
        llm, _ = _make_llm(settings)
        if isinstance(llm, OllamaClient):
            docs = docs or open_docs(settings, llm)

    tracker = ScenarioTracker(runner.resource_dir(opts.resource_type) / "results.json")
    tools = build_tools(opts.resource_type, runner, memory, docs, tracker, allow_apply=opts.apply)
    auto = {Risk.READ, Risk.WRITE} | ({Risk.DESTRUCTIVE} if settings.auto_approve_destructive else set())
    agent = Agent(llm, ToolRegistry(tools), Policy(approver, auto_approve=auto), tracer,
                  SYSTEM_PROMPT, max_steps=settings.max_steps, redact=redact, echo=echo)
    try:
        result = agent.run(build_task(opts.resource_type, opts.positive, opts.negative, opts.apply),
                           done_check=make_done_check(tracker, opts.positive, opts.negative, opts.apply))
    finally:
        memory.close()
    return RunReport(result, tracker, str(tracer.path))


def _make_llm(settings: Settings) -> tuple[LLMClient, str]:
    """Return (llm, provider_name). Prefers Claude when ANTHROPIC_API_KEY is set."""
    if settings.anthropic_api_key:
        return ClaudeClient(model=settings.model, api_key=settings.anthropic_api_key), "Claude"
    return OllamaClient(settings.ollama_url, settings.model, num_ctx=settings.num_ctx), "Ollama"
