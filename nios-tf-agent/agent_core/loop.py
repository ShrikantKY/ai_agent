"""
The agent loop.

  1. Send the conversation and tool list to the model.
  2. If the model requests tools: validate arguments, check policy, run, feed results back.
  3. If the model stops requesting tools: ask CODE whether the task is really done.
     If not, tell the model what's missing and continue.
  4. Stop at the step limit no matter what.

The loop never crashes because of a tool or a bad model output. Every failure becomes
a message the model can read and react to.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from .llm import LLMClient, LLMError, ToolCall
from .policy import Policy
from .tools import ToolArgumentError, ToolRegistry, ToolResult
from .tracing import Tracer

DoneCheck = Callable[[], tuple[bool, str]]
COMPACTED = "[older tool output removed to save context; call the tool again if you need it]"


@dataclass
class RunResult:
    status: str  # "done" | "incomplete" | "max_steps" | "llm_error"
    summary: str
    steps: int
    tool_calls: int
    seconds: float


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} characters truncated] ...\n{text[-half:]}"


def _brief(args: dict[str, Any], limit: int = 120) -> str:
    text = json.dumps(args, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


class Agent:
    def __init__(self, llm: LLMClient, tools: ToolRegistry, policy: Policy, tracer: Tracer,
                 system_prompt: str, max_steps: int = 25, max_tool_output: int = 4000,
                 keep_recent_messages: int = 16, redact: Callable[[str], str] = lambda s: s,
                 echo: Callable[[str], None] | None = print):
        self.llm = llm
        self.tools = tools
        self.policy = policy
        self.tracer = tracer
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_tool_output = max_tool_output
        self.keep_recent_messages = keep_recent_messages
        self.redact = redact
        self.echo = echo

    # ── Public API ───────────────────────────────────────────────────────────
    def run(self, task: str, done_check: DoneCheck | None = None, max_nudges: int = 6) -> RunResult:
        started = time.monotonic()
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        self.tracer.log("run_start", task=task, tools=self.tools.names(), max_steps=self.max_steps)
        tool_calls = 0
        nudges = 0

        for step in range(1, self.max_steps + 1):
            self._compact(messages)
            try:
                response = self.llm.chat(messages, self.tools.schemas())
            except LLMError as e:
                self.tracer.log("llm_error", step=step, error=str(e))
                return self._finish("llm_error", str(e), step, tool_calls, started)

            self.tracer.log("model_turn", step=step, content=response.content,
                            tool_calls=[{"name": c.name, "arguments": c.arguments} for c in response.tool_calls])
            messages.append(self.llm.assistant_message(response))

            if not response.tool_calls:
                done, reason = done_check() if done_check else (True, "")
                if done:
                    return self._finish("done", response.content, step, tool_calls, started)
                if nudges >= max_nudges:
                    return self._finish("incomplete", f"model stopped but the task is not done: {reason}",
                                        step, tool_calls, started)
                nudges += 1
                self._say(f"   ↻ not done yet: {reason}")
                messages.append({"role": "user",
                                 "content": f"The task is not complete yet. {reason} You MUST call the tools — write_scenario and plan_scenario — do not write text describing what you would do. Call the tools now."})
                continue

            for call in response.tool_calls:
                tool_calls += 1
                output = self._execute(step, call)
                messages.append(self.llm.tool_message(call.name, output))

        done, reason = done_check() if done_check else (False, "")
        status = "done" if done else "max_steps"
        return self._finish(status, reason or "step limit reached", self.max_steps, tool_calls, started)

    # ── Internals ────────────────────────────────────────────────────────────
    def _execute(self, step: int, call: ToolCall) -> str:
        tool = self.tools.get(call.name)
        if tool is None:
            result = ToolResult.failure(
                f"unknown tool '{call.name}'. Available tools: {', '.join(self.tools.names())}")
        elif call.raw_arguments is not None:
            result = ToolResult.failure(
                f"arguments for {call.name} were not valid JSON: {call.raw_arguments[:200]}")
        else:
            result = self._validated_call(step, tool, call)

        text = _truncate(self.redact(result.for_model()), self.max_tool_output)
        self.tracer.log("tool_result", step=step, tool=call.name, ok=result.ok, output=text)
        if not result.ok:
            self._say(f"     ✗ {text.splitlines()[0][:140]}")
        return text

    def _validated_call(self, step: int, tool, call: ToolCall) -> ToolResult:
        try:
            args = tool.validate(call.arguments)
        except ToolArgumentError as e:
            return ToolResult.failure(f"invalid arguments for {tool.name}: {e}")

        denial = self.policy.check(tool, args)
        if denial:
            self.tracer.log("denied", step=step, tool=tool.name, reason=denial)
            return ToolResult.failure(f"DENIED: {denial}")

        self._say(f"   → {tool.name}({_brief(args)})")
        try:
            return tool.fn(**args)
        except Exception as e:  # a tool bug must never kill the run
            self.tracer.log("tool_exception", step=step, tool=tool.name, traceback=traceback.format_exc())
            return ToolResult.failure(f"{tool.name} crashed: {type(e).__name__}: {e}")

    def _compact(self, messages: list[dict]) -> None:
        """Replace old tool outputs with a placeholder so the context window doesn't overflow.
        Message structure is kept intact, because each tool message must follow its tool call."""
        cutoff = len(messages) - self.keep_recent_messages
        for msg in messages[2:max(cutoff, 2)]:
            if msg.get("role") == "tool" and msg.get("content") != COMPACTED:
                msg["content"] = COMPACTED

    def _finish(self, status: str, summary: str, steps: int, tool_calls: int, started: float) -> RunResult:
        result = RunResult(status, summary, steps, tool_calls, round(time.monotonic() - started, 1))
        self.tracer.log("run_end", status=status, summary=summary, steps=steps,
                        tool_calls=tool_calls, seconds=result.seconds)
        return result

    def _say(self, text: str) -> None:
        if self.echo:
            self.echo(text)
