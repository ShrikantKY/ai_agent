"""
Guardrails live in code, not in the prompt.

The model can ask for anything. The policy decides what actually runs. A prompt that
says "never destroy production" is a suggestion; a check in code is a rule.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from .tools import Risk, Tool

Approver = Callable[[str, dict[str, Any]], bool]


def deny_all(tool_name: str, args: dict[str, Any]) -> bool:
    return False


def console_approver(tool_name: str, args: dict[str, Any]) -> bool:
    print(f"\n⚠️  The agent wants to run {tool_name} with {args}")
    try:
        return input("   Allow? [y/N] ").strip().lower() == "y"
    except EOFError:  # non-interactive session: deny
        return False


class Policy:
    def __init__(self, approver: Approver = deny_all,
                 auto_approve: Iterable[Risk] = (Risk.READ, Risk.WRITE),
                 blocked_tools: Iterable[str] = ()):
        self.approver = approver
        self.auto_approve = set(auto_approve)
        self.blocked_tools = set(blocked_tools)

    def check(self, tool: Tool, args: dict[str, Any]) -> str | None:
        """Return None if the call may run, or a reason it may not."""
        if tool.name in self.blocked_tools:
            return f"tool '{tool.name}' is disabled for this run. Do not call it again."
        if tool.risk in self.auto_approve:
            return None
        if self.approver(tool.name, args):
            return None
        return f"the user did not approve {tool.name}. Do not retry it; continue without it or finish."


def safe_path(root: Path, *parts: str) -> Path:
    """Resolve a path under root, refusing anything that escapes it (../, absolute paths, symlinks)."""
    root = root.resolve()
    target = root.joinpath(*parts).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path escapes the workspace: {'/'.join(parts)}")
    return target


class Redactor:
    """Replaces known secret values in any text before it reaches the model or a log."""

    def __init__(self, secrets: Iterable[str]):
        self._secrets = sorted({s for s in secrets if s and len(s) >= 4}, key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "***")
        return text
