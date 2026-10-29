"""
Tools are the only way the agent can affect the world.

Each tool declares its parameters explicitly, so what the model sees is exactly what
the code enforces. Arguments are validated (and lightly coerced) before the tool runs,
which means tool code never has to defend against malformed model output.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class Risk(str, Enum):
    READ = "read"                # no side effects
    WRITE = "write"              # changes local files or memory; reversible
    DESTRUCTIVE = "destructive"  # touches real infrastructure


@dataclass
class ToolResult:
    ok: bool
    output: str
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, output: str, **data: Any) -> "ToolResult":
        return cls(True, output, data)

    @classmethod
    def failure(cls, output: str, **data: Any) -> "ToolResult":
        return cls(False, output, data)

    def for_model(self) -> str:
        return self.output if self.ok else f"ERROR: {self.output}"


class ToolArgumentError(ValueError):
    pass


_PY_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str, "integer": int, "number": (int, float),
    "boolean": bool, "object": dict, "array": list,
}


def _short(value: Any, limit: int = 80) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


@dataclass
class Param:
    name: str
    type: str  # JSON schema type: string | integer | number | boolean | object | array
    description: str
    required: bool = True
    enum: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.type not in _PY_TYPES:
            raise ValueError(f"unsupported parameter type {self.type!r}")

    def to_schema(self) -> dict:
        schema: dict[str, Any] = {"type": self.type, "description": self.description}
        if self.enum:
            schema["enum"] = self.enum
        return schema

    def coerce(self, value: Any) -> Any:
        """Accept common small-model mistakes, such as numbers or objects sent as strings."""
        if not isinstance(value, str):
            return value
        s = value.strip()
        if self.type == "integer" and s.lstrip("-").isdigit():
            return int(s)
        if self.type == "number":
            try:
                return float(s)
            except ValueError:
                return value
        if self.type == "boolean" and s.lower() in {"true", "false"}:
            return s.lower() == "true"
        if self.type in {"object", "array"}:
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return value
        return value

    def matches(self, value: Any) -> bool:
        # bool is a subclass of int in Python; don't let True pass as a number.
        if self.type in {"integer", "number"} and isinstance(value, bool):
            return False
        if not isinstance(value, _PY_TYPES[self.type]):
            return False
        return not self.enum or value in self.enum


@dataclass
class Tool:
    name: str
    description: str
    params: list[Param]
    fn: Callable[..., ToolResult]
    risk: Risk = Risk.READ

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {p.name: p.to_schema() for p in self.params},
                "required": [p.name for p in self.params if p.required],
            },
        }}

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        known = {p.name for p in self.params}
        problems: list[str] = []
        unknown = sorted(set(args) - known)
        if unknown:
            problems.append(f"unknown argument(s) {unknown}; valid arguments are {sorted(known)}")
        clean: dict[str, Any] = {}
        for p in self.params:
            if args.get(p.name) is None:
                if p.required:
                    problems.append(f"missing required argument '{p.name}' ({p.type})")
                continue
            value = p.coerce(args[p.name])
            if not p.matches(value):
                expected = f"one of {p.enum}" if p.enum else f"a {p.type}"
                problems.append(f"'{p.name}' must be {expected}, got {_short(value)}")
                continue
            clean[p.name] = value
        if problems:
            raise ToolArgumentError("; ".join(problems))
        return clean


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        names = [t.name for t in tools]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate tool names: {duplicates}")
        self._tools = {t.name: t for t in tools}

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)
