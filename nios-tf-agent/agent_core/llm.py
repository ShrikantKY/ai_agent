"""
Model-agnostic LLM interface.

The agent loop only talks to `LLMClient`. Swapping Ollama for another provider means
writing one new adapter class with the same four methods. Nothing else changes.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw_arguments: str | None = None  # set only when the model sent arguments that weren't valid JSON


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMError(RuntimeError):
    """The model could not be reached or returned something unusable."""


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...
    def assistant_message(self, response: LLMResponse) -> dict: ...
    def tool_message(self, tool_name: str, content: str) -> dict: ...


class OllamaClient:
    def __init__(self, url: str, model: str, num_ctx: int = 8192, temperature: float = 0.0,
                 timeout: int = 300, retries: int = 2):
        self.url = url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries

    # ── Chat ────────────────────────────────────────────────────────────────
    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            # Model parameters must go under "options"; top-level keys are ignored.
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        body = self._post("/api/chat", payload)
        return self.parse(body)

    @staticmethod
    def parse(body: dict) -> LLMResponse:
        msg = body.get("message") or {}
        calls: list[ToolCall] = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function") or {}
            args: Any = fn.get("arguments", {})
            raw = None
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    raw, args = args, {}
            if not isinstance(args, dict):
                raw, args = json.dumps(args), {}
            calls.append(ToolCall(name=fn.get("name", ""), arguments=args, raw_arguments=raw))
        return LLMResponse(content=msg.get("content") or "", tool_calls=calls)

    def assistant_message(self, response: LLMResponse) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": response.content}
        if response.tool_calls:
            msg["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments}}
                                 for c in response.tool_calls]
        return msg

    def tool_message(self, tool_name: str, content: str) -> dict:
        return {"role": "tool", "tool_name": tool_name, "content": content}

    # ── Embeddings (used by document search) ────────────────────────────────
    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        body = self._post("/api/embed", {"model": model, "input": texts})
        vectors = body.get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise LLMError(f"embedding model {model!r} returned {len(vectors or [])} vectors for {len(texts)} inputs")
        return vectors

    # ── Health check ────────────────────────────────────────────────────────
    def available_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=5)
            r.raise_for_status()
        except requests.RequestException as e:
            raise LLMError(f"cannot reach Ollama at {self.url}. Is 'ollama serve' running?") from e
        return [m.get("name", "") for m in r.json().get("models", [])]

    # ── HTTP with retries for transient failures only ───────────────────────
    def _post(self, path: str, payload: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(f"{self.url}{path}", json=payload, timeout=self.timeout)
                if 400 <= r.status_code < 500:
                    # Client errors (bad model name, bad request) won't fix themselves.
                    raise LLMError(f"Ollama {path} returned {r.status_code}: {r.text[:300]}")
                r.raise_for_status()
                return r.json()
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                last_error = e
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
        raise LLMError(f"Ollama {path} failed after {self.retries + 1} attempts: {last_error}")


class ClaudeClient:
    """Anthropic Claude adapter. Drop-in replacement for OllamaClient."""

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise LLMError("anthropic package not installed. Run: pip install anthropic")
        self.model = model
        self._anthropic = _anthropic
        self._client = _anthropic.Anthropic(api_key=api_key)

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        filtered = [m for m in messages if m["role"] != "system"]
        # Convert tool results to Anthropic format
        converted = []
        for m in filtered:
            if m["role"] == "tool":
                converted.append({
                    "role": "user",
                    "content": [{"type": "tool_result",
                                 "tool_use_id": m.get("tool_use_id", "call_0"),
                                 "content": m["content"]}]
                })
            else:
                converted.append(m)
        claude_tools = [
            {"name": t["function"]["name"],
             "description": t["function"]["description"],
             "input_schema": t["function"]["parameters"]}
            for t in tools
        ] if tools else []
        try:
            kwargs: dict[str, Any] = dict(
                model=self.model, max_tokens=4096,
                messages=converted,
            )
            if system:
                kwargs["system"] = system
            if claude_tools:
                kwargs["tools"] = claude_tools
            response = self._client.messages.create(**kwargs)
        except self._anthropic.APIError as e:
            raise LLMError(f"Claude API error: {e}")

        calls = [ToolCall(name=b.name, arguments=dict(b.input))
                 for b in response.content if b.type == "tool_use"]
        text = next((b.text for b in response.content if b.type == "text"), "")
        return LLMResponse(content=text, tool_calls=calls)

    def assistant_message(self, response: LLMResponse) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": response.content}
        if response.tool_calls:
            msg["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments}}
                                 for c in response.tool_calls]
        return msg

    def tool_message(self, tool_name: str, content: str) -> dict:
        return {"role": "tool", "tool_name": tool_name, "content": content}

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        raise LLMError("Claude does not support embeddings. Use OllamaClient for doc indexing.")
