"""
All settings come from environment variables.
Nothing secret lives in code, config files, prompts, or generated Terraform files.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    # Model
    ollama_url: str
    model: str
    embed_model: str
    num_ctx: int
    anthropic_api_key: str
    # Agent limits
    max_steps: int
    auto_approve_destructive: bool
    # Where everything the agent writes goes. The agent cannot write outside this folder.
    workspace: Path
    # Terraform / NIOS
    terraform_bin: str
    terraform_timeout: int
    provider_source: str
    provider_version: str
    nios_host_url: str
    nios_username: str
    nios_password: str

    @property
    def trace_dir(self) -> Path:
        return self.workspace / "traces"

    @property
    def memory_db(self) -> Path:
        return self.workspace / "memory.sqlite"

    @property
    def docs_index(self) -> Path:
        return self.workspace / "docs_index.json"

    @property
    def secrets(self) -> list[str]:
        return [self.nios_password]


def load_settings() -> Settings:
    return Settings(
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        model=os.getenv("AGENT_MODEL", "llama3.1:8b"),
        embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
        num_ctx=int(os.getenv("AGENT_NUM_CTX", "8192")),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        max_steps=int(os.getenv("AGENT_MAX_STEPS", "25")),
        auto_approve_destructive=_bool("AGENT_AUTO_APPROVE_DESTRUCTIVE", False),
        workspace=Path(os.getenv("AGENT_WORKSPACE", "./workspace")).resolve(),
        terraform_bin=os.getenv("TERRAFORM_BIN", "terraform"),
        terraform_timeout=int(os.getenv("TERRAFORM_TIMEOUT", "600")),
        provider_source=os.getenv("NIOS_PROVIDER_SOURCE", "infobloxopen/nios"),
        provider_version=os.getenv("NIOS_PROVIDER_VERSION", ""),
        nios_host_url=os.getenv("NIOS_HOST_URL", ""),
        nios_username=os.getenv("NIOS_USERNAME", ""),
        nios_password=os.getenv("NIOS_PASSWORD", ""),
    )
