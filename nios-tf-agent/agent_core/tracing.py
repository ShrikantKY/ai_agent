"""
Every run writes a trace: one JSON object per line for each model turn and tool call.
When the agent does something strange, the trace tells you exactly why.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


class Tracer:
    def __init__(self, trace_dir: Path, redact: Callable[[str], str] = lambda s: s,
                 run_id: str | None = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.redact = redact
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = trace_dir / f"{self.run_id}.jsonl"

    def log(self, event: str, **data: Any) -> None:
        record = {"ts": round(time.time(), 3), "run_id": self.run_id, "event": event, **data}
        line = self.redact(json.dumps(record, default=str, ensure_ascii=False))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


class NullTracer(Tracer):
    """For tests and dry runs."""

    def __init__(self) -> None:  # noqa: D401 - intentionally skips parent init
        self.run_id = "null"
        self.events: list[dict[str, Any]] = []

    def log(self, event: str, **data: Any) -> None:
        self.events.append({"event": event, **data})
