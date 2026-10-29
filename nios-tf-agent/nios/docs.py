"""
Document search (RAG), exposed to the agent as a tool it calls when it decides to.

Index plain-text, Markdown, or PDF files directly:
    python cli.py index-docs docs/wapi_reference.pdf

A JSON file with linear cosine search is fine up to tens of thousands of chunks.
Move to a vector database only when you measure that search is slow.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

EmbedFn = Callable[[list[str]], list[list[float]]]


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("pypdf is required to read PDF files. Run: pip install pypdf>=4.0")
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    """Split on paragraph boundaries into chunks of roughly `size` characters, with overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for p in paragraphs:
        if len(p) > size:  # a single huge paragraph: flush, then hard-split it
            if current:
                chunks.append(current)
                current = ""
            while len(p) > size:
                chunks.append(p[:size])
                p = p[size - overlap:]
        if current and len(current) + len(p) + 2 > size:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current:
        chunks.append(current)
    return chunks


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-10)


@dataclass
class Hit:
    score: float
    source: str
    text: str


class DocIndex:
    def __init__(self, path: Path, embed: EmbedFn, model_name: str):
        self.path = path
        self.embed = embed
        self.model_name = model_name
        self.chunks: list[dict] = []
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("model") != model_name:
                raise ValueError(f"{path} was built with {data.get('model')!r}, not {model_name!r}. Rebuild it.")
            self.chunks = data["chunks"]

    @property
    def ready(self) -> bool:
        return bool(self.chunks)

    def build(self, files: list[Path], batch_size: int = 16, echo: Callable[[str], None] = print) -> int:
        pending: list[tuple[str, str]] = []
        for f in files:
            for chunk in chunk_text(_read_file(f)):
                pending.append((f.name, chunk))
        chunks = []
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            vectors = self.embed([text for _, text in batch])
            chunks.extend({"source": src, "text": text, "vec": vec} for (src, text), vec in zip(batch, vectors))
            echo(f"  embedded {min(i + batch_size, len(pending))}/{len(pending)} chunks")
        self.chunks = chunks
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"model": self.model_name, "chunks": chunks}), encoding="utf-8")
        return len(chunks)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if not self.chunks:
            return []
        qvec = self.embed([query])[0]
        hits = [Hit(_cosine(qvec, c["vec"]), c["source"], c["text"]) for c in self.chunks]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
