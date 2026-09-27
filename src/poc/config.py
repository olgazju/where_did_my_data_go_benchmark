"""Benchmark configuration (loaded from YAML, kept separate from implementation).

The official prompt (P2) and the official validator are fixed in code and are NOT
configurable. Only execution parameters live here.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class BenchmarkConfig(BaseModel):
    model: str = "claude-sonnet-5"
    max_tokens: int = 8192
    # temperature is intentionally omitted (the model rejects it; the provider omits it).
    k_general: int = 3
    k_calibration: int = 5
    output_dir: Path = Field(default=Path("benchmark_output"))
    budget_usd: float = 5.0
    max_consecutive_failures: int = 3

    @classmethod
    def load(cls, path: str | Path | None) -> "BenchmarkConfig":
        if path and Path(path).exists():
            data = yaml.safe_load(Path(path).read_text()) or {}
            return cls.model_validate(data)
        return cls()
