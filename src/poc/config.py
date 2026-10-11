"""Benchmark execution parameters loaded from YAML; the prompt and validator are fixed in code."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class BenchmarkConfig(BaseModel):
    """Execution parameters for build/run/replay."""

    model: str = "claude-sonnet-5"
    max_tokens: int = 8192
    # No temperature: the target model rejects it.
    output_dir: Path = Path("benchmark_output")
    budget_usd: float = 5.0
    max_consecutive_failures: int = 3

    @classmethod
    def load(cls, path: str | Path | None) -> BenchmarkConfig:
        """Load the config from a YAML file, falling back to defaults if it does not exist.

        Args:
            path — YAML file path, or None for defaults.

        Returns:
            The validated configuration.
        """
        if path and Path(path).exists():
            data = yaml.safe_load(Path(path).read_text()) or {}
            return cls.model_validate(data)
        return cls()
