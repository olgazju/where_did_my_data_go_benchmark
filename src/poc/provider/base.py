"""Minimal provider interface (Ref. #8)."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class LLMRequest(BaseModel):
    model: str
    max_tokens: int
    temperature: float
    prompt: str


class RawResponse(BaseModel):
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_params: dict = {}  # the EXACT kwargs sent to the provider (for provenance)
    raw: dict = {}  # verbatim provider response


class LLMProvider(Protocol):
    def preflight(self) -> None:
        """Validate credentials, model id, and the output mechanism BEFORE spending tokens."""
        ...

    def complete(self, req: LLMRequest) -> RawResponse:
        ...
