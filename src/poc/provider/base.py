"""Provider-neutral request and response models."""

from __future__ import annotations

from pydantic import BaseModel


class LLMRequest(BaseModel):
    """One single-turn completion request."""

    model: str
    max_tokens: int
    temperature: float  # sent only to models that accept it
    prompt: str


class RawResponse(BaseModel):
    """A completion plus the provenance needed to audit it."""

    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_params: dict = {}  # kwargs actually sent, excluding the prompt
    raw: dict = {}  # verbatim provider response
