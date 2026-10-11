"""Anthropic implementation of the provider seam.

The answer is requested as JSON in the prompt and parsed by the independent evaluator.
``temperature`` is sent only to models that accept it; the exact kwargs sent are recorded.
"""

from __future__ import annotations

import anthropic

from .base import LLMRequest, RawResponse

# Model-id prefixes known to reject the `temperature` sampling parameter.
_NO_SAMPLING_PREFIXES = ("claude-opus-5", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8",
                         "claude-sonnet-5", "claude-sonnet-4-6", "claude-fable-5", "claude-mythos-5")


class PreflightError(RuntimeError):
    """The configured model cannot be used; raised before any tokens are spent."""


class AnthropicProvider:
    """Single-turn Messages API client for one model."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = anthropic.Anthropic()  # resolves credentials from the environment / ant profiles

    def preflight(self) -> None:
        """Check that the configured model id is usable with the resolved credentials.

        Raises:
            PreflightError: If the model cannot be retrieved (bad id, auth failure, ...).
        """
        try:
            self._client.models.retrieve(self.model)
        except Exception as e:  # noqa: BLE001 - surface any auth/404 clearly
            available = ""
            try:
                available = ", ".join(m.id for m in self._client.models.list().data)
            except Exception:  # noqa: BLE001
                pass
            raise PreflightError(
                f"model id {self.model!r} is not usable: {e}."
                + (f" Available: {available}" if available else "")
            ) from e

    def complete(self, req: LLMRequest) -> RawResponse:
        """Send one prompt and return the text answer with usage and provenance.

        Args:
            req — The request; ``temperature`` is dropped for models that reject it.

        Returns:
            The concatenated text blocks, token usage, the kwargs sent and the raw response.
        """
        kwargs: dict = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "messages": [{"role": "user", "content": req.prompt}],
        }
        if not req.model.startswith(_NO_SAMPLING_PREFIXES):
            kwargs["temperature"] = req.temperature

        raw = self._client.messages.create(**kwargs).model_dump()
        text = "".join(
            block.get("text", "") for block in raw.get("content", []) if block.get("type") == "text"
        )
        usage = raw.get("usage") or {}
        return RawResponse(
            text=text,
            model=raw.get("model", req.model),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            request_params={k: v for k, v in kwargs.items() if k != "messages"},
            raw=raw,
        )
