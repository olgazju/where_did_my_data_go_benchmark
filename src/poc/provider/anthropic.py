"""Anthropic implementation of the provider seam.

Output mechanism: we request a small JSON object via the prompt and parse it in the
(independent) evaluator. We deliberately do NOT rely on a beta structured-outputs API
so the code stays compatible with the installed SDK; the evaluator re-validates every
response regardless.

Preflight (Ref. #7) validates, before spending any tokens:
  * credentials are resolvable,
  * the configured model id actually exists in the account's model list, and
  * the `messages.create` call path exists in the installed SDK.

Note on sampling: modern Claude models (Opus/Sonnet 5.x, 4.6+) reject `temperature`.
We therefore only send `temperature` for models that accept it (e.g. Haiku), and record
the exact kwargs that were sent for provenance.
"""

from __future__ import annotations

from .base import LLMRequest, RawResponse

# Model-id prefixes known to reject the `temperature` sampling parameter.
_NO_SAMPLING_PREFIXES = ("claude-opus-5", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8",
                         "claude-sonnet-5", "claude-sonnet-4-6", "claude-fable-5", "claude-mythos-5")


class PreflightError(RuntimeError):
    pass


class AnthropicProvider:
    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None

    def _client_or_create(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise PreflightError("the 'anthropic' package is not installed") from e
            # The SDK resolves credentials from the environment / ant profiles itself.
            self._client = anthropic.Anthropic()
        return self._client

    def preflight(self) -> None:
        client = self._client_or_create()
        if not hasattr(client, "messages") or not hasattr(client.messages, "create"):
            raise PreflightError("installed anthropic SDK has no messages.create call path")
        # Validate the model id against the live model list — never assume it is valid.
        try:
            client.models.retrieve(self.model)
        except Exception as e:  # noqa: BLE001 - surface any auth/404 clearly
            available = ""
            try:
                available = ", ".join(m.id for m in client.models.list().data)
            except Exception:  # noqa: BLE001
                pass
            raise PreflightError(
                f"model id {self.model!r} is not usable: {e}."
                + (f" Available: {available}" if available else "")
            ) from e

    def _sends_temperature(self) -> bool:
        return not any(self.model.startswith(p) for p in _NO_SAMPLING_PREFIXES)

    def complete(self, req: LLMRequest) -> RawResponse:
        client = self._client_or_create()
        kwargs: dict = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "messages": [{"role": "user", "content": req.prompt}],
        }
        if self._sends_temperature():
            kwargs["temperature"] = req.temperature

        msg = client.messages.create(**kwargs)
        raw = msg.model_dump()
        text = "".join(
            block.get("text", "") for block in raw.get("content", []) if block.get("type") == "text"
        )
        usage = raw.get("usage") or {}
        # Persist exactly what was sent (minus the full prompt, kept alongside the task).
        params = {k: v for k, v in kwargs.items() if k != "messages"}
        return RawResponse(
            text=text,
            model=raw.get("model", req.model),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            request_params=params,
            raw=raw,
        )
