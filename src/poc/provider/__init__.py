"""LLM provider seam. Only Anthropic is implemented; the interface keeps the pilot
runner and evaluator independent of any specific SDK."""

from .base import LLMProvider, LLMRequest, RawResponse

__all__ = ["LLMProvider", "LLMRequest", "RawResponse", "get_provider"]


def get_provider(name: str):
    """Return a provider instance by name. Only 'anthropic' exists for now."""
    if name == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider
    raise ValueError(f"unknown provider: {name!r} (only 'anthropic' is implemented)")
