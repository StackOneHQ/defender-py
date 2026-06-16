"""Tier 3 provider registry.

The defender package ships no Tier 3 implementations — proprietary model
endpoints (SageMaker, OpenAI, etc.) live in consumer code. Consumers call
``set_default_tier3_provider(provider)`` once at app startup; ``PromptDefense``
picks the registered provider up when callers opt in via ``enable_tier3=True``.

Module-level singleton because the defender is often instantiated per-request
and we don't want to pipe a provider object through that boundary on every call.
"""

from __future__ import annotations

from ..types import Tier3Provider

_default_provider: Tier3Provider | None = None


def set_default_tier3_provider(provider: Tier3Provider | None) -> None:
    """Register the process-wide default Tier 3 provider. Pass ``None`` to clear."""
    global _default_provider
    _default_provider = provider


def get_default_tier3_provider() -> Tier3Provider | None:
    """Return the registered default Tier 3 provider, or ``None`` if unset."""
    return _default_provider
