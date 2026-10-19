from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LLMResponse:
    """Provider output metadata; it intentionally contains no prompts or secrets."""

    text: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    provider_request_id: str | None = None


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        artifacts: list[Path],
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a response from prompts and evidence artifacts."""


class ProviderConfigError(ValueError):
    """Raised when a provider cannot be configured safely."""


class ArtifactError(ValueError):
    """Raised when an evidence artifact cannot be safely sent to a provider."""
