from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import LLMProvider, LLMResponse


class MockProvider(LLMProvider):
    """Deterministic provider for tests and offline workflow runs."""

    def __init__(self, payload: Any, *, model: str = "mock") -> None:
        self._payload = payload
        self._model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        artifacts: list[Path],
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        del system_prompt, user_prompt, artifacts, response_schema
        text = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return LLMResponse(
            text=text,
            model=self._model,
            usage={},
            provider_request_id="mock-request",
        )
