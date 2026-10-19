from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .base import ArtifactError, LLMProvider, LLMResponse, ProviderConfigError

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only without the optional dependency
    OpenAI = None  # type: ignore[assignment,misc]


class OpenAIProvider(LLMProvider):
    """Thin, deliberately quiet adapter around the OpenAI chat API."""

    max_artifact_bytes = 64 * 1024
    max_total_artifact_bytes = 1024 * 1024

    def __init__(
        self,
        *,
        api_key_env_var: str = "OPENAI_API_KEY",
        api_key_env: str | None = None,
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
    ) -> None:
        configured_env = api_key_env or api_key_env_var
        api_key = os.environ.get(configured_env)
        if not api_key:
            raise ProviderConfigError(
                f"API key environment variable '{configured_env}' is not set"
            )
        if OpenAI is None:
            raise ProviderConfigError("The optional 'openai' package is not installed")
        self._model = model
        self._client = OpenAI(api_key=api_key, timeout=timeout)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        artifacts: list[Path],
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        artifact_text = _serialize_artifacts(
            artifacts,
            self.max_artifact_bytes,
            self.max_total_artifact_bytes,
        )
        outgoing_user_prompt = user_prompt
        if artifact_text:
            outgoing_user_prompt = f"{user_prompt}\n\n{artifact_text}"
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": outgoing_user_prompt},
            ],
        }
        if response_schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _response_schema_name(response_schema),
                    "strict": True,
                    "schema": response_schema,
                },
            }

        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            # Never include request data, headers, or exception text that may echo them.
            raise RuntimeError("OpenAI provider request failed") from None

        choice = response.choices[0]
        usage = _usage_dict(getattr(response, "usage", None))
        return LLMResponse(
            text=choice.message.content or "",
            model=getattr(response, "model", self._model),
            usage=usage,
            provider_request_id=getattr(response, "id", None),
        )

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> "OpenAIProvider":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def _serialize_artifacts(
    artifacts: list[Path], max_bytes: int, max_total_bytes: int | None = None
) -> str:
    sections = []
    total_bytes = 0
    for path in artifacts:
        try:
            size = path.stat().st_size
            if size > max_bytes:
                raise ArtifactError(
                    f"Artifact '{path}' is too large; maximum is {max_bytes} bytes"
                )
            data = path.read_bytes()
            text = data.decode("utf-8")
        except ArtifactError:
            raise
        except UnicodeDecodeError as exc:
            raise ArtifactError(f"Artifact '{path}' is not a UTF-8 text file") from exc
        except OSError as exc:
            raise ArtifactError(f"Unable to read artifact '{path}'") from exc
        if "\x00" in text:
            raise ArtifactError(f"Artifact '{path}' is binary or contains unsupported data")
        total_bytes += len(data)
        if max_total_bytes is not None and total_bytes > max_total_bytes:
            raise ArtifactError(
                f"combined artifacts are too large; maximum is {max_total_bytes} bytes"
            )
        sections.append(f"--- ARTIFACT: {path} ---\n{text}\n--- END ARTIFACT: {path} ---")
    return "\n\n".join(sections)


def _response_schema_name(schema: dict[str, Any]) -> str:
    title = schema.get("title")
    if not isinstance(title, str):
        return "workflow_response"
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", title)[:64]
    return safe_name or "workflow_response"


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return dict(usage.model_dump())
    if isinstance(usage, dict):
        return dict(usage)
    return {
        key: getattr(usage, key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if hasattr(usage, key)
    }
