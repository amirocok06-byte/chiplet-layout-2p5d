from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from .contracts import RunConfig
from .validation import PayloadSchemaError, validate_run_payload


class ConfigError(ValueError):
    """Raised when a YAML configuration cannot be read or validated."""


def load_config(path: Path) -> RunConfig:
    try:
        with path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
    except OSError as exc:
        raise ConfigError(f"Unable to read config file '{path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in config file '{path}': {exc}") from exc

    try:
        return validate_run_payload(raw)
    except (PayloadSchemaError, ValidationError) as exc:
        if isinstance(exc, PayloadSchemaError):
            field = "config"
            message = str(exc)
        else:
            first = exc.errors()[0]
            field = ".".join(str(part) for part in first.get("loc", ())) or "config"
            message = first.get("msg", "invalid value")
        raise ConfigError(f"Invalid config field '{field}': {message}") from exc
