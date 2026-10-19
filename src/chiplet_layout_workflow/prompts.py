from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Any


_PROMPT_ASSET_PACKAGE = "chiplet_layout_workflow.prompt_assets"
_DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


class PromptError(ValueError):
    """Raised when a role prompt cannot be loaded or rendered."""


class PromptRenderer:
    _allowed_roles = frozenset({"reader", "retriever", "builder", "verifier", "version_controller"})

    def __init__(self, prompts_dir: Path | str | None = None) -> None:
        self._prompts_dir = None if prompts_dir is None else Path(prompts_dir)

    def render(self, role: str, context: dict[str, Any]) -> str:
        if role not in self._allowed_roles:
            raise PromptError(f"Unknown role '{role}'")
        if self._prompts_dir is None:
            template = self._load_packaged_prompt(role)
            if template is None:
                path = _DEFAULT_PROMPTS_DIR / f"{role}.md"
                template = self._read_prompt_file(path, role)
            return self._render_template(template, role, context)
        path = self._prompts_dir / f"{role}.md"
        return self._render_template(self._read_prompt_file(path, role), role, context)

    @staticmethod
    def _load_packaged_prompt(role: str) -> str | None:
        try:
            resource = importlib.resources.files(_PROMPT_ASSET_PACKAGE).joinpath(f"{role}.md")
            if resource.is_file():
                return resource.read_text(encoding="utf-8")
        except (ModuleNotFoundError, OSError):
            return None
        return None

    @staticmethod
    def _read_prompt_file(path: Path, role: str) -> str:
        if not path.is_file():
            raise PromptError(f"Missing prompt for role '{role}'")
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PromptError(f"Unable to render prompt for role '{role}'") from exc

    @staticmethod
    def _render_template(template: str, role: str, context: dict[str, Any]) -> str:
        try:
            return template.format(**context)
        except (KeyError, ValueError) as exc:
            raise PromptError(f"Unable to render prompt for role '{role}'") from exc
