from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .base import AgentProtocolError, BaseAgent, GitAdapter, RunContext


class VersionControllerAgent(BaseAgent):
    name = "version_controller"
    role = "version_controller"
    stage = "version_archiving"
    required_outputs = ("commit_manifest.json", "summary.md")

    def archive_request(
        self,
        context: RunContext,
        artifact_paths: Sequence[Path | str],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Pass only validated paths/metadata across the future Git boundary."""
        paths: list[Path] = []
        for raw_path in artifact_paths:
            try:
                path = context.run.assert_inside(raw_path)
            except ValueError as exc:
                raise AgentProtocolError(
                    "version controller accepts only artifacts in the current run"
                ) from exc
            paths.append(path)
        if context.git_adapter is None:
            return {"status": "deferred", "artifact_paths": [str(path) for path in paths]}
        return context.git_adapter.archive(tuple(paths), dict(metadata))

    def validate_inputs(self, context: RunContext) -> None:
        context.artifact_path("version_archiving", "commit_manifest.json")
