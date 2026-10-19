from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .artifacts.checksums import checksum_tree
from .artifacts.run_store import (
    DataRootMutationLock,
    PathBoundaryError,
    Run,
    _reject_link_components,
)


CHECKPOINT_STAGES = (
    "reading",
    "feasibility_review",
    "building",
    "verification",
    "version_archiving",
)
_DECISIONS = {"approve", "approved", "revise", "reject"}
_REQUIRED_FIELDS = {
    "stage",
    "decision",
    "artifact_checksums",
    "timestamp",
    "operator_note",
}


class CheckpointIntegrityError(ValueError):
    """Raised when a checkpoint is missing, malformed, or no longer true."""


@dataclass(frozen=True)
class Checkpoint:
    stage: str
    decision: str
    artifact_checksums: dict[str, str]
    timestamp: str
    operator_note: str


def _normalize_stage(stage: str) -> str:
    if stage not in CHECKPOINT_STAGES:
        raise CheckpointIntegrityError(f"unknown checkpoint stage: {stage!r}")
    return stage


def _normalize_decision(decision: str) -> str:
    if decision not in _DECISIONS:
        raise CheckpointIntegrityError(f"unknown checkpoint decision: {decision!r}")
    return "approve" if decision == "approved" else decision


def _validate_checksums(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise CheckpointIntegrityError("artifact_checksums must be an object")
    result: dict[str, str] = {}
    for relative, checksum in value.items():
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise CheckpointIntegrityError("artifact checksum paths must be relative")
        parts = Path(relative).parts
        if ".." in parts or not isinstance(checksum, str) or not checksum:
            raise CheckpointIntegrityError("artifact checksum records are invalid")
        result[Path(relative).as_posix()] = checksum
    return dict(sorted(result.items()))


def _validate_timestamp(timestamp: str) -> str:
    if not isinstance(timestamp, str) or not timestamp:
        raise CheckpointIntegrityError("checkpoint timestamp is required")
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise CheckpointIntegrityError("checkpoint timestamp is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise CheckpointIntegrityError("checkpoint timestamp must include a timezone")
    return timestamp


def _checkpoint_directory(run_or_directory: Run | Path | str) -> tuple[Path, Path]:
    if isinstance(run_or_directory, Run):
        directory = run_or_directory.checkpoints
        try:
            run_or_directory.assert_inside(directory)
        except PathBoundaryError:
            raise
        return run_or_directory.path, directory

    candidate = Path(run_or_directory)
    if candidate.name == "checkpoints":
        return candidate.parent, candidate
    return candidate, candidate / "checkpoints"


def checkpoint_path(run_or_directory: Run | Path | str, stage: str) -> Path:
    stage = _normalize_stage(stage)
    _, directory = _checkpoint_directory(run_or_directory)
    return directory / f"{stage}.json"


def write_checkpoint(
    run_or_directory: Run | Path | str,
    stage: str,
    decision: str,
    artifact_checksums: Mapping[str, str],
    *,
    timestamp: str | None = None,
    operator_note: str = "",
) -> Path:
    stage = _normalize_stage(stage)
    decision = _normalize_decision(decision)
    checksums = _validate_checksums(artifact_checksums)
    if not isinstance(operator_note, str):
        raise CheckpointIntegrityError("operator_note must be a string")
    timestamp = _validate_timestamp(
        timestamp or datetime.now(timezone.utc).isoformat()
    )
    payload = {
        "artifact_checksums": checksums,
        "decision": decision,
        "operator_note": operator_note,
        "stage": stage,
        "timestamp": timestamp,
    }
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")

    if isinstance(run_or_directory, Run):
        data_root = run_or_directory.data_root or run_or_directory.path.parents[1]
        with DataRootMutationLock(data_root):
            return _write_checkpoint_file(run_or_directory, stage, content)
    return _write_checkpoint_file(run_or_directory, stage, content)


def _write_checkpoint_file(
    run_or_directory: Run | Path | str,
    stage: str,
    content: bytes,
) -> Path:
    """Publish one checkpoint after all path checks for the selected API.

    The mutation lock coordinates workflow-owned Run mutations. A caller that
    changes the filesystem outside that lock is outside this coordination
    boundary; validation and os.replace reduce the race window but do not make
    ordinary path operations absolutely atomic.
    """

    _, directory = _checkpoint_directory(run_or_directory)
    target = checkpoint_path(run_or_directory, stage)
    if isinstance(run_or_directory, Run):
        _validate_run_checkpoint_path(run_or_directory, directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CheckpointIntegrityError(f"unable to write checkpoint under {directory}") from exc
    if isinstance(run_or_directory, Run):
        _validate_run_checkpoint_path(run_or_directory, directory)

    descriptor: int | None = None
    temporary_name: str | None = None
    primary_error: BaseException | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{stage}.", suffix=".tmp", dir=directory
        )
        temporary = Path(temporary_name)
        if isinstance(run_or_directory, Run):
            _validate_run_checkpoint_path(run_or_directory, temporary)
            _validate_run_checkpoint_path(run_or_directory, target)
        elif temporary.parent.resolve() != directory.resolve():
            raise PathBoundaryError("checkpoint temporary file escaped checkpoint directory")
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if isinstance(run_or_directory, Run):
            _validate_run_checkpoint_path(run_or_directory, temporary)
            _validate_run_checkpoint_path(run_or_directory, target)
        os.replace(temporary, target)
        if isinstance(run_or_directory, Run):
            _validate_run_checkpoint_path(run_or_directory, target)
            _validate_run_checkpoint_path(run_or_directory, target.parent)
        _sync_directory(directory)
    except (OSError, RuntimeError) as exc:
        primary_error = CheckpointIntegrityError(
            f"unable to write checkpoint under {directory}"
        )
        raise primary_error from exc
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_error: OSError | None = None
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                cleanup_error = exc
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if cleanup_error is not None:
            if primary_error is not None:
                primary_error.add_note(
                    f"checkpoint temporary cleanup failed for '{temporary_name}': "
                    f"{cleanup_error}"
                )
            else:
                raise CheckpointIntegrityError(
                    f"unable to clean up temporary checkpoint '{temporary_name}'"
                ) from cleanup_error
    return target


def read_checkpoint(path: Path | str) -> Checkpoint:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError(f"invalid checkpoint JSON: {path}") from exc
    if not isinstance(payload, dict) or set(payload) != _REQUIRED_FIELDS:
        raise CheckpointIntegrityError(f"checkpoint fields are invalid: {path}")
    try:
        stage = _normalize_stage(payload["stage"])
        decision = _normalize_decision(payload["decision"])
        checksums = _validate_checksums(payload["artifact_checksums"])
        timestamp = _validate_timestamp(payload["timestamp"])
        operator_note = payload["operator_note"]
        if not isinstance(operator_note, str):
            raise CheckpointIntegrityError("operator_note must be a string")
    except (KeyError, TypeError) as exc:
        raise CheckpointIntegrityError(f"checkpoint fields are invalid: {path}") from exc
    if path.suffix == ".json" and path.stem != stage:
        raise CheckpointIntegrityError(f"checkpoint filename does not match stage: {path}")
    return Checkpoint(stage, decision, checksums, timestamp, operator_note)


def verify_checkpoint_artifacts(checkpoint: Checkpoint, artifact_root: Path) -> None:
    if not artifact_root.exists() and checkpoint.artifact_checksums:
        raise CheckpointIntegrityError(
            f"artifact tree for checkpoint stage '{checkpoint.stage}' is missing"
        )
    actual = checksum_tree(artifact_root) if artifact_root.exists() else {}
    if actual != checkpoint.artifact_checksums:
        raise CheckpointIntegrityError(
            f"artifact checksums do not match checkpoint for stage '{checkpoint.stage}'"
        )


def read_all_checkpoints(run_or_directory: Run | Path | str) -> list[Checkpoint]:
    _, directory = _checkpoint_directory(run_or_directory)
    if isinstance(run_or_directory, Run):
        _validate_run_checkpoint_path(run_or_directory, directory)
    if not directory.exists():
        return []
    try:
        paths = sorted(directory.glob("*.json"))
    except OSError as exc:
        raise CheckpointIntegrityError(f"unable to inspect checkpoint directory: {directory}") from exc
    if isinstance(run_or_directory, Run):
        paths = [_validate_run_checkpoint_path(run_or_directory, path) for path in paths]
    return [read_checkpoint(path) for path in paths]


def _validate_run_checkpoint_path(run: Run, path: Path) -> Path:
    try:
        run.assert_inside(path)
        _reject_link_components(path)
    except PathBoundaryError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise PathBoundaryError(
            f"Checkpoint path must remain a non-link path inside run directory '{run.path}'"
        ) from exc
    return path


def _sync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
