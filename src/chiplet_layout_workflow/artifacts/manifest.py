from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any

from ..contracts import RunManifest
from ..validation import validate_manifest_payload
from .run_store import DataRootMutationLock, Run, _is_link_or_reparse, _reject_link_components


class ManifestConflictError(FileExistsError):
    """Raised when a manifest would overwrite different existing content."""


class InvalidStatusTransition(ValueError):
    """Raised when a run status moves backward or sideways."""


class ManifestWriteError(OSError):
    """Raised when a manifest target cannot be safely published."""


_STATUSES = {"pending", "approved", "rejected", "superseded", "failed"}
_FORWARD_TRANSITIONS = {
    "pending": {"approved", "rejected", "failed", "superseded"},
    "approved": {"superseded"},
    "rejected": {"superseded"},
    "failed": {"superseded"},
    "superseded": set(),
}


def transition_status(current: str, new: str) -> str:
    if current not in _STATUSES or new not in _STATUSES:
        raise InvalidStatusTransition(f"Unknown status transition: {current!r} -> {new!r}")
    if new not in _FORWARD_TRANSITIONS[current]:
        raise InvalidStatusTransition(f"Invalid status transition: {current!r} -> {new!r}")
    return new


def _manifest_model(manifest: RunManifest | dict[str, Any]) -> RunManifest:
    return validate_manifest_payload(manifest)


def _content_checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _validate_manifest_target(run: Run, path: Path) -> Path:
    try:
        run_root = run.path.resolve(strict=True)
        _reject_link_components(run.path)
        target = run.assert_inside(path)
        _reject_link_components(target.parent)
        if os.path.lexists(target) and _is_link_or_reparse(target):
            raise ManifestWriteError(f"Manifest target is a link or reparse point: '{target}'")
        target_parent = target.parent.resolve(strict=True)
        target_parent.relative_to(run_root)
        target.relative_to(run_root)
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, ManifestWriteError):
            raise
        raise ManifestWriteError(f"Manifest target is not safely inside run '{run.path}'") from exc
    return target


def _read_existing_manifest(path: Path, expected_checksum: str) -> Path:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ManifestWriteError(f"Unable to read existing manifest '{path}'") from exc
    actual_checksum = _content_checksum(content)
    if actual_checksum != expected_checksum:
        raise ManifestConflictError(f"Refusing to overwrite different manifest: {path}")
    try:
        validate_manifest_payload(json.loads(content))
    except json.JSONDecodeError as exc:
        raise ManifestWriteError(f"Existing manifest is not valid JSON: {path}") from exc
    return path


def _sync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)


def _write_manifest_locked(run: Run, manifest: RunManifest | dict[str, Any]) -> Path:
    model = _manifest_model(manifest)
    if model.run_id != run.run_id:
        raise ValueError(f"manifest run_id {model.run_id!r} does not match run {run.run_id!r}")

    path = _validate_manifest_target(run, run.path / "manifest.json")
    content = (json.dumps(model.model_dump(mode="json", by_alias=True), indent=2, sort_keys=True) + "\n").encode("utf-8")
    expected_checksum = _content_checksum(content)
    if path.exists():
        return _read_existing_manifest(path, expected_checksum)

    parent = path.parent
    temporary = parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    file_descriptor = None
    publication_succeeded = False
    primary_error = False
    try:
        _validate_manifest_target(run, path)
        file_descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(file_descriptor, "wb") as stream:
            file_descriptor = None
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _validate_manifest_target(run, temporary)
        try:
            os.link(temporary, path)
        except FileExistsError:
            _validate_manifest_target(run, path)
            publication_succeeded = True
            return _read_existing_manifest(path, expected_checksum)
        publication_succeeded = True
        _validate_manifest_target(run, path)
        _sync_directory(parent)
        return _read_existing_manifest(path, expected_checksum)
    except ManifestConflictError:
        primary_error = True
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        primary_error = True
        if isinstance(exc, ManifestWriteError):
            raise
        raise ManifestWriteError(f"Unable to safely publish manifest '{path}'") from exc
    finally:
        cleanup_error = None
        if file_descriptor is not None:
            try:
                os.close(file_descriptor)
            except BaseException as exc:
                cleanup_error = exc
        try:
            _validate_manifest_target(run, temporary)
            temporary.unlink()
        except FileNotFoundError:
            pass
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
        if cleanup_error is not None and not publication_succeeded and not primary_error:
            raise ManifestWriteError(f"Unable to clean up temporary manifest '{temporary}'") from cleanup_error


def write_manifest(run: Run, manifest: RunManifest | dict[str, Any]) -> Path:
    data_root = run.data_root or run.path.parents[1]
    with DataRootMutationLock(data_root):
        return _write_manifest_locked(run, manifest)


def read_manifest(run: Run) -> RunManifest:
    """Read and validate the manifest belonging to ``run``."""

    path = _validate_manifest_target(run, run.path / "manifest.json")
    try:
        payload = json.loads(path.read_bytes())
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestWriteError(f"Unable to read manifest '{path}'") from exc
    model = validate_manifest_payload(payload)
    if model.run_id != run.run_id:
        raise ValueError(f"manifest run_id {model.run_id!r} does not match run {run.run_id!r}")
    return model
