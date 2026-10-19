from __future__ import annotations

import re
import secrets
import unicodedata
import os
import stat
import threading
from contextlib import AbstractContextManager
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from ..research_target import ResearchTarget


class PathBoundaryError(ValueError):
    """Raised when an artifact path escapes its run directory."""


def _is_link_or_reparse(path: Path) -> bool:
    if os.path.islink(path):
        return True
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except FileNotFoundError:
        return False
    except AttributeError:
        raise OSError(f"Cannot inspect link or reparse state for '{path}'")
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _reject_link_components(path: Path) -> None:
    current = Path(path.anchor) if path.anchor else Path()
    for part in path.parts[1:] if path.anchor else path.parts:
        current /= part
        if os.path.lexists(current):
            try:
                is_link = _is_link_or_reparse(current)
            except OSError as exc:
                raise PathBoundaryError(f"Cannot inspect path component '{current}'") from exc
            if is_link:
                raise PathBoundaryError(f"Link or reparse component is not allowed: '{current}'")


def _canonical_path(path: Path) -> Path:
    return path.expanduser().absolute()


class DataRootMutationLock(AbstractContextManager["DataRootMutationLock"]):
    """Serialize workflow mutations; the configured data root is workflow-owned.

    External mutation of the data root that bypasses this lock is unsupported.
    """

    _FILENAME = ".chiplet-layout-workflow.lock"
    _THREAD_LOCKS: dict[str, threading.RLock] = {}
    _THREAD_LOCKS_GUARD = threading.Lock()

    def __init__(self, data_root: Path):
        self.data_root = _canonical_path(data_root)
        self.path = self.data_root / self._FILENAME
        self._stream = None
        with self._THREAD_LOCKS_GUARD:
            self._thread_lock = self._THREAD_LOCKS.setdefault(str(self.data_root), threading.RLock())

    def __enter__(self) -> "DataRootMutationLock":
        self._thread_lock.acquire()
        try:
            _reject_link_components(self.data_root)
            self.data_root.mkdir(parents=True, exist_ok=True)
            if os.path.lexists(self.path) and _is_link_or_reparse(self.path):
                raise PathBoundaryError(f"Mutation lock is a link or reparse point: '{self.path}'")
            self._stream = self.path.open("a+b")
            if os.name == "nt":
                self._stream.seek(0, os.SEEK_END)
                if self._stream.tell() == 0:
                    self._stream.write(b"0")
                    self._stream.flush()
                self._stream.seek(0)
                msvcrt.locking(self._stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX)
        except (OSError, RuntimeError, ValueError) as exc:
            try:
                self.close()
            except BaseException:
                pass
            finally:
                self._thread_lock.release()
            if isinstance(exc, PathBoundaryError):
                raise
            raise PathBoundaryError(f"Unable to acquire data-root mutation lock '{self.path}'") from exc
        return self

    def close(self) -> None:
        if self._stream is None:
            return
        try:
            if os.name == "nt":
                self._stream.seek(0)
                msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            self._stream.close()
            self._stream = None

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        cleanup_error = None
        try:
            self.close()
        except BaseException as exc:
            cleanup_error = exc
        finally:
            self._thread_lock.release()
        if cleanup_error is not None and exc_type is None:
            raise cleanup_error


_REQUIRED_DIRECTORIES = (
    "inputs",
    "paper_understanding",
    "feasibility",
    "source",
    "verification",
    "versioning",
    "logs",
    "checkpoints",
)


def _topic_slug(topic: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(topic)).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    return slug[:64] or "run"


@dataclass(frozen=True)
class Run:
    run_id: str
    path: Path
    inputs: Path
    paper_understanding: Path
    feasibility: Path
    source: Path
    verification: Path
    versioning: Path
    logs: Path
    checkpoints: Path
    research_target: ResearchTarget
    data_root: Path | None = None
    parent_run_id: str | None = None

    @property
    def subpaths(self) -> dict[str, Path]:
        return {name: getattr(self, name) for name in _REQUIRED_DIRECTORIES}

    def assert_inside(self, path: Path | str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.path / candidate
        try:
            candidate_resolved = candidate.resolve(strict=False)
            run_resolved = self.path.resolve(strict=True)
            candidate_resolved.relative_to(run_resolved)
        except (OSError, RuntimeError, ValueError) as exc:
            raise PathBoundaryError(f"Path must remain inside run directory '{self.path}'") from exc
        return candidate


class RunStore:
    def __init__(self, data_root: Path | str):
        self.data_root = _canonical_path(Path(data_root))
        self.runs_root = self.data_root / "runs"

    def _validate_storage_roots(self) -> None:
        _reject_link_components(self.data_root)
        _reject_link_components(self.runs_root)
        canonical_root = self.data_root.resolve(strict=False)
        canonical_runs = self.runs_root.resolve(strict=False)
        try:
            canonical_runs.relative_to(canonical_root)
        except ValueError as exc:
            raise PathBoundaryError(
                f"runs directory must remain inside configured data root '{canonical_root}'"
            ) from exc

    @staticmethod
    def _require_approved_target(research_target: ResearchTarget | None) -> ResearchTarget:
        if not isinstance(research_target, ResearchTarget):
            raise ValueError("an approved research target is required before run construction")
        try:
            return research_target.require_approved()
        except ValueError as exc:
            raise ValueError(
                "an approved research target with approval provenance is required before run construction"
            ) from exc

    def _create_locked(
        self,
        topic: str,
        *,
        research_target: ResearchTarget,
        parent_run_id: str | None = None,
    ) -> Run:
        self._validate_storage_roots()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._validate_storage_roots()
        slug = _topic_slug(topic)
        while True:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"{slug}-{timestamp}-{secrets.token_hex(4)}"
            path = self.runs_root / run_id
            try:
                path.mkdir()
            except FileExistsError:
                continue
            self._validate_storage_roots()
            _reject_link_components(path)
            break

        directories = {name: path / name for name in _REQUIRED_DIRECTORIES}
        for directory in directories.values():
            directory.mkdir()
        return Run(
            run_id=run_id,
            path=path,
            data_root=self.data_root,
            parent_run_id=parent_run_id,
            research_target=research_target,
            **directories,
        )

    def create(self, topic: str, *, research_target: ResearchTarget | None = None) -> Run:
        research_target = self._require_approved_target(research_target)
        with DataRootMutationLock(self.data_root):
            return self._create_locked(topic, research_target=research_target)

    def _validate_parent_run(self, parent_run: Run) -> Path:
        if not isinstance(parent_run, Run):
            raise PathBoundaryError("parent_run must be a Run created by RunStore")
        if parent_run.data_root is not None and _canonical_path(parent_run.data_root) != self.data_root:
            raise PathBoundaryError("parent_run must belong to this data root")
        try:
            parent_path = parent_run.path.resolve(strict=True)
            runs_root = self.runs_root.resolve(strict=False)
            parent_path.relative_to(runs_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise PathBoundaryError("parent_run must be inside this store's runs directory") from exc
        if parent_path.name != parent_run.run_id:
            raise PathBoundaryError("parent_run path does not match parent_run.run_id")
        _reject_link_components(parent_run.path)
        return parent_path

    @staticmethod
    def _reject_reused_paths(parent_path: Path, reused_paths: Iterable[Path | str]) -> None:
        if isinstance(reused_paths, (str, os.PathLike)):
            reused_paths = (reused_paths,)
        for reused_path in reused_paths:
            candidate = Path(reused_path)
            if not candidate.is_absolute():
                candidate = parent_path / candidate
            try:
                candidate = candidate.resolve(strict=False)
                candidate.relative_to(parent_path)
            except (OSError, RuntimeError) as exc:
                raise PathBoundaryError(
                    f"cannot inspect reused artifact path: '{reused_path}'"
                ) from exc
            except ValueError:
                continue
            raise PathBoundaryError(f"superseding run cannot reuse parent artifacts: '{reused_path}'")

    def create_superseding(
        self,
        parent_run: Run,
        topic: str,
        *,
        research_target: ResearchTarget | None = None,
        reused_paths: Iterable[Path | str] = (),
    ) -> Run:
        """Create an isolated child run linked to ``parent_run``.

        Every child directory is newly allocated. Any requested path under the
        parent run is rejected before allocation so source, metric, and result
        artifacts cannot leak across superseding runs.
        """
        research_target = self._require_approved_target(research_target)
        with DataRootMutationLock(self.data_root):
            parent_path = self._validate_parent_run(parent_run)
            self._reject_reused_paths(parent_path, reused_paths)
            return self._create_locked(
                topic,
                research_target=research_target,
                parent_run_id=parent_run.run_id,
            )
