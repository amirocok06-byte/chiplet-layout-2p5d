"""Runtime target and evidence policy for the currently approved workflow."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable
from pathlib import Path

from .paper_corpus import (
    CORPUS_MANIFEST_NAME,
    PaperCorpusError,
    read_ai42_5dic_corpus_manifest,
)
from .research_target import ResearchTarget


AI42_5DIC_TARGET_ID = "ai42.5Dic"
AI42_5DIC_DISPLAY_NAME = "ai42.5Dic"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class TargetPolicyError(ValueError):
    """Raised when a run is not bound to the required target evidence."""


def require_target_id(target: ResearchTarget, expected_target_id: str) -> ResearchTarget:
    """Fail closed when an orchestrated run carries another research target."""

    if not expected_target_id or not expected_target_id.strip():
        raise TargetPolicyError("expected target id must be non-empty")
    if target.target_id != expected_target_id:
        raise TargetPolicyError(
            f"workflow target is locked to {expected_target_id!r}; "
            f"received {target.target_id!r}"
        )
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _matching_corpus_manifest(candidate: Path, checksum: str, root: Path) -> bool:
    if candidate.name != CORPUS_MANIFEST_NAME:
        return False
    try:
        manifest = read_ai42_5dic_corpus_manifest(candidate, root)
    except PaperCorpusError:
        return False
    return manifest.corpus_checksum.lower() == checksum.lower()


def require_matching_evidence(
    target: ResearchTarget,
    evidence_paths: Iterable[Path],
    *,
    allowed_root: Path | str,
) -> Path:
    """Require one staged evidence file to match the approved target checksum.

    ``allowed_root`` is mandatory so this helper cannot accidentally hash a
    paper outside the run's staged-input boundary. It never accesses the
    authorized F-drive roots.
    """

    checksum = target.evidence_checksum
    if not _SHA256_RE.fullmatch(checksum):
        raise TargetPolicyError(
            "ai42.5Dic target evidence_checksum must be a SHA-256 digest "
            "bound to the staged paper"
        )

    root = Path(allowed_root).expanduser().absolute()
    try:
        root_real = Path(os.path.realpath(root))
    except OSError as exc:
        raise TargetPolicyError("target evidence root cannot be inspected") from exc
    if not root.is_dir() or root_real != root:
        raise TargetPolicyError("target evidence root must be a real directory")

    candidates = sorted({Path(path) for path in evidence_paths}, key=lambda path: str(path))
    for path in candidates:
        try:
            candidate = path.expanduser().absolute()
            candidate.relative_to(root)
            if not candidate.is_file() or os.path.islink(candidate):
                continue
            if Path(os.path.realpath(candidate)) != candidate:
                continue
            if candidate.name == CORPUS_MANIFEST_NAME:
                if _matching_corpus_manifest(candidate, checksum, root):
                    return candidate
                continue
            if _sha256(candidate).lower() == checksum.lower():
                return candidate
        except OSError:
            continue
        except ValueError:
            continue
    raise TargetPolicyError(
        "no staged evidence file matches the approved ai42.5Dic target checksum; "
        "the correct paper must be supplied before ReaderAgent can run"
    )


__all__ = [
    "AI42_5DIC_DISPLAY_NAME",
    "AI42_5DIC_TARGET_ID",
    "TargetPolicyError",
    "require_matching_evidence",
    "require_target_id",
]
