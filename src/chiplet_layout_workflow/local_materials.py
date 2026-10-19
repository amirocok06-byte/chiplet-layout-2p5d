"""Read-only indexing and safe staging for explicitly authorized local files.

The scanner treats local source trees as untrusted data.  It never imports,
executes, or follows links while walking an authorized root.  Only records
from the resulting index may be passed to :func:`stage_selected`.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MaterialError(ValueError):
    """Base error for local material indexing and staging."""


class MaterialBoundaryError(MaterialError):
    """Raised when an input, link, or output path violates the boundary."""


class MaterialManifestError(MaterialError):
    """Raised when a material manifest is malformed or tampered with."""


class AuthorizedMaterialRoot(BaseModel):
    """One user-authorized, read-only source root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    path: Path

    @field_validator("key")
    @classmethod
    def _key_is_safe(cls, value: str) -> str:
        if not value or value in {".", ".."} or any(char in value for char in "/\\"):
            raise ValueError("material root key must be a non-empty path component")
        return value


class MaterialRecord(BaseModel):
    """A deterministic snapshot of one regular file under an authorized root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root_key: str
    source_relative_path: str
    source_path: str
    file_kind: str
    byte_size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    safe_to_stage: bool
    text_like: bool
    category: str
    exclusion_reason: str | None = None

    @field_validator("source_relative_path")
    @classmethod
    def _relative_path_is_posix(cls, value: str) -> str:
        parsed = PurePosixPath(value)
        if not value or parsed.is_absolute() or "\\" in value or ".." in parsed.parts:
            raise ValueError("source_relative_path must be a relative POSIX path")
        return value


class MaterialIndex(BaseModel):
    """The serialized index produced by :func:`scan_authorized_roots`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = 1
    roots: tuple[AuthorizedMaterialRoot, ...]
    records: tuple[MaterialRecord, ...]
    index_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


class StagedMaterial(BaseModel):
    """One explicitly selected material copied into run inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root_key: str
    source_relative_path: str
    source_path: str
    staged_relative_path: str
    staged_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    staged_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_file_kind: str | None = None
    source_category: str | None = None
    staging_reason: str


class MaterialManifest(BaseModel):
    """Checksummed record of files staged under a run's ``inputs`` directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = 1
    index_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    materials: tuple[StagedMaterial, ...]
    manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


_TEXT_EXTENSIONS = {
    ".bib",
    ".blocks",
    ".csv",
    ".json",
    ".markdown",
    ".md",
    ".nets",
    ".pl",
    ".power",
    ".txt",
    ".yaml",
    ".yml",
}
_SCRIPT_EXTENSIONS = {
    ".bat",
    ".cmd",
    ".cmake",
    ".csh",
    ".exe",
    ".js",
    ".ps1",
    ".psm1",
    ".py",
    ".pyw",
    ".rb",
    ".sh",
    ".so",
    ".dll",
}
_SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".java",
    ".rs",
    ".go",
}
_PDF_EXTENSIONS = {".pdf"}
_ZIP_EXTENSIONS = {".7z", ".gz", ".rar", ".tar", ".tgz", ".zip"}
_MAKEFILE_NAMES = {"cmakelists.txt", "makefile", "gnumakefile"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _is_link_or_reparse(path: Path) -> bool:
    if os.path.islink(path):
        return True
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except AttributeError:
        return False
    except FileNotFoundError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _reject_link_components(path: Path) -> None:
    current = Path(path.anchor) if path.anchor else Path()
    parts = path.parts[1:] if path.anchor else path.parts
    for part in parts:
        current /= part
        if os.path.lexists(current) and _is_link_or_reparse(current):
            raise MaterialBoundaryError(f"link or reparse component is not allowed: '{current}'")


def _absolute(path: Path | str) -> Path:
    return Path(path).expanduser().absolute()


def _safe_root(root: AuthorizedMaterialRoot) -> AuthorizedMaterialRoot:
    path = _absolute(root.path)
    if not path.exists() or not path.is_dir():
        raise MaterialBoundaryError(f"authorized root must be an existing directory: '{path}'")
    _reject_link_components(path)
    return root.model_copy(update={"path": path})


def _normalize_roots(
    roots: Mapping[str, Path | str] | Iterable[AuthorizedMaterialRoot],
) -> tuple[AuthorizedMaterialRoot, ...]:
    if isinstance(roots, Mapping):
        roots = tuple(AuthorizedMaterialRoot(key=key, path=path) for key, path in roots.items())
    else:
        roots = tuple(roots)
    if not roots:
        raise MaterialBoundaryError("at least one authorized root is required")
    normalized = tuple(_safe_root(root) for root in roots)
    keys = [root.key for root in normalized]
    if len(keys) != len(set(keys)):
        raise MaterialBoundaryError("authorized root keys must be unique")
    canonical_paths = [root.path.resolve(strict=True) for root in normalized]
    if len(canonical_paths) != len(set(canonical_paths)):
        raise MaterialBoundaryError("authorized roots must be distinct")
    return normalized


def _classify(path: Path) -> tuple[str, bool, bool, str, str | None]:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in _ZIP_EXTENSIONS:
        return "zip", False, False, "unsupported", "zip_archive"
    if name in _MAKEFILE_NAMES or suffix in _SCRIPT_EXTENSIONS:
        return "script", False, True, "unsupported", "script_like"
    if suffix in _SOURCE_EXTENSIONS:
        return "source", False, True, "unsupported", "source_code"
    if suffix in _PDF_EXTENSIONS:
        return "pdf", True, False, "reference_paper", None
    if suffix in _TEXT_EXTENSIONS:
        if "AI相关论文" in path.parts:
            category = "ai_placement_note"
        elif name.startswith("readme"):
            category = "readme"
        elif any(token in name for token in ("guide", "指南", "说明")):
            category = "project_guide"
        elif any(token in name for token in ("source", "inventory", "清单")):
            category = "source_inventory"
        elif suffix in {".blocks", ".nets", ".pl", ".power"} or any(
            part.lower() in {"benchmark", "benchmarks", "cases", "case", "inputs"}
            for part in path.parts
        ):
            category = "benchmark_input"
        elif suffix in {".md", ".markdown"}:
            category = "paper_note"
        else:
            category = "unsupported"
        if suffix in {".md", ".markdown"}:
            file_kind = "markdown"
        elif suffix in {".blocks", ".nets", ".pl", ".power"}:
            file_kind = "bookshelf"
        else:
            file_kind = suffix.lstrip(".")
        return file_kind, True, True, category, None
    return (suffix.lstrip(".") or "unknown"), False, False, "unsupported", "unsupported_file_type"


def _iter_files(root: Path) -> Iterable[Path]:
    def walk(directory: Path) -> Iterable[Path]:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: (entry.name.casefold(), entry.name))
        except OSError as exc:
            raise MaterialBoundaryError(f"unable to scan authorized root '{directory}'") from exc
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink() or _is_link_or_reparse(path):
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    yield from walk(path)
                elif entry.is_file(follow_symlinks=False):
                    yield path
            except OSError as exc:
                raise MaterialBoundaryError(f"unable to inspect authorized path '{path}'") from exc

    yield from walk(root)


def _manifest_payload(manifest: MaterialManifest | dict[str, Any]) -> dict[str, Any]:
    if isinstance(manifest, MaterialManifest):
        return manifest.model_dump(mode="json", exclude={"manifest_checksum"})
    payload = dict(manifest)
    payload.pop("manifest_checksum", None)
    return payload


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path = _absolute(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    _reject_link_components(parent)
    if path.exists() and _is_link_or_reparse(path):
        raise MaterialBoundaryError(f"manifest path is a link or reparse point: '{path}'")
    encoded = _canonical_json(payload) + b"\n"
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        _reject_link_components(temporary)
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def scan_authorized_roots(
    roots: Mapping[str, Path | str] | Iterable[AuthorizedMaterialRoot],
    *,
    index_path: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> MaterialIndex:
    """Index regular files under explicit roots without executing their contents."""

    if index_path is not None and output_dir is not None:
        raise ValueError("pass either index_path or output_dir, not both")
    normalized_roots = _normalize_roots(roots)
    records: list[MaterialRecord] = []
    for root in normalized_roots:
        root_path = root.path
        for path in _iter_files(root_path):
            relative = path.relative_to(root_path).as_posix()
            file_kind, safe, text_like, category, exclusion_reason = _classify(path)
            try:
                stat_result = path.stat(follow_symlinks=False)
                digest = _sha256_file(path)
            except OSError as exc:
                raise MaterialBoundaryError(f"unable to hash authorized file '{path}'") from exc
            if safe and stat_result.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
                safe = False
                exclusion_reason = "executable"
            records.append(
                MaterialRecord(
                    root_key=root.key,
                    source_relative_path=relative,
                    source_path=str(path),
                    file_kind=file_kind,
                    byte_size=stat_result.st_size,
                    sha256=digest,
                    safe_to_stage=safe,
                    text_like=text_like,
                    category=category,
                    exclusion_reason=exclusion_reason,
                )
            )
    records.sort(key=lambda record: (record.root_key, record.source_relative_path))
    unsigned = {"version": 1, "roots": [root.model_dump(mode="json") for root in normalized_roots], "records": [record.model_dump(mode="json") for record in records]}
    checksum = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    index = MaterialIndex(roots=normalized_roots, records=tuple(records), index_checksum=checksum)
    destination = index_path
    if destination is None and output_dir is not None:
        destination = _absolute(output_dir) / "material_index.json"
    if destination is not None:
        destination_path = _absolute(destination)
        for root in normalized_roots:
            try:
                destination_path.resolve(strict=False).relative_to(root.path.resolve(strict=True))
            except ValueError:
                continue
            raise MaterialBoundaryError("material index must not be written inside an authorized source root")
        _atomic_json_write(destination_path, index.model_dump(mode="json"))
    return index


def _record_lookup(index: MaterialIndex) -> dict[tuple[str, str], MaterialRecord]:
    return {(record.root_key, record.source_relative_path): record for record in index.records}


def _record_source(index: MaterialIndex, record: MaterialRecord) -> tuple[AuthorizedMaterialRoot, Path]:
    if not isinstance(record, MaterialRecord):
        raise MaterialBoundaryError("stage_selected accepts MaterialRecord objects from the index")
    stored = _record_lookup(index).get((record.root_key, record.source_relative_path))
    if stored is None or stored != record:
        raise MaterialBoundaryError("selected material is not an exact record from the index")
    root = next(root for root in index.roots if root.key == record.root_key)
    relative = PurePosixPath(record.source_relative_path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in record.source_relative_path:
        raise MaterialBoundaryError("indexed material path is outside the authorized root")
    source = _absolute(root.path).joinpath(*relative.parts)
    try:
        source.resolve(strict=True).relative_to(root.path.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as exc:
        raise MaterialBoundaryError("indexed material path is outside the authorized root") from exc
    _reject_link_components(source)
    if source != _absolute(record.source_path) or not source.is_file():
        raise MaterialBoundaryError("indexed material source no longer matches the authorized file")
    if source.stat(follow_symlinks=False).st_size != record.byte_size or _sha256_file(source) != record.sha256:
        raise MaterialBoundaryError("indexed material changed after scanning")
    return root, source


def _safe_output_directory(path: Path, roots: Sequence[AuthorizedMaterialRoot]) -> Path:
    path = _absolute(path)
    for root in roots:
        try:
            path.resolve(strict=False).relative_to(root.path.resolve(strict=True))
        except ValueError:
            continue
        raise MaterialBoundaryError("run inputs must not be inside an authorized source root")
    path.mkdir(parents=True, exist_ok=True)
    _reject_link_components(path)
    return path


def stage_selected(
    index: MaterialIndex,
    selected: Iterable[MaterialRecord],
    run_inputs: Path | str,
    *,
    staging_reason: str = "explicitly_selected",
) -> MaterialManifest:
    """Copy only exact, safe index records below a run's ``inputs`` directory."""

    if not isinstance(index, MaterialIndex):
        raise MaterialBoundaryError("stage_selected requires a MaterialIndex")
    if not staging_reason.strip():
        raise ValueError("staging_reason must be non-empty")
    run_inputs_path = _safe_output_directory(Path(run_inputs), index.roots)
    staged: list[StagedMaterial] = []
    seen: set[tuple[str, str]] = set()
    for record in selected:
        root, source = _record_source(index, record)
        key = (record.root_key, record.source_relative_path)
        if key in seen:
            raise MaterialBoundaryError("selected material records must be unique")
        seen.add(key)
        if not record.safe_to_stage:
            raise MaterialBoundaryError(
                f"material is not safe to stage: {record.source_relative_path}"
            )
        destination_relative = PurePosixPath("materials") / root.key / PurePosixPath(record.source_relative_path)
        destination = run_inputs_path.joinpath(*destination_relative.parts)
        _reject_link_components(destination.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _reject_link_components(destination.parent)
        if destination.exists() or os.path.lexists(destination):
            raise MaterialBoundaryError(f"refusing to overwrite staged material: '{destination}'")
        try:
            with source.open("rb") as source_stream, destination.open("xb") as destination_stream:
                while chunk := source_stream.read(1024 * 1024):
                    destination_stream.write(chunk)
                destination_stream.flush()
                os.fsync(destination_stream.fileno())
        except OSError as exc:
            raise MaterialBoundaryError(f"unable to stage material '{source}'") from exc
        _reject_link_components(destination)
        staged_hash = _sha256_file(destination)
        if staged_hash != record.sha256:
            raise MaterialBoundaryError(f"staged material checksum mismatch: '{destination}'")
        staged.append(
            StagedMaterial(
                root_key=root.key,
                source_relative_path=record.source_relative_path,
                source_path=str(source),
                staged_relative_path=destination_relative.as_posix(),
                staged_path=str(destination),
                source_sha256=record.sha256,
                staged_sha256=staged_hash,
                source_file_kind=record.file_kind,
                source_category=record.category,
                staging_reason=staging_reason,
            )
        )
    staged.sort(key=lambda item: (item.root_key, item.source_relative_path))
    unsigned = {
        "version": 1,
        "index_checksum": index.index_checksum,
        "materials": [item.model_dump(mode="json") for item in staged],
    }
    manifest_checksum = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    manifest = MaterialManifest(
        index_checksum=index.index_checksum,
        materials=tuple(staged),
        manifest_checksum=manifest_checksum,
    )
    _atomic_json_write(run_inputs_path / "material_manifest.json", manifest.model_dump(mode="json"))
    return manifest


def read_material_manifest(path: Path | str) -> MaterialManifest:
    """Read a manifest using its own absolute file path, independent of cwd."""

    manifest_path = _absolute(path)
    if not manifest_path.is_file() or _is_link_or_reparse(manifest_path):
        raise MaterialManifestError(f"material manifest is not a regular file: '{manifest_path}'")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = MaterialManifest.model_validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise MaterialManifestError(f"invalid material manifest: '{manifest_path}'") from exc
    expected = hashlib.sha256(_canonical_json(_manifest_payload(manifest))).hexdigest()
    if expected != manifest.manifest_checksum:
        raise MaterialManifestError(f"material manifest checksum mismatch: '{manifest_path}'")
    return manifest


# Short aliases keep the public surface convenient without adding alternate behavior.
scan_materials = scan_authorized_roots
stage_materials = stage_selected


__all__ = [
    "AuthorizedMaterialRoot",
    "MaterialBoundaryError",
    "MaterialError",
    "MaterialIndex",
    "MaterialManifest",
    "MaterialManifestError",
    "MaterialRecord",
    "StagedMaterial",
    "read_material_manifest",
    "scan_authorized_roots",
    "scan_materials",
    "stage_materials",
    "stage_selected",
]
