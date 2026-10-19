"""Prepare a deterministic, checksummed evidence corpus for ``ai42.5Dic``.

The source trees are treated as untrusted input.  This module consumes only
files already copied below a run's ``inputs`` directory by
``local_materials.stage_selected``.  PDF bytes are never sent downstream:
they are converted into page-located UTF-8 text parts and verified through a
small manifest before a reader agent can use them.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .local_materials import (
    MaterialBoundaryError,
    MaterialManifest,
    StagedMaterial,
    _canonical_json,
    _is_link_or_reparse,
    _reject_link_components,
    _sha256_file,
    read_material_manifest,
)

try:  # Kept injectable for tests and optional local installations.
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - depends on the host environment.
    PdfReader = None  # type: ignore[assignment,misc]


CORPUS_MANIFEST_NAME = "ai42_5dic_corpus_manifest.json"
MAX_TEXT_PART_BYTES = 60 * 1024
_TARGET_ID = "ai42.5Dic"

# This is an approval boundary, not a discovery rule.  The Reader corpus is
# intentionally limited to the ten papers and ten AI-placement notes that the
# user approved.  The values are kept as paths relative to the two approved
# source directories so a manifest cannot replace the corpus with a similarly
# shaped collection from another directory.
APPROVED_SOURCE_ROOTS = {
    "reference_paper": Path(
        r"F:\GAME\2027-SRTP-面向2.5D系统集成的芯粒布局方法研究\1-参考文献"
    ),
    "ai_placement_note": Path(r"F:\GAME\SRTP\AI相关论文"),
}
APPROVED_REFERENCE_PAPERS = frozenset(
    {
        "2021-DATE-TAP-2.5D-A Thermally-Aware Chiplet Placement Methodology for 2.5D Systems.pdf",
        "2021-RLPlanner-MF_RL_based_Floorplanning_for_Chiplets_with_Multi-Modal_Fusion_and_Fast_Thermal_Analysis.pdf",
        "2024-ETPC-Optimizing_Chiplet_Placement_in_Thermally_Aware_Heterogeneous_2.5D_Systems_Using_RL.pdf",
        "2024-ICCAD-ATPlace2.5D-Analytical Thermal-Aware Chiplet Placement Framework for Large-Scale 2.5D-IC.pdf",
        "2024-ISEDA-Chiplet_Placement_Order_Exploration_based_on_Learning_to_Rank_with_Graph_Representation.pdf",
        "2025-Fast Thermal-Aware Chiplet Placement Assisted by Surrogate.pdf",
        "2025-GLSVLSI-TACPlace Ultrafast Thermal-Aware Chiplet Placement with Feasibility Seeking.pdf",
        "2025-TDPNavigator-Placer_Thermal-_and_Wirelength-Aware_Chiplet_Placement_in_2.5D_Systems_Through_Multi-Agent_RL.pdf",
        "2025-TODAES-Thermal-Aware Chiplet Placement for 2.5D ICs with Sequence Pair Based Tree.pdf",
        "2026-Integration-MaskPlace2. 5D-Mask-guided thermal-aware chiplet placement for large-scale 2.5 D ICs by RL.pdf",
    }
)
APPROVED_AI_PLACEMENT_NOTES = frozenset(
    {
        "00_总览.md",
        "01_EoH.md",
        "02_EvoPlace.md",
        "03_CoEvoP&R.md",
        "04_GoalEvolve.md",
        "05_GR-Evolve.md",
        "06_MAGE.md",
        "07_OrderPlace.md",
        "08_MacroAgent.md",
        "09_软模块迁移方案.md",
    }
)
APPROVED_SOURCE_SHA256 = {
    "2021-DATE-TAP-2.5D-A Thermally-Aware Chiplet Placement Methodology for 2.5D Systems.pdf": "b1df01100d80e83a2b0098080548265911dd42c12d7e8153c086433124f99e09",
    "2021-RLPlanner-MF_RL_based_Floorplanning_for_Chiplets_with_Multi-Modal_Fusion_and_Fast_Thermal_Analysis.pdf": "1e56fee7812dfffcbd2dd0c760543bda4bc12f4dc41479e727fd3269adfba1ee",
    "2024-ETPC-Optimizing_Chiplet_Placement_in_Thermally_Aware_Heterogeneous_2.5D_Systems_Using_RL.pdf": "cb0f987576a44bd09453f2bb43f302dd1cf9470f76f401e9952eb57e4d204a83",
    "2024-ICCAD-ATPlace2.5D-Analytical Thermal-Aware Chiplet Placement Framework for Large-Scale 2.5D-IC.pdf": "2c21c120c51bfa33b5c6bca629b7981633d34f295a11b2770de18687285afe74",
    "2024-ISEDA-Chiplet_Placement_Order_Exploration_based_on_Learning_to_Rank_with_Graph_Representation.pdf": "71b023ac6696cdab657bc56bbf794be4021413d0b70a19d967da465a3b099e77",
    "2025-Fast Thermal-Aware Chiplet Placement Assisted by Surrogate.pdf": "2d385a9122135eabf4984665bcffb10b5774f5552ae5f9d040cc2aa576d288df",
    "2025-GLSVLSI-TACPlace Ultrafast Thermal-Aware Chiplet Placement with Feasibility Seeking.pdf": "f8414a04e36e5a6ea0754eb1160e9a6863988465ef26fa64814ba9165a695fe0",
    "2025-TDPNavigator-Placer_Thermal-_and_Wirelength-Aware_Chiplet_Placement_in_2.5D_Systems_Through_Multi-Agent_RL.pdf": "ad1368004f8acc93c3923ef85242582ca9f426d9686a7992c34abba58df40b83",
    "2025-TODAES-Thermal-Aware Chiplet Placement for 2.5D ICs with Sequence Pair Based Tree.pdf": "a1ff18b8065a18b3b2b2ee02a4d702feafa351db0328f156f66f65c280ad55d9",
    "2026-Integration-MaskPlace2. 5D-Mask-guided thermal-aware chiplet placement for large-scale 2.5 D ICs by RL.pdf": "5d5da30f70e4b683dafa8bb21dc3755d13d985d0acad6d4ca9cccb9631f8369f",
    "00_总览.md": "4b4084048743ee2b7dbc36a1129bf5a50f3008c8a3af70c4846093eab8f0dbfd",
    "01_EoH.md": "95ee99bb0e061c0b40753bcafb2b7dad4704c38badb7be6cbb99d027f2ecbbb7",
    "02_EvoPlace.md": "3e8ade8221a7af3e990b237e9d4f4a7ac82785910c6fa474dbfe8e7f6ec10ef0",
    "03_CoEvoP&R.md": "ffc67af5c1fc54cd0362d8af72064013e1eb0e02772e104227335ef1f4a59e9a",
    "04_GoalEvolve.md": "f9d99fda18b3329ed7eb72b480bc8776dd6352d8ed817f432287aa4777f348e2",
    "05_GR-Evolve.md": "31329e79816ad672788a0234990b5f096223cd550ac58dee24983281f242a2ab",
    "06_MAGE.md": "70238fd9280aeedd265e64572b6880d407ddfa23020a2631f6e7868f4162cc28",
    "07_OrderPlace.md": "db18668496e14d61a95406d53401ffb420bcccd99d61c4a7e0e8acccea721a29",
    "08_MacroAgent.md": "2b45cecaad50320865fc1cb245ba04e712032996dc1070a6e1879b41679a5aee",
    "09_软模块迁移方案.md": "ec0b4f9a0b8e89987e62b758c00a46ce1e69e17068dc8e85387aab4e3139398e",
}


class PaperCorpusError(ValueError):
    """Raised when corpus preparation or verification fails."""


class CorpusTextPart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    byte_size: int = Field(ge=1)


class CorpusPaper(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_relative_path: str
    staged_relative_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: int = Field(ge=1)
    unicode_replacement_count: int = Field(ge=0)
    text_parts: tuple[CorpusTextPart, ...] = Field(min_length=1)

    @field_validator("source_relative_path", "staged_relative_path")
    @classmethod
    def _safe_paths(cls, value: str) -> str:
        _safe_relative(value, label="corpus paper path")
        return value


class CorpusNote(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_relative_path: str
    staged_relative_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("source_relative_path", "staged_relative_path")
    @classmethod
    def _safe_paths(cls, value: str) -> str:
        _safe_relative(value, label="corpus note path")
        return value


class Ai42CorpusManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    target_id: Literal["ai42.5Dic"] = _TARGET_ID
    papers: tuple[CorpusPaper, ...]
    ai_placement_notes: tuple[CorpusNote, ...]
    material_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


def _absolute(path: Path | str) -> Path:
    return Path(path).expanduser().absolute()


def _manifest_payload(manifest: Ai42CorpusManifest | dict[str, Any]) -> dict[str, Any]:
    if isinstance(manifest, Ai42CorpusManifest):
        return manifest.model_dump(mode="json", exclude={"corpus_checksum"})
    payload = dict(manifest)
    payload.pop("corpus_checksum", None)
    return payload


def _corpus_identity_payload(manifest: Ai42CorpusManifest | dict[str, Any]) -> dict[str, Any]:
    """Return the run-location-independent payload bound to the target."""

    payload = _manifest_payload(manifest)
    payload.pop("material_manifest_checksum", None)
    return payload


def _safe_relative(value: str, *, label: str) -> PurePosixPath:
    parsed = PurePosixPath(value)
    if (
        not value
        or parsed.is_absolute()
        or "\\" in value
        or any(":" in part for part in parsed.parts)
        or any(char in value for char in "\r\n\x00")
        or ".." in parsed.parts
        or any(part in {"", "."} for part in parsed.parts)
    ):
        raise PaperCorpusError(f"{label} must be a safe relative POSIX path")
    return parsed


def _run_relative(run_inputs: Path, relative: str, *, label: str) -> Path:
    parsed = _safe_relative(relative, label=label)
    candidate = run_inputs.joinpath(*parsed.parts)
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(run_inputs.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as exc:
        raise PaperCorpusError(f"{label} escapes run inputs") from exc
    try:
        _reject_link_components(candidate)
    except MaterialBoundaryError as exc:
        raise PaperCorpusError(f"{label} contains a link or reparse point") from exc
    if _is_link_or_reparse(candidate):
        raise PaperCorpusError(f"{label} is a link or reparse point")
    return candidate


def _ensure_run_inputs(path: Path | str) -> Path:
    run_inputs = _absolute(path)
    run_inputs.mkdir(parents=True, exist_ok=True)
    _reject_link_components(run_inputs)
    return run_inputs


def _existing_run_inputs(path: Path | str) -> Path:
    run_inputs = _absolute(path)
    if not run_inputs.is_dir() or _is_link_or_reparse(run_inputs):
        raise PaperCorpusError("run inputs must be a regular directory")
    try:
        _reject_link_components(run_inputs)
    except MaterialBoundaryError as exc:
        raise PaperCorpusError("run inputs contains a link or reparse point") from exc
    return run_inputs


def _require_pdf_reader():
    if PdfReader is None:
        raise PaperCorpusError(
            "pypdf is required to prepare PDF corpus evidence; install declared dependencies before running"
        )
    return PdfReader


def _normalize_pdf_text(raw: str) -> tuple[str, int]:
    nul_count = raw.count("\x00")
    normalized_raw = raw.replace("\x00", "\uFFFD")
    replacement_count = nul_count + sum(0xD800 <= ord(char) <= 0xDFFF for char in raw)
    normalized = normalized_raw.encode("utf-8", "replace").decode("utf-8")
    return normalized, replacement_count


def _material_lookup(material_manifest: MaterialManifest) -> dict[tuple[str, str], StagedMaterial]:
    lookup: dict[tuple[str, str], StagedMaterial] = {}
    for material in material_manifest.materials:
        key = (material.root_key, material.source_relative_path)
        if key in lookup:
            raise PaperCorpusError(f"duplicate staged material key: {key!r}")
        lookup[key] = material
    return lookup


def _approved_source_path(material: StagedMaterial, *, kind: Literal["paper", "note"]) -> bool:
    source_path = Path(material.source_path).expanduser().absolute()
    source_kind = "reference_paper" if kind == "paper" else "ai_placement_note"
    expected_root = APPROVED_SOURCE_ROOTS[source_kind]
    try:
        source_path.relative_to(expected_root.absolute())
    except ValueError:
        return False
    allowed_names = APPROVED_REFERENCE_PAPERS if kind == "paper" else APPROVED_AI_PLACEMENT_NOTES
    expected_hash = APPROVED_SOURCE_SHA256.get(source_path.name)
    relative_path = PurePosixPath(material.source_relative_path).as_posix()
    expected_relative_paths = {
        source_path.name,
        f"{'1-参考文献' if kind == 'paper' else 'AI相关论文'}/{source_path.name}",
    }
    return (
        source_path.parent == expected_root.absolute()
        and source_path.name in allowed_names
        and relative_path in expected_relative_paths
        and expected_hash is not None
        and material.source_sha256 == expected_hash
    )


def _verify_material_manifest(material_manifest: MaterialManifest) -> None:
    expected = hashlib.sha256(
        _canonical_json(material_manifest.model_dump(mode="json", exclude={"manifest_checksum"}))
    ).hexdigest()
    if expected != material_manifest.manifest_checksum:
        raise PaperCorpusError("staged material manifest checksum mismatch")


def _selected_material(
    lookup: dict[tuple[str, str], StagedMaterial],
    run_inputs: Path,
    key: tuple[str, str],
    *,
    kind: Literal["paper", "note"],
) -> StagedMaterial:
    material = lookup.get(key)
    if material is None:
        raise PaperCorpusError(f"selected material is not in the staged manifest: {key!r}")
    _safe_relative(material.source_relative_path, label="staged source path")
    expected_staged = (
        PurePosixPath("materials")
        / material.root_key
        / PurePosixPath(material.source_relative_path)
    ).as_posix()
    if material.staged_relative_path != expected_staged:
        raise PaperCorpusError(f"staged material path is not bound to its source: {key!r}")
    if material.source_sha256 != material.staged_sha256:
        raise PaperCorpusError(f"source/staged checksum mismatch: {key!r}")
    if not _approved_source_path(material, kind=kind):
        raise PaperCorpusError(
            f"selected material is outside the approved ai42.5Dic corpus: {key!r}"
        )
    staged_path = _run_relative(
        run_inputs, material.staged_relative_path, label="staged material path"
    )
    if not staged_path.is_file():
        raise PaperCorpusError(f"staged material is missing: '{staged_path}'")
    if _sha256_file(staged_path) != material.staged_sha256:
        raise PaperCorpusError(f"staged material checksum mismatch: '{staged_path}'")
    suffix = Path(material.source_relative_path).suffix.lower()
    if kind == "paper" and suffix != ".pdf":
        raise PaperCorpusError(f"reference paper is not a PDF: {key!r}")
    if kind == "paper" and (
        material.source_file_kind != "pdf" or material.source_category != "reference_paper"
    ):
        raise PaperCorpusError(f"staged material is not classified as a reference paper: {key!r}")
    if kind == "note" and suffix not in {".md", ".markdown"}:
        raise PaperCorpusError(f"AI placement note is not Markdown: {key!r}")
    if kind == "note" and (
        material.source_file_kind != "markdown" or material.source_category != "ai_placement_note"
    ):
        raise PaperCorpusError(f"staged material is not classified as an AI placement note: {key!r}")
    return material


def _header(
    *,
    source_relative_path: str,
    source_sha256: str,
    source_kind: str,
    page_start: int,
    page_end: int,
) -> str:
    return (
        "---\n"
        f"source_relative_path: {source_relative_path}\n"
        f"source_sha256: {source_sha256}\n"
        f"source_kind: {source_kind}\n"
        f"page_start: {page_start}\n"
        f"page_end: {page_end}\n"
        "---\n\n"
    )


def _split_page_text(
    text: str,
    *,
    material: StagedMaterial,
    page_number: int,
) -> list[bytes]:
    body = f"[PDF page {page_number}]\n\n{text}"
    header = _header(
        source_relative_path=material.source_relative_path,
        source_sha256=material.source_sha256,
        source_kind="reference_paper",
        page_start=page_number,
        page_end=page_number,
    )
    header_bytes = header.encode("utf-8")
    if len(header_bytes) >= MAX_TEXT_PART_BYTES:
        raise PaperCorpusError("corpus provenance header exceeds text part size limit")

    def encode(prefix: str) -> bytes:
        return (header + prefix).encode("utf-8")

    if len(encode(body)) <= MAX_TEXT_PART_BYTES:
        return [encode(body)]

    chunks: list[bytes] = []
    remaining = body
    while remaining:
        low, high = 1, len(remaining)
        best = 0
        while low <= high:
            middle = (low + high) // 2
            candidate = encode(remaining[:middle])
            if len(candidate) <= MAX_TEXT_PART_BYTES:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best == 0:
            raise PaperCorpusError("unable to split PDF page into a valid UTF-8 text part")
        chunks.append(encode(remaining[:best]))
        remaining = remaining[best:]
    return chunks


def _read_pdf_parts(
    material: StagedMaterial,
    staged_path: Path,
    output_directory: Path,
) -> tuple[int, int, tuple[CorpusTextPart, ...]]:
    reader_class = _require_pdf_reader()
    try:
        reader = reader_class(str(staged_path))
        pages = tuple(reader.pages)
    except Exception as exc:
        raise PaperCorpusError(f"unable to read PDF '{material.source_relative_path}'") from exc
    if not pages:
        raise PaperCorpusError(f"PDF has no pages: '{material.source_relative_path}'")

    replacement_count = 0
    text_parts: list[CorpusTextPart] = []
    paper_directory = output_directory
    part_number = 0
    for page_number, page in enumerate(pages, start=1):
        try:
            raw = page.extract_text() or ""
            normalized, replacements = _normalize_pdf_text(raw)
        except Exception as exc:
            raise PaperCorpusError(
                f"unable to extract PDF page {page_number} from '{material.source_relative_path}'"
            ) from exc
        replacement_count += replacements
        if not normalized:
            continue
        for encoded in _split_page_text(
            normalized,
            material=material,
            page_number=page_number,
        ):
            part_number += 1
            filename = f"page-{page_number:04d}-part-{part_number:04d}.txt"
            destination = paper_directory / filename
            destination.write_bytes(encoded)
            relative = destination.relative_to(output_directory.parent.parent).as_posix()
            text_parts.append(
                CorpusTextPart(
                    relative_path=relative,
                    sha256=hashlib.sha256(encoded).hexdigest(),
                    page_start=page_number,
                    page_end=page_number,
                    byte_size=len(encoded),
                )
            )
    if not text_parts:
        raise PaperCorpusError(
            f"PDF contains no extractable text: '{material.source_relative_path}'"
        )
    return len(pages), replacement_count, tuple(text_parts)


def _write_note_reference(material: StagedMaterial, run_inputs: Path) -> CorpusNote:
    path = _run_relative(run_inputs, material.staged_relative_path, label="staged note path")
    try:
        content = path.read_bytes()
        content.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise PaperCorpusError(f"AI placement note is not valid UTF-8: '{path}'") from exc
    if not content or b"\x00" in content:
        raise PaperCorpusError(f"AI placement note is empty or contains NUL: '{path}'")
    return CorpusNote(
        source_relative_path=material.source_relative_path,
        staged_relative_path=material.staged_relative_path,
        source_sha256=material.source_sha256,
    )


def _validate_selection(keys: tuple[tuple[str, str], ...], label: str) -> None:
    if not keys:
        raise PaperCorpusError(f"{label} selection must not be empty")
    if any(len(key) != 2 or not key[0] or not key[1] for key in keys):
        raise PaperCorpusError(f"{label} selection contains an invalid material key")
    if len(set(keys)) != len(keys):
        raise PaperCorpusError(f"{label} selection contains duplicate material keys")


def prepare_ai42_5dic_corpus(
    material_manifest: MaterialManifest,
    run_inputs: Path | str,
    *,
    reference_paper_keys: tuple[tuple[str, str], ...],
    ai_placement_note_keys: tuple[tuple[str, str], ...],
) -> Ai42CorpusManifest:
    """Create the prepared corpus atomically below ``run_inputs``."""

    if not isinstance(material_manifest, MaterialManifest):
        raise PaperCorpusError("prepare requires a MaterialManifest")
    _verify_material_manifest(material_manifest)
    run_inputs_path = _ensure_run_inputs(run_inputs)
    _validate_selection(reference_paper_keys, "reference paper")
    _validate_selection(ai_placement_note_keys, "AI placement note")
    if len(reference_paper_keys) != len(APPROVED_REFERENCE_PAPERS):
        raise PaperCorpusError("reference paper selection must contain the complete approved corpus")
    if len(ai_placement_note_keys) != len(APPROVED_AI_PLACEMENT_NOTES):
        raise PaperCorpusError("AI placement note selection must contain the complete approved corpus")
    lookup = _material_lookup(material_manifest)
    papers = [
        _selected_material(lookup, run_inputs_path, key, kind="paper")
        for key in sorted(reference_paper_keys)
    ]
    notes = [
        _selected_material(lookup, run_inputs_path, key, kind="note")
        for key in sorted(ai_placement_note_keys)
    ]

    final_directory = run_inputs_path / "prepared_papers"
    final_manifest = run_inputs_path / CORPUS_MANIFEST_NAME
    if final_directory.exists() or os.path.lexists(final_directory):
        raise PaperCorpusError("refusing to overwrite existing prepared_papers")
    if final_manifest.exists() or os.path.lexists(final_manifest):
        raise PaperCorpusError(f"refusing to overwrite existing corpus manifest: '{final_manifest}'")

    temporary_root = Path(tempfile.mkdtemp(prefix=".ai42-corpus-", dir=str(run_inputs_path)))
    temporary_directory = temporary_root / "prepared_papers"
    published_directory = False
    published_manifest = False
    try:
        temporary_directory.mkdir()
        corpus_papers: list[CorpusPaper] = []
        for paper_index, material in enumerate(papers, start=1):
            staged_path = _run_relative(
                run_inputs_path, material.staged_relative_path, label="staged paper path"
            )
            output_parent = temporary_directory / f"paper-{paper_index:03d}"
            output_parent.mkdir()
            # _read_pdf_parts uses the parent name as the stable paper directory key.
            page_count, replacement_count, parts = _read_pdf_parts(
                material,
                staged_path,
                output_parent,
            )
            corpus_papers.append(
                CorpusPaper(
                    source_relative_path=material.source_relative_path,
                    staged_relative_path=material.staged_relative_path,
                    source_sha256=material.source_sha256,
                    page_count=page_count,
                    unicode_replacement_count=replacement_count,
                    text_parts=parts,
                )
            )

        corpus_notes = tuple(_write_note_reference(material, run_inputs_path) for material in notes)
        unsigned = {
            "version": 1,
            "target_id": _TARGET_ID,
            "papers": [paper.model_dump(mode="json") for paper in corpus_papers],
            "ai_placement_notes": [note.model_dump(mode="json") for note in corpus_notes],
            "material_manifest_checksum": material_manifest.manifest_checksum,
        }
        corpus_checksum = hashlib.sha256(_canonical_json(_corpus_identity_payload(unsigned))).hexdigest()
        manifest = Ai42CorpusManifest(
            papers=tuple(corpus_papers),
            ai_placement_notes=corpus_notes,
            material_manifest_checksum=material_manifest.manifest_checksum,
            corpus_checksum=corpus_checksum,
        )
        temporary_manifest = temporary_root / CORPUS_MANIFEST_NAME
        temporary_manifest.write_bytes(_canonical_json(manifest.model_dump(mode="json")) + b"\n")

        _reject_link_components(final_directory.parent)
        os.replace(temporary_directory, final_directory)
        published_directory = True
        os.replace(temporary_manifest, final_manifest)
        published_manifest = True
        return manifest
    except Exception as exc:
        cleanup_errors: list[str] = []
        if published_manifest:
            try:
                final_manifest.unlink()
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                cleanup_errors.append(f"remove published manifest: {cleanup_error}")
        if published_directory:
            try:
                shutil.rmtree(final_directory)
            except OSError as cleanup_error:
                cleanup_errors.append(f"remove published directory: {cleanup_error}")
        if cleanup_errors:
            raise PaperCorpusError(
                "corpus publication failed and rollback was incomplete: "
                + "; ".join(cleanup_errors)
            ) from exc
        if isinstance(exc, PaperCorpusError):
            raise
        raise PaperCorpusError("failed to publish prepared ai42.5Dic corpus") from exc
    finally:
        try:
            shutil.rmtree(temporary_root)
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            active_error = sys.exc_info()[1]
            if active_error is not None:
                raise PaperCorpusError(
                    "corpus operation failed and temporary cleanup was incomplete"
                ) from active_error
            raise PaperCorpusError(
                "corpus publication succeeded but temporary cleanup failed"
            ) from cleanup_error


def _validate_manifest_paths(manifest: Ai42CorpusManifest, manifest_path: Path, run_inputs: Path) -> None:
    try:
        manifest_path.resolve(strict=True).relative_to(run_inputs.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as exc:
        raise PaperCorpusError("corpus manifest must be inside run inputs") from exc
    if manifest_path != run_inputs / CORPUS_MANIFEST_NAME:
        raise PaperCorpusError("unexpected corpus manifest filename")
    if not manifest.papers or not manifest.ai_placement_notes:
        raise PaperCorpusError("corpus manifest must contain papers and AI placement notes")
    if len(manifest.papers) != len(APPROVED_REFERENCE_PAPERS):
        raise PaperCorpusError("corpus manifest must contain exactly the approved reference papers")
    if len(manifest.ai_placement_notes) != len(APPROVED_AI_PLACEMENT_NOTES):
        raise PaperCorpusError("corpus manifest must contain exactly the approved AI placement notes")
    paper_names = [Path(paper.source_relative_path).name for paper in manifest.papers]
    note_names = [Path(note.source_relative_path).name for note in manifest.ai_placement_notes]
    if set(paper_names) != APPROVED_REFERENCE_PAPERS or len(paper_names) != len(set(paper_names)):
        raise PaperCorpusError("corpus manifest reference papers do not match the approved set")
    if set(note_names) != APPROVED_AI_PLACEMENT_NOTES or len(note_names) != len(set(note_names)):
        raise PaperCorpusError("corpus manifest AI placement notes do not match the approved set")
    material_manifest_path = run_inputs / "material_manifest.json"
    try:
        material_manifest = read_material_manifest(material_manifest_path)
    except (OSError, ValueError) as exc:
        raise PaperCorpusError("unable to verify the staged material manifest") from exc
    if material_manifest.manifest_checksum != manifest.material_manifest_checksum:
        raise PaperCorpusError("material manifest checksum is not bound to the corpus")
    material_by_staged_path = {
        material.staged_relative_path: material
        for material in material_manifest.materials
    }
    seen: set[str] = set()
    for paper in manifest.papers:
        _safe_relative(paper.source_relative_path, label="paper source path")
        staged_material = material_by_staged_path.get(paper.staged_relative_path)
        if staged_material is None or (
            staged_material.source_relative_path != paper.source_relative_path
            or staged_material.source_sha256 != paper.source_sha256
            or staged_material.source_sha256 != staged_material.staged_sha256
            or staged_material.source_file_kind != "pdf"
            or staged_material.source_category != "reference_paper"
        ):
            raise PaperCorpusError("paper provenance is not bound to material_manifest.json")
        if not _approved_source_path(staged_material, kind="paper"):
            raise PaperCorpusError("paper provenance is outside the approved ai42.5Dic corpus")
        staged = _run_relative(run_inputs, paper.staged_relative_path, label="staged paper path")
        if staged.suffix.lower() != ".pdf" or not staged.is_file():
            raise PaperCorpusError(f"staged paper is missing or is not PDF: '{staged}'")
        if _sha256_file(staged) != paper.source_sha256:
            raise PaperCorpusError(f"staged paper checksum mismatch: '{staged}'")
        for part in paper.text_parts:
            if part.relative_path in seen:
                raise PaperCorpusError(f"duplicate corpus artifact path: '{part.relative_path}'")
            seen.add(part.relative_path)
            part_path = _run_relative(run_inputs, part.relative_path, label="text part path")
            try:
                actual_size = part_path.stat().st_size
            except OSError as exc:
                raise PaperCorpusError(f"invalid corpus text part: '{part_path}'") from exc
            if (
                PurePosixPath(part.relative_path).parts[0] != "prepared_papers"
                or part_path.suffix.lower() != ".txt"
                or actual_size != part.byte_size
                or part.byte_size > MAX_TEXT_PART_BYTES
            ):
                raise PaperCorpusError(f"corpus text part size/content mismatch: '{part_path}'")
            try:
                data = part_path.read_bytes()
                data.decode("utf-8")
            except (OSError, UnicodeError) as exc:
                raise PaperCorpusError(f"invalid corpus text part: '{part_path}'") from exc
            if b"\x00" in data or len(data) != part.byte_size:
                raise PaperCorpusError(f"corpus text part contains NUL: '{part_path}'")
            if hashlib.sha256(data).hexdigest() != part.sha256:
                raise PaperCorpusError(f"corpus text part checksum mismatch: '{part_path}'")
            if part.page_start > part.page_end or part.page_end > paper.page_count:
                raise PaperCorpusError(f"invalid page range in corpus text part: '{part.relative_path}'")
    for note in manifest.ai_placement_notes:
        _safe_relative(note.source_relative_path, label="note source path")
        staged_material = material_by_staged_path.get(note.staged_relative_path)
        if staged_material is None or (
            staged_material.source_relative_path != note.source_relative_path
            or staged_material.source_sha256 != note.source_sha256
            or staged_material.source_sha256 != staged_material.staged_sha256
            or staged_material.source_file_kind != "markdown"
            or staged_material.source_category != "ai_placement_note"
        ):
            raise PaperCorpusError("note provenance is not bound to material_manifest.json")
        if not _approved_source_path(staged_material, kind="note"):
            raise PaperCorpusError("note provenance is outside the approved ai42.5Dic corpus")
        staged = _run_relative(run_inputs, note.staged_relative_path, label="staged note path")
        if staged.suffix.lower() not in {".md", ".markdown"} or not staged.is_file():
            raise PaperCorpusError(f"staged AI placement note is missing or not Markdown: '{staged}'")
        try:
            data = staged.read_bytes()
            data.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise PaperCorpusError(f"invalid AI placement note: '{staged}'") from exc
        if not data or b"\x00" in data:
            raise PaperCorpusError(f"AI placement note is empty or contains NUL: '{staged}'")
        if _sha256_file(staged) != note.source_sha256:
            raise PaperCorpusError(f"AI placement note checksum mismatch: '{staged}'")


def read_ai42_5dic_corpus_manifest(
    path: Path | str,
    run_inputs: Path | str,
) -> Ai42CorpusManifest:
    """Read and fully verify a prepared corpus manifest and its artifacts."""

    run_inputs_path = _existing_run_inputs(run_inputs)
    manifest_path = _absolute(path)
    try:
        _reject_link_components(run_inputs_path)
        _reject_link_components(manifest_path.parent)
    except MaterialBoundaryError as exc:
        raise PaperCorpusError("corpus manifest path contains a link or reparse point") from exc
    if not manifest_path.is_file() or _is_link_or_reparse(manifest_path):
        raise PaperCorpusError(f"corpus manifest is not a regular file: '{manifest_path}'")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = Ai42CorpusManifest.model_validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise PaperCorpusError(f"invalid corpus manifest: '{manifest_path}'") from exc
    expected = hashlib.sha256(_canonical_json(_corpus_identity_payload(manifest))).hexdigest()
    if expected != manifest.corpus_checksum:
        raise PaperCorpusError("corpus manifest checksum mismatch")
    _validate_manifest_paths(manifest, manifest_path, run_inputs_path)
    return manifest


def reader_artifacts_from_corpus(path: Path | str, run_inputs: Path | str) -> tuple[Path, ...]:
    """Return only verified manifest, text parts, and Markdown note paths."""

    manifest_path = _absolute(path)
    run_inputs_path = _absolute(run_inputs)
    manifest = read_ai42_5dic_corpus_manifest(manifest_path, run_inputs_path)
    artifacts: list[Path] = [manifest_path]
    for paper in manifest.papers:
        artifacts.extend(
            _run_relative(run_inputs_path, part.relative_path, label="text part path")
            for part in paper.text_parts
        )
    artifacts.extend(
        _run_relative(run_inputs_path, note.staged_relative_path, label="staged note path")
        for note in manifest.ai_placement_notes
    )
    return tuple(artifacts)


__all__ = [
    "Ai42CorpusManifest",
    "CORPUS_MANIFEST_NAME",
    "CorpusNote",
    "CorpusPaper",
    "CorpusTextPart",
    "MAX_TEXT_PART_BYTES",
    "PaperCorpusError",
    "prepare_ai42_5dic_corpus",
    "read_ai42_5dic_corpus_manifest",
    "reader_artifacts_from_corpus",
]
