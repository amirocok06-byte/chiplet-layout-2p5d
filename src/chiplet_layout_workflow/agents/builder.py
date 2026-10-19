from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from .. import builder_prototype as prototype_template
from ..artifacts.checksums import checksum_tree, sha256_file
from ..handoff_receipts import InvalidBuilderHandoff, validate_builder_approval_receipt
from ..local_materials import MaterialManifest, read_material_manifest
from .base import AgentProtocolError, AgentResult, BaseAgent, RunContext


class BuilderAgent(BaseAgent):
    name = "builder"
    role = "builder"
    stage = "building"
    required_outputs = (
        "implementation_plan.md",
        "directory_layout.md",
        "implementation_report.md",
        "builder_result.json",
        "source/ (approved source worktree only)",
        "agent_status.json",
        "builder_output_manifest.json",
    )

    def validate_inputs(self, context: RunContext) -> None:
        revised_plan = context.run.feasibility / "revised_plan.md"
        if not revised_plan.is_file():
            raise ValueError("builder requires an approved feasibility decision and revised plan")
        try:
            validate_builder_approval_receipt(context)
        except InvalidBuilderHandoff as exc:
            raise ValueError(str(exc)) from exc
        context.run.assert_inside(context.run.source / "implementation_report.md")

    def run(self, context: RunContext) -> AgentResult:
        validation_context = self._unbound_context(context)
        # Validate receipt and upstream manifests before touching run.inputs.
        self.validate_inputs(validation_context)
        try:
            receipt = validate_builder_approval_receipt(validation_context)
        except InvalidBuilderHandoff as exc:
            raise ValueError(str(exc)) from exc

        staged_case, material_manifest = self._discover_single_case(validation_context)
        building_context = validation_context.for_stage(self.stage)
        return self._build_and_publish(building_context, staged_case, material_manifest, receipt)

    def _unbound_context(self, context: RunContext) -> RunContext:
        if context.stage is not None and context.stage != self.stage:
            context.for_stage(self.stage)
        return RunContext(
            run=context.run,
            current_target=context.current_target,
            provider=self.provider,
            prompt_renderer=self.prompt_renderer,
            input_paths=context.input_paths,
            source_worktree=context.source_worktree,
            git_adapter=context.git_adapter,
            benchmark_executor=context.benchmark_executor,
            metadata=context.metadata,
        )

    def _discover_single_case(
        self, context: RunContext
    ) -> tuple[dict[str, Path], MaterialManifest]:
        manifest_path = context.run.inputs / "material_manifest.json"
        context.run.assert_inside(manifest_path)
        try:
            manifest = read_material_manifest(manifest_path)
        except (OSError, ValueError) as exc:
            raise ValueError("builder requires a valid staged material manifest") from exc

        staged: dict[str, Path] = {}
        for material in manifest.materials:
            relative = PurePosixPath(material.staged_relative_path)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in material.staged_relative_path
            ):
                raise ValueError("builder material manifest contains an unsafe staged path")
            candidate = context.run.inputs.joinpath(*relative.parts)
            try:
                candidate = context._validate_explicit_input_path(candidate)
            except Exception as exc:
                raise ValueError("builder material manifest contains an unsafe staged file") from exc
            if str(candidate) != material.staged_path:
                raise ValueError("builder material manifest staged path does not match run.inputs")
            if sha256_file(candidate) != material.staged_sha256:
                raise ValueError(f"staged material checksum mismatch: {relative.as_posix()}")
            staged[relative.as_posix()] = candidate

        block_paths = sorted(
            path for relative, path in staged.items() if relative.casefold().endswith(".blocks")
        )
        if len(block_paths) != 1:
            raise ValueError(
                "builder requires exactly one staged Bookshelf case; "
                f"found {len(block_paths)} .blocks files"
            )
        blocks = block_paths[0]
        stem = blocks.with_suffix("")
        required = {
            "blocks": blocks,
            "nets": stem.with_suffix(".nets"),
            "pl": stem.with_suffix(".pl"),
        }
        missing = [kind for kind, path in required.items() if path not in staged.values()]
        if missing:
            raise ValueError(
                "builder requires one complete staged Bookshelf case; missing "
                + ", ".join(sorted(missing))
            )
        power = stem.with_suffix(".power")
        case = dict(required)
        if power in staged.values():
            case["power"] = power
        return case, manifest

    def _build_and_publish(
        self,
        context: RunContext,
        case: dict[str, Path],
        material_manifest: MaterialManifest,
        receipt: Any,
    ) -> AgentResult:
        stage_root = context.artifact_root(self.stage)
        temporary_root = Path(tempfile.mkdtemp(prefix=".builder-", dir=str(context.run.path)))
        context.run.assert_inside(temporary_root)
        try:
            payload = self._write_candidate(
                temporary_root, context, case, material_manifest, receipt
            )
            self._publish_candidate(
                stage_root,
                temporary_root,
                validate_published=lambda: self._validate_published_result(
                    payload["result"], context
                ),
            )
        except Exception:
            if temporary_root.exists():
                shutil.rmtree(temporary_root, ignore_errors=True)
            raise

        return payload["result"]

    def _write_candidate(
        self,
        root: Path,
        context: RunContext,
        case: dict[str, Path],
        material_manifest: MaterialManifest,
        receipt: Any,
    ) -> dict[str, Any]:
        source_package = root / "source" / "chiplet_layout_builder"
        source_package.mkdir(parents=True)
        generated_source = Path(prototype_template.__file__).read_text(encoding="utf-8")
        (source_package / "builder_prototype.py").write_text(
            "# Generated from the package's internal dependency-free template.\n"
            + generated_source,
            encoding="utf-8",
        )
        (source_package / "__init__.py").write_text(
            "from .builder_prototype import *  # noqa: F401,F403\n",
            encoding="utf-8",
        )

        generated_module = self._load_generated_module(source_package / "builder_prototype.py")
        design = generated_module.parse_bookshelf(
            case["blocks"], case["nets"], case["pl"], case.get("power")
        )
        package_width = sum(chiplet.width for chiplet in design.chiplets)
        package_height = max(chiplet.height for chiplet in design.chiplets)
        smoke = generated_module.run_baseline(
            design,
            package_width=package_width,
            package_height=package_height,
            thermal_input=generated_module.PowerThermalInput(power_by_chiplet=design.power),
        )
        if not smoke["checks"]["package_boundary"] or not smoke["checks"]["non_overlap"]:
            raise AgentProtocolError("generated builder smoke case is not legal")

        case_name = case["blocks"].stem
        builder_result = {
            "agent": self.name,
            "approved": True,
            "approval_receipt_required": True,
            "approval_receipt_checksum": receipt.receipt_checksum,
            "run_id": context.run.run_id,
            "target": context.current_target.model_dump(mode="json", by_alias=True),
            "case": {"name": case_name, "blocks": context.run_relative(case["blocks"])},
            "smoke": {
                "case": case_name,
                "metrics": smoke["metrics"],
                "checks": smoke["checks"],
            },
            "deferred": [
                "thermal solve",
                "complete benchmark suite",
                "external solver and cloud/SSH execution",
            ],
        }
        self._write_text(
            root / "implementation_plan.md",
            """# Builder implementation plan

1. Validate the explicit Reader/Retriever handoff and current approved target.
2. Select exactly one staged Bookshelf case from `run.inputs`.
3. Generate an independent source tree and run its deterministic smoke case.
4. Publish all outputs atomically after validation.
""",
        )
        self._write_text(
            root / "directory_layout.md",
            """# Directory layout

- `source/chiplet_layout_builder/`: generated, dependency-free builder source
- `implementation_plan.md`: bounded construction plan
- `implementation_report.md`: evidence and engineering inference record
- `builder_result.json`: completion metadata and one-case smoke result
- `agent_status.json`: execution status
- `builder_output_manifest.json`: checksums and publication identity
""",
        )
        self._write_text(
            root / "implementation_report.md",
            """# Builder implementation report

## Evidence-backed decisions

- The explicit BuilderApprovalReceipt and Reader/Retriever manifests bind this run and target.
- The staged material manifest binds the one Bookshelf case consumed by the smoke run.
- The approved handoff limits this stage to an offline, one-case geometry smoke run.

## Engineering inference

- Because no package dimensions are present in the scoped case, package width is the sum of chiplet widths and package height is the maximum chiplet height.
- The generated baseline implements Bookshelf parsing, sequential rectangle placement, HPWL, package-boundary checks, and overlap checks.
- `builder_prototype.py` is generated from this package's internal template and is independently importable; no external source tree is copied.

## External methods and limits

TAP, ATPlace, MaskPlace, and RLPlanner are metadata-only references. Thermal values are inputs/interfaces; no thermal solver, complete benchmark, external solver, cloud job, SSH job, or formal comparison is executed. Those integrations remain deferred to later gated stages.
""",
        )
        self._write_json(root / "builder_result.json", builder_result)
        self._write_json(
            root / "agent_status.json",
            {
                "agent": self.name,
                "stage": self.stage,
                "status": "completed",
                "approved": True,
                "execution": "generated_source_smoke",
                "target": context.current_target.model_dump(mode="json", by_alias=True),
                "required_outputs": list(self.required_outputs),
                "material_manifest_checksum": material_manifest.manifest_checksum,
                "approval_receipt_checksum": receipt.receipt_checksum,
            },
        )

        output_names = [
            "implementation_plan.md",
            "directory_layout.md",
            "implementation_report.md",
            "builder_result.json",
            "agent_status.json",
            "source/chiplet_layout_builder/__init__.py",
            "source/chiplet_layout_builder/builder_prototype.py",
        ]
        output_checksums = {
            name: sha256_file(root / Path(*PurePosixPath(name).parts)) for name in output_names
        }
        manifest = {
            "agent": self.name,
            "complete": True,
            "approved": True,
            "run_id": context.run.run_id,
            "stage": self.stage,
            "status": "completed",
            "target": context.current_target.model_dump(mode="json", by_alias=True),
            "material_manifest_checksum": material_manifest.manifest_checksum,
            "approval_receipt_checksum": receipt.receipt_checksum,
            "case": context.run_relative(case["blocks"]),
            "outputs": output_names + ["builder_output_manifest.json"],
            "output_checksums": output_checksums,
            "source_tree_checksum": checksum_tree(root / "source"),
            "smoke": builder_result["smoke"],
        }
        self._write_json(root / "builder_output_manifest.json", manifest)
        self._validate_candidate(root, manifest)
        created_files = tuple(
            path.relative_to(root).as_posix()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        )
        result = AgentResult(
            stage=self.stage,
            status="completed",
            created_files=created_files,
            checksums=checksum_tree(root),
            warnings=(
                "thermal solve, complete benchmark, external methods, and cloud/SSH execution are deferred",
            ),
            human_review_questions=(
                "builder_result.approved is completion metadata; the approval receipt remains authoritative",
            ),
        )
        return {
            "warnings": (
                "thermal solve, complete benchmark, external methods, and cloud/SSH execution are deferred",
            ),
            "result": result,
        }

    @staticmethod
    def _validate_published_result(result: AgentResult, context: RunContext) -> None:
        root = context.artifact_root("building")
        actual_checksums = checksum_tree(root)
        for relative in result.created_files:
            candidate = root / relative
            try:
                candidate.resolve(strict=False).relative_to(root.resolve(strict=True))
            except (OSError, RuntimeError, ValueError) as exc:
                raise AgentProtocolError(
                    f"created file is outside Builder stage: {relative!r}"
                ) from exc
        if dict(result.checksums) != actual_checksums:
            raise AgentProtocolError("published Builder checksums do not match the stage")

    @staticmethod
    def _validate_candidate(root: Path, expected_manifest: dict[str, Any]) -> None:
        try:
            manifest = json.loads(
                (root / "builder_output_manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AgentProtocolError("builder output manifest cannot be re-read") from exc
        if manifest != expected_manifest:
            raise AgentProtocolError("builder output manifest changed before publication")
        output_checksums = manifest.get("output_checksums")
        if not isinstance(output_checksums, dict):
            raise AgentProtocolError("builder output manifest has no output checksums")
        for name, expected in output_checksums.items():
            path = root / Path(*PurePosixPath(name).parts)
            if not path.is_file() or sha256_file(path) != expected:
                raise AgentProtocolError(f"builder output checksum mismatch: {name}")
        if checksum_tree(root / "source") != manifest.get("source_tree_checksum"):
            raise AgentProtocolError("builder source tree checksum mismatch")

    @staticmethod
    def _load_generated_module(path: Path) -> Any:
        module_name = "_chiplet_layout_generated_builder"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise AgentProtocolError("generated builder source cannot be imported")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise AgentProtocolError("generated builder source failed to import") from exc
        finally:
            sys.dont_write_bytecode = previous
            sys.modules.pop(module_name, None)
        return module

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _publish_candidate(
        final_root: Path,
        temporary_root: Path,
        *,
        validate_published: Any,
    ) -> None:
        backup_root = final_root.parent / f".builder-backup-{temporary_root.name}"
        os.replace(final_root, backup_root)
        try:
            os.replace(temporary_root, final_root)
            try:
                validate_published()
            except Exception:
                failed_root = final_root.parent / f".builder-failed-{temporary_root.name}"
                os.replace(final_root, failed_root)
                os.replace(backup_root, final_root)
                shutil.rmtree(failed_root)
                raise
        except Exception:
            if backup_root.exists() and not final_root.exists():
                os.replace(backup_root, final_root)
            raise
        try:
            shutil.rmtree(backup_root)
        except Exception as exc:
            raise AgentProtocolError(
                "builder published successfully but old output cleanup failed"
            ) from exc
