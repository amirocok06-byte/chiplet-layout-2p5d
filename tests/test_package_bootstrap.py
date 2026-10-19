import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import zipfile

from chiplet_layout_workflow import __version__, PROJECT_TITLE


def _json_files(root):
    return {
        path.name: path
        for path in root.iterdir()
        if path.is_file() and path.name.endswith(".json")
    }


def _assert_schema_files_match(expected_root, actual_root):
    expected = _json_files(expected_root)
    actual = _json_files(actual_root)
    assert set(actual) == set(expected)
    for schema_name in sorted(expected):
        assert actual[schema_name].read_bytes() == expected[schema_name].read_bytes()


def _valid_target_payload():
    return {
        "id": "rectilinear-soft-module",
        "display_name": "rectilinear soft module",
        "revision": 0,
        "source": "user_seed",
        "evidence_checksum": "seed",
        "approved": False,
    }


def _valid_run_payload():
    return {"topic": "layout", "data_root": "D:/runs"}


def _valid_manifest_payload():
    return {
        "run_id": "run-1",
        "checksums": {"input": "abc"},
        "source_commit": "commit-1",
        "config_checksum": "config-1",
        "metric_schema_checksum": "metric-1",
        "executor": "local",
        "model_provider": "mock",
        "status": "pending",
        "research_target": _valid_target_payload(),
    }


def _validation_smoke_script():
    return textwrap.dedent(
        """
        import json
        import os
        from importlib import resources
        from pathlib import Path
        import tempfile

        import chiplet_layout_workflow
        from chiplet_layout_workflow import (
            validate_manifest_payload,
            validate_run_payload,
            validate_target_payload,
        )
        from chiplet_layout_workflow.config import load_config

        package_path = Path(chiplet_layout_workflow.__file__).resolve()
        assert package_path.is_relative_to(Path(os.environ["EXPECTED_PACKAGE_ROOT"]).resolve())
        repository_root = Path(os.environ["EXPECTED_SCHEMA_ROOT"])
        package_root = resources.files("chiplet_layout_workflow.schema_assets")
        repository_schemas = {
            path.name
            for path in repository_root.iterdir()
            if path.is_file() and path.name.endswith(".json")
        }
        package_schemas = {
            path.name
            for path in package_root.iterdir()
            if path.is_file() and path.name.endswith(".json")
        }
        assert package_schemas == repository_schemas
        for schema_name in sorted(repository_schemas):
            assert package_root.joinpath(schema_name).read_bytes() == (
                repository_root / schema_name
            ).read_bytes()

        target = json.loads(os.environ["VALID_TARGET_JSON"])
        run = validate_run_payload(json.loads(os.environ["VALID_RUN_JSON"]))
        validate_target_payload(target)
        validate_manifest_payload(json.loads(os.environ["VALID_MANIFEST_JSON"]))
        assert run.topic == "layout"
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(os.environ["VALID_CONFIG_YAML"], encoding="utf-8")
            assert load_config(config_path).topic == "layout"
        """
    )


def _validation_smoke_env(
    base_env,
    *,
    expected_package_root,
    pythonpath,
    schema_root,
):
    return base_env | {
        "EXPECTED_PACKAGE_ROOT": str(expected_package_root),
        "EXPECTED_SCHEMA_ROOT": str(schema_root),
        "PYTHONPATH": str(pythonpath),
        "PYTHONNOUSERSITE": "1",
        "VALID_TARGET_JSON": json.dumps(_valid_target_payload()),
        "VALID_RUN_JSON": json.dumps(_valid_run_payload()),
        "VALID_MANIFEST_JSON": json.dumps(_valid_manifest_payload()),
        "VALID_CONFIG_YAML": "topic: layout\ndata_root: D:/runs\n",
    }


def test_package_exposes_version():
    assert __version__ == "0.1.0"


def test_package_exposes_project_title():
    assert PROJECT_TITLE == "面向2.5D系统集成的芯粒布局方法"


def test_packaged_schemas_match_repository_schemas():
    from importlib import resources

    package_root = resources.files("chiplet_layout_workflow.schema_assets")
    repository_root = Path(__file__).parents[1] / "schemas"

    _assert_schema_files_match(repository_root, package_root)


def test_wheel_contains_schemas_and_works_from_isolated_install(tmp_path):
    project_root = Path(__file__).parents[1]
    wheel_dir = tmp_path / "wheel"
    install_dir = tmp_path / "install"
    wheel_dir.mkdir()
    install_dir.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
        }
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            str(project_root),
        ],
        check=True,
        cwd=tmp_path,
        env=env,
    )
    wheel_path = next(wheel_dir.glob("*.whl"))

    wheel_extract_dir = tmp_path / "wheel-extract"
    wheel_extract_dir.mkdir()
    with zipfile.ZipFile(wheel_path) as wheel:
        wheel.extractall(wheel_extract_dir)
    _assert_schema_files_match(
        project_root / "schemas",
        wheel_extract_dir / "chiplet_layout_workflow" / "schema_assets",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(install_dir),
            str(wheel_path),
        ],
        check=True,
        cwd=tmp_path,
        env=env,
    )

    child_env = _validation_smoke_env(
        env,
        expected_package_root=install_dir,
        pythonpath=install_dir,
        schema_root=project_root / "schemas",
    )
    subprocess.run(
        [sys.executable, "-c", _validation_smoke_script()],
        check=True,
        cwd=tmp_path,
        env=child_env,
    )


def test_source_checkout_smoke_is_independent_of_cwd(tmp_path):
    project_root = Path(__file__).parents[1]
    source_package_root = project_root / "src" / "chiplet_layout_workflow"
    env = _validation_smoke_env(
        os.environ.copy(),
        expected_package_root=source_package_root,
        pythonpath=project_root / "src",
        schema_root=project_root / "schemas",
    )

    subprocess.run(
        [sys.executable, "-c", _validation_smoke_script()],
        check=True,
        cwd=tmp_path,
        env=env,
    )
