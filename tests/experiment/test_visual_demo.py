import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from chiplet_layout_experiment.visual_demo import generate_synthetic_smoke_bundle


def test_generate_synthetic_smoke_bundle_writes_traceable_three_map_artifacts(tmp_path):
    manifest = generate_synthetic_smoke_bundle(
        tmp_path,
        grid_size=32,
        source_revision="test-revision",
    )

    assert manifest["synthetic"] is True
    assert manifest["purpose"] == "protocol-smoke-only"
    assert manifest["labels"] == ["sa", "analytical", "rl-compatible"]
    assert manifest["order_source"] == "explicit-priority-precedence-metadata"
    assert manifest["source_revision"] == "test-revision"

    snapshot = np.load(tmp_path / "snapshot_maps.npz", allow_pickle=False)["maps"]
    state_flow = np.load(tmp_path / "state_flow.npz", allow_pickle=False)["maps"]
    effects = np.load(tmp_path / "algorithm_effects.npz", allow_pickle=False)
    saved_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert snapshot.shape == (3, 32, 32)
    assert state_flow.shape == (4, 3, 32, 32)
    assert effects["before_maps"].shape == (3, 32, 32)
    assert effects["after_maps"].shape == (3, 3, 32, 32)
    assert saved_manifest == manifest


def test_visual_demo_module_entrypoint_generates_bundle(tmp_path):
    project_root = Path(__file__).parents[2]
    environment = {**os.environ, "PYTHONPATH": str(project_root / "src")}
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chiplet_layout_experiment.visual_demo",
            str(tmp_path),
            "--grid-size",
            "16",
            "--source-revision",
            "cli-test",
        ],
        check=False,
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["source_revision"] == "cli-test"
