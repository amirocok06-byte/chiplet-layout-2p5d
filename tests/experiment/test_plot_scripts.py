import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np


PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from plot_algorithm_effects import plot_algorithm_effects  # noqa: E402
from plot_snapshot_maps import plot_snapshot_maps  # noqa: E402
from plot_state_flow import plot_state_flow  # noqa: E402


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(7)
    snapshot = rng.normal(size=(3, 256, 256)).astype(np.float32)
    flow = rng.normal(size=(4, 3, 256, 256)).astype(np.float32)
    after = rng.normal(size=(3, 3, 256, 256)).astype(np.float32)
    snapshot_path = tmp_path / "snapshot.npz"
    flow_path = tmp_path / "flow.npz"
    effects_path = tmp_path / "effects.npz"
    np.savez_compressed(snapshot_path, maps=snapshot)
    np.savez_compressed(flow_path, maps=flow)
    np.savez_compressed(effects_path, before_maps=snapshot, after_maps=after)
    return snapshot_path, flow_path, effects_path


def test_scheme_a_snapshot_preserves_raw_grid_and_high_resolution(tmp_path):
    snapshot, _, _ = _fixture(tmp_path)
    output = tmp_path / "snapshot.png"

    plot_snapshot_maps(snapshot, output)

    image = mpimg.imread(output)
    assert image.shape[1] >= 3000
    assert image.shape[0] >= 900


def test_scheme_a_state_flow_preserves_raw_grid_and_high_resolution(tmp_path):
    _, flow, _ = _fixture(tmp_path)
    output = tmp_path / "flow.png"

    plot_state_flow(flow, output, max_frames=4)

    image = mpimg.imread(output)
    assert image.shape[1] >= 3000
    assert image.shape[0] >= 1800


def test_scheme_a_effects_preserves_raw_grid_and_high_resolution(tmp_path):
    _, _, effects = _fixture(tmp_path)
    output = tmp_path / "effects.png"

    plot_algorithm_effects(effects, output, ["sa", "analytical", "rl"])

    image = mpimg.imread(output)
    assert image.shape[1] >= 3000
    assert image.shape[0] >= 2500
