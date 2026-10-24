import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import PowerNorm, SymLogNorm


PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import plot_grid  # noqa: E402
import plot_algorithm_effects as effects_plot  # noqa: E402
import plot_snapshot_maps as snapshot_plot  # noqa: E402
import plot_state_flow as flow_plot  # noqa: E402


@pytest.mark.parametrize("channel", ["thermal", "hpwl"])
def test_zero_background_is_masked_without_mutating_raw_pixels(channel):
    values = np.array([[0.0, 0.001], [0.1, 1.0]])
    original = values.copy()
    values.flags.writeable = False
    fig, axis = plt.subplots()
    try:
        image = plot_grid.draw_map(axis, values, channel, limits=(0.0, 1.0))
        np.testing.assert_array_equal(np.ma.getmaskarray(image.get_array()), values == 0)
        np.testing.assert_array_equal(image.get_array().data, original)
        np.testing.assert_array_equal(values, original)
        assert image.get_interpolation() == "nearest"
        assert min(image.cmap.get_bad()[:3]) > 0.85
        assert image.norm(0.0) == 0.0
        assert image.norm(0.01) > 0.01
        assert isinstance(image.norm, PowerNorm if channel == "thermal" else SymLogNorm)
    finally:
        plt.close(fig)


def test_priority_zero_is_visible_and_only_sentinel_is_masked():
    values = np.array([[-1.0, 0.0], [0.5, 1.0]])
    original = values.copy()
    fig, axis = plt.subplots()
    try:
        image = plot_grid.draw_map(axis, values, "order")
        np.testing.assert_array_equal(np.ma.getmaskarray(image.get_array()), values == -1)
        np.testing.assert_array_equal(image.get_array().data, original)
        np.testing.assert_array_equal(values, original)
        assert image.cmap.name.startswith("viridis")
        assert image.cmap.N <= 32
        assert (image.norm.vmin, image.norm.vmax) == (0.0, 1.0)
        assert min(image.cmap.get_bad()[:3]) > 0.85
    finally:
        plt.close(fig)


def test_hpwl_display_masks_accumulation_residue_but_keeps_small_real_signal():
    raw = np.array([[0., 5e-15], [1e-8, 1.]])
    original = raw.copy()
    fig, ax = plt.subplots()
    try:
        image = plot_grid.draw_map(ax, raw, "hpwl", limits=(0., 1.))
        np.testing.assert_array_equal(
            np.ma.getmaskarray(image.get_array()), [[True, True], [False, False]])
        np.testing.assert_array_equal(raw, original)
    finally:
        plt.close(fig)


def test_global_limits_accept_already_sliced_arrays_of_arbitrary_shape():
    values = np.zeros((2, 4, 5, 7))
    values[-1, -1, -1, -1] = 20.0
    original = values.copy()
    assert plot_grid.compute_map_limits(values, "thermal") == (0.0, 20.0)
    assert plot_grid.compute_map_limits(values, "hpwl") == (0.0, 20.0)
    assert plot_grid.compute_map_limits(np.full((2, 4, 5), -1.0), "order") == (0.0, 1.0)
    np.testing.assert_array_equal(values, original)


def test_macro_grid_has_sixteen_cells_with_only_five_coordinate_labels():
    fig, axis = plt.subplots()
    try:
        plot_grid.add_raw_pixel_grid(axis, height=256, width=256)
        assert [t.get_text() for t in axis.get_xticklabels()] == ["0", "64", "128", "192", "256"]
        assert [t.get_text() for t in axis.get_yticklabels()] == ["0", "64", "128", "192", "256"]
        ticks = np.sort(np.r_[axis.get_xticks(), axis.get_xticks(minor=True)])
        np.testing.assert_array_equal(ticks, np.arange(17) * 16 - 0.5)
        assert all(t.get_fontsize() >= 6 for t in axis.get_xticklabels())
    finally:
        plt.close(fig)


@pytest.fixture
def captured_export(monkeypatch):
    captured = []

    def capture(fig, output_path, dpi=300, axes=None):
        captured.append((fig, list(axes), dpi))

    for module in (snapshot_plot, flow_plot, effects_plot):
        monkeypatch.setattr(module, "export_figure", capture)
    return captured


def _map_fixture():
    maps = np.zeros((3, 8, 8), dtype=np.float64)
    maps[0].fill(-1)
    maps[0, 1:4, 1:4] = 0
    maps[0, 4:6, 4:6] = 1
    maps[1, 1:4, 1:4] = 2.0
    maps[2, 1:6, 1:6] = 0.001
    return maps


def test_snapshot_accepts_custom_title_labels_and_global_limits(tmp_path, captured_export):
    maps = _map_fixture()
    path = tmp_path / "maps.npz"
    np.savez(path, maps=maps)
    limits = [(0.0, 1.0), (0.0, 20.0), (0.0, 0.1)]
    names = ("Priority", "Power density", "Net HPWL")
    snapshot_plot.plot_snapshot_maps(
        path, tmp_path / "snapshot.png", title="Checkpoint 4", channel_labels=names, limits=limits
    )
    fig, axes, dpi = captured_export[0]
    assert fig._suptitle.get_text() == "Checkpoint 4"
    assert len(axes) == 3
    assert len(fig.axes) == 6
    assert dpi == 300
    for index, axis in enumerate(axes):
        image = axis.images[0]
        assert names[index] in axis.get_title()
        assert (image.norm.vmin, image.norm.vmax) == limits[index]
        np.testing.assert_array_equal(image.get_array().data, maps[index])
    with np.load(path) as data:
        np.testing.assert_array_equal(data["maps"], maps)


def test_state_flow_scales_include_hidden_frames_and_colorbars_are_shared(tmp_path, captured_export):
    maps = np.stack([_map_fixture()] * 3)
    maps[1, 1, 2, 2] = 40
    maps[1, 2, 2, 2] = 0.5
    path = tmp_path / "flow.npz"
    np.savez(path, maps=maps)
    flow_plot.plot_state_flow(
        path,
        tmp_path / "flow.png",
        max_frames=2,
        frame_labels=["Initial", "Hidden maximum", "Final"],
        title="Placement trajectory",
    )
    fig, axes, _ = captured_export[0]
    assert fig._suptitle.get_text() == "Placement trajectory"
    assert len(axes) == 6
    assert len(fig.axes) == 9  # Six maps, exactly one colorbar per channel.
    for row, maximum in enumerate((1.0, 40.0, 0.5)):
        first, last = axes[row * 2 : row * 2 + 2]
        assert first.images[0].norm is last.images[0].norm
        assert first.images[0].norm.vmax == maximum
        assert "Initial" in first.get_title()
        assert "Final" in last.get_title()
        np.testing.assert_array_equal(first.images[0].get_array().data, maps[0, row])
        np.testing.assert_array_equal(last.images[0].get_array().data, maps[2, row])


def test_state_flow_respects_external_scales_and_custom_channel_labels(tmp_path, captured_export):
    path = tmp_path / "flow.npz"
    np.savez(path, maps=np.stack([_map_fixture()]))
    limits = [(0.0, 1.0), (0.0, 80.0), (0.0, 0.2)]
    names = ("Priority", "Power density", "Net HPWL")
    flow_plot.plot_state_flow(
        path, tmp_path / "flow.png", title=None, channel_labels=names, limits=limits
    )
    fig, axes, _ = captured_export[0]
    assert fig._suptitle is None
    for row, axis in enumerate(axes):
        assert names[row] in axis.get_title()
        assert (axis.images[0].norm.vmin, axis.images[0].norm.vmax) == limits[row]


def test_algorithm_deltas_share_symmetric_channel_scales_across_rows(tmp_path, captured_export):
    before = _map_fixture()
    change = np.ones_like(before)
    change[:, 0, 0] = -1
    after = np.stack([before + change, before + 10 * change])
    path = tmp_path / "effects.npz"
    np.savez(path, before_maps=before, after_maps=after)
    effects_plot.plot_algorithm_effects(path, tmp_path / "effects.png", ["Small", "Large"])
    fig, axes, _ = captured_export[0]
    assert len(axes) == 6
    assert len(fig.axes) == 9  # Exactly one colorbar per delta channel.
    for col in range(3):
        small = axes[col].images[0]
        large = axes[col + 3].images[0]
        assert small.norm is large.norm
        assert (small.norm.vmin, small.norm.vmax) == (-10.0, 10.0)
        assert small.norm(0.0) == 0.5
        assert small.norm(1.0) < small.norm(10.0)
        np.testing.assert_array_equal(small.get_array(), after[0, col] - before[col])
    with np.load(path) as data:
        np.testing.assert_array_equal(data["before_maps"], before)
        np.testing.assert_array_equal(data["after_maps"], after)


@pytest.mark.parametrize("channel", ["thermal", "hpwl"])
def test_all_zero_channel_has_a_finite_non_degenerate_scale(channel):
    limits = plot_grid.compute_map_limits(np.zeros((2, 3, 4)), channel)
    assert limits == (0.0, 1.0)
    fig, axis = plt.subplots()
    try:
        image = plot_grid.draw_map(axis, np.zeros((3, 4)), channel, limits=limits)
        assert np.ma.getmaskarray(image.get_array()).all()
        assert np.isfinite(image.norm([0, 0.5, 1])).all()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("limits", [(0, 0), (1, 0), (0, np.inf), (np.nan, 1)])
def test_invalid_limits_are_rejected(limits):
    fig, axis = plt.subplots()
    try:
        with pytest.raises(ValueError, match="limits"):
            plot_grid.draw_map(axis, np.ones((2, 2)), "hpwl", limits=limits)
    finally:
        plt.close(fig)


def test_state_flow_rejects_mismatched_frame_labels(tmp_path):
    path = tmp_path / "flow.npz"
    np.savez(path, maps=np.stack([_map_fixture()] * 2))
    with pytest.raises(ValueError, match="frame label"):
        flow_plot.plot_state_flow(path, tmp_path / "flow.png", frame_labels=["Only one"])


def test_snapshot_true_outlines_preserve_tied_priority_colors(tmp_path, captured_export):
    maps = _map_fixture()
    maps[0, 4:6, 4:6] = 0.0  # Both chiplets have the same priority.
    boxes = np.array([[0.125, 0.125, 0.375, 0.375], [0.5, 0.5, 0.25, 0.25]])
    original_boxes = boxes.copy()
    boxes.flags.writeable = False
    path = tmp_path / "maps.npz"
    np.savez(path, maps=maps)
    snapshot_plot.plot_snapshot_maps(path, tmp_path / "snapshot.png", chiplet_boxes=boxes)
    _, axes, _ = captured_export[0]
    assert len(axes[0].patches) == 2
    assert not axes[1].patches and not axes[2].patches
    for patch, (x, y, width, height) in zip(axes[0].patches, boxes):
        np.testing.assert_allclose(patch.get_xy(), (x * 8 - 0.5, y * 8 - 0.5))
        assert patch.get_width() == width * 8
        assert patch.get_height() == height * 8
        assert not patch.get_fill()
        assert patch.get_linewidth() <= 0.6
    image = axes[0].images[0]
    np.testing.assert_array_equal(image.get_array().data, maps[0])
    np.testing.assert_array_equal(image.to_rgba(image.get_array())[2, 2], image.to_rgba(image.get_array())[5, 5])
    np.testing.assert_array_equal(boxes, original_boxes)
    assert not axes[0].texts


def test_state_flow_uses_boxes_for_selected_original_frame(tmp_path, captured_export):
    path = tmp_path / "flow.npz"
    np.savez(path, maps=np.stack([_map_fixture()] * 3))
    boxes = [
        np.array([[0.0, 0.0, 0.25, 0.5]]),
        np.array([[0.2, 0.1, 0.25, 0.5]]),
        np.array([[0.5, 0.5, 0.25, 0.5]]),
    ]
    flow_plot.plot_state_flow(path, tmp_path / "flow.png", max_frames=2, chiplet_boxes=boxes)
    _, axes, _ = captured_export[0]
    np.testing.assert_allclose(axes[0].patches[0].get_xy(), (-0.5, -0.5))
    np.testing.assert_allclose(axes[1].patches[0].get_xy(), (3.5, 3.5))
    assert all(not axis.patches for axis in axes[2:])


def test_state_flow_rejects_mismatched_box_frame_count(tmp_path):
    path = tmp_path / "flow.npz"
    np.savez(path, maps=np.stack([_map_fixture()] * 2))
    with pytest.raises(ValueError, match="chiplet_boxes"):
        flow_plot.plot_state_flow(path, tmp_path / "flow.png", chiplet_boxes=[np.empty((0, 4))])


@pytest.mark.parametrize("kind", ["snapshot", "flow", "effects"])
def test_every_plot_invocation_exports_png_and_pdf(tmp_path, monkeypatch, kind):
    # These are export smoke fixtures, not empirical layout measurements.
    monkeypatch.delenv("CHIPLET_FIGURE_QA_TOOLS", raising=False)
    maps = _map_fixture()
    path = tmp_path / "maps.npz"
    output = tmp_path / f"{kind}.png"
    if kind == "snapshot":
        np.savez(path, maps=maps)
        snapshot_plot.plot_snapshot_maps(path, output, dpi=40)
    elif kind == "flow":
        np.savez(path, maps=np.stack([maps]))
        flow_plot.plot_state_flow(path, output, dpi=40)
    else:
        np.savez(path, before_maps=maps, after_maps=np.stack([maps]))
        effects_plot.plot_algorithm_effects(path, output, ["Unchanged"], dpi=40)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.with_suffix(".pdf").read_bytes().startswith(b"%PDF-")
