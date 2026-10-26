"""Small synthetic contract fixtures; these are not experiment measurements."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / 'scripts'))
import plot_grid as grid
import plot_snapshot_maps as snapshot
import plot_state_flow as flow
import plot_algorithm_effects as effects


@pytest.fixture
def captured(monkeypatch):
    result = []
    def capture(fig, output_path, dpi=300, axes=None):
        result.append((fig, list(axes)))
    for module in (snapshot, flow, effects):
        monkeypatch.setattr(module, 'export_figure', capture)
    return result


def maps(count=4):
    data = np.zeros((count, 4, 4))
    data[0, 0] = -1
    data[1, 1] = 2
    data[2, 2] = .1
    if count == 4:
        data[3, 3, 3] = .25
    return data


@pytest.mark.parametrize('count', [3, 4])
@pytest.mark.parametrize('kind', ['snapshot', 'flow', 'effects'])
def test_panel_shapes_labels_and_legacy_calls(tmp_path, captured, count, kind):
    data = maps(count)
    path = tmp_path / 'input.npz'
    output = tmp_path / 'output.png'
    if kind == 'snapshot':
        np.savez(path, maps=data)
        snapshot.plot_snapshot_maps(path, output)
        expected = (1, count)
    elif kind == 'flow':
        np.savez(path, maps=np.stack([data, data]))
        flow.plot_state_flow(path, output, 2)
        expected = (count, 2)
    else:
        np.savez(path, before_maps=data, after_maps=np.stack([data, data]))
        effects.plot_algorithm_effects(path, output, ['A', 'B'])
        expected = (2, count)
    fig, axes = captured[0]
    assert len(axes) == np.prod(expected)
    spec = axes[0].get_subplotspec().get_gridspec()
    assert (spec.nrows, spec.ncols) == expected
    assert 'Connectivity priority' in axes[0].get_title()
    assert any('Spatial thermal proxy' in ax.get_title() for ax in axes)
    assert len(fig.axes) == len(axes) + count
    if count == 4:
        fourth = [ax for ax in axes if 'Overlap / boundary violation' in ax.get_title()]
        assert fourth
        for ax in fourth:
            if kind == 'effects':
                assert 'Δ' in ax.get_title()
                assert ax.images[0].get_clim()[0] < 0 < ax.images[0].get_clim()[1]
            else:
                assert '0 = no displayed violation' in ax.get_title()
                assert ax.images[0].get_clim() == (0, .25)


def test_legality_white_zero_colored_positive_and_no_mutation():
    raw = np.array([[0., 1e-8], [.2, 2.]])
    original = raw.copy()
    raw.flags.writeable = False
    fig, ax = plt.subplots()
    try:
        image = grid.draw_map(ax, raw, 'legality', limits=(0., 4.))
        rgba = image.to_rgba(image.get_array())
        np.testing.assert_array_equal(rgba[0, 0], [1, 1, 1, 1])
        assert not np.allclose(rgba[0, 1, :3], 1)
        assert image.get_clim() == (0, 4)
        np.testing.assert_array_equal(np.ma.getmaskarray(image.get_array()), raw == 0)
        np.testing.assert_array_equal(raw, original)
        assert grid.compute_map_limits(np.zeros((2, 3)), 'legality') == (0, 1)
        assert grid.compute_map_limits(raw, 'legality') == (0, 2)
        assert grid.compute_map_limits(raw, 'priority') == (0, 1)
        assert grid.compute_map_limits(raw, 'thermal_proxy') == (0, 2)
    finally:
        plt.close(fig)


def test_hidden_frame_maximum_and_external_frozen_limits(tmp_path, captured):
    data = np.stack([maps()] * 3)
    data[1, 3, 1, 1] = 7
    path = tmp_path / 'flow.npz'
    np.savez(path, maps=data)
    flow.plot_state_flow(path, tmp_path / 'a.png', 2)
    fourth = captured[-1][1][-2:]
    assert fourth[0].images[0].norm is fourth[1].images[0].norm
    assert fourth[0].images[0].get_clim() == (0, 7)
    limits = [(0, 1), (0, 10), (0, 2), (0, 12)]
    names = ('P', 'T', 'H', 'V')
    flow.plot_state_flow(path, tmp_path / 'b.png', 2, limits=limits, channel_labels=names)
    assert captured[-1][1][-1].images[0].get_clim() == (0, 12)
    snapshot_path = tmp_path / 'snapshot.npz'
    np.savez(snapshot_path, maps=data[0])
    snapshot.plot_snapshot_maps(snapshot_path, tmp_path / 'c.png', limits=limits, channel_labels=names)
    assert captured[-1][1][-1].images[0].get_clim() == (0, 12)


def test_effects_fourth_is_signed_delta_with_shared_scale(tmp_path, captured):
    before = maps()
    before[3, 0, 0] = 5
    after = np.stack([maps(), maps()])
    after[0, 3] = 0
    path = tmp_path / 'effects.npz'
    np.savez(path, before_maps=before, after_maps=after)
    effects.plot_algorithm_effects(path, tmp_path / 'out.png', ['A', 'B'])
    axes = captured[-1][1]
    for row in range(2):
        image = axes[row * 4 + 3].images[0]
        assert image.get_clim() == (-5, 5)
        np.testing.assert_array_equal(image.get_array().data, after[row, 3]-before[3])
    np.testing.assert_array_equal(axes[0].images[0].get_array(), after[0, 0] - before[0])


@pytest.mark.parametrize('bad', [-.1, np.nan, np.inf])
def test_invalid_legality_values_rejected(bad):
    with pytest.raises(ValueError):
        grid.compute_map_limits(np.array([0, bad]), 'legality')


@pytest.mark.parametrize('limits', [(-1, 2), (0, 0), (0, .1), (0, np.inf)])
def test_invalid_fourth_limits_rejected(tmp_path, limits):
    path = tmp_path / 'input.npz'
    np.savez(path, maps=maps())
    with pytest.raises(ValueError, match='legality'):
        snapshot.plot_snapshot_maps(path, tmp_path / 'out.png', limits=[(0, 1), (0, 2), (0, 1), limits])


@pytest.mark.parametrize('count', [2, 5])
def test_unsupported_channel_counts_rejected(tmp_path, count):
    path = tmp_path / 'input.npz'
    np.savez(path, maps=np.zeros((count, 4, 4)))
    with pytest.raises(ValueError, match='shape'):
        snapshot.plot_snapshot_maps(path, tmp_path / 'out.png')
