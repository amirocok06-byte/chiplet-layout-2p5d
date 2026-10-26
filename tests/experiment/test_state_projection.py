import numpy as np
import pytest

from chiplet_layout_experiment.state_projection import legality_map, project_state
from chiplet_layout_experiment.trajectory_problem import TracedProblem


def test_same_pixel_neighbours_and_touching_edges_are_not_overlaps():
    x = np.array([[.01, .01], [.04, .01]])
    size = np.array([[.02, .02], [.02, .02]])
    assert not legality_map(x, size, 8).any()
    assert not legality_map([[0., 0.], [.2, 0.]], [[.2, .2], [.2, .2]], 20).any()


def test_map_integral_equals_exact_pair_area_and_containment():
    result = legality_map([[.1, .1], [.2, .2]], [[.6, .6], [.1, .2]], 32)
    assert result.sum(dtype=float)/32**2 == pytest.approx(.02, rel=1e-6)


def test_boundary_signal_visible_even_when_whole_chiplet_is_outside():
    result = legality_map([[-.4, .2]], [[.1, .1]], 20)
    assert result[:, 0].sum() > 0
    assert not result[:, 1:].any()


def test_priority_max_and_four_channels_preserve_physical_maps():
    p = TracedProblem(names=('a', 'b', 'c'), board_mm=np.array([10., 10.]),
        sizes=np.full((3, 2), .2), powers=np.ones(3),
        pin_indices=np.array([[0, 1], [0, 1]]), pin_offsets=np.full((2, 2, 2), .1),
        initial_positions=np.array([[.1, .1], [.1, .1], [.7, .7]]))
    old = p.maps(p.initial_positions, 32)
    maps = project_state(p, p.initial_positions, 32)
    assert maps.shape == (4, 32, 32)
    np.testing.assert_array_equal(maps[1:3], old[1:3])
    assert (maps[0] == -1).any() and (maps[0] == 0).any() and (maps[0] == 1).any()
    assert maps[3].max() > 0
    assert p.evaluator_calls == p.gradient_calls == 0
