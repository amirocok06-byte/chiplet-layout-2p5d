import numpy as np
from chiplet_layout_experiment.preference_probe import preference_summary


def test_preference_summary_identifies_switches_and_fixed_oracle_gap():
    result = preference_summary(np.array([[3., 1.], [1., 4.]]), ['a', 'b'])
    assert result['winner_counts'] == {'a': 1, 'b': 1}
    assert result['best_fixed_mean_regret'] == 1.


def test_preference_summary_keeps_ties():
    result = preference_summary(np.array([[1., 1.]]), ['a', 'b'])
    assert result['tied_states'] == 1
    assert result['best_fixed_mean_regret'] == 0.
