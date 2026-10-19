import pytest

from chiplet_layout_experiment.arms import Arm, default_arm_catalog


def test_default_arm_catalog_is_finite_and_hashable():
    arms = default_arm_catalog()

    assert 1 < len(arms) <= 12
    assert len({arm.arm_id for arm in arms}) == len(arms)
    assert all(arm.budget > 0 for arm in arms)
    assert all(len(arm.config_hash) == 64 for arm in arms)


def test_arm_rejects_empty_or_non_positive_budget():
    with pytest.raises(ValueError, match="arm_id"):
        Arm(arm_id="", worker_id="sa", config={}, budget=1)
    with pytest.raises(ValueError, match="budget"):
        Arm(arm_id="a", worker_id="sa", config={}, budget=0)
