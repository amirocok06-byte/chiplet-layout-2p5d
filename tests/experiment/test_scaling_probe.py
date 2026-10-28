import importlib.util
import numpy as np
import pytest

@pytest.mark.skipif(importlib.util.find_spec('torch') is None,reason='optional torch')
def test_training_constant_features_do_not_explode_on_new_state():
    from chiplet_layout_experiment.scaling_probe import bounded_scale
    train=np.array([[0.,1.,1.],[0.,2.,1.]])
    other=np.array([[1000.,999.,1.]])
    scaled=bounded_scale(train,other)
    np.testing.assert_array_equal(scaled,[[0.,5.,1.]])
