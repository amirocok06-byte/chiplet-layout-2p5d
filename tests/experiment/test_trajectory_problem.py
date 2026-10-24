import numpy as np

from chiplet_layout_experiment.trajectory_problem import TracedProblem


def problem():
    return TracedProblem(
        names=("a", "b"), board_mm=np.array([10., 8.]),
        sizes=np.array([[.2, .2], [.2, .2]]), powers=np.array([4., 8.]),
        pin_indices=np.array([[0, 1]]),
        pin_offsets=np.array([[[.1, .1], [.1, .1]]]),
        initial_positions=np.array([[.1, .2], [.6, .5]]),
    )


def test_pin_hpwl_and_map_sum_use_physical_board_axes():
    p = problem()
    e = p.evaluate(p.initial_positions)
    maps = p.maps(p.initial_positions, grid_size=32)
    assert np.isclose(e["hpwl_mm"], 5 + 2.4)
    assert np.isclose(maps[2].sum(), e["hpwl_mm"])
    assert np.isfinite(maps).all()
    assert (maps[0] == -1).any()


def test_spatial_thermal_proxy_changes_when_chiplets_approach():
    p = problem()
    separated = p.evaluate(p.initial_positions)
    close = p.evaluate(np.array([[.1, .2], [.31, .2]]))
    assert close["thermal_proxy_peak"] > separated["thermal_proxy_peak"]


def test_analytic_gradient_matches_finite_difference():
    p = problem()
    # Test overlap derivatives as well as spatial thermal and smooth wirelength.
    x = np.array([[.2, .2], [.31, .29]])
    g = p.gradient(x, smoothing=.015)
    numerical = np.zeros_like(x)
    for i in range(2):
        for j in range(2):
            d = np.zeros_like(x)
            d[i, j] = 1e-6
            numerical[i, j] = (
                p.smooth_objective(x + d, smoothing=.015)
                - p.smooth_objective(x - d, smoothing=.015)
            ) / 2e-6
    np.testing.assert_allclose(g, numerical, atol=2e-5, rtol=2e-4)


def test_unequal_containment_reports_exact_area_separately_from_penalty():
    p = problem()
    p.sizes = np.array([[.6, .4], [.2, .1]])
    p.__post_init__()
    metrics = p.evaluate(np.array([[.1, .1], [.2, .2]]))
    assert np.isclose(metrics["overlap_fraction"], .02 / .26)
    assert metrics["penetration_penalty"] > metrics["overlap_fraction"]
