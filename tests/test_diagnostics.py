import numpy as np
import pymc as pm
import pytest

import pymc_prop.sampler as sampler_module
from pymc_prop.diagnostics import compute_kgd, get_bandwidth, imq_kernel
from pymc_prop.sample import sample_pro


def test_imq_kernel_components():
    sq_dists = np.array([[0.0, 2.0], [2.0, 0.0]])
    dimension = 3
    bandwidth = 1.5

    kernel, cross_coeff, trace_term = imq_kernel(
        sq_dists, dimension, bandwidth
    )
    expected_kernel = (1.0 + sq_dists / bandwidth) ** -0.5

    np.testing.assert_allclose(kernel, expected_kernel)
    np.testing.assert_allclose(
        cross_coeff, expected_kernel**3 / bandwidth
    )
    np.testing.assert_allclose(
        trace_term,
        dimension * expected_kernel**3 / bandwidth
        - 3.0 * sq_dists * expected_kernel**5 / bandwidth**2,
    )


def test_get_bandwidth_median_heuristic():
    particles = np.array([[0.0], [1.0], [3.0]])
    diffs = particles[:, None, :] - particles[None, :, :]
    sq_dists = np.sum(diffs**2, axis=-1)

    expected = np.median([1.0, 9.0, 4.0]) / np.log(3)
    assert np.isclose(get_bandwidth(sq_dists), expected)
    assert get_bandwidth(sq_dists, bandwidth=2.5) == 2.5


def test_get_bandwidth_rejects_invalid_explicit_value():
    sq_dists = np.zeros((2, 2))

    for bandwidth in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and positive"):
            get_bandwidth(sq_dists, bandwidth=bandwidth)


def test_compute_kgd_matches_sqrt_direct_v_statistic():
    particles = np.array([[-1.0], [0.5], [2.0]])
    score = np.array([[0.25], [-0.75], [1.5]])
    bandwidth = 1.7

    stein_kernel = np.empty((len(particles), len(particles)))
    for i, (x_i,) in enumerate(particles):
        for j, (x_j,) in enumerate(particles):
            difference = x_i - x_j
            squared_difference = difference**2
            kernel = (1.0 + squared_difference / bandwidth) ** -0.5
            cross_terms = (
                (score[i, 0] - score[j, 0])
                * difference
                * kernel**3
                / bandwidth
            )
            derivative_trace = (
                kernel**3 / bandwidth
                - 3.0 * squared_difference * kernel**5 / bandwidth**2
            )
            stein_kernel[i, j] = (
                score[i, 0] * score[j, 0] * kernel
                + cross_terms
                + derivative_trace
            )

    actual = compute_kgd(particles, score, bandwidth=bandwidth)
    np.testing.assert_allclose(actual, np.sqrt(np.mean(stein_kernel)))


def test_compute_kgd_clips_negative_roundoff_before_sqrt():
    particles = np.array([[0.0], [1.0]])
    score = np.zeros_like(particles)

    def roundoff_kernel(sq_dists, dimension, bandwidth):
        shape = sq_dists.shape
        return np.zeros(shape), np.zeros(shape), np.full(shape, -1e-16)

    assert (
        compute_kgd(
            particles,
            score,
            kernel_fn=roundoff_kernel,
            bandwidth=1.0,
        )
        == 0.0
    )


def test_normal_target_has_lower_kgd_than_wrong_score():
    rng = np.random.default_rng(91)
    particles = rng.normal(size=(500, 1))
    sq_dists = (particles[:, None, 0] - particles[None, :, 0]) ** 2
    bandwidth = get_bandwidth(sq_dists)

    target_score = -particles
    shifted_score = -(particles - 2.0)

    target_kgd = compute_kgd(
        particles, target_score, bandwidth=bandwidth
    )
    shifted_kgd = compute_kgd(
        particles, shifted_score, bandwidth=bandwidth
    )

    assert target_kgd < shifted_kgd


def test_sample_pro_validates_kgd_options():
    with pm.Model() as model:
        location = pm.Normal("location", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=location, sigma=1.0, observed=np.array([0.0]))

    cases = (
        ({"kgd_interval": 0}, "positive integer"),
        ({"kgd_interval": True}, "positive integer"),
        (
            {"kgd_interval": 1, "include_sample_stats": False},
            "require sample_stats",
        ),
        ({"kgd_bandwidth": 1.0}, "requires kgd_interval"),
        (
            {"kgd_interval": 1, "kgd_bandwidth": 0.0},
            "finite and positive",
        ),
    )

    for kwargs, match in cases:
        with pytest.raises(ValueError, match=match):
            sample_pro(model=model, **kwargs)


def test_sample_pro_records_kgd_at_requested_interval():
    with pm.Model() as model:
        location = pm.Normal("location", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=location, sigma=1.0, observed=np.array([0.0]))

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=4,
        tune=1,
        step_size=1e-3,
        random_seed=42,
        include_log_likelihood=False,
        kgd_interval=2,
        kgd_bandwidth=1.0,
    )

    kgd = dt.sample_stats["kgd"]
    first_chain = kgd.isel(chain=0).values

    assert np.all(np.isfinite(first_chain[[0, 2]]))
    assert np.all(np.isnan(first_chain[[1, 3]]))
    np.testing.assert_allclose(
        kgd.values,
        np.broadcast_to(first_chain[:, None], kgd.shape),
        equal_nan=True,
    )
    np.testing.assert_array_equal(kgd.coords["draw"], np.arange(4))
    np.testing.assert_array_equal(kgd.coords["step"], np.arange(1, 5))


def test_kgd_disabled_skips_bandwidth_computation(monkeypatch):
    with pm.Model() as model:
        location = pm.Normal("location", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=location, sigma=1.0, observed=np.array([0.0]))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("get_bandwidth should not run when KGD is disabled")

    monkeypatch.setattr(sampler_module, "get_bandwidth", fail_if_called)

    dt = sample_pro(
        model=model,
        n_particles=4,
        n_steps=4,
        tune=0,
        step_size=1e-3,
        random_seed=42,
        include_log_likelihood=False,
    )

    assert "kgd" not in dt.sample_stats
