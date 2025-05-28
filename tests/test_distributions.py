"""Statistical sanity checks for the random‑number generators."""

from math import isclose

import numpy as np
from scipy.stats import skewnorm, t as student_t, lognorm, weibull_min

import module  # assumes module.py is at repo root

N = 25_000  # moderately large for stable sample moments
RNG_SEED = 42


def _set_seed():
    np.random.seed(RNG_SEED)


def test_generate_skewnormal_shape_and_mean():
    _set_seed()
    alpha = 4.0
    data = module.generate_skewnormal(N, alpha)
    assert data.shape == (N,)

    # Theoretical mean of skew‑normal
    delta = alpha / (1 + alpha ** 2) ** 0.5
    theor_mean = delta * (2 / np.pi) ** 0.5
    sample_mean = data.mean()
    assert isclose(sample_mean, theor_mean, rel_tol=0.05)


def test_generate_student_t_zero_mean():
    _set_seed()
    nu = 5
    data = module.generate_student_t(N, nu)
    assert data.shape == (N,)
    # Student‑t with nu>1 has mean 0
    assert isclose(data.mean(), 0.0, abs_tol=0.05)


def test_generate_lognormal_moment():
    _set_seed()
    sigma = np.log(5)
    data = module.generate_lognormal_zero_sigma(size=N, sigma=sigma)
    assert data.shape == (N,)
    theor_mean = np.exp(sigma ** 2 / 2)
    assert isclose(data.mean(), theor_mean, rel_tol=0.05)


def test_generate_weibull_mean():
    _set_seed()
    lam = 1.0
    k = 0.5
    data = module.generate_weibull_half_lambda(size=N, lambda_=lam, k=k)
    assert data.shape == (N,)

    # Theoretical mean of Weibull
    from math import gamma

    theor_mean = lam * gamma(1 + 1 / k)
    assert isclose(data.mean(), theor_mean, rel_tol=0.05)
