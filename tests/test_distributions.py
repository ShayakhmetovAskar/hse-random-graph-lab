from math import isclose

import numpy as np
from scipy.stats import skewnorm, t as student_t, lognorm, weibull_min

import src.utils as utils  # ← updated import

N = 25_000
RNG_SEED = 42


def _set_seed():
    np.random.seed(RNG_SEED)


def test_generate_skewnormal_shape_and_mean():
    _set_seed()
    alpha = 4.0
    data = utils.generate_skewnormal(N, alpha)
    assert data.shape == (N,)

    delta = alpha / (1 + alpha ** 2) ** 0.5
    theor_mean = delta * (2 / np.pi) ** 0.5
    assert isclose(data.mean(), theor_mean, rel_tol=0.05)


def test_generate_student_t_zero_mean():
    _set_seed()
    nu = 5
    data = utils.generate_student_t(N, nu)
    assert isclose(data.mean(), 0.0, abs_tol=0.05)


def test_generate_lognormal_moment():
    _set_seed()
    sigma = np.log(5)
    data = utils.generate_lognormal_zero_sigma(size=N, sigma=sigma)
    theor_mean = np.exp(sigma ** 2 / 2)
    assert isclose(data.mean(), theor_mean, rel_tol=0.05)


def test_generate_weibull_mean():
    _set_seed()
    lam, k = 1.0, 0.5
    data = utils.generate_weibull_half_lambda(size=N, lambda_=lam, k=k)

    from math import gamma
    theor_mean = lam * gamma(1 + 1 / k)
    assert isclose(data.mean(), theor_mean, rel_tol=0.05)
