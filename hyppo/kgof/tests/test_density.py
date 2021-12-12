import numpy as np
from numpy import testing
from scipy.linalg.misc import norm

from ..density import IsotropicNormal, Normal, GaussianMixture
import scipy.stats as stats
from numpy.random import default_rng

import pytest


class TestIsotropicNormal:
    @pytest.mark.parametrize("n", [7])
    @pytest.mark.parametrize("d", [3, 1])
    def test_log_den(self, n, d):
        rng = default_rng(16)
        variance = 1.1
        mean = rng.standard_normal(size=d)
        X = rng.random(size=(n, d)) + 1

        isonorm = IsotropicNormal(mean, variance)
        log_dens = isonorm.log_den(X)
        my_log_dens = -np.sum((X - mean) ** 2, 1) / (2.0 * variance)

        # check correctness
        testing.assert_almost_equal(log_dens, my_log_dens)

    @pytest.mark.parametrize("n", [8])
    @pytest.mark.parametrize("d", [4, 1])
    def test_grad_log(self, n, d):
        rng = default_rng(17)
        variance = 1.2
        mean = rng.standard_normal(size=d) + 1
        X = rng.random(size=(n, d)) - 2

        isonorm = IsotropicNormal(mean, variance)
        grad_log = isonorm.grad_log(X)
        my_grad_log = -(X - mean) / variance

        # check correctness
        testing.assert_almost_equal(grad_log, my_grad_log)


class TestNormal:
    @pytest.mark.parametrize("n", [7])
    @pytest.mark.parametrize("d", [3, 1])
    def test_log_den(self, n, d):
        rng = default_rng(16)
        variance = 1.1
        mean = rng.standard_normal(size=d)
        X = rng.random(size=(n, d)) + 1

        norm = Normal(mean, variance)
        log_dens = norm.log_den(X)
        my_log_dens = -np.sum((X - mean) ** 2, 1) / (2.0 * variance)

        # check correctness
        testing.assert_almost_equal(log_dens, my_log_dens)


class TestGaussianMixture:
    @pytest.mark.parametrize("i", [0, 1, 2, 3])
    def test_multivariate_normal_density(self, i):
        rng = default_rng(i + 8)
        d = i + 2
        cov = stats.wishart(df=10 + d, scale=np.eye(d)).rvs(size=1)
        mean = rng.standard_normal(size=d)
        X = rng.standard_normal(size=(11, d))
        den_estimate = GaussianMixture.multivariate_normal_density(mean, cov, X)

        mnorm = stats.multivariate_normal(mean=mean, cov=cov)
        den_truth = mnorm.pdf(X)

        testing.assert_almost_equal(den_estimate, den_truth)
