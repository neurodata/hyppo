import numpy as np
import matplotlib.pyplot as plt

from .. import data, density, _utils, kernel, fssd, h0simulator
import scipy.stats as stats

import unittest
from numpy.random import default_rng


class TestIsotropicNormal(unittest.TestCase):

    def test_log_den(self):
        n = 7
        rng = default_rng(16)
        for d in [3, 1]:
            variance = 1.1
            mean = rng.standard_normal(size=d)
            X = rng.random(size=(n, d)) + 1

            isonorm = density.IsotropicNormal(mean, variance)
            log_dens = isonorm.log_den(X)
            my_log_dens = -np.sum((X - mean) ** 2, 1) / (2.0 * variance)

            # check correctness
            np.testing.assert_almost_equal(log_dens, my_log_dens)

    def test_grad_log(self):
        n = 8
        rng = default_rng(17)
        for d in [4, 1]:
            variance = 1.2
            mean = rng.standard_normal(size=d) + 1
            X = rng.random(size=(n, d)) - 2

            isonorm = density.IsotropicNormal(mean, variance)
            grad_log = isonorm.grad_log(X)
            my_grad_log = -(X - mean) / variance

            # check correctness
            np.testing.assert_almost_equal(grad_log, my_grad_log)

    def tearDown(self):
        pass


class TestGaussianMixture(unittest.TestCase):
    def test_multivariate_normal_density(self):
        for i in range(4):
            rng = default_rng(i + 8)
            d = i + 2
            cov = stats.wishart(df=10 + d, scale=np.eye(d)).rvs(size=1)
            mean = rng.standard_normal(size=d)
            X = rng.standard_normal(size=(11, d))
            den_estimate = density.GaussianMixture.multivariate_normal_density(
                mean, cov, X
            )

            mnorm = stats.multivariate_normal(mean=mean, cov=cov)
            den_truth = mnorm.pdf(X)

            np.testing.assert_almost_equal(den_estimate, den_truth)
