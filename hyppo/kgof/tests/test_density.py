import numpy as np
from numpy import testing
from past.utils import old_div
from scipy.linalg.misc import norm

from ..datasource import DSNormal, DSIsotropicNormal
from ..density import IsotropicNormal, Normal
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
        ds_isonorm = DSIsotropicNormal(mean, variance)
        isonorm.log_normalized_den(X)
        isonorm.dim()
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
    @pytest.mark.parametrize("n", [2])
    @pytest.mark.parametrize("d", [2])
    def test_log_den(self, n, d):
        rng = default_rng(16)
        cov = np.array([[1.1, 1.2], [1.1, 1.2]])
        mean = rng.standard_normal(size=(n, d))
        X = rng.random(size=(n, d)) + 1

        test_mean = np.ones(2)
        test_cov = np.array([[1.1, 1.2], [1.1, 1.2]])

        norm = Normal(mean, cov)
        log_dens = norm.log_den(X)
        E, V = np.linalg.eigh(cov)
        prec = np.dot(np.dot(V, np.diag(old_div(1.0, E))), V.T)
        X0 = X - mean
        X0prec = np.dot(X0, prec)
        my_log_dens = old_div(-np.sum(X0prec * X0, 1), 2.0)

        ds_norm = DSNormal(test_mean, cov)
        ds_norm.sample(n=10)
        norm.get_datasource()
        norm.dim()

        # check correctness
        testing.assert_almost_equal(log_dens, my_log_dens)
