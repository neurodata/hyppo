import autograd.numpy as np
from numpy import testing
from ..kernel import KGauss
from ...tools import compute_kern, multi_compute_kern

import pytest
from numpy.random import default_rng


class TestKGauss:
    @pytest.mark.parametrize("n", [10])
    @pytest.mark.parametrize("d", [3])
    def test_basic(self, n, d):
        # sample
        rng = default_rng(29)
        X = rng.standard_normal(size=(n, d)) * 3
        k = KGauss(sigma2=1)
        K = k.eval(X, X)  # tests usage of compute_kern

        testing.assert_almost_equal(K.shape, (n, n))

    @pytest.mark.parametrize("n", [10])
    @pytest.mark.parametrize("d", [1, 3])
    def test_gradX_y(self, n, d):
        rng = default_rng(10)
        y = rng.standard_normal(size=d) * 2
        X = rng.random(size=(n, d)) * 3
        Y = rng.standard_normal(size=(n, d)) * 2

        sigma2 = 1.3
        k = KGauss(sigma2=sigma2)
        # n x d
        G = k.gradX_y(X, y)
        k.gradX_Y(X, Y, dim=d - 1)
        k.gradXY_sum(X, Y)
        k.pair_gradX_Y(X, Y)
        k.pair_gradXY_sum(X, Y)
        k.pair_eval(X, Y)
        # check correctness
        K = k.eval(X, y[np.newaxis, :])
        modG = -K / sigma2 * (X - y)

        testing.assert_equal(G.shape, modG.shape)
        testing.assert_almost_equal(G, modG)

    @pytest.mark.parametrize("n", [11])
    @pytest.mark.parametrize("d", [3, 1])
    def test_gradXY_sum(self, n, d):
        rng = default_rng(12)
        X = rng.standard_normal(size=(n, d))
        sigma2 = 1.4
        k = KGauss(sigma2=sigma2)

        # n x n
        modG = np.zeros((n, n))
        K = k.eval(X, X)
        for i in range(n):
            for j in range(n):
                diffi2 = np.sum((X[i, :] - X[j, :]) ** 2)
                modG[i, j] = K[i, j] / sigma2 * (d - diffi2 / sigma2)

        # check correctness of function that calculates sum of the gradient across x_i and y_i in Y
        G = k.gradXY_sum(X, X)

        testing.assert_equal(G.shape, modG.shape)
        testing.assert_almost_equal(G, modG)
