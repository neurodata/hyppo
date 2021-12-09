import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt

from .. import data, density, _utils, kernel, fssd, h0simulator
import scipy.stats as stats
import numpy.testing as testing
from numpy.random import default_rng

import unittest


class TestKGauss(unittest.TestCase):
    def setUp(self):
        pass

    def test_basic(self):
        # sample
        n = 10
        d = 3
        rng = default_rng(29)
        X = rng.standard_normal(size=(n, d)) * 3
        k = kernel.KGauss(sigma2=1)
        K = k.eval(X, X)

        self.assertEqual(K.shape, (n, n))
        self.assertTrue(np.all(K >= 0 - 1e-6))
        self.assertTrue(np.all(K <= 1 + 1e-6), "K not bounded by 1")

    def test_pair_gradX_Y(self):
        # sample
        n = 11
        d = 3
        rng = default_rng(20)
        X = rng.standard_normal(size=(n, d)) * 4
        Y = rng.standard_normal(size=(n, d)) * 2
        k = kernel.KGauss(sigma2=2.1)
        # n x d
        pair_grad = k.pair_gradX_Y(X, Y)
        loop_grad = np.zeros((n, d))
        for i in range(n):
            for j in range(d):
                loop_grad[i, j] = k.gradX_Y(X[[i], :], Y[[i], :], j)

        testing.assert_almost_equal(pair_grad, loop_grad)

    def test_gradX_y(self):
        n = 10
        rng = default_rng(10)
        for d in [1, 3]:
            y = rng.standard_normal(size=d) * 2
            X = rng.random(size=(n, d)) * 3

            sigma2 = 1.3
            k = kernel.KGauss(sigma2=sigma2)
            # n x d
            G = k.gradX_y(X, y)
            # check correctness
            K = k.eval(X, y[np.newaxis, :])
            myG = -K / sigma2 * (X - y)

            self.assertEqual(G.shape, myG.shape)
            testing.assert_almost_equal(G, myG)

    def test_gradXY_sum(self):
        n = 11
        rng = default_rng(12)
        for d in [3, 1]:
            X = rng.standard_normal(size=(n, d))
            sigma2 = 1.4
            k = kernel.KGauss(sigma2=sigma2)

            # n x n
            myG = np.zeros((n, n))
            K = k.eval(X, X)
            for i in range(n):
                for j in range(n):
                    diffi2 = np.sum((X[i, :] - X[j, :]) ** 2)
                    # myG[i, j] = -diffi2*K[i, j]/(sigma2**2)+ d*K[i, j]/sigma2
                    myG[i, j] = K[i, j] / sigma2 * (d - diffi2 / sigma2)

            # check correctness
            G = k.gradXY_sum(X, X)

            self.assertEqual(G.shape, myG.shape)
            testing.assert_almost_equal(G, myG)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
