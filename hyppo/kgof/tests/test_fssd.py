import numpy as np
import numpy.testing as testing
import matplotlib.pyplot as plt

from .. import fssd, data, density, _utils, kernel, h0simulator

from fssd import FSSD
from numpy.random import default_rng

import scipy.stats as stats 
import unittest

class TestFSSD(unittest.TestCase):
    def setUp(self):
        pass

    def test_basic(self):
        seed = 12
        # sample
        n = 100
        alpha = 0.01
        for d in [1, 4]:
            mean = np.zeros(d)
            variance = 1
            isonorm = density.IsotropicNormal(mean, variance)

            # only one dimension of the mean is shifted
            draw_mean = mean +0
            draw_variance = variance + 1

            rng = default_rng(seed)
            X = rng.standard_normal(size=(n, d))*np.sqrt(draw_variance) + draw_mean
            dat = data.Data(X)

            # Test
            for J in [1, 3]:
                sig2 = _utils.meddistance(X, subsample=1000)**2
                k = kernel.KGauss(sig2)

                # random test locations
                V = _utils.fit_gaussian_draw(X, J, seed=seed+1)
                null_sim = h0simulator.FSSDH0SimCovObs(n_simulate=200, seed=3)
                fssd = FSSD(isonorm, k, V, null_sim=null_sim, alpha=alpha)

                tresult = fssd.test(dat, return_simulated_stats=True)

                # assertions
                self.assertGreaterEqual(tresult['pvalue'], 0)
                self.assertLessEqual(tresult['pvalue'], 1)

    def test_ustat_h1_mean_variance(self):
        seed = 20
        # sample
        n = 200
        alpha = 0.01
        for d in [1, 4]:
            mean = np.zeros(d)
            variance = 1
            isonorm = density.IsotropicNormal(mean, variance)

            draw_mean = mean + 2
            draw_variance = variance + 1
            rng = default_rng(seed)
            X = rng.standard_normal(size=(n, d))*np.sqrt(draw_variance) + draw_mean
            dat = data.Data(X)

            # Test
            for J in [1, 3]:
                sig2 = _utils.meddistance(X, subsample=1000)**2
                k = kernel.KGauss(sig2)

                # random test locations
                V = _utils.fit_gaussian_draw(X, J, seed=seed+1)

                null_sim = h0simulator.FSSDH0SimCovObs(n_simulate=200, seed=3)
                fssd = FSSD(isonorm, k, V, null_sim=null_sim, alpha=alpha)
                fea_tensor = fssd.feature_tensor(X)

                u_mean, u_variance = FSSD.ustat_h1_mean_variance(fea_tensor)

                # assertions
                self.assertGreaterEqual(u_variance, 0)
                # should reject H0
                self.assertGreaterEqual(u_mean, 0)

    def tearDown(self):
        pass
