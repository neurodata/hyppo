"""
Module for testing goftest module.
"""

__author__ = 'wittawat'

import numpy as np
import numpy.testing as testing
import matplotlib.pyplot as plt
import kgof.data as data
import kgof.density as density
import kgof.util as util
import kgof.kernel as kernel
import kgof.goftest as gof
import kgof.glo as glo
import scipy.stats as stats

import unittest


class TestFSSD(unittest.TestCase):
    def setUp(self):
        pass

    def test_basic(self):
        """
        Nothing special. Just test basic things.
        """
        seed = 12
        # sample
        n = 100
        alpha = 0.01
        for d in [1, 4]:
            mean = np.zeros(d)
            variance = 1
            isonorm = density.IsotropicNormal(mean, variance)

            # only one dimension of the mean is shifted
            #draw_mean = mean + np.hstack((1, np.zeros(d-1)))
            draw_mean = mean +0
            draw_variance = variance + 1
            X = util.randn(n, d, seed=seed)*np.sqrt(draw_variance) + draw_mean
            dat = data.Data(X)

            # Test
            for J in [1, 3]:
                sig2 = util.meddistance(X, subsample=1000)**2
                k = kernel.KGauss(sig2)

                # random test locations
                V = util.fit_gaussian_draw(X, J, seed=seed+1)
                null_sim = gof.FSSDH0SimCovObs(n_simulate=200, seed=3)
                fssd = gof.FSSD(isonorm, k, V, null_sim=null_sim, alpha=alpha)

                tresult = fssd.perform_test(dat, return_simulated_stats=True)

                # assertions
                self.assertGreaterEqual(tresult['pvalue'], 0)
                self.assertLessEqual(tresult['pvalue'], 1)

    def test_optimized_fssd(self):
        """
        Test FSSD test with parameter optimization.
        """
        seed = 4
        # sample size
        n = 179 
        alpha = 0.01
        for d in [1, 3]:
            mean = np.zeros(d)
            variance = 1.0
            p = density.IsotropicNormal(mean, variance)
            # Mean difference. obvious reject
            ds = data.DSIsotropicNormal(mean+4, variance+0)
            dat = ds.sample(n, seed=seed)
            # test 
            for J in [1, 4]:
                opts = {
                    'reg': 1e-2,
                    'max_iter': 10, 
                    'tol_fun':1e-3, 
                    'disp':False
                }
                tr, te = dat.split_tr_te(tr_proportion=0.3, seed=seed+1)

                Xtr = tr.X
                gwidth0 = util.meddistance(Xtr, subsample=1000)**2
                # random test locations
                V0 = util.fit_gaussian_draw(Xtr, J, seed=seed+1)
                V_opt, gw_opt, opt_result = \
                gof.GaussFSSD.optimize_locs_widths(p, tr, gwidth0, V0, **opts)

                # construct a test
                k_opt = kernel.KGauss(gw_opt)
                null_sim = gof.FSSDH0SimCovObs(n_simulate=2000, seed=10)
                fssd_opt = gof.FSSD(p, k_opt, V_opt, null_sim=null_sim, alpha=alpha)
                fssd_opt_result = fssd_opt.perform_test(te, return_simulated_stats=True)
                assert fssd_opt_result['h0_rejected']

    def test_auto_init_opt_fssd(self):
        """
        Test FSSD-opt test with automatic parameter initialization.
        """
        seed = 5
        # sample size
        n = 191 
        alpha = 0.01
        for d in [1, 4]:
            mean = np.zeros(d)
            variance = 1.0
            p = density.IsotropicNormal(mean, variance)
            # Mean difference. obvious reject
            ds = data.DSIsotropicNormal(mean+4, variance+0)
            dat = ds.sample(n, seed=seed)
            # test 
            for J in [1, 3]:
                opts = {
                    'reg': 1e-2,
                    'max_iter': 10, 
                    'tol_fun': 1e-3, 
                    'disp':False
                }
                tr, te = dat.split_tr_te(tr_proportion=0.3, seed=seed+1)

                V_opt, gw_opt, opt_result = \
                gof.GaussFSSD.optimize_auto_init(p, tr, J, **opts)

                # construct a test
                k_opt = kernel.KGauss(gw_opt)
                null_sim = gof.FSSDH0SimCovObs(n_simulate=2000, seed=10)
                fssd_opt = gof.FSSD(p, k_opt, V_opt, null_sim=null_sim, alpha=alpha)
                fssd_opt_result = fssd_opt.perform_test(te, return_simulated_stats=True)
                assert fssd_opt_result['h0_rejected']

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
            X = util.randn(n, d, seed=seed)*np.sqrt(draw_variance) + draw_mean
            dat = data.Data(X)

            # Test
            for J in [1, 3]:
                sig2 = util.meddistance(X, subsample=1000)**2
                k = kernel.KGauss(sig2)

                # random test locations
                V = util.fit_gaussian_draw(X, J, seed=seed+1)

                null_sim = gof.FSSDH0SimCovObs(n_simulate=200, seed=3)
                fssd = gof.FSSD(isonorm, k, V, null_sim=null_sim, alpha=alpha)
                fea_tensor = fssd.feature_tensor(X)

                u_mean, u_variance = gof.FSSD.ustat_h1_mean_variance(fea_tensor)

                # assertions
                self.assertGreaterEqual(u_variance, 0)
                # should reject H0
                self.assertGreaterEqual(u_mean, 0)

    def tearDown(self):
        pass

# end class TestFSSD

class TestSteinWitness(unittest.TestCase):
    def test_basic(self):
        d = 3
        p = density.IsotropicNormal(mean=np.zeros(d), variance=3.0)
        q = density.IsotropicNormal(mean=np.zeros(d)+2, variance=3.0)
        k = kernel.KGauss(2.0)

        ds = q.get_datasource()
        n = 97
        dat = ds.sample(n, seed=3)

        witness = gof.SteinWitness(p, k, dat)
        # points to evaluate the witness
        J = 4
        V = np.random.randn(J, d)*2
        evals = witness(V)

        testing.assert_equal(evals.shape, (J, d))

# end class TestSteinWitness


if __name__ == '__main__':
   unittest.main()

