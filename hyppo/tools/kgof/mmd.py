"""
Module containing the MMD two-sample test of Gretton et al., 2012 
"A Kernel Two-Sample Test" disguised as goodness-of-fit tests. Require the
ability to sample from the specified density. This module depends on an external
package

freqopttest https://github.com/wittawatj/interpretable-test

providing an implementation to the MMD test.

"""

from builtins import str
__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import autograd
import autograd.numpy as np
# Require freqopttest https://github.com/wittawatj/interpretable-test
import freqopttest.tst as tst
import freqopttest.data as fdata
import kgof.data as data
import kgof.goftest as gof
import kgof.util as util
import kgof.kernel as kernel
import logging
import matplotlib.pyplot as plt

import scipy
import scipy.stats as stats

class QuadMMDGof(gof.GofTest):
    """
    Goodness-of-fit test by drawing sample from the density p and test with
    the MMD test of Gretton et al., 2012. 

    H0: the sample follows p
    H1: the sample does not follow p

    p is specified to the constructor in the form of an UnnormalizedDensity.
    """

    def __init__(self, p, k, n_permute=400, alpha=0.01, seed=28):
        """
        p: an instance of UnnormalizedDensity
        k: an instance of Kernel
        n_permute: number of times to permute the samples to simulate from the 
            null distribution (permutation test)
        alpha: significance level 
        seed: random seed
        """
        super(QuadMMDGof, self).__init__(p, alpha)
        # Construct the MMD test
        self.mmdtest = tst.QuadMMDTest(k, n_permute=n_permute, alpha=alpha)
        self.k = k
        self.seed = seed
        ds = p.get_datasource()
        if ds is None:
            raise ValueError('%s test requires a density p which implements get_datasource(', str(QuadMMDGof))


    def perform_test(self, dat):
        """
        dat: an instance of Data
        """
        with util.ContextTimer() as t:
            seed = self.seed
            mmdtest = self.mmdtest
            p = self.p

            # Draw sample from p. #sample to draw is the same as that of dat
            ds = p.get_datasource()
            p_sample = ds.sample(dat.sample_size(), seed=seed+12)

            # Run the two-sample test on p_sample and dat
            # Make a two-sample test data
            tst_data = fdata.TSTData(p_sample.data(), dat.data())
            # Test 
            results = mmdtest.perform_test(tst_data)

        results['time_secs'] = t.secs
        return results

    def compute_stat(self, dat):
        mmdtest = self.mmdtest
        p = self.p
        # Draw sample from p. #sample to draw is the same as that of dat
        ds = p.get_datasource()
        p_sample = ds.sample(dat.sample_size(), seed=self.seed)

        # Make a two-sample test data
        tst_data = fdata.TSTData(p_sample.data(), dat.data())
        s = mmdtest.compute_stat(tst_data)
        return s

        
# end QuadMMDGof

class QuadMMDGofOpt(gof.GofTest):
    """
    Goodness-of-fit test by drawing sample from the density p and test with the
    MMD test of Gretton et al., 2012. Optimize the kernel by the power
    criterion as in Sutherland et al., 2016. Need to split the data into
    training and test sets.

    H0: the sample follows p
    H1: the sample does not follow p

    p is specified to the constructor in the form of an UnnormalizedDensity.
    """

    def __init__(self, p, n_permute=400, alpha=0.01, seed=28):
        """
        p: an instance of UnnormalizedDensity
        k: an instance of Kernel
        n_permute: number of times to permute the samples to simulate from the 
            null distribution (permutation test)
        alpha: significance level 
        seed: random seed
        """
        super(QuadMMDGofOpt, self).__init__(p, alpha)
        self.n_permute = n_permute
        self.seed = seed
        ds = p.get_datasource()
        if ds is None:
            raise ValueError('%s test requires a density p which implements get_datasource(', str(QuadMMDGof))


    def perform_test(self, dat, candidate_kernels=None, return_mmdtest=False,
            tr_proportion=0.2, reg=1e-3):
        """
        dat: an instance of Data
        candidate_kernels: a list of Kernel's to choose from
        tr_proportion: proportion of sample to be used to choosing the best
            kernel
        reg: regularization parameter for the test power criterion 
        """
        with util.ContextTimer() as t:
            seed = self.seed
            p = self.p
            ds = p.get_datasource()
            p_sample = ds.sample(dat.sample_size(), seed=seed+77)
            xtr, xte = p_sample.split_tr_te(tr_proportion=tr_proportion, seed=seed+18)
            # ytr, yte are of type data.Data
            ytr, yte = dat.split_tr_te(tr_proportion=tr_proportion, seed=seed+12)

            # training and test data
            tr_tst_data = fdata.TSTData(xtr.data(), ytr.data())
            te_tst_data = fdata.TSTData(xte.data(), yte.data())

            if candidate_kernels is None:
                # Assume a Gaussian kernel. Construct a list of 
                # kernels to try based on multiples of the median heuristic
                med = util.meddistance(tr_tst_data.stack_xy(), 1000)
                list_gwidth = np.hstack( ( (med**2) *(2.0**np.linspace(-4, 4, 10) ) ) )
                list_gwidth.sort()
                candidate_kernels = [kernel.KGauss(gw2) for gw2 in list_gwidth]

            alpha = self.alpha

            # grid search to choose the best Gaussian width
            besti, powers = tst.QuadMMDTest.grid_search_kernel(tr_tst_data,
                    candidate_kernels, alpha, reg=reg)
            # perform test 
            best_ker = candidate_kernels[besti]
            mmdtest = tst.QuadMMDTest(best_ker, self.n_permute, alpha=alpha)
            results = mmdtest.perform_test(te_tst_data)
            if return_mmdtest:
                results['mmdtest'] = mmdtest

        results['time_secs'] = t.secs
        return results

    def compute_stat(self, dat):
        raise NotImplementedError('Not implemented yet.')

        
# end QuadMMDGofOpt
