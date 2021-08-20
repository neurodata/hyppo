"""
Module containing the two-sample tests of Jitkrittum et al., 2016 (NIPS 2016)
disguised as goodness-of-fit tests. Require the ability to
sample from the specified density. This module depends on external packages.

freqopttest https://github.com/wittawatj/interpretable-test

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


class GaussMETest(gof.GofTest):
    """
    Goodness-of-fit test by drawing sample from the density p and test with
    the mean embeddings test of Jitkrittum et al., 2016 (NIPS 2016). Use a
    Gaussian kernel. Test locations are specified, not optimized. 

    H0: the sample follows p
    H1: the sample does not follow p

    p is specified to the constructor in the form of an UnnormalizedDensity.
    """

    def __init__(self, p, gwidth2, test_locs, alpha=0.01, seed=28):
        """
        p: an instance of UnnormalizedDensity
        gwidth2: Gaussian width squared for the Gaussian kernel
        test_locs: J x d numpy array of J locations to test the difference
        alpha: significance level 
        """
        super(GaussMETest, self).__init__(p, alpha)
        self.gwidth2 = gwidth2
        self.test_locs = test_locs
        self.seed = seed
        ds = p.get_datasource()
        if ds is None:
            raise ValueError('%s test requires a density p which implements get_datasource(', str(GaussMETest))

        # Construct the ME test
        metest = tst.MeanEmbeddingTest(test_locs, gwidth2, alpha=alpha)
        self.metest = metest

    def perform_test(self, dat):
        """
        dat: an instance of Data
        """
        with util.ContextTimer() as t:
            seed = self.seed
            metest = self.metest
            p = self.p

            # Draw sample from p. #sample to draw is the same as that of dat
            ds = p.get_datasource()
            p_sample = ds.sample(dat.sample_size(), seed=seed)

            # Run the two-sample test on p_sample and dat
            # Make a two-sample test data
            tst_data = fdata.TSTData(p_sample.data(), dat.data())
            # Test 
            results = metest.perform_test(tst_data)

        results['time_secs'] = t.secs
        return results

    def compute_stat(self, dat):
        metest = self.metest
        p = self.p
        # Draw sample from p. #sample to draw is the same as that of dat
        ds = p.get_datasource()
        p_sample = ds.sample(dat.sample_size(), seed=self.seed)

        # Make a two-sample test data
        tst_data = fdata.TSTData(p_sample.data(), dat.data())
        s = metest.compute_stat(tst_data)
        return s

        
# end GaussMETest

class GaussMETestOpt(gof.GofTest):
    """
    Goodness-of-fit test by drawing sample from the density p and test with
    the mean embeddings test of Jitkrittum et al., 2016 (NIPS 2016). Use a
    Gaussian kernel. 
    
    For each given dataset dat, automatically optimize the test locations and
    the Gaussian width by dividing the dat into two disjoint halves: tr
    (training) and te (test set). The size of tr is specified by tr_proportion.

    H0: the sample follows p
    H1: the sample does not follow p

    p is specified to the constructor in the form of an UnnormalizedDensity.
    """

    def __init__(self, p, n_locs, tr_proportion=0.5, alpha=0.01, seed=29):
        """
        p: an instance of UnnormalizedDensity
        n_locs: number of test locations to use
        tr_proportion: proportion of the training set. A number in (0, 1).
        alpha: significance level 
        """
        super(GaussMETestOpt, self).__init__(p, alpha)
        if tr_proportion <= 0 or tr_proportion >= 1:
            raise ValueError('tr_proportion must be between 0 and 1 (exclusive)')
        self.n_locs = n_locs
        self.tr_proportion = tr_proportion
        self.seed = seed
        ds = p.get_datasource()
        if ds is None:
            raise ValueError('%s test requires a density p which implements get_datasource(', str(GaussMETest))

    def perform_test(self, dat, op=None, return_metest=False):
        """
        dat: an instance of Data
        op: a dictionary specifying options for the optimization of the ME test.
            Can be None (use default).
        """

        with util.ContextTimer() as t:
            metest, tr_tst_data, te_tst_data = self._get_metest_opt(dat, op)

            # Run the two-sample test 
            results = metest.perform_test(te_tst_data)

        results['time_secs'] = t.secs
        if return_metest:
            results['metest'] = metest
        return results

    def _get_metest_opt(self, dat, op=None):
        seed = self.seed
        if op is None:
            op = {'n_test_locs': self.n_locs, 'seed': seed+5, 'max_iter': 100, 
                 'batch_proportion': 1.0, 'locs_step_size': 1.0, 
                  'gwidth_step_size': 0.1, 'tol_fun': 1e-4, 'reg':1e-6}
        seed = self.seed
        alpha = self.alpha
        p = self.p
        # Draw sample from p. #sample to draw is the same as that of dat
        ds = p.get_datasource()
        p_sample = ds.sample(dat.sample_size(), seed=seed)
        xtr, xte = p_sample.split_tr_te(tr_proportion=self.tr_proportion, seed=seed+18)
        # ytr, yte are of type data.Data
        ytr, yte = dat.split_tr_te(tr_proportion=self.tr_proportion, seed=seed+12)

        # training and test data
        tr_tst_data = fdata.TSTData(xtr.data(), ytr.data())
        te_tst_data = fdata.TSTData(xte.data(), yte.data())

        # Train the ME test
        V_opt, gw2_opt, _ = tst.MeanEmbeddingTest.optimize_locs_width(tr_tst_data, alpha, **op)
        metest = tst.MeanEmbeddingTest(V_opt, gw2_opt, alpha)
        return metest, tr_tst_data, te_tst_data

    def compute_stat(self, dat, op=None):
        metest, tr_tst_data, te_tst_data = self._get_metest_opt(dat, op)

        # Make a two-sample test data
        s = metest.compute_stat(te_tst_data)
        return s

