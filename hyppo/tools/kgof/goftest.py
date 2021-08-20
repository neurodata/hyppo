"""
Module containing many types of goodness-of-fit test methods.
"""
from __future__ import division

from builtins import zip
from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object
from future.utils import with_metaclass
__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import autograd
import autograd.numpy as np
#import kgof.data as data
#import kgof.util as util
#import kgof.kernel as kernel
from . import data
from . import util
from . import kernel
import logging
import matplotlib.pyplot as plt

import scipy
import scipy.stats as stats

class GofTest(with_metaclass(ABCMeta, object)):
    """
    Abstract class for a goodness-of-fit test.
    """

    def __init__(self, p, alpha):
        """
        p: an UnnormalizedDensity
        alpha: significance level of the test
        """
        self.p = p
        self.alpha = alpha

    @abstractmethod
    def perform_test(self, dat):
        """perform the goodness-of-fit test and return values computed in a dictionary:
        {
            alpha: 0.01, 
            pvalue: 0.0002, 
            test_stat: 2.3, 
            h0_rejected: True, 
            time_secs: ...
        }

        dat: an instance of Data
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_stat(self, dat):
        """Compute the test statistic"""
        raise NotImplementedError()

# end of GofTest
#------------------------------------------------------

class H0Simulator(with_metaclass(ABCMeta, object)):
    """
    An abstract class representing a simulator to draw samples from the 
    null distribution. For some tests, these are needed to conduct the test.
    """

    def __init__(self, n_simulate, seed):
        """
        n_simulate: The number of times to simulate from the null distribution.
            Must be a positive integer.
        seed: a random seed
        """
        assert n_simulate > 0
        self.n_simulate = n_simulate
        self.seed = seed

    @abstractmethod
    def simulate(self, gof, dat):
        """
        gof: a GofTest
        dat: a Data (observed data)

        Simulate from the null distribution and return a dictionary. 
        One of the item is 
            sim_stats: a numpy array of stats.
        """
        raise NotImplementedError()

# end of H0Simulator
#-------------------

class FSSDH0SimCovObs(H0Simulator):
    """
    An asymptotic null distribution simulator for FSSD.  Simulate from the
    asymptotic null distribution given by the weighted sum of chi-squares. The
    eigenvalues (weights) are computed from the covarince matrix wrt. the
    observed sample. 
    This is not the correct null distribution; but has the correct asymptotic
    types-1 error at alpha.
    """
    def __init__(self, n_simulate=3000, seed=10):
        super(FSSDH0SimCovObs, self).__init__(n_simulate, seed)

    def simulate(self, gof, dat, fea_tensor=None):
        """
        fea_tensor: n x d x J feature matrix
        """
        assert isinstance(gof, FSSD)
        n_simulate = self.n_simulate
        seed = self.seed
        if fea_tensor is None:
            _, fea_tensor = gof.compute_stat(dat, return_feature_tensor=True)

        J = fea_tensor.shape[2]
        X = dat.data()
        n = X.shape[0]
        # n x d*J
        Tau = fea_tensor.reshape(n, -1)
        # Make sure it is a matrix i.e, np.cov returns a scalar when Tau is
        # 1d.
        cov = np.cov(Tau.T) + np.zeros((1, 1))
        #cov = Tau.T.dot(Tau/n)

        arr_nfssd, eigs = FSSD.list_simulate_spectral(cov, J, n_simulate,
                seed=self.seed)
        return {'sim_stats': arr_nfssd}

# end of FSSDH0SimCovObs
#-----------------------

class FSSDH0SimCovDraw(H0Simulator):
    """
    An asymptotic null distribution simulator for FSSD.  Simulate from the
    asymptotic null distribution given by the weighted sum of chi-squares. The
    eigenvalues (weights) are computed from the covarince matrix wrt. the
    sample drawn from p (the density to test against). 
    
    - The UnnormalizedDensity p is required to implement get_datasource() method.
    """
    def __init__(self, n_draw=2000, n_simulate=3000, seed=10):
        """
        n_draw: number of samples to draw from the UnnormalizedDensity p
        """
        super(FSSDH0SimCovDraw, self).__init__(n_simulate, seed)
        self.n_draw = n_draw

    def simulate(self, gof, dat, fea_tensor=None):
        """
        fea_tensor: n x d x J feature matrix

        This method does not use dat.
        """
        dat = None
        #assert isinstance(gof, FSSD)
        # p = an UnnormalizedDensity
        p = gof.p
        ds = p.get_datasource()
        if ds is None:
            raise ValueError('DataSource associated with p must be available.')
        Xdraw = ds.sample(n=self.n_draw, seed=self.seed)
        _, fea_tensor = gof.compute_stat(Xdraw, return_feature_tensor=True)

        X = Xdraw.data()
        J = fea_tensor.shape[2]
        n = self.n_draw
        # n x d*J
        Tau = fea_tensor.reshape(n, -1)
        # Make sure it is a matrix i.e, np.cov returns a scalar when Tau is
        # 1d.
        #cov = np.cov(Tau.T) + np.zeros((1, 1))
        cov = old_div(Tau.T.dot(Tau),n) + np.zeros((1, 1))
        n_simulate = self.n_simulate

        arr_nfssd, eigs = FSSD.list_simulate_spectral(cov, J, n_simulate,
                seed=self.seed)
        return {'sim_stats': arr_nfssd}

# end of FSSDH0SimCovDraw
#-----------------------

class FSSD(GofTest):
    """
    Goodness-of-fit test using The Finite Set Stein Discrepancy statistic.
    and a set of paired test locations. The statistic is n*FSSD^2. 
    The statistic can be negative because of the unbiased estimator.

    H0: the sample follows p
    H1: the sample does not follow p

    p is specified to the constructor in the form of an UnnormalizedDensity.
    """

    #NULLSIM_* are constants used to choose the way to simulate from the null 
    #distribution to do the test.


    # Same as NULLSIM_COVQ; but assume that sample can be drawn from p. 
    # Use the drawn sample to compute the covariance.
    NULLSIM_COVP = 1


    def __init__(self, p, k, V, null_sim=FSSDH0SimCovObs(n_simulate=3000,
        seed=101), alpha=0.01):
        """
        p: an instance of UnnormalizedDensity
        k: a DifferentiableKernel object
        V: J x dx numpy array of J locations to test the difference
        null_sim: an instance of H0Simulator for simulating from the null distribution.
        alpha: significance level 
        """
        super(FSSD, self).__init__(p, alpha)
        self.k = k
        self.V = V 
        self.null_sim = null_sim

    def perform_test(self, dat, return_simulated_stats=False):
        """
        dat: an instance of Data
        """
        with util.ContextTimer() as t:
            alpha = self.alpha
            null_sim = self.null_sim
            n_simulate = null_sim.n_simulate
            X = dat.data()
            n = X.shape[0]
            J = self.V.shape[0]

            nfssd, fea_tensor = self.compute_stat(dat, return_feature_tensor=True)
            sim_results = null_sim.simulate(self, dat, fea_tensor)
            arr_nfssd = sim_results['sim_stats']

            # approximate p-value with the permutations 
            pvalue = np.mean(arr_nfssd > nfssd)

        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': nfssd,
                'h0_rejected': pvalue < alpha, 'n_simulate': n_simulate,
                'time_secs': t.secs, 
                }
        if return_simulated_stats:
            results['sim_stats'] = arr_nfssd
        return results

    def compute_stat(self, dat, return_feature_tensor=False):
        """
        The statistic is n*FSSD^2.
        """
        X = dat.data()
        n = X.shape[0]

        # n x d x J
        Xi = self.feature_tensor(X)
        unscaled_mean = FSSD.ustat_h1_mean_variance(Xi, return_variance=False)
        stat = n*unscaled_mean

        #print 'Xi: {0}'.format(Xi)
        #print 'Tau: {0}'.format(Tau)
        #print 't1: {0}'.format(t1)
        #print 't2: {0}'.format(t2)
        #print 'stat: {0}'.format(stat)
        if return_feature_tensor:
            return stat, Xi
        else:
            return stat

    def get_H1_mean_variance(self, dat):
        """
        Return the mean and variance under H1 of the test statistic (divided by
        n).
        """
        X = dat.data()
        Xi = self.feature_tensor(X)
        mean, variance = FSSD.ustat_h1_mean_variance(Xi, return_variance=True)
        return mean, variance

    def feature_tensor(self, X):
        """
        Compute the feature tensor which is n x d x J.
        The feature tensor can be used to compute the statistic, and the
        covariance matrix for simulating from the null distribution.

        X: n x d data numpy array

        return an n x d x J numpy array
        """
        k = self.k
        J = self.V.shape[0]
        n, d = X.shape
        # n x d matrix of gradients
        grad_logp = self.p.grad_log(X)
        #assert np.all(util.is_real_num(grad_logp))
        # n x J matrix
        #print 'V'
        #print self.V
        K = k.eval(X, self.V)
        #assert np.all(util.is_real_num(K))

        list_grads = np.array([np.reshape(k.gradX_y(X, v), (1, n, d)) for v in self.V])
        stack0 = np.concatenate(list_grads, axis=0)
        #a numpy array G of size n x d x J such that G[:, :, J]
        #    is the derivative of k(X, V_j) with respect to X.
        dKdV = np.transpose(stack0, (1, 2, 0))

        # n x d x J tensor
        grad_logp_K = util.outer_rows(grad_logp, K)
        #print 'grad_logp'
        #print grad_logp.dtype
        #print grad_logp
        #print 'K'
        #print K
        Xi = old_div((grad_logp_K + dKdV),np.sqrt(d*J))
        #Xi = (grad_logp_K + dKdV)
        return Xi


    @staticmethod
    def power_criterion(p, dat, k, test_locs, reg=1e-2, use_unbiased=True, 
            use_2terms=False):
        """
        Compute the mean and standard deviation of the statistic under H1.
        Return mean/sd.
        use_2terms: True if the objective should include the first term in the power 
            expression. This term carries the test threshold and is difficult to 
            compute (depends on the optimized test locations). If True, then 
            the objective will be -1/(n**0.5*sigma_H1) + n**0.5 FSSD^2/sigma_H1, 
            which ignores the test threshold in the first term.
        """
        X = dat.data()
        n = X.shape[0]
        V = test_locs
        fssd = FSSD(p, k, V, null_sim=None)
        fea_tensor = fssd.feature_tensor(X)
        u_mean, u_variance = FSSD.ustat_h1_mean_variance(fea_tensor,
                return_variance=True, use_unbiased=use_unbiased)

        # mean/sd criterion 
        sigma_h1 = np.sqrt(u_variance + reg)
        ratio = old_div(u_mean,sigma_h1) 
        if use_2terms:
            obj = old_div(-1.0,(np.sqrt(n)*sigma_h1)) + np.sqrt(n)*ratio
            #print obj
        else:
            obj = ratio
        return obj

    @staticmethod
    def ustat_h1_mean_variance(fea_tensor, return_variance=True,
            use_unbiased=True):
        """
        Compute the mean and variance of the asymptotic normal distribution 
        under H1 of the test statistic.

        fea_tensor: feature tensor obtained from feature_tensor()
        return_variance: If false, avoid computing and returning the variance.
        use_unbiased: If True, use the unbiased version of the mean. Can be
            negative.

        Return the mean [and the variance]
        """
        Xi = fea_tensor
        n, d, J = Xi.shape
        #print 'Xi'
        #print Xi
        #assert np.all(util.is_real_num(Xi))
        assert n > 1, 'Need n > 1 to compute the mean of the statistic.'
        # n x d*J
        # Tau = Xi.reshape(n, d*J)
        Tau = np.reshape(Xi, [n, d*J])
        if use_unbiased:
            t1 = np.sum(np.mean(Tau, 0)**2)*(old_div(n,float(n-1)))
            t2 = old_div(np.sum(np.mean(Tau**2, 0)),float(n-1))
            # stat is the mean
            stat = t1 - t2
        else:
            stat = np.sum(np.mean(Tau, 0)**2)

        if not return_variance:
            return stat

        # compute the variance 
        # mu: d*J vector
        mu = np.mean(Tau, 0)
        variance = 4*np.mean(np.dot(Tau, mu)**2) - 4*np.sum(mu**2)**2
        return stat, variance

    @staticmethod 
    def list_simulate_spectral(cov, J, n_simulate=1000, seed=82):
        """
        Simulate the null distribution using the spectrums of the covariance
        matrix.  This is intended to be used to approximate the null
        distribution.

        Return (a numpy array of simulated n*FSSD values, eigenvalues of cov)
        """
        # eigen decompose 
        eigs, _ = np.linalg.eig(cov)
        eigs = np.real(eigs)
        # sort in decreasing order 
        eigs = -np.sort(-eigs)
        sim_fssds = FSSD.simulate_null_dist(eigs, J, n_simulate=n_simulate,
                seed=seed)
        return sim_fssds, eigs

    @staticmethod 
    def simulate_null_dist(eigs, J, n_simulate=2000, seed=7):
        """
        Simulate the null distribution using the spectrums of the covariance 
        matrix of the U-statistic. The simulated statistic is n*FSSD^2 where
        FSSD is an unbiased estimator.

        - eigs: a numpy array of estimated eigenvalues of the covariance
          matrix. eigs is of length d*J, where d is the input dimension, and 
        - J: the number of test locations.

        Return a numpy array of simulated statistics.
        """
        d = old_div(len(eigs),J)
        assert d>0
        # draw at most d x J x block_size values at a time
        block_size = max(20, int(old_div(1000.0,(d*J))))
        fssds = np.zeros(n_simulate)
        from_ind = 0
        with util.NumpySeedContext(seed=seed):
            while from_ind < n_simulate:
                to_draw = min(block_size, n_simulate-from_ind)
                # draw chi^2 random variables. 
                chi2 = np.random.randn(d*J, to_draw)**2

                # an array of length to_draw 
                sim_fssds = eigs.dot(chi2-1.0)
                # store 
                end_ind = from_ind+to_draw
                fssds[from_ind:end_ind] = sim_fssds
                from_ind = end_ind
        return fssds

    @staticmethod
    def fssd_grid_search_kernel(p, dat, test_locs, list_kernel):
        """
        Linear search for the best kernel in the list that maximizes 
        the test power criterion, fixing the test locations to V.

        - p: UnnormalizedDensity
        - dat: a Data object
        - list_kernel: list of kernel candidates 

        return: (best kernel index, array of test power criteria)
        """
        V = test_locs
        X = dat.data()
        n_cand = len(list_kernel)
        objs = np.zeros(n_cand)
        for i in range(n_cand):
            ki = list_kernel[i]
            objs[i] = FSSD.power_criterion(p, dat, ki, test_locs)
            logging.info('(%d), obj: %5.4g, k: %s' %(i, objs[i], str(ki)))

        #Widths that come early in the list 
        # are preferred if test powers are equal.
        #bestij = np.unravel_index(objs.argmax(), objs.shape)
        besti = objs.argmax()
        return besti, objs

# end of FSSD
# --------------------------------------

class GaussFSSD(FSSD):
    """
    FSSD using an isotropic Gaussian kernel.
    """
    def __init__(self, p, sigma2, V, alpha=0.01, n_simulate=3000, seed=10):
        k = kernel.KGauss(sigma2)
        null_sim = FSSDH0SimCovObs(n_simulate=n_simulate, seed=seed)
        super(GaussFSSD, self).__init__(p, k, V, null_sim, alpha)

    @staticmethod 
    def power_criterion(p, dat, gwidth, test_locs, reg=1e-2, use_2terms=False):
        """
        use_2terms: True if the objective should include the first term in the power 
            expression. This term carries the test threshold and is difficult to 
            compute (depends on the optimized test locations). If True, then 
            the objective will be -1/(n**0.5*sigma_H1) + n**0.5 FSSD^2/sigma_H1, 
            which ignores the test threshold in the first term.
        """
        k = kernel.KGauss(gwidth)
        return FSSD.power_criterion(p, dat, k, test_locs, reg, use_2terms=use_2terms)

    @staticmethod
    def optimize_auto_init(p, dat, J, **ops):
        """
        Optimize parameters by calling optimize_locs_widths(). Automatically 
        initialize the test locations and the Gaussian width.

        Return optimized locations, Gaussian width, optimization info
        """
        assert J>0
        # Use grid search to initialize the gwidth
        X = dat.data()
        n_gwidth_cand = 5
        gwidth_factors = 2.0**np.linspace(-3, 3, n_gwidth_cand) 
        med2 = util.meddistance(X, 1000)**2

        k = kernel.KGauss(med2*2)
        # fit a Gaussian to the data and draw to initialize V0
        V0 = util.fit_gaussian_draw(X, J, seed=829, reg=1e-6)
        list_gwidth = np.hstack( ( (med2)*gwidth_factors ) )
        besti, objs = GaussFSSD.grid_search_gwidth(p, dat, V0, list_gwidth)
        gwidth = list_gwidth[besti]
        assert util.is_real_num(gwidth), 'gwidth not real. Was %s'%str(gwidth)
        assert gwidth > 0, 'gwidth not positive. Was %.3g'%gwidth
        logging.info('After grid search, gwidth=%.3g'%gwidth)

        
        V_opt, gwidth_opt, info = GaussFSSD.optimize_locs_widths(p, dat,
                gwidth, V0, **ops) 

        # set the width bounds
        #fac_min = 5e-2
        #fac_max = 5e3
        #gwidth_lb = fac_min*med2
        #gwidth_ub = fac_max*med2
        #gwidth_opt = max(gwidth_lb, min(gwidth_opt, gwidth_ub))
        return V_opt, gwidth_opt, info

    @staticmethod
    def grid_search_gwidth(p, dat, test_locs, list_gwidth):
        """
        Linear search for the best Gaussian width in the list that maximizes 
        the test power criterion, fixing the test locations. 

        - V: a J x dx np-array for J test locations 

        return: (best width index, list of test power objectives)
        """
        list_gauss_kernel = [kernel.KGauss(gw) for gw in list_gwidth]
        besti, objs = FSSD.fssd_grid_search_kernel(p, dat, test_locs,
                list_gauss_kernel)
        return besti, objs

    @staticmethod
    def optimize_locs_widths(p, dat, gwidth0, test_locs0, reg=1e-2,
            max_iter=100,  tol_fun=1e-5, disp=False, locs_bounds_frac=100,
            gwidth_lb=None, gwidth_ub=None, use_2terms=False,
            ):
        """
        Optimize the test locations and the Gaussian kernel width by 
        maximizing a test power criterion. data should not be the same data as
        used in the actual test (i.e., should be a held-out set). 
        This function is deterministic.

        - data: a Data object
        - test_locs0: Jxd numpy array. Initial V.
        - reg: reg to add to the mean/sqrt(variance) criterion to become
            mean/sqrt(variance + reg)
        - gwidth0: initial value of the Gaussian width^2
        - max_iter: #gradient descent iterations
        - tol_fun: termination tolerance of the objective value
        - disp: True to print convergence messages
        - locs_bounds_frac: When making box bounds for the test_locs, extend
            the box defined by coordinate-wise min-max by std of each coordinate
            multiplied by this number.
        - gwidth_lb: absolute lower bound on the Gaussian width^2
        - gwidth_ub: absolute upper bound on the Gaussian width^2
        - use_2terms: If True, then besides the signal-to-noise ratio
          criterion, the objective function will also include the first term
          that is dropped.

        #- If the lb, ub bounds are None, use fraction of the median heuristics 
        #    to automatically set the bounds.
        
        Return (V test_locs, gaussian width, optimization info log)
        """
        J = test_locs0.shape[0]
        X = dat.data()
        n, d = X.shape

        # Parameterize the Gaussian width with its square root (then square later)
        # to automatically enforce the positivity.
        def obj(sqrt_gwidth, V):
            return -GaussFSSD.power_criterion(
                    p, dat, sqrt_gwidth**2, V, reg=reg, use_2terms=use_2terms)
        flatten = lambda gwidth, V: np.hstack((gwidth, V.reshape(-1)))
        def unflatten(x):
            sqrt_gwidth = x[0]
            V = np.reshape(x[1:], (J, d))
            return sqrt_gwidth, V

        def flat_obj(x):
            sqrt_gwidth, V = unflatten(x)
            return obj(sqrt_gwidth, V)
        # gradient
        #grad_obj = autograd.elementwise_grad(flat_obj)
        # Initial point
        x0 = flatten(np.sqrt(gwidth0), test_locs0)
        
        #make sure that the optimized gwidth is not too small or too large.
        fac_min = 1e-2 
        fac_max = 1e2
        med2 = util.meddistance(X, subsample=1000)**2
        if gwidth_lb is None:
            gwidth_lb = max(fac_min*med2, 1e-3)
        if gwidth_ub is None:
            gwidth_ub = min(fac_max*med2, 1e5)

        # Make a box to bound test locations
        X_std = np.std(X, axis=0)
        # X_min: length-d array
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        # V_lb: J x d
        V_lb = np.tile(X_min - locs_bounds_frac*X_std, (J, 1))
        V_ub = np.tile(X_max + locs_bounds_frac*X_std, (J, 1))
        # (J*d+1) x 2. Take square root because we parameterize with the square
        # root
        x0_lb = np.hstack((np.sqrt(gwidth_lb), np.reshape(V_lb, -1)))
        x0_ub = np.hstack((np.sqrt(gwidth_ub), np.reshape(V_ub, -1)))
        x0_bounds = list(zip(x0_lb, x0_ub))

        # optimize. Time the optimization as well.
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
        grad_obj = autograd.elementwise_grad(flat_obj)
        with util.ContextTimer() as timer:
            opt_result = scipy.optimize.minimize(
              flat_obj, x0, method='L-BFGS-B', 
              bounds=x0_bounds,
              tol=tol_fun, 
              options={
                  'maxiter': max_iter, 'ftol': tol_fun, 'disp': disp,
                  'gtol': 1.0e-07,
                  },
              jac=grad_obj,
            )

        opt_result = dict(opt_result)
        opt_result['time_secs'] = timer.secs
        x_opt = opt_result['x']
        sq_gw_opt, V_opt = unflatten(x_opt)
        gw_opt = sq_gw_opt**2

        assert util.is_real_num(gw_opt), 'gw_opt is not real. Was %s' % str(gw_opt)

        return V_opt, gw_opt, opt_result

# end of class GaussFSSD


def bootstrapper_rademacher(n):
    """
    Produce a sequence of i.i.d {-1, 1} random variables.
    Suitable for boostrapping on an i.i.d. sample.
    """
    return 2.0*np.random.randint(0, 1+1, n)-1.0

def bootstrapper_multinomial(n):
    """
    Produce a sequence of i.i.d Multinomial(n; 1/n,... 1/n) random variables.
    This is described on page 5 of Liu et al., 2016 (ICML 2016).
    """
    import warnings
    warnings.warn('Somehow bootstrapper_multinomial() does not give the right null distribution.')
    M = np.random.multinomial(n, old_div(np.ones(n),float(n)), size=1) 
    return M.reshape(-1) - old_div(1.0,n)



class IMQFSSD(FSSD):
    """
    FSSD using the inverse multiquadric kernel (IMQ).

    k(x,y) = (c^2 + ||x-y||^2)^b 
    where c > 0 and b < 0. 
    """
    def __init__(self, p, b, c, V, alpha=0.01, n_simulate=3000, seed=10):
        """
        n_simulate: number of times to draw from the null distribution.
        """
        k = kernel.KIMQ(b=b, c=c)
        null_sim = FSSDH0SimCovObs(n_simulate=n_simulate, seed=seed)
        super(IMQFSSD, self).__init__(p, k, V, null_sim, alpha)

    @staticmethod 
    def power_criterion(p, dat, b, c, test_locs, reg=1e-2):
        k = kernel.KIMQ(b=b, c=c)
        return FSSD.power_criterion(p, dat, k, test_locs, reg)

    #@staticmethod
    #def optimize_auto_init(p, dat, J, **ops):
    #    """
    #    Optimize parameters by calling optimize_locs_widths(). Automatically 
    #    initialize the test locations and the Gaussian width.

    #    Return optimized locations, Gaussian width, optimization info
    #    """
    #    assert J>0
    #    # Use grid search to initialize the gwidth
    #    X = dat.data()
    #    n_gwidth_cand = 5
    #    gwidth_factors = 2.0**np.linspace(-3, 3, n_gwidth_cand) 
    #    med2 = util.meddistance(X, 1000)**2

    #    k = kernel.KGauss(med2*2)
    #    # fit a Gaussian to the data and draw to initialize V0
    #    V0 = util.fit_gaussian_draw(X, J, seed=829, reg=1e-6)
    #    list_gwidth = np.hstack( ( (med2)*gwidth_factors ) )
    #    besti, objs = GaussFSSD.grid_search_gwidth(p, dat, V0, list_gwidth)
    #    gwidth = list_gwidth[besti]
    #    assert util.is_real_num(gwidth), 'gwidth not real. Was %s'%str(gwidth)
    #    assert gwidth > 0, 'gwidth not positive. Was %.3g'%gwidth
    #    logging.info('After grid search, gwidth=%.3g'%gwidth)

        
    #    V_opt, gwidth_opt, info = GaussFSSD.optimize_locs_widths(p, dat,
    #            gwidth, V0, **ops) 

    #    # set the width bounds
    #    #fac_min = 5e-2
    #    #fac_max = 5e3
    #    #gwidth_lb = fac_min*med2
    #    #gwidth_ub = fac_max*med2
    #    #gwidth_opt = max(gwidth_lb, min(gwidth_opt, gwidth_ub))
    #    return V_opt, gwidth_opt, info

    @staticmethod
    def optimize_locs(p, dat, b, c, test_locs0, reg=1e-5, max_iter=100,
            tol_fun=1e-5, disp=False, locs_bounds_frac=100):
        """
        Optimize just the test locations by maximizing a test power criterion,
        keeping the kernel parameters b, c fixed to the specified values. data
        should not be the same data as used in the actual test (i.e., should be
        a held-out set). This function is deterministic.

        - p: an UnnormalizedDensity specifying the problem
        - dat: a Data object
        - b, c: kernel parameters of the IMQ kernel. Not optimized.
        - test_locs0: Jxd numpy array. Initial V.
        - reg: reg to add to the mean/sqrt(variance) criterion to become
            mean/sqrt(variance + reg)
        - max_iter: #gradient descent iterations
        - tol_fun: termination tolerance of the objective value
        - disp: True to print convergence messages
        - locs_bounds_frac: When making box bounds for the test_locs, extend
            the box defined by coordinate-wise min-max by std of each coordinate
            multiplied by this number.
        
        Return (V test_locs, optimization info log)
        """
        J = test_locs0.shape[0]
        X = dat.data()
        n, d = X.shape

        def obj(V):
            return -IMQFSSD.power_criterion(p, dat, b, c, V, reg=reg)

        flatten = lambda V: np.reshape(V, -1)

        def unflatten(x):
            V = np.reshape(x, (J, d))
            return V

        def flat_obj(x):
            V = unflatten(x)
            return obj(V)

        # Initial point
        x0 = flatten(test_locs0)

        # Make a box to bound test locations
        X_std = np.std(X, axis=0)
        # X_min: length-d array
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        # V_lb: J x d
        V_lb = np.tile(X_min - locs_bounds_frac*X_std, (J, 1))
        V_ub = np.tile(X_max + locs_bounds_frac*X_std, (J, 1))
        # (J*d) x 2. 
        x0_bounds = list(zip(V_lb.reshape(-1)[:, np.newaxis], V_ub.reshape(-1)[:, np.newaxis]))

        # optimize. Time the optimization as well.
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
        grad_obj = autograd.elementwise_grad(flat_obj)
        with util.ContextTimer() as timer:
            opt_result = scipy.optimize.minimize(
              flat_obj, x0, method='L-BFGS-B', 
              bounds=x0_bounds,
              tol=tol_fun, 
              options={
                  'maxiter': max_iter, 'ftol': tol_fun, 'disp': disp,
                  'gtol': 1.0e-06,
                  },
              jac=grad_obj,
            )

        opt_result = dict(opt_result)
        opt_result['time_secs'] = timer.secs
        x_opt = opt_result['x']
        V_opt = unflatten(x_opt)
        return V_opt, opt_result

    @staticmethod
    def optimize_locs_params(p, dat, b0, c0, test_locs0, reg=1e-2,
            max_iter=100,  tol_fun=1e-5, disp=False, locs_bounds_frac=100,
            b_lb= -20.0, b_ub= -1e-4, c_lb=1e-6, c_ub=1e3,
            ):
        """
        Optimize the test locations and the the two parameters (b and c) of the
        IMQ kernel by maximizing the test power criterion. 
             k(x,y) = (c^2 + ||x-y||^2)^b 
            where c > 0 and b < 0. 
        data should not be the same data as used in the actual test (i.e.,
        should be a held-out set). This function is deterministic.

        - p: UnnormalizedDensity specifying the problem.
        - b0: initial parameter value for b (in the kernel)
        - c0: initial parameter value for c (in the kernel)
        - dat: a Data object (training set)
        - test_locs0: Jxd numpy array. Initial V.
        - reg: reg to add to the mean/sqrt(variance) criterion to become
            mean/sqrt(variance + reg)
        - max_iter: #gradient descent iterations
        - tol_fun: termination tolerance of the objective value
        - disp: True to print convergence messages
        - locs_bounds_frac: When making box bounds for the test_locs, extend
            the box defined by coordinate-wise min-max by std of each coordinate
            multiplied by this number.
        - b_lb: absolute lower bound on b. b is always < 0.
        - b_ub: absolute upper bound on b
        - c_lb: absolute lower bound on c. c is always > 0.
        - c_ub: absolute upper bound on c

        #- If the lb, ub bounds are None 
        
        Return (V test_locs, b, c, optimization info log)
        """

        """
        In the optimization, we will parameterize b with its square root.
        Square back and negate to form b. c is not parameterized in any special
        way since it enters to the kernel with c^2. Absolute value of c will be
        taken to make sure it is positive.
        """
        J = test_locs0.shape[0]
        X = dat.data()
        n, d = X.shape

        def obj(sqrt_neg_b, c, V):
            b = -sqrt_neg_b**2
            return -IMQFSSD.power_criterion(p, dat, b, c, V, reg=reg)

        flatten = lambda sqrt_neg_b, c, V: np.hstack((sqrt_neg_b, c, V.reshape(-1)))
        def unflatten(x):
            sqrt_neg_b = x[0]
            c = x[1]
            V = np.reshape(x[2:], (J, d))
            return sqrt_neg_b, c, V

        def flat_obj(x):
            sqrt_neg_b, c, V = unflatten(x)
            return obj(sqrt_neg_b, c, V)

        # gradient
        #grad_obj = autograd.elementwise_grad(flat_obj)
        # Initial point
        b02 = np.sqrt(-b0)
        x0 = flatten(b02, c0, test_locs0)
        
        # Make a box to bound test locations
        X_std = np.std(X, axis=0)
        # X_min: length-d array
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)

        # V_lb: J x d
        V_lb = np.tile(X_min - locs_bounds_frac*X_std, (J, 1))
        V_ub = np.tile(X_max + locs_bounds_frac*X_std, (J, 1))

        # (J*d+2) x 2. Make sure to bound the reparamterized values (not the original)
        """
        For b, b2 := sqrt(-b)
            lb <= b <= ub < 0 means 

            sqrt(-ub) <= b2 <= sqrt(-lb)
            Note the positions of ub, lb.
        """
        x0_lb = np.hstack((np.sqrt(-b_ub), c_lb, np.reshape(V_lb, -1)))
        x0_ub = np.hstack((np.sqrt(-b_lb), c_ub, np.reshape(V_ub, -1)))
        x0_bounds = list(zip(x0_lb, x0_ub))

        # optimize. Time the optimization as well.
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
        grad_obj = autograd.elementwise_grad(flat_obj)
        with util.ContextTimer() as timer:
            opt_result = scipy.optimize.minimize(
              flat_obj, x0, method='L-BFGS-B', 
              bounds=x0_bounds,
              tol=tol_fun, 
              options={
                  'maxiter': max_iter, 'ftol': tol_fun, 'disp': disp,
                  'gtol': 1.0e-06,
                  },
              jac=grad_obj,
            )

        opt_result = dict(opt_result)
        opt_result['time_secs'] = timer.secs
        x_opt = opt_result['x']
        sqrt_neg_b, c, V_opt = unflatten(x_opt)
        b = -sqrt_neg_b**2
        assert util.is_real_num(b), 'b is not real. Was {}'.format(b)
        assert b < 0
        assert util.is_real_num(c), 'c is not real. Was {}'.format(c)
        assert c > 0

        return V_opt, b, c, opt_result


# end of class IMQFSSD

class KernelSteinTest(GofTest):
    """
    Goodness-of-fit test using kernelized Stein discrepancy test of 
    Chwialkowski et al., 2016 and Liu et al., 2016 in ICML 2016.
    Mainly follow the details in Chwialkowski et al., 2016.
    The test statistic is n*V_n where V_n is a V-statistic.

    - This test runs in O(n^2 d^2) time.

    H0: the sample follows p
    H1: the sample does not follow p

    p is specified to the constructor in the form of an UnnormalizedDensity.
    """

    def __init__(self, p, k, bootstrapper=bootstrapper_rademacher, alpha=0.01,
            n_simulate=500, seed=11):
        """
        p: an instance of UnnormalizedDensity
        k: a KSTKernel object
        bootstrapper: a function: (n) |-> numpy array of n weights 
            to be multiplied in the double sum of the test statistic for generating 
            bootstrap samples from the null distribution.
        alpha: significance level 
        n_simulate: The number of times to simulate from the null distribution
            by bootstrapping. Must be a positive integer.
        """
        super(KernelSteinTest, self).__init__(p, alpha)
        self.k = k
        self.bootstrapper = bootstrapper
        self.n_simulate = n_simulate
        self.seed = seed

    def perform_test(self, dat, return_simulated_stats=False, return_ustat_gram=False):
        """
        dat: a instance of Data
        """
        with util.ContextTimer() as t:
            alpha = self.alpha
            n_simulate = self.n_simulate
            X = dat.data()
            n = X.shape[0]

            _, H = self.compute_stat(dat, return_ustat_gram=True)
            test_stat = n*np.mean(H)
            # bootrapping
            sim_stats = np.zeros(n_simulate)
            with util.NumpySeedContext(seed=self.seed):
                for i in range(n_simulate):
                   W = self.bootstrapper(n)
                   # n * [ (1/n^2) * \sum_i \sum_j h(x_i, x_j) w_i w_j ]
                   boot_stat = W.dot(H.dot(old_div(W,float(n))))
                   # This is a bootstrap version of n*V_n
                   sim_stats[i] = boot_stat
 
            # approximate p-value with the permutations 
            pvalue = np.mean(sim_stats > test_stat)
 
        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': test_stat,
                 'h0_rejected': pvalue < alpha, 'n_simulate': n_simulate,
                 'time_secs': t.secs, 
                 }
        if return_simulated_stats:
            results['sim_stats'] = sim_stats
        if return_ustat_gram:
            results['H'] = H
            
        return results


    def compute_stat(self, dat, return_ustat_gram=False):
        """
        Compute the V statistic as in Section 2.2 of Chwialkowski et al., 2016.
        return_ustat_gram: If True, then return the n x n matrix used to
            compute the statistic (by taking the mean of all the elements)
        """
        X = dat.data()
        n, d = X.shape
        k = self.k
        # n x d matrix of gradients
        grad_logp = self.p.grad_log(X)
        # n x n
        gram_glogp = grad_logp.dot(grad_logp.T)
        # n x n
        K = k.eval(X, X)

        B = np.zeros((n, n))
        C = np.zeros((n, n))
        for i in range(d):
            grad_logp_i = grad_logp[:, i]
            B += k.gradX_Y(X, X, i)*grad_logp_i
            C += (k.gradY_X(X, X, i).T * grad_logp_i).T

        H = K*gram_glogp + B + C + k.gradXY_sum(X, X)
        # V-statistic
        stat = n*np.mean(H)
        if return_ustat_gram:
            return stat, H
        else:
            return stat

        #print 't1: {0}'.format(t1)
        #print 't2: {0}'.format(t2)
        #print 't3: {0}'.format(t3)
        #print 't4: {0}'.format(t4)

# end KernelSteinTest

class LinearKernelSteinTest(GofTest):
    """
    Goodness-of-fit test using the linear-version of kernelized Stein
    discrepancy test of Liu et al., 2016 in ICML 2016. Described in Liu et al.,
    2016. 
    - This test runs in O(n d^2) time.
    - test stat = sqrt(n_half)*linear-time Stein discrepancy
    - Asymptotically normal under both H0 and H1.

    H0: the sample follows p
    H1: the sample does not follow p

    p is specified to the constructor in the form of an UnnormalizedDensity.
    """

    def __init__(self, p, k, alpha=0.01, seed=11):
        """
        p: an instance of UnnormalizedDensity
        k: a LinearKSTKernel object
        alpha: significance level 
        n_simulate: The number of times to simulate from the null distribution
            by bootstrapping. Must be a positive integer.
        """
        super(LinearKernelSteinTest, self).__init__(p, alpha)
        self.k = k
        self.seed = seed

    def perform_test(self, dat):
        """
        dat: a instance of Data
        """
        with util.ContextTimer() as t:
            alpha = self.alpha
            X = dat.data()
            n = X.shape[0]

            # H: length-n vector
            _, H = self.compute_stat(dat, return_pointwise_stats=True)
            test_stat = np.sqrt(old_div(n,2))*np.mean(H)
            stat_var = np.mean(H**2) 
            pvalue = stats.norm.sf(test_stat, loc=0, scale=np.sqrt(stat_var) )
 
        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': test_stat,
                 'h0_rejected': pvalue < alpha, 'time_secs': t.secs, 
                 }
        return results


    def compute_stat(self, dat, return_pointwise_stats=False):
        """
        Compute the linear-time statistic described in Eq. 17 of Liu et al., 2016
        """
        X = dat.data()
        n, d = X.shape
        k = self.k
        # Divide the sample into two halves of equal size. 
        n_half = old_div(n,2)
        X1 = X[:n_half, :]
        # May throw away last sample
        X2 = X[n_half:(2*n_half), :]
        assert X1.shape[0] == n_half
        assert X2.shape[0] == n_half
        # score vectors
        S1 = self.p.grad_log(X1)
        # n_half x d
        S2 = self.p.grad_log(X2)
        Kvec = k.pair_eval(X1, X2)

        A = np.sum(S1*S2, 1)*Kvec
        B = np.sum(S2*k.pair_gradX_Y(X1, X2), 1)
        C = np.sum(S1*k.pair_gradY_X(X1, X2), 1)
        D = k.pair_gradXY_sum(X1, X2)

        H = A + B + C + D
        assert len(H) == n_half
        stat = np.mean(H)
        if return_pointwise_stats:
            return stat, H
        else:
            return stat

# end LinearKernelSteinTest

class SteinWitness(object):
    """
    Construct a callable object representing the Stein witness function. 
    The witness function g is defined as in Eq. 1 of 

        A Linear-Time Kernel Goodness-of-Fit Test
        Wittawat Jitkrittum, Wenkai Xu, Zoltan Szabo, Kenji Fukumizu,
        Arthur Gretton
        NIPS 2017

    The witness function requires taking an expectation over the sample
    generating distribution. This is approximated by an empirical
    expectation using the sample in the input (dat). The witness function
    is a d-variate (d = dimension of the data) function, which depends on 
    the kernel k and the model p.

    The constructed object can be called as if it is a function: (J x d) numpy
    array |-> (J x d) outputs 
    """
    def __init__(self, p, k, dat):
        """
        :params p: an UnnormalizedDensity object
        :params k: a DifferentiableKernel
        :params dat: a kgof.data.Data
        """
        self.p = p
        self.k = k
        self.dat = dat

    def __call__(self, V):
        """
        :params V: a numpy array of size J x d (data matrix)

        :returns (J x d) numpy array representing witness evaluations at the J
            points.
        """
        J = V.shape[0]
        X = self.dat.data()
        n, d = X.shape
        # construct the feature tensor (n x d x J)
        fssd = FSSD(self.p, self.k, V, null_sim=None, alpha=None)

        # When X, V contain many points, this can use a lot of memory.
        # Process chunk by chunk.
        block_rows = util.constrain(50000//(d*J), 10, 5000)
        avg_rows = []
        for (f, t) in util.ChunkIterable(start=0, end=n, chunk_size=block_rows):
            assert f<t
            Xblock = X[f:t, :]
            b = Xblock.shape[0]
            F = fssd.feature_tensor(Xblock)
            Tau = np.reshape(F, [b, d*J])
            # witness evaluations computed on only a subset of data
            avg_rows.append(Tau.mean(axis=0))

        # an array of length d*J
        witness_evals = (float(b)/n)*np.sum(np.vstack(avg_rows), axis=0)
        assert len(witness_evals) == d*J
        return np.reshape(witness_evals, [J, d])



