from __future__ import division

from builtins import zip
from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object

from abc import ABC
import autograd
import autograd.numpy as np
import data
import _utils
import kernel
from fssd import FSSD
from h0simulator import FSSDH0SimCovObs
import logging
import time
import matplotlib.pyplot as plt

import scipy
import scipy.stats as stats

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
        med2 = _utils.meddistance(X, 1000)**2

        k = kernel.KGauss(med2*2)
        # fit a Gaussian to the data and draw to initialize V0
        V0 = _utils.fit_gaussian_draw(X, J, seed=829, reg=1e-6)
        list_gwidth = np.hstack( ( (med2)*gwidth_factors ) )
        besti, objs = GaussFSSD.grid_search_gwidth(p, dat, V0, list_gwidth)
        gwidth = list_gwidth[besti]
        assert _utils.is_real_num(gwidth), 'gwidth not real. Was %s'%str(gwidth)
        assert gwidth > 0, 'gwidth not positive. Was %.3g'%gwidth
        logging.info('After grid search, gwidth=%.3g'%gwidth)

        
        V_opt, gwidth_opt, info = GaussFSSD.optimize_locs_widths(p, dat,
                gwidth, V0, **ops) 

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
        - max_iter: # gradient descent iterations
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
        - If the lb, ub bounds are None, use fraction of the median heuristics 
          to automatically set the bounds.
        
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
        # grad_obj = autograd.elementwise_grad(flat_obj)
        # Initial point
        x0 = flatten(np.sqrt(gwidth0), test_locs0)
        
        # Make sure that the optimized gwidth is not too small or too large.
        fac_min = 1e-2 
        fac_max = 1e2
        med2 = _utils.meddistance(X, subsample=1000)**2
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

        # Optimize. Time the optimization as well.
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
        grad_obj = autograd.elementwise_grad(flat_obj)
        with _utils.ContextTimer() as timer:
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

        assert _utils.is_real_num(gw_opt), 'gw_opt is not real. Was %s' % str(gw_opt)

        return V_opt, gw_opt, opt_result

# end of class GaussFSSD