import warnings

import numpy as np
from scipy.stats import chi2

from .._utils import (contains_nan, check_ndarray_xy, convert_xy_float64,
                      check_reps, check_compute_distance)


class _CheckInputs:
    """Checks inputs for all independence tests"""
    def __init__(self, x, y, dim, reps=None, compute_distance=None):
        self.x = x
        self.y = y
        self.dim = dim
        self.compute_distance = compute_distance
        self.reps = reps

    def __call__(self):
        check_ndarray_xy(self.x, self.y)
        contains_nan(self.x)
        contains_nan(self.y)
        self.x, self.y = self.check_dim_xy()
        self.x, self.y = convert_xy_float64(self.x, self.y)
        self._check_min_samples()
        check_compute_distance(self.compute_distance)

        if self.reps:
            check_reps(self.reps)

        return self.x, self.y

    def check_dim_xy(self):
        """Convert x and y to proper dimensions"""
        # for kendall, pearson, and spearman
        if self.dim == 1:
            # check if x or y is shape (n,)
            if self.x.ndim > 1 or self.y.ndim > 1:
                self.x.shape = (-1,)
                self.y.shape = (-1,)

        # for other tests
        elif self.dim > 1:
            # convert arrays of type (n,) to (n, 1)
            if self.x.ndim == 1:
                self.x.shape = (-1, 1)
            if self.y.ndim == 1:
                self.y.shape = (-1, 1)

            self._check_nd_indeptest()

        return self.x, self.y

    def _check_nd_indeptest(self):
        """Check if number of samples is the same"""
        nx, _ = self.x.shape
        ny, _ = self.y.shape
        if nx != ny:
            raise ValueError("Shape mismatch, x and y must have shape "
                                "[n, p] and [n, q].")

    def _check_min_samples(self):
        """Check if the number of samples is at least 3"""
        nx = self.x.shape[0]
        ny = self.y.shape[0]

        if nx <= 3 or ny <= 3:
            raise ValueError("Number of samples is too low")


def _chi2_approx(stat, null_dist, samps):
    mu = np.mean(null_dist)
    sigma = np.std(null_dist)

    if sigma < 10e-4 and mu < 10e-4:
        x = 0.0
    else:
        x = samps*(stat - mu)/sigma + 1

    pvalue = 1 - chi2.cdf(x, 1)

    return pvalue
