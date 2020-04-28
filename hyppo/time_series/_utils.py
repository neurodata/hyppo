import warnings

import numpy as np

from .._utils import (
    contains_nan,
    check_ndarray_xy,
    convert_xy_float64,
    check_reps,
    check_compute_distance,
)
from ..independence import MGC


class _CheckInputs:
    def __init__(self, x, y, max_lag=None, reps=None, compute_distance=None):
        self.x = x
        self.y = y
        self.max_lag = max_lag
        self.compute_distance = compute_distance
        self.reps = reps

    def __call__(self):
        check_ndarray_xy(self.x, self.y)
        contains_nan(self.x)
        contains_nan(self.y)
        self.max_lag = self._check_max_lag()
        self.x, self.y = self.check_dim_xy()
        self.x, self.y = convert_xy_float64(self.x, self.y)
        self._check_min_samples()
        # self._check_variance()
        check_compute_distance(self.compute_distance)

        if self.reps:
            check_reps(self.reps)

        return self.x, self.y

    def check_dim_xy(self):
        # check if x and y are ndarrays
        # convert arrays of type (n,) to (n, 1)
        if self.x.ndim == 1:
            self.x = self.x[:, np.newaxis]
        elif self.x.ndim != 2:
            raise ValueError(
                "Expected a 2-D array `x`, found shape " "{}".format(self.x.shape)
            )
        if self.y.ndim == 1:
            self.y = self.y[:, np.newaxis]
        elif self.y.ndim != 2:
            raise ValueError(
                "Expected a 2-D array `y`, found shape " "{}".format(self.y.shape)
            )

        self._check_nd_indeptest()

        return self.x, self.y

    def _check_nd_indeptest(self):
        nx = self.x.shape[0]
        ny = self.y.shape[0]

        if nx != ny:
            raise ValueError(
                "Shape mismatch, x and y must have shape " "[n, p] and [n, q]."
            )

    def _check_max_lag(self):
        if not self.max_lag:
            self.max_lag = np.ceil(self.max_lag)

        return self.max_lag

    def _check_min_samples(self):
        nx = self.x.shape[0]
        ny = self.y.shape[0]

        if nx <= 3 or ny <= 3:
            raise ValueError("Number of samples is too low")

    # def _check_variance(self):
    #     for i in [self.x, self.y]:
    #         if np.var(i) == 0:
    #             raise ValueError("Test cannot be run, one of the inputs has 0 variance")


def compute_stat(x, y, indep_test, compute_distance, max_lag):
    """Compute time series test statistic"""
    # calculate distance matrices
    distx = compute_distance(x)
    disty = compute_distance(y)

    # calculate dep_lag when max_lag is 0
    dep_lag = []
    indep_test = indep_test(compute_distance=compute_distance)
    indep_test_stat = indep_test._statistic(x, y)
    dep_lag.append(np.maximum(0, indep_test_stat))

    # loop over time points and find max test statistic
    n = distx.shape[0]
    for j in range(1, max_lag + 1):
        slice_distx = distx[j:n, j:n]
        slice_disty = disty[0 : (n - j), 0 : (n - j)]
        stat = indep_test._statistic(slice_distx, slice_disty)
        dep_lag.append((n - j) * np.maximum(0, stat) / n)

    # calculate optimal lag and test statistic
    opt_lag = np.argmax(dep_lag)
    stat = np.sum(dep_lag)

    return stat, opt_lag


def compute_scale_at_lag(x, y, opt_lag, compute_distance):
    """Run the mgc test at the optimal scale (by shifting the series)."""
    n = x.shape[0]
    distx = compute_distance(x)
    disty = compute_distance(y)

    slice_distx = distx[opt_lag:n, opt_lag:n]
    slice_disty = disty[0 : (n - opt_lag), 0 : (n - opt_lag)]

    mgc = MGC()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        opt_scale = mgc.test(slice_distx, slice_disty, reps=0)[2]["opt_scale"]

    return opt_scale
