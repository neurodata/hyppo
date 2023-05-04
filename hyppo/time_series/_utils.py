import warnings

import numpy as np

from ..independence import MGC
from ..tools import (
    check_ndarray_xy,
    check_reps,
    compute_dist,
    contains_nan,
    convert_xy_float64,
)


class _CheckInputs:
    def __init__(self, x, y, max_lag=None, reps=None):
        self.x = x
        self.y = y
        self.max_lag = max_lag
        self.reps = reps

    def __call__(self):
        check_ndarray_xy(self.x, self.y)
        contains_nan(self.x)
        contains_nan(self.y)
        self.x, self.y = self.check_dim_xy()
        self.x, self.y = convert_xy_float64(self.x, self.y)
        self._check_min_samples()

        if self.reps:
            check_reps(self.reps)

        self.max_lag = check_max_lag(self.max_lag, self.x.shape[0])

        return self.x, self.y, self.max_lag

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
                "Shape mismatch, x and y must have shape [n, p] and [n, q]."
            )

    def _check_min_samples(self):
        nx = self.x.shape[0]
        ny = self.y.shape[0]

        if nx <= 3 or ny <= 3:
            raise ValueError("Number of samples is too low")


def check_max_lag(max_lag, n):
    if max_lag is None:
        max_lag = int(np.ceil(np.log(n)))
    elif not np.issubdtype(type(max_lag), np.integer):
        raise ValueError("max_lag must be an integer.")
    elif max_lag < 0:
        raise ValueError("max_lag must be >= 0.")

    return max_lag


def compute_stat(x, y, calc_stat, max_lag, is_distsim):
    """
    Computes time series test statistic and optimal lag.

    Parameters
    ----------
    x,y : ndarray of float
        Input data matrices. ``x`` and ``y`` must have the same number of
        samples. That is, the shapes must be ``(n, p)`` and ``(n, q)`` where
        `n` is the number of samples and `p` and `q` are the number of
        dimensions. Alternatively, ``x`` and ``y`` can be distance or similarity
        matrices, where the shapes must both be ``(n, n)``.
    calc_stat : callable
        The method used to calculate the test statistic (must use hyppo API).
    max_lag : int
        The maximum lag to consider when computing the test statistic.
    is_distsim : bool
        Whether or not ``x`` and ``y`` are distance or similarity matrices.
    """

    # loop over time points and find max test statistic
    dep_lag = []
    n = x.shape[0]
    for j in range(0, max_lag + 1):
        if is_distsim:
            slice_x = x[j:n, j:n]
            slice_y = y[0 : (n - j), 0 : (n - j)]
        else:
            slice_x = x[j:n, :]
            slice_y = y[0 : (n - j), :]
        stat = calc_stat(slice_x, slice_y)
        dep_lag.append((n - j) * stat / n)

    # calculate optimal lag and test statistic
    opt_lag = np.argmax(dep_lag)
    stat = np.sum(dep_lag)

    return stat, opt_lag


def compute_scale_at_lag(x, y, opt_lag, compute_distance, **kwargs):
    """Run the mgc test at the optimal scale (by shifting the series)."""
    n = x.shape[0]
    if not compute_distance:
        compute_distance = "precomputed"
    distx, disty = compute_dist(x, y, metric=compute_distance, **kwargs)

    slice_distx = distx[opt_lag:n, opt_lag:n]
    slice_disty = disty[0 : (n - opt_lag), 0 : (n - opt_lag)]

    mgc = MGC()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        opt_scale = mgc.test(slice_distx, slice_disty, reps=0)[2]["opt_scale"]

    return opt_scale
