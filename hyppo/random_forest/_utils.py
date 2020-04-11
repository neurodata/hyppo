import warnings

import numpy as np
from scipy.stats import chi2

from .._utils import (
    contains_nan,
    check_ndarray_xy,
    convert_xy_float64,
    check_reps,
    check_compute_distance,
)


class _CheckInputs:
    """Checks inputs for all independence tests"""

    def __init__(self, x, y, reps=None):
        self.x = x
        self.y = y
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

        return self.x, self.y

    def check_dim_xy(self):
        """Convert x and y to proper dimensions"""
        # convert arrays of type (n,) to (n, 1)
        if self.x.ndim == 1:
            self.x = self.x[:, np.newaxis]
        elif self.x.ndim != 2:
            raise ValueError(
                "Expected a 2-D array `x`, found shape " "{}".format(self.x.shape)
            )
        if self.y.ndim == 1:
            self.y = self.y[:, np.newaxis]
        elif self.y.ndim != 2 or self.y.shape[1] > 1:
            raise ValueError(
                "Expected a (n, 1) array `y`, found shape " "{}".format(self.y.shape)
            )

        self._check_nd_indeptest()

        return self.x, self.y

    def _check_nd_indeptest(self):
        """Check if number of samples is the same"""
        nx, _ = self.x.shape
        ny, _ = self.y.shape
        if nx != ny:
            raise ValueError(
                "Shape mismatch, x and y must have shape " "[n, p] and [n, q]."
            )

    def _check_min_samples(self):
        """Check if the number of samples is at least 3"""
        nx = self.x.shape[0]
        ny = self.y.shape[0]

        if nx <= 3 or ny <= 3:
            raise ValueError("Number of samples is too low")


def sim_matrix(model, x):
    terminals = model.apply(x)
    ntrees = terminals.shape[1]

    proxMat = 1 * np.equal.outer(terminals[:, 0], terminals[:, 0])
    for i in range(1, ntrees):
        proxMat += 1 * np.equal.outer(terminals[:, i], terminals[:, i])
    proxMat = proxMat / ntrees

    return proxMat
