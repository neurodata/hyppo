import warnings

import numpy as np

from .._utils import (contains_nan, check_ndarray_xy, convert_xy_float64,
                      check_reps, check_compute_distance)
from ..independence import Dcorr


class _CheckInputs:
    def __init__(self, x, y, max_lag=None, reps=None,
                 compute_distance=None):
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
        check_compute_distance(self.compute_distance)

        if self.reps:
            check_reps(self.reps)

        return self.x, self.y

    def check_dim_xy(self):
        # check if x and y are ndarrays
        # convert arrays of type (n,) to (n, 1)
        if self.x.ndim == 1:
            self.x.shape = (-1, 1)
        if self.y.ndim == 1:
            self.y.shape = (-1, 1)

        self._check_nd_indeptest()

        return self.x, self.y

    def _check_nd_indeptest(self):
        nx = self.x.shape[0]
        ny = self.y.shape[0]

        if nx != ny:
            raise ValueError("Shape mismatch, x and y must have shape "
                                "[n, p] and [n, q].")

    def _check_max_lag(self):
        if not self.max_lag:
            self.max_lag = np.ceil(self.max_lag)

        return self.max_lag

    def _check_min_samples(self):
        nx = self.x.shape[0]
        ny = self.y.shape[0]

        if nx <= 3 or ny <= 3:
            raise ValueError("Number of samples is too low")
