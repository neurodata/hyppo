import warnings

import numpy as np

from .._utils import (contains_nan, check_ndarray_xy, convert_xy_float64,
                      check_reps, check_compute_distance)


class _CheckInputs:
    def __init__(self, x, y, dim, reps=None, compute_distance=None):
        self.x = x
        self.y = y
        self.dim = dim
        self.compute_distance = compute_distance
        self.reps = reps

    def __call__(self, test_name):
        check_ndarray_xy(self.x, self.y)
        contains_nan(self.x)
        contains_nan(self.y)
        self.x, self.y = self.check_dim_xy(test_name)
        self.x, self.y = convert_xy_float64(self.x, self.y)
        self._check_min_samples()
        check_compute_distance(self.compute_distance)

        if self.reps:
            check_reps(self.reps)

        return self.x, self.y

    def check_dim_xy(self, test_name):
        # check if x and y are ndarrays
        if self.dim == 1:
            # check if x or y is shape (n,)
            if self.x.ndim > 1 or self.y.ndim > 1:
                msg = ("x and y must be of shape (n,). Will reshape")
                warnings.warn(msg, RuntimeWarning)
                self.x.shape = (-1)
                self.y.shape = (-1)
        elif self.dim > 1:
            # convert arrays of type (n,) to (n, 1)
            if self.x.ndim == 1:
                self.x.shape = (-1, 1)
            if self.y.ndim == 1:
                self.y.shape = (-1, 1)

            self._check_nd_indeptest(test_name)

        return self.x, self.y

    def _check_nd_indeptest(self, test_name):
        nx, px = self.x.shape
        ny, py = self.y.shape

        test_nsame = ["MGC", "Dcorr", "HHG", "CannCorr", "RVCorr"]
        if test_name in test_nsame:
            if nx != ny:
                raise ValueError("Shape mismatch, x and y must have shape "
                                 "[n, p] and [n, q].")
        else:
            if nx != ny or px != py:
                raise ValueError("Shape mismatch, x and y must have the same "
                                 "shape [n, p].")

    def _check_min_samples(self):
        nx = self.x.shape[0]
        ny = self.y.shape[0]

        if nx <= 3 or ny <= 3:
            raise ValueError("Number of samples is too low")
