import warnings

import numpy as np


# from scipy
def _contains_nan(a):
    try:
        # Calling np.sum to avoid creating a huge array into memory
        # e.g. np.isnan(a).any()
        with np.errstate(invalid='ignore'):
            contains_nan = np.isnan(np.sum(a))
    except TypeError:
        # This can happen when attempting to sum things which are not
        # numbers (e.g. as in the function `mode`). Try an alternative method:
        try:
            contains_nan = np.nan in set(a.ravel())
        except TypeError:
            # Don't know what to do. Fall back to omitting nan values and
            # issue a warning.
            contains_nan = False
            warnings.warn("The input array could not be properly checked for nan "
                          "values. nan values will be ignored.", RuntimeWarning)

    if contains_nan:
        raise ValueError("Input contains NaNs. Please omit and try again")

    return contains_nan


class _CheckInputs:
    def __init__(self, x, y, dim, reps=None, compute_distance=None):
        self.x = x
        self.y = y
        self.dim = dim
        self.compute_distance = compute_distance

        if reps:
            self.reps = reps

    def __call__(self, test_name):
        self.check_ndarray_xy()
        _contains_nan(self.x)
        _contains_nan(self.y)
        self.x, self.y = self.check_dim_xy(test_name)
        self.x, self.y = self.convert_xy_float64()
        self.check_compute_distance()

        if self.reps:
            self.check_reps()

        return self.x, self.y

    def check_ndarray_xy(self):
        if (not isinstance(self.x, np.ndarray) or
                not isinstance(self.y, np.ndarray)):
            raise ValueError("x and y must be ndarrays")

    def check_dim_xy(self, test_name):
        # check if x and y are ndarrays
        if self.dim == 1:
            # check if x or y is shape (n,)
            if self.x.ndim != 1 or self.y.ndim != 1:
                raise ValueError(
                    "x and y must be of shape (n,). Please reshape")
        elif self.dim > 1:
            # convert arrays of type (n,) to (n, 1)
            if self.x.ndim == 1:
                self.x.shape = (-1, 1)
            if self.y.ndim == 1:
                self.y.shape = (-1, 1)

            self._check_nd_indeptest(test_name)
        else:
            raise ValueError("dim must be 1 or greater than 1")

        return self.x, self.y

    def _check_nd_indeptest(self, test_name):
        nx, px = self.x.shape[0]
        ny, py = self.y.shape[0]

        test_nsame = ['MGC', 'Dcorr', 'HHG']
        if test_name in test_nsame:
            if nx != ny:
                raise ValueError("Shape mismatch, x and y must have shape [n, p] and "
                                 "[n, q].")
        else:
            if nx != ny or px != py:
                raise ValueError("Shape mismatch, x and y must have shape [n, p] and "
                                 "[n, p].")

    def convert_xy_float64(self):
        # convert x and y to floats
        self.x = np.asarray(self.x).astype(np.float64)
        self.y = np.asarray(self.y).astype(np.float64)

        return self.x, self.y

    def check_reps(self):
        if not isinstance(self.reps, int) or self.reps < 0:
            raise ValueError(
                "Number of reps must be an integer greater than 0.")
        elif self.reps < 1000:
            msg = ("The number of replications is low (under 1000), and p-value "
                   "calculations may be unreliable. Use the p-value result, with "
                   "caution!")
            warnings.warn(msg, RuntimeWarning)

    def check_compute_distance(self):
        # check if compute_distance_matrix if a callable()
        if (not callable(self.compute_distance) and
            self.compute_distance is not None):
            raise ValueError("Compute_distance must be a function.")
