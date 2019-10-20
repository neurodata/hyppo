import warnings

import numpy as np
from scipy.spatial.distance import cdist


# from scipy
def contains_nan(a):
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
            msg = ("The input array could not be properly checked for nan "
                   "values. nan values will be ignored.")
            warnings.warn(msg, RuntimeWarning)

    if contains_nan:
        raise ValueError("Input contains NaNs. Please omit and try again")

    return contains_nan


def check_ndarray_xy(x, y):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("x and y must be ndarrays")


def check_ndarray_inputs(inputs):
    if len(inputs) == 1:
        raise ValueError("there must be at least 2 inputs")
    for i in inputs:
        if not isinstance(i, np.ndarray):
            raise ValueError("x and y must be ndarrays")


def convert_xy_float64(x, y):
    # convert x and y to floats
    x = np.asarray(x).astype(np.float64)
    y = np.asarray(y).astype(np.float64)

    return x, y


def convert_inputs_float64(inputs):
    nd_inputs = []
    for i in inputs:
        nd_inputs.append(np.asarray(i).astype(np.float64))

    return nd_inputs


def check_reps(reps):
    if not isinstance(reps, int) or reps < 0:
        raise ValueError(
            "Number of reps must be an integer greater than 0.")
    elif reps < 1000:
        msg = ("The number of replications is low (under 1000), and p-value "
                "calculations may be unreliable. Use the p-value result, with "
                "caution!")
        warnings.warn(msg, RuntimeWarning)


def check_compute_distance(compute_distance):
    # check if compute_distance if a callable()
    if (not callable(compute_distance) and compute_distance is not None):
        raise ValueError("Compute_distance must be a function.")


def euclidean(x):
    return cdist(x, x, metric='euclidean')
