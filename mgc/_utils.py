import warnings

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel


# from scipy
def contains_nan(a):
    """Check if inputs contains NaNs"""
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
    """Check if x or y is an ndarray"""
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("x and y must be ndarrays")


def convert_xy_float64(x, y):
    """Convert x or y to np.float64 (if not already done)"""
    # convert x and y to floats
    x = np.asarray(x).astype(np.float64)
    y = np.asarray(y).astype(np.float64)

    return x, y


def check_reps(reps):
    """Check if reps is valid"""
    # check if reps is an integer > than 0
    if not isinstance(reps, int) or reps < 0:
        raise ValueError(
            "Number of reps must be an integer greater than 0.")

    # check if reps is under 1000 (recommended)
    elif reps < 1000:
        msg = ("The number of replications is low (under 1000), and p-value "
                "calculations may be unreliable. Use the p-value result, with "
                "caution!")
        warnings.warn(msg, RuntimeWarning)


def check_compute_distance(compute):
    """Check if compute distance/kernel function if a callable()"""
    if (not callable(compute) and compute is not None):
        raise ValueError("The compute distance/kernel must be a function.")


def check_xy_distmat(x, y):
    """Check if x and y are distance matrices"""
    nx, px = x.shape
    ny, py = y.shape
    if nx != px or ny != py or np.trace(x) != 0 or np.trace(y) != 0:
        raise ValueError("Shape mismatch, x and y must be distance matrices "
                         "have shape [n, n] and [n, n].")

def check_inputs_distmat(inputs):
    # check if x and y are distance matrices
    for i in inputs:
        n, p = i.shape
        if n != p or np.trace(i) != 0:
            raise ValueError("Shape mismatch, x and y must be distance matrices "
                        "have shape [n, n] and [n, n].")


def euclidean(x):
    """Default euclidean distance function calculation"""
    return cdist(x, x, metric='euclidean')


def gaussian(x):
    """Default medial gaussian kernel similarity calculation"""
    l1 = cdist(x, x, 'cityblock')
    gamma = 1.0 / (2 * (np.median(l1[l1!=0]) ** 2))
    return np.exp(-gamma * cdist(x, x, 'sqeuclidean'))
