import warnings

import numpy as np
from scipy.sparse.linalg import svds

from .base import IndependenceTest
from ._utils import _contains_nan


def _check_input_statistic(x, y):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("x and y must be ndarrays")

    # convert arrays of type (n,) to (n, 1)
    if x.ndim == 1:
        x.shape = (-1, 1)
    if y.ndim == 1:
        y.shape = (-1, 1)

    # check for NaNs
    _contains_nan(x)
    _contains_nan(y)

    nx = x.shape[0]
    ny = y.shape[0]
    if nx != ny:
        raise ValueError("Shape mismatch, x and y must have shape [n, p] and "
                         "[n, q].")

    # convert x and y to floats
    x = np.asarray(x).astype(np.float64)
    y = np.asarray(y).astype(np.float64)

    return x, y


def _check_input_pvalue(x, y, reps):
    x, y = _check_input_statistic(x, y)

    # check if number of reps exists, integer, or > 0 (if under 1000 raises
    # warning)
    if not isinstance(reps, int) or reps < 0:
        raise ValueError("Number of reps must be an integer greater than 0.")
    elif reps < 1000:
        msg = ("The number of replications is low (under 1000), and p-value "
               "calculations may be unreliable. Use the p-value result, with "
               "caution!")
        warnings.warn(msg, RuntimeWarning)

    return x, y


class CannCorr(IndependenceTest):
    """
    Compute the CCA test statistic and p-value.

    Attributes
    ----------
    stat : float
        The computed independence test statistic.
    pvalue : float
        The computed independence test p-value.
    """

    def __init__(self):
        super().__init__(self)

    def statistic(self, x, y):
        """
        Calulates the CCA test statistic.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices that have shapes depending on the particular
            independence tests (check desired test class for specifics).

        Returns
        -------
        stat : float
            The computed independence test statistic.
        """
        x, y = _check_input_statistic(x, y)

        centx = x - np.mean(x, axis=0)
        centy = y - np.mean(y, axis=0)

        # calculate covariance and variances for inputs
        covar = centx.T @ centy
        varx = centx.T @ centx
        vary = centy.T @ centy

        if varx.size == 1 or vary.size == 1 or covar.size == 1:
            covar = np.sum(covar ** 2)
            stat = np.divide(covar, np.sqrt(np.sum(varx ** 2),
                                            np.sum(vary ** 2)))
        else:
            covar = np.sum(svds(covar, 1)[1] ** 2)
            stat = np.divide(covar, np.sqrt(np.sum(svds(varx, 1)[1] ** 2)
                                            * np.sum(svds(vary, 1)[1] ** 2)))
        self.stat = stat

        return stat

    def p_value(self, x, y, reps=1000, workers=-1):
        """
        Calulates the CCA test p-value.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices that have shapes depending on the particular
            independence tests (check desired test class for specifics).
        reps : int, optional
            The number of replications used in permutation, by default 1000.

        Returns
        -------
        pvalue : float
            The computed independence test p-value.
        """
        x, y = _check_input_pvalue(x, y, reps)

        return super(CannCorr, self).p_value(x, y, reps, workers)
