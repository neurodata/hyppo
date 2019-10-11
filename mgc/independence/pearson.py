import warnings

import numpy as np
from scipy.stats import pearsonr

from .base import IndependenceTest
from ._utils import _contains_nan


def _check_input(x, y):
    # check if x and y are ndarrays
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("x and y must be ndarrays")

    # check if x or y is shape (n,)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be of shape (n,). Please reshape")

    # check for NaNs
    _contains_nan(x)
    _contains_nan(y)

    # convert x and y to floats
    x = np.asarray(x).astype(np.float64)
    y = np.asarray(y).astype(np.float64)

    return x, y


class Pearson(IndependenceTest):
    """
    Compute the Pearson test statistic and p-value.

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
        Calulates the Pearson test statistic.

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
        x, y = _check_input(x, y)
        stat, _ = pearsonr(x, y)
        self.stat = stat

        return stat

    def p_value(self, x, y):
        """
        Calulates the Pearson test p-value.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices that have shapes depending on the particular
            independence tests (check desired test class for specifics).

        Returns
        -------
        pvalue : float
            The computed independence test p-value.
        """
        x, y = _check_input(x, y)
        _, pvalue = pearsonr(x, y)
        self.pvalue = pvalue

        return pvalue
