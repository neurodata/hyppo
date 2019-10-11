import warnings

import numpy as np
from scipy.stats import spearmanr

from .base import IndependenceTest
from ._utils import _contains_nan, _CheckInputs


def _check_input(x, y):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("x and y must be ndarrays")

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be of shape (n,). Please reshape")

    # check for NaNs
    _contains_nan(x)
    _contains_nan(y)

    # convert x and y to floats
    x = np.asarray(x).astype(np.float64)
    y = np.asarray(y).astype(np.float64)

    return x, y


class Spearman(IndependenceTest):
    """
    Compute the Spearman test statistic and p-value.

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
        Calulates the Spearman test statistic.

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
        check_input = _CheckInputs(x, y, dim=1)
        x, y = check_input(Spearman.__name__)
        stat, _ = spearmanr(x, y)
        self.stat = stat

        return stat

    def p_value(self, x, y):
        """
        Calulates the Spearman test p-value.

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
        check_input = _CheckInputs(x, y, dim=1, is_pval=True)
        x, y = check_input(Spearman.__name__)
        _, pvalue = spearmanr(x, y)
        self.pvalue = pvalue

        return pvalue
