import warnings

import numpy as np
from scipy.stats import pearsonr

from .base import IndependenceTest
from ._utils import _contains_nan, _CheckInputs


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
        check_input = _CheckInputs(x, y, dim=1)
        x, y = check_input(Pearson.__name__)
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
        check_input = _CheckInputs(x, y, dim=1, is_pval=True)
        x, y = check_input(Pearson.__name__)
        _, pvalue = pearsonr(x, y)
        self.pvalue = pvalue

        return pvalue
