from scipy.stats import kendalltau

from .base import IndependenceTest
from ._utils import _CheckInputs


class Kendall(IndependenceTest):
    """
    Compute the Kendall test statistic and p-value.

    Attributes
    ----------
    stat : float
        The computed independence test statistic.
    pvalue : float
        The computed independence test p-value.
    """

    def __init__(self):
        IndependenceTest.__init__(self)

    def statistic(self, x, y):
        """
        Calulates the Kendall test statistic.

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
        x, y = check_input()
        stat, _ = kendalltau(x, y)
        self.stat = stat

        return stat

    def test(self, x, y):
        """
        Calulates the Kendall test p-value.

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
        check_input = _CheckInputs(x, y, dim=1)
        x, y = check_input()
        stat, pvalue = kendalltau(x, y)
        self.stat = stat
        self.pvalue = pvalue

        return stat, pvalue
