from abc import ABC, abstractmethod
from typing import NamedTuple


class ConditionalIndependenceTestOutput(NamedTuple):
    stat: float
    pvalue: float


class ConditionalIndependenceTest(ABC):
    """
    A base class for a conditional independence test.

    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def statistic(self, x, y, z):
        r"""
        Calculates the conditional independence test statistic.

        Parameters
        ----------
        x,y,z : ndarray of float
            Input data matrices.

        Returns
        -------
        stat : float
            The computed conditional independence test statistic.
        """

    @abstractmethod
    def test(self, x, y, z):
        r"""
        Calculates the conditional independence test statistic and p-value.

        Parameters
        ----------
        x,y,z : ndarray of float
            Input data matrices.

        Returns
        -------
        stat : float
            The computed conditional independence test statistic.
        pvalue : float
            The computed conditional independence test p-value.
        """
