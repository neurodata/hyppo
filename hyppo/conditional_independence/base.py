from abc import ABC, abstractmethod
from typing import NamedTuple


class ConditionalIndependenceTestOutput(NamedTuple):
    stat: float
    pvalue: float


class ConditionalIndependenceTest(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def statistic(self, x, y, z):
        r"""
        :param x:
        :param y:
        :param z:
        :return:
        """

    @abstractmethod
    def test(self, x, y, z):
        r"""
        :param x:
        :param y:
        :param z:
        :return:
        """
