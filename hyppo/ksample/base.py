from abc import ABC, abstractmethod

from .._utils import euclidean


class KSampleTest(ABC):
    """
    A base class for a k-sample test.
    """

    def __init__(self):
        # set statistic and p-value
        self.stat = None
        self.pvalue = None

        super().__init__()

    @abstractmethod
    def _statistic(self, inputs):
        r"""
        Calulates the *k*-sample test statistic.

        Parameters
        ----------
        inputs : ndarray
            Input data matrices.
        """

    @abstractmethod
    def test(self, inputs, reps=1000, workers=1):
        r"""
        Calulates the k-sample test p-value.

        Parameters
        ----------
        inputs : list of ndarray
            Input data matrices.
        reps : int, optional
            The number of replications used in permutation, by default 1000.
        workers : int, optional (default: 1)
            Evaluates method using `multiprocessing.Pool <multiprocessing>`).
            Supply `-1` to use all cores available to the Process.
        """
