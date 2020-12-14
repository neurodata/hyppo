from abc import ABC, abstractmethod


class RandomForestTest(ABC):
    r"""
    A base class for an random-forest based independence test.
    """

    def __init__(self):
        # set statistic and p-value
        self.stat = None
        self.pvalue = None

        super().__init__()

    @abstractmethod
    def _statistic(self, x, y):
        r"""
        Calulates the random-forest test statistic.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices.
        """

    @abstractmethod
    def test(self, x, y, reps=1000, workers=1):
        r"""
        Calulates the independence test p-value.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices.
        reps : int, optional
            The number of replications used in permutation, by default 1000.
        workers : int, optional (default: 1)
            Evaluates method using `multiprocessing.Pool <multiprocessing>`).
            Supply `-1` to use all cores available to the Process.

        Returns
        -------
        stat : float
            The computed independence test statistic.
        pvalue : float
            The pvalue obtained via permutation.
        """
