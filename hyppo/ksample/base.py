from abc import ABC, abstractmethod

from .._utils import euclidean


class KSampleTest(ABC):
    """
    A base class for a k-sample test.

    Parameters
    ----------
    compute_distance : callable(), optional (default: euclidean)
        A function that computes the distance or similarity among the samples
        within each data matrix. Set to `None` if `x` and `y` are already
        distance matrices. To call a custom function, either create the
        distance matrix before-hand or create a function of the form
        ``compute_distance(x)`` where `x` is the data matrix for which
        pairwise distances are calculated.
    bias : bool (default: False)
        Whether or not to use the biased or unbiased test statistics. Only
        applies to ``Dcorr`` and ``Hsic``.
    """

    def __init__(self, compute_distance=euclidean, bias=False):
        # set statistic and p-value
        self.stat = None
        self.pvalue = None
        self.compute_distance = compute_distance

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
