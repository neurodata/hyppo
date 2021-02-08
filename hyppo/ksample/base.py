from abc import ABC, abstractmethod


class KSampleTest(ABC):
    """
    A base class for a k-sample test.

    Parameters
    ----------
    compute_distance : callable(), optional (default: None)
        A function that computes the distance among the samples within each
        data matrix.
        Valid strings for ``metric`` are, as defined in
        ``sklearn.metrics.pairwise_distances``,

            - From scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’,
              ‘manhattan’] See the documentation for scipy.spatial.distance for details
              on these metrics.
            - From scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’,
              ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’,
              ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’,
              ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’] See the
              documentation for scipy.spatial.distance for details on these metrics.

        Set to `None` or `precomputed` if `x` and `y` are already distance
        matrices. To call a custom function, either create the distance matrix
        before-hand or create a function of the form ``metric(x, **kwargs)``
        where `x` is the data matrix for which pairwise distances are
        calculated and kwargs are extra arguements to send to your custom function.
    bias : bool (default: False)
        Whether or not to use the biased or unbiased test statistics. Only
        applies to ``Dcorr`` and ``Hsic``.
    """

    def __init__(self, compute_distance=None, bias=False, **kwargs):
        # set statistic and p-value
        self.stat = None
        self.pvalue = None
        self.bias = bias
        self.compute_distance = compute_distance
        self.kwargs = kwargs

        super().__init__()

    @abstractmethod
    def statistic(self, inputs):
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
