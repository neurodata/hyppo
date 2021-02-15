from ..independence import Dcorr
from ._utils import _CheckInputs, compute_stat
from .base import TimeSeriesTest


class DcorrX(TimeSeriesTest):
    r"""
    Cross Distance Correlation (DcorrX) test statistic and p-value.

    DcorrX is an independence test between two (paired) time series of
    not necessarily equal dimensions. The population parameter is 0 if and only if the
    time series are independent. It is based upon energy distance between distributions.

    The statistic can be derived as follows `[1]`_:

    Let :math:`x` and :math:`y` be :math:`(n, p)` and :math:`(n, q)` series
    respectively, which each contain :math:`y` observations of the series
    :math:`(X_t)` and :math:`(Y_t)`. Similarly, let :math:`x[j:n]` be the
    :math:`(n-j, p)` last :math:`n-j` observations of :math:`x`. Let :math:`y[0:(n-j)]`
    be the :math:`(n-j, p)` first :math:`n-j` observations of :math:`y`. Let :math:`M`
    be the maximum lag hyperparameter. The cross distance correlation is,

    .. math::

        \mathrm{DcorrX}_n (x, y) =  \sum_{j=0}^M \frac{n-j}{n}
                                    Dcorr_n (x[j:n], y[0:(n-j)])

    The p-value returned is calculated using a permutation test.

    .. _[1]: https://arxiv.org/abs/1908.06486

    Parameters
    ----------
    compute_distance : str, callable, or None, default: "euclidean"
        A function that computes the distance among the samples within each
        data matrix.
        Valid strings for ``compute_distance`` are, as defined in
        :func:`sklearn.metrics.pairwise_distances`,

            - From scikit-learn: [``"euclidean"``, ``"cityblock"``, ``"cosine"``,
              ``"l1"``, ``"l2"``, ``"manhattan"``] See the documentation for
              :mod:`scipy.spatial.distance` for details
              on these metrics.
            - From scipy.spatial.distance: [``"braycurtis"``, ``"canberra"``,
              ``"chebyshev"``, ``"correlation"``, ``"dice"``, ``"hamming"``,
              ``"jaccard"``, ``"kulsinski"``, ``"mahalanobis"``, ``"minkowski"``,
              ``"rogerstanimoto"``, ``"russellrao"``, ``"seuclidean"``,
              ``"sokalmichener"``, ``"sokalsneath"``, ``"sqeuclidean"``,
              ``"yule"``] See the documentation for :mod:`scipy.spatial.distance` for
              details on these metrics.

        Set to ``None`` or ``"precomputed"`` if ``x`` and ``y`` are already distance
        matrices. To call a custom function, either create the distance matrix
        before-hand or create a function of the form ``metric(x, **kwargs)``
        where ``x`` is the data matrix for which pairwise distances are
        calculated and ``**kwargs`` are extra arguements to send to your custom
        function.
    max_lag : int, default: 0
        The maximum number of lags in the past to check dependence between ``x`` and the
        shifted ``y``. Also the ``M`` hyperparmeter below.
    **kwargs
        Arbitrary keyword arguments for ``compute_distance``.
    """

    def __init__(self, compute_distance="euclidean", max_lag=0, **kwargs):
        TimeSeriesTest.__init__(
            self, compute_distance=compute_distance, max_lag=max_lag, **kwargs
        )

    def statistic(self, x, y):
        r"""
        Helper function that calculates the DcorrX test statistic.

        Parameters
        ----------
        x,y : ndarray
            Input data matrices. ``x`` and ``y`` must have the same number of
            samples. That is, the shapes must be ``(n, p)`` and ``(n, q)`` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions. Alternatively, ``x`` and ``y`` can be distance matrices,
            where the shapes must both be ``(n, n)``.

        Returns
        -------
        stat : float
            The computed DcorrX statistic.
        opt_lag : int
            The computed optimal lag.
        """
        stat, opt_lag = compute_stat(
            x, y, Dcorr, self.compute_distance, self.max_lag, **self.kwargs
        )
        self.stat = stat
        self.opt_lag = opt_lag

        return stat, opt_lag

    def test(self, x, y, reps=1000, workers=1):
        r"""
        Calculates the DcorrX test statistic and p-value.

        Parameters
        ----------
        x,y : ndarray
            Input data matrices. ``x`` and ``y`` must have the same number of
            samples. That is, the shapes must be ``(n, p)`` and ``(n, q)`` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions. Alternatively, ``x`` and ``y`` can be distance matrices,
            where the shapes must both be ``(n, n)``.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default: 1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.

        Returns
        -------
        stat : float
            The computed DcorrX statistic.
        pvalue : float
            The computed DcorrX p-value.
        dcorrx_dict : dict
            Contains additional useful returns containing the following keys:

                - opt_lag : int
                    The optimal lag that maximizes the strength of the relationship.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.time_series import DcorrX
        >>> np.random.seed(456)
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue, dcorrx_dict = DcorrX().test(x, y, reps = 100)
        >>> '%.1f, %.2f, %d' % (stat, pvalue, dcorrx_dict['opt_lag'])
        '1.0, 0.01, 0'
        """
        check_input = _CheckInputs(
            x,
            y,
            max_lag=self.max_lag,
        )
        x, y = check_input()

        stat, pvalue, stat_list = super(DcorrX, self).test(x, y, reps, workers)
        dcorrx_dict = {"opt_lag": stat_list[1]}
        return stat, pvalue, dcorrx_dict
