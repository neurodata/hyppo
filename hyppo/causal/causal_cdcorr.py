from ..conditional import ConditionalDcorr
from ..tools import VectorMatch
from .base import ConditionalDiscrepancyTest, ConditionalDiscrepancyTestOutput

class CausalCDcorr(ConditionalDiscrepancyTest):
    """
    Causal Conditional Distance Correlation test statistic and p-value.

    Causal CDcorr is a method for testing for causal effects in multivariate data
    across groups, given a third (conditioning) matrix. Under standard causal assumptions,
    including consistency, positivity on the conditioning variables, conditional 
    ignorability on the conditioning variables, and no interference, subsequent
    conclusions merit causal interpretations. This approach levels Vertex Matching to
    synthetically pre-process the data and ensure empirical positivity on the 
    covariates  :footcite:p:`Lopez2017Aug`, followed by conditional K-sample testing 
    using the conditional distance correlation  :footcite:p:`wang2015conditional`.

    Parameters
    ----------
    compute_distance : str, callable, or None, default: "euclidean"
        A function that computes the distance among the samples within the
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

        Set to ``None`` or ``"precomputed"`` if ``Ys`` is already a distance matrix. 
        To call a custom function, either create the distance matrix
        before-hand or create a function of the form ``metric(Ys, **kwargs)``
        where ``Ys`` is the data matrix for which pairwise distances are
        calculated and ``**kwargs`` are extra arguements to send to your custom
        function.
    use_cov : bool, default: True
        If `True`, then the statistic will compute the covariance rather than the
        correlation.
    bandwith : str, scalar, 1d-array
        The method used to calculate the bandwidth used for kernel density estimate of
        the conditional matrix. This can be ‘scott’, ‘silverman’, a scalar constant or a
        1d-array with length ``r`` which is the dimensions of the conditional matrix.
        If None (default), ‘scott’ is used.
    **kwargs
        Arbitrary keyword arguments for ``compute_distance``.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self, compute_distance="euclidean", use_cov=True, bandwidth=None, **kwargs
    ):
        self.use_cov = use_cov
        self.compute_distance = compute_distance
        self.bandwidth = bandwidth

        # Check bandwidth input
        if bandwidth is not None:
            if not isinstance(bandwidth, (int, float, np.ndarray, str)):
                raise ValueError(
                    "`bandwidth` should be 'scott', 'silverman', a scalar or 1d-array."
                )
            if isinstance(bandwidth, str):
                if bandwidth.lower() not in ["scott", "silverman"]:
                    raise ValueError(
                        f"`bandwidth` must be either 'scott' or 'silverman' not '{bandwidth}'"
                    )
                self.bandwidth = bandwidth.lower()

        # set is_distance to true if compute_distance is None
        self.is_distance = False
        if not compute_distance:
            self.is_distance = True

        ConditionalDiscrepancyTest.__init__(self, **kwargs)
