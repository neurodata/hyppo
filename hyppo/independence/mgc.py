import warnings

from scipy.stats import multiscale_graphcorr

from ..tools import compute_dist
from ._utils import _CheckInputs
from .base import IndependenceTest


class MGC(IndependenceTest):
    r"""
    Multiscale Graph Correlation (MGC) test statistic and p-value.

    Specifically, for each point, MGC finds the :math:`k`-nearest neighbors for
    one property (e.g. cloud density), and the :math:`l`-nearest neighbors for
    the other property (e.g. grass wetness) `[1]`_. This pair :math:`(k, l)` is
    called the "scale". A priori, however, it is not know which scales will be
    most informative. So, MGC computes all distance pairs, and then efficiently
    computes the distance correlations for all scales. The local correlations
    illustrate which scales are relatively informative about the relationship.
    The key, therefore, to successfully discover and decipher relationships
    between disparate data modalities is to adaptively determine which scales
    are the most informative, and the geometric implication for the most
    informative scales. Doing so not only provides an estimate of whether the
    modalities are related, but also provides insight into how the
    determination was made. This is especially important in high-dimensional
    data, where simple visualizations do not reveal relationships to the
    unaided human eye. Characterizations of this implementation in particular
    have been derived from and benchmarked within in `[2]`_.
    A description of the process of MGC and applications on neuroscience data
    can be found in `[1]`_. It is performed using the following steps:

    Let :math:`x` and :math:`y` be :math:`(n, p)` samples of random variables
    :math:`X` and :math:`Y`. Let :math:`D^x` be the :math:`n \times n`
    distance matrix of :math:`x` and :math:`D^y` be the :math:`n \times n` be
    the distance matrix of :math:`y`. :math:`D^x` and :math:`D^y` are
    modified to be mean zero columnwise. This results in two
    :math:`n \times n` distance matrices :math:`A` and :math:`B` (the
    centering and unbiased modification) `[3]`_.

    + For all values :math:`k` and :math:`l` from :math:`1, ..., n`,

       * The :math:`k`-nearest neighbor and :math:`l`-nearest neighbor graphs
         are calculated for each property. Here, :math:`G_k (i, j)` indicates
         the :math:`k`-smallest values of the :math:`i`-th row of :math:`A`
         and :math:`H_l (i, j)` indicates the :math:`l` smallested values of
         the :math:`i`-th row of :math:`B`

       * The local
         correlations are summed and normalized using the following statistic:

         .. math::

            c^{kl} = \frac{\sum_{ij} A G_k B H_l}
                            {\sqrt{\sum_{ij} A^2 G_k \times \sum_{ij} B^2 H_l}}

    + The MGC test statistic is the smoothed optimal local correlation of
      :math:`\{ c^{kl} \}`. Denote the smoothing operation as :math:`R(\cdot)`
      (which essentially set all isolated large correlations) as 0 and
      connected large correlations the same as before, see `[3]`_.) MGC is,

      .. math::

         \mathrm{MGC}_n (x, y) = \max_{(k, l)} R \left(c^{kl} \left( x_n, y_n \right)
                                                    \right)

    The test statistic returns a value between :math:`(-1, 1)` since it is
    normalized.

    The p-value returned is calculated using a permutation test using
    :meth:`hyppo.tools.perm_test`.

    .. _[1]: https://elifesciences.org/articles/41690
    .. _[2]: https://arxiv.org/abs/1907.02088
    .. _[3]: https://www.tandfonline.com/doi/abs/10.1080/01621459.2018.1543125

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
    **kwargs
        Arbitrary keyword arguments for ``compute_distance``.
    """

    def __init__(self, compute_distance="euclidean", **kwargs):
        # set is_distance to true if compute_distance is None
        self.is_distance = False
        if not compute_distance:
            self.is_distance = True
        IndependenceTest.__init__(self, compute_distance=compute_distance, **kwargs)

    def statistic(self, x, y):
        r"""
        Helper function that calculates the MGC test statistic.

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
            The computed MGC statistic.
        """
        distx = x
        disty = y

        if not self.is_distance:
            distx, disty = compute_dist(
                x, y, metric=self.compute_distance, **self.kwargs
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            mgc = multiscale_graphcorr(distx, disty, compute_distance=None, reps=0)

        stat = mgc.stat
        self.stat = stat

        return stat

    def test(self, x, y, reps=1000, workers=1):
        r"""
        Calculates the MGC test statistic and p-value.

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
            The computed MGC statistic.
        pvalue : float
            The computed MGC p-value.
        mgc_dict : dict
            Contains additional useful returns containing the following keys:

                - mgc_map : ndarray
                    A 2D representation of the latent geometry of the relationship.
                - opt_scale : (int, int)
                    The estimated optimal scale as a ``(x, y)`` pair.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.independence import MGC
        >>> x = np.arange(100)
        >>> y = x
        >>> stat, pvalue, _ = MGC().test(x, y)
        >>> '%.1f, %.3f' % (stat, pvalue)
        '1.0, 0.001'

        In addition, the inputs can be distance matrices. Using this is the,
        same as before, except the ``compute_distance`` parameter must be set
        to ``None``.

        >>> import numpy as np
        >>> from hyppo.independence import MGC
        >>> x = np.ones((10, 10)) - np.identity(10)
        >>> y = 2 * x
        >>> mgc = MGC(compute_distance=None)
        >>> stat, pvalue, _ = mgc.test(x, y)
        >>> '%.1f, %.2f' % (stat, pvalue)
        '0.0, 1.00'
        """
        check_input = _CheckInputs(
            x,
            y,
            reps=reps,
        )
        x, y = check_input()

        x, y = compute_dist(x, y, metric=self.compute_distance, **self.kwargs)
        self.is_distance = True

        # using our joblib implementation instead of multiprocessing backend in
        # scipy gives significantly faster results
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            _, _, mgc_dict = multiscale_graphcorr(x, y, compute_distance=None, reps=0)
        mgc_dict.pop("null_dist")

        stat, pvalue = super(MGC, self).test(x, y, reps, workers)
        self.mgc_dict = mgc_dict

        return stat, pvalue, mgc_dict
