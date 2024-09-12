from typing import NamedTuple

import numpy as np
from scipy.stats.distributions import chi2
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import pairwise_distances

from ._utils import _CheckInputs, sim_matrix
from .base import IndependenceTest
from .dcorr import _dcorr

FOREST_TYPES = {
    "classifier": RandomForestClassifier,
    "regressor": RandomForestRegressor,
}


class KMERFTestOutput(NamedTuple):
    stat: float
    pvalue: float
    kmerf_dict: dict


class KMERF(IndependenceTest):
    r"""
    Kernel Mean Embedding Random Forest (KMERF) test statistic and p-value.

    The KMERF test statistic is a kernel method for calculating independence by using
    a random forest induced similarity matrix as an input, and has been shown to have
    especially high gains in finite sample testing power in high dimensional settings
    :footcite:p:`shenLearningInterpretableCharacteristic2020`.

    Parameters
    ----------
    forest : "regressor", "classifier", default: "regressor"
        Type of forest used when running the independence test. If the `y` input in
        ``test`` is categorial, use the "classifier" keyword.
    ntrees : int, default: 500
        The number of trees used in the random forest.
    compute_distance : str, callable, or None, default: "euclidean"
        A function that computes the distance among the samples for `y`.
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

        Set to ``None`` or ``"precomputed"`` if ``y`` is already a distance
        matrices. To call a custom function, either create the distance matrix
        before-hand or create a function of the form ``metric(x, **kwargs)``
        where ``x`` is the data matrix for which pairwise distances are
        calculated and ``**kwargs`` are extra arguements to send to your custom
        function.
    distance_kwargs : dict
        Arbitrary keyword arguments for ``compute_distance``.
    **kwargs
        Additional arguments used for the forest (see
        :class:`sklearn.ensemble.RandomForestClassifier` or
        :class:`sklearn.ensemble.RandomForestRegressor`)

    Notes
    -----
    .. note::
        This algorithm is currently under review at a peer-reviewed conference.

    A description of KMERF in greater detail can be found in
    :footcite:p:`shenLearningInterpretableCharacteristic2020`. It is computed
    using the following steps:

    Let :math:`x` and :math:`y` be :math:`(n, p)` and :math:`(n, 1)` samples of random
    variables
    :math:`X` and :math:`Y`.

    + Run random forest with :math:`m` trees. Independent bootstrap samples of size
      :math:`n_{b} \leq n` are drawn to build a tree each time; each tree structure
      within the forest is denoted as :math:`\phi_w \in \mathbf{P}`,
      :math:`w \in \{ 1, \ldots, m \}`; :math:`\phi_w(x_i)` denotes the partition
      assigned to :math:`x_i`.

    + Calculate the proximity kernel:

      .. math::

         \mathbf{K}^{\mathbf{x}}_{ij} = \frac{1}{m} \sum_{w = 1}^{m} I(\phi_w(x_i)
         = \phi_w(x_j))

      where :math:`I(\cdot)`$` is the indicator function for how often two observations
      lie in the same partition.

    + Compute the induced kernel correlation: Let

      .. math::

         \mathbf{L}^{\mathbf{x}}_{ij}=
         \begin{cases}
             \mathbf{K}^{\mathbf{x}}_{ij}
             - \frac{1}{n-2} \sum_{t=1}^{n} \mathbf{K}^{\mathbf{x}}_{it}
             - \frac{1}{n-2} \sum_{s=1}^{n} \mathbf{K}^{\mathbf{x}}_{sj}
             + \frac{1}{(n-1)(n-2)} \sum_{s,t=1}^{n} \mathbf{K}^{\mathbf{x}}_{st}
             & \mbox{when} \ i \neq j \\
             0 & \mbox{ otherwise}
         \end{cases}

    + Then let :math:`\mathbf{K}^{\mathbf{y}}` be the Euclidean distance induced kernel,
      and similarly compute :math:`\mathbf{L}^{\mathbf{y}}` from
      :math:`\mathbf{K}^{\mathbf{y}}`. The unbiased kernel correlation equals

      .. math::

         \mathrm{KMERF}_n(\mathbf{x}, \mathbf{y}) = \frac{1}{n(n-3)}
         \mathrm{tr} \left( \mathbf{L}^{\mathbf{x}} \mathbf{L}^{\mathbf{y}} \right)

    The p-value returned is calculated using a permutation test using
    :meth:`hyppo.tools.perm_test`.

    References
    ----------
    .. footbibliography::
    """
    def __init__(self, forest="regressor", ntrees=500, **kwargs):
        self.is_ksamp = False
        if "is_ksamp" in kwargs.keys():
            del kwargs["is_ksamp"]
        if forest in FOREST_TYPES.keys():
            self.clf = FOREST_TYPES[forest](n_estimators=ntrees, **kwargs)
        else:
            raise ValueError("forest must be one of the following ")
        IndependenceTest.__init__(self)

    def statistic(self, x, y):
        r"""
        Helper function that calculates the KMERF test statistic.

        Parameters
        ----------
        x,y : ndarray of float
            Input data matrices. ``x`` and ``y`` must have the same number of
            samples. That is, the shapes must be ``(n, p)`` and ``(n, 1)`` where
            `n` is the number of samples and `p` is the number of
            dimensions.

        Returns
        -------
        stat : float
            The computed KMERF statistic.
        """
        self.clf.fit(x, y)
        self.distx = 1 - sim_matrix(self.clf, x)
        self.disty = pairwise_distances(y, metric="euclidean")
        self.stat = _dcorr(self.distx, self.disty, bias=False, is_fast=False)

        # get feature importances from gini-based random forest
        self.importances = self.clf.feature_importances_

        return self.stat

    def test(self, x, y, reps=1000, workers=1, auto=True, random_state=None):
        r"""
        Calculates the KMERF test statistic and p-value.

        Parameters
        ----------
        x,y : ndarray of float
            Input data matrices. ``x`` and ``y`` must have the same number of
            samples. That is, the shapes must be ``(n, p)`` and ``(n, 1)`` where
            `n` is the number of samples and `p` is the number of
            dimensions.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default: 1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.
        auto : bool, default: True
            Automatically uses fast approximation when `n` and size of array
            is greater than 20. If ``True``, and sample size is greater than 20, then
            :class:`hyppo.tools.chi2_approx` will be run. Parameters ``reps`` and
            ``workers`` are
            irrelevant in this case. Otherwise, :class:`hyppo.tools.perm_test` will be
            run.

        Returns
        -------
        stat : float
            The computed KMERF statistic.
        pvalue : float
            The computed KMERF p-value.
        kmerf_dict : dict
            Contains additional useful returns containing the following keys:

                - feat_importance : ndarray of float
                    An array containing the importance of each dimension

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.independence import KMERF
        >>> x = np.arange(100)
        >>> y = x
        >>> '%.1f, %.2f' % KMERF().test(x, y)[:1] # doctest: +SKIP
        '1.0, 0.001'
        """
        check_input = _CheckInputs(x, y, reps=reps)
        x, y = check_input()

        if auto and x.shape[0] > 20:
            n = x.shape[0]
            stat = self.statistic(x, y)
            statx = _dcorr(self.distx, self.distx, bias=False, is_fast=False)
            staty = _dcorr(self.disty, self.disty, bias=False, is_fast=False)
            pvalue = chi2.sf(stat / np.sqrt(statx * staty) * n + 1, 1)
            self.stat = stat
            self.pvalue = pvalue
            self.null_dist = None
        else:
            stat, pvalue = super(KMERF, self).test(
                x, y, reps, workers, is_distsim=False, random_state=random_state
            )
        kmerf_dict = {"feat_importance": self.importances}

        return KMERFTestOutput(stat, pvalue, kmerf_dict)
