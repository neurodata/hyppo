import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import pairwise_distances

from ._utils import _CheckInputs, sim_matrix
from .base import IndependenceTest
from .dcorr import _dcorr

FOREST_TYPES = {
    "classifier": RandomForestClassifier,
    "regressor": RandomForestRegressor,
}


class KMERF(IndependenceTest):
    r"""
    Kernel Mean Embedding Random Forest (KMERF) test statistic and p-value.

    The KMERF test statistic is a kernel method for calculating independence by using
    a random forest induced similarity matrix as an input, and has been shown to have
    especially high gains in finite sample testing power in high dimensional settings
    `[1]`_.

    A description of KMERF in greater detail can be found in `[1]`_. It is computed
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

    .. _[1]: https://arxiv.org/abs/1812.00029

    Parameters
    ----------
    forest : "regressor", "classifier", default: "regressor"
        Type of forest used when running the independence test. If the `y` input in
        ``test`` is categorial, use the "classifier" keyword.
    ntrees : int, default: 500
        The number of trees used in the random forest.
    **kwargs
        Additional arguments used for the forest (see
        :class:`sklearn.ensemble.RandomForestClassifier` or
        :class:`sklearn.ensemble.RandomForestRegressor`)
    """

    def __init__(self, forest="regressor", ntrees=500, **kwargs):
        if forest in FOREST_TYPES.keys():
            self.clf = FOREST_TYPES[forest](n_estimators=ntrees, **kwargs)
        else:
            raise ValueError("Forest must be of type classification or regression")
        IndependenceTest.__init__(self)

    def statistic(self, x, y):
        r"""
        Helper function that calculates the KMERF test statistic.

        Parameters
        ----------
        x,y : ndarray
            Input data matrices. ``x`` and ``y`` must have the same number of
            samples. That is, the shapes must be ``(n, p)`` and ``(n, 1)`` where
            `n` is the number of samples and `p` is the number of
            dimensions.

        Returns
        -------
        stat : float
            The computed KMERF statistic.
        """
        y = y.reshape(-1)
        self.clf.fit(x, y)
        distx = np.sqrt(1 - sim_matrix(self.clf, x))
        y = y.reshape(-1, 1)
        disty = pairwise_distances(y, metric="euclidean")
        stat = _dcorr(distx, disty, bias=False, is_fast=False)
        self.stat = stat

        # get normalalized feature importances
        importances = self.clf.feature_importances_
        importances -= np.min(importances)
        self.importances = importances / np.max(importances)

        return stat

    def test(self, x, y, reps=1000, workers=1):
        r"""
        Calculates the KMERF test statistic and p-value.

        Parameters
        ----------
        x,y : ndarray
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

        Returns
        -------
        stat : float
            The computed KMERF statistic.
        pvalue : float
            The computed KMERF p-value.
        kmerf_dict : dict
            Contains additional useful returns containing the following keys:

                - feat_importance : ndarray
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

        stat, pvalue = super(KMERF, self).test(x, y, reps, workers, is_distsim=False)
        kmerf_dict = {"feat_importance": self.importances}

        return stat, pvalue, kmerf_dict
