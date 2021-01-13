import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import pairwise_distances

from . import Dcorr
from ._utils import _CheckInputs, sim_matrix
from .base import IndependenceTest

FOREST_TYPES = {
    "classifier": RandomForestClassifier,
    "regressor": RandomForestRegressor,
}


class KMERF(IndependenceTest):
    r"""
    Class for calculating the KMERF test statistic and p-value.

    The KMERF test statistic is a kernel method for calculating independence by using
    a random forest induced similarity matrix as an input, and has been shown to have
    especially high gains in finite sample testing power in high dimensional settings
    [#1KMERF]_.

    Parameters
    ----------
    forest : {"classifier", "regressor"}
        Type of forest used when running the independence test. If the `y` input in
        ``test`` is categorial, use the "classifier" keyword.
    ntrees : int, optional (default: 500)
        The number of trees used in the random forest.
    **kwargs : optional
        Additional arguments used for the forest (see
        ``sklearn.ensemble.RandomForestClassifier`` or
        ``sklearn.ensemble.RandomForestRegressor``)

    Notes
    -----
    A description of KMERF in greater detail can be found in [#1KMERF]_. It is computed
    using the following steps:

    Let :math:`x` and :math:`y` be :math:`(n, p)` samples of random variables
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

         KMERF_n(\mathbf{x}, \mathbf{y}) = \frac{1}{n(n-3)}
         \mathrm{tr} \left( \mathbf{L}^{\mathbf{x}} \mathbf{L}^{\mathbf{y}} \right)

    The p-value returned is calculated using a permutation test using a
    `permutation test <https://hyppo.neurodata.io/reference/tools.html#permutation-test>`_.

    References
    ----------
    .. [#1KMERF] Shen, C., Panda, S., & Vogelstein, J. T. (2018). Learning
                 Interpretable Characteristic Kernels via Decision Forests.
                 arXiv preprint arXiv:1812.00029.
    """

    def __init__(self, forest="regressor", ntrees=500, **kwargs):
        if forest in FOREST_TYPES.keys():
            self.clf = FOREST_TYPES[forest](n_estimators=ntrees, **kwargs)
        else:
            raise ValueError("Forest must be of type classification or regression")
        IndependenceTest.__init__(self)

    def _statistic(self, x, y):
        r"""
        Helper function that calculates the KMERF test statistic.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices. `x` and `y` must have the same number of
            samples. That is, the shapes must be `(n, p)` and `(n, q)` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions. Alternatively, `x` and `y` can be distance matrices,
            where the shapes must both be `(n, n)`.

        Returns
        -------
        stat : float
            The computed KMERF statistic.

        y must be categorical
        """
        y = y.reshape(-1)
        self.clf.fit(x, y)
        distx = np.sqrt(1 - sim_matrix(self.clf, x))
        y = y.reshape(-1, 1)
        disty = pairwise_distances(y, metric="euclidean")
        stat = Dcorr(compute_distance=None)._statistic(distx, disty)
        self.stat = stat

        return stat

    def test(self, x, y, reps=1000, workers=1):
        r"""
        Calculates the KMERF test statistic and p-value.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices. `x` and `y` must have the same number of
            samples. That is, the shapes must be `(n, p)` and `(n, q)` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions. Alternatively, `x` and `y` can be distance matrices,
            where the shapes must both be `(n, n)`.
        reps : int, optional (default: 1000)
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, optional (default: 1)
            The number of cores to parallelize the p-value computation over.
            Supply -1 to use all cores available to the Process.

        Returns
        -------
        stat : float
            The computed KMERF statistic.
        pvalue : float
            The computed KMERF p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.independence import KMERF
        >>> x = np.arange(100)
        >>> y = x
        >>> '%.1f, %.2f' % KMERF().test(x, y) # doctest: +SKIP
        '1.0, 0.001'
        """
        check_input = _CheckInputs(x, y, reps=reps)
        x, y = check_input()

        return super(KMERF, self).test(x, y, reps, workers, is_distsim=False)
