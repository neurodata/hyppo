import numpy as np
from numba import jit
from scipy.sparse.construct import random

from ..tools import compute_dist
from ._utils import _CheckInputs
from .base import IndependenceTest, IndependenceTestOutput
from scipy.spatial.distance import cdist
from scipy.stats import rankdata


class HHG(IndependenceTest):
    r"""
    Heller Heller Gorfine (HHG) test statistic and p-value.

    This is a powerful test for independence based on calculating pairwise
    Euclidean distances and associations between these distance matrices. The
    test statistic is a function of ranks of these distances, and is
    consistent against similar tests
    :footcite:p:`hellerConsistentMultivariateTest2013`. It can also operate on multiple
    dimensions :footcite:p:`hellerConsistentMultivariateTest2013`.

    The fast version of this test can also be run, based on
    Heller 2016 paper on multivariate tests based on univariate tests.
    The test statistic is the Hoeffding's dependence statistic from the distances of
    sample points from a single center point. Center point is the center of mass of the samples.
    This version has relatively low power, but performs well in scenarios where the center of
    mass carries a lot of information (e.g. circular or elliptical geometry).

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

    fast : boolean, default: False
        Used to opt into fast version of test if desired.

    **kwargs
        Arbitrary keyword arguments for ``compute_distance``.

    Notes
    -----
    The statistic can be derived as follows
    :footcite:p:`hellerConsistentMultivariateTest2013`:

    Let :math:`x` and :math:`y` be :math:`(n, p)` samples of random variables
    :math:`X` and :math:`Y`. For every sample :math:`j \neq i`, calculate the
    pairwise distances in :math:`x` and :math:`y` and denote this as
    :math:`d_x(x_i, x_j)` and :math:`d_y(y_i, y_j)`. The indicator function is
    denoted as :math:`\mathbb{1} \{ \cdot \}`. The cross-classification
    between these two random variables can be calculated as

    .. math::

        A_{11} = \sum_{k=1, k \neq i,j}^n
                    \mathbb{1} \{ d_x(x_i, x_k) \leq d_x(x_i, x_j) \}
                    \mathbb{1} \{ d_y(y_i, y_k) \leq d_y(y_i, y_j) \}

    and :math:`A_{12}`, :math:`A_{21}`, and :math:`A_{22}` are defined
    similarly. This is organized within the following table:

    +--------------------------------------------+--------------------------------------------+--------------------------------------------+---------------------------+
    |                                            | :math:`d_x(x_i, \cdot) \leq d_x(x_i, x_j)` | :math:`d_x(x_i, \cdot) \leq d_x(x_i, x_j)` |                           |
    +--------------------------------------------+--------------------------------------------+--------------------------------------------+---------------------------+
    | :math:`d_x(x_i, \cdot) \leq d_x(x_i, x_j)` | :math:`A_{11} (i,j)`                       | :math:`A_{12} (i,j)`                       | :math:`A_{1 \cdot} (i,j)` |
    +--------------------------------------------+--------------------------------------------+--------------------------------------------+---------------------------+
    | :math:`d_x(x_i, \cdot) > d_x(x_i, x_j)`    | :math:`A_{21} (i,j)`                       | :math:`A_{22} (i,j)`                       | :math:`A_{2 \cdot} (i,j)` |
    +--------------------------------------------+--------------------------------------------+--------------------------------------------+---------------------------+
    |                                            | :math:`A_{\cdot 1} (i,j)`                  | :math:`A_{\cdot 2} (i,j)`                  | :math:`n - 2`             |
    +--------------------------------------------+--------------------------------------------+--------------------------------------------+---------------------------+

    Here, :math:`A_{\cdot 1}` and :math:`A_{\cdot 2}` are the column sums,
    :math:`A_{1 \cdot}` and :math:`A_{2 \cdot}` are the row sums, and
    :math:`n - 2` is the number of degrees of freedom. From this table, we can
    calculate the Pearson's chi squared test statistic using,

    .. math::

        S(i, j) = \frac{(n-2) (A_{12} A_{21} - A_{11} A_{22})^2}
                       {A_{1 \cdot} A_{2 \cdot} A_{\cdot 1} A_{\cdot 2}}

    and the HHG test statistic is then,

    .. math::

        \mathrm{HHG}_n (x, y) = \sum_{i=1}^n \sum_{j=1, j \neq i}^n S(i, j)

    The p-value returned is calculated using a permutation test using
    :math:`hyppo.tools.perm_test`.

    For the fast version of the test, the test statistic is derived as follows:

    Let :math:`x` and :math:`y` be :math:`(n, p)` samples of random variables
    :math:`X` and :math:`Y`. A center point - the center of mass of points in 'X' and 'Y'
    - is chosen. For every sample :math:`i`, calculate the distances from the center point
    in :math:`x` and :math:`y` and denote this as :math:`d_x(x_i)`
    and :math:`d_y(y_i)`. This will create a 1D collection of distances for each
    sample group.

    From these distances, we can calculate the Hoeffding's dependence score between
    the two groups - denoted as D - using,

    .. math::

        D = \frac{(n-2) (n-3) D_{1} + D_{2} - 2(n-2) D_{3}}
                 {n (n-1) (n-2) (n-3) (n-4)}

        D_{1} = \sum_{i} (Q_{i}-1) (Q_{i}-2)

        D_{2} = \sum_{i} (R_{i} - 1) (R_{i} - 2) (S_{i} - 1) (S_{i} - 2)

        D_{3} = \sum_{i} {R_{i} - 2} (S_{i} - 2) (Q_{i}-1)

    where :math:`R_{i}` is the rank of :math:`x_{i}`
    and :math:`D_{i}` is the rank of :math:`y_{i}`
    and :math:`Q_{i}` is the bivariate rank = 1 plus the number of points with both x and y
    values less than the :math:`i`-th point.

    D ranges between -0.5 and 1, with 1 indicating complete dependence. D is notably
    sensitive to ties and may get smaller the more pairs of variables with identical values.

    The p-value returned is calculated using a permutation test using
    :math:`hyppo.tools.perm_test`.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, compute_distance="euclidean", fast=False, **kwargs):
        self.is_distance = False
        if not compute_distance:
            self.is_distance = True
        self.fast = fast
        IndependenceTest.__init__(self, compute_distance=compute_distance, **kwargs)

    def statistic(self, x, y):
        r"""
        Helper function that calculates the HHG test statistic.

        Parameters
        ----------
        x,y : ndarray
            Input data matrices. ``x`` and ``y`` must have the same number of
            samples. That is, the shapes must be ``(n, p)`` and ``(n, q)`` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions.
            Alternatively, ``x`` and ``y`` can be distance matrices,
            where the shapes must both be ``(n, n)``.
            For fast version, ``x`` and ``y`` can be 1D collections of distances
            from a chosen center point, where the shapes must be ``(n,1)`` or ``(n-1,1)``
            depending on choice of center point.

        Returns
        -------
        stat : float
            The computed HHG statistic.
        """
        distx = x
        disty = y

        if not self.fast:
            if not self.is_distance:
                distx, disty = compute_dist(
                    x, y, metric=self.compute_distance, **self.kwargs
                )
            S = _pearson_stat(distx, disty)
            mask = np.ones(S.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            stat = np.sum(S[mask])
            self.stat = stat

        else:
            # Fast HHG
            if not self.is_distance:
                pointer = (np.mean(x, axis=0), np.mean(y, axis=0))
                zx, zy = pointer
                zx = np.array(zx).reshape(1, -1)
                zy = np.array(zy).reshape(1, -1)
                distx, disty = _point_distance(self, x, y, zx, zy)
                distx = distx[distx != 0]
                disty = disty[disty != 0]
            distx = distx.flatten()
            disty = disty.flatten()
            stat = hoeffdingsD(distx, disty)
            self.stat = stat

        return stat

    def test(self, x, y, reps=1000, workers=1, random_state=None):
        r"""
        Calculates the HHG test statistic and p-value.

        Parameters
        ----------
        x,y : ndarray
            Input data matrices. ``x`` and ``y`` must have the same number of
            samples. That is, the shapes must be ``(n, p)`` and ``(n, q)`` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions.
            Alternatively, ``x`` and ``y`` can be distance matrices,
            where the shapes must both be ``(n, n)``.
            For fast version, ``x`` and ``y`` can be 1D collections of distances
            from a chosen center point, where the shapes must be ``(n,1)`` or ``(n-1,1)``
            depending on choice of center point.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default: 1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.

        Returns
        -------
        stat : float
            The computed HHG statistic.
        pvalue : float
            The computed HHG p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.independence import HHG
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue = HHG().test(x, y)
        >>> '%.1f, %.2f' % (stat, pvalue)
        '160.0, 0.00'

        In addition, the inputs can be distance matrices. Using this is the,
        same as before, except the ``compute_distance`` parameter must be set
        to ``None``.

        >>> import numpy as np
        >>> from hyppo.independence import HHG
        >>> x = np.ones((10, 10)) - np.identity(10)
        >>> y = 2 * x
        >>> hhg = HHG(compute_distance=None)
        >>> stat, pvalue = hhg.test(x, y)
        >>> '%.1f, %.2f' % (stat, pvalue)
        '0.0, 1.00'
        """
        check_input = _CheckInputs(x, y, reps=reps)
        x, y = check_input()

        # Fast HHG Test
        if self.fast:
            pointer = (np.mean(x, axis=0), np.mean(y, axis=0))
            zx, zy = pointer
            zx = np.array(zx).reshape(1, -1)
            zy = np.array(zy).reshape(1, -1)
            distx, disty = _point_distance(self, x, y, zx, zy)
            x = distx[distx != 0]
            y = disty[disty != 0]
            self.is_distance = True
            stat, pvalue = super(HHG, self).test(x, y, reps, workers, is_distsim=False)

        else:
            x, y = compute_dist(x, y, metric=self.compute_distance, **self.kwargs)
            self.is_distance = True
            stat, pvalue = super(HHG, self).test(x, y, reps, workers)

        return IndependenceTestOutput(stat, pvalue)


@jit(nopython=True, cache=True)
def _pearson_stat(distx, disty):  # pragma: no cover
    """Calculate the Pearson chi square stats"""

    n = distx.shape[0]
    S = np.zeros((n, n))

    # iterate over all samples in the distance matrix
    for i in range(n):
        for j in range(n):
            if i != j:
                a = distx[i, :] <= distx[i, j]
                b = disty[i, :] <= disty[i, j]

                t11 = np.sum(a * b) - 2
                t12 = np.sum(a * (1 - b))
                t21 = np.sum((1 - a) * b)
                t22 = np.sum((1 - a) * (1 - b))

                denom = (t11 + t12) * (t21 + t22) * (t11 + t21) * (t12 + t22)
                if denom > 0:
                    S[i, j] = ((n - 2) * (t12 * t21 - t11 * t22) ** 2) / denom

    return S


def _point_distance(self, x, y, zx, zy, **kwargs):
    """
    For fast HHG,
    Returns a collection of distances between sample points and chosen centre point
    """
    distx = cdist(zx, x, metric=self.compute_distance, **kwargs)
    disty = cdist(zy, y, metric=self.compute_distance, **kwargs)

    return distx, disty


def hoeffdingsD(x, y):
    """
    For fast HHG, calculates the Hoeffding's dependence statistic
    """
    xin = x
    yin = y
    # crop data to the smallest array, length have to be equal
    if len(xin) < len(yin):
        yin = yin[: len(xin)]
    if len(xin) > len(yin):
        xin = xin[: len(yin)]
    # dropna
    x = xin[~(np.isnan(xin) | np.isnan(yin))]
    y = yin[~(np.isnan(xin) | np.isnan(yin))]

    # undersampling if length too long
    lenx = len(x)
    if lenx > 99999:
        factor = math.ceil(lenx / 100000)
        x = x[::factor]
        y = y[::factor]

    R = rankdata(x)
    S = rankdata(y)

    # core processing
    N = x.shape
    dico = {(np.nan, np.nan): np.nan}
    dicoRin = {np.nan: np.nan}
    dicoSin = {np.nan: np.nan}
    dicoRless = {np.nan: np.nan}
    dicoSless = {np.nan: np.nan}
    Q = np.ones(N[0])

    i = 0
    for r, s in np.nditer([R, S]):
        r = float(r)
        s = float(s)
        if (r, s) in dico.keys():
            Q[i] = dico[(r, s)]
        else:
            if r in dicoRin.keys():
                isinR = dicoRin[r]
                lessR = dicoRless[r]
            else:
                isinR = np.isin(R, r)
                dicoRin[r] = isinR
                lessR = np.less(R, r)
                dicoRless[r] = lessR

            if s in dicoSin.keys():
                isinS = dicoSin[s]
                lessS = dicoSless[s]
            else:
                isinS = np.isin(S, s)
                dicoSin[s] = isinS
                lessS = np.less(S, s)
                dicoSless[s] = lessS

            Q[i] = (
                Q[i]
                + np.count_nonzero(lessR & lessS)
                + 1 / 4 * (np.count_nonzero(isinR & isinS) - 1)
                + 1 / 2 * (np.count_nonzero(isinR & lessS))
                + 1 / 2 * (np.count_nonzero(lessR & isinS))
            )
            dico[(r, s)] = Q[i]
        i += 1

    D1 = np.sum(np.multiply((Q - 1), (Q - 2)))
    D2 = np.sum(
        np.multiply(np.multiply((R - 1), (R - 2)), np.multiply((S - 1), (S - 2)))
    )
    D3 = np.sum(np.multiply(np.multiply((R - 2), (S - 2)), (Q - 1)))

    D = (
        30
        * ((N[0] - 2) * (N[0] - 3) * D1 + D2 - 2 * (N[0] - 2) * D3)
        / (N[0] * (N[0] - 1) * (N[0] - 2) * (N[0] - 3) * (N[0] - 4))
    )
    return D
