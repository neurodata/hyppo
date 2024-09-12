import numpy as np
from numba import jit

from ..independence.dcorr import _center_distmat
from ..tools import compute_dist, perm_test
from ._utils import _CheckInputs
from .base import ConditionalIndependenceTest, ConditionalIndependenceTestOutput


class PartialDcorr(ConditionalIndependenceTest):
    r"""
    Partial Distance Covariance/Correlation (PDcov/PDcorr) test statistic and p-value.

    PDcorr is a measure of dependence between two paired random matrices
    given a third random matrix of not necessarily equal dimensions :footcite:p:`szekelyPartialDistanceCorrelation2014a`.

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
    use_cov : bool,
        If `True`, then the statistic will compute the covariance rather than the
        correlation.
    **kwargs
        Arbitrary keyword arguments for ``compute_distance``.

    Notes
    -----
    The statistic can be derived as follows:

    Let :math:`x`, :math:`y`, and :math:`z` be :math:`(n, p)` samples of random
    variables :math:`X`, :math:`Y` and :math:`Z`. Let :math:`D^x` be the :math:`n \times n`
    distance matrix of :math:`x`, :math:`D^y` be the :math:`n \times n` be
    the distance matrix of :math:`y`, and :math:`D^z` be the :math:`n \times n` distance
    matrix of :math:`z`. Let :math:`C^x`, :math:`C^y`, and :math:`C^z` be the unbiased centered
    distance matrices (see :class:`hyppo.independence.Dcorr` for more details). The
    partial distance covariance is defined as

    .. math::
        \mathrm{PDcov}_n (x, y; z) = \frac{1}{n(n-3)} \sum_{i\neq j}^n \left(P_{z^\perp}(x)\right)_{i,j} \left(P_{z^\perp}(y)\right)_{i,j}

    where

    .. math::
        P_{z^\perp}(x) = C^x - \frac{(C^x\cdot C^z)}{ C^z \cdot C^z) C^z

    is the orthogonal proejction of :math:`C^x` onto the subspace orthogonal to :math:`C^z`.
    The partial distance correlation is defined as

    .. math::
        \mathrm{PDcorr}_n (x, y; z) = \frac{P_{z^\perp}(x)\cdot P_{z^\perp}(y)}{|P_{z^\perp}(x)} |P_{z^\perp}(y)|}

    Equivalently, the partial distance correlation can be also defined as

    .. math::
        \mathrm{CDcorr}_n (x, y; z) =  \frac{R_{xy} - R_{xz} R_{yz}}{\sqrt{(1 - R_{xz}^2)(1 - R_{yz}^2)}}

    where :math:`R_{xy}` is the unbiased distance correlation between :math:`x` and :math:`y`.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, compute_distance="euclidean", use_cov=True, **kwargs):
        self.use_cov = use_cov
        self.compute_distance = compute_distance

        # set is_distance to true if compute_distance is None
        self.is_distance = False
        if not compute_distance:
            self.is_distance = True

        ConditionalIndependenceTest.__init__(self, **kwargs)

    def __repr__(self):
        return "PartialDcorr"

    def statistic(self, x, y, z):
        r"""
        Helper function that calculates the PDcov/PDcorr test statistic.

        Parameters
        ----------
        x,y,z : ndarray of float
            Input data matrices. ``x``, ``y`` and ``z`` must have the same number
            of samples. That is, the shapes must be ``(n, p)``, ``(n, q)`` and
            ``(n, r)`` where `n` is the number of samples and `p`, `q`, and `r`
            are the number of dimensions. Alternatively, ``x`` and ``y`` can be
            distance matrices and ``z`` can be a similarity matrix where the
            shapes must be ``(n, n)``.

        Returns
        -------
        stat : float
            The computed PDcov/PDcorr statistic.
        """
        check_input = _CheckInputs(x, y, z)
        x, y, z = check_input()

        if not self.is_distance:
            distx, disty, distz = compute_dist(
                x,
                y,
                z,
                metric=self.compute_distance,
            )
        else:
            distx = x
            disty = y
            distz = z

        if self.use_cov:
            stat = _pdcov(distx, disty, distz)
        else:
            stat = _pdcorr(distx, disty, distz)

        self.stat = stat
        return stat

    def test(
        self,
        x,
        y,
        z,
        reps=1000,
        workers=1,
        random_state=None,
    ):
        r"""
        Calculates the PDcov/PDcorr test statistic and p-value.

        Parameters
        ----------
        x,y,z : ndarray of float
            Input data matrices. ``x``, ``y`` and ``z`` must have the same number
            of samples. That is, the shapes must be ``(n, p)``, ``(n, q)`` and
            ``(n, r)`` where `n` is the number of samples and `p`, `q`, and `r`
            are the number of dimensions. Alternatively, ``x`` and ``y`` can be
            distance matrices and ``z`` can be a similarity matrix where the
            shapes must be ``(n, n)``.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default: 1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.
        random_state : int, default: None
            The random_state for permutation testing to be fixed for
            reproducibility.

        Returns
        -------
        stat : float
            The computed PDcov/PDcorr statistic.
        pvalue : float
            The computed PDcov/PDcorr p-value.
        """
        check_input = _CheckInputs(x, y, z, reps=reps)
        x, y, z = check_input()

        if not self.is_distance:
            x, y, z = compute_dist(x, y, z, metric=self.compute_distance, **self.kwargs)
            self.is_distance = True

        stat, pvalue, null_dist = perm_test(
            self.statistic,
            x=x,
            y=y,
            z=z,
            reps=reps,
            workers=workers,
            is_distsim=self.is_distance,
            random_state=random_state,
        )
        self.stat = stat
        self.pvalue = pvalue
        self.null_dist = null_dist

        return ConditionalIndependenceTestOutput(stat, pvalue)


@jit(nopython=True, cache=True)
def _pdcov(distx, disty, distz):  # pragma: no cover
    """Calculate the PDcov test statistic"""
    N = distx.shape[0]
    denom = N * (N - 3)

    distx = _center_distmat(distx, bias=False)
    disty = _center_distmat(disty, bias=False)
    distz = _center_distmat(distz, bias=False)

    cov_xz = np.sum(distx * distz)
    cov_yz = np.sum(disty * distz)
    var_z = np.sum(distz * distz)

    if var_z == 0:
        stat = 0
    else:
        proj_xz = distx - (cov_xz / var_z) * distz
        proj_yz = disty - (cov_yz / var_z) * distz
        stat = np.sum(proj_xz * proj_yz) / denom

    return stat


@jit(nopython=True, cache=True)
def _pdcorr(distx, disty, distz):  # pragma: no cover
    """Calculate the PDcorr test statistic"""

    distx = _center_distmat(distx, bias=False)
    disty = _center_distmat(disty, bias=False)
    distz = _center_distmat(distz, bias=False)

    cov_xy = np.sum(distx * disty)
    cov_xz = np.sum(distx * distz)
    cov_yz = np.sum(disty * distz)
    var_x = np.sum(distx * distx)
    var_y = np.sum(disty * disty)
    var_z = np.sum(distz * distz)

    if var_x <= 0 or var_y <= 0:
        rxy = 0
    else:
        rxy = cov_xy / np.real(np.sqrt(var_x * var_y))

    if var_x <= 0 or var_z <= 0:
        rxz = 0
    else:
        rxz = cov_xz / np.real(np.sqrt(var_x * var_z))

    if var_y <= 0 or var_z <= 0:
        ryz = 0
    else:
        ryz = cov_yz / np.real(np.sqrt(var_y * var_z))

    stat = (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))

    return stat
