import numpy as np
from numba import jit

from ..independence.dcorr import _center_distmat
from ..tools import compute_dist, perm_test
from ._utils import _CheckInputs
from .base import ConditionalIndependenceTest, ConditionalIndependenceTestOutput


class PartialDcorr(ConditionalIndependenceTest):
    r"""
    Partial Distance Covariance/Correlation (PDcov/PDcorr) test statistic and p-value.

    CDcorr is a measure of dependence between two paired random matrices
    given a third random matrix of not necessarily equal dimensions :footcite:p:`szekelyPartialDistanceCorrelation2014a`.
    The coefficient is 0 if and only if the matrices are independent given
    third matrix.

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
            The computed CDcov/CDcorr statistic.
        pvalue : float
            The computed CDcov/CDcorr p-value.
        """
        check_input = _CheckInputs(x, y, z, reps=reps)
        x, y, z = check_input()

        if not self.is_distance:
            x, y, z = compute_dist(x, y, z, metric=self.compute_distance, **self.kwargs)
            self.is_distance = True

        stat, pvalue, null_dist = perm_test(
            self.statistic,
            x,
            y,
            z,
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
def _pdcov(distx, disty, distz):
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
def _pdcorr(distx, disty, distz):
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
