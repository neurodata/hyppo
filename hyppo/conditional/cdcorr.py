from functools import partial

import numpy as np
from scipy.spatial.distance import pdist, squareform

from ..tools import compute_dist, perm_test
from ._utils import _CheckInputs
from .base import ConditionalIndependenceTest, ConditionalIndependenceTestOutput


class ConditionalDcorr(ConditionalIndependenceTest):
    r"""
    Conditional Distance Covariance/Correlation (CDcov/CDcorr) test statistic and p-value.

    CDcorr is a measure of dependence between two paired random matrices
    given a third random matrix of not necessarily equal dimensions :footcite:p:`wang2015conditional`.
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

        ConditionalIndependenceTest.__init__(self, **kwargs)

    def __repr__(self):
        return "ConditionalDcorr"

    def statistic(self, x, y, z):
        r"""
        Helper function that calculates the CDcov/CDcorr test statistic.

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
            The computed CDcov/CDcorr statistic.
        """
        check_input = _CheckInputs(x, y, z, ignore_z_var=True)
        x, y, z = check_input()

        if not self.is_distance:
            distx, disty = compute_dist(
                x,
                y,
                metric=self.compute_distance,
            )
            distz = self._compute_kde(z)
        else:
            distx = x
            disty = y
            distz = z

        if self.use_cov:
            stat = _cdcov(distx, disty, distz).mean()
        else:
            stat = _cdcorr(distx, disty, distz).mean()

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
        Calculates the CDcov/CDcorr test statistic and p-value.

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
        check_input = _CheckInputs(x, y, z, reps=reps, ignore_z_var=True)
        x, y, z = check_input()

        if not self.is_distance:
            x, y = compute_dist(x, y, metric=self.compute_distance, **self.kwargs)
            z = self._compute_kde(z)

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
            permuter=partial(self._permuter, probs=z),
        )
        self.stat = stat
        self.pvalue = pvalue
        self.null_dist = null_dist

        return ConditionalIndependenceTestOutput(stat, pvalue)

    def _compute_kde(self, data):
        n, d = data.shape

        if isinstance(self.bandwidth, (int, float)):
            bandwidth_ = np.repeat(self.bandwidth, d)
        elif isinstance(self.bandwidth, np.ndarray):
            if not d == self.bandwidth.size:
                raise ValueError(f"`bandwidth` must be of length {d}.")
            bandwidth_ = self.bandwidth
        elif self.bandwidth is None or self.bandwidth == "scott":
            factor = np.power(n, (-1.0 / (d + 4)))
            stds = np.std(data, axis=0, ddof=1)
            bandwidth_ = factor * stds
        elif self.bandwidth == "silverman":
            factor = (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))
            stds = np.std(data, axis=0, ddof=1)
            bandwidth_ = factor * stds

        # Compute constants
        denom = np.power(2 * np.pi, d / 2) * np.power(bandwidth_.prod(), 0.5)

        # Compute weighted kernel
        sim_mat = pdist(
            data, "sqeuclidean", w=1 / bandwidth_**2
        )  # Section 4.2 equation
        sim_mat = squareform(-0.5 * sim_mat)
        np.exp(sim_mat, sim_mat)
        sim_mat /= denom

        return sim_mat

    def _permuter(self, probs, rng=None, axis=1):
        """
        Weighted sampling with replacement
        """
        if rng is None:
            rng = np.random.default_rng()

        n = probs.shape[1 - axis]
        sums = probs.sum(axis=1, keepdims=True)
        idx = ((probs / sums).cumsum(axis=1) > rng.random(n)[:, None]).argmax(axis=1)

        return idx


def _weighted_center_distmat(distx, weights): # pragma: no cover
    """Centers the distance matrices with weights"""

    n = distx.shape[0]

    scl = np.sum(weights)
    row_sum = np.sum(np.multiply(distx, weights), axis=1) / scl
    total_sum = weights @ row_sum / scl

    cent_distx = distx - row_sum.reshape(-1, n).T - row_sum.reshape(-1, n) + total_sum

    return cent_distx


def _cdcov(distx, disty, distz): # pragma: no cover
    """Calculate the CDcov test statistic"""
    n = distx.shape[0]

    cdcov = np.zeros(n)

    for i in range(n):
        r = distz[[i]]
        cdx = _weighted_center_distmat(distx, distz[i])
        cdy = _weighted_center_distmat(disty, distz[i])
        cdcov[i] = (cdx * cdy * r * r.T).sum() / r.sum() ** 2

    cdcov *= 12 * np.power(distz.mean(axis=0), 4)

    return cdcov


def _cdcorr(distx, disty, distz): # pragma: no cover
    """Calculate the CDcorr test statistic"""
    varx = _cdcov(distx, distx, distz)
    vary = _cdcov(disty, disty, distz)
    covar = _cdcov(distx, disty, distz)

    # stat is 0 with negative variances (would make denominator undefined)
    denom = varx * vary
    non_positives = denom <= 0
    if np.any(non_positives):
        denom[non_positives] = 0
    stat = np.divide(
        covar, np.sqrt(denom), out=np.zeros_like(covar), where=np.invert(non_positives)
    )

    return stat
