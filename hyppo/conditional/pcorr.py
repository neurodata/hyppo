import numpy as np
from scipy.stats import t

from ..tools import perm_test
from ._utils import _CheckInputs
from .base import ConditionalIndependenceTest, ConditionalIndependenceTestOutput


class PartialCorr(ConditionalIndependenceTest):
    r"""
    Conditional Pearson's correlation test.

    Partial correlation is a measure of the association between two univariate
    variables given a third univariate variable.

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

    Notes
    -----
    The statistic is computed as follows:

    .. math::
        r_{x, y ; z} = \frac{\rho_{xy} - \rho_{xz} \rho_{yz}}{\sqrt{(1 - \rho_{xz}^2)(1 - \rho_{yz}^2)}}

    where :math:`\rho_{xy}` is the Pearson correlation coefficient between :math:`x` and
    :math:`y`. The partial correlation test is implemented as a t-test footcite:p:`legendre2000`:.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, **kwargs):
        ConditionalIndependenceTest.__init__(self, **kwargs)

    def statistic(self, x, y, z):
        r"""
        Helper function that calculates the partial correlation test statistic.

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
            The computed partial correlation test statistic.
        """
        check_input = _CheckInputs(x, y, z, max_dims=1)
        x, y, z = check_input()

        corr = np.corrcoef(np.hstack([x, y, z]), rowvar=False)
        cov_xy, cov_xz, cov_yz = corr[np.triu_indices_from(corr, k=1)]

        if cov_xz <= 0 or cov_yz <= 0:
            stat = 0
        else:
            stat = (cov_xy - cov_xz * cov_yz) / np.sqrt(
                (1 - cov_xz**2) * (1 - cov_yz**2)
            )

        self.stat = stat
        return stat

    def test(
        self,
        x,
        y,
        z,
        reps=1000,
        workers=1,
        auto=True,
        perm_blocks=None,
        random_state=None,
    ):
        r"""
        Calculates the partial correlation test statistic and p-value.

        Parameters
        ----------
        x,y,z : ndarray of float
            Input data matrices. ``x``, ``y`` and ``z`` must have the same number
            of samples. That is, the shapes must be ``(n, 1)``, ``(n, 1)`` and
            ``(n, 1)`` where `n` is the number of samples and `p`, `q`, and `r`
            are the number of dimensions.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default: 1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.
        auto : bool, default: True
            If True, the p-value is computed by t-distribution approximation.
            Parameters ``reps`` and ``workers`` are irrelevant in this case.
            Otherwise, :class:`hyppo.tools.perm_test` will be run.
        perm_blocks : None or ndarray, default: None
            Defines blocks of exchangeable samples during the permutation test.
            If None, all samples can be permuted with one another. Requires `n`
            rows. At each column, samples with matching column value are
            recursively partitioned into blocks of samples. Within each final
            block, samples are exchangeable. Blocks of samples from the same
            partition are also exchangeable between one another. If a column
            value is negative, that block is fixed and cannot be exchanged.
        random_state : int, default: None
            The random_state for permutation testing to be fixed for
            reproducibility.

        Returns
        -------
        stat : float
            The computed partial correlation test statistic.
        pvalue : float
            The computed partial correlation test p-value.
        """
        check_input = _CheckInputs(x, y, z, reps=reps, max_dims=1)
        x, y, z = check_input()

        if auto:  # run t-stat
            stat = self.statistic(x, y, z)

            n = x.shape[0]
            dof = n - 3
            tstat = stat * np.sqrt(dof) / np.sqrt(1 - stat**2)
            pvalue = 1 - t.cdf(np.abs(tstat), dof)

            self.stat = stat
            self.pvalue = pvalue
            self.null_dist = None
        else:  # run permutation
            stat, pvalue, null_dist = perm_test(
                self.statistic,
                x=x,
                y=y,
                z=z,
                reps=reps,
                workers=workers,
                is_distsim=False,
                perm_blocks=perm_blocks,
                random_state=random_state,
            )

            self.stat = stat
            self.pvalue = pvalue
            self.null_dist = null_dist

        return ConditionalIndependenceTestOutput(stat, pvalue)
