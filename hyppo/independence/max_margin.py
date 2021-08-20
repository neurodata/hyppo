import numpy as np

from ..tools import chi2_approx
from ._utils import _CheckInputs
from .base import IndependenceTest
from .cca import CCA
from .dcorr import Dcorr
from .hhg import HHG
from .hsic import Hsic
from .kmerf import KMERF
from .mgc import MGC
from .rv import RV

INDEP_NOT_MAXMARGIN = {
    "rv": RV,
    "cca": CCA,
    "hhg": HHG,
    "hsic": Hsic,
    "dcorr": Dcorr,
    "mgc": MGC,
    "kmerf": KMERF,
}


class MaxMargin(IndependenceTest):
    r"""
    Maximal Margin test statistic and p-value.

    This test loops over each of the dimensions of the inputs :math:`x` and :math:`y`
    and computes the desired independence test statistic. Then, the maximial test
    statistic is chosen `[1]`_.

    The p-value returned is calculated using a permutation test using
    :meth:`hyppo.tools.perm_test`.

    .. _[1]: https://arxiv.org/abs/2001.01095

    Parameters
    ----------
    indep_test : "CCA", "Dcorr", "HHG", "RV", "Hsic", "MGC", "KMERF"
        A string corresponding to the desired independence test from
        :mod:`hyppo.independence`. This is not case sensitive.
    compute_distkern : str, callable, or None, default: "euclidean" or "gaussian"
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

        Alternatively, this function computes the kernel similarity among the
        samples within each data matrix.
        Valid strings for ``compute_kernel`` are, as defined in
        :func:`sklearn.metrics.pairwise.pairwise_kernels`,

            [``"additive_chi2"``, ``"chi2"``, ``"linear"``, ``"poly"``,
            ``"polynomial"``, ``"rbf"``,
            ``"laplacian"``, ``"sigmoid"``, ``"cosine"``]

        Note ``"rbf"`` and ``"gaussian"`` are the same metric.
    bias : bool, default: False
        Whether or not to use the biased or unbiased test statistics (for
        ``indep_test="Dcorr"`` and ``indep_test="Hsic"``).
    **kwargs
        Arbitrary keyword arguments for ``compute_distkern``.
    """

    def __init__(self, indep_test, compute_distkern="euclidean", bias=False, **kwargs):
        indep_test = indep_test.lower()
        if indep_test not in INDEP_NOT_MAXMARGIN.keys():
            raise ValueError("Test is not a valid independence test")
        if indep_test == "hsic" and compute_distkern == "euclidean":
            compute_distkern = "gaussian"
        self.indep_test_name = indep_test

        indep_kwargs = {
            "dcorr": {"bias": bias, "compute_distance": compute_distkern},
            "hsic": {"bias": bias, "compute_kernel": compute_distkern},
            "hhg": {"compute_distance": compute_distkern},
            "mgc": {"compute_distance": compute_distkern},
            "kmerf": {"forest_type": "classifier"},
            "rv": {},
            "cca": {},
        }

        self.indep_test = INDEP_NOT_MAXMARGIN[indep_test](
            **indep_kwargs[indep_test], **kwargs
        )

        IndependenceTest.__init__(self, compute_distance=compute_distkern, **kwargs)

    def statistic(self, x, y):
        r"""
        Helper function that calculates the Maximal Margin test statistic.

        Parameters
        ----------
        x,y : ndarray
            Input data matrices. ``x`` and ``y`` must have the same number of
            samples. That is, the shapes must be ``(n, p)`` and ``(n, q)`` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions.

        Returns
        -------
        stat : float
            The computed Maximal Margin statistic.
        """
        stat = np.max(
            [
                self.indep_test.statistic(
                    x[:, i].reshape(-1, 1), y[:, j].reshape(-1, 1)
                )
                for i in range(x.shape[1])
                for j in range(y.shape[1])
            ]
        )
        self.stat = stat

        return stat

    def test(self, x, y, reps=1000, workers=1, auto=True):
        r"""
        Calculates the Maximal Margin test statistic and p-value.

        Parameters
        ----------
        x,y : ndarray
            Input data matrices. ``x`` and ``y`` must have the same number of
            samples. That is, the shapes must be ``(n, p)`` and ``(n, q)`` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default: 1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.
        auto : bool, default: True
            Only applies to ``"Dcorr"`` and ``"Hsic"``.
            Automatically uses fast approximation when `n` and size of array
            is greater than 20. If ``True``, and sample size is greater than 20, then
            :class:`hyppo.tools.chi2_approx` will be run. Parameters ``reps`` and
            ``workers`` are
            irrelevant in this case. Otherwise, :class:`hyppo.tools.perm_test` will be
            run.

        Returns
        -------
        stat : float
            The computed Maximal Margin statistic.
        pvalue : float
            The computed Maximal Margin p-value.
        dict
            A dictionary containing optional parameters for tests that return them.
            See the relevant test in :mod:`hyppo.independence`.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.independence import MaxMargin
        >>> x = np.arange(100)
        >>> y = x
        >>> stat, pvalue = MaxMargin("Dcorr").test(x, y)
        >>> '%.1f, %.3f' % (stat, pvalue)
        '1.0, 0.000'
        """
        check_input = _CheckInputs(
            x,
            y,
            reps=reps,
        )
        x, y = check_input()

        if auto and x.shape[0] > 20 and self.indep_test_name in ["dcorr", "hsic"]:
            stat, pvalue = chi2_approx(self.statistic, x, y)
            self.stat = stat
            self.pvalue = pvalue
            self.null_dist = None
        else:
            stat, pvalue = super(MaxMargin, self).test(
                x, y, reps, workers, is_distsim=False
            )

        return stat, pvalue
