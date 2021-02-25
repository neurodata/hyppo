import numpy as np
from numba import jit
from scipy._lib._util import MapWrapper

from ._utils import _CheckInputs
from .base import DiscriminabilityTest


class DiscrimTwoSample(DiscriminabilityTest):
    r"""
    A class that compares the discriminability of two datasets.
    Two sample test measures whether the discriminability is different for
    one dataset compared to another. More details can be described in `[1]`_.

    Let :math:`\hat D_{x_1}` denote the sample discriminability of one approach,
    and :math:`\hat D_{x_2}` denote the sample discriminability of another approach.
    Then,
    .. math::
        H_0: D_{x_1} &= D_{x_2} \\
        H_A: D_{x_1} &> D_{x_2}
    Alternatively, tests can be done for :math:`D_{x_1} < D_{x_2}` and
    :math:`D_{x_1} \neq D_{x_2}`.

    .. _[1]: https://www.biorxiv.org/content/10.1101/802629v1

    Parameters
    ----------
    is_dist : bool, default: False
        Whether `x1` and `x2` are distance matrices or not.
    remove_isolates : bool, default: True
        Whether to remove the measurements with a single instance or not.
    """

    def __init__(self, is_dist=False, remove_isolates=True):
        self.is_distance = is_dist
        self.remove_isolates = remove_isolates
        DiscriminabilityTest.__init__(self)

    def statistic(self, x, y):
        """
        Helper function that calculates the discriminability test statistic.

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
            The computed two sample discriminability statistic.
        """
        stat = super(DiscrimTwoSample, self).statistic(x, y)

        return stat

    def test(self, x1, x2, y, reps=1000, alt="neq", workers=-1):
        r"""
        Calculates the test statistic and p-value for a two sample test for
        discriminability.

        Parameters
        ----------
        x1, x2 : ndarray
            Input data matrices. `x1` and `x2` must have the same number of
            samples. That is, the shapes must be `(n, p)` and `(n, q)` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions. Alternatively, `x1` and `x2` can be distance matrices,
            where the shapes must both be `(n, n)`, and ``is_dist`` must set
            to ``True`` in this case.
        y : ndarray
            A vector containing the sample ids for our `n` samples. Should be matched
            to the inputs such that ``y[i]`` is the corresponding label for
            ``x_1[i, :]`` and ``x_2[i, :]``.
        reps : int, optional (default: 1000)
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        alt : {"greater", "less", "neq"} (default: "neq")
            The alternative hypothesis for the test. Can test that first dataset is
            more discriminable (alt = "greater"), less discriminable (alt = "less")
            or unequal discriminability (alt = "neq").
        workers : int, optional (default: -1)
            The number of cores to parallelize the p-value computation over.
            Supply -1 to use all cores available to the Process.

        Returns
        -------
        d1 : float
            The computed discriminability score for ``x1``.
        d2 : float
            The computed discriminability score for ``x2``.
        pvalue : float
            The computed two sample test p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.discrim import DiscrimTwoSample
        >>> x1 = np.ones((100,2), dtype=float)
        >>> x2 = np.concatenate([np.zeros((50, 2)), np.ones((50, 2))], axis=0)
        >>> y = np.concatenate([np.zeros(50), np.ones(50)], axis=0)
        >>> discrim1, discrim2, pvalue = DiscrimTwoSample().test(x1, x2, y)
        >>> '%.1f, %.1f, %.2f' % (discrim1, discrim2, pvalue)
        '0.5, 1.0, 0.00'
        """

        check_input = _CheckInputs(
            [x1, x2],
            y,
            reps=reps,
            is_dist=self.is_distance,
            remove_isolates=self.remove_isolates,
        )
        x, y = check_input()
        self.x1 = np.asarray(x[0])
        self.x2 = np.asarray(x[1])
        self.y = y

        self.d1 = self.statistic(self.x1, y)
        self.d2 = self.statistic(self.x2, y)
        self.da = self.d1 - self.d2

        # use all cores to create function that parallelizes over number of reps
        mapwrapper = MapWrapper(workers)
        null_dist = np.array(list(mapwrapper(self._perm_stat, range(reps))))

        self.diff_null = np.asarray(calculate_diff_null(null_dist, reps))

        if alt == "greater":
            pvalue = (self.diff_null > self.da).mean()
        elif alt == "less":
            pvalue = (self.diff_null < self.da).mean()
        elif alt == "neq":
            pvalue = (abs(self.diff_null) > abs(self.da)).mean()
        else:
            msg = "You have not entered a valid alternative."
            raise ValueError(msg)

        if pvalue == 0:
            pvalue = 1 / reps

        self.pvalue = pvalue

        return self.d1, self.d2, self.pvalue

    def _get_convex_comb(self, x):  # pragma: no cover
        """Get random convex combination of input x."""
        n, _ = x.shape

        q1 = np.random.choice(n, n)
        q2 = np.random.choice(n, n)
        lamda = np.random.uniform(size=n)

        return (lamda * (x[q1]).T + (1 - lamda) * (x[q2]).T).T

    def _perm_stat(self, index):  # pragma: no cover
        r"""
        Helper function that is used to calculate parallel permuted test
        statistics.

        Parameters
        ----------
        index : int
            Iterator used for parallel statistic calculation

        Returns
        -------
        perm_stat1, perm_stat2 : float
            Test statistic for each value in the null distribution.
        """
        permx1 = self._get_convex_comb(self.x1)
        permx2 = self._get_convex_comb(self.x2)

        perm_stat1 = self.statistic(permx1, self.y)
        perm_stat2 = self.statistic(permx2, self.y)

        return perm_stat1, perm_stat2


@jit(nopython=True, cache=True)
def calculate_diff_null(null_dist, reps):  # pragma: no cover
    """
    Helper function to calculate the distribution of thedifference under
    null.
    """
    diff_null = []

    for i in range(0, reps - 1):
        for j in range(i + 1, reps):
            diff_null.append(null_dist[i][0] - null_dist[j][1])
            diff_null.append(null_dist[j][1] - null_dist[i][0])

    return diff_null
