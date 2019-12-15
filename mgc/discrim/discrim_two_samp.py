from ._utils import _CheckInputs
import numpy as np
import random
from numba import njit
from .base import DiscriminabilityTest
from scipy._lib._util import MapWrapper


class DiscrimTwoSample(DiscriminabilityTest):
    r"""
     A class that compares the discriminability of two dataset.
     
     Two sample test measures whether the discriminability is different for 
     one dataset than that of another. More details can be described in [#1Dscr]_.
    
    Parameters
    ---------- 
    is_dist : Boolean, optional (default: False)
        whether `x1` and `x2` are distance matrices or not.
    remove_isolates : Bool, optional (default: True)
        whether to remove the measurements with single instance or not.

    See Also
    --------
    DiscrimOneSample : One sample test for accessing discriminability of a single measurement
    
    Notes
    -----
    With :math:`\hat D_{X_1}` the sample discriminability of one approach, 
    and :math:`\hat D_{X_2}` the sample discriminability of another approach:
    
     .. math::

         H_0: D_{X_1} = D_{X_2}

    and

     .. math::

         H_A: D_{X_1} > D_{X_2}  
    
    Also implemented are tests of :math:`<` and :math:`\neq`.
    """

    def __init__(self, is_dist=False, remove_isolates=True):
        self.is_distance = is_dist
        self.remove_isolates = remove_isolates
        DiscriminabilityTest.__init__(self)

    def _statistic(self, x, y):
        """
        Helper function that calculates the discriminability test statistics.
        """
        stat = super(DiscrimTwoSample, self)._statistic(x, y)

        return stat

    def test(self, x1, x2, y, reps=1000, alt="neq", workers=-1):
        r"""
        Calculates the test statistic and p-value for Discriminability two sample test.

        Parameters
        ----------
        x1, x2 : ndarray
            An `(n, d)` data matrix with `n` samples in `d` dimensions,
            if flag is_dist = Flase and an `(n, n)` distance matrix,
            if flag is_dist = True
        y : ndarray
            A vector containing the sample ids for our :math:`n` samples. Should be matched such that :math:`y[i]` 
            is the corresponding label for :math:`x_1[i,]` and :math:`x_2[i,]`.
        reps : int, optional (default: 1000)
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        alt : options: {'greater', 'less', 'neq'}, default: 'neq'
            The alternative hypothesis for the test. Can be that first dataset is more discriminable (alt = "greater"),
            less discriminable (alt = "less") or just non-equal (alt = "neq").
        workers : int, optional (default: -1)
            The number of cores to parallelize the p-value computation over.
            Supply -1 to use all cores available to the Process.

        Returns
        -------
        d1 : float
            The computed discriminability score for `x_1`.
        d2 : float
            The computed discriminability score for `x_2`.
        pvalue : float
            The computed two sample test p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from mgc.discrim import DiscrimTwoSample
        >>> x1 = np.ones((100,2),dtype=float)
        >>> x2 = np.concatenate((np.zeros((50,2)),np.ones((50,2))), axis=0)
        >>> y = np.concatenate((np.zeros(50),np.ones(50)), axis=0)
        >>> d1, d2, p = DiscrimTwoSample().test(x1,x2,y) 
        >>> '%.1f, %.lf, %.2f' % (d1, d2, p)
        '0.5, 1, 1.00'
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

        self.d1 = self._statistic(self.x1, y)
        self.d2 = self._statistic(self.x2, y)
        self.da = self.d1 - self.d2

        # use all cores to create function that parallelizes over number of reps
        mapwrapper = MapWrapper(workers)
        null_dist = np.array(list(mapwrapper(self._perm_stat, range(reps))))
        self.null_dist = null_dist

        self.diff_null = []

        for i in range(0, reps - 1):
            for j in range(i + 1, reps):
                self.diff_null.append(self.null_dist[i][0] - self.null_dist[j][1])
                self.diff_null.append(self.null_dist[j][1] - self.null_dist[i][0])

        self.diff_null = np.asarray(self.diff_null)

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

    def _get_convex_comb(self, x):
        n, _ = x.shape

        q1 = np.random.choice(n, n)
        q2 = np.random.choice(n, n)
        lamda = np.random.uniform(size=n)

        return (lamda * (x[q1]).T + (1 - lamda) * (x[q2]).T).T

    def _perm_stat(self, index):
        permx1 = self._get_convex_comb(self.x1)
        permx2 = self._get_convex_comb(self.x2)
        permy = self.y

        perm_stat1 = self._statistic(permx1, permy)
        perm_stat2 = self._statistic(permx2, permy)

        return perm_stat1, perm_stat2
