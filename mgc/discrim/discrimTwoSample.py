from sklearn.metrics import euclidean_distances
from ._utils import _CheckInputs
import numpy as np
import random
from .base import DiscriminabilityTest
from scipy._lib._util import MapWrapper


class DiscrimTwoSample(DiscriminabilityTest):
    r"""
     A class that performs a two-sample test for whether the discriminability is different for that of
     one dataset vs another, as described in [#1Dscr]_.
    
    Parameters
    ---------- 
    is_dist : Boolean, optional (default: False)
        whether `x1` and `x2` are distance matrices or not.
    remove_isolates : Boolean, optional (default: True)
        whether remove the samples with single instance or not.

    See Also
    --------
    DiscrimOneSample : Onesample test for accessing discriminability of a single measurement
    
    Notes
    -----
    With :math:`\hat D_{X_1}` the sample discriminability of one approach, and
    :math:`\hat D_{X_2}` the sample discriminability of another approach:
    
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
        stat_ = super(DiscrimTwoSample, self)._statistic(x, y)

        return stat_

    def test(self, x1, x2, y, reps=1000, alt="greater", workers=-1):
        r"""
        Calculates the test statistic and p-value for Discriminability two sample test.

        Parameters
        ----------
        x1 : ndarray

            * An `n \times d` data matrix with `n` samples in `d` dimensions. Should not be a distance matrix.  

        x2 : ndarray

            * An `n \times d` data matrix with `n` samples in `d` dimensions. Should not be a distance matrix.

        y : ndarray
            A vector containing the sample ids for our :math:`n` samples. Should be matched such that :math:`y[i]` 
            is the corresponding label for :math:`x_1[i,]` and :math:`x_2[i,]`.
        remove_isolates : Boolean, optional (default: True)
            whether remove the samples with single instance or not.
        reps : int, optional (default: 1000)
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        alt : string, optional(default: "greater")
            The alternative hypothesis for the test. Can be that first dataset is more discriminable (alt = "greater"),
            less discriminable (alt = "less") or just non-equal (alt = "neq").
        workers : int, optional (default: -1)
            The number of cores to parallelize the p-value computation over.
            Supply -1 to use all cores available to the Process.

        Returns
        -------
        D1 : float
            The computed discriminability score for :math:`x_1`.
        D2 : float
            The computed discriminability score for :math:`x_2`.
        pvalue : float
            The computed two sample test p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from mgc.discrim import DiscrimTwoSample
        >>> x1 = np.ones((100,2),dtype=float)
        >>> x2 = np.concatenate((np.zeros((50,2)),np.ones((50,2))), axis= 0)
        >>> y = np.concatenate((np.zeros(50),np.ones(50)), axis= 0)
        >>> D1, D2, p = DiscrimTwoSample().test(x1,x2,y) 
        >>> '%1f, %lf, %lf' % (D1, D2, p)
        '0.5, 1.0, 1.0'
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

        self.D1_ = self._statistic(self.x1, y)
        self.D2_ = self._statistic(self.x2, y)
        self.Da_ = self.D1_ - self.D2_

        # use all cores to create function that parallelizes over number of reps
        mapwrapper = MapWrapper(workers)
        null_dist = np.array(list(mapwrapper(self._perm_stat, range(reps))))
        self.null_dist = null_dist

        self.diffNull = []

        for i in range(0, reps - 1):
            for j in range(i + 1, reps):
                self.diffNull.append(self.null_dist[i][0] - self.null_dist[j][1])
                self.diffNull.append(self.null_dist[j][1] - self.null_dist[i][0])

        self.diffNull = np.asarray(self.diffNull)

        if alt == "greater":
            pvalue = (self.diffNull > self.Da_).mean()
        elif alt == "less":
            pvalue = (self.diffNull < self.Da_).mean()
        elif alt == "neq":
            pvalue = (abs(self.diffNull) > abs(self.Da_)).mean()
        else:
            msg = "You have not entered a valid alternative."
            raise ValueError(msg)

        if pvalue == 0:
            pvalue = 1 / reps

        self.pvalue_ = pvalue

        return self.D1_, self.D2_, self.pvalue_

    def getConvexComb(self, x):
        N, _ = x.shape

        q1 = np.random.choice(N, N)
        q2 = np.random.choice(N, N)
        lamda = np.random.uniform(size=N)

        return (lamda * (x[q1]).T + (1 - lamda) * (x[q2]).T).T

    def _perm_stat(self, index):
        permx1 = self.getConvexComb(self.x1)
        permx2 = self.getConvexComb(self.x2)
        permy = self.y

        perm_stat1 = self._statistic(permx1, permy)
        perm_stat2 = self._statistic(permx2, permy)

        return perm_stat1, perm_stat2
