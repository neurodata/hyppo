from ._utils import _CheckInputs
import numpy as np
import random
from .base import DiscriminabilityTest
from scipy._lib._util import MapWrapper


class DiscrimOneSample(DiscriminabilityTest):
    r"""
     A class that performs a one-sample test for discriminability.
     
     Discriminability index is a measure of whether a data acquisition and 
     preprocessing pipeline is more discriminable among different subjects.
     The key insight is that each measurement of the same item should be more 
     similar to other measurements of that item, as compared to measurements 
     of any other item. One sample test measures whether the discriminability 
     for a dataset differs from random chance. More details can be described 
     in [#1Dscr]_.
    
    Parameters
    ---------- 
    is_dist : bool, optional (default: False)
        whether `x` is a distance matrix or not.
    remove_isolates : bool, optional (default: True)
        whether to remove the measurements with single instance or not.

    See Also
    --------
    DiscrimTwoSample : Two sample test for comparing the discriminability of two data
    
    Notes
    -----
    With :math:`D_X` as the sample discriminability of :math:`X`, 
    one sample test verifies whether
    
     .. math::

         H_0: D_X = D_0

    and

     .. math::

         H_A: D_X > D_0 
    
    where :math:`D_0` is the discriminability that would be observed by random chance.

    References
    ----------
    .. [#1Dscr] Eric W. Bridgeford, et al. "Optimal Decisions for Reference 
                Pipelines and Datasets: Applications in Connectomics." Bioarxiv (2019).
    """

    def __init__(self, is_dist=False, remove_isolates=True):
        # set is_distance to true if compute_distance is None
        self.is_distance = is_dist
        self.remove_isolates = remove_isolates
        DiscriminabilityTest.__init__(self)

    def _statistic(self, x, y):
        """
        Helper function that calculates the discriminability test statistics.
        """
        stat = super(DiscrimOneSample, self)._statistic(x, y)

        return stat

    def test(self, x, y, reps=1000, workers=-1):
        r"""
        Calculates the test statistic and p-value for Discriminability one sample test.

        Parameters
        ----------
        x: ndarray
            An `(n, d)` data matrix with `n` samples in `d` dimensions,
            if flag is_dist = Flase and an `(n, n)` distance matrix,
            if flag is_dist = True
            
        y : ndarray
            a vector containing the sample ids for our :math:`n` samples.
        reps : int, optional (default: 1000)
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, optional (default: -1)
            The number of cores to parallelize the p-value computation over.
            Supply -1 to use all cores available to the Process.

        Returns
        -------
        stat : float
            The computed discriminability statistic.
        pvalue : float
            The computed one sample test p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from mgc.discrim import DiscrimOneSample
        >>> x = np.concatenate((np.zeros((50,2)) ,np.ones((50,2))), axis=0)
        >>> y = np.concatenate((np.zeros(50),np.ones(50)), axis= 0)
        >>> stat, p = DiscrimOneSample().test(x,y)
        >>> '%.1f, %.2f' % (stat, p)
        '1.0, 0.00'
        """

        check_input = _CheckInputs(
            [x],
            y,
            reps=reps,
            is_dist=self.is_distance,
            remove_isolates=self.remove_isolates,
        )
        x, y = check_input()

        self.x = np.asarray(x[0])
        self.y = y

        stat = self._statistic(self.x, self.y)
        self.stat = stat

        # use all cores to create function that parallelizes over number of reps
        mapwrapper = MapWrapper(workers)
        null_dist = np.array(list(mapwrapper(self._perm_stat, range(reps))))
        self.null_dist = null_dist

        # calculate p-value and significant permutation map through list
        pvalue = ((null_dist >= stat).sum()) / reps

        # correct for a p-value of 0. This is because, with bootstrapping
        # permutations, a p-value of 0 is incorrect
        if pvalue == 0:
            pvalue = 1 / reps

        self.pvalue_ = pvalue

        return stat, pvalue

    def _perm_stat(self, index):
        permy = np.random.permutation(self.y)

        perm_stat = self._statistic(self.x, permy)

        return perm_stat
