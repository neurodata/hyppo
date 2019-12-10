from sklearn.metrics import euclidean_distances
from ._utils import _CheckInputs
import numpy as np
import random
from .base import DiscriminabilityTest
from scipy._lib._util import MapWrapper



class DiscrimOneSample(DiscriminabilityTest):
    r"""
     A class that performs a one-sample test for whether the discriminability 
     differs from random chance, as described in [#1Dscr]_.
     
     Discriminability index is a masure of whether a data acquisition and 
     preprocessing pipeline is more discriminable among different subjects.
     The key insight is that each measurement of the same item should be more 
     similar to other measurements of that item, as compared to measurements 
     of any other item. 
    
    See Also
    --------
    DiscrimTwoSample : Two sample test for comparing the discriminability of two data
    
    Notes
    -----
    With :math:`D_X` as the sample discriminability of :math:`X`, one sample test verifies 
    whether
    
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

    def __init__(self):
        DiscriminabilityTest.__init__(self)

    def _statistic(self):
        """
        """
        

    def test(self, X, Y, is_dist = False, remove_isolates=True, reps=1000, workers=-1):
        r"""
        Calculates the test statistic and p-value for Discriminability one sample test.

        Parameters
        ----------
        X : ndarray

            * An `(n, d)` data matrix with `n` samples in `d` dimensions,
              if flag `(is\_dist = Flase)`

            * An `(n, n)` distance matrix, if flag `(is\_dist = True)`
            
        Y : ndarray
            a vector containing the sample ids for our :math:`n` samples.
            
        is_dist : Boolean, optional (default: False)
            Whether `X` is a distance matrix or not.
        remove_isolates : Boolean, optional (default: True)
            whether remove the samples with single instance or not.
        reps : int, optional (default: 1000)
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, optional (default: -1)
            The number of cores to parallelize the p-value computation over.
            Supply -1 to use all cores available to the Process.

        Returns
        -------
        stat : float
            The computed Discriminability statistic.
        pvalue : float
            The computed one sample test p-value.
        """

        check_input = _CheckInputs(X, Y, reps=reps)
        X, Y = check_input()

        _, counts = np.unique(Y, return_counts=True)

        if (counts != 1).sum() <= 1:
            msg = "You have passed a vector containing only a single unique sample id."
            raise ValueError(msg)

        self.X = X
        self.Y = Y
        
        stat = super(DiscrimOneSample,self)._statistic(self.X, self.Y, is_dist, remove_isolates)
        self.stat_ = stat

        # use all cores to create function that parallelizes over number of reps
        mapwrapper = MapWrapper(workers)
        null_dist = np.array(list(mapwrapper(self._perm_stat, range(reps))))
        self.null_dist = null_dist
        
        # calculate p-value and significant permutation map through list
        pvalue = ((null_dist >= stat).sum() + 1) / (reps + 1)

        # correct for a p-value of 0. This is because, with bootstrapping
        # permutations, a p-value of 0 is incorrect
        if pvalue == 0:
            pvalue = 1 / reps
        self.pvalue_ = pvalue

        return stat, pvalue

    def _perm_stat(self, index):
        permx = self.X
        permy = np.random.permutation(self.Y)

        perm_stat = super(DiscrimOneSample,self)._statistic(permx, permy)

        return perm_stat