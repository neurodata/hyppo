from sklearn.metrics import euclidean_distances
from ._utils import _CheckInputs
import numpy as np
import random
from .base import DiscriminabilityTest
from scipy._lib._util import MapWrapper



class oneSample(DiscriminabilityTest):
    r"""
     A class that performs a one-sample test for whether the discriminability differs from random chance,
     as described in [1]. With :math:`D_X` as the sample discriminability of :math:`X`, it tests whether:

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
        self.stat = None
        self.pvalue = None
        DiscriminabilityTest.__init__(self)



    def test(self, X, Y, isDist = False, remove_isolates=True, reps=1000, workers=-1):
        r"""
        Calculates the test statistic and p-value for Discriminability one sample test.

        Parameters
        ----------
        X : ndarray

            * An :math:`n \times d` dimensional data matrix with :math:`n` samples in :math:`d` dimensions, if flag :math:`(isDist = Flase)` 

            * An :math:`n \times n` dimensional distance matrix if :math:`X` is a distance matrix. Use flag :math:`(isDist = True)`
            
        isDist : Boolean, optional (default: False)
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

        check_input = _CheckInputs(X, Y, reps = reps)
        X, Y = check_input()

        _, counts = np.unique(Y, return_counts=True)

        if (counts != 1).sum() <= 1:
            msg = "You have passed a vector containing only a single unique sample id."
            raise ValueError(msg)


        self.X = X
        self.Y = Y
        
        stat = super(oneSample,self)._statistic(self.X, self.Y,isDist, remove_isolates, return_rdfs=False)
        self.stat = stat

        # use all cores to create function that parallelizes over number of reps
        mapwrapper = MapWrapper(workers)
        null_dist = np.array(list(mapwrapper(self._perm_stat, range(reps))))
        self.null_dist = null_dist

        # calculate p-value and significant permutation map through list
        pvalue = (null_dist >= stat).sum() / reps

        # correct for a p-value of 0. This is because, with bootstrapping
        # permutations, a p-value of 0 is incorrect
        if pvalue == 0:
            pvalue = 1 / reps
        self.pvalue = pvalue

        return stat, pvalue




    def _perm_stat(self, index):
        permx = self.X
        permy = np.random.permutation(self.Y)

        perm_stat = super(oneSample,self)._statistic(permx, permy)

        return perm_stat