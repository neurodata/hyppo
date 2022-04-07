import numpy as np
from hyppo.ksample._utils import _CheckInputs
from hyppo.ksample.base import KSampleTest, KSampleTestOutput
from scipy.stats import f
from sklearn.metrics import pairwise_distances
from scipy.stats import anderson_ksamp

class FastHHG(KSampleTest):
    r"""
    Fast HHG 2-Sample Test
    """
    def __init__(self, compute_distance="euclidean", **kwargs):
       self.compute_distance = compute_distance
       KSampleTest.__init__(self, compute_distance=compute_distance, **kwargs)
       
    def statistic(self, y1, y2):
        #if y1 is data and y2 is K onehoc-encoding (as done in power test)
        if y2.shape[1] == 1:
            a = np.hstack((y1,y2))
            a = a[a[:, -1].argsort()]
            a = np.split(a[:,:-1], np.unique(a[:, -1], return_index=True)[1][1:])
            y1 = a[0]
            y2 = a[1]
        
        disty1, disty2 = self._centerpoint_dist(y1, y2, self.compute_distance, 1)
        stat, crit, pvalue = anderson_ksamp([disty1, disty2])
        self.stat=stat
        return stat

    def test(self, x, y):
        """
        x,y : ndarray of float
            Input data matrices. ``x`` and ``y`` must have the same number of
            dimensions. That is, the shapes must be ``(n, p)`` and ``(m, p)`` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions.
        """
        distx, disty = self._centerpoint_dist(x, y, self.compute_distance, 1)
        #Simple AD Implementation
        stat, crit, pvalue = anderson_ksamp([distx,disty])
        return stat, crit, pvalue
        
    def _centerpoint_dist(self, y1, y2, metric, workers=1, **kwargs):
        #CM of one group as center point
        zy = np.mean(y1, axis=0)
        zy = np.array(zy).reshape(1,-1)
        yin1 = np.concatenate((zy, y1))
        yin2 = np.concatenate((zy, y2))
        if callable(metric):
            disty1 = metric(yin1, **kwargs)
            disty1 = metric(yin2, **kwargs)
        else:
            disty1 = pairwise_distances(yin1, metric=metric, n_jobs=workers, **kwargs)
            disty2 = pairwise_distances(yin2, metric=metric, n_jobs=workers, **kwargs)
        disty1 = np.delete(disty1[0],0)
        disty2 = np.delete(disty2[0],0)
        disty1 = disty1.reshape(-1)
        disty2 = disty2.reshape(-1)
        return disty1, disty2