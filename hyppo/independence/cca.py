import numpy as np
from hyppo.independence._utils import _CheckInputs
from hyppo.independence.base import IndependenceTest


class CCA(IndependenceTest):
    """
    Canonical Correlation Analysis (CCA) test statistic and p-value.
    """

    def __init__(self):
        super().__init__()

    def statistic(self, x, y):
        """
        Calculates the CCA test statistic.

        Parameters
        ----------
        x, y : ndarray of float
            Input data matrices with shapes (n_samples, n_features_x) and
            (n_samples, n_features_y), respectively. Monodimensional arrays are
            allowed and will be reshaped to (n_samples, 1).

        Returns
        -------
        stat : float
            The largest canonical correlation.
        """
        centx = x - np.mean(x, axis=0)
        centy = y - np.mean(y, axis=0)

        # calculate covariance and variances for inputs
        covar = centx.T @ centy
        varx = centx.T @ centx
        vary = centy.T @ centy

        # if 1-d, don't calculate the svd
        if varx.size == 1 or vary.size == 1 or covar.size == 1:
            covar = np.sum(np.abs(covar))
            stat = covar / np.sqrt(np.sum(np.abs(varx)) * np.sum(np.abs(vary)))
        else:
            covar = np.sum(np.abs(np.linalg.svd(covar, 1)[1]))
            stat = covar / np.sqrt(
                np.sum(np.abs(np.linalg.svd(varx, 1)[1]))
                * np.sum(np.abs(np.linalg.svd(vary, 1)[1]))
            )
        self.stat = stat

        return stat



    def test(self, x, y, reps=1000, workers=1, random_state=None):
        """
        Calculates the CCA test statistic and p-value.
        """
        check_input = _CheckInputs(x, y, reps=reps)
        x, y = check_input()

        # Use default permutation test
        return super(CCA, self).test(
            x, y, reps, workers, is_distsim=False, random_state=random_state
        )
