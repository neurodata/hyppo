from typing import NamedTuple

import numpy as np
from honest_forests import HonestForestClassifier  # change this to scikit-tree later
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ._utils import _CheckInputs
from .base import IndependenceTest


class MIRFTestOutput(NamedTuple):
    stat: float
    pvalue: float
    # mirf_dict: dict


class MIRF(IndependenceTest):
    r"""
    Independence test using mutual information as the test statistic.

    Parameters
    ----------
    n_estimators : int, default: 100
        The number of trees in the forest.
    honest_fraction : float, default=0.5
        Fraction of training samples used for estimates in the trees. The
        remaining samples will be used to learn the tree structure. A larger
        fraction creates shallower trees with lower variance estimates.
    honest_prior : {"ignore", "uniform", "empirical"}, default="empirical"
        Method for dealing with empty leaves during evaluation of a test
        sample. If "ignore", the tree is ignored. If "uniform", the prior tree
        posterior is 1/(number of classes). If "empirical", the prior tree
        posterior is the relative class frequency in the voting subsample.
        If all trees are ignored, the empirical estimate is returned.
    **kwargs
        Additional arguments used for the forest (see
        :class:`honest_forests.HonestForestClassifier`)
    """

    def __init__(
        self, n_estimators=500, honest_fraction=0.5, honest_prior="empirical", **kwargs
    ):
        self.clf = HonestForestClassifier(
            n_estimators=n_estimators,
            honest_fraction=honest_fraction,
            honest_prior=honest_prior,
            **kwargs,
        )
        IndependenceTest.__init__(self)

    def statistic(self, x, y):
        r"""
        Helper function that calculates the MI test statistic.

        Parameters
        ----------
        x,y : ndarray of float
            Input data matrices. ``x`` and ``y`` must have the same number of
            samples. That is, the shapes must be ``(n, p)`` and ``(n, 1)`` where
            `n` is the number of samples and `p` is the number of
            dimensions.

        Returns
        -------
        stat : float
            The computed MI statistic.
        """
        self.clf.fit(x, y.ravel())
        H_YX = np.mean(entropy(self.clf.predict_proba(x), base=np.exp(1), axis=1))
        _, counts = np.unique(y, return_counts=True)
        H_Y = entropy(counts, base=np.exp(1))
        stat = max(H_Y - H_YX, 0)

        self.stat = stat

        return stat

    def test(self, x, y, reps=1000, workers=1, random_state=None):
        r"""
        Calculates the MI test statistic and p-value.

        Parameters
        ----------
        x,y : ndarray of float
            Input data matrices. ``x`` and ``y`` must have the same number of
            samples. That is, the shapes must be ``(n, p)`` and ``(n, 1)`` where
            `n` is the number of samples and `p` is the number of
            dimensions.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default: 1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.

        Returns
        -------
        stat : float
            The computed MI statistic.
        pvalue : float
            The computed MI p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.independence import MI
        >>> x = np.arange(100)
        >>> y = x
        >>> '%.1f, %.2f' % MI().test(x, y)[:1] # doctest: +SKIP
        '1.0, 0.001'
        """
        check_input = _CheckInputs(x, y, reps=reps)
        x, y = check_input()

        stat, pvalue = super(MIRF, self).test(
            x, y, reps, workers, is_distsim=False, random_state=random_state
        )
        # mirf_dict = {}

        return MIRFTestOutput(stat, pvalue)  # , mirf_dict)


class MIRF_AUC(IndependenceTest):
    def __init__(
        self,
        n_estimators=500,
        honest_fraction=0.5,
        honest_prior="empirical",
        limit=0.05,
        **kwargs,
    ):
        self.clf = HonestForestClassifier(
            n_estimators=n_estimators,
            honest_fraction=honest_fraction,
            honest_prior=honest_prior,
            **kwargs,
        )
        self.limit = limit
        IndependenceTest.__init__(self)

    def statistic(self, x, y):
        stats = []

        # 5-fold cross validation for truncated AUROC
        cv = StratifiedKFold()
        for fold, (train, test) in enumerate(cv.split(x, y)):
            self.clf.fit(x[train], y[train].ravel())
            y_pred = self.clf.predict_proba(x[test])[:, 1]
            stats.append(roc_auc_score(y[test], y_pred, max_fpr=self.limit))

        self.stat = np.mean(stats)
        return self.stat

    def test(self, x, y, reps=1000, workers=1, random_state=None):
        check_input = _CheckInputs(x, y, reps=reps)
        x, y = check_input()

        stat, pvalue = super(MIRF, self).test(
            x, y, reps, workers, is_distsim=False, random_state=random_state
        )

        return MIRFTestOutput(stat, pvalue)
