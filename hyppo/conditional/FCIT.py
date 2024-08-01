import time
import joblib

import numpy as np
from scipy.stats import ttest_1samp
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor

from .base import ConditionalIndependenceTest, ConditionalIndependenceTestOutput


class FCIT(ConditionalIndependenceTest):
    r"""
    Fast Conditional Independence test statistic and p-value

    The Fast Conditional Independence Test is a non-parametric
    conditional independence test :footcite:p:`chalupka2018FastConditionalIndependence`.

    Parameters
    ----------
    model: Sklearn regressor
        Regressor used to predict input data :math:`Y`
    cv_grid: dict
       Dictionary of parameters to cross-validate over when training regressor.
    num_perm: int
        Number of data permutations to estimate the p-value from marginal stats.
    prop_test: float
        Proportion of data to evaluate test stat on.
    discrete: tuple of string
        Whether :math:`X` or :math:`Y` are discrete
    Notes
    -----
    The motivation for the test rests on the assumption that if :math:`X \not\!\perp\!\!\!\perp Y \mid Z`,
    then :math:`Y` should be more accurately predicted by using both
    :math:`X` and :math:`Z` as covariates as opposed to only using
    :math:`Z` as a covariate. Likewise, if :math:`X \perp \!\!\! \perp Y \mid Z`,
    then :math:`Y` should be predicted just as accurately by solely
    using :math:`X` or soley using :math:`Z` :footcite:p:`chalupka2018FastConditionalIndependence`.
    Thus, the test works by using a regressor (the default is decision tree) to
    to predict input :math:`Y` using both :math:`X` and :math:`Z` and using
    only :math:`Z` :footcite:p:`chalupka2018FastConditionalIndependence`. Then,
    accuracy of both predictions are measured via mean-squared error (MSE).
    :math:`X \perp \!\!\! \perp Y \mid Z` if and only if MSE of the algorithm
    using both :math:`X` and :math:`Z` is not smaller than the MSE of the
    algorithm trained using only :math:`Z` :footcite:p:`chalupka2018FastConditionalIndependence`.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        model=DecisionTreeRegressor(),
        cv_grid={"min_samples_split": [2, 8, 64, 512, 1e-2, 0.2, 0.4]},
        num_perm=8,
        prop_test=0.1,
        discrete=(False, False),
    ):

        self.model = model
        self.cv_grid = cv_grid
        self.num_perm = num_perm
        self.prop_test = prop_test
        self.discrete = discrete
        ConditionalIndependenceTest.__init__(self)

    def statistic(self, x, y, z=None):
        r"""
        Calculates the FCIT test statistic.

        Parameters
        ----------
        x,y,z : ndarray of float
            Input data matrices.

        Returns
        -------
        stat : float
            The computed FCIT test statistic.
        two_sided: float
            Two-sided p-value associated with test statistic
        """

        n_samples = x.shape[0]
        n_test = int(n_samples * self.prop_test)

        data_permutations = [
            np.random.permutation(x.shape[0]) for i in range(self.num_perm)
        ]

        clf = _cross_val(x, y, z, self.cv_grid, self.model, prop_test=self.prop_test)
        datadict = {
            "x": x,
            "y": y,
            "z": z,
            "data_permutation": data_permutations,
            "n_test": n_test,
            "reshuffle": False,
            "clf": clf,
        }
        d1_stats = np.array(
            joblib.Parallel(n_jobs=-1, max_nbytes=100e6)(
                joblib.delayed(_obtain_error)((datadict, i))
                for i in range(self.num_perm)
            )
        )

        if z.shape[1] == 0:
            x_indep_y = x[np.random.permutation(n_samples)]
        else:
            x_indep_y = np.empty([x.shape[0], 0])

        clf = _cross_val(
            x_indep_y, y, z, self.cv_grid, self.model, prop_test=self.prop_test
        )

        datadict["reshuffle"] = True
        datadict["x"] = x_indep_y
        d0_stats = np.array(
            joblib.Parallel(n_jobs=-1, max_nbytes=100e6)(
                joblib.delayed(_obtain_error)((datadict, i))
                for i in range(self.num_perm)
            )
        )

        stat, two_sided = ttest_1samp(d0_stats / d1_stats, 1)

        return stat, two_sided

    def test(self, x, y, z=None):
        r"""
        Calculates the FCIT test statistic and p-value.

        Parameters
        ----------
        x,y,z : ndarray of float
            Input data matrices.

        Returns
        -------
        stat : float
            The computed FCIT statistic.
        pvalue : float
            The computed FCIT p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.conditional import FCIT
        >>> from sklearn.tree import DecisionTreeRegressor
        >>> np.random.seed(1234)
        >>> dim = 2
        >>> n = 100000
        >>> z1 = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=(n))
        >>> A1 = np.random.normal(loc=0, scale=1, size=dim * dim).reshape(dim, dim)
        >>> B1 = np.random.normal(loc=0, scale=1, size=dim * dim).reshape(dim, dim)
        >>> x1 = (A1 @ z1.T + np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=(n)).T)
        >>> y1 = (B1 @ z1.T + np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=(n)).T)
        >>> model = DecisionTreeRegressor()
        >>> cv_grid = {"min_samples_split": [2, 8, 64, 512, 1e-2, 0.2, 0.4]}
        >>> stat, pvalue = FCIT(model=model, cv_grid=cv_grid).test(x1.T, y1.T, z1)
        >>> '%.2f, %.3f' % (stat, pvalue)
        '-3.59, 0.995'
        """

        n_samples = x.shape[0]

        if z is None:
            z = np.empty([n_samples, 0])

        if self.discrete[0] and not self.discrete[1]:
            x, y = y, x
        elif x.shape[1] < y.shape[1]:
            x, y = y, x

        y = StandardScaler().fit_transform(y)

        stat, two_sided = self.statistic(x, y, z)

        if stat < 0:
            pvalue = 1 - two_sided / 2
        else:
            pvalue = two_sided / 2

        return ConditionalIndependenceTestOutput(stat, pvalue)


def _cross_val(x, y, z, cv_grid, model, prop_test):
    """
    Choose the regression hyperparameters by
    cross-validation.
    """

    splitter = ShuffleSplit(n_splits=3, test_size=prop_test)
    cv = GridSearchCV(estimator=model, cv=splitter, param_grid=cv_grid, n_jobs=-1)
    cv.fit(_interleave(x, z), y)

    return type(model)(**cv.best_params_)


def _interleave(x, z, seed=None):
    """Interleave x and z dimension-wise."""
    state = np.random.get_state()
    np.random.seed(seed or int(time.time()))
    total_ids = np.random.permutation(x.shape[1] + z.shape[1])
    np.random.set_state(state)
    out = np.zeros([x.shape[0], x.shape[1] + z.shape[1]])
    out[:, total_ids[: x.shape[1]]] = x
    out[:, total_ids[x.shape[1] :]] = z
    return out


def _obtain_error(data_and_i):
    """
    A function used for multithreaded computation of the fcit test statistic.
    Calculates MSE error for both trained regressors.
    """
    data, i = data_and_i
    x = data["x"]
    y = data["y"]
    z = data["z"]
    if data["reshuffle"]:
        perm_ids = np.random.permutation(x.shape[0])
    else:
        perm_ids = np.arange(x.shape[0])
    data_permutation = data["data_permutation"][i]
    n_test = data["n_test"]
    clf = data["clf"]

    x_z = _interleave(x[perm_ids], z, seed=i)

    clf.fit(x_z[data_permutation][n_test:], y[data_permutation][n_test:])
    return mse(
        y[data_permutation][:n_test], clf.predict(x_z[data_permutation][:n_test])
    )
