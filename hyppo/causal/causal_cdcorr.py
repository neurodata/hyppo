from ..conditional import ConditionalDcorr
from .propensity_model import GeneralisedPropensityModel
from .base import ConditionalDiscrepancyTest, ConditionalDiscrepancyTestOutput
from ..tools.common import compute_dist, _check_distmat
from ._utils import _CleanInputsConditionalDiscrepancy
import pandas as pd
import numpy as np

class CausalCDcorr(ConditionalDiscrepancyTest):
    """
    Causal Conditional Distance Correlation test statistic and p-value.

    Causal CDcorr is a method for testing for causal effects in multivariate data
    across groups, given a third (conditioning) matrix. Under standard causal assumptions,
    including consistency, positivity on the conditioning variables, conditional 
    ignorability on the conditioning variables, and no interference, subsequent
    conclusions merit causal interpretations. This approach levels Vertex Matching to
    synthetically pre-process the data and ensure empirical positivity on the 
    covariates  :footcite:p:`Lopez2017Aug`, followed by conditional K-sample testing 
    using the conditional distance correlation  :footcite:p:`wang2015conditional`.

    Parameters
    ----------
    compute_distance : str, callable, or None, default: "euclidean"
        A function that computes the distance among the samples within the
        outcome matrix.
        Valid strings for ``compute_distance`` are, as defined in
        :func:`sklearn.metrics.pairwise_distances`,

            - From scikit-learn: [``"euclidean"``, ``"cityblock"``, ``"cosine"``,
              ``"l1"``, ``"l2"``, ``"manhattan"``] See the documentation for
              :mod:`scipy.spatial.distance` for details
              on these metrics.
            - From scipy.spatial.distance: [``"braycurtis"``, ``"canberra"``,
              ``"chebyshev"``, ``"correlation"``, ``"dice"``, ``"hamming"``,
              ``"jaccard"``, ``"kulsinski"``, ``"mahalanobis"``, ``"minkowski"``,
              ``"rogerstanimoto"``, ``"russellrao"``, ``"seuclidean"``,
              ``"sokalmichener"``, ``"sokalsneath"``, ``"sqeuclidean"``,
              ``"yule"``] See the documentation for :mod:`scipy.spatial.distance` for
              details on these metrics.

        Set to ``None`` or ``"precomputed"`` if ``Ys`` is already a distance matrix. 
        To call a custom function, either create the distance matrix
        before-hand or create a function of the form ``metric(Ys, **kwargs)``
        where ``Ys`` is the data matrix for which pairwise distances are
        calculated and ``**kwargs`` are extra arguements to send to your custom
        function.
    use_cov : bool, default: True
        If `True`, then the statistic will compute the covariance rather than the
        correlation.
    bandwith : str, scalar, 1d-array
        The method used to calculate the bandwidth used for kernel density estimate of
        the conditional matrix. This can be ‘scott’, ‘silverman’, a scalar constant or a
        1d-array with length ``r`` which is the dimensions of the conditional matrix.
        If None (default), ‘scott’ is used.
    **kwargs
        Arbitrary keyword arguments for ``compute_distance``.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self, 
        compute_distance="euclidean",
        use_cov=True, 
        bandwidth=None,
        **kwargs,
    ):
        self.use_cov = use_cov
        self.compute_distance = compute_distance
        self.bandwidth = bandwidth
        self.kwargs = kwargs

        # initialize cdcorr, but skip computing distances and compute
        # internally here to properly handle categorical variables
        self.cdcorr = ConditionalDcorr(self.use_cov, self.bandwidth, compute_distance=None)

        ConditionalDiscrepancyTest.__init__(self, **kwargs)

    def __repr__(self):
        return "CausalCDcorr"

    def _preprocess(self, Ys, Ts, Xs, prop_form_rhs=None, ddx=False, niter=100, retain_ratio=0.05):
        """
        An internal function to pre-process a dataset before assessing conditional discrepancies.
        """
        # initialize VM instance and clean outcomes, treatments, and covariates
        vm = VectorMatch(self.retain_ratio)
        try:
            vm._clean(Ts, Xs, prop_form_rhs=prop_form_rhs)
            clean_outcomes = _CleanInputsConditionalDiscrepancy(Ys, vm.cleaned_inputs.Ts_factor, vm.cleaned_inputs.Xs_df)
        except Exception as e:
            exc_type = type(e)
            new_message = f"Error cleaning `Ys', `Ts', or `Xs'. Error: {str(e)}"
            raise exc_type(new_message) from e
        # obtain balancing IDs
        try:
            self.balanced_ids = vm._fit(ddx=ddx, niter=niter, retain_ratio=retain_ratio)
        except Exception as e:
            exc_type = type(e)
            new_message = f"Error with vector matching. Error: {str(e)}"
            raise exc_type(new_message) from e
    
        # handle distances and kernel matrices
        if not self.is_distance:
            try:
                Ys_df_tilde = clean_outcomes.Ys_df.iloc[self.balanced_ids, :]
                categorical_cols = Ys_df_tilde.select_dtypes(include=['category']).columns.tolist()
                Y_dummies = pd.get_dummies(clean_outcomes, columns=categorical_cols, drop_first=True)
                DYtilde = compute_dist(Y_dummies, metric=self.compute_distance, **self.kwargs)
            except Exception as e:
                exc_type = type(e)
                new_message = f"Error converting `Ys' to a distance matrix. Error: {str(e)}"
                raise exc_type(new_message) from e
            self.is_distance = True
        else:
            try:
                DY = _check_dist(np.array(clean_outcomes.Ys_df))
                DYtilde = DY[self.balanced_ids, :][:, self.balanced_ids]
            except Exception as e:
                raise TypeError(f"`Ys' is not a valid distance matrix. Error: {e}")
        try:
            DTtilde = compute_dist(vm.Ts_factor, metric=self.compute_distance, **self.kwargs)
        except Exception as e:
            exc_type = type(e)
            new_message = f"Error converting `Ts' to a distance matrix. Error: {str(e)}"
            raise exc_type(new_message) from e

        try:
            KXtilde = self.cdcorr._compute_kde(vm.Xs_df.iloc[self.balanced_ids,:])
        except Exception as e:
            exc_type = type(e)
            new_message = f"Error converting `Xs' to a kernel similarity matrix. Error: {str(e)}"
            raise exc_type(new_message) from e

        # returns distance matrix for outcomes, distance matrix for treatments,
        # and kernel matrix for the covariates
        return DYtilde, DTtilde, KXtilde
    
    def statistic(self, Ys, Ts, Xs, prop_form_rhs=None, ddx=False, niter=100, retain_ratio=0.05):
        """
        Computes the Causal CDcorr/CDcov statistic.

        Parameters
        ----------
        Ys : pandas DataFrame or array-like
            Outcome matrix, as an array. Should have a shape ``(n, r)``, where ``n`` is the number of samples, and ``r`` is the number of outcome dimensions. Alternatively, if ``compute_distance'' is None or `"precomputed"', an ``(n, n)`` distance matrix for the outcomes between samples.
        Ts : array-like
            Treatment assignment vector, where entries are one of K-possible treatment indicators. Should have a shape castable to an ``n'' vector, where ``n'' is the number of samples.
        Xs : pandas DataFrame or array-like
            Covariates/features matrix, as an array. Should have a shape ``(n, r)``, where ``n`` is the number of samples, and ``r`` is the number of covariates. 
        prop_form_rhs : str, or None, default: None
                - Set to `None` to default to a propensity model which includes a single regressor for each column of the covariate matrix.
                - This option is only functional if `outcome_only' is set to False.
        ddx : bool, optional, default: False
            Whether to print diagnostic debugging information during propensity model fitting.
        niter : int, optional, default: 100
            The number of iterations for the multinomial logit propensity model fitting.

        Returns
        -------
        stat : float
            The computed Causal CDcorr/CDcov statistic.
        """
        DYtilde, DTtilde, KXtilde = self._preprocess(Ys, Ts, Xs, prop_form_rhs=prop_form_rhs, ddx=ddx, niter=niter, retain_ratio=retain_ratio)
        return self.cdcorr.statistic(DYtilde, DTtilde, KXtilde)

    def test(
        self, 
        Ys,
        Ts, 
        Xs,
        prop_form_rhs=None,
        reps=1000,
        workers=1,
        random_state=None,
        ddx=False,
        niter=100,
        retain_ratio=0.05,
    ):
        """
        Computes the Causal CDcov/CDcorr statistic.

        Parameters
        ----------
        Ys : pandas DataFrame or array-like
            Outcome matrix, as an array. Should have a shape ``(n, r)``, where ``n`` is the number of samples, and ``r`` is the number of outcome dimensions. Alternatively, if ``compute_distance'' is None or `"precomputed"', an ``(n, n)`` distance matrix for the outcomes between samples.
        Ts : array-like
            Treatment assignment vector, where entries are one of K-possible treatment indicators. Should have a shape castable to an ``n'' vector, where ``n'' is the number of samples.
        Xs : pandas DataFrame or array-like
            Covariates/features matrix, as an array. Should have a shape ``(n, r)``, where ``n`` is the number of samples, and ``r`` is the number of covariates. 
        prop_form_rhs : str, or None, default: None
                - Set to `None` to default to a propensity model which includes a single regressor for each column of the covariate matrix.
                - This option is only functional if `outcome_only' is set to False.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default: 1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.
        random_state : int, default: None
            The random_state for permutation testing to be fixed for
            reproducibility.
        ddx : bool, optional, default: False
            Whether to print diagnostic debugging information during propensity model fitting.
        niter : int, optional, default: 100
            The number of iterations for the multinomial logit propensity model fitting.

        Returns
        -------
        stat : float
            The computed Causal CDcorr/CDcov statistic.
        pvalue : float
            The computed Causal CDcorr/CDcov p-value.
        """
        DYtilde, DTtilde, KXtilde = self._preprocess(Ys, Ts, Xs, prop_form_rhs=prop_form_rhs, ddx=ddx, niter=niter, retain_ratio=retain_ratio)
        return self.cdcorr.test(DYtilde, DTtilde, KXtilde, reps=reps, workers=workers, random_state=random_state)