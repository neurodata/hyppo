from abc import ABC, abstractmethod
import warnings
from statsmodels.discrete.discrete_model import MNLogit
import pandas as pd
import numpy as np
import patsy
from ..tools import contains_nan, check_min_samples, check_2d_array, check_categorical, check_ndarray_or_dataframe

class _CleanInputsPM:
    """
    Cleans inputs for Propensity Model.

    Parameters
    ----------
    Ts : array-like
        Treatment assignment vector, where entries are one of K-possible treatment indicators. Should have a shape castable to an ``n'' vector, where ``n'' is the number of samples.
    Xs : pandas DataFrame or array-like
        Covariates/features matrix, as an array. Should have a shape ``(n, r)``, where ``n`` is the number of samples, and ``r`` is the number of covariates.
    prop_form_rhs : str, or None, default: None
        the right-hand side of a formula for a generalized propensity score, an extension of the concept of a propensity score to (optionally) more than two groups.
            - Set to `None` to default to a propensity model which includes a single regressor for each column of the covariate matrix.

    Attributes
    ----------
    Ts_factor: pandas series
        Cleaned treatment assignment vector, as a categorical pandas series.
    unique_treatments: list
        the unique treatment levels of `Ts_factor'.
    K: int
        the number of unique treatments.
    formula: str
        A propensity model.
    Xs_df:  pandas DataFrame
        Cleaned covariates/features matrix, as a dataframe with named columns.
    Xs_design: patsy.DesignMatrix
        Design matrix for the covariates/features.
    Ts_design: patsy.DesignMatrix
        Design matrix for the treatment variables.
    """

    def __init__(self, Ts, Xs, prop_form_rhs=None):
        self.validate_inputs(Ts, Xs, prop_form_rhs=prop_form_rhs)

    def validate_inputs(self, Ts, Xs, prop_form_rhs=None):
        # check covariates
        try:
            Xs = check_2d_array(Xs)
            contains_nan(Xs)
            Xs_df = self.check_Xs_ndarray_or_dataframe(Xs, prop_form_rhs=prop_form_rhs)
        except Exception as e:
            exc_type = type(e)
            new_message = f"Error checking `Xs'. Error: {e}"
            raise exc_type(new_message) from e
        # check treatments
        try:
            contains_nan(Ts)
            Ts_factor, unique_treatments, K = check_categorical(Ts)
        except Exception as e:
            exc_type = type(e)
            new_message = f"Error checking `Ts'. Error: {e}"
            raise exc_type(new_message) from e

        check_min_samples(Ts=Ts, Xs=Xs)
        # check that Xs is a ndarray or pandas dataframe, and
        # remove zero-variance columns if possible
        # generate a formula using user-specified values
        self.Ts_design, self.Xs_design, self.formula = self.generate_formula(
            Xs_df, Ts_factor, prop_form_rhs
        )

        self.Xs_df = Xs_df
        self.Ts_factor = Ts_factor
        self.unique_treatments = unique_treatments
        self.K = K

    def check_Xs_ndarray_or_dataframe(self, Xs, prop_form_rhs=None):
        """Ensure that Xs is a pandas dataframe, or can be cast to one."""
        # Ensure Xs is a DataFrame
        if not isinstance(Xs, pd.DataFrame):
            if prop_form_rhs is not None:
                raise TypeError(
                    "Specified a propensity formula `prop_form_rhs' upon initialization, but `Xs' covariate matrix is not a pandas dataframe."
                )
            Xs = check_ndarray_or_dataframe(Xs, "X")
        return Xs

    def generate_formula(self, Xs, Ts, prop_form_rhs=None):
        """
        Check if the right-hand side of a patsy formula is valid.
        If not specified, make one.

        Parameters
        ----------
        Xs: pandas dataframe
        Ts: pandas series
        prop_form_rhs: str
        """
        try:
            # Create the right-hand side of the formula
            if prop_form_rhs is None:
                prop_form_rhs = " + ".join(Xs.columns)
            # Create the design matrix using patsy
            formula = f"Ts ~ {prop_form_rhs}"
            Ts_design, Xs_design = patsy.dmatrices(
                formula,
                pd.concat([pd.Series(Ts, name="Ts"), Xs], axis=1),
                return_type="dataframe",
            )
        except Exception as e:
            exc_type = type(e)
            new_message = f"Error generating propensity model formula: {e}"
            raise exc_type(new_message) from e
        return Ts_design, Xs_design, formula


class GeneralisedPropensityModel(ABC):
    """
    Vector matching causal data pre-processing.

    A propensity score-based algorithm for pre-processing observational data with multiple
    treatment groups :footcite:p:`Lopez2017Aug`. The algorithm works as follows:

    1. Estimates generalized propensity scores using multinomial logistic regression
    2. For each treatment group `k`, identifies the range of propensity scores
    3. Determines the overlapping region of propensity scores across all groups
    4. Retains only observations that fall within the common support region

    This effectively removes observations without comparable matches in other treatment
    groups, improving covariate balance for downstream causal inference.

    Parameters
    ----------
    retain_ratio: float, default: 0.05
        If the number of samples retained is less than `retain_ratio`, throws a warning.

    Attributes
    ----------
    balanced_ids : array-like
        Indices of observations that are retained after vector matching.
    model : statsmodels.MNLogit
        Fitted multinomial logistic regression model.
    model_result : statsmodels.discrete.discrete_model.MNLogitResults
        Result of model fitting from MNLogit regression.
    pred_probs : pandas.DataFrame
        Predicted probabilities from the propensity model.
    Rtable : numpy.ndarray
        Table of propensity score ranges for each treatment group.
    is_fitted : bool
        Whether the model has been fitted.
    retain_ratio : float
        Minimum proportion of samples that should be retained.
    prop_form_rhs : str or None
        Formula used for propensity score estimation.
    ddx : bool
        Whether debugging information was displayed during fitting.
    cleaned_inputs : _CleanInputsPM
        Processed input data used for model fitting.
    unique_treatments : array-like
        Unique treatment values found in the data.

    Notes
    -----
    Vector matching can be viewed as a K-way extension of propensity score
    trimming, removing observations where treatment groups do not overlap
    in their covariate distributions.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self):
        """
        Initialize the VectorMatcher.
        """
        self.retain_ratio = np.nan
        self.is_fitted = False
        self.balanced_ids = None
        self.model = None
        self.pred_probs = None
        self.Rtable = None
        self.cleaned_inputs = None

    def _clean(self, Ts, Xs, prop_form_rhs=None):
        """
        An internal function to handle cleaning separately.
        """
        # Process inputs
        try:
            self.cleaned_inputs = _CleanInputsPM(Ts, Xs, prop_form_rhs=prop_form_rhs)
        except Exception as e:
            raise ValueError(f"Failed to clean inputs: {str(e)}") from e
        # set attributes after cleaning
        self.prop_form_rhs = prop_form_rhs
        self.nsamples = len(self.cleaned_inputs.Ts_factor)
        return self

    def fit(self, Ts, Xs, prop_form_rhs=None, ddx=False, niter=100, retain_ratio=0.05):
        """
        Fit the vector matching model and identify balanced observations.

        Parameters:
        -----------
        Ts : array-like
            Treatment assignment vector, where entries are one of K-possible treatment indicators. Should have a shape castable to an ``n'' vector, where ``n'' is the number of samples.
        Xs : pandas DataFrame or array-like
            Covariates/features matrix, as an array. Should have a shape ``(n, r)``, where ``n`` is the number of samples, and ``r`` is the number of covariates.
        prop_form_rhs : str, or None, default: None
            the right-hand side of a formula for a generalized propensity score, an extension of the concept of a propensity score to (optionally) more than two groups.
                - See the documentation for :mod:`statsmodels.discrete.discrete_model.MNLogit` for details on the use of formulas. If a propensity model is specified, anticipates that the covariate matrix specified for the `fit()` method will be a pandas dataframe.
                - Set to `None` to default to a propensity model which includes a single regressor for each column of the covariate matrix.
        ddx : bool, optional, default: False
            Whether to print diagnostic debugging information for model fitting.
        niter : int, optional, default: 100
            The number of iterations for the multinomial logit model.
        retain_ratio : float, optional
            Minimum proportion of samples to retain

        Returns:
        --------
        balance_ids: list of int
        - the positional indices of samples to include for subsequent analysis.
        """
        if self.is_fitted:
            raise ValueError(
                "This VectorMatch instance has already been fit. "
                "Create a new instance for a new dataset."
            )
        self._clean(Ts, Xs, prop_form_rhs=prop_form_rhs)
        return self._fit(ddx=ddx, niter=niter, retain_ratio=0.05)

    
    def _fit(self, ddx=False, niter=100, retain_ratio=0.05):
        """
        Internal method for fitting the vector matching model and identify balanced observations.

        Parameters:
        -----------
        ddx : bool, optional, default: False
            Whether to print diagnostic debugging information for model fitting.
        niter : int, optional, default: 100
            The number of iterations for the multinomial logit model.
        retain_ratio : float, optional
            Minimum proportion of samples to retain

        Returns:
        --------
        balance_ids: list of int
        - the positional indices of samples to include for subsequent analysis.
        """
        # Validate that retain_ratio is valid
        if not isinstance(retain_ratio, (float, int)):
            raise TypeError("retain_ratio must be a number (float or int)")
            
        if not (0 <= retain_ratio <= 1):
            raise ValueError("retain_ratio should be a fraction between 0 and 1.")

        # Fit the multinomial logit model
        try:
            model = MNLogit(self.cleaned_inputs.Ts_design, self.cleaned_inputs.Xs_design)
            if ddx:
                result = model.fit(disp=True, maxiter=niter)
            else:
                result = model.fit(disp=False, maxiter=niter)
        except Exception as e:
            exc_type = type(e)
            new_message = f"Failed to fit generalised propensity model: {str(e)}"
            raise exc_type(new_message) from e

        # Get predictions
        try:
            pred_probs = result.predict()
            pred_probs = pd.DataFrame(
                pred_probs, columns=[str(t) for t in range(self.cleaned_inputs.K)]
            )
        except Exception as e:
            exc_type = type(e)
            new_message = f"Failed to generate generalised propensity score predictions: {str(e)}"
            raise exc_type(new_message) from e

        # Identify overlapping regions
        try:
            Rtable = np.zeros((self.cleaned_inputs.K, 2))
            for i, t in enumerate(self.cleaned_inputs.unique_treatments):
                t_idx = np.where(self.cleaned_inputs.Ts_factor == t)[0]
                for j in range(self.cleaned_inputs.K):
                    # Min and max probabilities for treatment t when actual treatment is j
                    min_prob = np.min(pred_probs.iloc[t_idx, j])
                    max_prob = np.max(pred_probs.iloc[t_idx, j])

                    # Update Rtable with max of mins and min of maxes
                    Rtable[j, 0] = max(Rtable[j, 0], min_prob)
                    Rtable[j, 1] = min(
                        Rtable[j, 1] if Rtable[j, 1] > 0 else float("inf"),
                        max_prob,
                    )
        except Exception as e:
            exc_type = type(e)
            new_message = f"Failed to calculate overlap regions: {str(e)}"
            raise exc_type(new_message) from e

        # Identify balanced observations
        try:
            balanced_ids = []
            for i in range(self.nsamples):
                is_balanced = True
                for j in range(self.cleaned_inputs.K):
                    if not (
                        pred_probs.iloc[i, j] >= Rtable[j, 0]
                        and pred_probs.iloc[i, j] <= Rtable[j, 1]
                    ):
                        is_balanced = False
                        break
                if is_balanced:
                    balanced_ids.append(i)
        except Exception as e:
            exc_type = type(e)
            new_message = f"Failed to identify balanced observations: {str(e)}"
            raise exc_type(new_message) from e

        # Check if enough samples are retained
        if len(balanced_ids) < retain_ratio * self.nsamples:
            percentage = 100 * len(balanced_ids) / self.nsamples
            warnings.warn(
                f"Few samples retained by vector matching ({len(balanced_ids)} out of {self.nsamples}, {percentage:.1f}%)."
            )

        if len(balanced_ids) == 0:
            raise ValueError("No samples retained by vector matching.")

        self.retain_ratio = retain_ratio
        self.ddx = ddx
        self.model = model
        self.model_result = result
        self.pred_probs = pred_probs
        self.Rtable = Rtable
        self.balanced_ids = balanced_ids
        self.is_fitted = True

        return balanced_ids