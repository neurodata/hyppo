from abc import ABC, abstractmethod
import warnings
from statsmodels.discrete.discrete_model import MNLogit
import pandas as pd
import numpy as np
import patsy
from ..tools import contains_nan


class _CleanInputsVM:
    """Cleans inputs for VM."""

    def __init__(self, Ts, Xs, prop_form_rhs=None):
        self.Ts = Ts
        self.Xs = Xs
        self.prop_form_rhs = prop_form_rhs
        self.validate_inputs()

    def validate_inputs(self):
        self.check_dim_x()
        try:
            contains_nan(self.Ts)
        except Exception as e:
            raise ValueError(f"There was an issue checking `Ts' for NaNs. Error: {e}")
        try:
            contains_nan(self.Xs)
        except Exception as e:
            raise ValueError(f"There was an issue checking `Xs' for NaNs. Error: {e}")
        self._check_min_samples()
        # check that Xs is a ndarray or pandas dataframe, and
        # remove zero-variance columns if possible
        self.check_Xs_ndarray_or_pandas()
        # check that Ts is a categorical vector, or castable
        # to a categorical variable
        self.check_Ts_categorical()
        # generate a formula using user-specified values
        self.generate_formula()

    def check_Xs_ndarray_or_pandas(self):
        """Ensure that Xs is a pandas dataframe, or can be cast to one."""
        # Ensure Xs is a DataFrame
        if not isinstance(self.Xs, pd.DataFrame):
            if self.prop_form_rhs is not None:
                raise ValueError(
                    "Specified a propensity formula `prop_form_rhs' upon initialization, but `Xs' covariate matrix is not a pandas dataframe."
                )
            try:
                # Create column names based on the shape of Xs
                column_names = [f"X{i}" for i in range(self.Xs.shape[1])]
                # Convert Xs to DataFrame with these column names
                self.Xs = pd.DataFrame(self.Xs, columns=column_names)
            except Exception as e:
                raise TypeError(
                    f"Cannot cast covariates `Xs' to a dataframe. Error: {e}"
                )

    def check_Ts_categorical(self):
        """Cast Ts to a categorical vector."""
        try:
            self.unique_treatments = np.unique(self.Ts)
            self.K = len(self.unique_treatments)

            self.Ts_factor = pd.Categorical(
                self.Ts, categories=self.unique_treatments
            ).codes
        except Exception as e:
            raise TypeError(
                f"Cannot cast `Ts' to a categorical treatment vector. Error: {e}"
            )

    def generate_formula(self):
        """
        Check if the right-hand side of a patsy formula is valid.
        If not specified, make one.
        """
        try:
            # Create the right-hand side of the formula
            if self.prop_form_rhs is None:
                self.prop_form_rhs = " + ".join(self.Xs.columns)
            # Create the design matrix using patsy
            self.formula = f"Ts_factor ~ {self.prop_form_rhs}"
            self.Ts_design, self.Xs_design = patsy.dmatrices(
                self.formula,
                pd.concat(
                    [pd.Series(self.Ts_factor, name="Ts_factor"), self.Xs], axis=1
                ),
                return_type="dataframe",
            )
        except Exception as e:
            raise ValueError(f"Error generating propensity model formula: {e}")

    def check_dim_x(self):
        """Convert x proper dimensions"""
        if self.Xs.ndim == 1:
            self.Xs = self.Xs[:, np.newaxis]
        elif self.Xs.ndim != 2:
            raise ValueError(
                "Expected a 2-D array `Xs', found shape " "{}".format(self.Xs.shape)
            )

    def _check_min_samples(self):
        """Check if the number of samples is at least 3"""
        nt = self.Ts.shape[0]
        nx = self.Xs.shape[0]

        if nt <= 3 or nx <= 3:
            raise ValueError("Number of samples is too low")

        if nt != nx:
            raise ValueError(
                "The number of samples does not match between the covariates `Xs' and the Treatments `Ts'. The number of rows of `Xs' and the number of elements of `Ts' should be equal."
            )


class VectorMatch(ABC):
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
    cleaned_inputs : _CleanInputsVM
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

    def __init__(self, retain_ratio=0.05):
        """
        Initialize the VectorMatcher.

        Parameters:
        -----------
        retain_ratio : float, optional
            Minimum proportion of samples to retain
        """
        self.retain_ratio = retain_ratio
        self.is_fitted = False
        self.balanced_ids = None
        self.model = None
        self.pred_probs = None
        self.Rtable = None
        self.unique_treatments = None
        self.cleaned_inputs = None

    def fit(self, Ts, Xs, prop_form_rhs=None, ddx=False, niter=100):
        """
        Fit the vector matching model and identify balanced observations.

        Parameters:
        -----------
        Ts : array-like
            Treatment assignment vector, where entries are one of K-possible treatment indicators. Should have a shape castable to an ``n'' vector, where ``n'' is the number of samples.
        Xs : pandas DataFrame or array-like
            Covariates/features matrix, as an array. Should have a shape ``(n, r)``, where ``n`` is the number of samples, and ``r`` is the number of covariates.
        prop_form_rhs : str, or None, default: None
            the right-hand side of a formula for a generalized propensity score, an extension of the concept of a propensity score to more than two groups.
                - See the documentation for :mod:`statsmodels.discrete.discrete_model.MNLogit` for details on the use of formulas. If a propensity model is specified, anticipates that the covariate matrix specified for the `fit()` method will be a pandas dataframe.
                - Set to `None` to default to a propensity model which includes a single regressor for each column of the covariate matrix.
        ddx : bool, optional, default: False
            Whether to print diagnostic debugging information for model fitting.
        niter : int, optional, default: 100
            The number of iterations for the multinomial logit model.

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

        # Process inputs
        try:
            cleaned_inputs = _CleanInputsVM(Ts, Xs, prop_form_rhs=prop_form_rhs)
        except Exception as e:
            raise ValueError(f"Failed to clean inputs: {str(e)}") from e

        # Fit the multinomial logit model
        try:
            model = MNLogit(cleaned_inputs.Ts_design, cleaned_inputs.Xs_design)
            if ddx:
                result = model.fit(disp=True, maxiter=niter)
            else:
                result = model.fit(disp=False, maxiter=niter)
        except Exception as e:
            raise ValueError(f"Failed to fit propensity model: {str(e)}") from e

        # Get predictions
        try:
            pred_probs = result.predict()
            pred_probs = pd.DataFrame(
                pred_probs, columns=[str(t) for t in range(cleaned_inputs.K)]
            )
        except Exception as e:
            raise ValueError(
                f"Failed to generate propensity score predictions: {str(e)}"
            ) from e

        # Identify overlapping regions
        try:
            Rtable = np.zeros((cleaned_inputs.K, 2))
            for i, t in enumerate(cleaned_inputs.unique_treatments):
                t_idx = np.where(Ts == t)[0]
                for j in range(cleaned_inputs.K):
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
            raise ValueError(f"Failed to calculate overlap regions: {str(e)}") from e

        # Identify balanced observations
        try:
            balanced_ids = []
            for i in range(len(Ts)):
                is_balanced = True
                for j in range(cleaned_inputs.K):
                    if not (
                        pred_probs.iloc[i, j] >= Rtable[j, 0]
                        and pred_probs.iloc[i, j] <= Rtable[j, 1]
                    ):
                        is_balanced = False
                        break
                if is_balanced:
                    balanced_ids.append(i)
        except Exception as e:
            raise ValueError(
                f"Failed to identify balanced observations: {str(e)}"
            ) from e

        # Check if enough samples are retained
        if len(balanced_ids) < self.retain_ratio * len(Ts):
            percentage = 100 * len(balanced_ids) / len(Ts)
            warnings.warn(
                f"Few samples retained by vector matching ({len(balanced_ids)} out of {len(Ts)}, {percentage:.1f}%)."
            )

        if len(balanced_ids) == 0:
            raise ValueError("No samples retained by vector matching.")

        # set attributes after successful rfitting
        self.prop_form_rhs = prop_form_rhs
        self.ddx = ddx
        self.cleaned_inputs = cleaned_inputs
        self.model = model
        self.pred_probs = pred_probs
        self.Rtable = Rtable
        self.balanced_ids = balanced_ids
        self.is_fitted = True

        return balanced_ids
