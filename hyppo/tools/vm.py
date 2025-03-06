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
                pd.concat([pd.Series(self.Ts_factor, name="Ts_factor"), self.Xs], axis=1),
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
            raise ValueError("The number of samples does not match between the covariates `Xs' and the Treatments `Ts'. The number of rows of `Xs' and the number of elements of `Ts' should be equal.")


class VectorMatch(ABC):
    """
    Vector matching causal data pre-processing.

    A function for implementing the vector matching procedure, a data pre-processing step for K-sample causal discrepancy testing :footcite:p:`Lopez2017Aug`. Uses propensity scores to strategically include/exclude samples from subsequent inference, based on whether (or not) there are samples with similar propensity scores across all treatment levels. Conceptually, this is an algorithmic heuristic for a K-way propensity trimming. Care should be taken to pursue including covariates which conceptually might satisfy the conditional ignorability criterion.

    Parameters
    ----------
    prop_form_rhs : str, or None, default: None
        the right-hand side of a formula for a generalized propensity score, an extension of the concept of a propensity score to more than two groups. 
            - See the documentation for :mod:`statsmodels.discrete.discrete_model.MNLogit` for details on the use of formulas. If a propensity model is specified, anticipates that the covariate matrix specified for the `fit()` method will be a pandas dataframe.
            - Set to `None` to default to a propensity model which includes a single regressor for each column of the covariate matrix.
    retain_ratio: float, default: 0.05
        If the number of samples retained is less than `retain_ratio`, throws a warning. Defaults to `0.05`.
    ddx : bool, optional, default: False
        Whether to print debugging information for model fitting. Defaults to `False`.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, retain_ratio=0.05, prop_form_rhs=None, ddx=False):
        """
        Initialize the VectorMatcher.
        
        Parameters:
        -----------
        retain_ratio : float, optional
            Minimum proportion of samples to retain
        prop_form : str, optional
            Formula for propensity score model (patsy formula syntax for RHS)
        ddx : bool, optional
            Whether to print debugging information
        reference : any, optional
            Reference level for treatment (not used in current implementation)
        """
        self.retain_ratio = retain_ratio
        self.prop_form_rhs = prop_form_rhs
        self.ddx = ddx
        self.balanced_ids = None
        self.model = None
        self.pred_probs = None
        self.Rtable = None
        self.unique_treatments = None
        self.cleaned_inputs = None

    def fit(self, Ts, Xs):
        """
        Fit the vector matching model and identify balanced observations.
        
        Parameters:
        -----------
        Ts : array-like
            Treatment assignment vector, where entries are one of K-possible treatment indicators. Should have a shape castable to an ``n'' vector, where ``n'' is the number of samples.
        Xs : pandas DataFrame or array-like
            Covariates/features matrix, as an array. Should have a shape ``(n, r)``, where ``n`` is the number of samples, and ``r`` is the number of covariates.

        Returns:
        --------
        balance_ids: list of int
          - the positional indices of samples to include for subsequent analysis.
        """
        self.Ts = Ts; self.Xs = Xs
        self.cleaned_inputs = _CleanInputsVM(
            self.Ts, self.Xs, prop_form_rhs=self.prop_form_rhs
        )

        # Fit the multinomial logit model
        self.model = MNLogit(
            self.cleaned_inputs.Ts_design, self.cleaned_inputs.Xs_design
        )
        if self.ddx:
            result = self.model.fit(disp=True)
        else:
            result = self.model.fit(disp=False)

        # Make predictions
        self.pred_probs = result.predict()

        # Convert to DataFrame for easier handling
        self.pred_probs = pd.DataFrame(
            self.pred_probs, columns=[str(t) for t in range(self.cleaned_inputs.K)]
        )

        # Create Rtable to store the range of predicted probabilities for each treatment
        self.Rtable = np.zeros((self.cleaned_inputs.K, 2))
        for i, t in enumerate(self.cleaned_inputs.unique_treatments):
            t_idx = np.where(Ts == t)[0]
            for j in range(self.cleaned_inputs.K):
                # Min and max probabilities for treatment t when actual treatment is j
                min_prob = np.min(self.pred_probs.iloc[t_idx, j])
                max_prob = np.max(self.pred_probs.iloc[t_idx, j])

                # Update Rtable with max of mins and min of maxes
                self.Rtable[j, 0] = max(self.Rtable[j, 0], min_prob)
                self.Rtable[j, 1] = min(
                    self.Rtable[j, 1] if self.Rtable[j, 1] > 0 else float("inf"),
                    max_prob,
                )

        # Check balance condition for each observation and each treatment
        self.balanced_ids = []
        for i in range(len(Ts)):
            is_balanced = True
            for j in range(self.cleaned_inputs.K):
                if not (
                    self.pred_probs.iloc[i, j] >= self.Rtable[j, 0]
                    and self.pred_probs.iloc[i, j] <= self.Rtable[j, 1]
                ):
                    is_balanced = False
                    break
            if is_balanced:
                self.balanced_ids.append(i)

        # Check if enough samples are retained
        if len(self.balanced_ids) < self.retain_ratio * len(Ts):
            warnings.warn("Few samples retained by vector matching.")

        if len(self.balanced_ids) == 0:
            raise ValueError("No samples retained by vector matching.")

        return self.balanced_ids