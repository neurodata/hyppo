import warnings
from statsmodels.discrete.discrete_model import MNLogit
import pandas as pd
import numpy as np
from .cleaners import CleanInputsPM


class GeneralisedPropensityModel:
    """
    This is a lightweight class for fitting generalised propensity models.

    A generalised propensity score is a length `K` vector `r(x)` for covariates `x`
    where each element `rk(x)` indicates the probability of a sample with covariates
    `x` being assigned to treatment group `k`.

    Parameters
    ----------
    retain_ratio: float, default: 0.05
        If the number of samples retained is less than `retain_ratio`, throws a warning.

    Attributes
    ----------
    model : statsmodels.MNLogit
        Fitted multinomial logistic regression model.
    model_result : statsmodels.discrete.discrete_model.MNLogitResults
        Result of model fitting from MNLogit regression.
    pred_probs : pandas.DataFrame
        Predicted probabilities from the propensity model.
    is_fitted : bool
        Whether the model has been fitted.
    prop_form_rhs : str or None
        Formula used for propensity score estimation.
    ddx : bool
        Whether debugging information was displayed during fitting.
    cleaned_inputs : CleanInputsPM
        Processed input data used for model fitting.
    unique_treatments : array-like
        Unique treatment values found in the data.
    """

    def __init__(self):
        """
        Initialize the GeneralisedPropensityModel.
        """
        self.is_fitted = False
        self.model = None
        self.model_result = None
        self.pred_probs = None
        self.cleaned_inputs = None
        self.ddx = False
        self.niter = np.nan

    def _clean(self, Ts, Xs, prop_form_rhs=None):
        """
        An internal function to handle cleaning separately.

        Parameters
        ----------
        Ts : array-like
            Treatment assignment vector
        Xs : pandas DataFrame or array-like
            Covariates/features matrix
        prop_form_rhs : str, or None, default: None
            The right-hand side of a formula for a generalised propensity score

        Returns
        -------
        self : GeneralisedPropensityModel
            The instance with cleaned inputs
        """
        # Process inputs
        try:
            self.cleaned_inputs = CleanInputsPM(Ts, Xs, prop_form_rhs=prop_form_rhs)
        except Exception as e:
            raise ValueError(f"Failed to clean inputs: {str(e)}") from e
        # set attributes after cleaning
        self.prop_form_rhs = prop_form_rhs
        self.nsamples = len(self.cleaned_inputs.Ts_factor)
        return self

    def _fit(self, ddx=False, niter=100):
        """
        Internal method for fitting the generalised propensity model and
        estimating generalised propensity scores using multinomial logistic regression.

        This method assumes that cleaning has already been performed
        and self.cleaned_inputs is available.

        Parameters
        ----------
        ddx : bool, optional, default: False
            Whether to print diagnostic debugging information for model fitting.
        niter : int, optional, default: 100
            The number of iterations for the multinomial logit model.

        Returns
        -------
        self : GeneralisedPropensityModel
            The fitted model instance
        """
        if not hasattr(self, "cleaned_inputs") or self.cleaned_inputs is None:
            raise ValueError(
                "Inputs must be cleaned before fitting. Call _clean() first."
            )

        # Fit the multinomial logit model
        try:
            model = MNLogit(
                self.cleaned_inputs.Ts_design, self.cleaned_inputs.Xs_design
            )
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
            new_message = (
                f"Failed to generate generalised propensity score predictions: {str(e)}"
            )
            raise exc_type(new_message) from e

        self.model = model
        self.model_result = result
        self.pred_probs = pred_probs
        self.is_fitted = True

        return self

    def fit(self, Ts, Xs, prop_form_rhs=None, ddx=False, niter=100):
        """
        Fit a generalised propensity score model (multinomial logistic regression).

        This method cleans the inputs and fits a propensity model.

        Parameters
        ----------
        Ts : array-like
            Treatment assignment vector, where entries are one of K-possible treatment indicators.
            Should have a shape castable to an "n" vector, where "n" is the number of samples.
        Xs : pandas DataFrame or array-like
            Covariates/features matrix, as an array. Should have a shape "(n, r)",
            where "n" is the number of samples, and "r" is the number of covariates.
        prop_form_rhs : str, or None, default: None
            The right-hand side of a formula for a generalised propensity score, an extension
            of the concept of a propensity score to (optionally) more than two groups.
            - See the documentation for :mod:`statsmodels.discrete.discrete_model.MNLogit` for details
              on the use of formulas. If a propensity model is specified, anticipates that the
              covariate matrix specified for the `fit()` method will be a pandas dataframe.
            - Set to `None` to default to a propensity model which includes a single regressor
              for each column of the covariate matrix.
        ddx : bool, optional, default: False
            Whether to print diagnostic debugging information for model fitting.
        niter : int, optional, default: 100
            The number of iterations for the multinomial logit model.

        Returns
        -------
        self : GeneralisedPropensityModel
            The fitted model instance
        """
        if self.is_fitted:
            raise ValueError(
                "This GeneralisedPropensityModel instance has already been fit. "
                "Create a new instance for a new dataset."
            )

        self._clean(Ts, Xs, prop_form_rhs=prop_form_rhs)
        self.ddx = ddx
        self.niter = niter

        return self._fit(ddx=ddx, niter=niter)

    def vector_match(self, retain_ratio=0.05):
        """
        Perform vector matching to identify balanced observations.

        A propensity score-based algorithm for pre-processing observational data with multiple
        treatment groups :footcite:p:`Lopez2017Aug`. The algorithm works as follows, given estimated propensity scores from `fit()`:
        1. For each treatment group `k`, identifies the range of propensity scores
        2. Determines the overlapping region of propensity scores across all groups
        3. Retains only observations that fall within the common support region

        This effectively removes observations without comparable matches in other treatment
        groups, improving covariate balance for downstream causal inference.

        This method should be called after fitting the propensity model with `fit()`.

        Parameters
        ----------
        retain_ratio : float, default 0.05
            Minimum proportion of samples to retain. Defaults to 0.05.

        Returns
        -------
        balanced_ids : list of int
            The positional indices of samples to include for subsequent analysis.

        Attributes
        ----------
        balanced_ids : array-like
            Indices of observations that are retained after vector matching.
        Rtable : numpy.ndarray
            Table of propensity score ranges for each treatment group.
        retain_ratio : float
            Minimum proportion of samples that should be retained.

        References
        ----------
        .. footbibliography::
        """
        if not self.is_fitted:
            raise ValueError(
                "Model must be fitted before performing vector matching. "
                "Call fit() method first."
            )

        # Validate that retain_ratio is valid
        if not isinstance(retain_ratio, (float, int)):
            raise TypeError("retain_ratio must be a number (float or int)")

        if not (0 <= retain_ratio <= 1):
            raise ValueError("retain_ratio should be a fraction between 0 and 1.")

        Rtable = np.zeros((self.cleaned_inputs.K, 2))

        try:
            # For each treatment group t
            for t in range(self.cleaned_inputs.K):
                # Arrays to store min and max propensity scores for treatment t across all treatment groups
                min_probs_across_groups = []
                max_probs_across_groups = []

                # For each actual treatment group t'
                for code in self.cleaned_inputs.treatment_maps["code_to_value"].keys():
                    # Indices of samples with actual treatment t'
                    indices = np.where(self.cleaned_inputs.Ts_factor == code)[0]

                    # Skip if no samples have this treatment
                    if len(indices) == 0:
                        continue

                    # Get propensity scores for treatment t among samples with actual treatment t'
                    propensity_scores = self.pred_probs.iloc[indices, t]

                    # Find min and max
                    min_prob = np.min(propensity_scores)
                    max_prob = np.max(propensity_scores)

                    # Add to lists
                    min_probs_across_groups.append(min_prob)
                    max_probs_across_groups.append(max_prob)

                # Compute l(t) and h(t)
                # l(t) = max of mins, h(t) = min of maxes
                Rtable[t, 0] = np.max(min_probs_across_groups)
                Rtable[t, 1] = np.min(max_probs_across_groups)

        except Exception as e:
            exc_type = type(e)
            new_message = f"Failed to calculate overlap regions: {str(e)}"
            raise exc_type(new_message) from e

        # Identify balanced observations
        try:
            balanced_ids = []

            # For each sample
            for i in range(self.nsamples):
                is_balanced = True

                # Check if propensity scores for all treatments fall within bounds
                for t in range(self.cleaned_inputs.K):
                    # Get propensity score for treatment t
                    prop_score = self.pred_probs.iloc[i, t]

                    # Check if it's within bounds
                    if prop_score < Rtable[t, 0] or prop_score > Rtable[t, 1]:
                        is_balanced = False
                        break

                # If sample is balanced, add to list
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
        self.Rtable = Rtable
        self.balanced_ids = balanced_ids

        return balanced_ids

    def fit_and_vector_match(
        self, Ts, Xs, prop_form_rhs=None, ddx=False, niter=100, retain_ratio=0.05
    ):
        """
        Convenience method to both fit the model and perform vector matching in one call. Vector matching is a propensity score-based algorithm for pre-processing observational data with multiple treatment groups :footcite:p:`Lopez2017Aug`.

        This is equivalent to calling fit() followed by vector_match().

        Parameters
        ----------
        Ts : array-like
            Treatment assignment vector
        Xs : pandas DataFrame or array-like
            Covariates/features matrix
        prop_form_rhs : str, or None, default: None
            The right-hand side of a formula for a generalised propensity score
        ddx : bool, optional, default: False
            Whether to print diagnostic debugging information for model fitting.
        niter : int, optional, default: 100
            The number of iterations for the multinomial logit model.
        retain_ratio : float, optional, default: 0.05
            Minimum proportion of samples to retain

        Returns
        -------
        balanced_ids : list of int
            The positional indices of samples to include for subsequent analysis.

        References
        ----------
        .. footbibliography::
        """
        self.fit(Ts, Xs, prop_form_rhs=prop_form_rhs, ddx=ddx, niter=niter)
        return self.vector_match(retain_ratio=retain_ratio)

    def _fit_from_cleaned(self, cleaned_inputs, ddx=False, niter=100):
        """
        Fits a generalised propensity model using pre-cleaned inputs. Internal method to streamline the use of classes leveraging input cleaners inheriting the CleanInputsPM class.

        This method allows for external cleaning of inputs before fitting,
        which can be useful when integrating with other preprocessing steps.

        Parameters
        ----------
        cleaned_inputs : CleanInputsPM
            A properly initialized CleanInputsPM instance with validated inputs
        ddx : bool, optional, default: False
            Whether to print diagnostic debugging information for model fitting
        niter : int, optional, default: 100
            The number of iterations for the multinomial logit model

        Returns
        -------
        self : GeneralisedPropensityModel
            The fitted model instance
        """
        if self.is_fitted:
            raise ValueError(
                "This GeneralisedPropensityModel instance has already been fit. "
                "Create a new instance for a new dataset."
            )

        if not isinstance(cleaned_inputs, CleanInputsPM):
            raise TypeError("cleaned_inputs must be an instance of CleanInputsPM")

        self.cleaned_inputs = cleaned_inputs
        self.nsamples = len(self.cleaned_inputs.Ts_factor)
        self.prop_form_rhs = getattr(cleaned_inputs, "prop_form_rhs", None)
        self.ddx = ddx
        self.niter = niter

        return self._fit(ddx=ddx, niter=niter)
