import pandas as pd
import numpy as np
import patsy
from ..tools import contains_nan, check_min_samples, check_2d_array, check_ndarray_or_dataframe
from ..tools.vm import _CleanInputsPM

class _CleanInputsConditionalDiscrepancy:
    """
    Cleans inputs for Conditional discrepancy testing.

    Parameters
    ----------
    Ys : pandas DataFrame or array-like
        Outcome matrix, as an array. Should have a shape ``(n, r)``, where ``n`` is the number of samples, and ``r`` is the number of outcome dimensions.
    Ts : array-like
        Treatment assignment vector, where entries are one of K-possible treatment indicators. Should have a shape castable to an ``n'' vector, where ``n'' is the number of samples.
    Xs : pandas DataFrame or array-like
        Covariates/features matrix, as an array. Should have a shape ``(n, r)``, where ``n`` is the number of samples, and ``r`` is the number of covariates.
    outcome_only : bool, default: False
        Whether to only thoroughly clean and check outcomes `Ys'. Useful, for instance, if
        a propensity model will be used to separately clean and check the treatments and
        covariates, to avoid repetition of computations. If True, it is advisable to clean
        and check the treatments `Ts' and covariates `Xs' first, and pass the cleaned and
        checked versions into this this utility.
    prop_form_rhs : str, or None, default: None
            - Set to `None` to default to a propensity model which includes a single regressor for each column of the covariate matrix.
            - This option is only functional if `outcome_only' is set to False.
    outcome_isdist: bool, default: False
        Whether the outcome matrix `Ys' is a data matrix with shape ``(n, r)'' or a distance matrix with shape ``(n, n)''.
    
    Attributes
    ----------
    Ys_df:  pandas DataFrame
        Cleaned outcomes matrix, as a dataframe with named columns.
    Ts_factor: pandas series
        Cleaned treatment assignment vector, as a categorical pandas series.
    Xs_df:  pandas DataFrame
        Cleaned covariates/features matrix, as a dataframe with named columns.
    unique_treatments: list
        the unique treatment levels of `Ts_factor'.
    K: int
        the number of unique treatments.
    formula: str
        A propensity model.
    Xs_design: patsy.DesignMatrix
        Design matrix for the covariates/features.
    Ts_design: patsy.DesignMatrix
        Design matrix for the treatment variables.
    """
    def __init__(self, Ys, Ts, Xs, outcome_only=False, prop_form_rhs=None, outcome_isdist=False):
        # check outcomes
        self.validate_inputs(Ys, Ts, Xs, outcome_isdist=outcome_isdist)            
        if not outcome_only:
            # if not outcome only, clean the treatments and covariates
            cleaned_pm = _CleanInputsPM(Ts, Xs, prop_form_rhs=prop_form_rhs)
            self.Xs_df = cleaned_pm.Xs_df; self.Ts_factor = cleaned_pm.Ts_factor
            self.Xs_design = cleaned_pm.Xs_design; self.Ts_design = cleaned_pm.Ts_design
            self.unique_treatments = cleaned_pm.unique_treatments
            self.K = cleaned_pm.K
            self.formula = cleaned_pm.formula
        # check the minimum number of samples across all
        check_min_samples(Ys=self.Ys_df, Ts=self.Ts_factor, Xs=self.Xs_df)

    def validate_outcome(self, Ys, outcome_isdist=False):
        try:
            Ys = check_2d_array(Ys)
            contains_nan(Ys)
            # if a distance matrix, check it's a ndarray and square
            if outcome_isdist:
                Ys_df = _check_distmat(Ys)
            # if not a distance matrix, check it's a ndarray or pandas df
            else:
                Ys_df = check_ndarray_or_dataframe(Ys, "Y")
            
        except Exception as e:
            exc_type = type(e)
            new_message = f"Error checking `Ys'. Error: {e}"
            raise exc_type(new_message) from e

        self.Ys_df = Ys_df