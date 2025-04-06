import pandas as pd
import numpy as np
import patsy
from ..tools import (
    contains_nan,
    check_min_samples,
    check_2d_array,
    check_ndarray_or_dataframe,
    check_categorical,
)

from ..tools.common import _check_distmat


class CleanInputsPM:
    """
    Cleans inputs for Propensity Model.

    Parameters
    ----------
    Ts : array-like
        Treatment assignment vector, where entries are one of K-possible treatment indicators. Should have a shape castable to an ``n'' vector, where ``n'' is the number of samples.
    Xs : pandas DataFrame or array-like
        Covariates/features matrix, as an array. Should have a shape ``(n, r)``, where ``n'' is the number of samples, and ``r'' is the number of covariates.
    prop_form_rhs : str, or None, default: None
        the right-hand side of a formula for a generalized propensity score, an extension of the concept of a propensity score to (optionally) more than two groups.
            - Set to `None` to default to a propensity model which includes a single regressor for each column of the covariate matrix.

    Attributes
    ----------
    Ts_factor: pandas series
        Cleaned treatment assignment vector, as a categorical pandas series.
    unique_treatments: list
        a dictionary, whose keys are the remapped treatment names after cleaning and values are the original treatment names.
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
        self.validate_tx_covars(Ts, Xs, prop_form_rhs=prop_form_rhs)

    def validate_tx_covars(self, Ts, Xs, prop_form_rhs=None):
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
            Ts_factor, treatment_maps, K = check_categorical(Ts)
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
        self.treatment_maps = treatment_maps
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


class CleanInputsConditionalDiscrepancy(CleanInputsPM):
    """
    Cleans inputs for Conditional discrepancy testing.

    Inherits from _CleanInputsPM to handle treatment and covariate cleaning.
    Additionally validates and cleans outcome data.

    Parameters
    ----------
    Ys : pandas DataFrame or array-like
        Outcome matrix, as an array. Should have a shape ``(n, r)``, where ``n`` is the number of samples, and ``r`` is the number of outcome dimensions.
    Ts : array-like
        Treatment assignment vector, where entries are one of K-possible treatment indicators. Should have a shape castable to an ``n'' vector, where ``n'' is the number of samples.
    Xs : pandas DataFrame or array-like
        Covariates/features matrix, as an array. Should have a shape ``(n, r)``, where ``n'' is the number of samples, and ``r'' is the number of covariates.
    prop_form_rhs : str, or None, default: None
            - Set to `None` to default to a propensity model which includes a single regressor for each column of the covariate matrix.
    outcome_isdist: bool, default: False
        Whether the outcome matrix `Ys' is a data matrix with shape ``(n, r)'' or a distance matrix with shape ``(n, n)''.

    Attributes
    ----------
    Ys_df:  pandas DataFrame
        Cleaned outcomes matrix, as a dataframe with named columns.
    Ts_factor: pandas series (inherited)
        Cleaned treatment assignment vector, as a categorical pandas series.
    Xs_df:  pandas DataFrame (inherited)
        Cleaned covariates/features matrix, as a dataframe with named columns.
    treatment_maps: dict (inherited)
        a dictionary, whose keys are the remapped treatment names after cleaning and values are the original treatment names.
    K: int (inherited)
        the number of unique treatments.
    formula: str (inherited)
        A propensity model.
    Xs_design: patsy.DesignMatrix (inherited)
        Design matrix for the covariates/features.
    Ts_design: patsy.DesignMatrix (inherited)
        Design matrix for the treatment variables.
    """

    def __init__(self, Ys, Ts, Xs, prop_form_rhs=None, outcome_isdist=False):
        # Initialize the parent class first to clean treatments and covariates
        super().__init__(Ts, Xs, prop_form_rhs=prop_form_rhs)

        # Validate outcome
        self.validate_outcome(Ys, outcome_isdist=outcome_isdist)

        # Check the minimum number of samples across all
        check_min_samples(Ys=self.Ys_df, Ts=self.Ts_factor, Xs=self.Xs_df)

    def validate_outcome(self, Ys, outcome_isdist=False):
        """
        Validate and clean the outcome data.

        Parameters
        ----------
        Ys : array-like
            Outcome matrix
        outcome_isdist : bool, default: False
            Whether the outcome is a distance matrix

        Returns
        -------
        None, but sets self.Ys_df
        """
        try:
            Ys = check_2d_array(Ys)
            contains_nan(Ys)
            # if a distance matrix, check it's a ndarray and square
            if outcome_isdist:
                _check_distmat(Ys)
                Ys_df = Ys
            # if not a distance matrix, check it's a ndarray or pandas df
            else:
                Ys_df = check_ndarray_or_dataframe(Ys, "Y")

        except Exception as e:
            exc_type = type(e)
            new_message = f"Error checking `Ys'. Error: {e}"
            raise exc_type(new_message) from e

        self.Ys_df = Ys_df
