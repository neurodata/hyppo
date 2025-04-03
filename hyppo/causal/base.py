from abc import ABC, abstractmethod
from typing import NamedTuple
import numpy as np

class ConditionalDiscrepancyTestOutput(NamedTuple):
    stat: float
    pvalue: float

class ConditionalDiscrepancyTest(ABC):
    """
    A base class for a conditional discrepancy test.
    
    This class provides a framework for implementing conditional discrepancy 
    tests that evaluate differences in outcome distributions across treatment 
    groups, conditioning on covariates.
    """

    def __init__(self, **kwargs):
        self.stat = None
        self.pvalue = None
        self.kwargs = kwargs

        super().__init__()
        
    def _convert_to_numpy(self, data):
        """
        Convert pandas Series/DataFrame to numpy array if needed.
        
        Parameters
        ----------
        data : array-like
            The input data, which could be a pandas DataFrame, Series, 
            or numpy array.
            
        Returns
        -------
        ndarray
            The input data as a numpy array.
        """
        if hasattr(data, 'values'):
            return data.values
        return np.asarray(data)

    @abstractmethod
    def statistic(self, Ys, Ts, Xs):
        r"""
        Calculates the conditional discrepancy test statistic.

        Parameters
        ----------
        Ys : array-like
            Outcome variables. Can be a numpy array or pandas DataFrame/Series.
        Ts : array-like
            Treatment assignments. Can be a numpy array or pandas Series.
        Xs : array-like
            Covariate/feature matrix. Can be a numpy array or pandas DataFrame.

        Returns
        -------
        stat : float
            The computed conditional discrepancy test statistic.
        """
        pass

    @abstractmethod
    def test(self, Ys, Ts, Xs):
        r"""
        Calculates the conditional discrepancy test statistic and p-value.

        Parameters
        ----------
        Ys : array-like
            Outcome variables. Can be a numpy array or pandas DataFrame/Series.
        Ts : array-like
            Treatment assignments. Can be a numpy array or pandas Series.
        Xs : array-like
            Covariate/feature matrix. Can be a numpy array or pandas DataFrame.

        Returns
        -------
        ConditionalDiscrepancyTestOutput
            A named tuple containing:
            - stat : The computed conditional discrepancy test statistic.
            - pvalue : The computed conditional discrepancy test p-value.
        """
        pass