"""
Module that will contain several goodness-of-fit test methods
"""

from __future__ import division

from abc import ABC, abstractmethod

import data
import _utils
import density
import kernel 
import h0simulator

class GofTest(ABC):
    """
    Abstract class for a goodness-of-fit test.
    """

    def __init__(self, p, alpha):
        """
        p: an UnnormalizedDensity
        alpha: significance level of the test
        """
        self.p = p
        self.alpha = alpha

    @abstractmethod
    def test(self, dat):
        """perform the goodness-of-fit test and return values computed in a dictionary:
        {
            alpha: 0.01, 
            pvalue: 0.0002, 
            test_stat: 2.3, 
            h0_rejected: True, 
            time_secs: ...
        }
        dat: an instance of Data
        """
        raise NotImplementedError()

    @abstractmethod
    def statistic(self, dat):
        """Compute the test statistic"""
        raise NotImplementedError()

