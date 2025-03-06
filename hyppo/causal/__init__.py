from .causal_cdcorr import CausalCDcorr

__all__ = [s for s in dir()]  # add imported tests to __all__

CAUSAL_KSAMPLE_TESTS = {
    "ccdcorr": CausalCDcorr,
}