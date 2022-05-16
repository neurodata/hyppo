from .FCIT import FCIT
from .kci import KCI


__all__ = [s for s in dir()]  # add imported tests to __all__

COND_INDEP_TESTS = {"fcit": FCIT, "kci": KCI}
