from .kci import KCI
from .FCIT import FCIT


__all__ = [s for s in dir()]  # add imported tests to __all__


COND_INDEP_TESTS = {
    "kci": KCI,
    "fcit":FCIT
}
