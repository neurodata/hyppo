from ._utils import sim_matrix
from .kci import KCI


__all__ = [s for s in dir()]  # add imported tests to __all__


INDEP_TESTS = {
    "kci": KCI,
}
