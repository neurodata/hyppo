from .cdcorr import ConditionalDcorr
from .FCIT import FCIT
from .kci import KCI
from .pcorr import PartialCorr
from .pdcorr import PartialDcorr

__all__ = [s for s in dir()]  # add imported tests to __all__

COND_INDEP_TESTS = {
    "fcit": FCIT,
    "kci": KCI,
    "conditionaldcorr": ConditionalDcorr,
    "partialcorr": PartialCorr,
    "partialdcorr": PartialDcorr,
}
