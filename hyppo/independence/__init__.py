from ._utils import sim_matrix
from .cca import CCA
from .dcorr import Dcorr
from .hhg import HHG
from .hsic import Hsic
from .kmerf import KMERF
from .max_margin import MaxMargin
from .mgc import MGC
from .rv import RV

__all__ = [s for s in dir()]  # add imported tests to __all__


INDEP_TESTS = {
    "rv": RV,
    "cca": CCA,
    "hhg": HHG,
    "hsic": Hsic,
    "dcorr": Dcorr,
    "mgc": MGC,
    "kmerf": KMERF,
    "maxmargin": MaxMargin,
}
