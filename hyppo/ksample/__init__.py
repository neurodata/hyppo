from ._utils import k_sample_transform
from .disco import DISCO
from .energy import Energy
from .hotelling import Hotelling
from .ksamp import KSample
from .manova import MANOVA
from .mmd import MMD

__all__ = [s for s in dir()]  # add imported tests to __all__


KSAMP_TESTS = {
    "disco": DISCO,
    "energy": Energy,
    "hotelling": Hotelling,
    "ksample": KSample,
    "manova": MANOVA,
    "mmd": MMD,
}
