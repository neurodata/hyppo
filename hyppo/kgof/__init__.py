from ._utils import meddistance
from .fssd import FSSD
from .data import Data
from .density import Normal
from .kernel import KGauss

__all__ = [s for s in dir()]  # add imported tests to all

GOF_TESTS = {
    "fssd": FSSD,
}
