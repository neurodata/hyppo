from ._utils import meddistance
from ._utils import fit_gaussian_draw
from .fssd import FSSD

__all__ = [s for s in dir()]  # add imported tests to all

GOF_TESTS = {
    "fssd": FSSD,
}
