from .dhsic import Dhsic

__all__ = [s for s in dir()]  # add imported tests to __all__

MULTI_TESTS = {
    "dhsic": Dhsic,
}
