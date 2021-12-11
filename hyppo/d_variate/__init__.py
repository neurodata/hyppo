from .dhsic import dHsic

__all__ = [s for s in dir()]  # add imported tests to __all__

MULTI_TESTS = {
    "dhsic": dHsic,
}
