from ._utils import k_sample_transform
from .disco import DISCO
from .energy import Energy
from .hotelling import Hotelling
from .ksamp import KSample
from .manova import MANOVA
from .mmd import MMD

from .smoothCF import SmoothCFTest, smooth_cf_distance
from .mean_embedding import MeanEmbeddingTest, mean_embed_distance

__all__ = [s for s in dir()]  # add imported tests to __all__


KSAMP_TESTS = {
    "disco": DISCO,
    "energy": Energy,
    "hotelling": Hotelling,
    "ksample": KSample,
    "manova": MANOVA,
    "mmd": MMD,
    "smoothCF": SmoothCFTest,
    "mean_embedding": MeanEmbeddingTest,
}
