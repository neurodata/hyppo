from .pearson import Pearson
from .rv import RV
from .cca import CCA
from .kendall import Kendall
from .spearman import Spearman
from .hhg import HHG
from .dcorr import Dcorr
from .hsic import Hsic

__all__ = ["Pearson", "RV", "CCA", "Kendall", "Spearman", "HHG",
           "Dcorr", "Hsic"]