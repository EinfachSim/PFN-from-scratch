"""
Prior-Data Fitted Networks (PFNs)
===================================
Implementation of "Transformers Can Do Bayesian Inference"
Müller et al., ICLR 2022. https://arxiv.org/abs/2112.10510
"""

from .model import PFN
from .train import train_pfn
from .inference import predict, compute_log_likelihood, compare_with_gp
from .priors import GPPrior, BNNPrior

__version__ = "1.0.0"
__all__ = [
    "PFN",
    "train_pfn",
    "predict",
    "compute_log_likelihood",
    "compare_with_gp",
    "GPPrior",
    "BNNPrior",
]
