"""
Gaussian Process Prior for PFN Training
========================================
Samples synthetic supervised learning tasks from a GP prior.

For each task:
  1. Sample GP hyperparameters (length-scale, output-scale, noise) from hyper-priors.
  2. Sample N input points x ~ Uniform([0,1]^d).
  3. Compute the GP covariance matrix K(x, x).
  4. Sample y ~ N(0, K + noise*I).

This gives us ground-truth Bayesian posterior data to train on.
The paper shows PFNs trained this way can near-perfectly mimic GPs at inference time.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def rbf_kernel(
    x1: torch.Tensor,  # (..., n, d)
    x2: torch.Tensor,  # (..., m, d)
    length_scale: torch.Tensor,  # (..., 1) or scalar
    output_scale: torch.Tensor,  # (..., 1) or scalar
) -> torch.Tensor:
    """Squared-exponential (RBF) kernel: k(x,x') = σ² exp(-||x-x'||² / (2l²))"""
    # Pairwise squared distances
    diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)  # (..., n, m, d)
    sq_dist = (diff ** 2).sum(-1)               # (..., n, m)
    return output_scale ** 2 * torch.exp(-0.5 * sq_dist / length_scale ** 2)


def matern52_kernel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    length_scale: torch.Tensor,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    """Matérn 5/2 kernel."""
    diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
    dist = (diff ** 2).sum(-1).clamp(min=1e-12).sqrt()
    sqrt5 = 5 ** 0.5
    r = sqrt5 * dist / length_scale
    return output_scale ** 2 * (1 + r + r ** 2 / 3) * torch.exp(-r)


class GPPrior:
    """
    Samples batches of supervised tasks from a GP prior.

    Each task = (x_train, y_train, x_test, y_test) where y is drawn from
    a GP with randomly sampled hyperparameters.
    """

    def __init__(
        self,
        x_dim: int = 1,
        kernel: str = "rbf",  # "rbf" or "matern52"
        # Hyper-priors (log-normal)
        length_scale_mean: float = 0.0,
        length_scale_std: float = 0.5,
        output_scale_mean: float = 0.0,
        output_scale_std: float = 0.5,
        noise_std_mean: float = -2.0,   # log-normal -> typically small
        noise_std_std: float = 0.5,
        x_range: Tuple[float, float] = (0.0, 1.0),
    ):
        self.x_dim = x_dim
        self.kernel_type = kernel
        self.ls_mean = length_scale_mean
        self.ls_std = length_scale_std
        self.os_mean = output_scale_mean
        self.os_std = output_scale_std
        self.ns_mean = noise_std_mean
        self.ns_std = noise_std_std
        self.x_range = x_range

    def _kernel(self, x1, x2, length_scale, output_scale):
        if self.kernel_type == "rbf":
            return rbf_kernel(x1, x2, length_scale, output_scale)
        elif self.kernel_type == "matern52":
            return matern52_kernel(x1, x2, length_scale, output_scale)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")

    def sample_batch(
        self,
        batch_size: int,
        n_context: int,
        n_query: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_context: (batch, n_context, x_dim)
            y_context: (batch, n_context, 1)
            x_query:   (batch, n_query,   x_dim)
            y_query:   (batch, n_query,   1)
        """
        n_total = n_context + n_query

        # Sample hyperparameters from log-normal hyper-priors
        length_scale = torch.exp(
            torch.randn(batch_size, 1, device=device) * self.ls_std + self.ls_mean
        ).unsqueeze(-1)  # (B, 1, 1)
        output_scale = torch.exp(
            torch.randn(batch_size, 1, device=device) * self.os_std + self.os_mean
        ).unsqueeze(-1)
        noise_std = torch.exp(
            torch.randn(batch_size, 1, device=device) * self.ns_std + self.ns_mean
        )  # (B, 1)

        # Sample input points uniformly
        lo, hi = self.x_range
        x_all = torch.rand(batch_size, n_total, self.x_dim, device=device) * (hi - lo) + lo
        # (B, n_total, x_dim)

        # Compute GP covariance
        K = self._kernel(x_all, x_all, length_scale, output_scale)
        # Add noise + jitter for numerical stability
        noise_var = noise_std ** 2  # (B, 1)
        K = K + (noise_var.unsqueeze(-1) + 1e-6) * torch.eye(n_total, device=device).unsqueeze(0)
        # (B, n_total, n_total)

        # Sample from GP: y ~ N(0, K)
        # Using Cholesky decomposition
        try:
            L = torch.linalg.cholesky(K)  # (B, n_total, n_total)
        except Exception:
            # Fallback: add more jitter
            K = K + 1e-4 * torch.eye(n_total, device=device).unsqueeze(0)
            L = torch.linalg.cholesky(K)

        eps = torch.randn(batch_size, n_total, 1, device=device)
        y_all = torch.bmm(L, eps)  # (B, n_total, 1)

        x_context = x_all[:, :n_context]
        y_context = y_all[:, :n_context]
        x_query = x_all[:, n_context:]
        y_query = y_all[:, n_context:]

        return x_context, y_context, x_query, y_query
