"""
PFN Inference
==============
Utility functions for running inference with a trained PFN model.

Key property: inference is a SINGLE forward pass — no MCMC, no VI, no iterative
optimization. The entire posterior predictive is computed in O(n²) time w.r.t.
context size (Transformer attention complexity).
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
from pfn.model import PFN


@torch.no_grad()
def predict(
    model: PFN,
    x_context: Union[torch.Tensor, np.ndarray],  # (n_context, x_dim)
    y_context: Union[torch.Tensor, np.ndarray],  # (n_context, 1 or y_dim)
    x_query: Union[torch.Tensor, np.ndarray],    # (n_query, x_dim)
    device: torch.device = torch.device("cpu"),
    batch_size: Optional[int] = None,            # chunk queries to avoid OOM
) -> dict:
    """
    Run PFN inference on a single task (no batch dimension needed).

    Converts numpy arrays automatically. Returns predictions as numpy arrays.

    Returns dict with:
      - For regression: 'mean', 'std', 'lower_95', 'upper_95'
      - For classification: 'probs', 'labels'
    """
    model = model.to(device)
    model.eval()

    # Convert to tensors and add batch dim
    def to_tensor(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return x.unsqueeze(0).to(device)  # (1, n, dim)

    x_ctx = to_tensor(x_context)
    y_ctx = to_tensor(y_context)
    x_qry = to_tensor(x_query)

    if batch_size is not None:
        # Process queries in chunks for long query sets
        n_query = x_qry.shape[1]
        chunks = x_qry.split(batch_size, dim=1)
        outputs = [model(x_ctx, y_ctx, chunk) for chunk in chunks]
        output = torch.cat(outputs, dim=1)
    else:
        output = model(x_ctx, y_ctx, x_qry)
    # output: (1, n_query, ...)

    output = output.squeeze(0)  # (n_query, ...)

    if model.num_classes is not None:
        probs = torch.softmax(output, dim=-1).cpu().numpy()
        labels = probs.argmax(axis=-1)
        return {"probs": probs, "labels": labels}
    else:
        y_dim = model.y_dim
        mean = output[:, :y_dim].cpu().numpy()
        log_var = output[:, y_dim:].cpu().numpy()
        std = np.exp(0.5 * log_var).clip(min=1e-4)
        return {
            "mean": mean,
            "std": std,
            "lower_95": mean - 1.96 * std,
            "upper_95": mean + 1.96 * std,
        }


@torch.no_grad()
def compute_log_likelihood(
    model: PFN,
    x_context: torch.Tensor,  # (batch, n_context, x_dim)
    y_context: torch.Tensor,  # (batch, n_context, y_dim)
    x_query: torch.Tensor,    # (batch, n_query, x_dim)
    y_query: torch.Tensor,    # (batch, n_query, y_dim)
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Compute the average log-predictive likelihood on query points.
    Useful for evaluating how well the PFN approximates the posterior.

    Returns: scalar tensor (mean log-likelihood per query point)
    """
    import math

    model = model.to(device)
    model.eval()

    x_context = x_context.to(device)
    y_context = y_context.to(device)
    x_query = x_query.to(device)
    y_query = y_query.to(device)

    output = model(x_context, y_context, x_query)  # (B, n_query, 2*y_dim)

    if model.num_classes is not None:
        # Classification log-likelihood
        log_probs = torch.log_softmax(output, dim=-1)
        targets = y_query.long().squeeze(-1)
        ll = -torch.nn.CrossEntropyLoss()(
            log_probs.reshape(-1, model.num_classes),
            targets.reshape(-1),
        )
    else:
        y_dim = model.y_dim
        mean = output[..., :y_dim]
        log_var = output[..., y_dim:]
        var = torch.exp(log_var).clamp(min=1e-6)
        ll = -0.5 * (log_var + (y_query - mean) ** 2 / var + math.log(2 * math.pi))
        ll = ll.mean()

    return ll


def compare_with_gp(
    model: PFN,
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
    kernel_type: str = "rbf",
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Compare PFN predictions against exact GP inference.
    Requires scikit-learn for the GP baseline.

    Returns dict with PFN and GP predictions for side-by-side comparison.
    """
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
    except ImportError:
        raise ImportError("scikit-learn required for GP comparison: pip install scikit-learn")

    # PFN prediction
    pfn_preds = predict(model, x_context, y_context, x_query, device=device)

    # GP prediction
    if kernel_type == "rbf":
        kernel = RBF() + WhiteKernel()
    else:
        kernel = Matern(nu=2.5) + WhiteKernel()

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
    gp.fit(x_context, y_context)
    gp_mean, gp_std = gp.predict(x_query, return_std=True)

    # Compute MSE of means
    pfn_mse = np.mean((pfn_preds["mean"].squeeze() - y_query.squeeze()) ** 2)
    gp_mse = np.mean((gp_mean.squeeze() - y_query.squeeze()) ** 2)

    return {
        "pfn": pfn_preds,
        "gp_mean": gp_mean,
        "gp_std": gp_std,
        "pfn_mse": pfn_mse,
        "gp_mse": gp_mse,
        "mse_ratio": pfn_mse / (gp_mse + 1e-10),
    }
