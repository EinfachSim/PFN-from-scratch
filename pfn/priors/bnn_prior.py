"""
Bayesian Neural Network (BNN) Prior for PFN Training
======================================================
Samples supervised tasks from a BNN prior.

For each task:
  1. Sample a random neural network architecture (weights drawn from N(0,1)).
  2. Sample N input points x.
  3. Compute y = f_w(x) + noise, where f_w is the sampled BNN.

This defines a prior over functions (much richer than GPs) and lets PFNs
do approximate Bayesian inference over BNN posteriors.

The paper uses this prior for classification on small tabular datasets,
resulting in the "TabPFN" model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class SampledBNN(nn.Module):
    """A randomly-initialized BNN with sampled weights. Used as a function prior."""

    def __init__(
        self,
        x_dim: int,
        hidden_dims: List[int],
        y_dim: int,
        activation: str = "relu",
        weight_std: float = 1.0,
        bias_std: float = 0.1,
    ):
        super().__init__()
        dims = [x_dim] + hidden_dims + [y_dim]
        layers = []
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1])
            # Sample weights from scaled normal (Neal's infinite-width prior)
            nn.init.normal_(layer.weight, 0, weight_std / (dims[i] ** 0.5))
            nn.init.normal_(layer.bias, 0, bias_std)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x


class BNNPrior:
    """
    Samples batches of supervised tasks from a BNN prior.

    Each task draws a new random BNN and uses it to generate (x, y) pairs.
    """

    def __init__(
        self,
        x_dim: int = 5,
        y_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        noise_std: float = 0.01,
        weight_std: float = 1.0,
        bias_std: float = 0.1,
        x_range: Tuple[float, float] = (-2.0, 2.0),
        num_classes: Optional[int] = None,  # if set, do classification
    ):
        self.x_dim = x_dim
        self.y_dim = y_dim if num_classes is None else num_classes
        self.hidden_dims = hidden_dims or [64, 64]
        self.activation = activation
        self.noise_std = noise_std
        self.weight_std = weight_std
        self.bias_std = bias_std
        self.x_range = x_range
        self.num_classes = num_classes

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
            y_context: (batch, n_context, y_dim or 1 for class labels)
            x_query:   (batch, n_query, x_dim)
            y_query:   (batch, n_query, y_dim or 1)
        """
        n_total = n_context + n_query
        lo, hi = self.x_range
        x_all = torch.rand(batch_size, n_total, self.x_dim, device=device) * (hi - lo) + lo

        y_list = []
        for b in range(batch_size):
            # Sample a new random BNN for each task
            bnn = SampledBNN(
                x_dim=self.x_dim,
                hidden_dims=self.hidden_dims,
                y_dim=self.y_dim,
                activation=self.activation,
                weight_std=self.weight_std,
                bias_std=self.bias_std,
            ).to(device)

            with torch.no_grad():
                out = bnn(x_all[b])  # (n_total, y_dim)

            if self.num_classes is not None:
                # Convert to class labels via argmax
                labels = out.argmax(dim=-1, keepdim=True).float()  # (n_total, 1)
                y_list.append(labels)
            else:
                # Add observation noise
                noise = torch.randn_like(out) * self.noise_std
                y_list.append(out + noise)

        y_all = torch.stack(y_list, dim=0)  # (B, n_total, y_dim or 1)

        x_context = x_all[:, :n_context]
        y_context = y_all[:, :n_context]
        x_query = x_all[:, n_context:]
        y_query = y_all[:, n_context:]

        return x_context, y_context, x_query, y_query
