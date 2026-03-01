"""
Prior-Data Fitted Networks (PFN) - Transformer Model
=====================================================
Implements the core PFN architecture from:
  "Transformers Can Do Bayesian Inference" (Müller et al., ICLR 2022)

The model takes a set-valued context (x_train, y_train) and a query x_test,
and outputs a predictive distribution over y_test in a single forward pass.

Key idea: treat (x_i, y_i) pairs as tokens, append query token with masked y,
and predict the distribution of the masked y using a Transformer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (used optionally)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PFNEncoder(nn.Module):
    """
    Encodes (x, y) context pairs and query x into a fixed-dimensional embedding.

    For context points: embed [x, y] -> d_model
    For query points:   embed [x, 0] -> d_model  (y is masked/zeroed)
    """

    def __init__(self, x_dim: int, y_dim: int, d_model: int):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        # Context encoder: takes (x, y)
        self.context_encoder = nn.Sequential(
            nn.Linear(x_dim + y_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        # Query encoder: takes x only (y is unknown)
        self.query_encoder = nn.Sequential(
            nn.Linear(x_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        x_context: torch.Tensor,  # (batch, n_context, x_dim)
        y_context: torch.Tensor,  # (batch, n_context, y_dim)
        x_query: torch.Tensor,    # (batch, n_query, x_dim)
    ) -> torch.Tensor:
        """
        Returns:
            tokens: (batch, n_context + n_query, d_model)
            n_context: int (so caller knows where queries start)
        """
        ctx_tokens = self.context_encoder(
            torch.cat([x_context, y_context], dim=-1)
        )  # (batch, n_context, d_model)
        qry_tokens = self.query_encoder(x_query)  # (batch, n_query, d_model)
        tokens = torch.cat([ctx_tokens, qry_tokens], dim=1)
        return tokens


class PFN(nn.Module):
    """
    Prior-Data Fitted Network.

    Architecture:
      1. Encode context (x,y) pairs and query x into tokens.
      2. Pass through a Transformer encoder (permutation-equivariant over the set).
      3. Read out the query token(s) and decode into a predictive distribution.

    For classification: outputs logits over num_classes.
    For regression:     outputs (mean, log_var) for a Gaussian predictive.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.0,
        num_classes: int = None,  # if None -> regression
        max_context_len: int = 1000,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.d_model = d_model
        self.num_classes = num_classes

        # Input encoding
        self.encoder = PFNEncoder(x_dim, y_dim, d_model)

        # Transformer (encoder-only, attends over all tokens including queries)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for stability (used in paper)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model)
        )

        # Output head
        if num_classes is not None:
            # Classification: logits over classes
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, num_classes),
            )
        else:
            # Regression: predict Gaussian mean and log-variance
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 2 * y_dim),  # (mean, log_var) per output dim
            )

    def forward(
        self,
        x_context: torch.Tensor,  # (batch, n_context, x_dim)
        y_context: torch.Tensor,  # (batch, n_context, y_dim)
        x_query: torch.Tensor,    # (batch, n_query, x_dim)
    ) -> torch.Tensor:
        """
        Returns logits (classification) or (mean, log_var) (regression)
        for each query point. Shape: (batch, n_query, num_classes or 2*y_dim).
        """
        n_context = x_context.shape[1]

        # Encode all tokens
        tokens = self.encoder(x_context, y_context, x_query)
        # (batch, n_context + n_query, d_model)

        # Transformer over all tokens (full attention -- set-valued input)
        out = self.transformer(tokens)
        # (batch, n_context + n_query, d_model)

        # Extract query token outputs only
        query_out = out[:, n_context:, :]  # (batch, n_query, d_model)

        return self.output_head(query_out)

    def predict_classification(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_query: torch.Tensor,
    ) -> torch.Tensor:
        """Returns class probabilities via softmax. Shape: (batch, n_query, num_classes)."""
        logits = self.forward(x_context, y_context, x_query)
        return torch.softmax(logits, dim=-1)

    def predict_regression(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_query: torch.Tensor,
    ):
        """Returns (mean, std) for Gaussian predictive. Both: (batch, n_query, y_dim)."""
        out = self.forward(x_context, y_context, x_query)
        mean = out[..., : self.y_dim]
        log_var = out[..., self.y_dim :]
        std = torch.exp(0.5 * log_var).clamp(min=1e-4)
        return mean, std
