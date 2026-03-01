"""
PFN Training
=============
Trains a PFN by repeatedly:
  1. Sampling a task (context + query) from the prior.
  2. Running a forward pass on context + query.
  3. Computing the loss on the query predictions.
  4. Backpropagating.

For regression: Gaussian NLL loss (learns both mean and uncertainty).
For classification: Cross-entropy loss.

The training objective is:
    L = E_{task ~ prior} [ -log p(y_query | x_query, context) ]

This is equivalent to amortized variational inference where the PFN
approximates the Bayesian posterior predictive.
"""

import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Callable
from pfn.model import PFN


def gaussian_nll_loss(
    y_true: torch.Tensor,    # (batch, n_query, y_dim)
    mean: torch.Tensor,      # (batch, n_query, y_dim)
    log_var: torch.Tensor,   # (batch, n_query, y_dim)
) -> torch.Tensor:
    """Gaussian negative log-likelihood: -log N(y | mean, exp(log_var))"""
    var = torch.exp(log_var).clamp(min=1e-6)
    nll = 0.5 * (log_var + (y_true - mean) ** 2 / var + math.log(2 * math.pi))
    return nll.mean()


def train_pfn(
    model: PFN,
    prior_sampler,              # GPPrior or BNNPrior instance
    n_steps: int = 10_000,
    batch_size: int = 32,
    n_context_range: tuple = (5, 50),  # randomly sample context size each step
    n_query: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    device: torch.device = torch.device("mps"),
    log_interval: int = 100,
    scheduler: bool = True,
    grad_clip: float = 1.0,
    checkpoint_path: Optional[str] = None,
    checkpoint_interval: int = 1000,
) -> dict:
    """
    Main PFN training loop.

    Args:
        model: PFN model to train.
        prior_sampler: Object with .sample_batch(batch_size, n_context, n_query, device)
        n_steps: Number of gradient steps.
        batch_size: Tasks per gradient step.
        n_context_range: Min/max context size to randomly sample each step.
        n_query: Number of query points per task.
        lr: Learning rate.
        weight_decay: L2 regularization.
        device: Training device.
        log_interval: Print loss every N steps.
        scheduler: Use cosine LR schedule.
        grad_clip: Max gradient norm.
        checkpoint_path: If set, save checkpoints here.
        checkpoint_interval: Save every N steps.

    Returns:
        dict with loss history.
    """
    model = model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler:
        sched = CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr * 0.01)

    is_classification = model.num_classes is not None
    loss_history = []

    print(f"Training PFN on {device}")
    print(f"  Steps: {n_steps} | Batch: {batch_size} | Mode: {'cls' if is_classification else 'reg'}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    start_time = time.time()

    for step in range(1, n_steps + 1):
        # Randomly sample context size (key: train on varied context lengths)
        n_context = torch.randint(n_context_range[0], n_context_range[1] + 1, (1,)).item()

        # Sample a batch of tasks from the prior
        x_ctx, y_ctx, x_qry, y_qry = prior_sampler.sample_batch(
            batch_size=batch_size,
            n_context=n_context,
            n_query=n_query,
            device=device,
        )

        optimizer.zero_grad()

        if is_classification:
            # y_qry contains integer class labels, shape (B, n_query, 1)
            logits = model(x_ctx, y_ctx, x_qry)  # (B, n_query, num_classes)
            targets = y_qry.long().squeeze(-1)    # (B, n_query)
            loss = nn.CrossEntropyLoss()(
                logits.reshape(-1, model.num_classes),
                targets.reshape(-1),
            )
        else:
            # Regression: predict mean and log_var
            out = model(x_ctx, y_ctx, x_qry)  # (B, n_query, 2*y_dim)
            y_dim = model.y_dim
            mean = out[..., :y_dim]
            log_var = out[..., y_dim:]
            loss = gaussian_nll_loss(y_qry, mean, log_var)

        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        if scheduler:
            sched.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if step % log_interval == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(loss_history[-log_interval:]) / log_interval
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Step {step:6d}/{n_steps} | Loss: {avg_loss:.4f} | "
                f"LR: {lr_now:.2e} | n_ctx: {n_context} | "
                f"Time: {elapsed:.0f}s"
            )

        if checkpoint_path and step % checkpoint_interval == 0:
            torch.save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss_history": loss_history,
                },
                f"{checkpoint_path}_step{step}.pt",
            )
            print(f"  Saved checkpoint at step {step}")

    print(f"\nTraining complete. Final loss: {loss_history[-1]:.4f}")
    return {"loss_history": loss_history}
