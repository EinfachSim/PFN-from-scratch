"""
Demo: PFN vs Exact GP for 1D Regression
=========================================
Trains a small PFN on a GP prior and compares its predictions
to those of an exact GP. Reproduces the core experiment from Section 4.1
of the paper.

Run: python examples/demo_gp_regression.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pfn import PFN, GPPrior, train_pfn, predict, compare_with_gp


def main():
    # ---- Config ----
    N_STEPS = 3000
    D_MODEL = 128
    N_LAYERS = 4
    N_HEADS = 4
    BATCH_SIZE = 64
    DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")

    print("=" * 60)
    print("PFN Demo: GP Regression")
    print("=" * 60)

    # ---- Build model and prior ----
    prior = GPPrior(x_dim=1, kernel="rbf")
    model = PFN(
        x_dim=1,
        y_dim=1,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_MODEL * 4,
        dropout=0.0,
        num_classes=None,  # Regression
    )

    # ---- Train ----
    print(f"\nTraining for {N_STEPS} steps (small demo)...")
    history = train_pfn(
        model=model,
        prior_sampler=prior,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_context_range=(5, 20),
        n_query=10,
        lr=3e-4,
        device=DEVICE,
        log_interval=300,
    )

    # ---- Generate a test task from the GP prior ----
    print("\nGenerating a test task...")
    x_ctx, y_ctx, x_qry, y_qry = prior.sample_batch(
        batch_size=1, n_context=10, n_query=50, device=torch.device("cpu")
    )
    x_ctx_np = x_ctx[0].numpy()
    y_ctx_np = y_ctx[0].numpy()
    x_qry_np = x_qry[0].numpy()
    y_qry_np = y_qry[0].numpy()

    # ---- Compare PFN vs GP ----
    results = compare_with_gp(
        model=model,
        x_context=x_ctx_np,
        y_context=y_ctx_np,
        x_query=x_qry_np,
        y_query=y_qry_np,
        device=DEVICE,
    )

    print(f"\nResults:")
    print(f"  PFN MSE: {results['pfn_mse']:.4f}")
    print(f"  GP MSE:  {results['gp_mse']:.4f}")
    print(f"  MSE ratio (PFN/GP): {results['mse_ratio']:.2f}x")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sort_idx = np.argsort(x_qry_np.squeeze())
    x_plot = x_qry_np.squeeze()[sort_idx]
    y_true = y_qry_np.squeeze()[sort_idx]

    pfn_mean = results["pfn"]["mean"].squeeze()[sort_idx]
    pfn_lower = results["pfn"]["lower_95"].squeeze()[sort_idx]
    pfn_upper = results["pfn"]["upper_95"].squeeze()[sort_idx]
    gp_mean = results["gp_mean"].squeeze()[sort_idx]
    gp_std = results["gp_std"].squeeze()[sort_idx]

    for ax, title, mean, lower, upper in [
        (axes[0], f"PFN (MSE={results['pfn_mse']:.3f})", pfn_mean, pfn_lower, pfn_upper),
        (axes[1], f"Exact GP (MSE={results['gp_mse']:.3f})", gp_mean,
         gp_mean - 1.96 * gp_std, gp_mean + 1.96 * gp_std),
    ]:
        ax.fill_between(x_plot, lower, upper, alpha=0.3, color="blue", label="95% CI")
        ax.plot(x_plot, mean, "b-", label="Predicted mean")
        ax.plot(x_plot, y_true, "k--", alpha=0.5, label="True function")
        ax.scatter(x_ctx_np, y_ctx_np, c="red", zorder=5, label="Context points", s=50)
        ax.set_title(title)
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"PFN vs Exact GP (trained {N_STEPS} steps)", fontsize=14)
    plt.tight_layout()
    plt.savefig("pfn_vs_gp.png", dpi=150, bbox_inches="tight")
    print("\nSaved plot to pfn_vs_gp.png")
    plt.show()

    # ---- Plot training loss ----
    fig2, ax = plt.subplots(figsize=(8, 4))
    losses = history["loss_history"]
    # Smooth
    window = 50
    smoothed = [
        np.mean(losses[max(0, i - window): i + 1]) for i in range(len(losses))
    ]
    ax.plot(losses, alpha=0.3, color="gray", label="Raw loss")
    ax.plot(smoothed, color="blue", label=f"Smoothed (w={window})")
    ax.set_xlabel("Step")
    ax.set_ylabel("NLL Loss")
    ax.set_title("PFN Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pfn_training_loss.png", dpi=150)
    print("Saved training loss plot to pfn_training_loss.png")
    plt.show()


if __name__ == "__main__":
    main()
