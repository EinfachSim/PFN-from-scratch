"""
Train a Prior-Data Fitted Network (PFN)

Usage examples:

  # Train on GP prior (regression)
  python train_pfn.py --prior gp --steps 10000 --output checkpoints/gp_pfn.pt

  # Train on BNN prior (regression)
  python train_pfn.py --prior bnn --steps 20000 --output checkpoints/bnn_pfn.pt

  # Train on BNN prior (classification, like TabPFN)
  python train_pfn.py --prior bnn --mode classification --num_classes 10 --output checkpoints/tabpfn.pt
"""

import argparse
import torch
import json
import os

from pfn.model import PFN
from pfn.priors import GPPrior, BNNPrior
from pfn.train import train_pfn


def parse_args():
    p = argparse.ArgumentParser(description="Train a Prior-Data Fitted Network")

    # Prior
    p.add_argument("--prior", choices=["gp", "bnn"], default="gp")
    p.add_argument("--mode", choices=["regression", "classification"], default="regression")
    p.add_argument("--x_dim", type=int, default=1)
    p.add_argument("--num_classes", type=int, default=None,
                   help="Number of classes (sets classification mode)")
    p.add_argument("--kernel", choices=["rbf", "matern52"], default="rbf",
                   help="GP kernel type (only for --prior gp)")

    # Model
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.0)

    # Training
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_context_min", type=int, default=5)
    p.add_argument("--n_context_max", type=int, default=50)
    p.add_argument("--n_query", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--no_scheduler", action="store_true")

    # Output
    p.add_argument("--output", type=str, default="pfn_model.pt")
    p.add_argument("--log_interval", type=int, default=200)
    p.add_argument("--checkpoint_interval", type=int, default=2000)
    p.add_argument("--device", type=str, default=None,
                   help="Device: cuda / mps / cpu (auto-detected if not set)")

    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Classification override
    num_classes = args.num_classes
    if args.mode == "classification" and num_classes is None:
        num_classes = 10
        print(f"Classification mode: defaulting to {num_classes} classes")

    # Build prior
    if args.prior == "gp":
        prior = GPPrior(x_dim=args.x_dim, kernel=args.kernel)
        y_dim = 1
    else:
        prior = BNNPrior(
            x_dim=args.x_dim,
            y_dim=1,
            num_classes=num_classes,
        )
        y_dim = 1

    # Build model
    model = PFN(
        x_dim=args.x_dim,
        y_dim=y_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        num_classes=num_classes,
    )

    # Train
    history = train_pfn(
        model=model,
        prior_sampler=prior,
        n_steps=args.steps,
        batch_size=args.batch_size,
        n_context_range=(args.n_context_min, args.n_context_max),
        n_query=args.n_query,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        log_interval=args.log_interval,
        scheduler=not args.no_scheduler,
        grad_clip=args.grad_clip,
        checkpoint_path=args.output.replace(".pt", "_ckpt"),
        checkpoint_interval=args.checkpoint_interval,
    )

    # Save final model
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": vars(args),
            "loss_history": history["loss_history"],
        },
        args.output,
    )
    print(f"\nSaved model to: {args.output}")

    # Save loss history
    loss_path = args.output.replace(".pt", "_losses.json")
    with open(loss_path, "w") as f:
        json.dump(history["loss_history"], f)
    print(f"Saved loss history to: {loss_path}")


if __name__ == "__main__":
    main()
