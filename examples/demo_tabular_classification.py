"""
Demo: PFN for Small Tabular Classification (TabPFN-style)
===========================================================
Trains a PFN on a BNN prior for classification, then evaluates
on a small sklearn dataset. This replicates the spirit of the
TabPFN experiment from Section 4.3 of the paper.

Run: python examples/demo_tabular_classification.py
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from pfn import PFN, BNNPrior, train_pfn, predict


def evaluate_pfn_on_dataset(model, dataset_loader, device, dataset_name):
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}")

    data = dataset_loader()
    X, y = data.data, data.target
    n_classes = len(np.unique(y))

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # PFN inference (single forward pass!)
    pfn_preds = predict(
        model,
        x_context=X_train,
        y_context=y_train.reshape(-1, 1).astype(np.float32),
        x_query=X_test,
        device=device,
    )
    pfn_acc = accuracy_score(y_test, pfn_preds["labels"])

    # Baseline: Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}, Classes: {n_classes}")
    print(f"  PFN accuracy:           {pfn_acc:.3f}")
    print(f"  Random Forest accuracy: {rf_acc:.3f}")
    return pfn_acc, rf_acc


def main():
    DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")

    # Iris: 4 features, 3 classes
    # We'll adapt the model to handle variable feature dims via zero-padding
    # For simplicity in this demo, retrain a smaller model per dataset
    # In the paper, TabPFN uses a single model with fixed max feature size

    small_model = PFN(
        x_dim=4,
        y_dim=1,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        num_classes=3,
    )
    small_prior = BNNPrior(x_dim=4, y_dim=1, num_classes=3)
    train_pfn(
        small_model, small_prior,
        n_steps=3000, batch_size=32, n_context_range=(10, 50), n_query=16,
        lr=1e-4, device=DEVICE, log_interval=600
    )
    pfn_acc, rf_acc = evaluate_pfn_on_dataset(
        small_model, load_iris, DEVICE, "Iris (4 features, 3 classes)"
    )

    print("\n\nSummary:")
    print(f"  PFN: {pfn_acc:.3f}  |  Random Forest: {rf_acc:.3f}")
    print("\nNote: PFN inference is a SINGLE forward pass (no training on the real data).")
    print("With more training steps and a larger model, PFN matches sklearn baselines.")


if __name__ == "__main__":
    main()
