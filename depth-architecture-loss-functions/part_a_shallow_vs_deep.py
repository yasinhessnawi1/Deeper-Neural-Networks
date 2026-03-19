"""
Part A: Shallow vs Deep Networks on Wine Quality.

Compares a shallow network (1 hidden layer, wide) against
a deep network (7 hidden layers, narrow) with roughly the
same parameter budget. Each is trained with two loss functions:
CrossEntropy and MSE.
"""

import torch
import torch.nn as nn
from data_loading import load_wine_quality
from utils import (
    count_parameters, train_model, full_evaluation,
    plot_training_curves, plot_results_table, DEVICE
)


# ──────────────────────────────────────────────
# Model Definitions
# ──────────────────────────────────────────────

class ShallowNet(nn.Module):
    """
    Shallow network: 1 hidden layer with many neurons.
    Wide single layer to compensate for lack of depth.
    Both models use BatchNorm for fair comparison.
    """
    def __init__(self, input_dim, num_classes, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class DeepNet(nn.Module):
    """
    Deep network: 7 hidden layers with fewer neurons per layer.
    Each layer: Linear -> BatchNorm -> ReLU.
    Narrower layers to roughly match ShallowNet's parameter budget.
    """
    def __init__(self, input_dim, num_classes, width=64, depth=7):
        super().__init__()
        layers = []

        # First layer: input_dim -> width
        layers.extend([
            nn.Linear(input_dim, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        ])

        # Hidden layers: width -> width
        for _ in range(depth - 1):
            layers.extend([
                nn.Linear(width, width),
                nn.BatchNorm1d(width),
                nn.ReLU(),
            ])

        # Output layer
        layers.append(nn.Linear(width, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────
# Main Experiment
# ──────────────────────────────────────────────

def run_part_a():
    print("=" * 60)
    print("PART A: Shallow vs Deep Networks")
    print("=" * 60)

    # Load data
    train_loader, val_loader, test_loader, num_features, num_classes = load_wine_quality()

    # Define models and check parameter counts
    # We tune widths so both models have ~similar parameter count
    # Shallow: 1 wide layer (~6,300 params), Deep: 7 narrow layers (~7,400 params)
    # Not exactly equal, but "as far as possible" per assignment spec
    shallow = ShallowNet(num_features, num_classes, width=300)
    deep = DeepNet(num_features, num_classes, width=32, depth=7)

    print(f"\nShallow net parameters: {count_parameters(shallow):,}")
    print(f"Deep net parameters:    {count_parameters(deep):,}")

    # Loss functions to compare
    loss_functions = {
        "CrossEntropy": nn.CrossEntropyLoss(),
        "MSE": nn.MSELoss(),
    }

    # Store all results
    all_histories = {}
    all_results = {}

    for loss_name, loss_fn in loss_functions.items():
        for model_name, ModelClass, kwargs in [
            ("Shallow", ShallowNet, {"width": 300}),
            ("Deep", DeepNet, {"width": 32, "depth": 7}),
        ]:
            label = f"{model_name} + {loss_name}"
            print(f"\n--- Training: {label} ---")

            # Create fresh model and optimizer
            model = ModelClass(num_features, num_classes, **kwargs)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Train
            history = train_model(
                model, train_loader, val_loader, loss_fn, optimizer,
                epochs=100, device=DEVICE
            )
            all_histories[label] = history

            # Evaluate on test set
            results = full_evaluation(model, test_loader, loss_fn, device=DEVICE)
            all_results[label] = results
            print(f"  Test Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")

    # Plot training curves
    plot_training_curves(all_histories,
                         title="Part A: Shallow vs Deep",
                         filename="part_a_training_curves.png")

    # Plot results comparison
    plot_results_table(all_results,
                       title="Part A: Test Performance",
                       filename="part_a_results.png")

    # Print summary
    print("\n" + "=" * 60)
    print("PART A SUMMARY")
    print("=" * 60)
    for label, res in all_results.items():
        print(f"  {label:30s} | Acc: {res['accuracy']:.4f} | F1: {res['f1']:.4f}")

    return all_histories, all_results


if __name__ == "__main__":
    run_part_a()
