"""
Part B: Loss Function Experiments on Wine Quality.

Tests 4 loss functions: CrossEntropy, MSE, MAE, Huber.
Analyzes convergence speed, stability, and robustness to outliers.
Uses the Shallow network from Part A for consistency.
"""

import torch
import torch.nn as nn
import numpy as np
from data_loading import load_wine_quality
from utils import (
    count_parameters, train_model, full_evaluation,
    plot_training_curves, plot_results_table, DEVICE
)


class ShallowNet(nn.Module):
    """Same shallow network from Part A for fair comparison."""
    def __init__(self, input_dim, num_classes, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def run_part_b():
    print("=" * 60)
    print("PART B: Loss Function Experiments")
    print("=" * 60)

    train_loader, val_loader, test_loader, num_features, num_classes = load_wine_quality()

    # 4 loss functions to compare
    loss_functions = {
        "CrossEntropy": nn.CrossEntropyLoss(),
        "MSE": nn.MSELoss(),
        "MAE (L1)": nn.L1Loss(),
        "Huber": nn.SmoothL1Loss(),  # SmoothL1 = Huber with delta=1
    }

    all_histories = {}
    all_results = {}

    for loss_name, loss_fn in loss_functions.items():
        print(f"\n--- Training with {loss_name} ---")

        model = ShallowNet(num_features, num_classes, width=256)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        history = train_model(
            model, train_loader, val_loader, loss_fn, optimizer,
            epochs=100, device=DEVICE
        )
        all_histories[loss_name] = history

        results = full_evaluation(model, test_loader, loss_fn, device=DEVICE)
        all_results[loss_name] = results
        print(f"  Test Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")

    # ── Outlier Robustness Test ──
    # Inject 10% label noise into training data and retrain
    print("\n" + "-" * 60)
    print("OUTLIER ROBUSTNESS TEST (10% label noise)")
    print("-" * 60)

    # Reload data and corrupt labels
    train_loader_noisy, val_loader, test_loader, num_features, num_classes = load_wine_quality()

    # Corrupt 10% of training labels
    dataset = train_loader_noisy.dataset
    tensors = list(dataset.tensors)
    labels = tensors[1].clone()
    n_corrupt = int(0.1 * len(labels))
    corrupt_idx = np.random.RandomState(42).choice(len(labels), n_corrupt, replace=False)
    # Assign random wrong labels
    for idx in corrupt_idx:
        original = labels[idx].item()
        new_label = np.random.RandomState(idx).choice(
            [c for c in range(num_classes) if c != original]
        )
        labels[idx] = new_label
    tensors[1] = labels
    noisy_dataset = torch.utils.data.TensorDataset(*tensors)
    train_loader_noisy = torch.utils.data.DataLoader(
        noisy_dataset, batch_size=64, shuffle=True
    )

    noisy_histories = {}
    noisy_results = {}

    for loss_name, loss_fn in loss_functions.items():
        print(f"\n--- {loss_name} (with noisy labels) ---")

        model = ShallowNet(num_features, num_classes, width=256)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        history = train_model(
            model, train_loader_noisy, val_loader, loss_fn, optimizer,
            epochs=100, device=DEVICE
        )
        noisy_histories[f"{loss_name} (noisy)"] = history

        results = full_evaluation(model, test_loader, loss_fn, device=DEVICE)
        noisy_results[f"{loss_name} (noisy)"] = results
        print(f"  Test Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")

    # Plot clean training curves
    plot_training_curves(all_histories,
                         title="Part B: Loss Function Comparison (Clean)",
                         filename="part_b_clean_curves.png")

    # Plot noisy training curves
    plot_training_curves(noisy_histories,
                         title="Part B: Loss Function Comparison (Noisy Labels)",
                         filename="part_b_noisy_curves.png")

    # Combined results bar chart
    combined_results = {**all_results, **noisy_results}
    plot_results_table(combined_results,
                       title="Part B: Clean vs Noisy Test Performance",
                       filename="part_b_results.png")

    # Summary
    print("\n" + "=" * 60)
    print("PART B SUMMARY")
    print("=" * 60)
    print(f"  {'Loss Function':25s} | {'Clean Acc':>10s} | {'Noisy Acc':>10s} | {'Drop':>8s}")
    print("  " + "-" * 60)
    for loss_name in loss_functions.keys():
        clean_acc = all_results[loss_name]["accuracy"]
        noisy_acc = noisy_results[f"{loss_name} (noisy)"]["accuracy"]
        drop = clean_acc - noisy_acc
        print(f"  {loss_name:25s} | {clean_acc:10.4f} | {noisy_acc:10.4f} | {drop:8.4f}")

    return all_histories, all_results, noisy_histories, noisy_results


if __name__ == "__main__":
    run_part_b()
