"""
Part D: Custom Loss Function.

Implements Focal Loss — a modification of CrossEntropy that
down-weights easy examples and focuses learning on hard ones.

Mathematical definition:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Where:
    p_t = predicted probability for the true class
    gamma = focusing parameter (gamma=0 reduces to standard CE)
    alpha_t = class weight (optional, for class imbalance)

Why Focal Loss?
    - Wine Quality has imbalanced classes (most samples are quality 5-6)
    - Standard CE treats all samples equally
    - Focal Loss reduces loss for well-classified samples, so the model
      spends more capacity learning the rare/hard classes
    - Originally proposed in "Focal Loss for Dense Object Detection"
      (Lin et al., 2017) for handling extreme class imbalance

Compared against CrossEntropy on architectures from Parts A, B, and C.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loading import load_wine_quality, load_fashion_mnist, FASHION_CLASSES
from utils import (
    count_parameters, train_model, full_evaluation,
    plot_training_curves, plot_results_table, DEVICE
)

# Import models from other parts
from part_a_shallow_vs_deep import ShallowNet, DeepNet
from part_c_cnns import ShallowCNN, ResidualCNN


# ──────────────────────────────────────────────
# Custom Loss: Focal Loss
# ──────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: focusing parameter. Higher gamma = more focus on hard examples.
               gamma=0 is equivalent to standard CrossEntropy.
               gamma=2 is the default from the original paper.
        alpha: per-class weights (tensor of size num_classes), or None.
        reduction: 'mean' or 'sum'
    """
    takes_class_indices = True  # Flag for our training loop

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: raw logits, shape (batch_size, num_classes)
            targets: class indices, shape (batch_size,)
        """
        # Compute softmax probabilities
        p = F.softmax(inputs, dim=1)

        # Gather the probability of the true class for each sample
        # p_t = p[i, targets[i]] for each sample i
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal weight: (1 - p_t)^gamma
        # When p_t is high (easy example), weight is small
        # When p_t is low (hard example), weight is large
        focal_weight = (1 - p_t) ** self.gamma

        # Standard cross-entropy term: -log(p_t)
        ce_loss = -torch.log(p_t + 1e-8)

        # Combine: focal_weight * ce_loss
        loss = focal_weight * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device).gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ──────────────────────────────────────────────
# Main Experiment
# ──────────────────────────────────────────────

def run_part_d():
    print("=" * 60)
    print("PART D: Custom Loss Function (Focal Loss)")
    print("=" * 60)

    # ── Experiment 1: On Wine Quality (tabular) ──
    print("\n--- Wine Quality Dataset ---")
    train_loader, val_loader, test_loader, num_features, num_classes = load_wine_quality()

    loss_functions = {
        "CrossEntropy": nn.CrossEntropyLoss(),
        "FocalLoss (gamma=2)": FocalLoss(gamma=2.0),
    }

    tabular_models = {
        "Shallow": (ShallowNet, {"input_dim": num_features, "num_classes": num_classes, "width": 300}),
        "Deep": (DeepNet, {"input_dim": num_features, "num_classes": num_classes, "width": 32, "depth": 7}),
    }

    wine_histories = {}
    wine_results = {}

    for loss_name, loss_fn in loss_functions.items():
        for model_name, (ModelClass, kwargs) in tabular_models.items():
            label = f"{model_name} + {loss_name}"
            print(f"\n--- Training: {label} ---")

            model = ModelClass(**kwargs)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Focal loss uses same interface as CrossEntropy (logits + class indices)
            history = train_model(
                model, train_loader, val_loader, loss_fn, optimizer,
                epochs=100, device=DEVICE
            )
            wine_histories[label] = history

            results = full_evaluation(model, test_loader, loss_fn, device=DEVICE)
            wine_results[label] = results
            print(f"  Test Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")

    # ── Experiment 2: On Fashion-MNIST (CNN) ──
    print("\n--- Fashion-MNIST Dataset ---")
    train_loader_img, val_loader_img, test_loader_img = load_fashion_mnist(
        batch_size=128, augment=True
    )

    cnn_models = {
        "Shallow CNN": (ShallowCNN, {}),
        "Residual CNN": (ResidualCNN, {}),
    }

    fmnist_histories = {}
    fmnist_results = {}

    for loss_name, loss_fn in loss_functions.items():
        for model_name, (ModelClass, kwargs) in cnn_models.items():
            label = f"{model_name} + {loss_name}"
            print(f"\n--- Training: {label} ---")

            model = ModelClass(**kwargs)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            history = train_model(
                model, train_loader_img, val_loader_img, loss_fn, optimizer,
                epochs=20, device=DEVICE
            )
            fmnist_histories[label] = history

            results = full_evaluation(
                model, test_loader_img, loss_fn, device=DEVICE,
                label_names=FASHION_CLASSES
            )
            fmnist_results[label] = results
            print(f"  Test Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")

    # Plots
    plot_training_curves(wine_histories,
                         title="Part D: Focal Loss vs CE (Wine Quality)",
                         filename="part_d_wine_curves.png")
    plot_results_table(wine_results,
                       title="Part D: Wine Quality Results",
                       filename="part_d_wine_results.png")

    plot_training_curves(fmnist_histories,
                         title="Part D: Focal Loss vs CE (Fashion-MNIST)",
                         filename="part_d_fmnist_curves.png")
    plot_results_table(fmnist_results,
                       title="Part D: Fashion-MNIST Results",
                       filename="part_d_fmnist_results.png")

    # Summary
    print("\n" + "=" * 60)
    print("PART D SUMMARY — Wine Quality")
    print("=" * 60)
    for label, res in wine_results.items():
        print(f"  {label:35s} | Acc: {res['accuracy']:.4f} | F1: {res['f1']:.4f}")

    print("\nPART D SUMMARY — Fashion-MNIST")
    print("=" * 60)
    for label, res in fmnist_results.items():
        print(f"  {label:35s} | Acc: {res['accuracy']:.4f} | F1: {res['f1']:.4f}")

    return wine_histories, wine_results, fmnist_histories, fmnist_results


if __name__ == "__main__":
    run_part_d()
