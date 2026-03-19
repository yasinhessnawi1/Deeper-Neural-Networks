"""
Part C: CNN Architectures on Fashion-MNIST.

Compares three architectures:
1. Shallow CNN — 3 conv layers, no skip connections
2. Deep CNN — 10 conv layers, no skip connections (plain)
3. Residual CNN — 10 conv layers, with skip connections

Each trained with two loss functions: CrossEntropy and MSE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loading import load_fashion_mnist, FASHION_CLASSES
from utils import (
    count_parameters, train_model, full_evaluation,
    plot_training_curves, plot_results_table, DEVICE
)


# ──────────────────────────────────────────────
# 1. Shallow CNN (3 conv layers)
# ──────────────────────────────────────────────

class ShallowCNN(nn.Module):
    """
    Simple CNN: Conv -> BN -> ReLU -> Pool, repeated 3 times, then FC.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # 1x28x28 -> 32x14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 32x14x14 -> 64x7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 64x7x7 -> 128x3x3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ──────────────────────────────────────────────
# 2. Deep CNN (10 conv layers, NO skip connections)
# ──────────────────────────────────────────────

class DeepCNN(nn.Module):
    """
    Deep plain CNN: 12 conv layers without any skip connections.
    No BatchNorm — this makes gradient flow harder and demonstrates
    the degradation problem that residual connections solve.
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # Initial conv to get to 64 channels
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 10 more conv layers at 64 channels (no BN, no skip connections)
        conv_layers = []
        for _ in range(10):
            conv_layers.extend([
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
            ])
        self.deep_convs = nn.Sequential(*conv_layers)

        # Final conv + pooling
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling -> 128x1x1
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.deep_convs(x)
        x = self.final_conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ──────────────────────────────────────────────
# 3. Residual CNN (10 conv layers, WITH skip connections)
# ──────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    A residual block: two conv layers with a skip connection.

        input ──> Conv -> BN -> ReLU -> Conv -> BN ──> (+) -> ReLU -> output
          │                                              │
          └──────────────── (identity) ──────────────────┘

    The skip connection adds the input directly to the output.
    This lets gradients flow through the shortcut during backprop.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x  # Save input for skip connection
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # Skip connection: add input back
        out = F.relu(out)
        return out


class ResidualCNN(nn.Module):
    """
    Residual CNN: same depth as DeepCNN (12 conv layers),
    but with skip connections every 2 layers.
    Should train much better than the plain DeepCNN.
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # Initial conv to get to 64 channels
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 5 residual blocks = 10 conv layers (same as DeepCNN's 10 middle layers)
        self.res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
        )

        # Final conv + pooling
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.res_blocks(x)
        x = self.final_conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ──────────────────────────────────────────────
# Main Experiment
# ──────────────────────────────────────────────

def run_part_c():
    print("=" * 60)
    print("PART C: CNN Architectures on Fashion-MNIST")
    print("=" * 60)

    train_loader, val_loader, test_loader = load_fashion_mnist(
        batch_size=128, augment=True
    )

    # Model definitions
    models = {
        "Shallow CNN": (ShallowCNN, {}),
        "Deep CNN (plain)": (DeepCNN, {}),
        "Residual CNN": (ResidualCNN, {}),
    }

    # Print parameter counts
    print("\nParameter counts:")
    for name, (ModelClass, kwargs) in models.items():
        m = ModelClass(**kwargs)
        print(f"  {name}: {count_parameters(m):,}")

    # Two loss functions
    loss_functions = {
        "CrossEntropy": nn.CrossEntropyLoss(),
        "MSE": nn.MSELoss(),
    }

    all_histories = {}
    all_results = {}

    for loss_name, loss_fn in loss_functions.items():
        for model_name, (ModelClass, kwargs) in models.items():
            label = f"{model_name} + {loss_name}"
            print(f"\n--- Training: {label} ---")

            model = ModelClass(**kwargs)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            history = train_model(
                model, train_loader, val_loader, loss_fn, optimizer,
                epochs=30, device=DEVICE
            )
            all_histories[label] = history

            results = full_evaluation(
                model, test_loader, loss_fn, device=DEVICE,
                label_names=FASHION_CLASSES
            )
            all_results[label] = results
            print(f"  Test Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")

    # Plot CrossEntropy curves
    ce_histories = {k: v for k, v in all_histories.items() if "CrossEntropy" in k}
    plot_training_curves(ce_histories,
                         title="Part C: CNNs with CrossEntropy",
                         filename="part_c_ce_curves.png")

    # Plot MSE curves
    mse_histories = {k: v for k, v in all_histories.items() if "MSE" in k}
    plot_training_curves(mse_histories,
                         title="Part C: CNNs with MSE",
                         filename="part_c_mse_curves.png")

    # Results comparison
    plot_results_table(all_results,
                       title="Part C: CNN Test Performance",
                       filename="part_c_results.png")

    # Summary
    print("\n" + "=" * 60)
    print("PART C SUMMARY")
    print("=" * 60)
    for label, res in all_results.items():
        print(f"  {label:35s} | Acc: {res['accuracy']:.4f} | F1: {res['f1']:.4f}")

    return all_histories, all_results


if __name__ == "__main__":
    run_part_c()
