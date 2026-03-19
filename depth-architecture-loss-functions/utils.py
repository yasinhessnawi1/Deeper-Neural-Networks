"""
Shared utilities: training loop, evaluation, plotting.
Used across all parts of the assignment.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
import time

# Create output directory for plots
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _takes_class_indices(loss_fn):
    """Check if a loss function takes class indices (not one-hot vectors)."""
    # CrossEntropyLoss and any custom loss with 'takes_class_indices' attribute
    return (isinstance(loss_fn, nn.CrossEntropyLoss) or
            getattr(loss_fn, "takes_class_indices", False))


def count_parameters(model):
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_loader, val_loader, loss_fn, optimizer,
                epochs=50, device=DEVICE, verbose=True):
    """
    Train a model and track loss + accuracy on train and validation sets.

    Returns:
        history: dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    model = model.to(device)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            # Handle different loss function input requirements
            # CrossEntropy and FocalLoss take class indices directly
            # MSE, MAE, Huber need one-hot encoded targets
            if _takes_class_indices(loss_fn):
                loss = loss_fn(outputs, y_batch)
            else:
                num_classes = outputs.shape[1]
                y_onehot = torch.zeros(y_batch.size(0), num_classes, device=device)
                y_onehot.scatter_(1, y_batch.unsqueeze(1), 1.0)
                loss = loss_fn(outputs, y_onehot)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)

        # --- Validation phase ---
        val_loss, val_acc = evaluate_model(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    return history


def evaluate_model(model, data_loader, loss_fn, device=DEVICE):
    """Evaluate model on a dataset. Returns (loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)

            if _takes_class_indices(loss_fn):
                loss = loss_fn(outputs, y_batch)
            else:
                num_classes = outputs.shape[1]
                y_onehot = torch.zeros(y_batch.size(0), num_classes, device=device)
                y_onehot.scatter_(1, y_batch.unsqueeze(1), 1.0)
                loss = loss_fn(outputs, y_onehot)

            running_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = running_loss / len(data_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def full_evaluation(model, data_loader, loss_fn, device=DEVICE, label_names=None):
    """Full evaluation with accuracy, F1, and classification report."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(all_labels, all_preds,
                                   target_names=label_names, zero_division=0)
    return {"accuracy": acc, "f1": f1, "report": report}


def plot_training_curves(histories, title="Training Curves", filename=None):
    """
    Plot loss and accuracy curves for multiple experiments.

    Args:
        histories: dict of {label: history_dict}
        title: plot title
        filename: if given, save to PLOT_DIR/filename
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    for label, h in histories.items():
        axes[0].plot(h["train_loss"], label=f"{label} (train)", linestyle="-")
        axes[0].plot(h["val_loss"], label=f"{label} (val)", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    for label, h in histories.items():
        axes[1].plot(h["train_acc"], label=f"{label} (train)", linestyle="-")
        axes[1].plot(h["val_acc"], label=f"{label} (val)", linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        path = os.path.join(PLOT_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {path}")
    plt.close()


def plot_results_table(results, title="Results Summary", filename=None):
    """
    Create a bar chart comparing accuracy/F1 across experiments.

    Args:
        results: dict of {label: {"accuracy": float, "f1": float}}
    """
    labels = list(results.keys())
    accs = [results[l]["accuracy"] for l in labels]
    f1s = [results[l]["f1"] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 5))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="steelblue")
    bars2 = ax.bar(x + width/2, f1s, width, label="F1-score", color="coral")

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    if filename:
        path = os.path.join(PLOT_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {path}")
    plt.close()
