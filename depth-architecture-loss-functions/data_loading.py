"""
Data loading and preprocessing for Wine Quality (Parts A, B)
and Fashion-MNIST (Part C).
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# Wine Quality Dataset (tabular, multi-class)
# ──────────────────────────────────────────────

def load_wine_quality(batch_size=64, test_size=0.2, val_size=0.15, random_state=42):
    """
    Load Wine Quality (red + white combined) from UCI repository.

    Labels: quality scores 3–9, remapped to 0-based class indices.
    Split: 65% train, 15% val, 20% test (approximately).

    Returns:
        train_loader, val_loader, test_loader, num_features, num_classes
    """
    # Download both red and white wine datasets
    url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

    red = pd.read_csv(url_red, sep=";")
    white = pd.read_csv(url_white, sep=";")

    # Combine both datasets
    df = pd.concat([red, white], ignore_index=True)
    print(f"Wine Quality dataset: {len(df)} samples, {df.shape[1]-1} features")
    print(f"Class distribution:\n{df['quality'].value_counts().sort_index()}\n")

    # Features and labels
    X = df.drop("quality", axis=1).values.astype(np.float32)
    y = df["quality"].values

    # Remap labels to 0-based (quality 3 -> 0, quality 4 -> 1, etc.)
    unique_classes = sorted(np.unique(y))
    class_map = {c: i for i, c in enumerate(unique_classes)}
    y = np.array([class_map[val] for val in y], dtype=np.int64)
    num_classes = len(unique_classes)

    # Split: first into train+val and test, then train+val into train and val
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=relative_val_size,
        random_state=random_state, stratify=y_trainval
    )

    # Standardize features (fit on train only, transform all)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert to tensors and create data loaders
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    print(f"Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"Features: {X_train.shape[1]}, Classes: {num_classes}")

    return train_loader, val_loader, test_loader, X_train.shape[1], num_classes


# ──────────────────────────────────────────────
# Fashion-MNIST Dataset (images, 10 classes)
# ──────────────────────────────────────────────

FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def load_fashion_mnist(batch_size=128, augment=False):
    """
    Load Fashion-MNIST with normalization and optional augmentation.

    Images: 28x28 grayscale, 10 classes.
    Split: 50k train, 10k val (from original train), 10k test.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Normalization values for Fashion-MNIST
    mean, std = 0.2860, 0.3530

    # Training transforms (with optional augmentation)
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    # Download datasets
    full_train = torchvision.datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=train_transform
    )
    test_ds = torchvision.datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=test_transform
    )

    # Split training into train (50k) and validation (10k)
    train_ds, val_ds = torch.utils.data.random_split(
        full_train, [50000, 10000],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    print(f"Fashion-MNIST: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"Image size: 1x28x28, Classes: 10")

    return train_loader, val_loader, test_loader
