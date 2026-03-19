"""
Assignment 3 - Part A: Image Classification with Residual + Inception + Fire
=============================================================================
Trains and evaluates five models on CIFAR-10:
  1. Plain CNN baseline
  2. ResNet-like model (residual blocks)
  3. Inception-like model (inception blocks)
  4. SqueezeNet-like model (fire modules)
  5. SuperNet (all three modules combined)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time

torch.manual_seed(42)
np.random.seed(42)

# ── Data ─────────────────────────────────────────────────────────────────────

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

CLASSES = train_set.classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Training samples: {len(train_set)}, Test samples: {len(test_set)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Conv-BN-ReLU -> Conv-BN + skip connection."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class InceptionBlock(nn.Module):
    """Parallel 1x1, 3x3, 5x5, and pooling branches -> concat."""
    def __init__(self, in_ch, out_1x1, out_3x3, out_5x5, out_pool):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_1x1, 1, bias=False), nn.BatchNorm2d(out_1x1), nn.ReLU())
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, out_3x3, 3, padding=1, bias=False), nn.BatchNorm2d(out_3x3), nn.ReLU())
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_ch, out_5x5, 5, padding=2, bias=False), nn.BatchNorm2d(out_5x5), nn.ReLU())
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, out_pool, 1, bias=False), nn.BatchNorm2d(out_pool), nn.ReLU())

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch3(x),
                          self.branch5(x), self.branch_pool(x)], dim=1)


class FireModule(nn.Module):
    """Squeeze 1x1 -> expand 1x1 and 3x3 -> concat."""
    def __init__(self, in_ch, squeeze, expand):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_ch, squeeze, 1, bias=False), nn.BatchNorm2d(squeeze), nn.ReLU())
        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze, expand, 1, bias=False), nn.BatchNorm2d(expand), nn.ReLU())
        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze, expand, 3, padding=1, bias=False), nn.BatchNorm2d(expand), nn.ReLU())

    def forward(self, x):
        s = self.squeeze(x)
        return torch.cat([self.expand1x1(s), self.expand3x3(s)], dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class PlainCNN(nn.Module):
    """Simple 6-layer CNN baseline."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class ResNetLike(nn.Module):
    """Small ResNet: stem -> 3 stages of residual blocks."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.stage1 = nn.Sequential(ResidualBlock(32), ResidualBlock(32))
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.stage2 = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.stage3 = nn.Sequential(ResidualBlock(128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(self.stage1(x))
        x = self.down2(self.stage2(x))
        x = self.stage3(x)
        return self.fc(self.pool(x).flatten(1))


class InceptionLike(nn.Module):
    """Small Inception-like model with 3 inception blocks."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        # InceptionBlock outputs: 8+12+8+4 = 32 channels
        self.inc1 = InceptionBlock(32, 8, 12, 8, 4)
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        # 16+24+16+8 = 64
        self.inc2 = InceptionBlock(64, 16, 24, 16, 8)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU())
        # 32+48+32+16 = 128
        self.inc3 = InceptionBlock(128, 32, 48, 32, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(self.inc1(x))
        x = self.down2(self.inc2(x))
        x = self.inc3(x)
        return self.fc(self.pool(x).flatten(1))


class SqueezeNetLike(nn.Module):
    """Small SqueezeNet-like model with fire modules."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        # Fire(32, squeeze=8, expand=16) -> output 32 ch
        self.fire1 = FireModule(32, 8, 16)
        self.fire2 = FireModule(32, 8, 16)
        self.down1 = nn.MaxPool2d(2)
        # Fire(32, squeeze=16, expand=32) -> output 64 ch
        self.fire3 = FireModule(32, 16, 32)
        self.fire4 = FireModule(64, 16, 32)
        self.down2 = nn.MaxPool2d(2)
        # Fire(64, squeeze=32, expand=64) -> output 128 ch
        self.fire5 = FireModule(64, 32, 64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.fire1(x)
        x = self.down1(self.fire2(x))
        x = self.fire3(x)
        x = self.down2(self.fire4(x))
        x = self.fire5(x)
        return self.fc(self.pool(x).flatten(1))


class SuperNet(nn.Module):
    """Hybrid network combining Residual, Inception, and Fire modules.

    Architecture:
      Stem (3->32) -> InceptionBlock (multi-scale features)
      -> downsample -> ResidualBlock x2 (deep feature refinement)
      -> downsample -> FireModule (parameter-efficient final features)
      -> global pool -> classifier

    Design rationale:
      - Inception at the start captures multi-scale patterns early
      - Residual blocks in the middle enable deep feature learning with skip connections
      - Fire module at the end compresses features efficiently (fewer params)
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        # Inception: multi-scale early features (8+12+8+4=32 ch)
        self.inception = InceptionBlock(32, 8, 12, 8, 4)
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        # Residual: deep refinement
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        # Fire: efficient final features (squeeze=16, expand=64 -> 128 ch)
        self.fire = FireModule(64, 16, 64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception(x)
        x = self.down1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.down2(x)
        x = self.fire(x)
        return self.fc(self.pool(x).flatten(1))


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_model(name, model, epochs=20):
    model = model.to(device)
    params = count_params(model)
    print(f"\n{'='*60}")
    print(f"  {name}  ({params:,} parameters)")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_losses, train_accs, test_accs = [], [], []
    best_acc = 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        train_losses.append(loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        best_acc = max(best_acc, test_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{epochs}  loss={loss:.4f}  "
                  f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    elapsed = time.time() - start
    print(f"  Best test accuracy: {best_acc:.4f}  ({elapsed:.1f}s)")
    return {
        "name": name, "params": params, "best_acc": best_acc,
        "train_losses": train_losses, "train_accs": train_accs, "test_accs": test_accs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

EPOCHS = 20

models = [
    ("Plain CNN", PlainCNN()),
    ("ResNet-like", ResNetLike()),
    ("Inception-like", InceptionLike()),
    ("SqueezeNet-like", SqueezeNetLike()),
    ("SuperNet (hybrid)", SuperNet()),
]

results = []
for name, model in models:
    r = train_model(name, model, epochs=EPOCHS)
    results.append(r)

# ── Summary table ────────────────────────────────────────────────────────────
print("\n\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  {'Model':<22s} {'Params':>10s} {'Best Test Acc':>14s}")
print(f"  {'-'*22} {'-'*10} {'-'*14}")
for r in results:
    print(f"  {r['name']:<22s} {r['params']:>10,} {r['best_acc']:>14.4f}")

# ── Plots ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for r in results:
    axes[0].plot(r["train_losses"], label=r["name"])
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Training Loss")
axes[0].set_title("Training Loss Curves")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

for r in results:
    axes[1].plot(r["test_accs"], label=r["name"])
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Test Accuracy")
axes[1].set_title("Test Accuracy Curves")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("modern-architectures/part_a_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure saved -> modern-architectures/part_a_training_curves.png")
