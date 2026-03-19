"""
Assignment 4 - Part C: Mixture of Experts with Trainable Router
================================================================
Trains a gating network on top of frozen expert classifiers
from Assignment 3 (ResNet-like, Inception-like, SqueezeNet-like, SuperNet).
The router learns to weight each expert's output per input image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Data (same CIFAR-10 setup as Assignment 3) ──────────────────────────────

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

print(f"Training: {len(train_set)}, Test: {len(test_set)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERT DEFINITIONS (copied from Assignment 3)
# ══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
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


class ResNetLike(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.inc1 = InceptionBlock(32, 8, 12, 8, 4)
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.inc2 = InceptionBlock(64, 16, 24, 16, 8)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU())
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
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.fire1 = FireModule(32, 8, 16)
        self.fire2 = FireModule(32, 8, 16)
        self.down1 = nn.MaxPool2d(2)
        self.fire3 = FireModule(32, 16, 32)
        self.fire4 = FireModule(64, 16, 32)
        self.down2 = nn.MaxPool2d(2)
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
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.inception = InceptionBlock(32, 8, 12, 8, 4)
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
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
# STEP 1: Train each expert independently
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  Step 1: Training Expert Models (20 epochs each)")
print("=" * 60)

EXPERT_EPOCHS = 20


def train_expert(name, model):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EXPERT_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    start = time.time()

    for epoch in range(1, EXPERT_EPOCHS + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(1) == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        best_acc = max(best_acc, acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{name}] Epoch {epoch:2d}/{EXPERT_EPOCHS}  test_acc={acc:.4f}")

    elapsed = time.time() - start
    print(f"  [{name}] Best: {best_acc:.4f}  ({elapsed:.1f}s)")
    return model, best_acc


experts_info = [
    ("ResNet", ResNetLike()),
    ("Inception", InceptionLike()),
    ("SqueezeNet", SqueezeNetLike()),
    ("SuperNet", SuperNet()),
]

trained_experts = []
expert_accs = []
for name, model in experts_info:
    trained_model, acc = train_expert(name, model)
    trained_experts.append(trained_model)
    expert_accs.append(acc)
    print()

# Freeze all experts
for expert in trained_experts:
    for param in expert.parameters():
        param.requires_grad = False
    expert.eval()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Mixture of Experts with trainable router
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  Step 2: Training MoE Router/Gating Network")
print("=" * 60)

NUM_EXPERTS = len(trained_experts)
NUM_CLASSES = 10


class MoERouter(nn.Module):
    """Simple CNN-based gating network.
    Takes an image, outputs softmax weights over K experts.
    """
    def __init__(self, num_experts):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 32->16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 16->8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.gate = nn.Linear(32, num_experts)

    def forward(self, x):
        h = self.features(x)
        return F.softmax(self.gate(h), dim=1)


class MixtureOfExperts(nn.Module):
    def __init__(self, experts, router):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.router = router

    def forward(self, x):
        # Get gating weights: (batch, num_experts)
        weights = self.router(x)

        # Get each expert's logits: list of (batch, num_classes)
        expert_outputs = []
        for expert in self.experts:
            with torch.no_grad():
                expert_outputs.append(expert(x))

        # Stack: (batch, num_experts, num_classes)
        stacked = torch.stack(expert_outputs, dim=1)

        # Weighted combination: (batch, num_classes)
        # weights: (batch, num_experts, 1) * stacked: (batch, num_experts, num_classes)
        combined = (weights.unsqueeze(2) * stacked).sum(dim=1)

        return combined, weights


router = MoERouter(NUM_EXPERTS).to(device)
moe = MixtureOfExperts(trained_experts, router).to(device)

# Only train the router
moe_optimizer = torch.optim.Adam(router.parameters(), lr=1e-3)
moe_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(moe_optimizer, T_max=15)
criterion = nn.CrossEntropyLoss()

MOE_EPOCHS = 15
moe_accs = []

for epoch in range(1, MOE_EPOCHS + 1):
    router.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        combined, _ = moe(images)
        loss = criterion(combined, labels)
        moe_optimizer.zero_grad()
        loss.backward()
        moe_optimizer.step()
        total_loss += loss.item() * labels.size(0)
    moe_scheduler.step()

    # Evaluate
    router.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            combined, _ = moe(images)
            correct += (combined.argmax(1) == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    moe_accs.append(acc)

    avg_loss = total_loss / len(train_set)
    if epoch % 3 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d}/{MOE_EPOCHS}  loss={avg_loss:.4f}  test_acc={acc:.4f}")

best_moe_acc = max(moe_accs)
print(f"\n  Best MoE accuracy: {best_moe_acc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Analyse router behaviour
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  Step 3: Router Analysis")
print("=" * 60)

expert_names = [name for name, _ in experts_info]

# Collect gating weights per class
router.eval()
all_weights = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        _, weights = moe(images)
        all_weights.append(weights.cpu().numpy())
        all_labels.append(labels.numpy())

all_weights = np.concatenate(all_weights)
all_labels = np.concatenate(all_labels)

CLASSES = test_set.classes

# Average gating weights per class
print(f"\n  Average gating weights per class:")
print(f"  {'Class':<14s}", end="")
for name in expert_names:
    print(f"  {name:>10s}", end="")
print()
print(f"  {'-'*14}", end="")
for _ in expert_names:
    print(f"  {'-'*10}", end="")
print()

class_weights = np.zeros((10, NUM_EXPERTS))
for c in range(10):
    mask = all_labels == c
    avg_w = all_weights[mask].mean(axis=0)
    class_weights[c] = avg_w
    print(f"  {CLASSES[c]:<14s}", end="")
    for w in avg_w:
        print(f"  {w:>10.3f}", end="")
    print()

# ── Summary table ────────────────────────────────────────────────────────────
print("\n\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"  {'Model':<22s} {'Test Acc':>10s}")
print(f"  {'-'*22} {'-'*10}")
for (name, _), acc in zip(experts_info, expert_accs):
    print(f"  {name:<22s} {acc:>10.4f}")
print(f"  {'MoE (router)':<22s} {best_moe_acc:>10.4f}")

# ── Visualise gating weights ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(10)
width = 0.2
for i, name in enumerate(expert_names):
    ax.bar(x + i * width, class_weights[:, i], width, label=name)

ax.set_xlabel("CIFAR-10 Class")
ax.set_ylabel("Average Gate Weight")
ax.set_title("Router Gating Weights per Class")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(CLASSES, rotation=45, ha="right")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("embeddings-moe/part_c_gating_weights.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Saved -> embeddings-moe/part_c_gating_weights.png")
