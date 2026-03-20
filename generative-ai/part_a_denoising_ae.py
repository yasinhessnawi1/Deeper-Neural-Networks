"""
Assignment 6 - Part A: Denoising Autoencoder
=============================================
Trains an autoencoder on Fashion-MNIST, adds noise/labels
to images, and reconstructs clean versions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Data ─────────────────────────────────────────────────────────────────────

transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

CLASS_NAMES = train_set.classes
print(f"Train: {len(train_set)}, Test: {len(test_set)}\n")


# ── Noise functions ──────────────────────────────────────────────────────────

def add_gaussian_noise(images, noise_factor=0.4):
    noisy = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy, 0.0, 1.0)


def add_label_overlay(images, labels):
    """Stamp a small text label on top of the image."""
    noisy = images.clone()
    for i in range(images.size(0)):
        # Draw a white rectangle + class index at top-left
        noisy[i, :, 0:6, 0:10] = 1.0  # white patch
        # Simple: draw the digit as a pattern
        digit = labels[i].item()
        # Draw horizontal lines based on digit value
        row = 1
        for bit in range(4):
            if digit & (1 << bit):
                noisy[i, :, row, 1:9] = 0.0
            row += 1
    return noisy


# ── Model ────────────────────────────────────────────────────────────────────

class DenoisingAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 28->14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14->7
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 7->14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),  # 14->28
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── Training ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Training Denoising Autoencoder (Gaussian noise)")
print("=" * 60)

model_noise = DenoisingAE().to(device)
optimizer = torch.optim.Adam(model_noise.parameters(), lr=1e-3)

EPOCHS = 15
for epoch in range(1, EPOCHS + 1):
    model_noise.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        noisy = add_gaussian_noise(images, 0.4)
        recon = model_noise(noisy)
        loss = F.mse_loss(recon, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={total_loss/len(train_set):.6f}")

print("\nTraining Denoising AE (label overlay) ...")

model_label = DenoisingAE().to(device)
optimizer2 = torch.optim.Adam(model_label.parameters(), lr=1e-3)

for epoch in range(1, EPOCHS + 1):
    model_label.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        noisy = add_label_overlay(images, labels)
        recon = model_label(noisy)
        loss = F.mse_loss(recon, images)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        total_loss += loss.item() * images.size(0)
    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={total_loss/len(train_set):.6f}")

# ── Generate and visualise ───────────────────────────────────────────────────

print("\nGenerating samples ...")
model_noise.eval()
model_label.eval()

test_imgs, test_labels = next(iter(test_loader))
test_imgs = test_imgs[:10].to(device)
test_labels_batch = test_labels[:10]

noisy_gauss = add_gaussian_noise(test_imgs, 0.4)
noisy_label = add_label_overlay(test_imgs, test_labels_batch)

with torch.no_grad():
    recon_gauss = model_noise(noisy_gauss)
    recon_label = model_label(noisy_label)

fig, axes = plt.subplots(4, 10, figsize=(16, 6.5))
titles = ["Original", "Gaussian Noisy", "Denoised (Gauss)", "Label Overlay -> Denoised"]
data = [test_imgs, noisy_gauss, recon_gauss, recon_label]

for row in range(4):
    for col in range(10):
        axes[row, col].imshow(data[row][col].cpu().squeeze(), cmap="gray")
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(titles[row], fontsize=9, rotation=0, labelpad=80, va="center")

plt.suptitle("Part A: Denoising Autoencoder", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("generative-ai/part_a_denoising.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved -> generative-ai/part_a_denoising.png")

# Save generated samples for final comparison
torch.save(recon_gauss.cpu(), "generative-ai/samples_dae.pt")
print("Done.")
