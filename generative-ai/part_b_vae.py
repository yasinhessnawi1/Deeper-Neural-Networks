"""
Assignment 6 - Part B: Variational Autoencoder (VAE)
=====================================================
Implements a VAE on Fashion-MNIST with mu/logvar encoder,
reparameterization trick, and sample generation.
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
print(f"Train: {len(train_set)}, Test: {len(test_set)}\n")

LATENT_DIM = 32


# ── Model ────────────────────────────────────────────────────────────────────

class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        # Encoder
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 28->14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14->7
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 64 * 7 * 7)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 7->14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),  # 14->28
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc_conv(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc_dec(z))
        h = h.view(-1, 64, 7, 7)
        return self.dec_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kl_loss) / x.size(0)


# ── Training ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Training VAE")
print("=" * 60)

model = VAE(LATENT_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 20
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for images, _ in train_loader:
        images = images.to(device)
        recon, mu, logvar = model(images)
        loss = vae_loss(recon, images, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={total_loss/len(train_set):.2f}")

# ── Generate samples ─────────────────────────────────────────────────────────

print("\nGenerating samples ...")
model.eval()

# 1. Random samples from latent space
z_random = torch.randn(20, LATENT_DIM).to(device)
with torch.no_grad():
    generated = model.decode(z_random)

# 2. Reconstructions
test_imgs, _ = next(iter(test_loader))
test_imgs = test_imgs[:10].to(device)
with torch.no_grad():
    recon, _, _ = model(test_imgs)

# Plot
fig, axes = plt.subplots(4, 10, figsize=(16, 6.5))

for i in range(10):
    axes[0, i].imshow(test_imgs[i].cpu().squeeze(), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(recon[i].cpu().squeeze(), cmap="gray")
    axes[1, i].axis("off")
    axes[2, i].imshow(generated[i].cpu().squeeze(), cmap="gray")
    axes[2, i].axis("off")
    axes[3, i].imshow(generated[i + 10].cpu().squeeze(), cmap="gray")
    axes[3, i].axis("off")

axes[0, 0].set_ylabel("Original", fontsize=9, rotation=0, labelpad=60, va="center")
axes[1, 0].set_ylabel("Reconstructed", fontsize=9, rotation=0, labelpad=60, va="center")
axes[2, 0].set_ylabel("Generated (1)", fontsize=9, rotation=0, labelpad=60, va="center")
axes[3, 0].set_ylabel("Generated (2)", fontsize=9, rotation=0, labelpad=60, va="center")

plt.suptitle("Part B: VAE Samples", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("generative-ai/part_b_vae.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved -> generative-ai/part_b_vae.png")

# Save for comparison
torch.save(generated[:10].cpu(), "generative-ai/samples_vae.pt")
print("Done.")
