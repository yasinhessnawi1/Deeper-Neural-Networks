"""
Assignment 6 - Part C: Generative Adversarial Network (GAN)
=============================================================
Basic DCGAN-style GAN on Fashion-MNIST.
Generator G(z) -> image, Discriminator D(x) -> real/fake.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Data ─────────────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # scale to [-1, 1]
])
train_set = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
print(f"Train: {len(train_set)}\n")

LATENT_DIM = 64


# ── Generator ────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 7->14
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1),  # 14->28
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


# ── Discriminator ────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),  # 28->14
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 14->7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ── Training ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Training GAN")
print("=" * 60)

G = Generator(LATENT_DIM).to(device)
D = Discriminator().to(device)

opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()

EPOCHS = 30
fixed_z = torch.randn(20, LATENT_DIM).to(device)

for epoch in range(1, EPOCHS + 1):
    G.train()
    D.train()
    d_loss_sum, g_loss_sum, batches = 0, 0, 0

    for real_imgs, _ in train_loader:
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)

        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # Train Discriminator
        z = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_imgs = G(z).detach()

        d_real = D(real_imgs)
        d_fake = D(fake_imgs)
        d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # Train Generator
        z = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_imgs = G(z)
        g_loss = criterion(D(fake_imgs), real_labels)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        d_loss_sum += d_loss.item()
        g_loss_sum += g_loss.item()
        batches += 1

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d}/{EPOCHS}  D_loss={d_loss_sum/batches:.4f}  "
              f"G_loss={g_loss_sum/batches:.4f}")

# ── Generate samples ─────────────────────────────────────────────────────────

print("\nGenerating samples ...")
G.eval()
with torch.no_grad():
    generated = G(fixed_z)

# Rescale from [-1,1] to [0,1]
generated = (generated + 1) / 2

fig, axes = plt.subplots(2, 10, figsize=(16, 3.5))
for i in range(10):
    axes[0, i].imshow(generated[i].cpu().squeeze(), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(generated[i + 10].cpu().squeeze(), cmap="gray")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Row 1", fontsize=9, rotation=0, labelpad=40, va="center")
axes[1, 0].set_ylabel("Row 2", fontsize=9, rotation=0, labelpad=40, va="center")

plt.suptitle("Part C: GAN Generated Samples", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("generative-ai/part_c_gan.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved -> generative-ai/part_c_gan.png")

# Save for comparison
torch.save(generated[:10].cpu(), "generative-ai/samples_gan.pt")
print("Done.")
