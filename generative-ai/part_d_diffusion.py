"""
Assignment 6 - Part D: Diffusion Model (DDPM)
===============================================
Implements a small DDPM on Fashion-MNIST:
  - Forward diffusion with linear noise schedule
  - Small U-Net-style denoiser with timestep embedding
  - Reverse sampling with intermediate steps saved
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
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
print(f"Train: {len(train_set)}\n")


# ── Noise schedule ───────────────────────────────────────────────────────────

T = 300  # total timesteps
beta_start = 1e-4
beta_end = 0.02

betas = torch.linspace(beta_start, beta_end, T).to(device)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)


def forward_diffusion(x0, t, noise=None):
    """Add noise to x0 at timestep t: x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps"""
    if noise is None:
        noise = torch.randn_like(x0)
    ab = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise, noise


# ── Show forward diffusion process ──────────────────────────────────────────

print("=" * 60)
print("  Forward Diffusion Process")
print("=" * 60)

sample_img = train_set[0][0].unsqueeze(0).to(device)
timesteps_show = [0, 25, 50, 100, 150, 200, 250, 299]

fig, axes = plt.subplots(1, len(timesteps_show), figsize=(16, 2.5))
fig.suptitle("Forward Diffusion: x_t for increasing t", fontsize=12, fontweight="bold")

for i, t_val in enumerate(timesteps_show):
    t_tensor = torch.tensor([t_val], device=device)
    noisy, _ = forward_diffusion(sample_img, t_tensor)
    axes[i].imshow(noisy[0].cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[i].set_title(f"t={t_val}", fontsize=9)
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("generative-ai/part_d_forward_diffusion.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved -> generative-ai/part_d_forward_diffusion.png")


# ── Denoising model (small U-Net-style CNN with timestep embedding) ─────────

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = np.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.SiLU(),
        )
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))

    def forward(self, x, t_emb):
        h = self.conv(x)
        t = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        return h + t


class SmallUNet(nn.Module):
    """Tiny U-Net: down(28->14->7) -> bottleneck -> up(7->14->28)."""
    def __init__(self, time_dim=64):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        # Down
        self.down1 = ConvBlock(1, 32, time_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(32, 64, time_dim)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bot = ConvBlock(64, 128, time_dim)

        # Up
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64, time_dim)  # 64+64 from skip
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32, time_dim)  # 32+32 from skip

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x, t):
        t_emb = self.time_emb(t)

        # Down
        h1 = self.down1(x, t_emb)       # (B,32,28,28)
        h2 = self.down2(self.pool1(h1), t_emb)  # (B,64,14,14)

        # Bottleneck
        hb = self.bot(self.pool2(h2), t_emb)  # (B,128,7,7)

        # Up
        u2 = self.up2(hb)               # (B,64,14,14)
        u2 = self.dec2(torch.cat([u2, h2], dim=1), t_emb)
        u1 = self.up1(u2)               # (B,32,28,28)
        u1 = self.dec1(torch.cat([u1, h1], dim=1), t_emb)

        return self.out(u1)


# ── Training ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Training Denoising Model")
print("=" * 60)

model = SmallUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

EPOCHS = 20
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    count = 0

    for images, _ in train_loader:
        images = images.to(device)
        batch_size = images.size(0)

        # Random timesteps
        t = torch.randint(0, T, (batch_size,), device=device)

        # Forward diffusion
        noisy, noise = forward_diffusion(images, t)

        # Predict noise
        pred_noise = model(noisy, t)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        count += batch_size

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={total_loss/count:.6f}")


# ── Sampling (reverse diffusion) ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Sampling (Reverse Diffusion)")
print("=" * 60)

@torch.no_grad()
def sample(model, n_samples=10, save_every=30):
    """DDPM sampling: start from noise, iteratively denoise."""
    model.eval()
    x = torch.randn(n_samples, 1, 28, 28, device=device)
    intermediates = [(T, x.cpu().clone())]

    for t_val in reversed(range(T)):
        t_batch = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
        pred_noise = model(x, t_batch)

        beta_t = betas[t_val]
        alpha_t = alphas[t_val]
        alpha_bar_t = alpha_bar[t_val]

        # DDPM update
        x = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise
        )

        if t_val > 0:
            noise = torch.randn_like(x)
            x = x + torch.sqrt(beta_t) * noise

        if t_val % save_every == 0 or t_val == 0:
            intermediates.append((t_val, x.cpu().clone()))

    return x, intermediates


generated, intermediates = sample(model, n_samples=10, save_every=30)
generated = torch.clamp(generated, 0, 1)

# Plot generation movie
n_steps = len(intermediates)
fig, axes = plt.subplots(n_steps, 10, figsize=(16, 2 * n_steps))
fig.suptitle("Part D: Reverse Diffusion (Sampling Process)", fontsize=13, fontweight="bold")

for row, (t_val, imgs) in enumerate(intermediates):
    imgs = torch.clamp(imgs, 0, 1)
    for col in range(10):
        axes[row, col].imshow(imgs[col].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(f"t={t_val}", fontsize=8, rotation=0, labelpad=30, va="center")

plt.tight_layout()
plt.savefig("generative-ai/part_d_sampling_process.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved -> generative-ai/part_d_sampling_process.png")

# Final generated samples
fig, axes = plt.subplots(1, 10, figsize=(16, 2))
for i in range(10):
    axes[i].imshow(generated[i].cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[i].axis("off")
plt.suptitle("Part D: Diffusion Model - Final Generated Samples", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("generative-ai/part_d_diffusion_samples.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved -> generative-ai/part_d_diffusion_samples.png")

# Save for comparison
torch.save(generated[:10].cpu(), "generative-ai/samples_diffusion.pt")
print("Done.")
