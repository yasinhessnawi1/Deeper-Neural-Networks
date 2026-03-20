"""
Assignment 6 - Comparison of All Generative Models
====================================================
Loads saved samples from DAE, VAE, GAN, and Diffusion,
and displays them side by side.
"""

import torch
import matplotlib.pyplot as plt

print("Loading saved samples from all models ...")

samples_dae = torch.load("generative-ai/samples_dae.pt", weights_only=True)
samples_vae = torch.load("generative-ai/samples_vae.pt", weights_only=True)
samples_gan = torch.load("generative-ai/samples_gan.pt", weights_only=True)
samples_diff = torch.load("generative-ai/samples_diffusion.pt", weights_only=True)

# Clamp all to [0,1]
samples_dae = torch.clamp(samples_dae, 0, 1)
samples_vae = torch.clamp(samples_vae, 0, 1)
samples_gan = torch.clamp(samples_gan, 0, 1)
samples_diff = torch.clamp(samples_diff, 0, 1)

fig, axes = plt.subplots(4, 10, figsize=(16, 6.5))
labels = ["Denoising AE", "VAE", "GAN", "Diffusion"]
all_samples = [samples_dae, samples_vae, samples_gan, samples_diff]

for row in range(4):
    for col in range(10):
        axes[row, col].imshow(all_samples[row][col].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(labels[row], fontsize=10, rotation=0, labelpad=70, va="center")

plt.suptitle("Comparison of Generated Samples (Fashion-MNIST)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("generative-ai/comparison_all_models.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved -> generative-ai/comparison_all_models.png")
print("Done.")
