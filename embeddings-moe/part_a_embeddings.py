"""
Assignment 4 - Part A: Learning Image Embeddings from Scratch
==============================================================
A1. Autoencoder-based embeddings on Fashion-MNIST
A2. Contrastive (Siamese) embeddings on Fashion-MNIST
Both with t-SNE visualisation of the learned embedding spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Data ─────────────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])

train_set = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

CLASS_NAMES = train_set.classes
print(f"Training: {len(train_set)}, Test: {len(test_set)}")
print(f"Classes: {CLASS_NAMES}\n")


# ══════════════════════════════════════════════════════════════════════════════
# A1: AUTOENCODER
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  A1: Autoencoder-Based Image Embeddings")
print("=" * 60)

LATENT_DIM = 16


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 28->14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 14->7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 7->14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 14->28
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


ae_model = Autoencoder(LATENT_DIM).to(device)
ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3)

# Need unnormalized data for reconstruction loss
transform_ae = transforms.Compose([transforms.ToTensor()])
train_set_ae = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_ae)
test_set_ae = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_ae)
train_loader_ae = DataLoader(train_set_ae, batch_size=256, shuffle=True, num_workers=2)
test_loader_ae = DataLoader(test_set_ae, batch_size=256, shuffle=False, num_workers=2)

AE_EPOCHS = 10
for epoch in range(1, AE_EPOCHS + 1):
    ae_model.train()
    total_loss = 0
    for images, _ in train_loader_ae:
        images = images.to(device)
        recon, _ = ae_model(images)
        loss = F.mse_loss(recon, images)
        ae_optimizer.zero_grad()
        loss.backward()
        ae_optimizer.step()
        total_loss += loss.item() * images.size(0)
    avg_loss = total_loss / len(train_set_ae)
    if epoch % 2 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d}/{AE_EPOCHS}  recon_loss={avg_loss:.6f}")

# Extract embeddings
print("\n  Extracting autoencoder embeddings ...")
ae_model.eval()
ae_embeddings, ae_labels = [], []
with torch.no_grad():
    for images, labels in test_loader_ae:
        images = images.to(device)
        _, z = ae_model(images)
        ae_embeddings.append(z.cpu().numpy())
        ae_labels.append(labels.numpy())
ae_embeddings = np.concatenate(ae_embeddings)
ae_labels = np.concatenate(ae_labels)

# Show some reconstructions
ae_model.eval()
sample_images, _ = next(iter(test_loader_ae))
sample_images = sample_images[:8].to(device)
with torch.no_grad():
    recon_images, _ = ae_model(sample_images)

fig, axes = plt.subplots(2, 8, figsize=(14, 3.5))
for i in range(8):
    axes[0, i].imshow(sample_images[i].cpu().squeeze(), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(recon_images[i].cpu().squeeze(), cmap="gray")
    axes[1, i].axis("off")
axes[0, 0].set_ylabel("Original", fontsize=10)
axes[1, 0].set_ylabel("Recon", fontsize=10)
plt.suptitle("Autoencoder Reconstructions", fontsize=12)
plt.tight_layout()
plt.savefig("embeddings-moe/ae_reconstructions.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved -> embeddings-moe/ae_reconstructions.png")


# ══════════════════════════════════════════════════════════════════════════════
# A2: CONTRASTIVE EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  A2: Contrastive Image Embeddings")
print("=" * 60)

EMBED_DIM = 16


class SiameseNet(nn.Module):
    def __init__(self, embed_dim=16):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 28->14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 14->7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, embed_dim),
        )

    def forward(self, x):
        return self.backbone(x)


class PairDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = dataset.data.numpy() if hasattr(dataset.data, 'numpy') else np.array(dataset.data)
        self.targets = np.array(dataset.targets)
        self.labels_set = set(self.targets)
        self.label_to_indices = {
            l: np.where(self.targets == l)[0] for l in self.labels_set
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img1 = self.data[idx]
        label1 = self.targets[idx]

        # 50% positive, 50% negative pairs
        if np.random.random() < 0.5:
            # Positive pair (same class)
            idx2 = np.random.choice(self.label_to_indices[label1])
            target = 1.0
        else:
            # Negative pair (different class)
            neg_label = np.random.choice(list(self.labels_set - {label1}))
            idx2 = np.random.choice(self.label_to_indices[neg_label])
            target = 0.0

        img2 = self.data[idx2]

        # Convert to tensor
        img1 = torch.FloatTensor(img1).unsqueeze(0) / 255.0
        img2 = torch.FloatTensor(img2).unsqueeze(0) / 255.0

        return img1, img2, torch.FloatTensor([target])


def contrastive_loss(emb1, emb2, target, margin=1.0):
    dist = F.pairwise_distance(emb1, emb2)
    pos_loss = target * dist.pow(2)
    neg_loss = (1 - target) * F.relu(margin - dist).pow(2)
    return (pos_loss + neg_loss).mean()


pair_train = PairDataset(train_set_ae)
pair_loader = DataLoader(pair_train, batch_size=256, shuffle=True, num_workers=2)

siamese = SiameseNet(EMBED_DIM).to(device)
siam_optimizer = torch.optim.Adam(siamese.parameters(), lr=1e-3)

SIAM_EPOCHS = 10
for epoch in range(1, SIAM_EPOCHS + 1):
    siamese.train()
    total_loss = 0
    count = 0
    for img1, img2, target in pair_loader:
        img1, img2, target = img1.to(device), img2.to(device), target.to(device)
        emb1 = siamese(img1)
        emb2 = siamese(img2)
        loss = contrastive_loss(emb1, emb2, target.squeeze())
        siam_optimizer.zero_grad()
        loss.backward()
        siam_optimizer.step()
        total_loss += loss.item() * img1.size(0)
        count += img1.size(0)
    avg_loss = total_loss / count
    if epoch % 2 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d}/{SIAM_EPOCHS}  contrastive_loss={avg_loss:.6f}")

# Extract contrastive embeddings
print("\n  Extracting contrastive embeddings ...")
siamese.eval()
con_embeddings, con_labels = [], []
with torch.no_grad():
    for images, labels in test_loader_ae:
        images = images.to(device)
        emb = siamese(images)
        con_embeddings.append(emb.cpu().numpy())
        con_labels.append(labels.numpy())
con_embeddings = np.concatenate(con_embeddings)
con_labels = np.concatenate(con_labels)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION: t-SNE for both embedding spaces
# ══════════════════════════════════════════════════════════════════════════════

print("\n  Computing t-SNE projections ...")

# Use a subset for t-SNE (faster)
N_VIS = 3000
idx = np.random.choice(len(ae_labels), N_VIS, replace=False)

tsne_ae = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(ae_embeddings[idx])
tsne_con = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(con_embeddings[idx])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for c in range(10):
    mask = ae_labels[idx] == c
    axes[0].scatter(tsne_ae[mask, 0], tsne_ae[mask, 1], s=5, alpha=0.6, label=CLASS_NAMES[c])
axes[0].set_title("Autoencoder Embeddings (t-SNE)")
axes[0].legend(fontsize=7, markerscale=3, loc="best")
axes[0].set_xticks([])
axes[0].set_yticks([])

for c in range(10):
    mask = con_labels[idx] == c
    axes[1].scatter(tsne_con[mask, 0], tsne_con[mask, 1], s=5, alpha=0.6, label=CLASS_NAMES[c])
axes[1].set_title("Contrastive Embeddings (t-SNE)")
axes[1].legend(fontsize=7, markerscale=3, loc="best")
axes[1].set_xticks([])
axes[1].set_yticks([])

plt.tight_layout()
plt.savefig("embeddings-moe/part_a_embeddings_tsne.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved -> embeddings-moe/part_a_embeddings_tsne.png")

# Save embeddings for Part B comparison
torch.save({
    "ae_embeddings": ae_embeddings, "ae_labels": ae_labels,
    "con_embeddings": con_embeddings, "con_labels": con_labels,
}, "embeddings-moe/part_a_embeddings.pt")
print("  Saved embeddings -> embeddings-moe/part_a_embeddings.pt")
