"""
Assignment 8 - Part D: Self-supervised Learning (Contrastive)
==============================================================
Graph augmentation -> two views -> InfoNCE loss -> learn embeddings
-> evaluate with linear probe + k-NN clustering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import sys
sys.path.insert(0, ".")

from data_utils import prepare_all, build_knn_edges

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load data ────────────────────────────────────────────────────────────────

df, label_map, img_embeds, text_embeds, labels = prepare_all(subset_size=5000)
N = len(labels)
NUM_CLASSES = len(label_map)
EMBED_DIM = 64
CLASSES = sorted(label_map.keys())

fused_embeds = F.normalize(torch.cat([img_embeds, text_embeds], dim=1), dim=1)
base_edges = build_knn_edges(fused_embeds, k=5)

fused_d = fused_embeds.to(device)
base_edges_d = base_edges.to(device)
labels_d = labels.to(device)


# ── Graph augmentations ──────────────────────────────────────────────────────

def edge_dropout(edge_index, drop_rate=0.2):
    """Randomly drop edges."""
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) > drop_rate
    return edge_index[:, mask]


def feature_masking(x, mask_rate=0.2):
    """Randomly mask feature dimensions."""
    mask = torch.rand(x.size(1), device=x.device) > mask_rate
    return x * mask.float().unsqueeze(0)


def feature_jitter(x, noise_std=0.1):
    """Add Gaussian noise to features."""
    return x + torch.randn_like(x) * noise_std


def augment_graph(x, edge_index):
    """Apply random augmentations to create a graph view."""
    x_aug = feature_masking(x, mask_rate=0.2)
    x_aug = feature_jitter(x_aug, noise_std=0.05)
    edge_aug = edge_dropout(edge_index, drop_rate=0.2)
    return x_aug, edge_aug.to(x.device)


# ── Model ────────────────────────────────────────────────────────────────────

class MessagePassingLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_msg = nn.Linear(in_dim, out_dim, bias=False)
        self.W_upd = nn.Linear(in_dim + out_dim, out_dim)

    def forward(self, h, edge_index):
        num_nodes = h.size(0)
        src, dst = edge_index[0], edge_index[1]
        messages = self.W_msg(h[src])
        agg = torch.zeros(num_nodes, messages.size(1), device=h.device)
        agg.index_add_(0, dst, messages)
        return F.relu(self.W_upd(torch.cat([h, agg], dim=1)))


class ContrastiveGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, proj_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.mp = MessagePassingLayer(hidden_dim, hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x, edge_index):
        h = self.encoder(x)
        h = self.mp(h, edge_index)
        return h, self.projector(h)

    def get_embeddings(self, x, edge_index):
        h = self.encoder(x)
        h = self.mp(h, edge_index)
        return h


def info_nce_loss(z1, z2, temperature=0.5):
    """InfoNCE loss between two sets of normalized embeddings."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Use a random subset for tractability
    if z1.size(0) > 2048:
        idx = torch.randperm(z1.size(0))[:2048]
        z1 = z1[idx]
        z2 = z2[idx]

    N = z1.size(0)
    # Positive pairs: (z1_i, z2_i)
    sim_pos = (z1 * z2).sum(dim=1) / temperature  # (N,)

    # Negative pairs: all cross-combinations
    sim_matrix = torch.mm(z1, z2.t()) / temperature  # (N, N)

    # InfoNCE: -log(exp(pos) / sum(exp(all)))
    labels = torch.arange(N, device=z1.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss


# ── Contrastive pretraining ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Contrastive Pretraining (InfoNCE)")
print("=" * 60)

model = ContrastiveGNN(128, EMBED_DIM, proj_dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

PRETRAIN_EPOCHS = 100
losses = []

for epoch in range(1, PRETRAIN_EPOCHS + 1):
    model.train()

    # Create two augmented views
    x1, e1 = augment_graph(fused_d, base_edges_d)
    x2, e2 = augment_graph(fused_d, base_edges_d)

    _, z1 = model(x1, e1)
    _, z2 = model(x2, e2)

    loss = info_nce_loss(z1, z2, temperature=0.5)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if epoch % 20 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{PRETRAIN_EPOCHS}  InfoNCE_loss={loss.item():.4f}")


# ── Extract learned embeddings ───────────────────────────────────────────────

print("\n  Extracting learned embeddings ...")
model.eval()
with torch.no_grad():
    learned_embeds = model.get_embeddings(fused_d, base_edges_d)
    learned_embeds = F.normalize(learned_embeds, dim=1).cpu()

print(f"  Learned embeddings shape: {learned_embeds.shape}")

# ── Linear probe evaluation ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Linear Probe Evaluation")
print("=" * 60)

train_idx, test_idx = train_test_split(
    np.arange(N), test_size=0.2, random_state=42, stratify=labels.numpy()
)

# Train linear classifier on frozen embeddings
linear_clf = nn.Linear(EMBED_DIM, NUM_CLASSES).to(device)
linear_opt = torch.optim.Adam(linear_clf.parameters(), lr=1e-2)

embeds_d = learned_embeds.to(device)
train_mask = torch.zeros(N, dtype=torch.bool)
test_mask = torch.zeros(N, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

for epoch in range(1, 51):
    linear_clf.train()
    logits = linear_clf(embeds_d)
    loss = F.cross_entropy(logits[train_mask], labels_d[train_mask])
    linear_opt.zero_grad()
    loss.backward()
    linear_opt.step()

linear_clf.eval()
with torch.no_grad():
    logits = linear_clf(embeds_d)
    linear_acc = (logits[test_mask].argmax(1) == labels_d[test_mask]).float().mean().item()
print(f"  Linear probe accuracy: {linear_acc:.4f}")

# ── k-NN evaluation ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  k-NN Clustering Evaluation")
print("=" * 60)

embeds_np = learned_embeds.numpy()
labels_np = labels.numpy()

for k in [3, 5, 10]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(embeds_np[train_idx], labels_np[train_idx])
    knn_acc = knn.score(embeds_np[test_idx], labels_np[test_idx])
    print(f"  k={k:2d}  kNN accuracy: {knn_acc:.4f}")

# ── t-SNE visualisation ─────────────────────────────────────────────────────

print("\n  Computing t-SNE ...")
N_VIS = 2000
vis_idx = np.random.choice(N, N_VIS, replace=False)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
coords = tsne.fit_transform(embeds_np[vis_idx])

fig, ax = plt.subplots(figsize=(8, 6))
for c in range(NUM_CLASSES):
    mask = labels_np[vis_idx] == c
    ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6, label=CLASSES[c])
ax.set_title("Part D: Contrastive GNN Embeddings (t-SNE)")
ax.legend(fontsize=8, markerscale=3)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig("multimodal-gnn/part_d_tsne.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved -> multimodal-gnn/part_d_tsne.png")

# ── Pretraining loss curve ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses)
ax.set_xlabel("Epoch")
ax.set_ylabel("InfoNCE Loss")
ax.set_title("Contrastive Pretraining Loss")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("multimodal-gnn/part_d_loss.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved -> multimodal-gnn/part_d_loss.png")

torch.save({
    "linear_acc": linear_acc, "learned_embeds": learned_embeds,
}, "multimodal-gnn/part_d_results.pt")
print("\nDone.")
