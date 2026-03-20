"""
Assignment 8 - Part A: GNN on Bipartite Graphs
================================================
Image nodes + Text nodes with three edge types:
  - Image-Image (k-NN cosine)
  - Text-Text (k-NN cosine)
  - Image-Text (same product)
Message passing + pooling -> predict masterCategory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
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

# ── Build bipartite graph ────────────────────────────────────────────────────

print("\n[3] Building bipartite graph ...")

K = 5  # k-NN

# Image-Image edges (k-NN)
print("  Building Image-Image k-NN edges ...")
img_edges = build_knn_edges(img_embeds, k=K)
print(f"    {img_edges.size(1)} edges")

# Text-Text edges (k-NN)
print("  Building Text-Text k-NN edges ...")
text_edges = build_knn_edges(text_embeds, k=K)
# Offset text node indices by N (bipartite: nodes 0..N-1 are images, N..2N-1 are text)
text_edges_offset = text_edges + N
print(f"    {text_edges_offset.size(1)} edges")

# Image-Text edges (same product, bidirectional)
img_text_src = list(range(N)) + list(range(N, 2 * N))
img_text_dst = list(range(N, 2 * N)) + list(range(N))
img_text_edges = torch.tensor([img_text_src, img_text_dst], dtype=torch.long)
print(f"    Image-Text edges: {img_text_edges.size(1)}")

# Combine all edges
all_edges = torch.cat([img_edges, text_edges_offset, img_text_edges], dim=1)
print(f"  Total edges: {all_edges.size(1)}")

# Node features: stack image and text embeddings
# Nodes 0..N-1: image nodes, N..2N-1: text nodes
node_features = torch.cat([img_embeds, text_embeds], dim=0)  # (2N, 64)

# ── Train/test split ─────────────────────────────────────────────────────────

train_idx, test_idx = train_test_split(
    np.arange(N), test_size=0.2, random_state=42, stratify=labels.numpy()
)
train_mask = torch.zeros(N, dtype=torch.bool)
test_mask = torch.zeros(N, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

print(f"  Train: {train_mask.sum().item()}, Test: {test_mask.sum().item()}")


# ── Message Passing Layer ────────────────────────────────────────────────────

class MessagePassingLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_msg = nn.Linear(in_dim, out_dim, bias=False)
        self.W_upd = nn.Linear(in_dim + out_dim, out_dim)

    def forward(self, h, edge_index):
        num_nodes = h.size(0)
        if edge_index.size(1) == 0:
            zero_msg = torch.zeros(num_nodes, self.W_msg.out_features, device=h.device)
            return F.relu(self.W_upd(torch.cat([h, zero_msg], dim=1)))

        src, dst = edge_index[0], edge_index[1]
        messages = self.W_msg(h[src])
        agg = torch.zeros(num_nodes, messages.size(1), device=h.device)
        agg.index_add_(0, dst, messages)
        return F.relu(self.W_upd(torch.cat([h, agg], dim=1)))


class BipartiteGNN(nn.Module):
    """GNN on bipartite graph -> pool image+text node pairs -> classify."""
    def __init__(self, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.mp1 = MessagePassingLayer(embed_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, n_products):
        h = self.mp1(x, edge_index)
        # Pool: average image node and text node features for each product
        img_h = h[:n_products]
        txt_h = h[n_products:2 * n_products]
        pooled = (img_h + txt_h) / 2  # (N, hidden_dim)
        return self.classifier(pooled)


# ── Training ─────────────────────────────────────────────────────────────────

print("\n[4] Training bipartite GNN ...")

model = BipartiteGNN(EMBED_DIM, EMBED_DIM, NUM_CLASSES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()

node_features_d = node_features.to(device)
all_edges_d = all_edges.to(device)
labels_d = labels.to(device)

EPOCHS = 50
train_accs, test_accs = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    logits = model(node_features_d, all_edges_d, N)
    loss = criterion(logits[train_mask], labels_d[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(node_features_d, all_edges_d, N)
        train_acc = (logits[train_mask].argmax(1) == labels_d[train_mask]).float().mean().item()
        test_acc = (logits[test_mask].argmax(1) == labels_d[test_mask]).float().mean().item()
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={loss.item():.4f}  "
              f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

best_test = max(test_accs)
print(f"\n  Best test accuracy: {best_test:.4f}")

# Save model embeddings for Part D
model.eval()
with torch.no_grad():
    h = model.mp1(node_features_d, all_edges_d)
    img_h = h[:N]
    txt_h = h[N:2*N]
    gnn_embeds = ((img_h + txt_h) / 2).cpu()
torch.save(gnn_embeds, "multimodal-gnn/gnn_bipartite_embeds.pt")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(train_accs, label="Train")
plt.plot(test_accs, label="Test")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Part A: Bipartite GNN Training")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("multimodal-gnn/part_a_training.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved -> multimodal-gnn/part_a_training.png")

# Save results for comparison
torch.save({
    "best_test": best_test, "train_accs": train_accs, "test_accs": test_accs,
    "train_mask": train_mask, "test_mask": test_mask,
    "labels": labels, "node_features": node_features, "all_edges": all_edges,
    "N": N, "NUM_CLASSES": NUM_CLASSES,
}, "multimodal-gnn/part_a_results.pt")
print("Done.")
