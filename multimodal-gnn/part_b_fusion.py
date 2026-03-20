"""
Assignment 8 - Part B: Fusion for Multimodalities
===================================================
- Early fusion: concat image+text embeddings -> single GNN
- Late fusion: separate image GNN + text GNN -> merge outputs
- Cross-modal retrieval via fused embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, ".")

from data_utils import prepare_all, build_knn_edges

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load data and Part A results ─────────────────────────────────────────────

df, label_map, img_embeds, text_embeds, labels = prepare_all(subset_size=5000)
N = len(labels)
NUM_CLASSES = len(label_map)
EMBED_DIM = 64
CLASSES = sorted(label_map.keys())

part_a = torch.load("multimodal-gnn/part_a_results.pt", weights_only=False)
train_mask = part_a["train_mask"]
test_mask = part_a["test_mask"]
bipartite_acc = part_a["best_test"]

print(f"\n  Part A bipartite GNN best test acc: {bipartite_acc:.4f}")

# ── Build graphs ─────────────────────────────────────────────────────────────

print("\n[3] Building graphs ...")
K = 5

img_edges = build_knn_edges(img_embeds, k=K)
text_edges = build_knn_edges(text_embeds, k=K)

# Fused embeddings for early fusion
fused_embeds = F.normalize(torch.cat([img_embeds, text_embeds], dim=1), dim=1)  # (N, 128)
fused_edges = build_knn_edges(fused_embeds, k=K)


# ── Message Passing ──────────────────────────────────────────────────────────

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


# ══════════════════════════════════════════════════════════════════════════════
# EARLY FUSION
# ══════════════════════════════════════════════════════════════════════════════

class EarlyFusionGNN(nn.Module):
    """Concat image+text -> MLP -> GNN on fused graph -> classify."""
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.fuse = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.mp = MessagePassingLayer(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        h = self.fuse(x)
        h = self.mp(h, edge_index)
        return self.classifier(h)


print("\n" + "=" * 60)
print("  Early Fusion GNN")
print("=" * 60)

ef_model = EarlyFusionGNN(128, EMBED_DIM, NUM_CLASSES).to(device)
ef_optimizer = torch.optim.Adam(ef_model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()

fused_d = fused_embeds.to(device)
fused_edges_d = fused_edges.to(device)
labels_d = labels.to(device)

EPOCHS = 50
ef_test_accs = []

for epoch in range(1, EPOCHS + 1):
    ef_model.train()
    logits = ef_model(fused_d, fused_edges_d)
    loss = criterion(logits[train_mask], labels_d[train_mask])
    ef_optimizer.zero_grad()
    loss.backward()
    ef_optimizer.step()

    ef_model.eval()
    with torch.no_grad():
        logits = ef_model(fused_d, fused_edges_d)
        test_acc = (logits[test_mask].argmax(1) == labels_d[test_mask]).float().mean().item()
    ef_test_accs.append(test_acc)

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={loss.item():.4f}  test_acc={test_acc:.4f}")

ef_best = max(ef_test_accs)
print(f"  Best test accuracy: {ef_best:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# LATE FUSION
# ══════════════════════════════════════════════════════════════════════════════

class LateFusionGNN(nn.Module):
    """Two separate GNNs (image graph, text graph) -> concat+MLP -> classify."""
    def __init__(self, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.img_mp = MessagePassingLayer(embed_dim, hidden_dim)
        self.txt_mp = MessagePassingLayer(embed_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, img_feats, img_edges, txt_feats, txt_edges):
        h_img = self.img_mp(img_feats, img_edges)
        h_txt = self.txt_mp(txt_feats, txt_edges)
        merged = torch.cat([h_img, h_txt], dim=1)
        return self.classifier(merged)


print("\n" + "=" * 60)
print("  Late Fusion GNN")
print("=" * 60)

lf_model = LateFusionGNN(EMBED_DIM, EMBED_DIM, NUM_CLASSES).to(device)
lf_optimizer = torch.optim.Adam(lf_model.parameters(), lr=1e-2)

img_d = img_embeds.to(device)
txt_d = text_embeds.to(device)
img_edges_d = img_edges.to(device)
txt_edges_d = text_edges.to(device)

lf_test_accs = []

for epoch in range(1, EPOCHS + 1):
    lf_model.train()
    logits = lf_model(img_d, img_edges_d, txt_d, txt_edges_d)
    loss = criterion(logits[train_mask], labels_d[train_mask])
    lf_optimizer.zero_grad()
    loss.backward()
    lf_optimizer.step()

    lf_model.eval()
    with torch.no_grad():
        logits = lf_model(img_d, img_edges_d, txt_d, txt_edges_d)
        test_acc = (logits[test_mask].argmax(1) == labels_d[test_mask]).float().mean().item()
    lf_test_accs.append(test_acc)

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={loss.item():.4f}  test_acc={test_acc:.4f}")

lf_best = max(lf_test_accs)
print(f"  Best test accuracy: {lf_best:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-MODAL RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  Cross-Modal Retrieval")
print("=" * 60)

# Get GNN-learnt embeddings from early fusion model
ef_model.eval()
with torch.no_grad():
    h_fused = ef_model.fuse(fused_d)
    gnn_embeds = ef_model.mp(h_fused, fused_edges_d)
    gnn_embeds = F.normalize(gnn_embeds, dim=1).cpu()

# Text-to-Image retrieval: use text embedding as query, find nearest in GNN space
print("\n  Text-to-Image retrieval (5 examples):")
print(f"  {'Query (text)':<40s} {'Retrieved Category':<20s} {'Match'}")
print(f"  {'-'*40} {'-'*20} {'-'*5}")

test_indices = torch.where(test_mask)[0][:5]
for idx in test_indices:
    # Query: zero image + valid text -> fused, then project through model
    query_text = text_embeds[idx]
    query_img = torch.zeros(EMBED_DIM)
    query_fused = torch.cat([query_img, query_text]).unsqueeze(0).to(device)
    with torch.no_grad():
        query_proj = F.normalize(ef_model.fuse(query_fused), dim=1).cpu()

    # Compare with all GNN embeddings
    sims = torch.mm(query_proj, gnn_embeds.t()).squeeze()
    sims[idx] = -1  # exclude self

    top_idx = sims.argmax().item()
    true_cat = CLASSES[labels[idx].item()]
    ret_cat = CLASSES[labels[top_idx].item()]
    match = "YES" if true_cat == ret_cat else "NO"
    name = str(df.loc[idx, "productDisplayName"])[:38]
    print(f"  {name:<40s} {ret_cat:<20s} {match}")

# Image-to-Text retrieval
print("\n  Image-to-Text retrieval (5 examples):")
print(f"  {'Query (image ID)':<40s} {'Retrieved Product':<40s} {'Match'}")
print(f"  {'-'*40} {'-'*40} {'-'*5}")

for idx in test_indices:
    query_img = img_embeds[idx]
    query_text = torch.zeros(EMBED_DIM)
    query_fused = torch.cat([query_img, query_text]).unsqueeze(0).to(device)
    with torch.no_grad():
        query_proj = F.normalize(ef_model.fuse(query_fused), dim=1).cpu()

    sims = torch.mm(query_proj, gnn_embeds.t()).squeeze()
    sims[idx] = -1

    top_idx = sims.argmax().item()
    true_cat = CLASSES[labels[idx].item()]
    ret_cat = CLASSES[labels[top_idx].item()]
    match = "YES" if true_cat == ret_cat else "NO"
    img_id = str(df.loc[idx, "id"])
    ret_name = str(df.loc[top_idx, "productDisplayName"])[:38]
    print(f"  {'ID: ' + img_id:<40s} {ret_name:<40s} {match}")

# ── Summary ──────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 60)
print("  COMPARISON")
print("=" * 60)
print(f"  {'Model':<30s} {'Best Test Acc':>14s}")
print(f"  {'-'*30} {'-'*14}")
print(f"  {'Bipartite GNN (Part A)':<30s} {bipartite_acc:>14.4f}")
print(f"  {'Early Fusion GNN':<30s} {ef_best:>14.4f}")
print(f"  {'Late Fusion GNN':<30s} {lf_best:>14.4f}")

# Save for later parts
torch.save({
    "bipartite_acc": bipartite_acc, "ef_best": ef_best, "lf_best": lf_best,
    "ef_test_accs": ef_test_accs, "lf_test_accs": lf_test_accs,
    "gnn_embeds": gnn_embeds,
}, "multimodal-gnn/part_b_results.pt")

plt.figure(figsize=(8, 5))
plt.plot(part_a["test_accs"], label="Bipartite GNN")
plt.plot(ef_test_accs, label="Early Fusion")
plt.plot(lf_test_accs, label="Late Fusion")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("Part B: Fusion Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("multimodal-gnn/part_b_fusion.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved -> multimodal-gnn/part_b_fusion.png")
print("Done.")
