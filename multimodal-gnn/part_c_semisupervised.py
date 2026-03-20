"""
Assignment 8 - Part C: Semi-supervised Learning
=================================================
Train with limited labels (5%, 10%, 20%) using:
  1. Supervised only (cross-entropy on labeled nodes)
  2. + Pseudo-labels (generated every 5 epochs)
  3. + Mean-Teacher consistency
Compare across combinations and label fractions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
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

# Fused features
fused_embeds = F.normalize(torch.cat([img_embeds, text_embeds], dim=1), dim=1)
fused_edges = build_knn_edges(fused_embeds, k=5)

fused_d = fused_embeds.to(device)
fused_edges_d = fused_edges.to(device)
labels_d = labels.to(device)

# Fixed test set (20%)
from sklearn.model_selection import train_test_split
all_idx = np.arange(N)
train_pool, test_idx = train_test_split(all_idx, test_size=0.2, random_state=42, stratify=labels.numpy())
test_mask = torch.zeros(N, dtype=torch.bool)
test_mask[test_idx] = True
test_mask = test_mask.to(device)


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


class FusionGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.fuse = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.mp = MessagePassingLayer(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        h = self.fuse(x)
        h = self.mp(h, edge_index)
        return self.classifier(h)


# ── Training functions ───────────────────────────────────────────────────────

EPOCHS = 50


def train_supervised(label_frac):
    """Method 2: Supervised only."""
    n_labeled = int(len(train_pool) * label_frac)
    labeled_idx = train_pool[:n_labeled]
    labeled_mask = torch.zeros(N, dtype=torch.bool)
    labeled_mask[labeled_idx] = True
    labeled_mask = labeled_mask.to(device)

    model = FusionGNN(128, EMBED_DIM, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        logits = model(fused_d, fused_edges_d)
        loss = F.cross_entropy(logits[labeled_mask], labels_d[labeled_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(fused_d, fused_edges_d)
            acc = (logits[test_mask].argmax(1) == labels_d[test_mask]).float().mean().item()
        best_acc = max(best_acc, acc)

    return best_acc


def train_pseudo_labels(label_frac):
    """Method 3: Supervised + pseudo-labels every 5 epochs."""
    n_labeled = int(len(train_pool) * label_frac)
    labeled_idx = train_pool[:n_labeled]
    unlabeled_idx = train_pool[n_labeled:]

    labeled_mask = torch.zeros(N, dtype=torch.bool, device=device)
    labeled_mask[labeled_idx] = True
    unlabeled_mask = torch.zeros(N, dtype=torch.bool, device=device)
    unlabeled_mask[unlabeled_idx] = True

    model = FusionGNN(128, EMBED_DIM, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    pseudo_labels = labels_d.clone()
    pseudo_mask = labeled_mask.clone()

    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        # Generate pseudo-labels every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(fused_d, fused_edges_d)
                probs = F.softmax(logits, dim=1)
                max_probs, preds = probs.max(dim=1)
                confident = max_probs > 0.9
                pseudo_mask = labeled_mask | (unlabeled_mask & confident)
                pseudo_labels[unlabeled_mask & confident] = preds[unlabeled_mask & confident]

        model.train()
        logits = model(fused_d, fused_edges_d)
        loss = F.cross_entropy(logits[pseudo_mask], pseudo_labels[pseudo_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(fused_d, fused_edges_d)
            acc = (logits[test_mask].argmax(1) == labels_d[test_mask]).float().mean().item()
        best_acc = max(best_acc, acc)

    return best_acc


def train_mean_teacher(label_frac):
    """Method 4: Supervised + Mean-Teacher consistency."""
    n_labeled = int(len(train_pool) * label_frac)
    labeled_idx = train_pool[:n_labeled]

    labeled_mask = torch.zeros(N, dtype=torch.bool, device=device)
    labeled_mask[labeled_idx] = True
    unlabeled_mask = torch.zeros(N, dtype=torch.bool, device=device)
    unlabeled_mask[train_pool[n_labeled:]] = True

    student = FusionGNN(128, EMBED_DIM, NUM_CLASSES).to(device)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    ema_decay = 0.99

    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        student.train()
        s_logits = student(fused_d, fused_edges_d)

        # Supervised loss on labeled data
        sup_loss = F.cross_entropy(s_logits[labeled_mask], labels_d[labeled_mask])

        # Consistency loss: student vs teacher on unlabeled
        teacher.eval()
        with torch.no_grad():
            t_logits = teacher(fused_d, fused_edges_d)
        cons_loss = F.mse_loss(
            F.softmax(s_logits[unlabeled_mask], dim=1),
            F.softmax(t_logits[unlabeled_mask], dim=1),
        )

        loss = sup_loss + 0.5 * cons_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA update teacher
        with torch.no_grad():
            for t_param, s_param in zip(teacher.parameters(), student.parameters()):
                t_param.data.mul_(ema_decay).add_(s_param.data, alpha=1 - ema_decay)

        student.eval()
        with torch.no_grad():
            logits = student(fused_d, fused_edges_d)
            acc = (logits[test_mask].argmax(1) == labels_d[test_mask]).float().mean().item()
        best_acc = max(best_acc, acc)

    return best_acc


def train_pseudo_and_teacher(label_frac):
    """Method 2+3+4: All combined."""
    n_labeled = int(len(train_pool) * label_frac)
    labeled_idx = train_pool[:n_labeled]
    unlabeled_idx = train_pool[n_labeled:]

    labeled_mask = torch.zeros(N, dtype=torch.bool, device=device)
    labeled_mask[labeled_idx] = True
    unlabeled_mask = torch.zeros(N, dtype=torch.bool, device=device)
    unlabeled_mask[unlabeled_idx] = True

    student = FusionGNN(128, EMBED_DIM, NUM_CLASSES).to(device)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    ema_decay = 0.99
    pseudo_labels = labels_d.clone()
    pseudo_mask = labeled_mask.clone()

    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        # Pseudo-labels every 5 epochs
        if epoch % 5 == 0:
            teacher.eval()
            with torch.no_grad():
                t_logits = teacher(fused_d, fused_edges_d)
                probs = F.softmax(t_logits, dim=1)
                max_probs, preds = probs.max(dim=1)
                confident = max_probs > 0.9
                pseudo_mask = labeled_mask | (unlabeled_mask & confident)
                pseudo_labels[unlabeled_mask & confident] = preds[unlabeled_mask & confident]

        student.train()
        s_logits = student(fused_d, fused_edges_d)

        # Supervised + pseudo-label loss
        sup_loss = F.cross_entropy(s_logits[pseudo_mask], pseudo_labels[pseudo_mask])

        # Consistency loss
        teacher.eval()
        with torch.no_grad():
            t_logits = teacher(fused_d, fused_edges_d)
        cons_loss = F.mse_loss(
            F.softmax(s_logits[unlabeled_mask], dim=1),
            F.softmax(t_logits[unlabeled_mask], dim=1),
        )

        loss = sup_loss + 0.5 * cons_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for t_param, s_param in zip(teacher.parameters(), student.parameters()):
                t_param.data.mul_(ema_decay).add_(s_param.data, alpha=1 - ema_decay)

        student.eval()
        with torch.no_grad():
            logits = student(fused_d, fused_edges_d)
            acc = (logits[test_mask].argmax(1) == labels_d[test_mask]).float().mean().item()
        best_acc = max(best_acc, acc)

    return best_acc


# ── Run experiments ──────────────────────────────────────────────────────────

fractions = [0.05, 0.10, 0.20]
methods = {
    "Supervised only": train_supervised,
    "+ Pseudo-labels": train_pseudo_labels,
    "+ Mean-Teacher": train_mean_teacher,
    "+ Both": train_pseudo_and_teacher,
}

results = {name: [] for name in methods}

for frac in fractions:
    print(f"\n{'='*60}")
    print(f"  Label fraction: {frac*100:.0f}%  ({int(len(train_pool)*frac)} labeled)")
    print(f"{'='*60}")
    for name, fn in methods.items():
        acc = fn(frac)
        results[name].append(acc)
        print(f"  {name:<25s}  test_acc={acc:.4f}")

# ── Summary table ────────────────────────────────────────────────────────────

print("\n\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
header = f"  {'Method':<25s}"
for frac in fractions:
    header += f"  {frac*100:.0f}%{'':<8s}"
print(header)
print(f"  {'-'*25}" + f"  {'-'*10}" * len(fractions))
for name in methods:
    row = f"  {name:<25s}"
    for acc in results[name]:
        row += f"  {acc:>10.4f}"
    print(row)

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
x = [f"{f*100:.0f}%" for f in fractions]
for name in methods:
    ax.plot(x, results[name], "o-", label=name)
ax.set_xlabel("Label Fraction")
ax.set_ylabel("Test Accuracy")
ax.set_title("Part C: Semi-supervised Learning Comparison")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("multimodal-gnn/part_c_semisupervised.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved -> multimodal-gnn/part_c_semisupervised.png")

torch.save(results, "multimodal-gnn/part_c_results.pt")
print("Done.")
