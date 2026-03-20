"""
Assignment 7: Graph Neural Networks + State Space Models
=========================================================
Dataset: Fashion Product Images (Small) - text classification
  - Input: productDisplayName (text)
  - Target: masterCategory (Apparel, Accessories, Footwear, etc.)

Part A: GNN baseline (text -> word graph -> message passing -> classify)
Part B: SSM-only baseline (text -> token embeddings -> SSM -> classify)
Part C: Hybrid SSM+GNN (text -> SSM -> word graph -> GNN -> classify)
Part D: Compare all three
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import time
import os

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1] Loading dataset ...")

# Find styles.csv
DATA_PATHS = [
    "gnn-ssm/data/styles.csv",
    "gnn-ssm/data/fashion-product-images-small/styles.csv",
]
csv_path = None
for p in DATA_PATHS:
    if os.path.exists(p):
        csv_path = p
        break

if csv_path is None:
    print("ERROR: styles.csv not found. Run download_data.py first or place it in gnn-ssm/data/")
    exit(1)

df = pd.read_csv(csv_path, on_bad_lines="skip")
df = df[["productDisplayName", "masterCategory"]].dropna()

# Keep top categories with enough samples
cat_counts = df["masterCategory"].value_counts()
top_cats = cat_counts[cat_counts >= 100].index.tolist()
df = df[df["masterCategory"].isin(top_cats)].reset_index(drop=True)

# Label encoding
label_map = {cat: i for i, cat in enumerate(sorted(top_cats))}
df["label"] = df["masterCategory"].map(label_map)
NUM_CLASSES = len(label_map)

print(f"  Samples: {len(df)}, Classes: {NUM_CLASSES}")
print(f"  Categories: {label_map}")

# Build vocabulary
all_words = []
for text in df["productDisplayName"]:
    all_words.extend(str(text).lower().split())

word_counts = Counter(all_words)
# Keep words appearing >= 2 times
vocab = {w: i + 1 for i, (w, c) in enumerate(word_counts.most_common()) if c >= 2}
vocab["<unk>"] = 0
VOCAB_SIZE = len(vocab)
print(f"  Vocabulary size: {VOCAB_SIZE}")

# Tokenize
MAX_LEN = 20

def tokenize(text):
    words = str(text).lower().split()[:MAX_LEN]
    tokens = [vocab.get(w, 0) for w in words]
    return tokens


df["tokens"] = df["productDisplayName"].apply(tokenize)

# Train/test split
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
print(f"  Train: {len(train_df)}, Test: {len(test_df)}")


# ── Dataset class ────────────────────────────────────────────────────────────

class TextGraphDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data.loc[idx, "tokens"]
        label = self.data.loc[idx, "label"]
        return tokens, label


def collate_fn(batch):
    """Collate variable-length token lists into padded tensors + edge lists."""
    tokens_list, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)

    # Pad tokens
    max_len = max(len(t) for t in tokens_list)
    padded = torch.zeros(len(tokens_list), max_len, dtype=torch.long)
    lengths = []
    for i, t in enumerate(tokens_list):
        padded[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        lengths.append(len(t))

    lengths = torch.tensor(lengths, dtype=torch.long)

    # Build edge indices for each graph (consecutive word edges, both directions)
    edge_indices = []
    for t in tokens_list:
        n = len(t)
        if n <= 1:
            edge_indices.append(torch.zeros(2, 0, dtype=torch.long))
        else:
            src = list(range(n - 1)) + list(range(1, n))
            dst = list(range(1, n)) + list(range(n - 1))
            edge_indices.append(torch.tensor([src, dst], dtype=torch.long))

    return padded, lengths, edge_indices, labels


train_set = TextGraphDataset(train_df)
test_set = TextGraphDataset(test_df)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, collate_fn=collate_fn)


# ══════════════════════════════════════════════════════════════════════════════
# PART A: GNN-ONLY MODEL
# ══════════════════════════════════════════════════════════════════════════════

EMBED_DIM = 64
HIDDEN_DIM = 64


class MessagePassingLayer(nn.Module):
    """One layer of message passing from scratch.
    For each node i:
      m_i = SUM_{j in N(i)} W_msg * h_j     (aggregate messages)
      h_i' = ReLU(W_upd * [h_i || m_i])      (update)
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_msg = nn.Linear(in_dim, out_dim, bias=False)
        self.W_upd = nn.Linear(in_dim + out_dim, out_dim)

    def forward(self, h, edge_index):
        """h: (num_nodes, in_dim), edge_index: (2, num_edges)"""
        num_nodes = h.size(0)

        if edge_index.size(1) == 0:
            # No edges: just transform
            zero_msg = torch.zeros(num_nodes, self.W_msg.out_features, device=h.device)
            return F.relu(self.W_upd(torch.cat([h, zero_msg], dim=1)))

        src, dst = edge_index[0], edge_index[1]

        # Message: transform source node features
        messages = self.W_msg(h[src])  # (num_edges, out_dim)

        # Aggregate: sum messages at each destination node
        agg = torch.zeros(num_nodes, messages.size(1), device=h.device)
        agg.index_add_(0, dst, messages)

        # Update: combine original features with aggregated messages
        updated = F.relu(self.W_upd(torch.cat([h, agg], dim=1)))
        return updated


class GNNClassifier(nn.Module):
    """Embedding -> 1x MessagePassing -> MeanPool -> Classifier"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.mp = MessagePassingLayer(embed_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, tokens, lengths, edge_indices):
        batch_size = tokens.size(0)
        outputs = []

        for i in range(batch_size):
            n = lengths[i].item()
            h = self.embedding(tokens[i, :n].to(device))  # (n, embed_dim)
            edge_idx = edge_indices[i].to(device)

            # Message passing
            h = self.mp(h, edge_idx)

            # Mean pool
            pooled = h.mean(dim=0)  # (hidden_dim,)
            outputs.append(pooled)

        pooled_batch = torch.stack(outputs)  # (batch, hidden_dim)
        return self.classifier(pooled_batch)


# ══════════════════════════════════════════════════════════════════════════════
# PART B: SSM-ONLY MODEL (Simple S4-style SSM)
# ══════════════════════════════════════════════════════════════════════════════

# Try to import Mamba; fall back to a simple SSM if not available
USE_MAMBA = False
try:
    from mamba_ssm import Mamba
    USE_MAMBA = True
    print("\n  Using Mamba SSM")
except ImportError:
    print("\n  Mamba not available, using simple SSM (diagonal state space)")


class SimpleSSM(nn.Module):
    """Simple diagonal state space model.
    Discrete SSM: h_t = A * h_{t-1} + B * x_t, y_t = C * h_t
    A is diagonal (learnable), making this efficient.
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Learnable parameters
        self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.B = nn.Linear(d_model, d_state, bias=False)
        self.C = nn.Linear(d_state, d_model, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))  # skip connection

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)"""
        batch, seq_len, d = x.shape

        # Discretize A (ensure stability with negative real part)
        A = -torch.exp(self.A_log)  # (d_model, d_state), all negative

        outputs = []
        h = torch.zeros(batch, self.d_model, self.d_state, device=x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, d_model)
            b_t = self.B(x_t)  # (batch, d_state)

            # State update: h = A * h + B(x) (element-wise on state dim)
            h = h * torch.exp(A.unsqueeze(0)) + b_t.unsqueeze(1).expand_as(h)

            # Output: y = C * h + D * x
            y = self.C(h.mean(dim=1)) + self.D * x_t  # simplified readout
            outputs.append(y)

        out = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        return self.norm(out)


class SSMClassifier(nn.Module):
    """Embedding -> SSM -> MeanPool -> Classifier"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if USE_MAMBA:
            self.ssm = Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
        else:
            self.ssm = SimpleSSM(embed_dim, d_state=16)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens, lengths, edge_indices=None):
        x = self.embedding(tokens.to(device))  # (batch, max_len, embed_dim)

        # SSM
        x = self.ssm(x)  # (batch, max_len, embed_dim)

        # Masked mean pool (ignore padding)
        mask = torch.arange(x.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1).to(device)
        mask = mask.unsqueeze(2).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.classifier(pooled)


# ══════════════════════════════════════════════════════════════════════════════
# PART C: HYBRID SSM + GNN
# ══════════════════════════════════════════════════════════════════════════════

class HybridSSMGNN(nn.Module):
    """Embedding -> SSM (contextual features) -> GNN message passing -> MeanPool -> Classifier"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if USE_MAMBA:
            self.ssm = Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
        else:
            self.ssm = SimpleSSM(embed_dim, d_state=16)
        self.mp = MessagePassingLayer(embed_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, tokens, lengths, edge_indices):
        x = self.embedding(tokens.to(device))  # (batch, max_len, embed_dim)

        # SSM: produce contextual features
        x = self.ssm(x)  # (batch, max_len, embed_dim)

        batch_size = tokens.size(0)
        outputs = []

        for i in range(batch_size):
            n = lengths[i].item()
            h = x[i, :n, :]  # (n, embed_dim) -- SSM-enriched features
            edge_idx = edge_indices[i].to(device)

            # GNN message passing on word graph
            h = self.mp(h, edge_idx)

            # Mean pool
            pooled = h.mean(dim=0)
            outputs.append(pooled)

        pooled_batch = torch.stack(outputs)
        return self.classifier(pooled_batch)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for tokens, lengths, edge_indices, labels in loader:
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(tokens, lengths, edge_indices)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for tokens, lengths, edge_indices, labels in loader:
            labels = labels.to(device)
            logits = model(tokens, lengths, edge_indices)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_model(name, model, epochs=20):
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  {name}  ({params:,} params)")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_losses, train_accs, test_accs = [], [], []
    best_acc = 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_acc = evaluate(model, test_loader)
        scheduler.step()

        train_losses.append(loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        best_acc = max(best_acc, test_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{epochs}  loss={loss:.4f}  "
                  f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    elapsed = time.time() - start
    print(f"  Best test accuracy: {best_acc:.4f}  ({elapsed:.1f}s)")
    return {
        "name": name, "params": params, "best_acc": best_acc,
        "train_losses": train_losses, "test_accs": test_accs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART D: EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════

EPOCHS = 20

print("\n\n" + "#" * 60)
print("  PART D: EXPERIMENTS")
print("#" * 60)

# D1: GNN-only
r_gnn = train_model(
    "D1: GNN-only",
    GNNClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES),
    epochs=EPOCHS,
)

# D2: SSM-only
r_ssm = train_model(
    "D2: SSM-only",
    SSMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES),
    epochs=EPOCHS,
)

# D3: Hybrid SSM+GNN
r_hybrid = train_model(
    "D3: Hybrid SSM+GNN",
    HybridSSMGNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES),
    epochs=EPOCHS,
)

results = [r_gnn, r_ssm, r_hybrid]

# ── Summary ──────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  {'Model':<25s} {'Params':>10s} {'Best Test Acc':>14s}")
print(f"  {'-'*25} {'-'*10} {'-'*14}")
for r in results:
    print(f"  {r['name']:<25s} {r['params']:>10,} {r['best_acc']:>14.4f}")

# ── Plots ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for r in results:
    axes[0].plot(r["train_losses"], label=r["name"])
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Training Loss")
axes[0].set_title("Training Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

for r in results:
    axes[1].plot(r["test_accs"], label=r["name"])
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Test Accuracy")
axes[1].set_title("Test Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gnn-ssm/training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved -> gnn-ssm/training_curves.png")
