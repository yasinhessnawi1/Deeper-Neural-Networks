"""
Assignment 2 - Part A: Self-Attention From "Scratch"
=====================================================
Implements a minimal self-attention mechanism for a single sentence,
without training any matrices. All Q, K, V projections use fixed
(random but seeded) weight matrices so results are reproducible.

Reference: https://mohdfaraaz.medium.com/implementing-self-attention-from-scratch-in-pytorch-776ef7b8f13e
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Reproducibility ──────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── 2. Sentence & tokenisation ──────────────────────────────────────────────
sentence = "Darth Vader is the villain"
words = sentence.split()
print(f"Sentence : {sentence}")
print(f"Tokens   : {words}")
print(f"Num tokens: {len(words)}\n")

# ── 3. Word embeddings (random, 4-d) ────────────────────────────────────────
embed_dim = 4
num_tokens = len(words)

# Create a simple vocabulary -> embedding lookup (fixed random vectors)
embeddings = torch.randn(num_tokens, embed_dim)
print("Word embeddings (each row = one token):")
for w, e in zip(words, embeddings):
    print(f"  {w:>8s} -> {e.tolist()}")
print()

# ── 4. Define Q, K, V weight matrices (not trained) ─────────────────────────
d_k = 3  # dimension of queries / keys / values (small on purpose)

W_q = torch.randn(embed_dim, d_k)
W_k = torch.randn(embed_dim, d_k)
W_v = torch.randn(embed_dim, d_k)

print(f"Projection dimensions: embed_dim={embed_dim} -> d_k={d_k}")
print(f"W_q shape: {W_q.shape}")
print(f"W_k shape: {W_k.shape}")
print(f"W_v shape: {W_v.shape}\n")

# ── 5. Compute Q, K, V ──────────────────────────────────────────────────────
Q = embeddings @ W_q   # (num_tokens, d_k)
K = embeddings @ W_k
V = embeddings @ W_v

print("Query matrix Q (tokens × d_k):")
print(Q)
print("\nKey matrix K:")
print(K)
print("\nValue matrix V:")
print(V)
print()

# ── 6. Compute scaled dot-product attention scores ──────────────────────────
#    score(i,j) = Q_i · K_j / sqrt(d_k)
scores = Q @ K.T / (d_k ** 0.5)   # (num_tokens, num_tokens)

print("Raw attention scores (before softmax):")
print(scores)
print()

# ── 7. Apply softmax row-wise ───────────────────────────────────────────────
attention_weights = F.softmax(scores, dim=-1)   # each row sums to 1

print("Attention weights (after softmax):")
print(attention_weights)
print()

# Verify rows sum to 1
row_sums = attention_weights.sum(dim=-1)
print(f"Row sums (should all be 1.0): {row_sums.tolist()}\n")

# ── 8. Compute context vectors ──────────────────────────────────────────────
context = attention_weights @ V   # (num_tokens, d_k)

print("Context vectors (weighted combination of V):")
for w, c in zip(words, context):
    print(f"  {w:>8s} -> {c.tolist()}")
print()

# ── 9. Interpretation ───────────────────────────────────────────────────────
print("=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("""
Each context vector for a word is a weighted sum of ALL value
vectors, where the weights come from how much that word's query
"matches" every other word's key.

For example, the attention weights for "villain" show how much
it attends to each other word:
""")
villain_idx = words.index("villain")
for w, weight in zip(words, attention_weights[villain_idx]):
    bar = "#" * int(weight.item() * 40)
    print(f"  villain -> {w:>8s}: {weight.item():.4f}  {bar}")

print("""
These weights are NOT trained — they come from random projections.
In a real Transformer, W_q, W_k, W_v would be learned so that
semantically related words attend more strongly to each other.
""")

# ── 10. Visualisation ───────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Attention heatmap
sns.heatmap(
    attention_weights.detach().numpy(),
    xticklabels=words,
    yticklabels=words,
    annot=True,
    fmt=".3f",
    cmap="YlOrRd",
    ax=axes[0],
    cbar_kws={"label": "Attention weight"},
)
axes[0].set_title("Self-Attention Weights", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Key (attending to)")
axes[0].set_ylabel("Query (from)")

# (b) Raw scores heatmap (before softmax)
sns.heatmap(
    scores.detach().numpy(),
    xticklabels=words,
    yticklabels=words,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    ax=axes[1],
    cbar_kws={"label": "Score"},
)
axes[1].set_title("Raw Attention Scores (before softmax)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Key")
axes[1].set_ylabel("Query")

# (c) Per-word attention bar chart
x = np.arange(num_tokens)
width = 0.15
for i, (w, color) in enumerate(zip(words, plt.cm.tab10.colors)):
    axes[2].bar(x + i * width, attention_weights[i].detach().numpy(),
                width, label=w, color=color)
axes[2].set_xticks(x + width * (num_tokens - 1) / 2)
axes[2].set_xticklabels(words)
axes[2].set_ylabel("Attention weight")
axes[2].set_title("Attention Distribution per Query Word", fontsize=13, fontweight="bold")
axes[2].legend(title="Query word", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.savefig("attention-transformers/attention_visualisation.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure saved -> attention-transformers/attention_visualisation.png")

# ── 11. Step-by-step walkthrough for one token ──────────────────────────────
print("\n" + "=" * 60)
print("STEP-BY-STEP: How 'Vader' attends to all words")
print("=" * 60)
vader_idx = words.index("Vader")
q_vader = Q[vader_idx]
print(f"\n1. Query vector for 'Vader':  q = {q_vader.tolist()}")
print(f"2. Key vectors:")
for w, k in zip(words, K):
    print(f"     K_{w:>8s} = {k.tolist()}")

print(f"\n3. Dot products  q · k_j  (scaled by 1/sqrtd_k = 1/sqrt{d_k} ~= {1/d_k**0.5:.4f}):")
for w, s in zip(words, scores[vader_idx]):
    print(f"     score(Vader, {w:>8s}) = {s.item():.4f}")

print(f"\n4. After softmax:")
for w, a in zip(words, attention_weights[vader_idx]):
    print(f"     a(Vader, {w:>8s}) = {a.item():.4f}")

print(f"\n5. Context vector for 'Vader' = SUM a_j · v_j")
print(f"   = {context[vader_idx].tolist()}")
