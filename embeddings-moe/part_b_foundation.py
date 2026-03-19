"""
Assignment 4 - Part B: Foundation Model Embeddings (CLIP)
==========================================================
Extracts CLIP embeddings from the same Fashion-MNIST test set,
visualises with t-SNE, and compares to Part A embeddings.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load CLIP ────────────────────────────────────────────────────────────────
print("\n[1/4] Loading CLIP model ...")

try:
    import clip
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    print("  Loaded CLIP ViT-B/32 via openai/clip")
    USE_CLIP_PACKAGE = True
except ImportError:
    from transformers import CLIPProcessor, CLIPModel
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("  Loaded CLIP ViT-B/32 via transformers")
    USE_CLIP_PACKAGE = False

# ── Load Fashion-MNIST (same as Part A) ──────────────────────────────────────
print("\n[2/4] Loading Fashion-MNIST ...")

# Raw dataset for CLIP (needs PIL images, 3-channel)
raw_test = datasets.FashionMNIST(root="./data", train=False, download=True)
CLASS_NAMES = raw_test.classes
print(f"  Test set: {len(raw_test)} images, {len(CLASS_NAMES)} classes")

# ── Extract CLIP embeddings ──────────────────────────────────────────────────
print("\n[3/4] Extracting CLIP embeddings ...")

clip_embeddings = []
clip_labels = []

clip_model.eval()
BATCH = 128

for start in range(0, len(raw_test), BATCH):
    end = min(start + BATCH, len(raw_test))
    batch_imgs = []
    batch_labels = []

    for i in range(start, end):
        img, label = raw_test[i]
        # Convert grayscale to RGB
        img_rgb = img.convert("RGB")
        batch_imgs.append(img_rgb)
        batch_labels.append(label)

    if USE_CLIP_PACKAGE:
        processed = torch.stack([clip_preprocess(img) for img in batch_imgs]).to(device)
        with torch.no_grad():
            features = clip_model.encode_image(processed)
        features = features.cpu().numpy()
    else:
        inputs = clip_processor(images=batch_imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
        features = features.cpu().numpy()

    clip_embeddings.append(features)
    clip_labels.extend(batch_labels)

    if (start // BATCH) % 20 == 0:
        print(f"  Processed {end}/{len(raw_test)} images ...")

clip_embeddings = np.concatenate(clip_embeddings)
clip_labels = np.array(clip_labels)

# Normalize CLIP embeddings (standard practice)
clip_embeddings = clip_embeddings / np.linalg.norm(clip_embeddings, axis=1, keepdims=True)

print(f"  CLIP embedding shape: {clip_embeddings.shape}")

# ── Load Part A embeddings ───────────────────────────────────────────────────
print("\n[4/4] Comparing embeddings ...")

part_a = torch.load("embeddings-moe/part_a_embeddings.pt", weights_only=True)
ae_embeddings = part_a["ae_embeddings"]
ae_labels = part_a["ae_labels"]
con_embeddings = part_a["con_embeddings"]
con_labels = part_a["con_labels"]

# ── kNN accuracy as embedding quality metric ─────────────────────────────────
# Use first 5000 as train, rest as test for kNN
KNN_SPLIT = 5000

def knn_accuracy(embeddings, labels, k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(embeddings[:KNN_SPLIT], labels[:KNN_SPLIT])
    acc = knn.score(embeddings[KNN_SPLIT:], labels[KNN_SPLIT:])
    return acc

ae_knn = knn_accuracy(ae_embeddings, ae_labels)
con_knn = knn_accuracy(con_embeddings, con_labels)
clip_knn = knn_accuracy(clip_embeddings, clip_labels)

print(f"\n  kNN (k=5) accuracy on embeddings:")
print(f"    Autoencoder (16-d):    {ae_knn:.4f}")
print(f"    Contrastive (16-d):    {con_knn:.4f}")
print(f"    CLIP ViT-B/32 (512-d): {clip_knn:.4f}")

# ── t-SNE visualisation (all three) ─────────────────────────────────────────
print("\n  Computing t-SNE projections ...")

N_VIS = 3000
idx = np.random.choice(len(clip_labels), N_VIS, replace=False)

tsne_ae = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(ae_embeddings[idx])
tsne_con = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(con_embeddings[idx])
tsne_clip = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(clip_embeddings[idx])

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for c in range(10):
    mask = ae_labels[idx] == c
    axes[0].scatter(tsne_ae[mask, 0], tsne_ae[mask, 1], s=5, alpha=0.6, label=CLASS_NAMES[c])
axes[0].set_title("Autoencoder (16-d)")
axes[0].legend(fontsize=6, markerscale=3)
axes[0].set_xticks([])
axes[0].set_yticks([])

for c in range(10):
    mask = con_labels[idx] == c
    axes[1].scatter(tsne_con[mask, 0], tsne_con[mask, 1], s=5, alpha=0.6, label=CLASS_NAMES[c])
axes[1].set_title("Contrastive (16-d)")
axes[1].legend(fontsize=6, markerscale=3)
axes[1].set_xticks([])
axes[1].set_yticks([])

for c in range(10):
    mask = clip_labels[idx] == c
    axes[2].scatter(tsne_clip[mask, 0], tsne_clip[mask, 1], s=5, alpha=0.6, label=CLASS_NAMES[c])
axes[2].set_title("CLIP ViT-B/32 (512-d)")
axes[2].legend(fontsize=6, markerscale=3)
axes[2].set_xticks([])
axes[2].set_yticks([])

plt.suptitle("Embedding Comparison: Autoencoder vs Contrastive vs CLIP", fontsize=13)
plt.tight_layout()
plt.savefig("embeddings-moe/part_b_embedding_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved -> embeddings-moe/part_b_embedding_comparison.png")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  {'Method':<25s} {'Dim':>5s} {'kNN Acc':>10s}")
print(f"  {'-'*25} {'-'*5} {'-'*10}")
print(f"  {'Autoencoder':<25s} {'16':>5s} {ae_knn:>10.4f}")
print(f"  {'Contrastive':<25s} {'16':>5s} {con_knn:>10.4f}")
print(f"  {'CLIP ViT-B/32':<25s} {'512':>5s} {clip_knn:>10.4f}")
