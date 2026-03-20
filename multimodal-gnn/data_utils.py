"""
Shared data loading and embedding extraction for Assignment 8.
Loads Fashion Product Images (Small), extracts CNN image embeddings
and text embeddings, builds graphs, and saves everything to disk.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import os

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset():
    """Load Fashion Product dataset and return DataFrame with valid image paths."""
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
        raise FileNotFoundError("styles.csv not found. Run gnn-ssm/download_data.py first.")

    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df = df[["id", "productDisplayName", "masterCategory"]].dropna()

    # Keep categories with enough samples
    cat_counts = df["masterCategory"].value_counts()
    top_cats = cat_counts[cat_counts >= 100].index.tolist()
    df = df[df["masterCategory"].isin(top_cats)].reset_index(drop=True)

    # Find image directory
    img_dirs = [
        "gnn-ssm/data/images/",
        "gnn-ssm/data/fashion-product-images-small/images/",
    ]
    img_dir = None
    for d in img_dirs:
        if os.path.exists(d):
            img_dir = d
            break

    if img_dir is None:
        raise FileNotFoundError("Image directory not found.")

    # Filter to products with existing images
    df["img_path"] = df["id"].apply(lambda x: os.path.join(img_dir, f"{x}.jpg"))
    df = df[df["img_path"].apply(os.path.exists)].reset_index(drop=True)

    # Label encoding
    label_map = {cat: i for i, cat in enumerate(sorted(top_cats))}
    df["label"] = df["masterCategory"].map(label_map)

    return df, label_map


def extract_image_embeddings(df, embed_dim=64, batch_size=128):
    """Extract image embeddings using a pretrained ResNet18."""
    cache_path = "multimodal-gnn/image_embeddings.pt"
    if os.path.exists(cache_path):
        data = torch.load(cache_path, weights_only=True)
        print(f"  Loaded cached image embeddings: {data.shape}")
        return data

    print("  Extracting image embeddings (ResNet18) ...")

    # Use pretrained ResNet18, replace fc with projection to embed_dim
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device).eval()

    projector = nn.Linear(512, embed_dim).to(device)
    nn.init.xavier_uniform_(projector.weight)

    transform = transforms.Compose([
        transforms.Resize((80, 60)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    all_embeds = []
    paths = df["img_path"].tolist()

    for start in range(0, len(paths), batch_size):
        end = min(start + batch_size, len(paths))
        batch_imgs = []
        for p in paths[start:end]:
            try:
                img = Image.open(p).convert("RGB")
                batch_imgs.append(transform(img))
            except Exception:
                batch_imgs.append(torch.zeros(3, 80, 60))

        batch = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            feats = resnet(batch)  # (B, 512)
            embeds = projector(feats)  # (B, embed_dim)
        all_embeds.append(embeds.cpu())

        if (start // batch_size) % 50 == 0:
            print(f"    {end}/{len(paths)} ...")

    all_embeds = torch.cat(all_embeds)
    # L2 normalize
    all_embeds = F.normalize(all_embeds, dim=1)
    torch.save(all_embeds, cache_path)
    print(f"  Image embeddings: {all_embeds.shape}")
    return all_embeds


def extract_text_embeddings(df, embed_dim=64):
    """Extract text embeddings using a simple learnable bag-of-words."""
    cache_path = "multimodal-gnn/text_embeddings.pt"
    if os.path.exists(cache_path):
        data = torch.load(cache_path, weights_only=True)
        print(f"  Loaded cached text embeddings: {data.shape}")
        return data

    print("  Extracting text embeddings (BoW + projection) ...")
    from collections import Counter

    # Build vocab
    all_words = []
    for text in df["productDisplayName"]:
        all_words.extend(str(text).lower().split())
    word_counts = Counter(all_words)
    vocab = {w: i for i, (w, c) in enumerate(word_counts.most_common()) if c >= 2}
    vocab_size = len(vocab)

    # BoW encoding
    bow_matrix = np.zeros((len(df), vocab_size), dtype=np.float32)
    for i, text in enumerate(df["productDisplayName"]):
        for w in str(text).lower().split():
            if w in vocab:
                bow_matrix[i, vocab[w]] += 1

    # TF-IDF-like: normalize by document length
    row_sums = bow_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    bow_matrix /= row_sums

    # Project to embed_dim via SVD (deterministic, no training needed)
    bow_tensor = torch.FloatTensor(bow_matrix)
    U, S, V = torch.svd_lowrank(bow_tensor, q=embed_dim)
    text_embeds = U * S.unsqueeze(0)
    text_embeds = F.normalize(text_embeds, dim=1)

    torch.save(text_embeds, cache_path)
    print(f"  Text embeddings: {text_embeds.shape}")
    return text_embeds


def build_knn_edges(embeddings, k=5):
    """Build k-NN graph edges based on cosine similarity."""
    # Compute cosine similarity matrix in chunks to save memory
    N = embeddings.size(0)
    edges_src, edges_dst = [], []

    chunk_size = 1000
    emb_norm = F.normalize(embeddings, dim=1)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        sim = torch.mm(emb_norm[start:end], emb_norm.t())  # (chunk, N)
        # Zero out self-similarity
        for i in range(end - start):
            sim[i, start + i] = -1

        _, topk_idx = sim.topk(k, dim=1)
        for i in range(end - start):
            global_i = start + i
            for j in topk_idx[i]:
                edges_src.append(global_i)
                edges_dst.append(j.item())

    return torch.tensor([edges_src, edges_dst], dtype=torch.long)


def prepare_all(subset_size=5000):
    """Load data, extract embeddings, return everything needed."""
    print("[1] Loading dataset ...")
    df, label_map = load_dataset()

    # Use a subset for tractability
    if len(df) > subset_size:
        df = df.sample(subset_size, random_state=42).reset_index(drop=True)
        print(f"  Using subset of {subset_size} products")

    print(f"  Products: {len(df)}, Classes: {len(label_map)}")
    print(f"  Categories: {label_map}")

    print("\n[2] Extracting embeddings ...")
    img_embeds = extract_image_embeddings(df, embed_dim=64)
    text_embeds = extract_text_embeddings(df, embed_dim=64)
    labels = torch.tensor(df["label"].values, dtype=torch.long)

    return df, label_map, img_embeds, text_embeds, labels


if __name__ == "__main__":
    df, label_map, img_embeds, text_embeds, labels = prepare_all()
    print(f"\nImage embeddings: {img_embeds.shape}")
    print(f"Text embeddings: {text_embeds.shape}")
    print(f"Labels: {labels.shape}, {len(label_map)} classes")
