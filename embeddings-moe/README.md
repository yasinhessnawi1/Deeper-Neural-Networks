# Embeddings and Mixture of Experts

Assignment 4 for the Deeper Neural Networks course. Covers learned embeddings (autoencoder, contrastive), foundation model embeddings (CLIP), and mixture-of-experts with a trainable router.

## Contents

| File | Description |
|------|-------------|
| `part_a_embeddings.py` | A1: Autoencoder embeddings on Fashion-MNIST; A2: Contrastive (Siamese) embeddings. Saves embeddings for Part B. |
| `part_b_foundation.py` | Extracts CLIP ViT-B/32 embeddings, compares with Part A via k-NN and t-SNE. Requires Part A to be run first. |
| `part_c_moe.py` | Mixture of Experts: trains a gating network on frozen ResNet-like, Inception-like, SqueezeNet-like, and SuperNet experts on CIFAR-10. |
| `report.tex` | LaTeX report for the assignment |

## Running

From the project root (where `requirements.txt` lives):

```bash
pip install -r requirements.txt
python embeddings-moe/part_a_embeddings.py
python embeddings-moe/part_b_foundation.py
python embeddings-moe/part_c_moe.py
```

**Order matters:** Part B loads `part_a_embeddings.pt` produced by Part A. Run Part A first.

Part A and B use Fashion-MNIST; Part C uses CIFAR-10. Part B downloads CLIP on first run.
