# Depth, Architecture, and Loss Functions in Neural Networks

## Structure

| File | Description |
|------|-------------|
| `run_all.py` | Main runner — runs all parts or a specific one |
| `data_loading.py` | Data loading for Wine Quality and Fashion-MNIST |
| `utils.py` | Training loop, evaluation, plotting utilities |
| `part_a_shallow_vs_deep.py` | Part A: Shallow vs Deep FC networks |
| `part_b_loss_functions.py` | Part B: Loss function experiments |
| `part_c_cnns.py` | Part C: CNN architectures (Shallow, Deep, Residual) |
| `part_d_custom_loss.py` | Part D: Focal Loss (custom loss function) |
| `plots/` | Generated plots |

## Usage

```bash
# Install dependencies (from repo root)
pip install -r requirements.txt

# Run everything
cd depth-architecture-loss-functions
python run_all.py

# Run individual parts
python run_all.py --part a
python run_all.py --part b
python run_all.py --part c
python run_all.py --part d
```

## Datasets
- **Wine Quality** (Parts A, B, D): UCI ML Repository, red + white wine combined
- **Fashion-MNIST** (Parts C, D): 28x28 grayscale images, 10 clothing categories
