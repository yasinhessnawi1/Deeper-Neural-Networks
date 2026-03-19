"""
Assignment 5 - Part D: Shapley (Model-Agnostic Global Explanation)
==================================================================
1. SHAP on Titanic dataset (tabular) for global feature importance
2. SHAP on CIFAR-10 images for pixel-wise explanations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

torch.manual_seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# D1: SHAP on Titanic (Tabular)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  D1: SHAP on Titanic Dataset")
print("=" * 60)

# Load Titanic data
print("\n[1/2] Loading Titanic dataset ...")
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
try:
    df = pd.read_csv(url)
except Exception:
    # Fallback: use seaborn's titanic
    import seaborn as sns
    df = sns.load_dataset("titanic")
    df = df.rename(columns={"survived": "Survived", "pclass": "Pclass", "sex": "Sex",
                            "age": "Age", "sibsp": "SibSp", "parch": "Parch",
                            "fare": "Fare", "embarked": "Embarked"})

# Preprocess
df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].dropna()
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])
df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"  Samples: {len(X_train)} train, {len(X_test)} test")
print(f"  Features: {list(X.columns)}")

# Train a Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print(f"  RF accuracy: {acc:.4f}")

# SHAP TreeExplainer
print("\n  Computing SHAP values ...")
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Summary plot
fig, ax = plt.subplots(figsize=(8, 5))
shap.summary_plot(shap_values[:, :, 1], X_test, show=False)
plt.title("SHAP Feature Importance (Titanic - Survived)")
plt.tight_layout()
plt.savefig("explainable-ai/shap_titanic_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> explainable-ai/shap_titanic_summary.png")

# Bar plot
fig, ax = plt.subplots(figsize=(8, 5))
shap.summary_plot(shap_values[:, :, 1], X_test, plot_type="bar", show=False)
plt.title("SHAP Mean Absolute Feature Importance (Titanic)")
plt.tight_layout()
plt.savefig("explainable-ai/shap_titanic_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> explainable-ai/shap_titanic_bar.png")


# ══════════════════════════════════════════════════════════════════════════════
# D2: SHAP on CIFAR-10 Images
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("  D2: SHAP on CIFAR-10 Images")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model ────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ResNetLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.stage1 = nn.Sequential(ResidualBlock(32), ResidualBlock(32))
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.stage2 = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.stage3 = nn.Sequential(ResidualBlock(128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(self.stage1(x))
        x = self.down2(self.stage2(x))
        x = self.stage3(x)
        return self.fc(self.pool(x).flatten(1))


print("\n[1/2] Loading model ...")
model = ResNetLike().to(device)
model.load_state_dict(torch.load("explainable-ai/resnet_cifar10.pt", map_location=device, weights_only=True))
model.eval()
print("  Loaded resnet_cifar10.pt")

# ── Data ─────────────────────────────────────────────────────────────────────

MEAN = np.array([0.4914, 0.4822, 0.4465])
STD = np.array([0.2470, 0.2435, 0.2616])
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

raw_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())

# Select a small set of images
N_IMAGES = 8
test_images = []
test_labels = []
for i in range(50):
    img = raw_test[i][0].permute(1, 2, 0).numpy()  # (32,32,3) in [0,1]
    test_images.append(img)
    test_labels.append(raw_test[i][1])
    if len(test_images) >= N_IMAGES:
        break

test_images = np.array(test_images)
print(f"  Selected {N_IMAGES} test images")

# Background for SHAP (small set)
bg_images = []
for i in range(100, 200):
    img = raw_test[i][0].permute(1, 2, 0).numpy()
    bg_images.append(img)
bg_images = np.array(bg_images)

# ── Prediction function ──────────────────────────────────────────────────────

def predict_fn(images_np):
    """Takes (N, H, W, 3) in [0,1], returns (N, 10) probabilities."""
    normed = (images_np - MEAN) / STD
    batch = torch.FloatTensor(normed.transpose(0, 3, 1, 2)).to(device)
    with torch.no_grad():
        probs = F.softmax(model(batch), dim=1)
    return probs.cpu().numpy()


# ── SHAP Partition Explainer ─────────────────────────────────────────────────

print("\n[2/2] Computing SHAP values for images ...")

masker = shap.maskers.Image("inpaint_telea", test_images[0].shape)
explainer = shap.Explainer(predict_fn, masker, output_names=CLASSES)

shap_values = explainer(
    test_images,
    max_evals=500,
    batch_size=64,
)

# Plot
fig, axes = plt.subplots(N_IMAGES, 2, figsize=(8, 3 * N_IMAGES))

for i in range(N_IMAGES):
    pred = predict_fn(test_images[i:i+1]).argmax(1)[0]
    label = test_labels[i]

    # Original
    axes[i, 0].imshow(test_images[i])
    status = "OK" if pred == label else "WRONG"
    axes[i, 0].set_title(f"True: {CLASSES[label]}, Pred: {CLASSES[pred]} ({status})", fontsize=8)
    axes[i, 0].axis("off")

    # SHAP values for predicted class
    sv = shap_values.values[i, :, :, :, pred]
    # Sum over channels for a single heatmap
    sv_sum = sv.sum(axis=-1)

    vmax = max(abs(sv_sum.min()), abs(sv_sum.max()))
    im = axes[i, 1].imshow(sv_sum, cmap="bwr", vmin=-vmax, vmax=vmax)
    axes[i, 1].set_title(f"SHAP (class: {CLASSES[pred]})", fontsize=8)
    axes[i, 1].axis("off")
    plt.colorbar(im, ax=axes[i, 1], fraction=0.046)

plt.suptitle("SHAP Pixel-wise Explanations (CIFAR-10)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("explainable-ai/shap_cifar10_images.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> explainable-ai/shap_cifar10_images.png")

print("\nDone.")
