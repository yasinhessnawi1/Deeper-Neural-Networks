"""
Assignment 5 - Part C: LIME (Model-Agnostic Local Explanation)
===============================================================
Uses LIME to explain individual CIFAR-10 predictions by
identifying important superpixels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Model (same as Part A) ──────────────────────────────────────────────────

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


# ── Load model ──────────────────────────────────────────────────────────────

print("\n[1/3] Loading trained model ...")
model = ResNetLike().to(device)
model.load_state_dict(torch.load("explainable-ai/resnet_cifar10.pt", map_location=device, weights_only=True))
model.eval()
print("  Loaded resnet_cifar10.pt")

# ── Data ─────────────────────────────────────────────────────────────────────

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
raw_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())
test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
CLASSES = test_set.classes
MEAN = np.array([0.4914, 0.4822, 0.4465])
STD = np.array([0.2470, 0.2435, 0.2616])


# ── Prediction function for LIME ─────────────────────────────────────────────

def predict_batch(images_np):
    """LIME passes numpy arrays (N, H, W, 3) in [0, 1]. Return class probabilities."""
    # Normalize
    images_np = (images_np - MEAN) / STD
    # to torch: (N, 3, H, W)
    batch = torch.FloatTensor(images_np.transpose(0, 3, 1, 2)).to(device)
    with torch.no_grad():
        probs = F.softmax(model(batch), dim=1)
    return probs.cpu().numpy()


# ── Select images ────────────────────────────────────────────────────────────

print("\n[2/3] Selecting images ...")
selected = []

with torch.no_grad():
    for i in range(len(test_set)):
        img_tensor = test_set[i][0].unsqueeze(0).to(device)
        label = test_set[i][1]
        pred = model(img_tensor).argmax(1).item()
        raw_img = raw_test[i][0].permute(1, 2, 0).numpy()  # (H,W,3) in [0,1]

        selected.append((raw_img, label, pred))
        if len(selected) >= 8:
            break

print(f"  Selected {len(selected)} images")

# ── LIME explanations ────────────────────────────────────────────────────────

print("\n[3/3] Computing LIME explanations ...")
explainer = lime_image.LimeImageExplainer()

n = len(selected)
fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n))
if n == 1:
    axes = axes.reshape(1, 3)

fig.suptitle("LIME Explanations", fontsize=13, fontweight="bold")

for i, (img_np, label, pred) in enumerate(selected):
    print(f"  Image {i+1}/{n}: true={CLASSES[label]}, pred={CLASSES[pred]}")

    explanation = explainer.explain_instance(
        img_np, predict_batch,
        top_labels=3, hide_color=0,
        num_samples=500, batch_size=64,
    )

    # Get mask for predicted class
    temp, mask = explanation.get_image_and_mask(
        pred, positive_only=True, num_features=5, hide_rest=False
    )

    # Also get positive+negative
    temp_pn, mask_pn = explanation.get_image_and_mask(
        pred, positive_only=False, num_features=5, hide_rest=False
    )

    # Original
    axes[i, 0].imshow(img_np)
    status = "OK" if pred == label else "WRONG"
    axes[i, 0].set_title(f"True: {CLASSES[label]}\nPred: {CLASSES[pred]} ({status})", fontsize=8)
    axes[i, 0].axis("off")

    # Positive superpixels only
    axes[i, 1].imshow(mark_boundaries(temp, mask))
    axes[i, 1].set_title("Positive superpixels", fontsize=8)
    axes[i, 1].axis("off")

    # Positive + negative
    axes[i, 2].imshow(mark_boundaries(temp_pn, mask_pn))
    axes[i, 2].set_title("Positive + Negative", fontsize=8)
    axes[i, 2].axis("off")

plt.tight_layout()
plt.savefig("explainable-ai/lime_explanations.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved -> explainable-ai/lime_explanations.png")

print("\nDone.")
