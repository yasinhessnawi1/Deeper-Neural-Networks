"""
Assignment 5 - Part B: Grad-CAM (Model-Specific Explanation)
=============================================================
Uses pytorch-grad-cam to generate class activation heatmaps
for correct and incorrect predictions on CIFAR-10.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

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


# ── Load trained model ──────────────────────────────────────────────────────

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

test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
raw_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())
CLASSES = test_set.classes

# ── Find correct and incorrect predictions ───────────────────────────────────

print("\n[2/3] Finding correct and misclassified images ...")
correct_imgs = []
incorrect_imgs = []

with torch.no_grad():
    for i in range(len(test_set)):
        img_tensor = test_set[i][0].unsqueeze(0).to(device)
        label = test_set[i][1]
        pred = model(img_tensor).argmax(1).item()
        raw_img = raw_test[i][0]  # [0,1] range

        if pred == label and len(correct_imgs) < 5:
            correct_imgs.append((raw_img, test_set[i][0], label, pred))
        elif pred != label and len(incorrect_imgs) < 5:
            incorrect_imgs.append((raw_img, test_set[i][0], label, pred))

        if len(correct_imgs) >= 5 and len(incorrect_imgs) >= 5:
            break

print(f"  Found {len(correct_imgs)} correct, {len(incorrect_imgs)} misclassified")

# ── Grad-CAM ────────────────────────────────────────────────────────────────

print("\n[3/3] Computing Grad-CAM heatmaps ...")

# Target the last conv layer in stage3
target_layers = [model.stage3[-1].conv2]


def plot_gradcam(images_list, title, filename):
    n = len(images_list)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 3)

    fig.suptitle(title, fontsize=13, fontweight="bold")

    cam = GradCAM(model=model, target_layers=target_layers)

    for i, (raw_img, norm_img, label, pred) in enumerate(images_list):
        input_tensor = norm_img.unsqueeze(0).to(device)

        # Generate Grad-CAM for predicted class
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0]

        # Original image
        img_np = raw_img.permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"True: {CLASSES[label]}\nPred: {CLASSES[pred]}", fontsize=9)
        axes[i, 0].axis("off")

        # Grad-CAM heatmap
        im = axes[i, 1].imshow(grayscale_cam, cmap="jet")
        axes[i, 1].set_title("Grad-CAM", fontsize=9)
        axes[i, 1].axis("off")
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046)

        # Overlay
        overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay", fontsize=9)
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(f"explainable-ai/{filename}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved -> explainable-ai/{filename}")


plot_gradcam(correct_imgs, "Grad-CAM -- Correct Predictions", "gradcam_correct.png")
plot_gradcam(incorrect_imgs, "Grad-CAM -- Misclassified", "gradcam_incorrect.png")

print("\nDone.")
