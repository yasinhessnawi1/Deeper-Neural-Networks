"""
Assignment 5 - Part A: Occlusion Sensitivity (From Scratch)
============================================================
Slides a patch over CIFAR-10 images and measures prediction
confidence change. Produces heatmaps for correct and incorrect
predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Model (same ResNet-like from Assignment 3) ──────────────────────────────

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


# ── Data ─────────────────────────────────────────────────────────────────────

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

CLASSES = test_set.classes
MEAN = np.array([0.4914, 0.4822, 0.4465])
STD = np.array([0.2470, 0.2435, 0.2616])


# ── Train model ──────────────────────────────────────────────────────────────

print("\n[1/3] Training ResNet-like model (15 epochs) ...")
model = ResNetLike().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, 16):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    scheduler.step()

    if epoch % 5 == 0 or epoch == 1:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(1) == labels).sum().item()
                total += labels.size(0)
        print(f"  Epoch {epoch:2d}/15  test_acc={correct/total:.4f}")

# Save model for other parts
torch.save(model.state_dict(), "explainable-ai/resnet_cifar10.pt")
print("  Model saved -> explainable-ai/resnet_cifar10.pt")

# ── Find correct and incorrect predictions ───────────────────────────────────

print("\n[2/3] Finding correct and misclassified images ...")
model.eval()

# Raw test set for display
raw_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())

correct_imgs = []
incorrect_imgs = []

with torch.no_grad():
    for i in range(len(test_set)):
        img_tensor = test_set[i][0].unsqueeze(0).to(device)
        label = test_set[i][1]
        pred = model(img_tensor).argmax(1).item()
        raw_img = raw_test[i][0]  # unnormalized

        if pred == label and len(correct_imgs) < 5:
            correct_imgs.append((raw_img, img_tensor.squeeze(0), label, pred))
        elif pred != label and len(incorrect_imgs) < 5:
            incorrect_imgs.append((raw_img, img_tensor.squeeze(0), label, pred))

        if len(correct_imgs) >= 5 and len(incorrect_imgs) >= 5:
            break

print(f"  Found {len(correct_imgs)} correct, {len(incorrect_imgs)} misclassified")


# ── Occlusion sensitivity ────────────────────────────────────────────────────

def occlusion_sensitivity(model, img_tensor, true_class, patch_size=4, stride=2):
    """Slide a patch over the image and measure confidence drop."""
    model.eval()
    _, h, w = img_tensor.shape

    # Baseline confidence
    with torch.no_grad():
        base_probs = F.softmax(model(img_tensor.unsqueeze(0).to(device)), dim=1)
        base_conf = base_probs[0, true_class].item()

    heatmap = np.zeros((h, w))
    count = np.zeros((h, w))

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            occluded = img_tensor.clone()
            # Replace patch with mean (gray)
            occluded[:, y:y+patch_size, x:x+patch_size] = 0.0

            with torch.no_grad():
                probs = F.softmax(model(occluded.unsqueeze(0).to(device)), dim=1)
                conf = probs[0, true_class].item()

            # Confidence drop = how important that region is
            drop = base_conf - conf
            heatmap[y:y+patch_size, x:x+patch_size] += drop
            count[y:y+patch_size, x:x+patch_size] += 1

    count[count == 0] = 1
    heatmap /= count
    return heatmap, base_conf


print("\n[3/3] Computing occlusion heatmaps ...")

def plot_occlusion_results(images_list, title, filename):
    n = len(images_list)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 3)

    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i, (raw_img, norm_img, label, pred) in enumerate(images_list):
        heatmap, base_conf = occlusion_sensitivity(model, norm_img, label, patch_size=4, stride=2)

        # Original image
        img_np = raw_img.permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"True: {CLASSES[label]}\nPred: {CLASSES[pred]}", fontsize=9)
        axes[i, 0].axis("off")

        # Heatmap alone
        im = axes[i, 1].imshow(heatmap, cmap="hot", interpolation="bilinear")
        axes[i, 1].set_title(f"Occlusion map\n(conf={base_conf:.3f})", fontsize=9)
        axes[i, 1].axis("off")
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046)

        # Overlay
        axes[i, 2].imshow(img_np)
        axes[i, 2].imshow(heatmap, cmap="jet", alpha=0.5, interpolation="bilinear")
        axes[i, 2].set_title("Overlay", fontsize=9)
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(f"explainable-ai/{filename}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved -> explainable-ai/{filename}")


plot_occlusion_results(correct_imgs, "Occlusion Sensitivity -- Correct Predictions", "occlusion_correct.png")
plot_occlusion_results(incorrect_imgs, "Occlusion Sensitivity -- Misclassified", "occlusion_incorrect.png")

print("\nDone.")
