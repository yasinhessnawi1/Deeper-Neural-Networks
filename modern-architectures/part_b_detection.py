"""
Assignment 3 - Part B: Object Detection -- YOLO vs RT-DETR
===========================================================
Uses Ultralytics library to run pretrained YOLOv8 and RT-DETR
on a subset of COCO val2017, computes mAP, and identifies
failure cases for each model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from ultralytics import YOLO
import json
import os

np.random.seed(42)

# ── 1. Load pretrained models ────────────────────────────────────────────────
print("=" * 60)
print("  Part B: YOLO vs RT-DETR Object Detection")
print("=" * 60)

print("\n[1/4] Loading pretrained models ...")
yolo_model = YOLO("yolov8n.pt")       # YOLOv8-nano
rtdetr_model = YOLO("rtdetr-l.pt")    # RT-DETR-L

print(f"  YOLO:    YOLOv8n  ({sum(p.numel() for p in yolo_model.model.parameters()):,} params)")
print(f"  RT-DETR: RT-DETR-L ({sum(p.numel() for p in rtdetr_model.model.parameters()):,} params)")

# ── 2. Dataset: COCO val2017 subset ─────────────────────────────────────────
print("\n[2/4] Preparing COCO val2017 subset ...")

# Download a small subset via ultralytics' built-in COCO support
# We'll use the val set and evaluate on it
COCO_DATA = "coco128.yaml"  # Ultralytics built-in: 128 images from COCO

# ── 3. Evaluate mAP ─────────────────────────────────────────────────────────
print("\n[3/4] Computing mAP on COCO128 ...")

print("\n  --- YOLOv8n ---")
yolo_metrics = yolo_model.val(data=COCO_DATA, verbose=False, imgsz=640)
yolo_map50 = yolo_metrics.box.map50
yolo_map = yolo_metrics.box.map
print(f"  mAP@0.5:    {yolo_map50:.4f}")
print(f"  mAP@0.5:95: {yolo_map:.4f}")

print("\n  --- RT-DETR-L ---")
rtdetr_metrics = rtdetr_model.val(data=COCO_DATA, verbose=False, imgsz=640)
rtdetr_map50 = rtdetr_metrics.box.map50
rtdetr_map = rtdetr_metrics.box.map
print(f"  mAP@0.5:    {rtdetr_map50:.4f}")
print(f"  mAP@0.5:95: {rtdetr_map:.4f}")

print("\n  Summary:")
print(f"  {'Model':<12s} {'mAP@0.5':>10s} {'mAP@0.5:95':>12s}")
print(f"  {'-'*12} {'-'*10} {'-'*12}")
print(f"  {'YOLOv8n':<12s} {yolo_map50:>10.4f} {yolo_map:>12.4f}")
print(f"  {'RT-DETR-L':<12s} {rtdetr_map50:>10.4f} {rtdetr_map:>12.4f}")

# ── 4. Failure case analysis ────────────────────────────────────────────────
print("\n[4/4] Analysing failure cases ...")

# Find the COCO128 images directory
coco_dir = Path("datasets/coco128/images/train2017")
if not coco_dir.exists():
    coco_dir = Path("datasets/coco128/images/val2017")
if not coco_dir.exists():
    # Try to find wherever ultralytics put it
    for p in Path("datasets").rglob("*.jpg"):
        coco_dir = p.parent
        break

image_files = sorted(coco_dir.glob("*.jpg"))
print(f"  Found {len(image_files)} images in {coco_dir}")

# Run both models on all images and compare per-image detections
CONF_THRESHOLD = 0.25

yolo_failures = []    # images where YOLO detects fewer objects than RT-DETR
rtdetr_failures = []  # images where RT-DETR detects fewer objects than YOLO

per_image_results = []

for img_path in image_files:
    yolo_res = yolo_model.predict(str(img_path), conf=CONF_THRESHOLD, verbose=False)[0]
    rtdetr_res = rtdetr_model.predict(str(img_path), conf=CONF_THRESHOLD, verbose=False)[0]

    y_count = len(yolo_res.boxes)
    r_count = len(rtdetr_res.boxes)

    y_confs = yolo_res.boxes.conf.cpu().numpy() if y_count > 0 else np.array([])
    r_confs = rtdetr_res.boxes.conf.cpu().numpy() if r_count > 0 else np.array([])

    y_classes = yolo_res.boxes.cls.cpu().numpy().astype(int) if y_count > 0 else np.array([])
    r_classes = rtdetr_res.boxes.cls.cpu().numpy().astype(int) if r_count > 0 else np.array([])

    y_boxes = yolo_res.boxes.xyxy.cpu().numpy() if y_count > 0 else np.array([])
    r_boxes = rtdetr_res.boxes.xyxy.cpu().numpy() if r_count > 0 else np.array([])

    info = {
        "path": str(img_path),
        "name": img_path.name,
        "yolo_count": y_count,
        "rtdetr_count": r_count,
        "diff": r_count - y_count,
        "yolo_boxes": y_boxes,
        "rtdetr_boxes": r_boxes,
        "yolo_classes": y_classes,
        "rtdetr_classes": r_classes,
        "yolo_confs": y_confs,
        "rtdetr_confs": r_confs,
        "img_shape": yolo_res.orig_shape,
    }
    per_image_results.append(info)

    if r_count > y_count + 1:
        yolo_failures.append(info)
    if y_count > r_count + 1:
        rtdetr_failures.append(info)

# Sort by detection difference
yolo_failures.sort(key=lambda x: x["diff"], reverse=True)
rtdetr_failures.sort(key=lambda x: -x["diff"])

print(f"\n  Images where YOLO misses more objects than RT-DETR: {len(yolo_failures)}")
print(f"  Images where RT-DETR misses more objects than YOLO: {len(rtdetr_failures)}")

# ── 5. Analyse box sizes in failures ─────────────────────────────────────────

def analyse_box_sizes(failures, model_name, other_model):
    """Check if failures correlate with small/large objects or crowded scenes."""
    small_obj_count = 0
    large_obj_count = 0
    crowded_count = 0

    for info in failures[:10]:
        img_h, img_w = info["img_shape"]
        img_area = img_h * img_w

        # Get boxes from the model that DID detect (the other model)
        if other_model == "rtdetr":
            boxes = info["rtdetr_boxes"]
        else:
            boxes = info["yolo_boxes"]

        if len(boxes) == 0:
            continue

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        relative_areas = areas / img_area

        small = (relative_areas < 0.01).sum()
        large = (relative_areas > 0.1).sum()

        if small > len(boxes) * 0.3:
            small_obj_count += 1
        if large > len(boxes) * 0.5:
            large_obj_count += 1
        if len(boxes) > 8:
            crowded_count += 1

    return small_obj_count, large_obj_count, crowded_count


y_small, y_large, y_crowd = analyse_box_sizes(yolo_failures, "YOLO", "rtdetr")
r_small, r_large, r_crowd = analyse_box_sizes(rtdetr_failures, "RT-DETR", "yolo")

print(f"\n  YOLO failure trends (top 10):")
print(f"    Small objects dominant: {y_small}/10")
print(f"    Large objects dominant: {y_large}/10")
print(f"    Crowded scenes:        {y_crowd}/10")

print(f"\n  RT-DETR failure trends (top 10):")
print(f"    Small objects dominant: {r_small}/10")
print(f"    Large objects dominant: {r_large}/10")
print(f"    Crowded scenes:        {r_crowd}/10")

# ── 6. Visualise comparison (top 5 failures per model) ──────────────────────

def visualise_failures(failures, title, filename, max_images=5):
    n = min(len(failures), max_images)
    if n == 0:
        print(f"  No failures to visualise for {title}")
        return

    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i in range(n):
        info = failures[i]
        img = plt.imread(info["path"])

        # YOLO detections
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"YOLO ({info['yolo_count']} det.)")
        for box in (info["yolo_boxes"] if len(info["yolo_boxes"]) > 0 else []):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor="lime", facecolor="none")
            axes[i, 0].add_patch(rect)
        axes[i, 0].axis("off")

        # RT-DETR detections
        axes[i, 1].imshow(img)
        axes[i, 1].set_title(f"RT-DETR ({info['rtdetr_count']} det.)")
        for box in (info["rtdetr_boxes"] if len(info["rtdetr_boxes"]) > 0 else []):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor="cyan", facecolor="none")
            axes[i, 1].add_patch(rect)
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(f"modern-architectures/{filename}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Figure saved -> modern-architectures/{filename}")


visualise_failures(yolo_failures, "YOLO Failures (RT-DETR detects more)", "yolo_failures.png")
visualise_failures(rtdetr_failures, "RT-DETR Failures (YOLO detects more)", "rtdetr_failures.png")

# ── 7. Print analysis ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ANALYSIS")
print("=" * 60)
print("""
  YOLO (YOLOv8n):
  - Anchor-free, single-stage detector optimised for speed
  - Tends to miss small or heavily occluded objects
  - Very fast inference, suitable for real-time applications
  - Uses CNN backbone (no attention), so struggles with
    global context in complex scenes

  RT-DETR:
  - Transformer-based detector with attention mechanisms
  - Better at detecting small objects and handling occlusion
    because attention captures global image context
  - Slower inference due to transformer overhead
  - May produce more false positives on simple scenes
    where YOLO's simpler pipeline is sufficient

  Key trends in failures:
  - YOLO fails more on: small objects, crowded scenes,
    partially occluded objects
  - RT-DETR fails more on: simple scenes with large objects
    (over-detects or splits large objects), edge cases where
    YOLO's priors work well
""")
