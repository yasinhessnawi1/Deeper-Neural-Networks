# Modern Neural Network Architectures and Object Detection

Assignment 3 for the Deeper Neural Networks course. Covers modern CNN architectures (ResNet, Inception, SqueezeNet) and object detection (YOLO vs RT-DETR).

## Contents

| File | Description |
|------|-------------|
| `part_a_classification.py` | Image classification on CIFAR-10: trains Plain CNN, ResNet-like, Inception-like, SqueezeNet-like, and SuperNet (hybrid) models |
| `part_b_detection.py` | Object detection: compares pretrained YOLOv8n and RT-DETR-L on COCO128, computes mAP, and analyses failure cases |
| `report.tex` | LaTeX report for the assignment |

## Running

From the project root (where `requirements.txt` lives):

```bash
pip install -r requirements.txt
python modern-architectures/part_a_classification.py
python modern-architectures/part_b_detection.py
```

Part A downloads CIFAR-10 on first run and saves training curves to `part_a_training_curves.png`. Part B downloads pretrained YOLOv8 and RT-DETR weights and COCO128 on first run.
