"""
Download the Fashion Product Images (Small) dataset from Kaggle.
Requires: pip install kaggle
          ~/.kaggle/kaggle.json with your API credentials

If Kaggle API is not available, download manually from:
  https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
and extract to gnn-ssm/data/
"""

import os
import subprocess
import sys

DATA_DIR = "gnn-ssm/data"

if os.path.exists(os.path.join(DATA_DIR, "styles.csv")):
    print("Dataset already exists in gnn-ssm/data/")
    sys.exit(0)

os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading Fashion Product Images (Small) from Kaggle ...")
try:
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "paramaggarwal/fashion-product-images-small",
        "-p", DATA_DIR, "--unzip"
    ], check=True)
    print("Done.")
except FileNotFoundError:
    print("ERROR: kaggle CLI not found. Install with: pip install kaggle")
    print("Then place your API key in ~/.kaggle/kaggle.json")
    sys.exit(1)
except subprocess.CalledProcessError as e:
    print(f"ERROR: Download failed: {e}")
    print("Download manually from:")
    print("  https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small")
    print(f"Extract to {DATA_DIR}/")
    sys.exit(1)
