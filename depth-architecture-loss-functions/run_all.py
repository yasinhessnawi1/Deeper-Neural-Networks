"""
Main runner: executes all 4 parts of the assignment sequentially.
Generates all plots in the plots/ directory.

Usage:
    python run_all.py          # Run all parts
    python run_all.py --part a # Run only Part A
    python run_all.py --part b # Run only Part B
    python run_all.py --part c # Run only Part C
    python run_all.py --part d # Run only Part D
"""

import argparse
import torch
from utils import DEVICE

def main():
    parser = argparse.ArgumentParser(description="Run assignment experiments")
    parser.add_argument("--part", type=str, default="all",
                        choices=["all", "a", "b", "c", "d"],
                        help="Which part to run (default: all)")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    if args.part in ("all", "a"):
        from part_a_shallow_vs_deep import run_part_a
        run_part_a()

    if args.part in ("all", "b"):
        from part_b_loss_functions import run_part_b
        run_part_b()

    if args.part in ("all", "c"):
        from part_c_cnns import run_part_c
        run_part_c()

    if args.part in ("all", "d"):
        from part_d_custom_loss import run_part_d
        run_part_d()

    print("\n" + "=" * 60)
    print("ALL DONE! Plots saved in plots/ directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
