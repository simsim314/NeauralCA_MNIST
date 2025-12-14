#!/usr/bin/env python3

# Convert mnist_train_digit_means.npy (10,28,28) → means.json (10,784) for the browser.
#
# Usage:
#   python3 tools/npy_to_json.py --npy out/mnist_train_digit_means.npy --out site/assets/means.json
#
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if not args.npy.is_file():
        raise FileNotFoundError(args.npy)

    means = np.load(args.npy).astype(np.float32)
    if means.shape != (10, 28, 28):
        raise ValueError(f"Expected (10,28,28), got {means.shape}")

    means = means.reshape(10, 28 * 28)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(means.tolist(), f)

    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
