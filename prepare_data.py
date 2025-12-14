# save_mnist_train_digit_means.py
# Save MNIST train per-digit mean images to: out/mnist_train_digit_means.npy
# Adds MNIST auto-download (train)

import os
import numpy as np

MNIST_ROOT = "./mnist"
OUT_PATH = "out/mnist_train_digit_means.npy"


def ensure_mnist_downloaded(root=MNIST_ROOT):
    os.makedirs(root, exist_ok=True)
    try:
        from torchvision.datasets import MNIST
    except ImportError as e:
        raise ImportError(
            "torchvision is required to download/load MNIST. Install with: pip install torchvision"
        ) from e
    MNIST(root=root, train=True, download=True)


def load_mnist_train(root=MNIST_ROOT):
    ensure_mnist_downloaded(root)
    from torchvision.datasets import MNIST
    tr = MNIST(root=root, train=True, download=False)
    x = tr.data.numpy().astype(np.float32) / 255.0  # (60000,28,28)
    y = tr.targets.numpy().astype(np.int64)
    return x, y


def main():
    x, y = load_mnist_train(MNIST_ROOT)

    means = np.zeros((10, 28, 28), dtype=np.float32)
    counts = np.zeros(10, dtype=np.int64)

    for d in range(10):
        idxs = np.where(y == d)[0]
        counts[d] = idxs.size
        means[d] = x[idxs].mean(axis=0)

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    np.save(OUT_PATH, means)
    np.save(OUT_PATH.replace(".npy", "_counts.npy"), counts)

    print("saved:", OUT_PATH)
    print("saved:", OUT_PATH.replace(".npy", "_counts.npy"))
    print("shape:", means.shape, "dtype:", means.dtype)


if __name__ == "__main__":
    main()
