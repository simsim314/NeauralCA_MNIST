# vis_dense_concat50_targets_mse_loop.py
# Infinite loop: pick random MNIST sample, show block outputs.
# Every 5 blocks also show the closest mean digit template (by MSE) on the RIGHT.

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


MNIST_ROOT = "./mnist"
MEANS_PATH = "out/mnist_train_digit_means.npy"

MODEL_PATH = "out/mnist_dense_concat50_targets_mse_1.pt"
NUM_BLOCKS = 50
HIDDEN = 16
SHOW_MEAN_EVERY = 5 

KERNEL = 9
PAUSE_S = 0.12
SPLIT = "test"
SEED = None

SHOW_BLOCKS = list(range(1, NUM_BLOCKS + 1))  # 1..50


def load_mnist(split, root=MNIST_ROOT):
    from torchvision.datasets import MNIST
    ds = MNIST(root=root, train=(split == "train"), download=True)
    x = (ds.data.numpy().astype(np.float32) / 255.0)[:, None, :, :]  # (N,1,28,28)
    y = ds.targets.numpy().astype(np.int64)
    return x, y


def load_means(path=MEANS_PATH):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing means: {path}")
    means = np.load(path).astype(np.float32)  # (10,28,28)
    if means.shape != (10, 28, 28):
        raise ValueError(f"Expected means (10,28,28), got {means.shape}")
    return means


class DenseConcatBlocks(nn.Module):
    def __init__(self, num_blocks=NUM_BLOCKS, hidden=HIDDEN, kernel=KERNEL):
        super().__init__()
        self.num_blocks = int(num_blocks)
        self.kernel = int(kernel)
        pad = self.kernel // 2
        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            in_ch = 1 + i
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, hidden, kernel_size=self.kernel, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 1, kernel_size=1, padding=0, bias=True),
            ))

    def forward(self, x):
        feats = [x]
        outs = []
        for blk in self.blocks:
            inp = torch.cat(feats, dim=1)
            y = torch.sigmoid(blk(inp))
            outs.append(y)
            feats.append(y)
        return outs


def closest_mean_digit(img28, means10):
    # img28: (28,28) float32
    # means10: (10,28,28) float32
    diff = means10 - img28[None, :, :]
    mse = np.mean(diff * diff, axis=(1, 2))  # (10,)
    k = int(np.argmin(mse))
    return k, float(mse[k])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    means = load_means(MEANS_PATH)
    x, y = load_mnist(SPLIT, MNIST_ROOT)
    rng = np.random.default_rng(SEED)

    model = DenseConcatBlocks(NUM_BLOCKS, HIDDEN, KERNEL).to(device)
    sd = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    show = SHOW_BLOCKS if SHOW_BLOCKS is not None else list(range(1, NUM_BLOCKS + 1))
    show = [b for b in show if 1 <= b <= NUM_BLOCKS]

    plt.ion()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8, 4))
    axL.axis("off")
    axR.axis("off")

    imL = axL.imshow(np.zeros((28, 28), np.float32), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    imR = axR.imshow(np.zeros((28, 28), np.float32), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")

    title = fig.suptitle("", fontsize=12)
    axL.set_title("model output", fontsize=10)
    axR.set_title("closest mean digit", fontsize=10)
    plt.tight_layout()

    n = y.shape[0]

    while True:
        idx = int(rng.integers(0, n))
        true_d = int(y[idx])
        xb = torch.from_numpy(x[idx:idx+1]).to(device)  # (1,1,28,28)

        # show input on left, blank right
        imL.set_data(x[idx, 0])
        imR.set_data(np.zeros((28, 28), np.float32))
        title.set_text(f"true={true_d} idx={idx}  input")
        plt.pause(2*PAUSE_S)

        with torch.no_grad():
            outs = model(xb)

        for b in show:
            img = outs[b - 1][0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
            imL.set_data(img)

            right_txt = ""
            if (b % SHOW_MEAN_EVERY) == 0:
                pred_d, pred_mse = closest_mean_digit(img, means)
                imR.set_data(means[pred_d])
                right_txt = f"  | closest={pred_d} mse={pred_mse:.5f}"
            else:
                # keep previous right image; comment next line if you want it to freeze only on updates
                # imR.set_data(np.zeros((28, 28), np.float32))
                pass

            #title.set_text(f"true={true_d} idx={idx}  block={b:02d}/{NUM_BLOCKS}{right_txt}")
            plt.pause(PAUSE_S)


if __name__ == "__main__":
    main()
