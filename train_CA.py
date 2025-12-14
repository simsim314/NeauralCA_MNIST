# mnist_dense_concat50_targets_mse.py
# 50 dense-concat blocks, each outputs (B,1,28,28) via sigmoid.
# Every 5 blocks (5,10,...,50) output is trained to match ideal digit template (MSE).
# Loss = mean MSE over all target checkpoints + small smoothness penalty between consecutive blocks.
# Also saves a checkpoint EACH epoch: out/mnist_dense_concat50_targets_mse_eXXX.pt
#
# Adds:
# - MNIST auto-download (train+test)
# - Auto-generate out/mnist_train_digit_means.npy if missing

import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


MNIST_ROOT = "./mnist"
IDEAL_PATH = "out/mnist_train_digit_means.npy"
OUT_DIR    = "out"
OUT_PATH   = os.path.join(OUT_DIR, "mnist_dense_concat50_targets_mse.pt")

NUM_BLOCKS = 50
HIDDEN = 16
TARGET_EVERY = 5  # 5,10,...,50

EPOCHS = 50
LR = 5e-4
BATCH_TRAIN = 256
BATCH_TEST  = 512
NUM_WORKERS = 4

KERNEL = 9
PAD = 4

LAMBDA_SMOOTH = 0.1  # small weight for mean-abs diff between outs[i] and outs[i+1]


def ensure_mnist_downloaded(root=MNIST_ROOT):
    os.makedirs(root, exist_ok=True)
    try:
        from torchvision.datasets import MNIST
    except ImportError as e:
        raise ImportError(
            "torchvision is required to download/load MNIST. Install with: pip install torchvision"
        ) from e
    MNIST(root=root, train=True, download=True)
    MNIST(root=root, train=False, download=True)


def load_mnist(root=MNIST_ROOT):
    ensure_mnist_downloaded(root)
    from torchvision.datasets import MNIST

    tr = MNIST(root=root, train=True, download=False)
    te = MNIST(root=root, train=False, download=False)

    xtr = (tr.data.numpy().astype(np.float32) / 255.0)[:, None, :, :]
    ytr = tr.targets.numpy().astype(np.int64)

    xte = (te.data.numpy().astype(np.float32) / 255.0)[:, None, :, :]
    yte = te.targets.numpy().astype(np.int64)
    return (xtr, ytr), (xte, yte)


def compute_and_save_digit_means(mnist_root=MNIST_ROOT, out_path=IDEAL_PATH):
    ensure_mnist_downloaded(mnist_root)
    from torchvision.datasets import MNIST

    tr = MNIST(root=mnist_root, train=True, download=False)
    x = tr.data.numpy().astype(np.float32) / 255.0  # (60000,28,28)
    y = tr.targets.numpy().astype(np.int64)

    means = np.zeros((10, 28, 28), dtype=np.float32)
    counts = np.zeros(10, dtype=np.int64)

    for d in range(10):
        idxs = np.where(y == d)[0]
        counts[d] = idxs.size
        means[d] = x[idxs].mean(axis=0)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, means)
    np.save(out_path.replace(".npy", "_counts.npy"), counts)
    print("generated:", out_path)
    print("generated:", out_path.replace(".npy", "_counts.npy"))


def load_ideals(device):
    if not os.path.isfile(IDEAL_PATH):
        compute_and_save_digit_means(MNIST_ROOT, IDEAL_PATH)

    means = np.load(IDEAL_PATH).astype(np.float32)  # (10,28,28)
    if means.shape != (10, 28, 28):
        raise ValueError(f"Expected (10,28,28), got {means.shape}")
    return torch.from_numpy(means)[:, None, :, :].to(device)


class DenseConcatBlocks(nn.Module):
    def __init__(self, num_blocks=NUM_BLOCKS, hidden=HIDDEN):
        super().__init__()
        self.num_blocks = int(num_blocks)
        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            in_ch = 1 + i
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, hidden, kernel_size=KERNEL, padding=PAD, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 1, kernel_size=1, padding=0, bias=True),
            ))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        feats = [x]
        outs = []
        for blk in self.blocks:
            inp = torch.cat(feats, dim=1)
            y = torch.sigmoid(blk(inp))
            outs.append(y)
            feats.append(y)
        return outs


def loss_terms(outs, tgt, target_blocks):
    mse = 0.0
    for b in target_blocks:
        mse = mse + F.mse_loss(outs[b - 1], tgt)
    mse = mse / float(len(target_blocks))

    smooth = 0.0
    for i in range(len(outs) - 1):
        smooth = smooth + torch.mean(torch.abs(outs[i + 1] - outs[i]))
    smooth = smooth / float(len(outs) - 1)

    total = mse + (LAMBDA_SMOOTH * smooth)
    return total, mse, smooth


@torch.no_grad()
def eval_loss(model, loader, device, ideals, target_blocks):
    model.eval()
    tot = 0.0
    tot_mse = 0.0
    tot_sm = 0.0
    seen = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        tgt = ideals.index_select(0, yb)
        outs = model(xb)
        loss, mse, sm = loss_terms(outs, tgt, target_blocks)

        bs = yb.size(0)
        tot += loss.item() * bs
        tot_mse += mse.item() * bs
        tot_sm += sm.item() * bs
        seen += bs

    return (tot / max(1, seen)), (tot_mse / max(1, seen)), (tot_sm / max(1, seen))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_pin = (device == "cuda")

    (xtr, ytr), (xte, yte) = load_mnist(MNIST_ROOT)
    ideals = load_ideals(device)

    train_ds = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr))
    test_ds  = TensorDataset(torch.from_numpy(xte), torch.from_numpy(yte))

    train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=use_pin)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_TEST, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=use_pin)

    model = DenseConcatBlocks(NUM_BLOCKS, HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    target_blocks = list(range(TARGET_EVERY, NUM_BLOCKS + 1, TARGET_EVERY))
    print("targets:", target_blocks)
    print(f"smooth penalty: lambda={LAMBDA_SMOOTH}")

    os.makedirs(OUT_DIR, exist_ok=True)

    for ep in range(1, EPOCHS + 1):
        model.train()
        run = 0.0
        run_mse = 0.0
        run_sm = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc=f"epoch {ep:03d}", unit="batch")
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            tgt = ideals.index_select(0, yb)

            opt.zero_grad(set_to_none=True)
            outs = model(xb)

            loss, mse, sm = loss_terms(outs, tgt, target_blocks)
            loss.backward()
            opt.step()

            bs = yb.size(0)
            run += loss.item() * bs
            run_mse += mse.item() * bs
            run_sm += sm.item() * bs
            seen += bs
            pbar.set_postfix(loss=run/seen, mse=run_mse/seen, smooth=run_sm/seen)

        te, te_mse, te_sm = eval_loss(model, test_loader, device, ideals, target_blocks)
        print(f"epoch {ep:03d} test_loss={te:.6f}  test_mse={te_mse:.6f}  test_smooth={te_sm:.6f}")

        ep_path = os.path.join(OUT_DIR, f"mnist_dense_concat50_targets_mse_e{ep:03d}.pt")
        torch.save(model.state_dict(), ep_path)
        print("saved:", ep_path)

    torch.save(model.state_dict(), OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
