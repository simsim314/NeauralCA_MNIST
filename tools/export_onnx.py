#!/usr/bin/env python3

# Export your DenseConcatBlocks(.pt) to ONNX with 50 outputs (out00..out49).
#
# Usage:
#   python3 tools/export_onnx.py \
#     --pt  out/mnist_dense_concat50_targets_mse_1.pt \
#     --onnx site/assets/model.onnx \
#     --kernel 9 --hidden 16 --blocks 50
#
# Notes:
# - Exports a graph with fixed 50 steps (unrolled), returning all 50 intermediate outputs.
# - Input name: "x" shape [B,1,28,28]
# - Output names: out00..out49 each shape [B,1,28,28]
#
import argparse
from pathlib import Path
import torch
import torch.nn as nn

class DenseConcatBlocks(nn.Module):
    def __init__(self, num_blocks: int, hidden: int, kernel: int):
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

class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        outs = self.model(x)
        return tuple(outs)  # fixed outputs for ONNX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, type=Path, help="Path to .pt state_dict")
    ap.add_argument("--onnx", required=True, type=Path, help="Output ONNX path")
    ap.add_argument("--blocks", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--kernel", type=int, default=9)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    if not args.pt.is_file():
        raise FileNotFoundError(args.pt)

    model = DenseConcatBlocks(args.blocks, args.hidden, args.kernel).cpu().eval()
    sd = torch.load(str(args.pt), map_location="cpu")
    model.load_state_dict(sd, strict=True)

    wrapper = ExportWrapper(model).cpu().eval()

    dummy = torch.zeros(1, 1, 28, 28, dtype=torch.float32)

    output_names = [f"out{i:02d}" for i in range(args.blocks)]

    args.onnx.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(args.onnx),
        input_names=["x"],
        output_names=output_names,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes={"x": {0: "batch"}, **{n: {0: "batch"} for n in output_names}},
    )

    print("Wrote:", args.onnx)

if __name__ == "__main__":
    main()
