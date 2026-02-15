"""Generate evaluation predictions from a trained CH4Net model."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models import Unet
from train import get_device


def _pad_to_multiple(x: torch.Tensor, k: int = 16) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad spatial dims (H, W) of a (C, H, W) tensor up to the next multiple of *k*."""
    _, h, w = x.shape
    ph = (math.ceil(h / k) * k) - h
    pw = (math.ceil(w / k) * k) - w
    # pad order: (left, right, top, bottom)
    return F.pad(x, (0, pw, 0, ph), mode="reflect"), (ph, pw)


def main():
    parser = argparse.ArgumentParser(description="Generate CH4Net predictions")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory with best.pt and losses.npy")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save predictions")
    parser.add_argument("--channels", type=int, default=12, choices=[2, 5, 12])
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load model
    model = Unet(in_channels=args.channels, out_channels=1, div_factor=1, prob_output=False)
    checkpoint = torch.load(Path(args.model_dir) / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']} (loss={checkpoint['loss']:.4f})")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load samples directly — pad to UNet-compatible size instead of cropping
    split_dir = Path(args.data_dir) / args.split
    label_dir = split_dir / "label"
    s2_dir = split_dir / "s2"
    sample_ids = sorted(int(p.stem) for p in label_dir.iterdir() if p.suffix == ".npy")

    preds, targets = [], []
    with torch.no_grad():
        for sid in sample_ids:
            label = np.load(str(label_dir / f"{sid}.npy"))
            s2 = np.load(str(s2_dir / f"{sid}.npy"))

            # Channel selection
            if args.channels == 2:
                s2 = s2[..., 10:]
            elif args.channels == 5:
                s2 = np.concatenate([s2[..., 1:4], s2[..., 10:]], axis=-1)

            x = torch.from_numpy(s2.copy()).float().permute(2, 0, 1) / 255.0  # (C, H, W)
            y = torch.from_numpy(label.copy()).float()

            x_padded, (ph, pw) = _pad_to_multiple(x)
            pred = model(x_padded.unsqueeze(0).to(device))[0, ..., 0].cpu()  # (H_pad, W_pad)

            # Remove padding
            h, w = label.shape
            pred = pred[:h, :w]

            preds.append(pred.numpy())
            targets.append(y.numpy())

    np.save(out / "preds.npy", np.array(preds, dtype=object), allow_pickle=True)
    np.save(out / "targets.npy", np.array(targets, dtype=object), allow_pickle=True)
    print(f"Saved {len(preds)} predictions to {out}")


if __name__ == "__main__":
    main()
