"""Run CH4Net inference on real-world Sentinel-2 data and evaluate against plume masks.

Loads (H, W, 12) uint8 patches and optional label masks, runs the trained model,
and produces evaluation metrics + visualizations.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models import Unet
from train import get_device


def _pad_to_multiple(x: torch.Tensor, k: int = 16) -> tuple[torch.Tensor, tuple[int, int]]:
    _, h, w = x.shape
    ph = (math.ceil(h / k) * k) - h
    pw = (math.ceil(w / k) * k) - w
    return F.pad(x, (0, pw, 0, ph), mode="reflect"), (ph, pw)


def main():
    parser = argparse.ArgumentParser(description="Run CH4Net on real-world S2 data")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with s2/*.npy and optionally label/*.npy")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory with best.pt")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--channels", type=int, default=12, choices=[2, 5, 12])
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load model
    model = Unet(in_channels=args.channels, out_channels=1, div_factor=1, prob_output=False)
    ckpt = torch.load(Path(args.model_dir) / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded (epoch {ckpt['epoch']}, val_loss={ckpt['loss']:.4f})")

    data_dir = Path(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    s2_dir = data_dir / "s2"
    label_dir = data_dir / "label"
    has_labels = label_dir.exists()

    sample_ids = sorted(int(p.stem) for p in s2_dir.iterdir() if p.suffix == ".npy")
    print(f"Found {len(sample_ids)} samples (labels={'yes' if has_labels else 'no'})")

    # Load metadata if available
    meta_path = data_dir / "metadata.json"
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta_list = json.load(f)
            metadata = {m["idx"]: m for m in meta_list}

    preds_all, targets_all, results = [], [], []

    with torch.no_grad():
        for sid in sample_ids:
            s2 = np.load(str(s2_dir / f"{sid}.npy"))

            # Channel selection
            if args.channels == 2:
                s2 = s2[..., 10:]
            elif args.channels == 5:
                s2 = np.concatenate([s2[..., 1:4], s2[..., 10:]], axis=-1)

            x = torch.from_numpy(s2.copy()).float().permute(2, 0, 1) / 255.0
            x_padded, (ph, pw) = _pad_to_multiple(x)
            logits = model(x_padded.unsqueeze(0).to(device))[0, ..., 0].cpu()

            h, w = s2.shape[:2]
            logits = logits[:h, :w].numpy()
            prob = 1 / (1 + np.exp(-logits))
            binary = (prob >= args.threshold).astype(float)

            info = metadata.get(sid, {})
            plume_id = info.get("plume_id", f"sample_{sid}")

            result = {
                "idx": sid,
                "plume_id": plume_id,
                "pred_max_prob": float(prob.max()),
                "pred_plume_pixels": int(binary.sum()),
                "patch_shape": list(s2.shape[:2]),
            }

            if has_labels:
                label = np.load(str(label_dir / f"{sid}.npy"))
                label = label[:h, :w]
                gt_pixels = int((label > 0).sum())

                tp = ((binary == 1) & (label > 0)).sum()
                fp = ((binary == 1) & (label == 0)).sum()
                fn = ((binary == 0) & (label > 0)).sum()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

                result.update({
                    "gt_plume_pixels": gt_pixels,
                    "tp": int(tp), "fp": int(fp), "fn": int(fn),
                    "precision": float(precision),
                    "recall": float(recall),
                    "iou": float(iou),
                    "detected": bool(binary.sum() > 0 and gt_pixels > 0),
                })

                preds_all.append(logits)
                targets_all.append(label)

            results.append(result)

            detected = "DETECTED" if binary.sum() > 0 else "no detection"
            gt_str = f", GT={result.get('gt_plume_pixels', '?')}px" if has_labels else ""
            print(f"  [{sid}] {plume_id}: max_p={prob.max():.3f}, "
                  f"pred={int(binary.sum())}px{gt_str} → {detected}")

    # Save results
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    n_detected = sum(1 for r in results if r["pred_plume_pixels"] > 0)
    print(f"  Samples: {len(results)}")
    print(f"  Detections: {n_detected}/{len(results)}")

    if has_labels:
        n_with_gt = sum(1 for r in results if r.get("gt_plume_pixels", 0) > 0)
        n_correct = sum(1 for r in results if r.get("detected", False))
        n_false_alarm = sum(1 for r in results
                           if r["pred_plume_pixels"] > 0 and r.get("gt_plume_pixels", 0) == 0)

        print(f"  GT positives: {n_with_gt}")
        print(f"  Correctly detected: {n_correct}/{n_with_gt}")
        print(f"  False alarms: {n_false_alarm}")

        # Aggregate pixel metrics
        total_tp = sum(r.get("tp", 0) for r in results)
        total_fp = sum(r.get("fp", 0) for r in results)
        total_fn = sum(r.get("fn", 0) for r in results)

        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0

        print(f"\n  Aggregate pixel metrics:")
        print(f"    Precision: {p:.4f}")
        print(f"    Recall:    {r:.4f}")
        print(f"    F1:        {f1:.4f}")
        print(f"    IoU:       {iou:.4f}")

    # Visualization
    n_show = min(10, len(results))
    cols = 3 if has_labels else 2
    fig, axes = plt.subplots(n_show, cols, figsize=(4 * cols, 3.5 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for row, sid in enumerate(sample_ids[:n_show]):
        s2 = np.load(str(s2_dir / f"{sid}.npy"))
        logits_path = preds_all[row] if has_labels and row < len(preds_all) else None

        # RGB composite (bands 3,2,1 = R,G,B)
        rgb = s2[..., [3, 2, 1]].astype(float)
        rgb = np.clip(rgb / rgb.max() * 2, 0, 1) if rgb.max() > 0 else rgb

        axes[row, 0].imshow(rgb)
        info = metadata.get(sid, {})
        axes[row, 0].set_title(f"RGB ({info.get('plume_id', sid)})", fontsize=8)
        axes[row, 0].axis("off")

        # Prediction overlay
        if logits_path is not None:
            prob = 1 / (1 + np.exp(-logits_path))
        else:
            # Recompute
            if args.channels == 2:
                s2_c = s2[..., 10:]
            elif args.channels == 5:
                s2_c = np.concatenate([s2[..., 1:4], s2[..., 10:]], axis=-1)
            else:
                s2_c = s2
            x = torch.from_numpy(s2_c.copy()).float().permute(2, 0, 1) / 255.0
            x_padded, _ = _pad_to_multiple(x)
            with torch.no_grad():
                l = model(x_padded.unsqueeze(0).to(device))[0, ..., 0].cpu().numpy()
            prob = 1 / (1 + np.exp(-l[:s2.shape[0], :s2.shape[1]]))

        axes[row, 1].imshow(prob, cmap="hot", vmin=0, vmax=1)
        axes[row, 1].set_title(f"Pred (max={prob.max():.2f})", fontsize=8)
        axes[row, 1].axis("off")

        if has_labels and cols == 3:
            label = np.load(str(label_dir / f"{sid}.npy"))[:s2.shape[0], :s2.shape[1]]
            axes[row, 2].imshow(label, cmap="Reds", vmin=0, vmax=1)
            axes[row, 2].set_title(f"GT ({int((label > 0).sum())}px)", fontsize=8)
            axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out / "real_world_predictions.png", dpi=150)
    plt.close()
    print(f"\n  Saved visualization to {out / 'real_world_predictions.png'}")
    print(f"  Saved results to {out / 'results.json'}")


if __name__ == "__main__":
    main()
