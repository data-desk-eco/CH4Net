"""Comprehensive evaluation of CH4Net model predictions.

Computes pixel-level and sample-level segmentation metrics on val/test splits.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    roc_curve,
)


def pixel_metrics(preds: list[np.ndarray], targets: list[np.ndarray], threshold: float = 0.5):
    """Compute pixel-level TP/FP/FN/TN across all samples."""
    tp = fp = fn = tn = 0
    for pred, target in zip(preds, targets):
        prob = 1 / (1 + np.exp(-pred))  # sigmoid (preds are logits)
        binary = (prob >= threshold).astype(float)
        t = (target > 0).astype(float)

        tp += ((binary == 1) & (t == 1)).sum()
        fp += ((binary == 1) & (t == 0)).sum()
        fn += ((binary == 0) & (t == 1)).sum()
        tn += ((binary == 0) & (t == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def sample_detection_metrics(preds: list[np.ndarray], targets: list[np.ndarray], threshold: float = 0.5):
    """Sample-level detection: does the model detect the presence of a plume at all?"""
    true_pos = false_pos = false_neg = true_neg = 0
    for pred, target in zip(preds, targets):
        has_plume_gt = target.max() > 0
        prob = 1 / (1 + np.exp(-pred))
        has_plume_pred = prob.max() >= threshold

        if has_plume_gt and has_plume_pred:
            true_pos += 1
        elif not has_plume_gt and has_plume_pred:
            false_pos += 1
        elif has_plume_gt and not has_plume_pred:
            false_neg += 1
        else:
            true_neg += 1

    total = true_pos + false_pos + false_neg + true_neg
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (true_pos + true_neg) / total if total > 0 else 0.0,
        "tp": true_pos,
        "fp": false_pos,
        "fn": false_neg,
        "tn": true_neg,
    }


def threshold_sweep(preds: list[np.ndarray], targets: list[np.ndarray]):
    """Sweep thresholds and return best pixel-F1 and its threshold."""
    best_f1, best_thresh = 0.0, 0.5
    results = []
    for t in np.arange(0.1, 0.95, 0.05):
        m = pixel_metrics(preds, targets, threshold=t)
        results.append((t, m))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = t
    return best_thresh, best_f1, results


def plot_curves(preds: list[np.ndarray], targets: list[np.ndarray], output_dir: Path):
    """Plot precision-recall and ROC curves."""
    all_pred = np.concatenate([1 / (1 + np.exp(-p.ravel())) for p in preds])
    all_target = np.concatenate([(t > 0).ravel().astype(int) for t in targets])

    # Subsample for speed if huge
    if len(all_pred) > 2_000_000:
        idx = np.random.default_rng(42).choice(len(all_pred), 2_000_000, replace=False)
        all_pred = all_pred[idx]
        all_target = all_target[idx]

    # Precision-Recall curve
    pr_precision, pr_recall, _ = precision_recall_curve(all_target, all_pred)
    ap = average_precision_score(all_target, all_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(pr_recall, pr_precision, lw=2)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title(f"Precision-Recall (AP={ap:.3f})")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    # ROC curve
    fpr, tpr, _ = roc_curve(all_target, all_pred)
    auc = roc_auc_score(all_target, all_pred)

    axes[1].plot(fpr, tpr, lw=2)
    axes[1].plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(f"ROC (AUC={auc:.3f})")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "curves.png", dpi=150)
    plt.close()

    return ap, auc


def plot_predictions(preds: list[np.ndarray], targets: list[np.ndarray],
                     output_dir: Path, threshold: float = 0.5, n: int = 10):
    """Save a grid of sample predictions vs. ground truth."""
    # Pick positive samples
    pos_idx = [i for i, t in enumerate(targets) if t.max() > 0]
    rng = np.random.default_rng(42)
    chosen = rng.choice(pos_idx, min(n, len(pos_idx)), replace=False)

    fig, axes = plt.subplots(len(chosen), 3, figsize=(10, 3.5 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(chosen):
        prob = 1 / (1 + np.exp(-preds[idx]))
        binary = (prob >= threshold).astype(float)
        target = targets[idx]

        axes[row, 0].imshow(target, cmap="Reds", vmin=0, vmax=1)
        axes[row, 0].set_title(f"GT (sample {idx})")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(prob, cmap="hot", vmin=0, vmax=1)
        axes[row, 1].set_title("Predicted prob")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(binary, cmap="Reds", vmin=0, vmax=1)
        axes[row, 2].set_title("Predicted binary")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "sample_predictions.png", dpi=150)
    plt.close()


def plot_loss_curve(losses: np.ndarray, output_dir: Path):
    """Plot training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title(f"Training Curve (best={losses.min():.4f} @ epoch {losses.argmin()})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate CH4Net predictions")
    parser.add_argument("--preds-dir", type=str, required=True, help="Dir with preds.npy and targets.npy")
    parser.add_argument("--model-dir", type=str, default=None, help="Dir with losses.npy for loss curve plot")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    preds_dir = Path(args.preds_dir)
    preds = np.load(preds_dir / "preds.npy", allow_pickle=True)
    targets = np.load(preds_dir / "targets.npy", allow_pickle=True)
    preds = list(preds)
    targets = list(targets)

    n_pos = sum(1 for t in targets if t.max() > 0)
    n_neg = len(targets) - n_pos
    print(f"Loaded {len(preds)} samples ({n_pos} positive, {n_neg} negative)")

    # 1. Pixel-level metrics at default threshold
    print(f"\n{'='*60}")
    print(f"PIXEL-LEVEL METRICS (threshold={args.threshold})")
    print(f"{'='*60}")
    pm = pixel_metrics(preds, targets, args.threshold)
    for k, v in pm.items():
        print(f"  {k:>12s}: {v:.4f}" if isinstance(v, float) else f"  {k:>12s}: {v}")

    # 2. Sample-level detection
    print(f"\n{'='*60}")
    print(f"SAMPLE-LEVEL DETECTION (threshold={args.threshold})")
    print(f"{'='*60}")
    sm = sample_detection_metrics(preds, targets, args.threshold)
    for k, v in sm.items():
        print(f"  {k:>12s}: {v:.4f}" if isinstance(v, float) else f"  {k:>12s}: {v}")

    # 3. Threshold sweep
    print(f"\n{'='*60}")
    print("THRESHOLD SWEEP (pixel F1)")
    print(f"{'='*60}")
    best_thresh, best_f1, sweep = threshold_sweep(preds, targets)
    for t, m in sweep:
        marker = " <-- best" if abs(t - best_thresh) < 0.01 else ""
        print(f"  t={t:.2f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  IoU={m['iou']:.3f}{marker}")
    print(f"\n  Best pixel F1: {best_f1:.4f} at threshold={best_thresh:.2f}")

    # 4. PR/ROC curves
    print(f"\n{'='*60}")
    print("CURVES")
    print(f"{'='*60}")
    ap, auc = plot_curves(preds, targets, preds_dir)
    print(f"  Average Precision (AP): {ap:.4f}")
    print(f"  ROC AUC:                {auc:.4f}")

    # 5. Sample predictions visualization
    plot_predictions(preds, targets, preds_dir, threshold=best_thresh)
    print(f"\n  Saved sample predictions to {preds_dir / 'sample_predictions.png'}")

    # 6. Loss curve
    if args.model_dir:
        losses = np.load(Path(args.model_dir) / "losses.npy")
        plot_loss_curve(losses, preds_dir)
        print(f"  Saved loss curve to {preds_dir / 'loss_curve.png'}")

    print(f"\n  All plots saved to {preds_dir}/")


if __name__ == "__main__":
    main()
