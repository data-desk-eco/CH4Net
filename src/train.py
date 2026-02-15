"""CH4Net training script — methane plume segmentation from Sentinel-2 imagery."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import Unet


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MethaneDataset(Dataset):
    """Loads the HuggingFace-hosted CH4Net dataset.

    Expected layout:
        data_dir/{split}/label/{id}.npy   — (H, W) float64 binary mask
        data_dir/{split}/s2/{id}.npy      — (H, W, 12) uint8 Sentinel-2 bands
    """

    def __init__(self, data_dir: str, split: str, channels: int = 12, crop_size: int = 100):
        self.split_dir = Path(data_dir) / split
        self.channels = channels
        self.crop_size = crop_size
        self.is_train = split == "train"

        # Discover all sample IDs
        label_dir = self.split_dir / "label"
        all_ids = sorted(
            int(p.stem) for p in label_dir.iterdir() if p.suffix == ".npy"
        )

        # Separate positive and negative samples for balanced sampling
        self.pos_ids: list[int] = []
        self.neg_ids: list[int] = []
        for sid in all_ids:
            label = np.load(str(label_dir / f"{sid}.npy"))
            if label.max() > 0:
                self.pos_ids.append(sid)
            else:
                self.neg_ids.append(sid)

        self.ids: list[int] = []
        self.resample()

    def resample(self):
        """Balance negatives with positives (train only)."""
        if not self.is_train:
            self.ids = self.pos_ids + self.neg_ids
        else:
            neg = list(self.neg_ids)
            random.shuffle(neg)
            self.ids = self.pos_ids + neg[: len(self.pos_ids)]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sid = self.ids[index]

        label = np.load(str(self.split_dir / "label" / f"{sid}.npy"))
        s2 = np.load(str(self.split_dir / "s2" / f"{sid}.npy"))

        # Channel selection
        if self.channels == 2:
            s2 = s2[..., 10:]
        elif self.channels == 5:
            s2 = np.concatenate([s2[..., 1:4], s2[..., 10:]], axis=-1)

        # Random crop (train) or center crop (val/test)
        h, w = label.shape
        cs = self.crop_size
        if self.is_train:
            cx = random.randint(0, h - cs)
            cy = random.randint(0, w - cs)
        else:
            cx = (h - cs) // 2
            cy = (w - cs) // 2

        label = label[cx : cx + cs, cy : cy + cs]
        s2 = s2[cx : cx + cs, cy : cy + cs, :]

        # To tensors — (C, H, W) float32, normalised to [0, 1]
        x = torch.from_numpy(s2.copy()).float().permute(2, 0, 1) / 255.0
        y = torch.from_numpy(label.copy()).float()

        return {"input": x, "target": y}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(args, on_save=None):
    device = get_device()
    print(f"Using device: {device}")

    torch.manual_seed(0)

    # Datasets
    print("Loading datasets...")
    train_ds = MethaneDataset(args.data_dir, split="train", channels=args.channels, crop_size=args.crop_size)
    val_ds = MethaneDataset(args.data_dir, split="val", channels=args.channels, crop_size=args.crop_size)

    print(f"Train: {len(train_ds)} samples ({len(train_ds.pos_ids)} pos, {len(train_ds.neg_ids)} neg)")
    print(f"Val:   {len(val_ds)} samples ({len(val_ds.pos_ids)} pos, {len(val_ds.neg_ids)} neg)")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Model — prob_output=False so we get raw logits, paired with BCEWithLogitsLoss
    model = Unet(in_channels=args.channels, out_channels=1, div_factor=1, prob_output=False)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Output directory
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    losses = []

    for epoch in range(args.epochs):
        # Resample negatives each epoch
        train_ds.resample()
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

        # --- Train ---
        model.train()
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch") as pbar:
            for batch in pbar:
                x = batch["input"].to(device)
                y = batch["target"].to(device)

                pred = model(x)[..., 0]  # (B, H, W)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device)
                y = batch["target"].to(device)
                pred = model(x)[..., 0]
                val_losses.append(loss_fn(pred, y).item())

        val_loss = np.mean(val_losses)
        losses.append(val_loss)
        print(f"  Val loss: {val_loss:.4f} (best: {best_loss:.4f})")

        # Save best
        if val_loss <= best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                },
                out / "best.pt",
            )
            print(f"  Saved best model (epoch {epoch+1})")
            if on_save:
                on_save()

        np.save(out / "losses.npy", np.array(losses))

    print("Training complete!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train CH4Net")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to dataset root")
    parser.add_argument("--output-dir", type=str, default="output", help="Where to save checkpoints")
    parser.add_argument("--channels", type=int, default=12, choices=[2, 5, 12], help="Number of input bands")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--crop-size", type=int, default=100, help="Square crop size in pixels")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
