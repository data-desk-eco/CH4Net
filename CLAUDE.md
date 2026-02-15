# CH4Net - Agent Reference

Fork of [anna-allen/CH4Net](https://github.com/anna-allen/CH4Net). Paper: Vaughan et al. (2024), *Atmos. Meas. Tech.*, 17, 2583–2593 (`amt-17-2583-2024.pdf` in repo root).

## Project setup

```bash
uv sync          # install deps (uses pyproject.toml + uv.lock)
uv sync --group dev  # also installs modal for remote training
```

Python >=3.10. Uses `uv` exclusively (not pip/conda).

## Architecture

U-Net binary segmentation model for detecting methane plumes in Sentinel-2 imagery. Trained on 23 known super-emitter sites in Turkmenistan (2017-2020), tested on 2021 images from the same sites.

- **Input**: (H, W, 12) uint8 Sentinel-2 L1C bands scaled to [0,255]
- **Output**: (H, W, 1) logits (use sigmoid for probabilities)
- **Band order** (indices 0-11): B01(coastal), B02(blue), B03(green), B04(red), B05-B07(rededge), B08(nir), B8A(nir08), B09(wvp), B11(swir16), B12(swir22)
- **Loss**: BCEWithLogitsLoss, **Optimizer**: Adam (lr=1e-4), **Crop**: 100x100 random (train), center (val/test)

## Key files

| File | Purpose |
|---|---|
| `src/models.py` | UNet + MLP definitions |
| `src/train.py` | Training pipeline + `MethaneDataset` class |
| `src/gen_eval_preds.py` | Generate predictions on val/test splits (pads to multiple of 16) |
| `src/evaluate.py` | Comprehensive metrics: pixel F1/IoU, sample detection, PR/ROC curves, threshold sweep, visualizations |
| `src/fetch_methanes2cm.py` | Download S2 L2A imagery at MethaneS2CM plume locations from Earth Search STAC |
| `src/fetch_real_world_data.py` | Download S2 at Carbon Mapper plume locations (less useful - those plumes are from EMIT, not S2) |
| `src/infer_real_world.py` | Run model on arbitrary S2 patches with optional labels, produces metrics + visualizations |
| `src/export_onnx.py` | Export PyTorch checkpoint to standalone ONNX file |
| `modal_train.py` | Modal remote training (T4 GPU, `ch4net-vol` volume) |

## Data (gitignored)

Training data is from HuggingFace `av555/ch4net`. On Modal, it's cached at `/vol/data` in the `ch4net-vol` volume.

```
data/
├── train/  (8256 samples)
├── val/    (256 samples)
└── test/   (2473 samples, 275 positive)
    ├── label/{id}.npy   # (H, W) float64 binary mask
    ├── s2/{id}.npy      # (H, W, 12) uint8
    └── mbmp/{id}.npy    # (H, W) multiband multipass (unused by model)
```

External datasets for real-world testing:
- **MethaneS2CM** (`Lucab95/MethaneS2CM_plume_only` on HF): 7251 georeferenced plume masks (512x512 GeoTIFF, 20m). Freely accessible. Use `fetch_methanes2cm.py` to download matching S2 imagery.
- **MARS-S2L** (`UNEP-IMEO/MARS-S2L` on HF): 87k S2/Landsat pairs, 5600+ plumes. Gated - requires access approval.

## Model checkpoints (gitignored)

```
output/modal_v2/best.pt       # current best (epoch 134, val_loss=0.072)
output/modal_v2/losses.npy    # 225-epoch loss curve
```

Download from Modal: `uv run modal volume get ch4net-vol output_v2/best.pt output/modal_v2/best.pt --force`

Checkpoint format: `{'epoch': int, 'model_state_dict': ..., 'optimizer_state_dict': ..., 'loss': float}`

Loading:
```python
from models import Unet
model = Unet(in_channels=12, out_channels=1, div_factor=1, prob_output=False)
ckpt = torch.load("output/modal_v2/best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
```

## Current performance (modal_v2)

### vs. paper (Table 1, ALL bands)

| Metric | Paper | Ours | Notes |
|---|---|---|---|
| Scene-level recall | 0.84 | **0.90** | We find more plumes |
| Scene-level precision | 0.30 | 0.31 | Comparable |
| Scene-level FPR | **0.24** | 0.26 | Comparable |
| **Pixel IoU** | **0.57** | **0.25** | Big gap - main issue |
| Pixel balanced acc | **0.66** | - | - |
| ROC AUC | - | 0.94 | Strong ranking |

### Why pixel IoU is lower than the paper

1. **Model capacity**: paper's encoder starts at 128 channels, ours at 64 (div_factor=1 gives 64//1=64). Paper is ~4x larger.
2. **Upsampling**: paper uses transposed convolutions; our rewrite uses bilinear upsampling (loses spatial precision).
3. **Resolution**: paper interpolates all bands to 10m; unclear if our data pipeline matches this exactly.

### Priority improvements

- Increase `div_factor` denominator effect or change initial channel count to 128 to match paper capacity
- Restore transposed convolution upsampling in decoder (set `self.bilinear = True` won't do it - need to change the `up` class default)
- Add learning rate scheduling (cosine annealing) - loss curve shows instability after epoch 150
- Try Dice/Focal loss to improve pixel-level segmentation
- Add data augmentation (rotations, flips)
- Hard negative mining to reduce FPR

### Real-world test (MethaneS2CM, Permian Basin)

20/20 known plume sites detected at sample level (max prob 0.57-0.99). Pixel-level comparison not meaningful because S2 scenes are from different dates than the airborne plume detections.

## Common commands

```bash
# Train locally
uv run python src/train.py --data-dir data --output-dir output --channels 12 --epochs 250

# Train on Modal
uv run modal run modal_train.py --epochs 250

# Generate predictions
uv run python src/gen_eval_preds.py --model-dir output/modal_v2 --output-dir output/modal_v2/eval_test --split test

# Evaluate
uv run python src/evaluate.py --preds-dir output/modal_v2/eval_test --model-dir output/modal_v2

# Fetch real-world S2 data at known plume sites
uv run python src/fetch_methanes2cm.py --n-samples 20

# Run inference on fetched data
uv run python src/infer_real_world.py --data-dir data/methanes2cm_test --model-dir output/modal_v2 --output-dir output/modal_v2/real_world_eval

# Export to ONNX
uv run python src/export_onnx.py --model-dir output/modal_v2
```

## S2 data access notes

- **Earth Search STAC** (`earth-search.aws.element84.com/v1`): No auth needed, COGs on public S3. Use collection `sentinel-2-l2a`.
- S2 L2A reflectance is uint16 0-10000. Scale by `255/10000` to match training data uint8 range.
- When reading S2 COGs with rasterio, always reproject mask bounds to the scene's CRS (UTM zones differ across tiles).
- 10m/20m/60m bands have different native resolutions - resample to target shape when mixing.
