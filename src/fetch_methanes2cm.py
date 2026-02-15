"""Fetch Sentinel-2 L2A imagery for MethaneS2CM plume mask locations.

Uses the georeferenced plume masks from MethaneS2CM to download coincident
Sentinel-2 L2A data from Earth Search STAC. Outputs (H, W, 12) uint8 arrays
matching the CH4Net training data format, paired with binary plume masks.
"""

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds, transform
from rasterio.windows import from_bounds
from pystac_client import Client


BAND_KEYS = [
    "coastal", "blue", "green", "red",
    "rededge1", "rededge2", "rededge3", "nir",
    "nir08", "nir09", "swir16", "swir22",
]


def load_plume_catalog(csv_path: str, plume_dir: str, n: int = 50) -> list[dict]:
    """Load plume metadata from MethaneS2CM CSV, deduplicate by plume_id."""
    seen = set()
    plumes = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["plume_id"]
            sid = row["sample_id"]
            if pid in seen:
                continue

            mask_path = Path(plume_dir) / sid / "plume.tif"
            if not mask_path.exists():
                continue

            # Read mask georef
            with rasterio.open(str(mask_path)) as src:
                mask_data = src.read(1)
                if mask_data.max() == 0:
                    continue  # skip empty masks
                bounds = src.bounds
                crs = src.crs

            seen.add(pid)
            plumes.append({
                "plume_id": pid,
                "sample_id": sid,
                "mask_path": str(mask_path),
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "datetime": row["datetime"],
                "emission_kg_hr": float(row["emission_auto"]),
                "bounds": bounds,
                "crs": str(crs),
                "mask_shape": mask_data.shape,
            })
            if len(plumes) >= n:
                break

    return plumes


def find_s2_scene(bounds, crs_str: str, date_str: str,
                  days_window: int = 30, max_cloud: float = 20.0):
    """Find Sentinel-2 L2A scene covering the plume mask bounds."""
    client = Client.open("https://earth-search.aws.element84.com/v1")

    # Convert mask bounds to EPSG:4326 for STAC search
    lon_min, lat_min, lon_max, lat_max = transform_bounds(
        crs_str, "EPSG:4326",
        bounds.left, bounds.bottom, bounds.right, bounds.top
    )

    if "T" in date_str:
        dt = datetime.fromisoformat(date_str.replace("+00", "+00:00").replace("Z", "+00:00"))
    else:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")

    start = (dt - timedelta(days=days_window)).strftime("%Y-%m-%d")
    end = (dt + timedelta(days=days_window)).strftime("%Y-%m-%d")

    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=[lon_min, lat_min, lon_max, lat_max],
        datetime=f"{start}/{end}",
        max_items=10,
    )

    items = list(search.items())
    if not items:
        return None

    valid = [i for i in items if i.properties.get("eo:cloud_cover", 100) <= max_cloud]
    if not valid:
        return None

    # Closest in time
    ts = dt.timestamp()
    valid.sort(key=lambda i: abs(i.datetime.timestamp() - ts))
    return valid[0]


def download_s2_for_mask(item, bounds, mask_crs: str,
                         target_shape: tuple[int, int]) -> np.ndarray | None:
    """Download 12 S2 bands matching the plume mask footprint.

    Handles CRS differences between mask and S2 scene by reprojecting bounds.
    All bands are read at 20m and resampled to target_shape.

    Returns (H, W, 12) uint8 array scaled from L2A reflectance.
    """
    target_h, target_w = target_shape

    # Use the swir16 band (20m native) as reference for scene CRS
    ref_key = next(k for k in ["swir16", "swir22", "blue"] if k in item.assets)
    with rasterio.open(item.assets[ref_key].href) as src:
        scene_crs = str(src.crs)

    # Reproject mask bounds to scene CRS if needed
    if mask_crs != scene_crs:
        left, bottom, right, top = transform_bounds(
            mask_crs, scene_crs,
            bounds.left, bounds.bottom, bounds.right, bounds.top
        )
    else:
        left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top

    bands = []
    for key in BAND_KEYS:
        if key not in item.assets:
            bands.append(np.zeros((target_h, target_w), dtype=np.uint8))
            continue

        with rasterio.open(item.assets[key].href) as src:
            window = from_bounds(left, bottom, right, top, transform=src.transform)
            data = src.read(1, window=window).astype(np.float32)

            # Resample to target shape if resolution differs
            if data.shape != (target_h, target_w) and data.size > 0:
                from PIL import Image
                img = Image.fromarray(data)
                img = img.resize((target_w, target_h), Image.BILINEAR)
                data = np.array(img)

            # Scale L2A reflectance (0-10000) to uint8 (0-255)
            data = np.clip(data, 0, 10000) * (255.0 / 10000.0)
            bands.append(data.astype(np.uint8))

    # Verify shapes
    shapes_ok = all(b.shape == (target_h, target_w) for b in bands)
    if not shapes_ok:
        # Crop to minimum as fallback
        min_h = min(b.shape[0] for b in bands)
        min_w = min(b.shape[1] for b in bands)
        if min_h < 16 or min_w < 16:
            return None
        bands = [b[:min_h, :min_w] for b in bands]

    return np.stack(bands, axis=-1)


def main():
    parser = argparse.ArgumentParser(description="Fetch S2 data for MethaneS2CM plume locations")
    parser.add_argument("--csv", type=str, default="data/methanes2cm/all.csv")
    parser.add_argument("--plume-dir", type=str, default="data/methanes2cm/plumes")
    parser.add_argument("--output-dir", type=str, default="data/methanes2cm_test")
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--days-window", type=int, default=30)
    parser.add_argument("--max-cloud", type=float, default=20.0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    (out / "s2").mkdir(parents=True, exist_ok=True)
    (out / "label").mkdir(parents=True, exist_ok=True)

    print("Loading plume catalog...")
    plumes = load_plume_catalog(args.csv, args.plume_dir, n=args.n_samples * 5)
    print(f"  {len(plumes)} unique plumes with non-empty masks")

    metadata = []
    downloaded = 0
    for i, plume in enumerate(plumes):
        if downloaded >= args.n_samples:
            break

        print(f"\n[{i+1}/{len(plumes)}] {plume['plume_id']}: "
              f"({plume['lat']:.4f}, {plume['lon']:.4f}), "
              f"{plume['emission_kg_hr']:.0f} kg/hr, {plume['datetime'][:10]}")

        item = find_s2_scene(
            plume["bounds"], plume["crs"], plume["datetime"],
            days_window=args.days_window, max_cloud=args.max_cloud,
        )
        if item is None:
            print("  No S2 scene found, skipping.")
            continue

        cloud = item.properties.get("eo:cloud_cover", 0)
        print(f"  S2: {item.id} (cloud={cloud:.1f}%)")

        try:
            patch = download_s2_for_mask(item, plume["bounds"], plume["crs"],
                                         plume["mask_shape"])
        except Exception as e:
            print(f"  Error: {e}")
            continue

        if patch is None:
            print("  Patch too small, skipping.")
            continue

        # Load and resize mask to match patch
        with rasterio.open(plume["mask_path"]) as src:
            mask = src.read(1).astype(np.float64)

        # Crop mask to match patch dimensions
        h, w = patch.shape[:2]
        mask = mask[:h, :w]

        print(f"  Patch: {patch.shape}, mask: {mask.shape}, "
              f"plume pixels: {(mask > 0).sum()}")

        np.save(str(out / "s2" / f"{downloaded}.npy"), patch)
        np.save(str(out / "label" / f"{downloaded}.npy"), mask)

        metadata.append({
            "idx": downloaded,
            "plume_id": plume["plume_id"],
            "lat": plume["lat"],
            "lon": plume["lon"],
            "datetime": plume["datetime"],
            "emission_kg_hr": plume["emission_kg_hr"],
            "s2_scene": item.id,
            "s2_cloud": cloud,
            "patch_shape": list(patch.shape),
            "plume_pixels": int((mask > 0).sum()),
        })
        downloaded += 1

    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nDone! Downloaded {downloaded} S2 patches with plume masks to {out}/")


if __name__ == "__main__":
    main()
