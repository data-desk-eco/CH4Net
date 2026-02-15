"""Fetch real-world Sentinel-2 data at known methane plume locations.

Uses Carbon Mapper API for plume coordinates, then downloads coincident
Sentinel-2 L2A imagery from the Element84 Earth Search STAC catalog.
Outputs patches in the same (H, W, 12) uint8 format as the training data.
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform
from rasterio.windows import from_bounds
import requests
from pystac_client import Client


# Band keys in Earth Search STAC, ordered to match our 12-channel training data:
#   idx 0:B01, 1:B02, 2:B03, 3:B04, 4:B05, 5:B06, 6:B07, 7:B08, 8:B8A, 9:B09, 10:B11, 11:B12
BAND_KEYS = [
    "coastal", "blue", "green", "red",
    "rededge1", "rededge2", "rededge3", "nir",
    "nir08", "nir09", "swir16", "swir22",
]


def fetch_carbon_mapper_plumes(n: int = 20, min_emission: float = 100.0) -> list[dict]:
    """Fetch high-confidence CH4 plumes from Carbon Mapper API."""
    url = "https://api.carbonmapper.org/api/v1/catalog/plumes/annotated"
    params = {
        "plume_gas": "CH4",
        "status": "published",
        "limit": min(n * 3, 1000),
        "sort": "emissions_desc",
    }
    print(f"Fetching plumes from Carbon Mapper (min_emission={min_emission} kg/hr)...")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    plumes = []
    for item in data.get("items", []):
        emission = item.get("emission_auto")
        if emission is None or emission < min_emission:
            continue

        geom = item.get("geometry_json", {})
        coords = geom.get("coordinates")
        if not coords or len(coords) < 2:
            continue

        lon, lat = float(coords[0]), float(coords[1])
        dt = item.get("scene_timestamp", "")
        name = item.get("plume_id", "unknown")

        plumes.append({
            "name": name,
            "lat": lat,
            "lon": lon,
            "datetime": dt,
            "emission_kg_hr": float(emission),
        })
        if len(plumes) >= n:
            break

    print(f"  Found {len(plumes)} plumes with emission >= {min_emission} kg/hr")
    return plumes


def find_sentinel2_scene(lat: float, lon: float, date_str: str,
                         days_window: int = 15, max_cloud: float = 30.0):
    """Find the best Sentinel-2 L2A scene near a location and date."""
    client = Client.open("https://earth-search.aws.element84.com/v1")

    if "T" in date_str:
        plume_dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    else:
        plume_dt = datetime.strptime(date_str[:10], "%Y-%m-%d")

    start = (plume_dt - timedelta(days=days_window)).strftime("%Y-%m-%d")
    end = (plume_dt + timedelta(days=days_window)).strftime("%Y-%m-%d")

    delta = 0.05
    bbox = [lon - delta, lat - delta, lon + delta, lat + delta]

    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start}/{end}",
        max_items=10,
    )

    items = list(search.items())
    if not items:
        return None

    # Filter by cloud cover and prefer closest in time
    valid = [i for i in items if i.properties.get("eo:cloud_cover", 100) <= max_cloud]
    if not valid:
        valid = items[:1]  # fall back to first result

    # Sort by time proximity to plume detection
    plume_ts = plume_dt.timestamp()
    valid.sort(key=lambda i: abs(i.datetime.timestamp() - plume_ts))

    return valid[0]


def download_s2_patch(item, lat: float, lon: float, patch_pixels: int = 256) -> np.ndarray | None:
    """Download a patch of all 12 bands centered on (lat, lon).

    S2 L2A reflectance values (uint16, ~0-10000) are scaled to uint8 [0-255]
    to match the training data format.

    Returns (H, W, 12) uint8 array.
    """
    # Get CRS from first available band
    first_key = next(k for k in BAND_KEYS if k in item.assets)
    with rasterio.open(item.assets[first_key].href) as src:
        crs = src.crs

    xs, ys = transform("EPSG:4326", crs, [lon], [lat])
    cx, cy = xs[0], ys[0]

    # All bands read at 20m resolution → patch_pixels * 20m
    half = (patch_pixels * 20) / 2
    bounds = (cx - half, cy - half, cx + half, cy + half)

    bands = []
    for key in BAND_KEYS:
        if key not in item.assets:
            print(f"    Warning: band {key} missing, filling zeros")
            bands.append(np.zeros((patch_pixels, patch_pixels), dtype=np.uint8))
            continue

        with rasterio.open(item.assets[key].href) as src:
            window = from_bounds(*bounds, transform=src.transform)
            data = src.read(1, window=window).astype(np.float32)

            # Scale S2 L2A reflectance (0-10000) to uint8 (0-255)
            # Clip to [0, 10000] then scale
            data = np.clip(data, 0, 10000) * (255.0 / 10000.0)
            bands.append(data.astype(np.uint8))

    # Ensure consistent shape (crop to minimum)
    min_h = min(b.shape[0] for b in bands)
    min_w = min(b.shape[1] for b in bands)
    if min_h < 16 or min_w < 16:
        return None

    bands = [b[:min_h, :min_w] for b in bands]
    return np.stack(bands, axis=-1)


def main():
    parser = argparse.ArgumentParser(description="Fetch real-world S2 data at known plume locations")
    parser.add_argument("--output-dir", type=str, default="data/real_world")
    parser.add_argument("--n-plumes", type=int, default=10)
    parser.add_argument("--min-emission", type=float, default=500.0)
    parser.add_argument("--days-window", type=int, default=15)
    parser.add_argument("--max-cloud", type=float, default=20.0)
    parser.add_argument("--patch-pixels", type=int, default=256,
                        help="Patch size in pixels at 20m resolution")
    args = parser.parse_args()

    out = Path(args.output_dir)
    (out / "s2").mkdir(parents=True, exist_ok=True)

    # Step 1: Get plume locations
    plumes = fetch_carbon_mapper_plumes(n=args.n_plumes, min_emission=args.min_emission)
    if not plumes:
        print("No plumes found! Try lowering --min-emission.")
        return

    # Step 2: Download Sentinel-2 data for each plume
    metadata = []
    for i, plume in enumerate(plumes):
        print(f"\n[{i+1}/{len(plumes)}] {plume['name']}: "
              f"({plume['lat']:.4f}, {plume['lon']:.4f}), "
              f"{plume['emission_kg_hr']:.0f} kg/hr, {plume['datetime'][:10]}")

        item = find_sentinel2_scene(
            plume["lat"], plume["lon"], plume["datetime"],
            days_window=args.days_window, max_cloud=args.max_cloud,
        )
        if item is None:
            print("  No suitable S2 scene found, skipping.")
            continue

        cloud = item.properties.get("eo:cloud_cover", "?")
        date_diff = abs((item.datetime - datetime.fromisoformat(
            plume["datetime"].replace("Z", "+00:00"))).days)
        print(f"  S2 scene: {item.id} (cloud={cloud:.1f}%, {date_diff}d from plume)")

        try:
            patch = download_s2_patch(item, plume["lat"], plume["lon"],
                                      patch_pixels=args.patch_pixels)
        except Exception as e:
            print(f"  Error downloading: {e}")
            continue

        if patch is None:
            print("  Patch too small, skipping.")
            continue

        print(f"  Downloaded: {patch.shape}, range=[{patch.min()}, {patch.max()}]")

        np.save(str(out / "s2" / f"{i}.npy"), patch)
        entry = {
            **plume,
            "s2_scene": item.id,
            "s2_date": str(item.datetime),
            "s2_cloud_cover": cloud,
            "date_diff_days": date_diff,
            "patch_shape": list(patch.shape),
        }
        metadata.append(entry)

    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nDone! Saved {len(metadata)} patches to {out}/")


if __name__ == "__main__":
    main()
