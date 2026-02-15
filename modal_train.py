"""Remote GPU training on Modal.

Setup (one-time):
    modal setup
    modal secret create huggingface-token HF_TOKEN=hf_xxxxx

Run:
    modal run modal_train.py [--epochs 250] [--channels 12] [--batch-size 16]

Download results:
    modal volume get ch4net-vol /output/best.pt ./output/
    modal volume get ch4net-vol /output/losses.npy ./output/
"""

import modal

app = modal.App("ch4net")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "numpy", "scikit-learn", "tqdm", "matplotlib")
    .apt_install("git", "git-lfs")
    .add_local_dir("src", remote_path="/root/src")
)

vol = modal.Volume.from_name("ch4net-vol", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/vol": vol},
    timeout=6 * 3600,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def run_training(
    epochs: int = 250,
    channels: int = 12,
    batch_size: int = 16,
    lr: float = 1e-4,
    crop_size: int = 100,
):
    import os
    import subprocess
    import sys
    from pathlib import Path

    data_path = Path("/vol/data")
    output_path = Path("/vol/output_v2")
    output_path.mkdir(parents=True, exist_ok=True)

    # Clone HF dataset on first run (cached in volume for subsequent runs)
    if not (data_path / "train").exists():
        print("Downloading dataset from HuggingFace...")
        subprocess.run(["git", "lfs", "install"], check=True)
        token = os.environ["HF_TOKEN"]
        subprocess.run(
            [
                "git", "clone",
                f"https://user:{token}@huggingface.co/datasets/av555/ch4net",
                str(data_path),
            ],
            check=True,
        )
        vol.commit()
        print("Dataset cached in volume.")

    # Run training
    subprocess.run(
        [
            sys.executable, "/root/src/train.py",
            "--data-dir", str(data_path),
            "--output-dir", str(output_path),
            "--epochs", str(epochs),
            "--channels", str(channels),
            "--batch-size", str(batch_size),
            "--lr", str(lr),
            "--crop-size", str(crop_size),
        ],
        check=True,
    )

    vol.commit()
    print("Done! Results saved to volume 'ch4net-vol' at /output/")


@app.local_entrypoint()
def main(
    epochs: int = 250,
    channels: int = 12,
    batch_size: int = 16,
    lr: float = 1e-4,
    crop_size: int = 100,
):
    run_training.remote(
        epochs=epochs,
        channels=channels,
        batch_size=batch_size,
        lr=lr,
        crop_size=crop_size,
    )
