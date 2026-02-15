"""Export a trained CH4Net checkpoint to a standalone ONNX file."""

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch

from models import Unet


def export(model_dir: str, output_path: str, channels: int = 12):
    model = Unet(in_channels=channels, out_channels=1, div_factor=1, prob_output=True)
    ckpt = torch.load(Path(model_dir) / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, channels, 256, 256)

    # Export with external data, then re-save as single file
    tmp_path = output_path + ".tmp"
    torch.onnx.export(
        model,
        dummy,
        tmp_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 1: "height", 2: "width"},
        },
        opset_version=18,
    )

    # Merge into standalone file
    onnx_model = onnx.load(tmp_path, load_external_data=True)
    onnx.save(onnx_model, output_path)

    # Clean up temp files
    Path(tmp_path).unlink(missing_ok=True)
    Path(tmp_path + ".data").unlink(missing_ok=True)

    # Verify
    onnx.checker.check_model(onnx.load(output_path))

    import onnxruntime as ort
    sess = ort.InferenceSession(output_path)
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
    torch_out = model(dummy).detach().numpy()
    diff = np.abs(onnx_out - torch_out).max()

    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"Exported: {output_path} ({size_mb:.1f} MB)")
    print(f"Epoch {ckpt['epoch']}, val_loss={ckpt['loss']:.4f}")
    print(f"Max diff vs PyTorch: {diff:.1e}")


def main():
    parser = argparse.ArgumentParser(description="Export CH4Net to ONNX")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: <model-dir>/ch4net.onnx)")
    parser.add_argument("--channels", type=int, default=12, choices=[2, 5, 12])
    args = parser.parse_args()

    output = args.output or str(Path(args.model_dir) / "ch4net.onnx")
    export(args.model_dir, output, args.channels)


if __name__ == "__main__":
    main()
