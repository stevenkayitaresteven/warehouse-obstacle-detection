"""
Fine-tune the warehouse detector for the real-image camera simulation and
export it to ONNX (weights/yolo/best.onnx).

Training data ("cut, paste and learn"):
    * the original real training photos (datasets/processed/warehouse3), so
      the detector keeps what it knows about real images, and
    * composite frames of the real-image camera whose objects are cut-outs
      from the TRAIN split only (scripts/make_sim_dataset.py).

No checkpoint is selected on validation data: the weights of the last
epoch are exported. That keeps the val / test cut-outs untouched for the
held-out evaluation (scripts/eval_real_image_sim.py).

    python -m scripts.make_sim_dataset --out datasets/processed/simcam \\
        --split train --frames 2600 --sprites train --styles a,b
    python -m scripts.make_sim_dataset --out datasets/processed/simcam \\
        --split val --frames 300 --sprites val --styles a,b --seed0 800000
    python training/finetune_sim_detector.py --epochs 8

Needs: ultralytics (PyTorch). About 8 minutes per epoch on 4 CPU cores.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=str(ROOT / "datasets" / "processed"
                                          / "warehouse3"))
    ap.add_argument("--sim", default=str(ROOT / "datasets" / "processed"
                                         / "simcam"))
    ap.add_argument("--weights", default=str(ROOT / "weights" / "yolo"
                                             / "best.pt"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "results" / "yolo_simcam"))
    ap.add_argument("--export", default=str(ROOT / "weights" / "yolo"
                                            / "best.onnx"))
    a = ap.parse_args()
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import strip_optimizer

    real, sim, out = Path(a.real).resolve(), Path(a.sim).resolve(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = out / "data.yaml"
    data.write_text(
        "train:\n"
        f"  - {real / 'images' / 'train'}\n"
        f"  - {sim / 'images' / 'train'}\n"
        "val:\n"
        f"  - {real / 'images' / 'val'}\n"
        f"  - {sim / 'images' / 'val'}\n"
        "names:\n  0: person\n  1: forklift\n  2: package\n")

    model = YOLO(a.weights)
    res = model.train(
        data=str(data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        device=a.device, workers=3, optimizer="AdamW", lr0=0.0008, lrf=0.05,
        cos_lr=True, warmup_epochs=1.0, close_mosaic=2, patience=100,
        amp=False, project=str(out), name="finetune", exist_ok=True,
        seed=0, save_period=1)
    last = Path(res.save_dir) / "weights" / "last.pt"
    final = out / "simcam_last.pt"
    shutil.copy(last, final)
    strip_optimizer(str(final))
    onnx = YOLO(str(final)).export(format="onnx", imgsz=(384, 640), opset=12,
                                   simplify=True, dynamic=False)
    shutil.copy(onnx, a.export)
    print("fine-tuned weights:", final)
    print("ONNX for the simulation camera:", a.export)


if __name__ == "__main__":
    main()
