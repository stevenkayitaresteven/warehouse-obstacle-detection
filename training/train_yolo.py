"""
Train a YOLO11n warehouse object detector.

Example:
    python training/train_yolo.py \
        --data configs/data.yaml \
        --weights yolo11n.pt \
        --epochs 100 \
        --imgsz 640 \
        --batch 16 \
        --device 0 \
        --output /content/drive/MyDrive/warehouse-obstacle-detection/results/yolo_training
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO11n on the warehouse dataset."
    )

    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="loco_yolo11n")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    weights_path = Path(args.weights)
    output_path = Path(args.output)

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    if args.epochs < 1 or args.imgsz < 1 or args.batch < 1:
        raise ValueError("Epochs, image size and batch must be positive.")

    output_path.mkdir(parents=True, exist_ok=True)

    print("YOLO TRAINING CONFIGURATION")
    print(f"Dataset: {data_path}")
    print(f"Initial weights: {weights_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {args.device}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Learning rate: {args.lr0}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Output: {output_path}")

    model = YOLO(str(weights_path))

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        amp=True,
        cache=False,
        project=str(output_path),
        name=args.name,
        exist_ok=True,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        seed=args.seed,
    )

    print("\nTRAINING COMPLETED")
    print(f"Results directory: {results.save_dir}")
    print(f"Best weights: {Path(results.save_dir) / 'weights' / 'best.pt'}")
    print(f"Last weights: {Path(results.save_dir) / 'weights' / 'last.pt'}")


if __name__ == "__main__":
    main()
