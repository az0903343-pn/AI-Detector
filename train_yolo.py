"""Fine-tune YOLOv8 on an office dataset (YOLO format, data.yaml from Roboflow etc.).

    python training/train_yolo.py --data data/office/data.yaml --epochs 50 --imgsz 960

Then point detector.person_model / detector.phone_model at weights/best.pt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for person/phone detection")
    parser.add_argument("--data", required=True, help="path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="", help="0 for the first GPU, empty = auto")
    parser.add_argument("--project", default="runs/yolo")
    parser.add_argument("--name", default="office")
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=args.project,
        name=args.name,
        patience=15,
        cos_lr=True,
        plots=True,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"best weights: {best}")
    if args.export_onnx:
        YOLO(str(best)).export(format="onnx", imgsz=args.imgsz, opset=12)
        print("onnx exported next to best.pt")


if __name__ == "__main__":
    main()
