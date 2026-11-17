"""Train MobileNetV3/ResNet on person crops and export ONNX for the app.

Kaggle (GPU):
    python training/train_activity_classifier.py \
        --data /kaggle/working/activity/crops \
        --epochs 10 --batch 64 \
        --out /kaggle/working/activity_model \
        --export-onnx /kaggle/working/activity_model/activity_classifier.onnx

The .onnx file + matching .json go into models/ and
activity.classifier_model in config.yaml points at the .onnx file.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise SystemExit(f"unknown --arch {arch!r}")
    return model


def build_transforms(size: int):
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(size * 1.15)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_tf, val_tf


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        predictions = model(images).argmax(dim=1)
        correct += int((predictions == targets).sum().item())
        total += len(targets)
    return correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the activity classifier")
    parser.add_argument("--data", required=True, help="ImageFolder root (working/phone/idle)")
    parser.add_argument("--out", default="runs/activity")
    parser.add_argument("--arch", default="mobilenet_v3_small")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    parser.add_argument("--export-onnx", nargs="?", const="", default=None)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_tf, val_tf = build_transforms(args.img_size)
    train_full = datasets.ImageFolder(args.data, transform=train_tf)
    val_full = datasets.ImageFolder(args.data, transform=val_tf)
    if len(train_full.classes) < 2:
        raise SystemExit(f"need at least 2 class folders in {args.data}")
    print("classes:", train_full.classes, "images:", len(train_full))

    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(train_full), generator=generator).tolist()
    n_val = max(1, int(len(order) * args.val_split))
    val_idx, train_idx = order[:n_val], order[n_val:]

    loader_args = dict(num_workers=args.workers)
    train_loader = DataLoader(
        Subset(train_full, train_idx), batch_size=args.batch, shuffle=True,
        pin_memory=True, **loader_args,
    )
    val_loader = DataLoader(
        Subset(val_full, val_idx), batch_size=args.batch, **loader_args
    )

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print("device:", device)

    model = build_model(args.arch, len(train_full.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_acc = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * len(images)
            seen += len(images)
        scheduler.step()
        accuracy = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch}/{args.epochs} loss {running_loss / max(1, seen):.4f} "
            f"val_acc {accuracy:.4f}"
        )
        if accuracy >= best_acc:
            best_acc = accuracy
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"best val_acc {best_acc:.4f}")

    (out / "classes.json").write_text(
        json.dumps(
            {"labels": train_full.classes, "img_size": args.img_size, "arch": args.arch},
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save(model.state_dict(), out / "activity_classifier.pt")

    if args.export_onnx is not None:
        onnx_path = Path(args.export_onnx) if args.export_onnx else out / "activity_classifier.onnx"
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        model.cpu().eval()
        dummy = torch.randn(1, 3, args.img_size, args.img_size)
        export_kwargs = dict(
            input_names=["input"],
            output_names=["logits"],
            opset_version=12,
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        )
        try:
            torch.onnx.export(
                model, dummy, str(onnx_path), dynamo=False, **export_kwargs
            )
        except TypeError:  # torch < 2.5 has no dynamo flag
            torch.onnx.export(model, dummy, str(onnx_path), **export_kwargs)
        shutil.copy(out / "classes.json", onnx_path.with_suffix(".json"))
        print(f"exported {onnx_path} (+ sidecar .json)")


if __name__ == "__main__":
    main()
