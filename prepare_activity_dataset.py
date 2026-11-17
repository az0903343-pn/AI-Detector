"""Sample person crops from office videos and auto-label them.

Usage (extract + auto label):
    python training/prepare_activity_dataset.py --videos data/videos --out data/activity

Outputs:
    <out>/crops/working/*.jpg, <out>/crops/phone/*.jpg
    <out>/review.csv    every crop + auto label

Fix labels in review.csv (working / phone / idle), then rebuild the folders:
    python training/prepare_activity_dataset.py --from-csv data/activity/review.csv \
        --out data/activity
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitor.config import Settings
from monitor.detector import Detector

LABELS = ("working", "phone", "idle")
FIELDS = ["relpath", "label", "video", "frame", "track_id", "conf"]


def video_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in extensions)


def margin_box(box, margin: float, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return (
        int(max(0, x1 - margin * bw)),
        int(max(0, y1 - margin * bh)),
        int(min(w, x2 + margin * bw)),
        int(min(h, y2 + margin * bh)),
    )


def extract(args: argparse.Namespace) -> None:
    out = Path(args.out)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    cfg = Settings.from_dict(
        {
            "detector": {
                "person_model": args.model,
                "conf": args.conf,
                "imgsz": args.imgsz,
                "device": args.device,
            }
        }
    )
    detector = Detector(cfg)

    rows: list[dict] = []
    for video in video_paths(Path(args.videos)):
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"skip {video}: cannot open")
            continue
        saved = 0
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % args.stride == 0:
                for person in detector.track(frame):
                    x1, y1, x2, y2 = margin_box(person.box, args.margin, frame.shape)
                    if x2 - x1 < args.min_size or y2 - y1 < args.min_size:
                        continue
                    label = "phone" if person.phone else "working"
                    name = f"{video.stem}_{index:06d}_{person.track_id}.jpg"
                    rel = Path("crops") / label / name
                    dest = out / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dest), frame[y1:y2, x1:x2])
                    rows.append(
                        {
                            "relpath": rel.as_posix(),
                            "label": label,
                            "video": video.name,
                            "frame": index,
                            "track_id": person.track_id,
                            "conf": round(person.conf, 3),
                        }
                    )
                    saved += 1
            index += 1
        cap.release()
        print(f"{video.name}: {saved} crops ({out / 'crops'})")

    csv_path = out / "review.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} crops")
    print("add some 'idle' crops manually, then run:")
    print(f"  python training/prepare_activity_dataset.py --from-csv {csv_path} --out {out}")


def rebuild(csv_file: Path, out: Path) -> None:
    with csv_file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    moved = 0
    for row in rows:
        label = (row.get("label") or "").strip().lower()
        if label not in LABELS:
            print(f"skip invalid label {label!r}: {row.get('relpath')}")
            continue
        src = out / row["relpath"]
        if not src.is_file():
            continue
        dest = out / "crops" / label / src.name
        if dest.resolve() != src.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            row["relpath"] = dest.relative_to(out).as_posix()
            moved += 1

    with csv_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"moved {moved} crops into label folders")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an activity-classifier dataset")
    parser.add_argument("--videos", help="video file or folder of videos")
    parser.add_argument("--out", default="data/activity", help="dataset output folder")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="")
    parser.add_argument("--stride", type=int, default=15, help="sample every Nth frame")
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--min-size", type=int, default=64)
    parser.add_argument("--from-csv", type=Path, help="rebuild label folders from reviewed CSV")
    args = parser.parse_args()

    if args.from_csv:
        rebuild(args.from_csv, Path(args.out))
        return
    if not args.videos:
        parser.error("--videos is required unless --from-csv is used")
    extract(args)


if __name__ == "__main__":
    main()
