"""Render an annotated monitoring video from a file, webcam or RTSP stream."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitor.config import Settings
from monitor.pipeline import MonitorPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI employee monitoring on a video source")
    parser.add_argument("--source", required=True, help="video path, RTSP URL, or webcam index (0)")
    parser.add_argument("--out", default="runs/output.mp4", help="annotated output video")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N processed frames")
    parser.add_argument("--show", action="store_true", help="preview window, press q to quit")
    parser.add_argument("--alert-dir", default="alerts_out")
    return parser.parse_args()


def open_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def main() -> int:
    args = parse_args()
    cfg = Settings.from_file(args.config) if Path(args.config).is_file() else Settings()
    pipeline = MonitorPipeline(cfg, alert_dir=args.alert_dir)

    cap = open_source(args.source)
    if not cap.isOpened():
        print(f"error: cannot open source {args.source!r}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 240:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / max(1, args.stride),
        (width, height),
    )

    stride = max(1, args.stride)
    index = 0
    processed = 0
    started = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                result = pipeline.step(frame, index / fps)
                writer.write(result.annotated)
                processed += 1
                if args.show:
                    cv2.imshow("AI Employee Monitoring", result.annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if processed % 50 == 0:
                    print(f"processed {processed} frames ({time.time() - started:.1f}s)")
                if args.max_frames and processed >= args.max_frames:
                    break
            index += 1
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    csv_path = out_path.with_suffix(".alerts.csv")
    rows = pipeline.alerts.rows()
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["t", "clock", "desk", "track_id", "kind", "message", "snapshot"]
        csv_writer = csv.DictWriter(handle, fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    print(f"done: {processed} frames -> {out_path}")
    print(f"alerts: {len(rows)} -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
