"""Click desk polygons on a video frame and print config-ready zones.

Controls:
  left click  add point
  right click remove last point
  enter       close current desk polygon
  u           undo last point
  q           save and quit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

GREEN = (80, 210, 90)
YELLOW = (0, 200, 255)


class ZonePicker:
    def __init__(self, frame, target: int):
        self.frame = frame
        self.target = target
        self.points: list[tuple[int, int]] = []
        self.zones: list[list[tuple[int, int]]] = []

    def click(self, event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(self.zones) < self.target:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()

    def commit(self) -> bool:
        if len(self.points) >= 3:
            self.zones.append(list(self.points))
            self.points = []
            return True
        return False

    def canvas(self):
        img = self.frame.copy()
        overlay = img.copy()
        for i, zone in enumerate(self.zones):
            pts = np.array(zone, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], GREEN)
            cv2.polylines(img, [pts], True, GREEN, 2)
            cx = sum(p[0] for p in zone) // len(zone)
            cy = sum(p[1] for p in zone) // len(zone)
            cv2.putText(
                img, f"ID {i + 1}", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

        if len(self.points) > 1:
            cv2.polylines(img, [np.array(self.points, dtype=np.int32)], False, YELLOW, 2)
        for point in self.points:
            cv2.circle(img, point, 4, YELLOW, -1)

        help_text = (
            f"desk {len(self.zones) + 1}/{self.target} | click points, ENTER = close, "
            "right click / u = undo, q = save"
        )
        cv2.putText(img, help_text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, help_text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw desk zones on a reference frame")
    parser.add_argument("--source", required=True, help="video file or image")
    parser.add_argument("--out", default="zones.json")
    parser.add_argument("--desks", type=int, default=6)
    parser.add_argument("--time", type=float, default=0.0, help="video seek position in seconds")
    args = parser.parse_args()

    source = args.source
    if Path(source).suffix.lower() in (".jpg", ".jpeg", ".png"):
        frame = cv2.imread(source)
    else:
        cap = cv2.VideoCapture(source)
        if args.time > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000.0)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            frame = None
    if frame is None:
        print(f"error: cannot read a frame from {source!r}")
        return 1

    picker = ZonePicker(frame, args.desks)
    window = "desk zones"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, picker.click)

    while True:
        cv2.imshow(window, picker.canvas())
        key = cv2.waitKey(20) & 0xFF
        if key == 13:
            if not picker.commit():
                print("need at least 3 points for a polygon")
        elif key == ord("u") and picker.points:
            picker.points.pop()
        elif key == ord("q"):
            break
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

    if not picker.zones:
        print("no zones clicked, nothing saved")
        return 1

    height, width = frame.shape[:2]
    zones = [
        {
            "name": f"ID {i + 1}",
            "points": [[round(x / width, 4), round(y / height, 4)] for x, y in zone],
        }
        for i, zone in enumerate(picker.zones)
    ]
    Path(args.out).write_text(json.dumps({"zones": zones}, indent=2), encoding="utf-8")

    print(f"saved {len(zones)} zones -> {args.out}")
    print("\ncopy into config.yaml:\n\ndesks:\n  zones:")
    for zone in zones:
        print(f"    - name: {zone['name']}")
        print(f"      points: {zone['points']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
