from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Settings
from .detector import Person
from .state import ABSENT, PHONE, WORKING, DeskStatus

COLORS = {
    WORKING: (70, 200, 80),
    PHONE: (0, 150, 255),
    ABSENT: (45, 45, 235),
}
NEUTRAL = (185, 185, 185)
TEXT = (238, 238, 238)
MUTED = (198, 202, 208)


@dataclass
class RenderInfo:
    persons: list[Person]
    labels: dict[int, tuple[str, str | None]]
    statuses: list[DeskStatus]
    absent_bounds: list[tuple[str, tuple[int, int, int, int]]] = field(default_factory=list)
    working: int = 0
    alert_count: int = 0


class Renderer:
    """Draws the overlay used in the screenshots: boxes, chips, status panel."""

    def __init__(self, cfg: Settings):
        self.cfg = cfg.render

    def draw(self, frame: np.ndarray, info: RenderInfo) -> np.ndarray:
        img = frame.copy()
        h = img.shape[0]
        k = max(0.55, h / 900.0)

        for name, box in info.absent_bounds:
            self._box(img, box, COLORS[ABSENT], k, fill=0.06)
            self._chip(img, f"{name} - {ABSENT}", box, COLORS[ABSENT], k)

        for person in info.persons:
            name, state = info.labels.get(person.track_id, (f"ID {person.track_id}", None))
            color = COLORS.get(state or "", NEUTRAL)
            self._box(img, person.box, color, k)
            label = f"{name} - {state}" if state else f"{name} - TRACKING"
            self._chip(img, label, person.box, color, k)
            if person.phone_box is not None:
                self._box(img, person.phone_box, COLORS[PHONE], k, fill=0.0, corner=0.4)
                self._tag(img, "PHONE", person.phone_box, COLORS[PHONE], k)

        if self.cfg.show_panel:
            self._panel(img, info.statuses, info.working, info.alert_count, k)
        return img

    def _box(
        self,
        img: np.ndarray,
        box,
        color,
        k: float,
        fill: float = 0.09,
        corner: float = 0.18,
    ) -> None:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1 = max(0, min(x1, w - 2))
        y1 = max(0, min(y1, h - 2))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return
        if fill > 0:
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, fill, img, 1 - fill, 0, img)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, int(round(1.5 * k))), cv2.LINE_AA)
        length = max(6, int(round(min(x2 - x1, y2 - y1) * corner)))
        thickness = max(2, int(round(3.2 * k)))
        for cx, cy, sx, sy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(img, (cx, cy), (cx + sx * length, cy), color, thickness, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + sy * length), color, thickness, cv2.LINE_AA)

    def _chip(self, img: np.ndarray, text: str, box, color, k: float) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.52 * k
        th = max(1, int(round(1.5 * k)))
        (tw, tht), base = cv2.getTextSize(text, font, fs, th)
        pad_x = int(round(8 * k))
        pad_y = int(round(5 * k))
        h, w = img.shape[:2]
        x1, y1, _, y2 = (int(round(v)) for v in box)
        left = max(0, min(x1, w - (tw + 2 * pad_x) - 1))
        chip_h = tht + base + 2 * pad_y
        top = y1 - chip_h - 2
        if top < 0:
            top = min(h - chip_h - 1, y2 - chip_h - 2)
        top = max(0, top)
        right = min(w - 1, left + tw + 2 * pad_x)
        bottom = min(h - 1, top + chip_h)
        cv2.rectangle(img, (left, top), (right, bottom), color, -1, cv2.LINE_AA)
        tx = left + pad_x
        ty = top + pad_y + tht
        cv2.putText(img, text, (tx + 1, ty + 1), font, fs, (0, 0, 0), th + 1, cv2.LINE_AA)
        cv2.putText(img, text, (tx, ty), font, fs, (255, 255, 255), th, cv2.LINE_AA)

    def _tag(self, img: np.ndarray, text: str, box, color, k: float) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.38 * k
        th = max(1, int(round(1.2 * k)))
        (tw, tht), base = cv2.getTextSize(text, font, fs, th)
        x1 = max(0, min(int(round(box[0])), img.shape[1] - tw - 10))
        y1 = max(0, min(int(round(box[1])), img.shape[0] - tht - 10))
        cv2.rectangle(img, (x1, y1), (x1 + tw + 8, y1 + tht + base + 4), color, -1, cv2.LINE_AA)
        cv2.putText(
            img, text, (x1 + 4, y1 + tht + 2), font, fs, (15, 15, 15), th, cv2.LINE_AA
        )

    def _panel(
        self,
        img: np.ndarray,
        statuses: list[DeskStatus],
        working: int,
        alerts: int,
        k: float,
    ) -> None:
        if not statuses:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.46 * k
        th = max(1, int(round(1.3 * k)))
        title_fs = 0.56 * k
        title_th = max(1, int(round(1.7 * k)))
        pad = int(round(14 * k))
        line_h = int(round(24 * k))
        shown = statuses[:14]

        (tw, _), _ = cv2.getTextSize("AI EMPLOYEE MONITORING", font, title_fs, title_th)
        widths = [tw]
        for status in shown:
            (w, _), _ = cv2.getTextSize(f"{status.name} - {status.state}", font, fs, th)
            widths.append(w + int(34 * k))
        (fw, _), _ = cv2.getTextSize(f"Working: {working}    Alerts: {alerts}", font, fs, th)
        widths.append(fw)

        pw = min(max(widths) + 2 * pad, img.shape[1] - 2 * pad)
        ph = pad + int(round(40 * k)) + line_h * len(shown) + int(round(32 * k)) + pad
        ph = min(ph, img.shape[0] - 2 * pad)

        overlay = img.copy()
        cv2.rectangle(overlay, (pad, pad), (pad + pw, pad + ph), (12, 14, 16), -1)
        cv2.addWeighted(overlay, 0.62, img, 0.38, 0, img)
        cv2.rectangle(
            img, (pad, pad), (pad + pw, pad + ph), (72, 78, 84), max(1, int(round(k))), cv2.LINE_AA
        )
        cv2.putText(
            img,
            "AI EMPLOYEE MONITORING",
            (pad + int(10 * k), pad + int(25 * k)),
            font,
            title_fs,
            TEXT,
            title_th,
            cv2.LINE_AA,
        )

        y = pad + int(52 * k)
        for status in shown:
            cv2.circle(
                img,
                (pad + int(16 * k), y - int(4 * k)),
                max(3, int(round(4 * k))),
                COLORS.get(status.state, NEUTRAL),
                -1,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                f"{status.name} - {status.state}",
                (pad + int(30 * k), y),
                font,
                fs,
                TEXT,
                th,
                cv2.LINE_AA,
            )
            y += line_h

        y = min(y + int(6 * k), pad + ph - int(8 * k))
        cv2.putText(
            img,
            f"Working: {working}    Alerts: {alerts}",
            (pad + int(10 * k), y),
            font,
            fs,
            MUTED,
            th,
            cv2.LINE_AA,
        )
