from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Settings


@dataclass
class Person:
    track_id: int
    box: tuple[float, float, float, float]
    conf: float
    phone: bool = False
    phone_box: tuple[float, float, float, float] | None = None


class Detector:
    """YOLO person detection + ByteTrack IDs + cell-phone association."""

    def __init__(self, cfg: Settings):
        from ultralytics import YOLO

        self.cfg = cfg.detector
        self.activity = cfg.activity
        self.model = YOLO(self.cfg.person_model)
        self.phone_model = YOLO(self.cfg.phone_model) if self.cfg.phone_model else None

        names = self.model.names or {}
        self.person_class_ids = [c for c in self.cfg.person_classes if c in names] or [0]
        self._phone_class: int | None = None
        if self.phone_model is None and self.cfg.detect_phone_with_person_model:
            for idx, name in names.items():
                if str(name).lower() == "cell phone":
                    self._phone_class = int(idx)
                    break

    def reset(self) -> None:
        self.model.predictor = None

    def track(self, frame: np.ndarray) -> list[Person]:
        classes = list(self.person_class_ids)
        if self._phone_class is not None:
            classes.append(self._phone_class)

        results = self.model.track(
            frame,
            persist=True,
            classes=classes,
            conf=min(self.cfg.conf, self.activity.phone_conf),
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device or None,
            tracker=self.cfg.tracker,
            verbose=False,
        )
        persons: list[Person] = []
        phones: list[tuple[tuple[float, float, float, float], float]] = []

        boxes = getattr(results[0], "boxes", None) if results else None
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            classes_np = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
            for i in range(len(xyxy)):
                box = tuple(float(v) for v in xyxy[i])
                cls_id = int(classes_np[i])
                conf = float(confs[i])
                if cls_id in self.person_class_ids:
                    if ids is None or conf < self.cfg.conf:
                        continue
                    persons.append(Person(int(ids[i]), box, conf))
                elif self._phone_class is not None and cls_id == self._phone_class:
                    if conf >= self.activity.phone_conf:
                        phones.append((box, conf))

        if self.phone_model is not None:
            phones.extend(self._detect_phones(frame))

        self._attach_phones(persons, phones)
        return persons

    def _detect_phones(self, frame: np.ndarray):
        names = self.phone_model.names or {}
        wanted = [int(i) for i, name in names.items() if "phone" in str(name).lower()]
        result = self.phone_model.predict(
            frame,
            conf=self.activity.phone_conf,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device or None,
            verbose=False,
        )[0]
        phones = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return phones
        for i in range(len(boxes)):
            if wanted and int(boxes.cls[i].item()) not in wanted:
                continue
            box = tuple(float(v) for v in boxes.xyxy[i].tolist())
            phones.append((box, float(boxes.conf[i].item())))
        return phones

    def _attach_phones(self, persons: list[Person], phones) -> None:
        if not persons or not phones:
            return
        best: dict[int, tuple[float, tuple]] = {}
        for box, conf in phones:
            pcx = (box[0] + box[2]) / 2
            pcy = (box[1] + box[3]) / 2
            phone_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            for person in persons:
                x1, y1, x2, y2 = person.box
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue
                # upper body region where a hand-held phone can appear
                if not (x1 - 0.20 * w <= pcx <= x2 + 0.20 * w):
                    continue
                if not (y1 - 0.10 * h <= pcy <= y1 + 0.72 * h):
                    continue
                if phone_area > 0.5 * w * h:
                    continue
                if person.track_id not in best or conf > best[person.track_id][0]:
                    best[person.track_id] = (conf, box)
        for person in persons:
            hit = best.get(person.track_id)
            if hit is not None:
                person.phone = True
                person.phone_box = hit[1]
