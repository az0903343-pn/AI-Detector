from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class ActivityClassifier:
    """Optional ONNX person-crop classifier (working / phone / idle).

    Train it with training/train_activity_classifier.py, then point
    activity.classifier_model at the exported .onnx file.
    """

    def __init__(self, model_path: str):
        import onnxruntime as ort

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Activity model not found: {path}")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.size = int(shape[2]) if isinstance(shape[2], int) else 224
        self.labels = ["idle", "phone", "working"]

        meta = path.with_suffix(".json")
        if meta.is_file():
            data = json.loads(meta.read_text(encoding="utf-8"))
            self.labels = list(data.get("labels", self.labels))
            self.size = int(data.get("img_size", self.size))

    def predict_crop(self, frame: np.ndarray, box, margin: float = 0.08):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        x1 = int(max(0, x1 - margin * bw))
        y1 = int(max(0, y1 - margin * bh))
        x2 = int(min(w, x2 + margin * bw))
        y2 = int(min(h, y2 + margin * bh))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 8:
            return None, 0.0
        return self.predict(crop)

    def predict(self, crop_bgr: np.ndarray):
        img = cv2.resize(crop_bgr, (self.size, self.size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        blob = np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)
        logits = self.session.run(None, {self.input_name: blob})[0][0]
        probs = _softmax(np.asarray(logits, dtype=np.float32))
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    exp = np.exp(shifted)
    return exp / float(np.sum(exp))
