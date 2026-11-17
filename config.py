from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class DetectorConfig:
    person_model: str = "yolov8n.pt"
    phone_model: str = ""
    device: str = ""
    imgsz: int = 960
    conf: float = 0.35
    iou: float = 0.5
    tracker: str = "bytetrack.yaml"
    person_classes: list[int] = field(default_factory=lambda: [0])
    coco_phone_class: int = 67
    detect_phone_with_person_model: bool = True


@dataclass
class ActivityConfig:
    phone_conf: float = 0.35
    phone_dwell_s: float = 1.0
    phone_hit_ratio: float = 0.5
    phone_window_s: float = 1.5
    min_work_dwell_s: float = 0.6
    classifier_model: str = ""
    classifier_threshold: float = 0.6


@dataclass
class DeskConfig:
    absent_after_s: float = 4.0
    max_assign_dist: float = 0.14
    zones: list[dict] = field(default_factory=list)


@dataclass
class AlertConfig:
    cooldown_s: float = 30.0
    phone_alert: bool = True
    absent_alert: bool = True
    save_snapshots: bool = False
    snapshot_dir: str = "alerts"


@dataclass
class RenderConfig:
    show_panel: bool = True
    show_track_id: bool = False


@dataclass
class Settings:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    activity: ActivityConfig = field(default_factory=ActivityConfig)
    desks: DeskConfig = field(default_factory=DeskConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Settings":
        settings = cls()
        if data:
            _fill(settings, data)
        return settings

    @classmethod
    def from_file(cls, path: str | Path | None) -> "Settings":
        if path is None:
            return cls()
        path = Path(path)
        if not path.is_file():
            return cls()
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fill(obj: Any, data: dict[str, Any]) -> None:
    for key, value in data.items():
        if not hasattr(obj, key):
            raise ValueError(f"Unknown config key: {key!r}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _fill(current, value)
        else:
            setattr(obj, key, value)
