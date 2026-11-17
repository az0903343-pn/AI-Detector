from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .config import Settings
from .state import ABSENT, PHONE, Transition


@dataclass
class AlertEvent:
    t: float
    clock: str
    desk: str
    track_id: int | None
    kind: str
    message: str
    snapshot: str = ""


class AlertManager:
    def __init__(self, cfg: Settings, output_dir: str | Path = "alerts_out"):
        self.cfg = cfg.alerts
        self.output_dir = Path(output_dir)
        self.events: list[AlertEvent] = []
        self._last: dict[tuple[int, str], float] = {}
        self._pending: list[AlertEvent] = []

    @property
    def count(self) -> int:
        return len(self.events)

    def reset(self) -> None:
        self.events.clear()
        self._last.clear()
        self._pending.clear()

    def handle(self, t: float, transitions: list[Transition]) -> list[AlertEvent]:
        fresh: list[AlertEvent] = []
        for tr in transitions:
            kind = message = None
            if tr.new == PHONE and self.cfg.phone_alert:
                kind, message = "phone", f"{tr.name} phone usage"
            elif tr.new == ABSENT and self.cfg.absent_alert:
                kind, message = "absent", f"{tr.name} desk unattended"
            if kind is None:
                continue
            key = (tr.key, kind)
            if t - self._last.get(key, -1e9) < self.cfg.cooldown_s:
                continue
            self._last[key] = t
            event = AlertEvent(
                t=round(t, 2),
                clock=datetime.now().strftime("%H:%M:%S"),
                desk=tr.name,
                track_id=tr.track_id,
                kind=kind,
                message=message,
            )
            self.events.append(event)
            fresh.append(event)
        self._pending = fresh
        return fresh

    def save_snapshots(self, frame) -> list[str]:
        saved: list[str] = []
        if not self.cfg.save_snapshots or not self._pending:
            return saved
        import cv2

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for event in self._pending:
            slug = re.sub(r"[^A-Za-z0-9]+", "_", event.desk).strip("_").lower() or "desk"
            path = self.output_dir / f"{event.kind}_{slug}_{int(event.t * 1000)}.jpg"
            cv2.imwrite(str(path), frame)
            event.snapshot = str(path)
            saved.append(str(path))
        self._pending = []
        return saved

    def rows(self) -> list[dict]:
        return [asdict(event) for event in self.events]
