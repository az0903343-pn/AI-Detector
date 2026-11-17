from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .alerts import AlertEvent, AlertManager
from .config import Settings
from .detector import Detector, Person
from .render import RenderInfo, Renderer
from .state import ABSENT, WORKING, DeskStatus, Observation, StateEngine
from .zones import DeskAssigner, DeskZone


@dataclass
class FrameResult:
    annotated: np.ndarray
    statuses: list[DeskStatus]
    persons: list[Person]
    events: list[AlertEvent] = field(default_factory=list)
    working: int = 0
    total_alerts: int = 0


class MonitorPipeline:
    """frame in -> detection, tracking, desk states, alerts, annotated frame out."""

    def __init__(self, cfg: Settings, alert_dir: str | None = None):
        self.cfg = cfg
        self.detector = Detector(cfg)
        self.zones = [DeskZone.from_dict(z) for z in cfg.desks.zones]
        self.assigner = (
            DeskAssigner(self.zones, cfg.desks.max_assign_dist) if self.zones else None
        )
        self.engine = StateEngine(cfg, absent_enabled=bool(self.zones))
        self.alerts = AlertManager(cfg, alert_dir or cfg.alerts.snapshot_dir)
        self.renderer = Renderer(cfg)

        self.classifier = None
        if cfg.activity.classifier_model:
            from .activity import ActivityClassifier

            self.classifier = ActivityClassifier(cfg.activity.classifier_model)

    def reset(self) -> None:
        self.detector.reset()
        self.engine.reset()
        self.alerts.reset()
        if self.assigner is not None:
            self.assigner.reset()

    def step(self, frame: np.ndarray, t: float) -> FrameResult:
        h, w = frame.shape[:2]
        persons = self.detector.track(frame)

        clf: dict[int, tuple[str | None, float]] = {}
        if self.classifier is not None:
            for person in persons:
                clf[person.track_id] = self.classifier.predict_crop(frame, person.box)

        observations: list[Observation] = []
        if self.zones:
            feet = {}
            for person in persons:
                x1, _, x2, y2 = person.box
                feet[person.track_id] = np.array(
                    [(x1 + x2) / 2 / w, y2 / h], dtype=np.float32
                )
            track_to_desk = self.assigner.assign(feet)
            desk_to_track = {desk: tr for tr, desk in track_to_desk.items()}
            by_id = {person.track_id: person for person in persons}
            for i in range(len(self.zones)):
                track_id = desk_to_track.get(i)
                person = by_id.get(track_id) if track_id is not None else None
                label, prob = (
                    clf.get(track_id, (None, 0.0)) if track_id is not None else (None, 0.0)
                )
                observations.append(
                    Observation(
                        key=i,
                        name=f"ID {i + 1}",
                        track_id=track_id,
                        phone=bool(person and person.phone),
                        clf_label=label,
                        clf_prob=prob,
                    )
                )
        else:
            for person in persons:
                label, prob = clf.get(person.track_id, (None, 0.0))
                observations.append(
                    Observation(
                        key=person.track_id,
                        name=f"ID {person.track_id}",
                        track_id=person.track_id,
                        phone=person.phone,
                        clf_label=label,
                        clf_prob=prob,
                    )
                )
            self.engine.drop_missing({person.track_id for person in persons})

        transitions = self.engine.update(t, observations)
        statuses = self.engine.statuses()
        events = self.alerts.handle(t, transitions)

        labels: dict[int, tuple[str, str]] = {}
        for status in statuses:
            if status.track_id is not None:
                labels[status.track_id] = (status.name, status.state)

        absent_bounds: list[tuple[str, tuple[int, int, int, int]]] = []
        if self.zones:
            for status in statuses:
                if status.state == ABSENT and status.track_id is None:
                    absent_bounds.append(
                        (status.name, self.zones[status.key].pixel_box(w, h))
                    )

        working = sum(1 for status in statuses if status.state == WORKING)
        info = RenderInfo(
            persons=persons,
            labels=labels,
            statuses=statuses,
            absent_bounds=absent_bounds,
            working=working,
            alert_count=self.alerts.count,
        )
        annotated = self.renderer.draw(frame, info)
        self.alerts.save_snapshots(annotated)

        return FrameResult(
            annotated=annotated,
            statuses=statuses,
            persons=persons,
            events=events,
            working=working,
            total_alerts=self.alerts.count,
        )
