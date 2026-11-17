from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .config import Settings

WORKING = "WORKING"
PHONE = "PHONE_USAGE"
ABSENT = "ABSENT"


@dataclass
class Observation:
    key: int
    name: str
    track_id: int | None
    phone: bool = False
    clf_label: str | None = None
    clf_prob: float = 0.0


@dataclass
class DeskStatus:
    key: int
    name: str
    state: str
    track_id: int | None
    since: float


@dataclass
class Transition:
    key: int
    name: str
    old: str
    new: str
    track_id: int | None
    t: float


def is_phone_label(label: str | None) -> bool:
    return bool(label) and "phone" in str(label).lower()


class _Desk:
    def __init__(self, cfg: Settings, key: int, name: str, t0: float):
        self.key = key
        self.name = name
        self.absent_after = cfg.desks.absent_after_s
        self.phone_dwell = cfg.activity.phone_dwell_s
        self.work_dwell = cfg.activity.min_work_dwell_s
        self.hit_ratio = cfg.activity.phone_hit_ratio
        self.window = cfg.activity.phone_window_s
        self.clf_threshold = cfg.activity.classifier_threshold

        self.state = ABSENT
        self.since = t0
        self.track_id: int | None = None
        self._candidate = ABSENT
        self._candidate_since = t0
        self._frames: deque[float] = deque()
        self._hits: deque[float] = deque()
        self._first_seen = False

    def update(self, t: float, obs: Observation) -> Transition | None:
        if obs.track_id != self.track_id:
            self._frames.clear()
            self._hits.clear()
        self.track_id = obs.track_id

        phone_now = bool(obs.phone) or (
            obs.clf_label is not None
            and is_phone_label(obs.clf_label)
            and obs.clf_prob >= self.clf_threshold
        )
        self._frames.append(t)
        if phone_now:
            self._hits.append(t)
        self._prune(t)
        ratio = len(self._hits) / max(1, len(self._frames))
        phone_evidence = obs.track_id is not None and ratio >= self.hit_ratio

        if obs.track_id is None:
            candidate, need = ABSENT, self.absent_after
        elif phone_evidence:
            candidate, need = PHONE, self.phone_dwell
        else:
            candidate, need = WORKING, self.work_dwell

        # A worker who just sat down should not flash as ABSENT for a second.
        if obs.track_id is not None and not self._first_seen and self.state == ABSENT:
            self._first_seen = True
            return self._commit(candidate, t, t)

        if candidate != self._candidate:
            self._candidate = candidate
            self._candidate_since = t

        if candidate != self.state and (t - self._candidate_since) >= need:
            return self._commit(candidate, t, self._candidate_since)
        return None

    def _commit(self, new_state: str, t: float, since: float) -> Transition | None:
        old = self.state
        self.state = new_state
        self.since = since
        self._candidate = new_state
        self._candidate_since = since
        if old == new_state:
            return None
        return Transition(self.key, self.name, old, new_state, self.track_id, t)

    def _prune(self, t: float) -> None:
        while self._frames and t - self._frames[0] > self.window:
            self._frames.popleft()
        while self._hits and t - self._hits[0] > self.window:
            self._hits.popleft()


class StateEngine:
    def __init__(self, cfg: Settings, absent_enabled: bool = True):
        self.cfg = cfg
        self.absent_enabled = absent_enabled
        self._desks: dict[int, _Desk] = {}

    def reset(self) -> None:
        self._desks.clear()

    def update(self, t: float, observations: list[Observation]) -> list[Transition]:
        transitions: list[Transition] = []
        for obs in observations:
            desk = self._desks.get(obs.key)
            if desk is None:
                desk = _Desk(self.cfg, obs.key, obs.name, t)
                self._desks[obs.key] = desk
            desk.name = obs.name
            if obs.track_id is None and not self.absent_enabled:
                continue
            tr = desk.update(t, obs)
            if tr is not None:
                transitions.append(tr)
        return transitions

    def drop_missing(self, keep_keys: set[int]) -> None:
        for key in list(self._desks):
            if key not in keep_keys:
                del self._desks[key]

    def statuses(self) -> list[DeskStatus]:
        return [
            DeskStatus(d.key, d.name, d.state, d.track_id, d.since)
            for d in sorted(self._desks.values(), key=lambda d: d.key)
        ]
