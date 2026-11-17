from __future__ import annotations

from typing import Iterable

import numpy as np


class DeskZone:
    """A desk / seat region in normalized (0-1) frame coordinates."""

    def __init__(self, name: str, points: Iterable[Iterable[float]]):
        pts = np.asarray(list(points), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise ValueError(f"Zone {name!r} needs at least 3 (x, y) points")
        self.name = name
        self.points = pts
        self.centroid = pts.mean(axis=0)
        self.bounds = (
            float(pts[:, 0].min()),
            float(pts[:, 1].min()),
            float(pts[:, 0].max()),
            float(pts[:, 1].max()),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "DeskZone":
        name = str(data.get("name") or "D")
        return cls(name, data["points"])

    def contains(self, point: np.ndarray) -> bool:
        return point_in_polygon(point, self.points)

    def distance(self, point: np.ndarray) -> float:
        if self.contains(point):
            return 0.0
        pts = self.points
        best = np.inf
        for i in range(len(pts)):
            best = min(best, _segment_distance(point, pts[i], pts[(i + 1) % len(pts)]))
        return float(best)

    def pixel_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.bounds
        return (
            int(round(x0 * width)),
            int(round(y0 * height)),
            int(round(x1 * width)),
            int(round(y1 * height)),
        )


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0.0:
        return float(np.hypot(*(point - a)))
    t = float((point - a) @ ab) / denom
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return float(np.hypot(*(point - closest)))


class DeskAssigner:
    """Sticky person -> desk assignment using foot points (bottom-center)."""

    def __init__(self, zones: list[DeskZone], max_dist: float = 0.14):
        self.zones = zones
        self.max_dist = max_dist
        self._last: dict[int, int] = {}

    def reset(self) -> None:
        self._last.clear()

    def assign(self, feet: dict[int, np.ndarray]) -> dict[int, int]:
        occupied: dict[int, int] = {}
        for desk_idx, track_id in sorted(self._last.items()):
            point = feet.get(track_id)
            if point is None:
                continue
            if self.zones[desk_idx].distance(point) <= self.max_dist:
                occupied[desk_idx] = track_id

        taken = set(occupied.values())
        candidates: list[tuple[float, int, int]] = []
        for track_id, point in feet.items():
            if track_id in taken:
                continue
            for desk_idx, zone in enumerate(self.zones):
                if desk_idx in occupied:
                    continue
                dist = zone.distance(point)
                if dist <= self.max_dist:
                    candidates.append((dist, desk_idx, track_id))

        for _, desk_idx, track_id in sorted(candidates):
            if desk_idx in occupied or track_id in taken:
                continue
            occupied[desk_idx] = track_id
            taken.add(track_id)

        self._last = dict(occupied)
        return {track_id: desk_idx for desk_idx, track_id in occupied.items()}


def grid_zones(
    rows: int = 3,
    cols: int = 2,
    margin_x: float = 0.06,
    margin_y: float = 0.08,
    gap_x: float = 0.08,
    gap_y: float = 0.05,
) -> list[DeskZone]:
    """Quick rectangular layout, handy when you have no calibration frame."""
    cell_w = (1.0 - 2 * margin_x - (cols - 1) * gap_x) / cols
    cell_h = (1.0 - 2 * margin_y - (rows - 1) * gap_y) / rows
    zones: list[DeskZone] = []
    for r in range(rows):
        for c in range(cols):
            x0 = margin_x + c * (cell_w + gap_x)
            y0 = margin_y + r * (cell_h + gap_y)
            pts = [
                [x0, y0],
                [x0 + cell_w, y0],
                [x0 + cell_w, y0 + cell_h],
                [x0, y0 + cell_h],
            ]
            zones.append(DeskZone(f"ID {len(zones) + 1}", pts))
    return zones


def zones_to_json(zones: list[DeskZone]) -> list[dict]:
    return [
        {
            "name": zone.name,
            "points": [[round(float(x), 4), round(float(y), 4)] for x, y in zone.points],
        }
        for zone in zones
    ]
