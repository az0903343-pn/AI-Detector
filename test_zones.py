import unittest

import numpy as np

from monitor.zones import (
    DeskAssigner,
    grid_zones,
    point_in_polygon,
    zones_to_json,
)


class ZoneGeometryTest(unittest.TestCase):
    def test_grid_zones_are_normalized(self):
        zones = grid_zones(3, 2)
        self.assertEqual(len(zones), 6)
        for zone in zones:
            x0, y0, x1, y1 = zone.bounds
            self.assertTrue(0.0 <= x0 < x1 <= 1.0)
            self.assertTrue(0.0 <= y0 < y1 <= 1.0)

    def test_point_in_polygon(self):
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        self.assertTrue(point_in_polygon(np.array([0.5, 0.5]), square))
        self.assertFalse(point_in_polygon(np.array([1.5, 0.5]), square))

    def test_zones_to_json(self):
        data = zones_to_json(grid_zones(1, 2))
        self.assertEqual(len(data), 2)
        self.assertIn("points", data[0])
        self.assertEqual(len(data[0]["points"]), 4)


class AssignerTest(unittest.TestCase):
    def setUp(self):
        self.zones = grid_zones(3, 2)
        self.assigner = DeskAssigner(self.zones, max_dist=0.2)

    def test_assigns_person_to_nearest_desk(self):
        result = self.assigner.assign(
            {11: np.array([0.26, 0.20], dtype=np.float32)}
        )
        self.assertEqual(result, {11: 0})

    def test_assignment_is_sticky(self):
        self.assigner.assign({11: np.array([0.26, 0.20], dtype=np.float32)})
        moved = self.assigner.assign({11: np.array([0.30, 0.25], dtype=np.float32)})
        self.assertEqual(moved, {11: 0})

    def test_two_people_two_desks(self):
        result = self.assigner.assign(
            {
                1: np.array([0.26, 0.20], dtype=np.float32),
                2: np.array([0.26, 0.45], dtype=np.float32),
            }
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(len(set(result.values())), 2)

    def test_far_away_person_is_not_assigned(self):
        strict = DeskAssigner(self.zones, max_dist=0.02)
        result = strict.assign({5: np.array([0.5, 0.5], dtype=np.float32)})
        self.assertEqual(result, {})
