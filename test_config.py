import unittest

from monitor.config import Settings


class ConfigTest(unittest.TestCase):
    def test_partial_override_keeps_defaults(self):
        cfg = Settings.from_dict({"detector": {"conf": 0.5}})
        self.assertEqual(cfg.detector.conf, 0.5)
        self.assertEqual(cfg.detector.imgsz, 960)
        self.assertEqual(cfg.activity.phone_conf, 0.35)

    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError):
            Settings.from_dict({"nope": 1})

    def test_zones_round_trip(self):
        data = {"desks": {"zones": [{"name": "ID 1", "points": [[0, 0], [1, 0], [1, 1]]}]}}
        cfg = Settings.from_dict(data)
        self.assertEqual(cfg.desks.zones[0]["name"], "ID 1")
        self.assertEqual(Settings.from_dict(cfg.to_dict()).desks.zones, cfg.desks.zones)


if __name__ == "__main__":
    unittest.main()
