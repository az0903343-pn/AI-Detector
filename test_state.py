import unittest

from monitor.alerts import AlertManager
from monitor.config import Settings
from monitor.state import (
    ABSENT,
    PHONE,
    WORKING,
    Observation,
    StateEngine,
    Transition,
)


class StateEngineTest(unittest.TestCase):
    def setUp(self):
        cfg = Settings()
        cfg.activity.phone_dwell_s = 1.0
        cfg.activity.phone_hit_ratio = 0.5
        cfg.activity.phone_window_s = 1.0
        cfg.activity.min_work_dwell_s = 0.5
        cfg.desks.absent_after_s = 2.0
        self.engine = StateEngine(cfg, absent_enabled=True)

    def obs(self, track=7, phone=False, clf=None, prob=0.0):
        return Observation(
            key=0, name="ID 1", track_id=track, phone=phone, clf_label=clf, clf_prob=prob
        )

    def run(self, times, **kwargs):
        transitions = []
        for t in times:
            transitions.extend(self.engine.update(t, [self.obs(**kwargs)]))
        return transitions

    def test_sitting_down_commits_working_immediately(self):
        transitions = self.run([0.0])
        self.assertEqual(transitions[0].new, WORKING)
        self.assertEqual(self.engine.statuses()[0].state, WORKING)

    def test_phone_usage_needs_dwell(self):
        self.run([0.0])
        transitions = self.run([0.5, 1.0, 1.5, 2.0, 2.5], phone=True)
        self.assertTrue(any(tr.new == PHONE for tr in transitions))
        self.assertEqual(self.engine.statuses()[0].state, PHONE)

    def test_classifier_phone_evidence(self):
        self.run([0.0])
        transitions = self.run(
            [0.5, 1.0, 1.5, 2.0], clf="phone", prob=0.9
        )
        self.assertTrue(any(tr.new == PHONE for tr in transitions))

    def test_single_phone_frame_is_ignored(self):
        self.run([0.0])
        transitions = self.run([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], phone=False)
        transitions += self.engine.update(3.2, [self.obs(phone=True)])
        self.assertFalse(any(tr.new == PHONE for tr in transitions))
        self.assertEqual(self.engine.statuses()[0].state, WORKING)

    def test_desk_becomes_absent(self):
        self.run([0.0, 0.5, 1.0])
        transitions = self.run([1.5, 2.0, 3.0, 3.5, 4.0], track=None)
        self.assertTrue(any(tr.new == ABSENT for tr in transitions))
        self.assertEqual(self.engine.statuses()[0].state, ABSENT)

    def test_absent_disabled_skips_empty_desks(self):
        engine = StateEngine(Settings(), absent_enabled=False)
        transitions = []
        for t in (0.0, 5.0, 10.0):
            transitions += engine.update(t, [self.obs(track=None)])
        self.assertEqual(transitions, [])
        self.assertEqual(engine.statuses(), [])

    def test_name_is_reused(self):
        self.run([0.0])
        status = self.engine.statuses()[0]
        self.assertEqual((status.name, status.track_id), ("ID 1", 7))


class AlertManagerTest(unittest.TestCase):
    def setUp(self):
        cfg = Settings()
        cfg.alerts.cooldown_s = 10.0
        self.manager = AlertManager(cfg)
        self.transition = Transition(
            key=0, name="ID 1", old=WORKING, new=PHONE, track_id=3, t=5.0
        )

    def test_alert_and_cooldown(self):
        self.assertEqual(len(self.manager.handle(5.0, [self.transition])), 1)
        self.assertEqual(self.manager.handle(8.0, [self.transition]), [])
        self.assertEqual(len(self.manager.handle(16.0, [self.transition])), 1)
        self.assertEqual(self.manager.count, 2)
        self.assertEqual(self.manager.rows()[0]["kind"], "phone")

    def test_working_transitions_do_not_alert(self):
        transition = Transition(
            key=0, name="ID 1", old=PHONE, new=WORKING, track_id=3, t=1.0
        )
        self.assertEqual(self.manager.handle(1.0, [transition]), [])


if __name__ == "__main__":
    unittest.main()
