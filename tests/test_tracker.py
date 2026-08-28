"""Unit tests for TargetTracker state machine and occlusion extrapolation."""

import unittest
import numpy as np

from vgoal.tracker import TargetState, TargetTracker, TrackerConfig


class TestTracker(unittest.TestCase):

    def test_tracker_initial_state(self):
        tracker = TargetTracker()
        self.assertEqual(tracker.state, TargetState.SEARCHING)
        self.assertIsNone(tracker.goal_rel)

    def test_tracker_acquires_target(self):
        tracker = TargetTracker(TrackerConfig(success_dist_m=3.0))

        meas = [10.0, 2.0, 0.0, float(np.sqrt(104.0))]
        state = tracker.update(meas, dt=0.2, confidence=0.9)

        self.assertEqual(state, TargetState.TRACKING)
        self.assertIsNotNone(tracker.goal_rel)
        self.assertTrue(np.isclose(tracker.goal_rel[0], 10.0))
        self.assertTrue(np.isclose(tracker.goal_rel[1], 2.0))
        self.assertTrue(np.isclose(tracker.goal_rel[2], 0.0))

    def test_tracker_dead_reckoning_during_occlusion(self):
        tracker = TargetTracker(TrackerConfig(max_occlusion_s=2.0, success_dist_m=3.0))

        # Step 1: Target seen at 10m forward, 0m left
        meas = [10.0, 0.0, 0.0, 10.0]
        tracker.update(meas, dt=0.2)
        self.assertEqual(tracker.state, TargetState.TRACKING)

        # Step 2: Drone flies FORWARD by 2.0m, target is OCCLUDED (meas = None)
        state = tracker.update(None, dt=0.2, ego_delta_body=[2.0, 0.0, 0.0])
        self.assertEqual(state, TargetState.OCCLUDED)
        self.assertIsNotNone(tracker.goal_rel)
        self.assertTrue(np.isclose(tracker.goal_rel[0], 8.0, atol=1e-3))
        self.assertTrue(np.isclose(tracker.goal_rel[1], 0.0, atol=1e-3))

        # Step 3: Drone yaws LEFT by 90 degrees (pi/2 rad), moves 0 translation
        state = tracker.update(None, dt=0.2, ego_delta_body=[0.0, 0.0, 0.0], ego_delta_yaw=np.pi / 2.0)
        self.assertEqual(state, TargetState.OCCLUDED)
        self.assertTrue(np.isclose(tracker.goal_rel[0], 0.0, atol=1e-3))
        self.assertTrue(np.isclose(tracker.goal_rel[1], -8.0, atol=1e-3))

    def test_tracker_occlusion_timeout(self):
        tracker = TargetTracker(TrackerConfig(max_occlusion_s=1.0))

        tracker.update([10.0, 0.0, 0.0, 10.0], dt=0.2)
        self.assertEqual(tracker.state, TargetState.TRACKING)

        # Occluded for 0.6s (<= 1.0s) -> OCCLUDED
        tracker.update(None, dt=0.6)
        self.assertEqual(tracker.state, TargetState.OCCLUDED)

        # Another 0.6s passes (total 1.2s > 1.0s) -> SEARCHING
        state = tracker.update(None, dt=0.6)
        self.assertEqual(state, TargetState.SEARCHING)
        self.assertIsNone(tracker.goal_rel)

    def test_tracker_triggers_arrived(self):
        tracker = TargetTracker(TrackerConfig(success_dist_m=3.0))

        meas = [2.0, 1.0, 0.0, float(np.sqrt(5.0))]
        state = tracker.update(meas, dt=0.2)
        self.assertEqual(state, TargetState.ARRIVED)
        self.assertIsNotNone(tracker.goal_rel)


if __name__ == "__main__":
    unittest.main()
