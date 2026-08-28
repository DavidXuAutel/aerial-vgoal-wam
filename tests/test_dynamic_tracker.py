"""Unit tests for DynamicTargetTracker."""

import math
import unittest
import numpy as np

from vgoal.dynamic_tracker import DynamicTargetTracker, DynamicTrackerConfig, TrackingMode


class TestDynamicTargetTracker(unittest.TestCase):

    def test_static_target_initialization_and_standoff(self):
        cfg = DynamicTrackerConfig(standoff_dist_m=5.0, standoff_height_m=2.0)
        tracker = DynamicTargetTracker(cfg)

        # Drone at [0, 0, 10], yaw = 0. Target detected at 20m forward
        meas_body = [20.0, 0.0, 0.0, 20.0]
        mode, goal_rel = tracker.step(meas_body, drone_pos=[0.0, 0.0, 10.0], drone_yaw=0.0, dt=0.1)

        self.assertEqual(mode, TrackingMode.INTERCEPTING)
        self.assertIsNotNone(goal_rel)
        # Target is at world [20, 0, 10]. Since v ~ 0, standoff offset from drone is d_back along drone-target line
        # Standoff point is target - 5m towards drone -> [15, 0, 12].
        # Relative to drone [0, 0, 10], goal_rel = [15, 0, 2, sqrt(15^2+4)]
        self.assertAlmostEqual(goal_rel[0], 15.0, places=1)
        self.assertAlmostEqual(goal_rel[1], 0.0, places=1)
        self.assertAlmostEqual(goal_rel[2], 2.0, places=1)

    def test_moving_target_velocity_estimation(self):
        tracker = DynamicTargetTracker()

        # Simulate target moving east along X at 5 m/s: x(t) = 10 + 5*t
        drone_pos = [0.0, 0.0, 10.0]
        drone_yaw = 0.0
        dt = 0.1

        for step in range(30):
            t = step * dt
            tgt_x = 10.0 + 5.0 * t
            # Target measured in body frame
            meas_body = [tgt_x, 0.0, 0.0, tgt_x]
            mode, _ = tracker.step(meas_body, drone_pos, drone_yaw, dt)

        # Check estimated velocity
        v_est = tracker.target_velocity_world
        self.assertIsNotNone(v_est)
        # Should be approximately [5.0, 0.0, 0.0]
        self.assertAlmostEqual(v_est[0], 5.0, delta=0.5)
        self.assertAlmostEqual(v_est[1], 0.0, delta=0.5)

    def test_occlusion_and_timeout(self):
        cfg = DynamicTrackerConfig(max_occlusion_s=1.0)
        tracker = DynamicTargetTracker(cfg)

        # Step 1: see target
        mode, _ = tracker.step([10.0, 0.0, 0.0, 10.0], [0.0, 0.0, 5.0], 0.0, 0.1)
        self.assertIn(mode, (TrackingMode.INTERCEPTING, TrackingMode.FOLLOWING))

        # Step 2: visual loss for 0.5s (within 1.0s window) -> OCCLUDED
        for _ in range(5):
            mode, goal_rel = tracker.step(None, [0.0, 0.0, 5.0], 0.0, 0.1)
        self.assertEqual(mode, TrackingMode.OCCLUDED)
        self.assertIsNotNone(goal_rel)

        # Step 3: visual loss exceeds 1.0s -> LOST
        for _ in range(10):
            mode, goal_rel = tracker.step(None, [0.0, 0.0, 5.0], 0.0, 0.1)
        self.assertEqual(mode, TrackingMode.LOST)
        self.assertIsNone(goal_rel)


if __name__ == "__main__":
    unittest.main()
