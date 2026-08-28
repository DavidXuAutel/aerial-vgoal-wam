"""Unit tests for AreaSearchPlanner."""

import math
import unittest
import numpy as np

from vgoal.search_planner import AreaSearchPlanner, SearchAreaConfig


class TestAreaSearchPlanner(unittest.TestCase):

    def test_lawnmower_generation(self):
        cfg = SearchAreaConfig(
            min_x=0.0,
            max_x=30.0,
            min_y=0.0,
            max_y=20.0,
            altitude_z=15.0,
            sweep_spacing_m=10.0,
        )
        planner = AreaSearchPlanner(cfg)
        wps = planner.waypoints
        # xs: 0.0, 10.0, 20.0, 30.0 -> 4 lanes * 2 wps = 8 waypoints
        self.assertEqual(len(wps), 8)
        self.assertTrue(np.allclose(wps[0], [0.0, 0.0, 15.0]))
        self.assertTrue(np.allclose(wps[1], [0.0, 20.0, 15.0]))
        self.assertTrue(np.allclose(wps[2], [10.0, 20.0, 15.0]))
        self.assertTrue(np.allclose(wps[3], [10.0, 0.0, 15.0]))

    def test_waypoint_progression_and_goal_rel(self):
        cfg = SearchAreaConfig(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            altitude_z=10.0,
            waypoint_reach_radius_m=3.0,
            loop=False,
        )
        planner = AreaSearchPlanner(cfg)
        # First target is [0, 0, 10]
        # Drone at [0, -10, 10], facing North (yaw=pi/2)
        # World delta is [0, 10, 0]
        # In drone body frame (facing North), forward is Y_world, left is -X_world
        # So d_fwd = 10.0, d_left = 0.0
        drone_pos = [0.0, -10.0, 10.0]
        drone_yaw = math.pi / 2.0
        gr = planner.update(drone_pos, drone_yaw)
        self.assertTrue(np.isclose(gr[0], 10.0, atol=1e-3))
        self.assertTrue(np.isclose(gr[1], 0.0, atol=1e-3))
        self.assertTrue(np.isclose(gr[2], 0.0, atol=1e-3))
        self.assertTrue(np.isclose(gr[3], 10.0, atol=1e-3))
        self.assertEqual(planner.current_wp_idx, 0)

        # Move drone close to [0, 0, 10] (within 2m)
        gr2 = planner.update([0.0, -1.5, 10.0], math.pi / 2.0)
        # Should advance to next waypoint [0, 10, 10]
        self.assertEqual(planner.current_wp_idx, 1)

    def test_spiral_generation(self):
        planner = AreaSearchPlanner()
        wps = planner.generate_spiral_waypoints(center=[50.0, 50.0, 20.0], n_points=8)
        self.assertEqual(len(wps), 8)
        self.assertTrue(np.allclose(wps[0], [50.0, 50.0, 20.0]))


if __name__ == "__main__":
    unittest.main()
