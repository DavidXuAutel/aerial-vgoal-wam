"""Unit tests for CameraIntrinsics and 2D-to-3D back-projection."""

import unittest
import numpy as np

from vgoal.geometry import CameraIntrinsics, bbox_to_goal_rel, extract_target_depth, project_3d_to_pixel


class TestGeometry(unittest.TestCase):

    def test_intrinsics_from_fov(self):
        cam = CameraIntrinsics.from_fov(fov_deg=90.0, width=640, height=480)
        self.assertTrue(np.isclose(cam.fx, 320.0))
        self.assertTrue(np.isclose(cam.fy, 320.0))
        self.assertTrue(np.isclose(cam.cx, 320.0))
        self.assertTrue(np.isclose(cam.cy, 240.0))

    def test_extract_target_depth_robust_to_background(self):
        depth_map = np.full((100, 100), 50.0, dtype=np.float32)
        depth_map[35:65, 35:65] = 5.0
        d = extract_target_depth(depth_map, [30, 30, 70, 70], core_frac=0.5)
        self.assertTrue(np.isclose(d, 5.0))

    def test_bbox_to_goal_rel_centered(self):
        cam = CameraIntrinsics.from_fov(fov_deg=90.0, width=640, height=480)
        depth_map = np.full((480, 640), 10.0, dtype=np.float32)

        bbox = [300, 220, 340, 260]
        goal_rel = bbox_to_goal_rel(bbox, depth_map, cam)

        self.assertIsNotNone(goal_rel)
        self.assertTrue(np.isclose(goal_rel[0], 10.0, atol=1e-4))
        self.assertTrue(np.isclose(goal_rel[1], 0.0, atol=1e-4))
        self.assertTrue(np.isclose(goal_rel[2], 0.0, atol=1e-4))
        self.assertTrue(np.isclose(goal_rel[3], 10.0, atol=1e-4))

    def test_bbox_to_goal_rel_off_center(self):
        cam = CameraIntrinsics.from_fov(fov_deg=90.0, width=640, height=480)
        depth_map = np.full((480, 640), 10.0, dtype=np.float32)

        bbox = [140, 60, 180, 100]
        goal_rel = bbox_to_goal_rel(bbox, depth_map, cam)

        self.assertIsNotNone(goal_rel)
        self.assertTrue(np.isclose(goal_rel[0], 10.0, atol=1e-4))
        self.assertTrue(np.isclose(goal_rel[1], 5.0, atol=1e-4))
        self.assertTrue(np.isclose(goal_rel[2], 5.0, atol=1e-4))
        self.assertTrue(np.isclose(goal_rel[3], np.sqrt(150.0), atol=1e-4))

    def test_round_trip_projection(self):
        cam = CameraIntrinsics.from_fov(fov_deg=80.0, width=640, height=480)
        p_body = [12.0, 3.5, -1.2]

        u, v, z = project_3d_to_pixel(p_body, cam)
        self.assertTrue(np.isfinite(u) and np.isfinite(v))
        self.assertTrue(np.isclose(z, 12.0))

        depth_map = np.full((480, 640), 12.0, dtype=np.float32)
        bbox = [u - 5, v - 5, u + 5, v + 5]
        goal_rel = bbox_to_goal_rel(bbox, depth_map, cam)

        self.assertIsNotNone(goal_rel)
        self.assertTrue(np.isclose(goal_rel[0], p_body[0], atol=1e-3))
        self.assertTrue(np.isclose(goal_rel[1], p_body[1], atol=1e-3))
        self.assertTrue(np.isclose(goal_rel[2], p_body[2], atol=1e-3))


if __name__ == "__main__":
    unittest.main()
