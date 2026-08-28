"""Unit tests for VisualGoalWAMPolicy bridge."""

import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from vgoal.bridge import VisualGoalPolicyConfig, VisualGoalWAMPolicy
from vgoal.detector import MockDetector
from vgoal.geometry import CameraIntrinsics
from vgoal.tracker import TargetState


@dataclass
class MockObs:
    rgb: np.ndarray = field(default_factory=lambda: np.zeros((480, 640, 3), dtype=np.uint8))
    depth: Optional[np.ndarray] = None
    state: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    info: Dict[str, Any] = field(default_factory=dict)


class MockDynamics:
    def __init__(self, latent_dim: int = 8) -> None:
        self.latent_dim = latent_dim

    def encode(self, obs: Any) -> np.ndarray:
        return np.ones(self.latent_dim, dtype=np.float32)


class MockActorCritic:
    def __init__(self) -> None:
        self.last_query_z: Optional[np.ndarray] = None
        self.last_query_goal_rel: Optional[np.ndarray] = None

    def act_latent(
        self, z: np.ndarray, goal_rel: Optional[np.ndarray] = None, deterministic: bool = True
    ) -> np.ndarray:
        self.last_query_z = np.asarray(z).copy()
        self.last_query_goal_rel = np.asarray(goal_rel).copy() if goal_rel is not None else None
        # Return a simulated forward action
        return np.array([0.6, 0.0, 0.0, 0.0], dtype=np.float64)


class MockDepthPredictor:
    def __init__(self, const_depth: float = 12.0) -> None:
        self.const_depth = float(const_depth)

    def predict_depth(self, obs: Any) -> np.ndarray:
        return np.full((480, 640), self.const_depth, dtype=np.float32)


class TestBridge(unittest.TestCase):

    def test_search_mode_when_no_target(self):
        detector = MockDetector(target_bbox=None)
        dyn = MockDynamics()
        ac = MockActorCritic()
        cfg = VisualGoalPolicyConfig(search_yaw_rate=0.2, search_fwd_speed=0.3)
        policy = VisualGoalWAMPolicy(dyn, ac, detector, config=cfg)

        obs = MockObs()
        action = policy.act(obs)

        self.assertEqual(policy.last_target_state, TargetState.SEARCHING)
        self.assertIsNone(policy.last_goal_rel)
        # In search mode, returns search maneuver without querying actor
        self.assertTrue(np.allclose(action, [0.3, 0.0, 0.0, 0.2]))
        self.assertIsNone(ac.last_query_goal_rel)

    def test_target_detected_queries_actor_with_goal_rel(self):
        detector = MockDetector(target_bbox=[300, 220, 340, 260], confidence=0.9)
        dyn = MockDynamics()
        ac = MockActorCritic()
        depth_pred = MockDepthPredictor(const_depth=15.0)
        policy = VisualGoalWAMPolicy(dyn, ac, detector, depth_predictor=depth_pred)

        obs = MockObs()  # obs.depth is None, will use depth_predictor
        action = policy.act(obs)

        self.assertEqual(policy.last_target_state, TargetState.TRACKING)
        self.assertIsNotNone(policy.last_goal_rel)
        # Target was centered at 15m depth
        self.assertTrue(np.isclose(policy.last_goal_rel[0], 15.0, atol=1e-3))
        self.assertTrue(np.isclose(policy.last_goal_rel[1], 0.0, atol=1e-3))

        # Check actor received goal_rel
        self.assertIsNotNone(ac.last_query_goal_rel)
        self.assertTrue(np.isclose(ac.last_query_goal_rel[0], 15.0, atol=1e-3))
        self.assertTrue(np.allclose(action, [0.6, 0.0, 0.0, 0.0]))

    def test_target_occluded_continues_querying_with_dead_reckoning(self):
        detector = MockDetector(target_bbox=[300, 220, 340, 260], confidence=0.95)
        dyn = MockDynamics()
        ac = MockActorCritic()
        depth_pred = MockDepthPredictor(const_depth=10.0)
        policy = VisualGoalWAMPolicy(dyn, ac, detector, depth_predictor=depth_pred)

        # Step 1: Target seen at 10m forward, drone at x=0
        obs1 = MockObs(state=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        policy.act(obs1)
        self.assertEqual(policy.last_target_state, TargetState.TRACKING)

        # Step 2: Target OCCLUDED, drone moved forward 2.0m to x=2.0
        detector.set_target(None)
        obs2 = MockObs(state=np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        action2 = policy.act(obs2)

        self.assertEqual(policy.last_target_state, TargetState.OCCLUDED)
        self.assertIsNotNone(policy.last_goal_rel)
        # Distance should now be 10.0 - 2.0 = 8.0m
        self.assertTrue(np.isclose(policy.last_goal_rel[0], 8.0, atol=1e-3))
        self.assertTrue(np.isclose(ac.last_query_goal_rel[0], 8.0, atol=1e-3))
        self.assertTrue(np.allclose(action2, [0.6, 0.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
