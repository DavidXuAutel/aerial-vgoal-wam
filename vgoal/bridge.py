"""Visual Object-Goal WAM Bridge and Deployment Policy.

Integrates:
1. Target Detector (YOLO / Mock)
2. Monocular Depth Predictor / Head
3. 2D-to-3D Geometric Back-Projection
4. Spatial Target Tracker (with dead reckoning)
5. WAM World Model Encoder + Goal-Conditioned Actor-Critic Policy
6. Safety Shield (optional post-action collision clamping)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from vgoal.detector import BaseDetector, DetectionResult
from vgoal.geometry import CameraIntrinsics, bbox_to_goal_rel
from vgoal.tracker import TargetState, TargetTracker, TrackerConfig


@dataclass
class VisualGoalPolicyConfig:
    intrinsics: CameraIntrinsics = field(
        default_factory=lambda: CameraIntrinsics.from_fov(fov_deg=80.0, width=640, height=480)
    )
    tracker_config: TrackerConfig = field(default_factory=TrackerConfig)
    dt: float = 0.2  # 5 Hz control rate
    search_yaw_rate: float = 0.15  # rad/step during SEARCHING mode
    search_fwd_speed: float = 0.2  # m/step during SEARCHING mode


class VisualGoalWAMPolicy:
    """End-to-end visual object-goal deployment policy for Aerial-WAM.

    Wraps raw RGB observations, extracts visual goals via detection and depth back-projection,
    tracks targets across occlusions, and queries the learned WAM policy π(a | z, goal_rel).
    """

    def __init__(
        self,
        dynamics: Any,
        actor_critic: Any,
        detector: BaseDetector,
        *,
        depth_predictor: Optional[Any] = None,
        safety_shield: Optional[Any] = None,
        config: Optional[VisualGoalPolicyConfig] = None,
    ) -> None:
        self.dynamics = dynamics
        self.actor_critic = actor_critic
        self.detector = detector
        self.depth_predictor = depth_predictor
        self.safety_shield = safety_shield
        self.config = config or VisualGoalPolicyConfig()
        self.tracker = TargetTracker(self.config.tracker_config)

        self._last_state: Optional[np.ndarray] = None  # [x, y, z, vx, vy, vz, yaw]
        self.last_detection: Optional[DetectionResult] = None
        self.last_goal_rel: Optional[np.ndarray] = None
        self.last_target_state: TargetState = TargetState.SEARCHING

    def reset(self) -> None:
        """Reset policy state and target tracker at episode start."""
        self.tracker.reset()
        self._last_state = None
        self.last_detection = None
        self.last_goal_rel = None
        self.last_target_state = TargetState.SEARCHING
        if hasattr(self.dynamics, "reset") and callable(self.dynamics.reset):
            self.dynamics.reset()

    def _estimate_ego_delta(self, obs: Any) -> Tuple[np.ndarray, float]:
        """Estimate drone body displacement [dx, dy, dz] and dyaw since last step."""
        state = getattr(obs, "state", None)
        if state is None or self._last_state is None:
            if state is not None:
                self._last_state = np.asarray(state, dtype=np.float32).copy()
            return np.zeros(3, dtype=np.float32), 0.0

        cur_s = np.asarray(state, dtype=np.float32)
        prev_s = self._last_state

        # World displacement
        d_world = cur_s[:3] - prev_s[:3]
        prev_yaw = float(prev_s[6]) if len(prev_s) > 6 else 0.0
        cur_yaw = float(cur_s[6]) if len(cur_s) > 6 else 0.0
        dyaw = cur_yaw - prev_yaw
        # Wrap dyaw into [-pi, pi]
        dyaw = float((dyaw + np.pi) % (2.0 * np.pi) - np.pi)

        # Rotate world displacement into drone previous body frame
        c, s = np.cos(prev_yaw), np.sin(prev_yaw)
        d_fwd = c * d_world[0] + s * d_world[1]
        d_left = -s * d_world[0] + c * d_world[1]
        d_up = d_world[2]

        self._last_state = cur_s.copy()
        return np.array([d_fwd, d_left, d_up], dtype=np.float32), dyaw

    def _get_depth_map(self, obs: Any) -> Optional[np.ndarray]:
        """Extract or predict dense depth map from observation."""
        # 1. Direct depth from observation (if available from simulator or RGBD sensor)
        obs_depth = getattr(obs, "depth", None)
        if obs_depth is not None:
            return np.asarray(obs_depth, dtype=np.float32)

        # 2. Predicted depth from WAM monocular depth head
        if self.depth_predictor is not None:
            predict_fn = getattr(self.depth_predictor, "predict_depth", None) or getattr(
                self.depth_predictor, "predict", None
            )
            if callable(predict_fn):
                d = predict_fn(obs)
                if d is not None:
                    return np.asarray(d, dtype=np.float32)

        return None

    def act(self, obs: Any) -> np.ndarray:
        """Process observation, detect & track visual goal, query WAM actor policy."""
        rgb = getattr(obs, "rgb", None)
        if rgb is None:
            raise ValueError("Observation missing 'rgb' image.")
        rgb_arr = np.asarray(rgb, dtype=np.uint8)

        # Step 1: Detect visual target
        det = self.detector.detect(rgb_arr)
        self.last_detection = det

        # Step 2: Extract/Predict depth & back-project to 3D goal_rel
        measured_goal_rel = None
        det_conf = 0.0
        if det is not None:
            depth_map = self._get_depth_map(obs)
            if depth_map is not None:
                measured_goal_rel = bbox_to_goal_rel(
                    det.bbox, depth_map, self.config.intrinsics
                )
                det_conf = det.confidence

        # Step 3: Update spatial tracker with dead-reckoning
        ego_delta, dyaw = self._estimate_ego_delta(obs)
        state = self.tracker.update(
            measured_goal_rel,
            dt=self.config.dt,
            ego_delta_body=ego_delta,
            ego_delta_yaw=dyaw,
            confidence=det_conf,
        )
        self.last_target_state = state
        cur_goal_rel = self.tracker.goal_rel
        self.last_goal_rel = cur_goal_rel

        # Step 4: Handle SEARCHING mode (no active or extrapolated target)
        if cur_goal_rel is None or state == TargetState.SEARCHING:
            # Execute gentle search maneuver (forward scan + rotation)
            action = np.array([
                self.config.search_fwd_speed,
                0.0,
                0.0,
                self.config.search_yaw_rate,
            ], dtype=np.float64)
            return action

        # Step 5: Encode latent state z through WAM Dynamics
        if hasattr(self.dynamics, "encode") and callable(self.dynamics.encode):
            z = self.dynamics.encode(obs)
        else:
            # Fallback stub for unit testing
            z = np.zeros(8, dtype=np.float32)

        # Step 6: Query WAM Goal-Conditioned Policy: π(a | z, goal_rel)
        if hasattr(self.actor_critic, "act_latent") and callable(self.actor_critic.act_latent):
            action = self.actor_critic.act_latent(z, goal_rel=cur_goal_rel, deterministic=True)
        elif hasattr(self.actor_critic, "act") and callable(self.actor_critic.act):
            action = self.actor_critic.act(obs)
        else:
            # Proportional guidance fallback
            heading_err = np.arctan2(cur_goal_rel[1], cur_goal_rel[0])
            dyaw_cmd = float(np.clip(heading_err * 0.4, -0.314, 0.314))
            fwd_cmd = float(np.clip(cur_goal_rel[0] * 0.3, 0.1, 1.0))
            action = np.array([fwd_cmd, 0.0, 0.0, dyaw_cmd], dtype=np.float64)

        action = np.asarray(action, dtype=np.float64).reshape(4)

        # Step 7: Optional Safety Shield clamping
        if self.safety_shield is not None:
            apply_fn = getattr(self.safety_shield, "apply_action", None)
            if callable(apply_fn):
                action, _ = apply_fn(action, obs)

        return action
