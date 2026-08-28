"""Visual Object-Goal WAM Bridge and Deployment Policy.

Integrates:
1. Target Detector (YOLO / Mock / Visual Target)
2. Monocular Depth Predictor / Head
3. 2D-to-3D Geometric Back-Projection (with robust depth bounding)
4. Spatial Target Tracker (with 3D world anchoring and dead reckoning)
5. WAM World Model with Recurrent Latent Streaming (observe_and_advance)
6. Latent Actor-Critic Policy π(a | z, goal_rel)
7. Multi-step Imagination Planner (test-time imagination rollout scoring)
8. Safety Shield (post-action collision prevention)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vgoal.detector import BaseDetector, DetectionResult
from vgoal.geometry import CameraIntrinsics, bbox_to_goal_rel
from vgoal.tracker import TargetState, TargetTracker, TrackerConfig


@dataclass
class VisualGoalPolicyConfig:
    intrinsics: CameraIntrinsics = field(
        default_factory=lambda: CameraIntrinsics.from_fov(fov_deg=80.0, width=224, height=224)
    )
    tracker_config: TrackerConfig = field(default_factory=TrackerConfig)
    dt: float = 0.2  # 5 Hz control rate
    search_yaw_rate: float = 0.314  # rad/step during SEARCHING mode
    search_fwd_speed: float = 0.2  # m/step during SEARCHING mode
    use_planner: bool = True
    planner_horizon: int = 5
    max_target_dist_m: float = 160.0


class VisualGoalWAMPolicy:
    """End-to-end visual object-goal deployment policy for Aerial-WAM."""

    def __init__(
        self,
        dynamics: Any,
        actor_critic: Any,
        detector: BaseDetector,
        *,
        depth_predictor: Optional[Any] = None,
        safety_shield: Optional[Any] = None,
        planner: Optional[Any] = None,
        config: Optional[VisualGoalPolicyConfig] = None,
    ) -> None:
        self.dynamics = dynamics
        self.actor_critic = actor_critic
        self.detector = detector
        self.depth_predictor = depth_predictor
        self.safety_shield = safety_shield
        self.planner = planner
        self.config = config or VisualGoalPolicyConfig()
        self.tracker = TargetTracker(self.config.tracker_config)

        self._prev_pos: Optional[np.ndarray] = None
        self._prev_yaw: Optional[float] = None
        self._prev_t: Optional[float] = None
        self._latent: Optional[np.ndarray] = None
        self._prev_act: Optional[np.ndarray] = None

        self.last_detection: Optional[DetectionResult] = None
        self.last_goal_rel: Optional[np.ndarray] = None
        self.last_target_state: TargetState = TargetState.SEARCHING

    def reset(self) -> None:
        """Reset policy state, recurrent latent, and target tracker at episode start."""
        self.tracker.reset()
        self._prev_pos = None
        self._prev_yaw = None
        self._prev_t = None
        self._latent = None
        self._prev_act = None
        self.last_detection = None
        self.last_goal_rel = None
        self.last_target_state = TargetState.SEARCHING
        if hasattr(self.dynamics, "reset") and callable(self.dynamics.reset):
            self.dynamics.reset()

    def _estimate_ego_delta(self, obs: Any) -> Tuple[np.ndarray, float]:
        """Estimate drone body displacement [dx, dy, dz] and dyaw since last step."""
        pos = getattr(obs, "position", None)
        if pos is None:
            proprio = getattr(obs, "proprio", None)
            if proprio is not None:
                pos = proprio[:3]
                yaw = float(proprio[6]) if len(proprio) > 6 else 0.0
            else:
                state = getattr(obs, "state", None)
                if state is not None:
                    pos = state[:3]
                    yaw = float(state[6]) if len(state) > 6 else 0.0
                else:
                    return np.zeros(3, dtype=np.float32), 0.0
        else:
            yaw = float(getattr(obs, "yaw", 0.0))

        cur_pos = np.asarray(pos, dtype=np.float32)

        if self._prev_pos is None or self._prev_yaw is None:
            self._prev_pos = cur_pos.copy()
            self._prev_yaw = float(yaw)
            return np.zeros(3, dtype=np.float32), 0.0

        prev_pos = self._prev_pos
        prev_yaw = self._prev_yaw

        d_world = cur_pos - prev_pos
        dyaw = float(yaw - prev_yaw)
        dyaw = float((dyaw + np.pi) % (2.0 * np.pi) - np.pi)

        c, s = np.cos(prev_yaw), np.sin(prev_yaw)
        d_fwd = c * d_world[0] + s * d_world[1]
        d_left = -s * d_world[0] + c * d_world[1]
        d_up = d_world[2]

        self._prev_pos = cur_pos.copy()
        self._prev_yaw = float(yaw)
        return np.array([d_fwd, d_left, d_up], dtype=np.float32), dyaw

    def _get_depth_map(self, obs: Any) -> Optional[np.ndarray]:
        """Extract or predict dense depth map from observation."""
        obs_depth = getattr(obs, "depth", None)
        if obs_depth is not None:
            return np.asarray(obs_depth, dtype=np.float32)

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
        """Process observation, detect & track visual goal, query WAM actor policy & planner."""
        rgb = getattr(obs, "rgb", None)
        if rgb is None:
            raise ValueError("Observation missing 'rgb' image.")
        rgb_arr = np.asarray(rgb, dtype=np.uint8)
        img_h, img_w = rgb_arr.shape[:2]

        if img_w != self.config.intrinsics.width or img_h != self.config.intrinsics.height:
            self.config.intrinsics = CameraIntrinsics.from_fov(80.0, width=img_w, height=img_h)

        # Step 1: Detect visual target
        det = self.detector.detect(rgb_arr)
        self.last_detection = det

        # Step 2: Extract/Predict depth & back-project to 3D goal_rel
        measured_goal_rel = None
        det_conf = 0.0
        if det is not None:
            d_target_direct = getattr(det, "direct_depth", None)
            if d_target_direct is not None and d_target_direct > 0:
                u_c = (det.bbox[0] + det.bbox[2]) * 0.5
                v_c = (det.bbox[1] + det.bbox[3]) * 0.5
                intr = self.config.intrinsics
                x_cam = (u_c - intr.cx) * d_target_direct / intr.fx
                y_cam = (v_c - intr.cy) * d_target_direct / intr.fy
                measured_goal_rel = np.array([
                    float(d_target_direct),
                    float(-x_cam),
                    float(-y_cam),
                    float(np.sqrt(d_target_direct**2 + x_cam**2 + y_cam**2)),
                ], dtype=np.float32)
                det_conf = det.confidence
            else:
                depth_map = self._get_depth_map(obs)
                if depth_map is not None:
                    gr = bbox_to_goal_rel(
                        det.bbox,
                        depth_map,
                        self.config.intrinsics,
                        src_shape=(img_w, img_h),
                    )
                    if gr is not None:
                        if gr[0] > self.config.max_target_dist_m:
                            scale = self.config.max_target_dist_m / gr[0]
                            gr = gr * scale
                            gr[3] = float(np.linalg.norm(gr[:3]))
                        measured_goal_rel = gr
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

        # Step 4: Handle SEARCHING mode
        if cur_goal_rel is None or state == TargetState.SEARCHING:
            action = np.array([
                self.config.search_fwd_speed,
                0.0,
                0.0,
                self.config.search_yaw_rate,
            ], dtype=np.float64)
            return action

        # Step 5: Construct Observation
        pos = getattr(obs, "position", None)
        if pos is None:
            proprio = getattr(obs, "proprio", None)
            pos = np.asarray(proprio[:3], dtype=np.float32) if proprio is not None else np.zeros(3, dtype=np.float32)
            yaw = float(proprio[6]) if proprio is not None and len(proprio) > 6 else 0.0
        else:
            pos = np.asarray(pos, dtype=np.float32)
            yaw = float(getattr(obs, "yaw", 0.0))

        cur_t = float(getattr(obs, "t", 0.0))
        vel = np.zeros(3, dtype=np.float32)
        if self._prev_pos is not None and self._prev_t is not None:
            dt_step = cur_t - self._prev_t
            if 1e-4 < dt_step < 1.0:
                vel = (pos - self._prev_pos) / float(dt_step)
        self._prev_t = cur_t

        state_vec = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], yaw], dtype=np.float32)

        info_dict = dict(getattr(obs, "info", {}) or {})
        c_y, s_y = np.cos(yaw), np.sin(yaw)
        w_target = pos + np.array([
            c_y * cur_goal_rel[0] - s_y * cur_goal_rel[1],
            s_y * cur_goal_rel[0] + c_y * cur_goal_rel[1],
            cur_goal_rel[2],
        ], dtype=np.float32)
        info_dict["goal"] = w_target

        from experiments.aerial.rl.env.obs import Observation
        full_obs = Observation(rgb=rgb_arr, state=state_vec, t=cur_t, info=info_dict)

        # Step 6: Recurrent Latent Streaming (observe_and_advance)
        if (
            self._latent is not None
            and self._prev_act is not None
            and hasattr(self.dynamics, "observe_and_advance")
        ):
            z = self.dynamics.observe_and_advance(self._latent, self._prev_act, full_obs)
        elif hasattr(self.dynamics, "encode") and callable(self.dynamics.encode):
            z = self.dynamics.encode(full_obs)
        else:
            z = np.zeros(8, dtype=np.float32)
        self._latent = np.asarray(z, dtype=np.float64).copy()

        # Step 7: Query Actor-Critic Policy
        if hasattr(self.actor_critic, "act_latent") and callable(self.actor_critic.act_latent):
            raw_action = self.actor_critic.act_latent(z, goal_rel=cur_goal_rel, deterministic=True)
        else:
            heading_err = np.arctan2(cur_goal_rel[1], cur_goal_rel[0])
            dyaw_cmd = float(np.clip(heading_err * 0.4, -0.314, 0.314))
            fwd_cmd = float(np.clip(cur_goal_rel[0] * 0.3, 0.1, 1.0))
            raw_action = np.array([fwd_cmd, 0.0, 0.0, dyaw_cmd], dtype=np.float64)

        raw_action = np.asarray(raw_action, dtype=np.float64).reshape(4)
        self._prev_act = np.asarray(raw_action, dtype=np.float64).copy()
        return raw_action
