"""Spatial target tracker with state machine and dead-reckoning extrapolation.

Maintains continuous 3D relative goal estimates even during brief detector misses
or obstacle occlusions.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np


class TargetState(str, Enum):
    SEARCHING = "searching"    # No target visible or memory expired
    TRACKING = "tracking"      # Target actively detected in camera FOV
    OCCLUDED = "occluded"      # Target briefly hidden, dead-reckoning active
    ARRIVED = "arrived"        # Within success radius


@dataclass
class TrackerConfig:
    success_dist_m: float = 3.0       # Distance threshold to trigger ARRIVED
    max_occlusion_s: float = 2.0      # Max duration to hold target in OCCLUDED state
    ema_alpha: float = 0.7            # Smoothing factor for newly detected target (1.0 = no smoothing)
    min_confidence: float = 0.5       # Detection confidence threshold
    near_dist_m: float = 35.0         # Below this, trust fresh measurements more
    near_ema_alpha: float = 0.92      # EMA weight on new measurement when near / closing
    inflate_reject_m: float = 4.0       # Ignore sudden depth inflation beyond this (m)
    inflate_alpha: float = 0.15         # EMA weight when measurement jumps farther
    freeze_dist_on_occlude: bool = True  # Static target: OCCLUDED dead-reckoning may not inflate range


class TargetTracker:
    """Stateful 3D spatial target tracker."""

    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        self.config = config or TrackerConfig()
        self.state: TargetState = TargetState.SEARCHING
        self._target_body: Optional[np.ndarray] = None  # [d_fwd, d_left, d_up] in body frame
        self._time_since_last_seen: float = float("inf")

    def reset(self) -> None:
        """Reset tracker state to initial searching mode."""
        self.state = TargetState.SEARCHING
        self._target_body = None
        self._time_since_last_seen = float("inf")

    @property
    def goal_rel(self) -> Optional[np.ndarray]:
        """Current 4D goal relative vector [d_fwd, d_left, d_up, dist], or None."""
        if self._target_body is None or self.state == TargetState.SEARCHING:
            return None
        dist = float(np.linalg.norm(self._target_body))
        return np.array([
            self._target_body[0],
            self._target_body[1],
            self._target_body[2],
            dist
        ], dtype=np.float32)

    def _ema_alpha_for_measurement(self, raw_p: np.ndarray) -> float:
        cfg = self.config
        alpha = float(np.clip(cfg.ema_alpha, 0.05, 1.0))
        meas_dist = float(np.linalg.norm(raw_p))
        if meas_dist <= float(cfg.near_dist_m):
            alpha = max(alpha, float(cfg.near_ema_alpha))
        if self._target_body is not None:
            cur_dist = float(np.linalg.norm(self._target_body))
            if meas_dist < cur_dist - 0.5:
                alpha = max(alpha, float(cfg.near_ema_alpha))
            elif meas_dist > cur_dist + float(cfg.inflate_reject_m):
                alpha = min(alpha, float(cfg.inflate_alpha))
        return float(np.clip(alpha, 0.05, 1.0))

    def update(
        self,
        measured_goal_rel: Optional[Sequence[float]],
        dt: float,
        *,
        ego_delta_body: Optional[Sequence[float]] = None,
        ego_delta_yaw: float = 0.0,
        confidence: float = 1.0,
    ) -> TargetState:
        """Update tracker state given current step observations.

        Args:
            measured_goal_rel: Raw back-projected [d_fwd, d_left, d_up, dist] if target was detected.
            dt: Elapsed time since last update in seconds.
            ego_delta_body: Drone body-frame displacement [dx, dy, dz] during this step (for dead reckoning).
            ego_delta_yaw: Drone yaw rotation in radians during this step (positive = turn left).
            confidence: Detector confidence score.

        Returns:
            Current TargetState.
        """
        dt = max(1e-4, float(dt))

        # 1. If target is currently seen with sufficient confidence
        if measured_goal_rel is not None and float(confidence) >= self.config.min_confidence:
            raw_p = np.asarray(measured_goal_rel[:3], dtype=np.float32)

            if self._target_body is None or self.state == TargetState.SEARCHING:
                self._target_body = raw_p
            else:
                alpha = self._ema_alpha_for_measurement(raw_p)
                self._target_body = alpha * raw_p + (1.0 - alpha) * self._target_body

            self._time_since_last_seen = 0.0
            dist = float(np.linalg.norm(self._target_body))
            if dist <= self.config.success_dist_m:
                self.state = TargetState.ARRIVED
            else:
                self.state = TargetState.TRACKING
            return self.state

        # 2. Target NOT seen in this frame: apply dead-reckoning extrapolation if we have prior memory
        self._time_since_last_seen += dt

        if self._target_body is not None and self._time_since_last_seen <= self.config.max_occlusion_s:
            # Dead-reckoning update based on drone ego-motion
            prev_dist = float(np.linalg.norm(self._target_body))
            p_prev = self._target_body.copy()

            # Subtract drone translation in body frame
            if ego_delta_body is not None:
                d_ego = np.asarray(ego_delta_body[:3], dtype=np.float32)
                p_prev = p_prev - d_ego

            # Rotate relative target vector by inverse yaw change
            # (drone turns left by +dyaw => relative target shifts right by -dyaw)
            if abs(ego_delta_yaw) > 1e-6:
                c = np.cos(-ego_delta_yaw)
                s = np.sin(-ego_delta_yaw)
                # p_body: [0] = fwd(x), [1] = left(y), [2] = up(z)
                fwd_new = c * p_prev[0] - s * p_prev[1]
                left_new = s * p_prev[0] + c * p_prev[1]
                p_prev[0] = fwd_new
                p_prev[1] = left_new

            new_dist = float(np.linalg.norm(p_prev))
            if self.config.freeze_dist_on_occlude and new_dist > prev_dist + 1e-3:
                p_prev = p_prev * (prev_dist / max(new_dist, 1e-3))
            self._target_body = p_prev
            dist = float(np.linalg.norm(self._target_body))
            if dist <= self.config.success_dist_m:
                self.state = TargetState.ARRIVED
            else:
                self.state = TargetState.OCCLUDED
            return self.state

        # 3. Memory expired or no prior target
        self.state = TargetState.SEARCHING
        self._target_body = None
        return self.state
