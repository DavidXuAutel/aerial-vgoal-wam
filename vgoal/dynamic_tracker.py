"""Dynamic 3D Target State Estimator (EKF) and Standoff Following Controller.

Maintains 6-DoF continuous target state [px, py, pz, vx, vy, vz] in world coordinates,
performs Kalman filtering with process/measurement noise, handles occlusion dead-reckoning,
and computes dynamic standoff following goal vectors (e.g. 6m behind, 3m above).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence, Tuple

import numpy as np


class TrackingMode(str, Enum):
    SEARCHING = "searching"          # No active target locked
    INTERCEPTING = "intercepting"    # Closing in on target from long distance
    FOLLOWING = "following"          # Maintaining standoff following geometry
    OCCLUDED = "occluded"            # Dead-reckoning during visual loss
    LOST = "lost"                    # Target lost beyond max occlusion timeout


@dataclass
class DynamicTrackerConfig:
    standoff_dist_m: float = 6.0       # Distance behind target to follow
    standoff_height_m: float = 3.0     # Height above target to follow
    min_moving_speed_mps: float = 0.3  # Speed threshold to identify heading direction
    max_occlusion_s: float = 3.0       # Max dead-reckoning window before triggering LOST
    process_pos_noise: float = 0.1     # EKF process position noise
    process_vel_noise: float = 0.5     # EKF process velocity noise
    meas_pos_noise: float = 0.4        # EKF visual back-projection measurement noise
    intercept_dist_thresh_m: float = 12.0  # Threshold to switch between INTERCEPTING and FOLLOWING


class DynamicTargetTracker:
    """3D Kalman Filter + Dynamic Standoff Following Calculator."""

    def __init__(self, config: Optional[DynamicTrackerConfig] = None) -> None:
        self.config = config or DynamicTrackerConfig()
        self.mode: TrackingMode = TrackingMode.SEARCHING
        # State: [px, py, pz, vx, vy, vz]^T
        self.x: np.ndarray = np.zeros(6, dtype=np.float64)
        self.P: np.ndarray = np.eye(6, dtype=np.float64) * 10.0
        self._time_since_seen: float = float("inf")
        self._is_initialized: bool = False

    def reset(self) -> None:
        """Reset estimator state."""
        self.mode = TrackingMode.SEARCHING
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 10.0
        self._time_since_seen = float("inf")
        self._is_initialized = False

    @property
    def target_position_world(self) -> Optional[np.ndarray]:
        if not self._is_initialized or self.mode == TrackingMode.SEARCHING or self.mode == TrackingMode.LOST:
            return None
        return self.x[:3].copy()

    @property
    def target_velocity_world(self) -> Optional[np.ndarray]:
        if not self._is_initialized or self.mode == TrackingMode.SEARCHING or self.mode == TrackingMode.LOST:
            return None
        return self.x[3:6].copy()

    def predict(self, dt: float) -> None:
        """EKF Prediction step using constant-velocity motion model."""
        if not self._is_initialized:
            return
        dt = max(1e-4, float(dt))
        # F matrix
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Process noise Q
        Q = np.zeros((6, 6), dtype=np.float64)
        qp = (self.config.process_pos_noise * dt) ** 2
        qv = (self.config.process_vel_noise * dt) ** 2
        for i in range(3):
            Q[i, i] = qp
            Q[i + 3, i + 3] = qv

        # x = F * x
        self.x = F @ self.x
        # P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + Q

    def update_measurement(self, measured_pos_world: Sequence[float], dt: float) -> None:
        """EKF Measurement update step given 3D world target position."""
        dt = max(1e-4, float(dt))
        z = np.asarray(measured_pos_world, dtype=np.float64).reshape(3)

        if not self._is_initialized:
            self.x = np.zeros(6, dtype=np.float64)
            self.x[:3] = z
            self.P = np.eye(6, dtype=np.float64) * 1.0
            self._is_initialized = True
            self._time_since_seen = 0.0
            self.mode = TrackingMode.INTERCEPTING
            return

        # EKF Measurement matrix H: maps 6D state to 3D position
        H = np.zeros((3, 6), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        # Measurement noise R
        R = np.eye(3, dtype=np.float64) * (self.config.meas_pos_noise ** 2)

        # Innovation: y = z - H * x
        y = z - H @ self.x
        # S = H * P * H^T + R
        S = H @ self.P @ H.T + R
        # K = P * H^T * S^-1
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ H) @ self.P

        self._time_since_seen = 0.0

    def compute_standoff_target_world(self, drone_pos: Sequence[float]) -> Optional[np.ndarray]:
        """Compute the 3D world coordinate where the drone should fly to accompany the target."""
        if not self._is_initialized or self.mode in (TrackingMode.SEARCHING, TrackingMode.LOST):
            return None

        p_tgt = self.x[:3]
        v_tgt = self.x[3:6]
        speed = float(np.linalg.norm(v_tgt[:2]))  # Horizontal speed

        d_back = self.config.standoff_dist_m
        h_above = self.config.standoff_height_m

        if speed >= self.config.min_moving_speed_mps:
            # Moving target: follow behind along velocity heading
            heading_unit = v_tgt[:2] / speed
            offset_xy = -d_back * heading_unit
        else:
            # Stationary target: maintain standoff distance relative to current drone approach vector
            drone_xy = np.asarray(drone_pos[:2], dtype=np.float64)
            diff = drone_xy - p_tgt[:2]
            dist_xy = float(np.linalg.norm(diff))
            if dist_xy > 1e-3:
                offset_xy = d_back * (diff / dist_xy)
            else:
                offset_xy = np.array([-d_back, 0.0], dtype=np.float64)

        standoff_p = np.array([
            p_tgt[0] + offset_xy[0],
            p_tgt[1] + offset_xy[1],
            p_tgt[2] + h_above,
        ], dtype=np.float64)

        return standoff_p

    def step(
        self,
        measured_body_rel: Optional[Sequence[float]],
        drone_pos: Sequence[float],
        drone_yaw: float,
        dt: float,
        confidence: float = 1.0,
    ) -> Tuple[TrackingMode, Optional[np.ndarray]]:
        """Main tracker update step.

        Args:
            measured_body_rel: Raw back-projected [d_fwd, d_left, d_up, dist] if visual detection succeeded.
            drone_pos: Current drone world position [x, y, z].
            drone_yaw: Current drone yaw in radians.
            dt: Time elapsed in seconds.
            confidence: Detector confidence score.

        Returns:
            (TrackingMode, Optional[np.ndarray goal_rel [4D in body frame]])
        """
        pos_d = np.asarray(drone_pos, dtype=np.float64).reshape(3)

        # 1. Prediction step
        self.predict(dt)

        # 2. Measurement update if target seen
        if measured_body_rel is not None and confidence >= 0.4:
            b_rel = np.asarray(measured_body_rel[:3], dtype=np.float64)
            c, s = math.cos(drone_yaw), math.sin(drone_yaw)
            w_offset = np.array([
                c * b_rel[0] - s * b_rel[1],
                s * b_rel[0] + c * b_rel[1],
                b_rel[2],
            ], dtype=np.float64)
            measured_pos_w = pos_d + w_offset
            self.update_measurement(measured_pos_w, dt)

            # Update state machine
            dist_to_tgt = float(np.linalg.norm(self.x[:3] - pos_d))
            if dist_to_tgt <= self.config.intercept_dist_thresh_m:
                self.mode = TrackingMode.FOLLOWING
            else:
                self.mode = TrackingMode.INTERCEPTING
        else:
            # Target occluded/missing in this step
            if self._is_initialized:
                self._time_since_seen += dt
                if self._time_since_seen <= self.config.max_occlusion_s:
                    self.mode = TrackingMode.OCCLUDED
                else:
                    self.mode = TrackingMode.LOST
            else:
                self.mode = TrackingMode.SEARCHING

        # 3. Compute output body-frame goal_rel vector
        if self.mode in (TrackingMode.SEARCHING, TrackingMode.LOST):
            return self.mode, None

        standoff_w = self.compute_standoff_target_world(pos_d)
        if standoff_w is None:
            return self.mode, None

        # Transform standoff waypoint into drone body frame
        delta_w = standoff_w - pos_d
        c, s = math.cos(drone_yaw), math.sin(drone_yaw)
        d_fwd = float(c * delta_w[0] + s * delta_w[1])
        d_left = float(-s * delta_w[0] + c * delta_w[1])
        d_up = float(delta_w[2])
        dist = float(math.sqrt(d_fwd**2 + d_left**2 + d_up**2))

        goal_rel_body = np.array([d_fwd, d_left, d_up, dist], dtype=np.float32)
        return self.mode, goal_rel_body
