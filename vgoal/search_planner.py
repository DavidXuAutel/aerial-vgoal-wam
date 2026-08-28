"""Area Coverage Search Planner for autonomous visual exploration.

Generates lawnmower (boustrophedon) grid patterns and expanding loiter search paths
within arbitrary 2D bounding boxes / polygonal regions. Outputs body-relative goal vectors
to seamlessly drive the WAM low-level policy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np


class SearchPattern(str, Enum):
    LAWNMOWER = "lawnmower"
    EXPANDING_SPIRAL = "expanding_spiral"


@dataclass
class SearchAreaConfig:
    """Bounding box or polygon definition for autonomous search area."""
    min_x: float = -50.0
    max_x: float = 50.0
    min_y: float = -50.0
    max_y: float = 50.0
    altitude_z: float = 30.0  # Cruise altitude in meters
    sweep_spacing_m: float = 15.0  # Distance between parallel sweep lanes
    waypoint_reach_radius_m: float = 3.5  # Radius to consider waypoint reached
    loop: bool = True  # Whether to repeat pattern when finished


class AreaSearchPlanner:
    """Stateful waypoint generator and tracker for area coverage search."""

    def __init__(self, config: Optional[SearchAreaConfig] = None) -> None:
        self.config = config or SearchAreaConfig()
        self.waypoints: List[np.ndarray] = []
        self.current_wp_idx: int = 0
        self.is_completed: bool = False
        self.generate_lawnmower_waypoints()

    def reset(self) -> None:
        """Reset search progression to start."""
        self.current_wp_idx = 0
        self.is_completed = False

    def generate_lawnmower_waypoints(self) -> List[np.ndarray]:
        """Generate boustrophedon (lawnmower) survey grid."""
        cfg = self.config
        xs = np.arange(cfg.min_x, cfg.max_x + 1e-3, cfg.sweep_spacing_m)
        wps = []

        flip = False
        for x in xs:
            if not flip:
                wps.append(np.array([x, cfg.min_y, cfg.altitude_z], dtype=np.float64))
                wps.append(np.array([x, cfg.max_y, cfg.altitude_z], dtype=np.float64))
            else:
                wps.append(np.array([x, cfg.max_y, cfg.altitude_z], dtype=np.float64))
                wps.append(np.array([x, cfg.min_y, cfg.altitude_z], dtype=np.float64))
            flip = not flip

        self.waypoints = wps
        self.current_wp_idx = 0
        self.is_completed = False
        return self.waypoints

    def generate_spiral_waypoints(
        self,
        center: Sequence[float],
        max_radius: float = 25.0,
        radial_step: float = 5.0,
        n_points: int = 16,
    ) -> List[np.ndarray]:
        """Generate expanding Archimedean spiral search around a localized center."""
        cx, cy, cz = center[:3]
        wps = []
        thetas = np.linspace(0, 4.0 * math.pi, n_points)
        for th in thetas:
            r = min(max_radius, radial_step * (th / (2.0 * math.pi)))
            x = cx + r * math.cos(th)
            y = cy + r * math.sin(th)
            wps.append(np.array([x, y, cz], dtype=np.float64))

        self.waypoints = wps
        self.current_wp_idx = 0
        self.is_completed = False
        return self.waypoints

    @property
    def current_target_world(self) -> Optional[np.ndarray]:
        """Current target waypoint in world coordinates [x, y, z]."""
        if not self.waypoints:
            return None
        if self.current_wp_idx >= len(self.waypoints):
            return self.waypoints[-1] if not self.config.loop else self.waypoints[0]
        return self.waypoints[self.current_wp_idx]

    def update(self, drone_pos: Sequence[float], drone_yaw: float) -> np.ndarray:
        """Update tracker with drone position and compute body-frame goal_rel vector.

        Returns:
            np.ndarray [d_fwd, d_left, d_up, dist] in body frame.
        """
        if not self.waypoints:
            return np.array([10.0, 0.0, 0.0, 10.0], dtype=np.float32)

        pos = np.asarray(drone_pos, dtype=np.float64).reshape(3)
        target = self.current_target_world
        dist = float(np.linalg.norm(target - pos))

        # Waypoint arrival check
        if dist <= self.config.waypoint_reach_radius_m:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.waypoints):
                if self.config.loop:
                    self.current_wp_idx = 0
                else:
                    self.is_completed = True
                    self.current_wp_idx = len(self.waypoints) - 1
            target = self.current_target_world
            dist = float(np.linalg.norm(target - pos))

        # Convert target into drone body frame
        delta_w = target - pos
        c, s = math.cos(drone_yaw), math.sin(drone_yaw)
        d_fwd = float(c * delta_w[0] + s * delta_w[1])
        d_left = float(-s * delta_w[0] + c * delta_w[1])
        d_up = float(delta_w[2])
        d_norm = float(math.sqrt(d_fwd**2 + d_left**2 + d_up**2))

        return np.array([d_fwd, d_left, d_up, d_norm], dtype=np.float32)
