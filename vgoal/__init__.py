"""vgoal: Visual Object-Goal Aerial World-Action Model extension package."""

from vgoal.geometry import CameraIntrinsics, bbox_to_goal_rel, project_3d_to_pixel
from vgoal.tracker import TargetState, TargetTracker, TrackerConfig

__all__ = [
    "CameraIntrinsics",
    "bbox_to_goal_rel",
    "project_3d_to_pixel",
    "TargetState",
    "TargetTracker",
    "TrackerConfig",
]
