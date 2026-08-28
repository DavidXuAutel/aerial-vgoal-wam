"""vgoal: Visual Object-Goal Aerial World-Action Model extension package."""

from vgoal.bridge import VisualGoalPolicyConfig, VisualGoalWAMPolicy
from vgoal.detector import BaseDetector, DetectionResult, MockDetector, YOLOTargetDetector
from vgoal.dynamic_tracker import DynamicTargetTracker, DynamicTrackerConfig, TrackingMode
from vgoal.geometry import CameraIntrinsics, bbox_to_goal_rel, project_3d_to_pixel
from vgoal.search_planner import AreaSearchPlanner, SearchAreaConfig, SearchPattern
from vgoal.tracker import TargetState, TargetTracker, TrackerConfig

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "MockDetector",
    "YOLOTargetDetector",
    "CameraIntrinsics",
    "bbox_to_goal_rel",
    "project_3d_to_pixel",
    "TargetState",
    "TargetTracker",
    "TrackerConfig",
    "VisualGoalWAMPolicy",
    "VisualGoalPolicyConfig",
    "AreaSearchPlanner",
    "SearchAreaConfig",
    "SearchPattern",
    "DynamicTargetTracker",
    "DynamicTrackerConfig",
    "TrackingMode",
]
