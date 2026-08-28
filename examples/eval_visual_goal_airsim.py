"""Closed-loop evaluation script for Visual Object-Goal WAM.

Loads:
1. Trained Torch RSSM Dynamics (World Model)
2. Goal-Conditioned LatentActorCritic Policy
3. Trained Monocular Depth Predictor Head
4. Real YOLOv8n Target Detector

Runs visual closed-loop navigation episodes in AirSim or Mock environment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

from vgoal.bridge import VisualGoalPolicyConfig, VisualGoalWAMPolicy
from vgoal.detector import MockDetector, YOLOTargetDetector
from vgoal.geometry import CameraIntrinsics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Visual Object-Goal WAM in closed-loop")
    parser.add_argument("--wm-ckpt", type=str, default="", help="Path to trained Torch RSSM checkpoint (.pt)")
    parser.add_argument("--actor-ckpt", type=str, default="", help="Path to trained LatentActorCritic checkpoint (.pt)")
    parser.add_argument("--depth-ckpt", type=str, default="", help="Path to trained DepthHead checkpoint (.pt)")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--target-class", type=str, default="car", help="Target object class to track")
    parser.add_argument("--backend", type=str, choices=["airsim", "mock"], default="mock", help="Simulation backend")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--max-steps", type=int, default=120, help="Max steps per episode (24s at 5Hz)")
    parser.add_argument("--step-hz", type=float, default=5.0, help="Control loop rate")
    parser.add_argument("--fov-deg", type=float, default=80.0, help="Camera horizontal FOV")
    parser.add_argument("--out-report", type=str, default="artifacts/visual_goal_eval_report.json", help="Output path")
    return parser.parse_args()


def load_wam_components(args: argparse.Namespace):
    """Load WAM modules from aerial-wam-v2 codebase if available."""
    try:
        from experiments.aerial.rl.dynamics_torch import TorchRSSMDynamics
        from experiments.aerial.rl.actor_critic import LatentActorCritic
        from experiments.aerial.rl.depth_predictor import MonocularDepthPredictor
        from experiments.aerial.rl.safety import SafetyShield
    except ImportError:
        print("[Warn] aerial-wam-v2 modules not in sys.path; running in self-contained mock mode.")
        return None, None, None, None

    dynamics = None
    if args.wm_ckpt and os.path.isfile(args.wm_ckpt):
        print(f"Loading TorchRSSMDynamics from {args.wm_ckpt}...")
        dynamics = TorchRSSMDynamics.load_from_checkpoint(args.wm_ckpt)

    actor_critic = None
    if args.actor_ckpt and os.path.isfile(args.actor_ckpt):
        print(f"Loading LatentActorCritic from {args.actor_ckpt}...")
        actor_critic = LatentActorCritic.load_from_checkpoint(args.actor_ckpt)

    depth_predictor = None
    if args.depth_ckpt and os.path.isfile(args.depth_ckpt):
        print(f"Loading MonocularDepthPredictor from {args.depth_ckpt}...")
        depth_predictor = MonocularDepthPredictor.load_from_checkpoint(args.depth_ckpt)

    safety_shield = None
    if depth_predictor is not None:
        safety_shield = SafetyShield(depth_predictor=depth_predictor)

    return dynamics, actor_critic, depth_predictor, safety_shield


def main():
    args = parse_args()
    print("=====================================================================")
    print("🎯 Visual Object-Goal Aerial WAM Evaluator")
    print(f"Backend: {args.backend}, Target Class: {args.target_class}, Rate: {args.step_hz} Hz")
    print("=====================================================================")

    # 1. Setup Camera Intrinsics
    intrinsics = CameraIntrinsics.from_fov(fov_deg=args.fov_deg, width=640, height=480)
    cfg = VisualGoalPolicyConfig(intrinsics=intrinsics, dt=1.0 / args.step_hz)

    # 2. Setup Detector
    if args.backend == "mock" and not os.path.exists(args.yolo_model):
        print("Using MockDetector for target simulation.")
        detector = MockDetector(target_bbox=[280, 200, 360, 280], class_name=args.target_class)
    else:
        print(f"Loading YOLO detector: {args.yolo_model} (target={args.target_class})...")
        detector = YOLOTargetDetector(
            model_path=args.yolo_model,
            target_classes={args.target_class},
            conf_threshold=0.4,
            imgsz=640,
        )

    # 3. Load WAM Neural Models
    dyn, ac, depth_pred, shield = load_wam_components(args)

    # 4. Construct End-to-End Bridge Policy
    policy = VisualGoalWAMPolicy(
        dynamics=dyn,
        actor_critic=ac,
        detector=detector,
        depth_predictor=depth_pred,
        safety_shield=shield,
        config=cfg,
    )

    print("\n✅ VisualGoalWAMPolicy successfully assembled and ready for execution.")
    print("Interface signature: action = policy.act(obs)")


if __name__ == "__main__":
    main()
