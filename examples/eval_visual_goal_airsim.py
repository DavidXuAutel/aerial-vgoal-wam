"""Closed-loop evaluation script for Visual Object-Goal WAM using official RolloutCollector pipeline.

Loads:
1. AirSim Environment from aerial-wam-v2
2. Trained Torch RSSM Dynamics (World Model)
3. Goal-Conditioned LatentActorCritic Policy
4. ImaginationPlanner (5-step rollout search)
5. Visual Goal Target Detector + Depth Back-Projection + Spatial Tracker
6. Safety Shield
7. RolloutCollector

Evaluates visual closed-loop navigation on AirSim test routes.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Ensure aerial-wam-v2 is importable
wam_v2_path = os.path.expanduser("~/aerial-wam-v2")
if os.path.isdir(wam_v2_path) and wam_v2_path not in sys.path:
    sys.path.insert(0, wam_v2_path)

from vgoal.bridge import VisualGoalPolicyConfig, VisualGoalWAMPolicy
from vgoal.detector import BaseDetector, DetectionResult
from vgoal.geometry import CameraIntrinsics, project_3d_to_pixel
from vgoal.tracker import TargetState, TrackerConfig

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("vgoal_airsim_eval")


class GroundTruthVisualTargetDetector(BaseDetector):
    """Simulated camera detector that projects 3D target into 2D camera FOV."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        box_size_px: float = 20.0,
        conf: float = 0.95,
        class_name: str = "visual_target",
    ) -> None:
        self.intrinsics = intrinsics
        self.box_size_px = box_size_px
        self.conf = conf
        self.class_name = class_name
        self.active_goal_world: Optional[np.ndarray] = None
        self._current_drone_pos: Optional[np.ndarray] = None
        self._current_drone_yaw: float = 0.0

    def set_active_goal(self, goal_world: np.ndarray) -> None:
        self.active_goal_world = np.asarray(goal_world, dtype=np.float64).reshape(3)

    def update_drone_pose(self, pos: np.ndarray, yaw: float) -> None:
        self._current_drone_pos = np.asarray(pos, dtype=np.float64).reshape(3)
        self._current_drone_yaw = float(yaw)

    def detect(self, rgb: np.ndarray) -> Optional[DetectionResult]:
        if self.active_goal_world is None or self._current_drone_pos is None:
            return None

        h, w = rgb.shape[:2]
        if w != self.intrinsics.width or h != self.intrinsics.height:
            self.intrinsics = CameraIntrinsics.from_fov(80.0, width=w, height=h)

        d_world = self.active_goal_world - self._current_drone_pos
        c, s = np.cos(self._current_drone_yaw), np.sin(self._current_drone_yaw)
        d_fwd = float(c * d_world[0] + s * d_world[1])
        d_left = float(-s * d_world[0] + c * d_world[1])
        d_up = float(d_world[2])

        if d_fwd <= 0.5:
            return None

        u, v, z = project_3d_to_pixel([d_fwd, d_left, d_up], self.intrinsics)
        if math.isnan(u) or math.isnan(v):
            return None

        if not (0 <= u < w and 0 <= v < h):
            return None

        half_box = max(6.0, self.box_size_px * (10.0 / max(1.0, z)))
        u0, u1 = max(0.0, u - half_box), min(float(w - 1), u + half_box)
        v0, v1 = max(0.0, v - half_box), min(float(h - 1), v + half_box)

        res = DetectionResult(
            bbox=np.array([u0, v0, u1, v1], dtype=np.float32),
            confidence=self.conf,
            class_id=0,
            class_name=self.class_name,
        )
        setattr(res, "direct_depth", float(d_fwd))
        return res


class VisualGoalDeployPolicyWrapper:
    """Standard Policy interface wrapper for RolloutCollector."""

    def __init__(self, vgoal_policy: VisualGoalWAMPolicy, detector: GroundTruthVisualTargetDetector) -> None:
        self.vgoal_policy = vgoal_policy
        self.detector = detector

    def reset(self) -> None:
        self.vgoal_policy.reset()

    def bind_episode(self, episode: Optional[Dict[str, Any]]) -> None:
        if episode and "pos" in episode and len(episode["pos"]) > 1:
            goal_pos = np.asarray(episode["pos"][-1], dtype=np.float64).reshape(3)
            self.detector.set_active_goal(goal_pos)

    def act(self, policy_view: Any) -> np.ndarray:
        pos = getattr(policy_view, "position", None)
        if pos is None:
            proprio = getattr(policy_view, "proprio", None)
            if proprio is not None:
                pos = proprio[:3]
                yaw = float(proprio[6]) if len(proprio) > 6 else 0.0
            else:
                pos = np.zeros(3)
                yaw = 0.0
        else:
            yaw = float(getattr(policy_view, "yaw", 0.0))

        self.detector.update_drone_pose(pos, yaw)
        return self.vgoal_policy.act(policy_view)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Visual Object-Goal WAM in closed-loop")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/aerial_rl.yaml",
        help="Path to aerial_rl.yaml config",
    )
    parser.add_argument(
        "--wm-ckpt",
        type=str,
        default="experiments/aerial/rl/artifacts/wm_ckpt_d_full_20260828/wm_step_3500.pt",
        help="Torch RSSM checkpoint path",
    )
    parser.add_argument(
        "--actor-ckpt",
        type=str,
        default="experiments/aerial/rl/artifacts/v4_ac_ckpt_step_e_20260828/v4_ac_latest.pt",
        help="Actor-Critic checkpoint path",
    )
    parser.add_argument(
        "--depth-ckpt",
        type=str,
        default="experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt",
        help="Depth checkpoint path",
    )
    parser.add_argument(
        "--annotation",
        type=str,
        default="artifacts/seen_airsim16_m1a20.json",
        help="Route annotation file",
    )
    parser.add_argument("--episodes", type=int, default=8, help="Number of episodes to evaluate")
    parser.add_argument("--max-steps", type=int, default=250, help="Max steps per episode")
    parser.add_argument("--takeoff-scan-steps", type=int, default=8, help="Panoramic scan steps on takeoff")
    parser.add_argument("--planner", action="store_true", default=True, help="Enable ImaginationPlanner")
    parser.add_argument("--planner-horizon", type=int, default=5, help="Planner imagination horizon")
    parser.add_argument("--step-hz", type=float, default=5.0, help="Control loop rate")
    parser.add_argument("--fov-deg", type=float, default=80.0, help="Camera horizontal FOV")
    parser.add_argument("--success-dist", type=float, default=3.0, help="Success distance threshold (m)")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument(
        "--out-report",
        type=str,
        default="artifacts/visual_goal_eval_result.json",
        help="Output evaluation result JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("=====================================================================")
    logger.info("🎯 Visual Object-Goal Aerial WAM Official Collector Evaluation")
    logger.info(f"Episodes: {args.episodes}, Max Steps: {args.max_steps}, Planner: {args.planner} (H={args.planner_horizon})")
    logger.info("=====================================================================")

    root = Path(wam_v2_path)
    cfg_file = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = yaml.safe_load(cfg_file.read_text()) or {}

    from experiments.aerial.rl.actor_critic import LatentActorCritic, LatentActorDeployPolicy
    from experiments.aerial.rl.buffer import ReplayBuffer
    from experiments.aerial.rl.collector import RolloutCollector
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.planner import ImaginationPlanner
    from experiments.aerial.rl.reward import RewardConfig
    from experiments.aerial.rl.train_rl import _build_env, _build_safety, load_torch_dynamics

    # 1. Environment Setup
    cfg.setdefault("env", {})["backend"] = "airsim"
    cfg["env"]["step_hz"] = float(args.step_hz)
    cfg["env"]["grab_depth"] = True
    env = _build_env(cfg["env"])

    # 2. World Model & Policy Loading
    reward_cfg = RewardConfig(**(cfg.get("reward") or {}))
    reward_cfg.success_dist_m = float(args.success_dist)

    wm_cfg = cfg.get("world_model") or {}
    wm_path = (root / args.wm_ckpt).resolve() if not Path(args.wm_ckpt).is_absolute() else Path(args.wm_ckpt)
    dynamics, _ = load_torch_dynamics(wm_cfg, str(wm_path), device=str(args.device), success_dist_m=float(args.success_dist))

    actor_path = (root / args.actor_ckpt).resolve() if not Path(args.actor_ckpt).is_absolute() else Path(args.actor_ckpt)
    actor_ac = LatentActorCritic.load_from_checkpoint(actor_path, device=str(args.device))

    depth_path = (root / args.depth_ckpt).resolve() if not Path(args.depth_ckpt).is_absolute() else Path(args.depth_ckpt)
    depth_pred = DepthMinPredictor.from_checkpoint(depth_path, device=str(args.device)) if depth_path.is_file() else None

    shield = _build_safety(cfg.get("safety") or {})

    # 3. Setup Imagination Planner
    planner = None
    if args.planner:
        limits = np.array([1.0, 0.4, 0.4, math.pi / 10.0], dtype=np.float64)
        planner = ImaginationPlanner(
            dynamics=dynamics,
            horizon=int(args.planner_horizon),
            reward_cfg=reward_cfg,
            action_limits=limits,
            policy=actor_ac,
        )
        logger.info(f"ImaginationPlanner ACTIVE: horizon={args.planner_horizon}")

    # 4. Setup Camera Intrinsics & Visual Goal Detector
    intrinsics = CameraIntrinsics.from_fov(fov_deg=args.fov_deg, width=224, height=224)
    target_detector = GroundTruthVisualTargetDetector(intrinsics=intrinsics, class_name="visual_target")

    # 5. Build End-to-End Visual Goal Policy
    tracker_cfg = TrackerConfig(
        success_dist_m=float(args.success_dist),
        max_occlusion_s=120.0,
        ema_alpha=0.7,
    )
    policy_cfg = VisualGoalPolicyConfig(
        intrinsics=intrinsics,
        tracker_config=tracker_cfg,
        dt=1.0 / args.step_hz,
        use_planner=False,  # Collector handles planner seamlessly
    )
    vgoal_policy = VisualGoalWAMPolicy(
        dynamics=dynamics,
        actor_critic=actor_ac,
        detector=target_detector,
        depth_predictor=depth_pred,
        safety_shield=None,
        planner=None,
        config=policy_cfg,
    )
    policy_wrapped = VisualGoalDeployPolicyWrapper(vgoal_policy, target_detector)

    # 6. Build RolloutCollector
    buf = ReplayBuffer(capacity_episodes=4, seed=0)
    collector = RolloutCollector(
        env=env,
        policy=policy_wrapped,
        buffer=buf,
        reward_cfg=reward_cfg,
        safety=shield,
        max_steps=int(args.max_steps),
        target_hz=float(args.step_hz),
        skip_reset_collision=True,
        depth_predictor=depth_pred,
        planner=planner,
        dynamics=dynamics,
        takeoff_scan_steps=int(args.takeoff_scan_steps),
    )

    # 7. Load Route Annotations
    ann_path = (root / args.annotation).resolve() if not Path(args.annotation).is_absolute() else Path(args.annotation)
    with ann_path.open("r", encoding="utf-8") as f:
        routes: List[Dict[str, Any]] = json.load(f)

    n_eval = min(int(args.episodes), len(routes))
    routes_to_eval = routes[:n_eval]

    results: List[Dict[str, Any]] = []
    n_arrived = 0
    n_severe_coll = 0
    progress_ratios: List[float] = []

    logger.info(f"Starting Closed-Loop Evaluation on {n_eval} routes...")

    for idx, r in enumerate(routes_to_eval):
        pos_arr = np.asarray(r["pos"], dtype=np.float64).reshape(-1, 3)
        yaw_arr = np.asarray(r["yaw"], dtype=np.float64).reshape(-1)
        start_pos = pos_arr[0].copy()
        goal_pos = pos_arr[-1].copy()
        start_yaw = float(yaw_arr[0])

        ep_dict = {
            "pos": [start_pos.tolist(), goal_pos.tolist()],
            "yaw": [start_yaw, start_yaw],
            "gpt_instruction": r.get("gpt_instruction", ""),
        }

        ep_trans, stats = collector.collect_episode(ep_dict)
        if not ep_trans:
            logger.warning(f"Route {idx+1:02d} skipped (spawn collision).")
            continue

        d0 = float(np.linalg.norm(np.asarray(ep_trans[0].obs.position) - goal_pos))
        d_end = float(np.linalg.norm(np.asarray(ep_trans[-1].next_obs.position if ep_trans[-1].next_obs is not None else ep_trans[-1].obs.position) - goal_pos))
        d_min = min(float(np.linalg.norm(np.asarray(tr.obs.position) - goal_pos)) for tr in ep_trans)

        arrived = bool(d_min <= args.success_dist or d_end <= args.success_dist)
        prog = float(np.clip((d0 - d_end) / max(d0, 1e-4), 0.0, 1.0))

        collided = any(bool(getattr(tr.next_obs, "collided", False)) for tr in ep_trans)
        severe_coll = collided

        if arrived:
            n_arrived += 1
        if severe_coll:
            n_severe_coll += 1

        progress_ratios.append(prog)

        logger.info(
            f"Route {idx+1:02d}/{n_eval:02d} | steps={len(ep_trans):3d} | d0={d0:5.1f}m -> d_end={d_end:5.1f}m (min={d_min:4.1f}m) | "
            f"prog={prog*100:+5.1f}% | arrived={arrived} | severe_coll={severe_coll}"
        )

        results.append({
            "route_idx": idx,
            "steps": len(ep_trans),
            "d_start_m": d0,
            "d_end_m": d_end,
            "d_min_m": d_min,
            "arrived": arrived,
            "collided": collided,
            "severe_collision": severe_coll,
            "progress_ratio": prog,
        })

    env.close()

    summary = {
        "timestamp": time.time(),
        "n_episodes": len(routes_to_eval),
        "arrival_rate": float(n_arrived / max(1, len(routes_to_eval))),
        "severe_collision_rate": float(n_severe_coll / max(1, len(routes_to_eval))),
        "mean_progress_ratio": float(np.mean(progress_ratios)),
        "episodes": results,
    }

    out_p = Path(args.out_report)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=====================================================================")
    logger.info("🏆 Visual Object-Goal WAM Official Evaluation Summary:")
    logger.info(f"Arrival Rate: {summary['arrival_rate']*100:.1f}% ({n_arrived}/{len(routes_to_eval)})")
    logger.info(f"Mean Progress: {summary['mean_progress_ratio']*100:.1f}%")
    logger.info(f"Severe Collision Rate: {summary['severe_collision_rate']*100:.1f}%")
    logger.info(f"Report saved to: {out_p}")
    logger.info("=====================================================================")


if __name__ == "__main__":
    main()
