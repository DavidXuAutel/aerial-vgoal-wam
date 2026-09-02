#!/usr/bin/env python3
"""Indoor wiring of original vgoal AirSim eval.

Same stack as ``examples/eval_visual_goal_airsim.py``:
  VisualGoalWAMPolicy + YOLOTargetDetector + RolloutCollector

Only change vs outdoor eval: ``AERIAL_INDOOR_ROOT`` supplies env / ckpts / Building99
annotation, and the detector is real ``YOLOTargetDetector`` (original defaults)
instead of the outdoor GT projector stub.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vgoal.bridge import VisualGoalPolicyConfig, VisualGoalWAMPolicy
from vgoal.detector import YOLOTargetDetector
from vgoal.geometry import CameraIntrinsics
from vgoal.report_meta import semantic_nav_report_fields
from vgoal.tracker import TrackerConfig

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("eval_indoor_vgoal")


def _indoor_root(override: str = "") -> Path:
    if override.strip():
        return Path(override).expanduser()
    env = os.environ.get("AERIAL_INDOOR_ROOT", "").strip()
    if env:
        return Path(env).expanduser()
    for cand in (
        Path("/home/yao/aerial-indoor-wam"),
        Path.home() / "Projects" / "aerial-indoor-wam",
        Path.home() / "aerial-indoor-wam",
    ):
        if cand.is_dir():
            return cand
    return Path("/home/yao/aerial-indoor-wam")


class VisualGoalDeployPolicyWrapper:
    """Same wrapper as ``eval_visual_goal_airsim.VisualGoalDeployPolicyWrapper`` (YOLO path)."""

    def __init__(self, vgoal_policy: VisualGoalWAMPolicy) -> None:
        self.vgoal_policy = vgoal_policy
        self.policy_calls = 0
        self.detect_hits = 0
        self.last_det_name: Optional[str] = None

    def reset(self) -> None:
        self.vgoal_policy.reset()
        self.policy_calls = 0
        self.detect_hits = 0
        self.last_det_name = None

    def bind_episode(self, episode: Optional[Dict[str, Any]]) -> None:
        return None

    def act(self, policy_view: Any) -> np.ndarray:
        action = self.vgoal_policy.act(policy_view)
        self.policy_calls += 1
        det = self.vgoal_policy.last_detection
        if det is not None:
            self.detect_hits += 1
            self.last_det_name = str(det.class_name)
        return action


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Original vgoal eval wired to aerial-indoor-wam")
    p.add_argument("--indoor-root", default="", help="Override AERIAL_INDOOR_ROOT")
    p.add_argument("--config", default="configs/aerial_rl_indoor_shield_v3.yaml")
    p.add_argument(
        "--wm-ckpt",
        default="experiments/aerial/rl/artifacts/wm_ckpt_indoor_encode_e2i_b_20260901/wm_step_400.pt",
    )
    p.add_argument(
        "--actor-ckpt",
        default="experiments/aerial/rl/artifacts/v4_ac_ckpt_indoor_e2i_e_20260901/v4_ac_latest.pt",
    )
    p.add_argument(
        "--depth-ckpt",
        default="",
        help="Optional depth head ckpt; empty = use AirSim depth via obs.depth",
    )
    p.add_argument(
        "--annotation",
        default="building99_indoor_short_routes_clean_sg.json",
        help="Building99 route ann (clean_sg = west/south/east, n=3)",
    )
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--takeoff-scan-steps", type=int, default=8)
    p.add_argument(
        "--planner",
        action="store_true",
        default=False,
        help="Enable collector imagination planner (uses ann GT goal; off for true vision)",
    )
    p.add_argument("--planner-horizon", type=int, default=5)
    p.add_argument("--step-hz", type=float, default=5.0)
    p.add_argument("--fov-deg", type=float, default=80.0)
    p.add_argument("--success-dist", type=float, default=0.50)
    p.add_argument("--device", type=str, default="cuda")
    # Original YOLOTargetDetector defaults
    p.add_argument("--yolo-weights", default="yolov8n.pt")
    p.add_argument("--yolo-conf", type=float, default=0.4)
    p.add_argument("--yolo-imgsz", type=int, default=640)
    p.add_argument(
        "--out-report",
        default="artifacts/indoor_vgoal_eval_result.json",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    indoor = _indoor_root(args.indoor_root)
    if not indoor.is_dir():
        raise FileNotFoundError(f"indoor root missing: {indoor}")
    if str(indoor) not in sys.path:
        sys.path.insert(0, str(indoor))
    os.chdir(indoor)

    from experiments.aerial.rl.actor_critic import LatentActorCritic
    from experiments.aerial.rl.buffer import ReplayBuffer
    from experiments.aerial.rl.collector import RolloutCollector
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.planner import ImaginationPlanner
    from experiments.aerial.rl.reward import RewardConfig
    from experiments.aerial.rl.train_rl import _build_env, _build_safety, load_torch_dynamics

    cfg_file = Path(args.config)
    if not cfg_file.is_absolute():
        cfg_file = indoor / cfg_file
    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    cfg.setdefault("env", {})["backend"] = "airsim"
    cfg["env"]["step_hz"] = float(args.step_hz)
    cfg["env"]["grab_depth"] = True
    env = _build_env(cfg["env"])

    reward_cfg = RewardConfig(**(cfg.get("reward") or {}))
    reward_cfg.success_dist_m = float(args.success_dist)

    wm_path = Path(args.wm_ckpt)
    if not wm_path.is_absolute():
        wm_path = indoor / wm_path
    dynamics, _ = load_torch_dynamics(
        cfg.get("world_model") or {},
        str(wm_path),
        device=str(args.device),
        success_dist_m=float(args.success_dist),
    )
    if not hasattr(dynamics, "image_size"):
        object.__setattr__(dynamics, "image_size", 224)

    actor_path = Path(args.actor_ckpt)
    if not actor_path.is_absolute():
        actor_path = indoor / actor_path
    actor_ac = LatentActorCritic.load_from_checkpoint(actor_path, device=str(args.device))

    depth_pred = None
    if args.depth_ckpt:
        depth_path = Path(args.depth_ckpt)
        if not depth_path.is_absolute():
            depth_path = indoor / depth_path
        if depth_path.is_file():
            depth_pred = DepthMinPredictor.from_checkpoint(depth_path, device=str(args.device))

    shield = _build_safety(cfg.get("safety") or {})

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

    # Original vgoal camera + YOLO (defaults: conf=0.4, imgsz=640)
    intrinsics = CameraIntrinsics.from_fov(fov_deg=args.fov_deg, width=224, height=224)
    yolo_device = "0" if str(args.device).startswith("cuda") else "cpu"
    target_detector = YOLOTargetDetector(
        model_path=args.yolo_weights,
        conf_threshold=float(args.yolo_conf),
        imgsz=int(args.yolo_imgsz),
        device=yolo_device,
    )

    tracker_cfg = TrackerConfig(
        success_dist_m=float(args.success_dist),
        max_occlusion_s=120.0,
        ema_alpha=0.7,
    )
    policy_cfg = VisualGoalPolicyConfig(
        intrinsics=intrinsics,
        tracker_config=tracker_cfg,
        dt=1.0 / float(args.step_hz),
        use_planner=False,
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
    policy_wrapped = VisualGoalDeployPolicyWrapper(vgoal_policy)

    buf = ReplayBuffer(capacity_episodes=4, seed=0)
    collector_kwargs: Dict[str, Any] = dict(
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
        terminal_dock=False,
    )
    try:
        collector = RolloutCollector(**collector_kwargs)
    except TypeError as e:
        raise SystemExit(
            "indoor RolloutCollector lacks terminal_dock=; sync aerial-indoor-wam "
            f"collector.py before true vision eval ({e})"
        ) from e

    ann_path = Path(args.annotation)
    if not ann_path.is_absolute():
        ann_path = indoor / ann_path
        if not ann_path.is_file():
            ann_path = indoor / Path(args.annotation).name
    routes: List[Dict[str, Any]] = json.loads(ann_path.read_text(encoding="utf-8"))
    n_eval = min(int(args.episodes), len(routes))
    routes_to_eval = routes[:n_eval]

    results: List[Dict[str, Any]] = []
    n_arrived = 0
    n_severe_coll = 0
    progress_ratios: List[float] = []

    logger.info(
        "Indoor vgoal TRUE vision: n=%s success_dist=%.2f yolo=%s conf=%.2f imgsz=%s "
        "terminal_dock=False planner=%s",
        n_eval,
        args.success_dist,
        args.yolo_weights,
        args.yolo_conf,
        args.yolo_imgsz,
        bool(planner),
    )

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
            "trajectory_id": r.get("trajectory_id", ""),
        }
        policy_wrapped.reset()
        ep_trans, _stats = collector.collect_episode(ep_dict)
        if not ep_trans:
            logger.warning("Route %02d skipped (spawn collision).", idx + 1)
            continue

        d0 = float(np.linalg.norm(np.asarray(ep_trans[0].obs.position) - goal_pos))
        last = ep_trans[-1]
        end_obs = last.next_obs if last.next_obs is not None else last.obs
        d_end = float(np.linalg.norm(np.asarray(end_obs.position) - goal_pos))
        d_min = min(float(np.linalg.norm(np.asarray(tr.obs.position) - goal_pos)) for tr in ep_trans)
        arrived = bool(d_min <= args.success_dist or d_end <= args.success_dist)
        prog = float(np.clip((d0 - d_end) / max(d0, 1e-4), 0.0, 1.0))
        collided = any(bool(getattr(tr.next_obs, "collided", False)) for tr in ep_trans)
        if arrived:
            n_arrived += 1
        if collided:
            n_severe_coll += 1
        progress_ratios.append(prog)
        logger.info(
            "Route %02d/%02d id=%s steps=%d d0=%.2f d_end=%.2f arrived=%s coll=%s "
            "policy_calls=%d detect_hits=%d last_det=%s",
            idx + 1,
            n_eval,
            r.get("trajectory_id", ""),
            len(ep_trans),
            d0,
            d_end,
            arrived,
            collided,
            policy_wrapped.policy_calls,
            policy_wrapped.detect_hits,
            policy_wrapped.last_det_name,
        )
        results.append(
            {
                "route_idx": idx,
                "trajectory_id": r.get("trajectory_id", ""),
                "steps": len(ep_trans),
                "d_start_m": d0,
                "d_end_m": d_end,
                "d_min_m": d_min,
                "arrived": arrived,
                "collided": collided,
                "progress_ratio": prog,
                "policy_calls": int(policy_wrapped.policy_calls),
                "detect_hits": int(policy_wrapped.detect_hits),
                "last_det": policy_wrapped.last_det_name,
            }
        )

    env.close()

    summary = {
        **semantic_nav_report_fields(
            depth_source="airsim_depth" if depth_pred is None else "depth_head",
            visual_prompt="yolo_coco_unfiltered",
            phase="P0",
        ),
        "control_mode": "vision_policy",
        "terminal_dock": False,
        "collector_planner": bool(planner),
        "timestamp": time.time(),
        "indoor_root": str(indoor),
        "annotation": str(ann_path),
        "yolo_weights": args.yolo_weights,
        "yolo_conf": float(args.yolo_conf),
        "yolo_imgsz": int(args.yolo_imgsz),
        "success_dist_m": float(args.success_dist),
        "n_episodes": len(results),
        "arrival_rate": float(n_arrived / max(1, len(results))),
        "severe_collision_rate": float(n_severe_coll / max(1, len(results))),
        "mean_progress_ratio": float(np.mean(progress_ratios)) if progress_ratios else 0.0,
        "mean_policy_calls": float(np.mean([e["policy_calls"] for e in results])) if results else 0.0,
        "episodes": results,
    }

    out_p = Path(args.out_report)
    if not out_p.is_absolute():
        out_p = ROOT / out_p
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "done arrival=%.1f%% (%d/%d) mean_policy_calls=%.1f out=%s",
        100.0 * summary["arrival_rate"],
        n_arrived,
        len(results),
        summary["mean_policy_calls"],
        out_p,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
