#!/usr/bin/env python3
"""Indoor object-goal eval: YOLO → standoff waypoint → fly-to.

Ann JSON supplies **spawn only** (start pose / facing). Success is vision-sourced:
detect object → approach point ``standoff_m`` in front of it → arrive within
``success_dist`` of that waypoint. Do **not** score against ann polyline ends.
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
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vgoal.bridge import VisualGoalPolicyConfig, VisualGoalWAMPolicy
from vgoal.detector import OpenVocabPromptDetector, YOLOTargetDetector
from vgoal.geometry import CameraIntrinsics
from vgoal.report_meta import semantic_nav_report_fields
from vgoal.tracker import TargetState, TrackerConfig

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
    """YOLO → standoff goal_rel policy wrapper for RolloutCollector."""

    def __init__(self, vgoal_policy: VisualGoalWAMPolicy) -> None:
        self.vgoal_policy = vgoal_policy
        # indoor act_delta: pass full Observation (incl. depth) for bbox→goal_rel
        self.use_full_obs = True
        self.policy_calls = 0
        self.detect_hits = 0
        self.last_det_name: Optional[str] = None
        self.last_det_wh: Optional[tuple] = None
        self.ever_arrived = False
        self.min_standoff_goal_dist = float("inf")
        self.min_object_dist = float("inf")
        self.last_object_dist: Optional[float] = None
        self.last_standoff_goal_dist: Optional[float] = None

    def reset(self) -> None:
        self.vgoal_policy.reset()
        self.policy_calls = 0
        self.detect_hits = 0
        self.last_det_name = None
        self.last_det_wh = None
        self.ever_arrived = False
        self.min_standoff_goal_dist = float("inf")
        self.min_object_dist = float("inf")
        self.last_object_dist = None
        self.last_standoff_goal_dist = None

    def bind_episode(self, episode: Optional[Dict[str, Any]]) -> None:
        return None

    def act(self, policy_view: Any) -> np.ndarray:
        action = self.vgoal_policy.act(policy_view)
        self.policy_calls += 1
        det = self.vgoal_policy.last_detection
        if det is not None:
            self.detect_hits += 1
            self.last_det_name = str(det.class_name)
            # Log YOLO branch resolution once (should be capture WH, not 224)
            yolo = getattr(policy_view, "rgb_yolo", None)
            if yolo is not None:
                arr = np.asarray(yolo)
                self.last_det_wh = (int(arr.shape[1]), int(arr.shape[0]))
        obj = self.vgoal_policy.last_object_goal_rel
        if obj is not None:
            od = float(obj[3]) if len(obj) > 3 else float(np.linalg.norm(obj[:3]))
            self.last_object_dist = od
            self.min_object_dist = min(self.min_object_dist, od)
        gr = self.vgoal_policy.last_goal_rel
        if gr is not None:
            gd = float(gr[3]) if len(gr) > 3 else float(np.linalg.norm(gr[:3]))
            self.last_standoff_goal_dist = gd
            self.min_standoff_goal_dist = min(self.min_standoff_goal_dist, gd)
        if self.vgoal_policy.last_target_state == TargetState.ARRIVED:
            self.ever_arrived = True
        return action


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Indoor YOLO→standoff waypoint vgoal eval (ann = spawn only)"
    )
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
        help="Spawn poses only (start/yaw); ann end is dual-report, not success",
    )
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--takeoff-scan-steps", type=int, default=8)
    p.add_argument(
        "--planner",
        action="store_true",
        default=False,
        help="Enable collector imagination planner (uses env goal; off for vision)",
    )
    p.add_argument("--planner-horizon", type=int, default=5)
    p.add_argument("--step-hz", type=float, default=5.0)
    p.add_argument("--fov-deg", type=float, default=80.0)
    p.add_argument(
        "--standoff-m",
        type=float,
        default=1.0,
        help="Approach waypoint distance in front of detected object (indoor default 1m)",
    )
    p.add_argument(
        "--success-dist",
        type=float,
        default=0.50,
        help="Arrive if within this of the standoff waypoint (not the object surface)",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--yolo-weights", default="yolov8n.pt")
    p.add_argument("--yolo-conf", type=float, default=0.4)
    p.add_argument("--yolo-imgsz", type=int, default=640)
    p.add_argument(
        "--target-classes",
        default="",
        help="Fixed COCO class filter when not using YOLO-World. "
        "Prefer --visual-prompt with --yolo-weights *world* for open-vocab (e.g. pillar).",
    )
    p.add_argument(
        "--visual-prompt",
        default="",
        help="Fixed open-vocab prompt (e.g. 'pillar'). Uses OpenVocabPromptDetector; "
        "with *world* weights this is the exact object class.",
    )
    p.add_argument(
        "--prefer-nearest",
        action="store_true",
        default=True,
        help="Among fixed-class hits, lock the nearest-by-depth instance (default on)",
    )
    p.add_argument(
        "--no-prefer-nearest",
        action="store_true",
        default=False,
        help="Disable nearest-instance selection",
    )
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
    # Single-camera fan-out: capture → rgb_vio / rgb(WAM 224) / rgb_yolo (native).
    # Capture WH must match AirSim CaptureSettings (not the WAM encode size).
    cfg["env"]["fanout_rgb"] = True
    cfg["env"]["width"] = int(os.environ.get("INDOOR_CAPTURE_W", "640"))
    cfg["env"]["height"] = int(os.environ.get("INDOOR_CAPTURE_H", "480"))
    cfg["env"]["wam_encode_size"] = int(os.environ.get("WAM_ENCODE_SIZE", "224"))
    env = _build_env(cfg["env"])

    reward_cfg = RewardConfig(**(cfg.get("reward") or {}))
    # Reward still needs an env goal for reset; keep tight vs far dummy so ann end ≠ PASS.
    reward_cfg.success_dist_m = 0.05

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

    intrinsics = CameraIntrinsics.from_fov(fov_deg=args.fov_deg, width=224, height=224)
    yolo_device = "0" if str(args.device).startswith("cuda") else "cpu"
    visual_prompt = str(args.visual_prompt).strip()
    class_filter = [c.strip() for c in str(args.target_classes).split(",") if c.strip()]
    if not visual_prompt and not class_filter:
        raise SystemExit(
            "Need a fixed object: pass --visual-prompt 'pillar' (YOLO-World) "
            "or --target-classes 'potted plant' (COCO). Unfiltered is invalid."
        )
    if visual_prompt:
        target_detector = OpenVocabPromptDetector(
            visual_prompt=visual_prompt,
            model_path=args.yolo_weights,
            conf_threshold=float(args.yolo_conf),
            imgsz=int(args.yolo_imgsz),
            device=yolo_device,
        )
        prompt_label = visual_prompt
    else:
        target_detector = YOLOTargetDetector(
            model_path=args.yolo_weights,
            target_classes=class_filter,
            conf_threshold=float(args.yolo_conf),
            imgsz=int(args.yolo_imgsz),
            device=yolo_device,
        )
        prompt_label = ",".join(class_filter)

    prefer_nearest = not bool(args.no_prefer_nearest)
    tracker_cfg = TrackerConfig(
        success_dist_m=float(args.success_dist),
        max_occlusion_s=120.0,
        ema_alpha=0.7,
        min_confidence=float(args.yolo_conf),
    )
    policy_cfg = VisualGoalPolicyConfig(
        intrinsics=intrinsics,
        tracker_config=tracker_cfg,
        dt=1.0 / float(args.step_hz),
        use_planner=False,
        approach_standoff_m=float(args.standoff_m),
        prefer_nearest_target=prefer_nearest,
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
        "Indoor vgoal object-standoff: n=%s standoff=%.2fm success=%.2fm yolo=%s conf=%.2f "
        "imgsz=%s prompt=%s prefer_nearest=%s terminal_dock=False",
        n_eval,
        args.standoff_m,
        args.success_dist,
        args.yolo_weights,
        args.yolo_conf,
        args.yolo_imgsz,
        prompt_label,
        prefer_nearest,
    )

    for idx, r in enumerate(routes_to_eval):
        pos_arr = np.asarray(r["pos"], dtype=np.float64).reshape(-1, 3)
        yaw_arr = np.asarray(r["yaw"], dtype=np.float64).reshape(-1)
        start_pos = pos_arr[0].copy()
        start_yaw = float(yaw_arr[0])
        # Far dummy env goal so NavigationReward cannot false-PASS on ann end.
        far = start_pos + np.array(
            [100.0 * math.cos(start_yaw), 100.0 * math.sin(start_yaw), 0.0],
            dtype=np.float64,
        )
        ann_end = pos_arr[-1].copy()
        ep_dict = {
            "pos": [start_pos.tolist(), far.tolist()],
            "yaw": [start_yaw, start_yaw],
            "gpt_instruction": r.get("gpt_instruction", ""),
            "trajectory_id": r.get("trajectory_id", ""),
        }
        policy_wrapped.reset()
        ep_trans, _stats = collector.collect_episode(ep_dict)
        if not ep_trans:
            logger.warning("Route %02d skipped (spawn collision).", idx + 1)
            continue

        vision_arrived = bool(
            policy_wrapped.ever_arrived
            or (
                policy_wrapped.min_standoff_goal_dist < float("inf")
                and policy_wrapped.min_standoff_goal_dist <= float(args.success_dist)
            )
        )
        d_ann0 = float(np.linalg.norm(np.asarray(ep_trans[0].obs.position) - ann_end))
        last = ep_trans[-1]
        end_obs = last.next_obs if last.next_obs is not None else last.obs
        d_ann_end = float(np.linalg.norm(np.asarray(end_obs.position) - ann_end))
        collided = any(bool(getattr(tr.next_obs, "collided", False)) for tr in ep_trans)
        if vision_arrived:
            n_arrived += 1
        if collided:
            n_severe_coll += 1
        gmin = policy_wrapped.min_standoff_goal_dist
        if gmin < float("inf"):
            prog = float(np.clip(1.0 - gmin / max(gmin + 1e-3, 3.0), 0.0, 1.0))
        else:
            prog = 0.0
        progress_ratios.append(prog)
        logger.info(
            "Route %02d/%02d id=%s steps=%d vision_arrived=%s coll=%s "
            "min_standoff_goal=%.2f min_obj=%.2f last_det=%s yolo_wh=%s policy_calls=%d detect_hits=%d "
            "(ann_dual d0=%.2f d_end=%.2f)",
            idx + 1,
            n_eval,
            r.get("trajectory_id", ""),
            len(ep_trans),
            vision_arrived,
            collided,
            gmin if gmin < float("inf") else -1.0,
            policy_wrapped.min_object_dist if policy_wrapped.min_object_dist < float("inf") else -1.0,
            policy_wrapped.last_det_name,
            policy_wrapped.last_det_wh,
            policy_wrapped.policy_calls,
            policy_wrapped.detect_hits,
            d_ann0,
            d_ann_end,
        )
        results.append(
            {
                "route_idx": idx,
                "trajectory_id": r.get("trajectory_id", ""),
                "steps": len(ep_trans),
                "vision_arrived": vision_arrived,
                "collided": collided,
                "min_standoff_goal_dist_m": (
                    None if gmin == float("inf") else float(gmin)
                ),
                "min_object_dist_m": (
                    None
                    if policy_wrapped.min_object_dist == float("inf")
                    else float(policy_wrapped.min_object_dist)
                ),
                "last_object_dist_m": policy_wrapped.last_object_dist,
                "policy_calls": int(policy_wrapped.policy_calls),
                "detect_hits": int(policy_wrapped.detect_hits),
            "last_det": policy_wrapped.last_det_name,
            "yolo_rgb_wh": list(policy_wrapped.last_det_wh) if policy_wrapped.last_det_wh else None,
            "ann_dual_d_start_m": d_ann0,
                "ann_dual_d_end_m": d_ann_end,
                "progress_ratio": prog,
            }
        )

    env.close()

    summary = {
        **semantic_nav_report_fields(
            depth_source="airsim_depth" if depth_pred is None else "depth_head",
            visual_prompt=prompt_label,
            phase="P0",
        ),
        "control_mode": "vision_standoff",
        "terminal_dock": False,
        "approach_standoff_m": float(args.standoff_m),
        "success_dist_m": float(args.success_dist),
        "success_metric": "standoff_waypoint",
        "prefer_nearest_target": prefer_nearest,
        "collector_planner": bool(planner),
        "timestamp": time.time(),
        "indoor_root": str(indoor),
        "annotation_spawn_only": str(ann_path),
        "yolo_weights": args.yolo_weights,
        "yolo_conf": float(args.yolo_conf),
        "yolo_imgsz": int(args.yolo_imgsz),
        "target_classes": list(class_filter) if class_filter else [prompt_label],
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
        "done vision_arrival=%.1f%% (%d/%d) mean_policy_calls=%.1f out=%s",
        100.0 * summary["arrival_rate"],
        n_arrived,
        len(results),
        summary["mean_policy_calls"],
        out_p,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
