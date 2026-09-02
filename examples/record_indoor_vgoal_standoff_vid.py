#!/usr/bin/env python3
"""Record ego mp4 for indoor vgoal standoff runs (debug / collision audit)."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vgoal.bridge import VisualGoalPolicyConfig, VisualGoalWAMPolicy
from vgoal.detector import YOLOTargetDetector
from vgoal.geometry import CameraIntrinsics
from vgoal.tracker import TrackerConfig

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("indoor_vgoal_vid")


def _write_mp4(frames: List[np.ndarray], path: Path, fps: float = 5.0) -> None:
    if not frames:
        raise ValueError("no frames")
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", str(path),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert p.stdin is not None
    for fr in frames:
        if fr.shape[:2] != (h, w):
            fr = cv2.resize(fr, (w, h))
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed {path}")


def _hud(rgb: np.ndarray, *, step: int, det: str, obj_d: float, stand_d: float, coll: bool) -> np.ndarray:
    bgr = np.asarray(rgb[..., ::-1], dtype=np.uint8).copy()
    if max(bgr.shape[:2]) < 400:
        bgr = cv2.resize(bgr, (bgr.shape[1] * 3, bgr.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
    lines = [
        f"vgoal step={step:03d} det={det} coll={'Y' if coll else 'n'}",
        f"obj_d={obj_d:.2f} standoff_goal_d={stand_d:.2f}",
    ]
    for i, t in enumerate(lines):
        cv2.putText(bgr, t, (10, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
    return bgr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indoor-root", default=os.environ.get("AERIAL_INDOOR_ROOT", "/home/yao/aerial-indoor-wam"))
    ap.add_argument("--annotation", default="building99_indoor_short_routes_clean_sg.json")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--standoff-m", type=float, default=1.0)
    ap.add_argument("--success-dist", type=float, default=0.50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="artifacts/videos/indoor_vgoal_standoff_20260902")
    args = ap.parse_args()

    indoor = Path(args.indoor_root)
    sys.path.insert(0, str(indoor))
    os.chdir(indoor)

    from experiments.aerial.rl.actor_critic import LatentActorCritic
    from experiments.aerial.rl.buffer import ReplayBuffer
    from experiments.aerial.rl.collector import RolloutCollector
    from experiments.aerial.rl.reward import RewardConfig
    from experiments.aerial.rl.train_rl import _build_env, _build_safety, load_torch_dynamics

    # Import wrapper from eval script
    sys.path.insert(0, str(ROOT / "examples"))
    from eval_indoor_semantic_p0 import VisualGoalDeployPolicyWrapper  # type: ignore

    cfg = yaml.safe_load((indoor / "configs/aerial_rl_indoor_shield_v3.yaml").read_text()) or {}
    cfg.setdefault("env", {})["backend"] = "airsim"
    cfg["env"]["step_hz"] = 5.0
    cfg["env"]["grab_depth"] = True
    env = _build_env(cfg["env"])
    reward_cfg = RewardConfig(**(cfg.get("reward") or {}))
    reward_cfg.success_dist_m = 0.05
    dynamics, _ = load_torch_dynamics(
        cfg.get("world_model") or {},
        str(indoor / "experiments/aerial/rl/artifacts/wm_ckpt_indoor_encode_e2i_b_20260901/wm_step_400.pt"),
        device=str(args.device),
        success_dist_m=float(args.success_dist),
    )
    if not hasattr(dynamics, "image_size"):
        object.__setattr__(dynamics, "image_size", 224)
    actor = LatentActorCritic.load_from_checkpoint(
        indoor / "experiments/aerial/rl/artifacts/v4_ac_ckpt_indoor_e2i_e_20260901/v4_ac_latest.pt",
        device=str(args.device),
    )
    shield = _build_safety(cfg.get("safety") or {})
    det = YOLOTargetDetector(model_path="yolov8n.pt", conf_threshold=0.4, imgsz=640, device="0")
    pol = VisualGoalWAMPolicy(
        dynamics=dynamics,
        actor_critic=actor,
        detector=det,
        depth_predictor=None,
        safety_shield=None,
        planner=None,
        config=VisualGoalPolicyConfig(
            intrinsics=CameraIntrinsics.from_fov(80.0, 224, 224),
            tracker_config=TrackerConfig(
                success_dist_m=float(args.success_dist),
                max_occlusion_s=120.0,
                ema_alpha=0.7,
                min_confidence=0.4,
            ),
            dt=0.2,
            use_planner=False,
            approach_standoff_m=float(args.standoff_m),
        ),
    )

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    routes = json.loads((indoor / args.annotation).read_text())
    routes = routes[: int(args.episodes)]
    reports: List[Dict[str, Any]] = []

    for idx, r in enumerate(routes):
        frames: List[np.ndarray] = []
        start = np.asarray(r["pos"][0], dtype=np.float64)
        yaw0 = float(r["yaw"][0])
        far = start + np.array([100.0 * math.cos(yaw0), 100.0 * math.sin(yaw0), 0.0])
        ep = {
            "pos": [start.tolist(), far.tolist()],
            "yaw": [yaw0, yaw0],
            "gpt_instruction": r.get("gpt_instruction", ""),
            "trajectory_id": r.get("trajectory_id", f"r{idx}"),
        }

        class RecWrap(VisualGoalDeployPolicyWrapper):
            def act(self, policy_view: Any) -> np.ndarray:
                # Prefer full obs path: collector may pass Observation when use_full_obs
                a = super().act(policy_view)
                rgb = getattr(policy_view, "rgb", None)
                if rgb is not None:
                    obj = self.vgoal_policy.last_object_goal_rel
                    gr = self.vgoal_policy.last_goal_rel
                    od = float(obj[3]) if obj is not None and len(obj) > 3 else -1.0
                    sd = float(gr[3]) if gr is not None and len(gr) > 3 else -1.0
                    coll = bool(getattr(policy_view, "collided", False))
                    frames.append(
                        _hud(
                            np.asarray(rgb),
                            step=self.policy_calls,
                            det=str(self.last_det_name or "-"),
                            obj_d=od,
                            stand_d=sd,
                            coll=coll,
                        )
                    )
                return a

        wrap = RecWrap(pol)
        wrap.reset()
        buf = ReplayBuffer(capacity_episodes=2, seed=0)
        col = RolloutCollector(
            env=env,
            policy=wrap,
            buffer=buf,
            reward_cfg=reward_cfg,
            safety=shield,
            max_steps=int(args.max_steps),
            target_hz=5.0,
            skip_reset_collision=True,
            depth_predictor=None,
            planner=None,
            dynamics=dynamics,
            takeoff_scan_steps=8,
            terminal_dock=False,
        )
        ep_trans, _ = col.collect_episode(ep)
        collided = any(bool(getattr(tr.next_obs, "collided", False)) for tr in ep_trans) if ep_trans else False
        # also stamp last frame collide from transitions
        for ti, tr in enumerate(ep_trans):
            if tr.next_obs is not None and bool(getattr(tr.next_obs, "collided", False)):
                # annotate a red banner frame if we have matching policy frames
                if ti < len(frames):
                    cv2.putText(
                        frames[ti],
                        "COLLISION",
                        (10, frames[ti].shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (40, 40, 255),
                        2,
                        cv2.LINE_AA,
                    )

        tid = r.get("trajectory_id", f"r{idx}")
        mp4 = out_dir / f"{tid}_ego.mp4"
        if frames:
            _write_mp4(frames, mp4, fps=5.0)
        rep = {
            "trajectory_id": tid,
            "steps": len(ep_trans),
            "collided": collided,
            "policy_calls": wrap.policy_calls,
            "detect_hits": wrap.detect_hits,
            "last_det": wrap.last_det_name,
            "min_standoff_goal_dist_m": (
                None if wrap.min_standoff_goal_dist == float("inf") else wrap.min_standoff_goal_dist
            ),
            "min_object_dist_m": (
                None if wrap.min_object_dist == float("inf") else wrap.min_object_dist
            ),
            "vision_arrived": bool(
                wrap.ever_arrived
                or (
                    wrap.min_standoff_goal_dist < float("inf")
                    and wrap.min_standoff_goal_dist <= float(args.success_dist)
                )
            ),
            "ego_mp4": str(mp4) if frames else None,
            "n_frames": len(frames),
        }
        reports.append(rep)
        logger.info(
            "%s steps=%d coll=%s arr=%s frames=%d -> %s",
            tid,
            rep["steps"],
            collided,
            rep["vision_arrived"],
            len(frames),
            mp4.name if frames else "none",
        )

    env.close()
    summary = out_dir / "summary.json"
    summary.write_text(json.dumps(reports, indent=2) + "\n")
    logger.info("wrote %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
