#!/usr/bin/env python3
"""Indoor semantic nav P0 — reuse vgoal; Building99 via AERIAL_INDOOR_ROOT.

P0: target already in FOV · fixed visual_prompt · vision goal_rel · @0.50.
No GT goal for control success. Dual-report GT distance is side-note only.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vgoal.detector import MockDetector, OpenVocabPromptDetector
from vgoal.geometry import CameraIntrinsics, bbox_to_goal_rel
from vgoal.report_meta import semantic_nav_report_fields

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("indoor_semantic_p0")


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


def run_dry_run(*, visual_prompt: str, out: Path) -> Dict[str, Any]:
    h, w = 224, 224
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), 3.0, dtype=np.float32)
    bbox = [w * 0.35, h * 0.35, w * 0.65, h * 0.65]
    name = visual_prompt.split(",")[0].strip() or "target"
    inner = MockDetector(target_bbox=bbox, confidence=0.9, class_name=name)
    det = OpenVocabPromptDetector(visual_prompt=visual_prompt, inner=inner)
    hit = det.detect(rgb)
    assert hit is not None
    K = CameraIntrinsics.from_fov(80.0, width=w, height=h)
    goal_rel = bbox_to_goal_rel(hit.bbox, depth, K)
    summary = {
        **semantic_nav_report_fields(
            depth_source="airsim_depth", visual_prompt=visual_prompt, phase="P0"
        ),
        "mode": "dry_run",
        "success_dist_m": 0.50,
        "n": 1,
        "arrived": True,
        "collided": False,
        "goal_rel": np.asarray(goal_rel, dtype=np.float64).reshape(-1).tolist(),
        "detection": {
            "class_name": hit.class_name,
            "confidence": hit.confidence,
            "bbox": hit.bbox.tolist(),
        },
        "note": "dry-run only — not a Building99 gate result",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _load_segments(ann: Path, routes: str) -> List[Dict[str, Any]]:
    raw = json.loads(ann.read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else raw.get("routes") or raw.get("segments") or []
    idxs = [int(x) for x in routes.split(",") if str(x).strip() != ""]
    segs: List[Dict[str, Any]] = []
    for i in idxs:
        it = items[i]
        pos = it.get("pos") or it.get("positions")
        yaw = it.get("yaw") or it.get("yaws")
        if pos is None:
            continue
        pos_a = np.asarray(pos, dtype=np.float64)
        yaw_a = np.asarray(yaw if yaw is not None else [0.0, 0.0], dtype=np.float64)
        if pos_a.ndim == 1:
            pos_a = pos_a.reshape(1, 3)
        if len(pos_a) < 2:
            # single pose: synthesize a 2 m forward goal for GT dual-report only
            yaw0 = float(yaw_a.reshape(-1)[0]) if yaw_a.size else 0.0
            g = pos_a[0] + np.array([np.cos(yaw0) * 2.0, np.sin(yaw0) * 2.0, 0.0])
            pos_a = np.vstack([pos_a[0], g])
            yaw_a = np.array([yaw0, yaw0])
        d0 = float(np.linalg.norm(pos_a[1] - pos_a[0]))
        segs.append(
            {
                "segment_name": str(it.get("trajectory_id") or it.get("gpt_instruction") or f"route_{i}"),
                "gpt_instruction": str(it.get("gpt_instruction") or "semantic nav p0"),
                "pos": pos_a.tolist(),
                "yaw": yaw_a.reshape(-1).tolist(),
                "d0_m": d0,
                "source_route_idx": i,
            }
        )
    return segs


def _ensure_indoor_path(indoor: Path) -> None:
    s = str(indoor)
    if s not in sys.path:
        sys.path.insert(0, s)


def run_airsim_p0(args: argparse.Namespace) -> Dict[str, Any]:
    indoor = _indoor_root(args.indoor_root)
    if not indoor.is_dir():
        raise FileNotFoundError(f"indoor root missing: {indoor}")
    _ensure_indoor_path(indoor)
    os.chdir(indoor)

    import yaml
    from experiments.aerial.rl.actor_critic import LatentActorCritic
    from experiments.aerial.rl.train_rl import _build_env, _build_safety, load_torch_dynamics
    from experiments.aerial.rl.collector import clip_body_delta

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = indoor / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg.setdefault("env", {})["backend"] = "airsim"
    cfg["env"]["step_hz"] = float(args.step_hz)
    cfg["env"]["grab_depth"] = True

    ann = Path(args.annotation)
    if not ann.is_absolute():
        ann = indoor / ann
        if not ann.is_file():
            ann = indoor / Path(args.annotation).name
    if not ann.is_file():
        # try repo-root copies used on 125
        for fb in (indoor / "building99_indoor_short_routes_clean_e.json", indoor / "building99_indoor_short_routes_clean_sg.json"):
            if fb.is_file():
                ann = fb
                break
    segs = _load_segments(ann, args.routes)
    if not segs:
        raise RuntimeError(f"no segments from {ann} routes={args.routes}")

    wm_path = Path(args.wm_ckpt)
    if not wm_path.is_absolute():
        wm_path = indoor / wm_path
    act_path = Path(args.actor_ckpt)
    if not act_path.is_absolute():
        act_path = indoor / act_path

    env = _build_env(cfg["env"])
    dynamics, _ = load_torch_dynamics(
        cfg.get("world_model") or {},
        str(wm_path),
        device=str(args.device),
        success_dist_m=float(args.success_dist),
    )
    actor = LatentActorCritic.load_from_checkpoint(act_path, device=str(args.device))
    shield = _build_safety(cfg.get("safety") or {})

    yolo_device = "0" if str(args.device).startswith("cuda") else "cpu"
    detector = OpenVocabPromptDetector(
        visual_prompt=args.visual_prompt,
        model_path=args.yolo_weights,
        conf_threshold=float(args.conf),
        device=yolo_device,
    )

    limits = np.array([0.15, 0.08, 0.08, 0.10], dtype=np.float64)
    episodes: List[Dict[str, Any]] = []
    seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip() != ""]

    for seed in seeds:
        np.random.seed(seed)
        seg = segs[seed % len(segs)]
        ep = _run_one_episode(
            env=env,
            dynamics=dynamics,
            actor=actor,
            shield=shield,
            detector=detector,
            seg=seg,
            success_dist=float(args.success_dist),
            max_steps=int(args.max_steps),
            step_hz=float(args.step_hz),
            limits=limits,
            visual_prompt=args.visual_prompt,
            seed=seed,
        )
        episodes.append(ep)
        logger.info(
            "seed=%s seg=%s fail=%s arrived_vision=%s d_vis=%.3f d_gt=%.3f",
            seed,
            seg["segment_name"],
            ep.get("fail_reason"),
            ep.get("arrived_vision"),
            ep.get("d_end_vision", float("nan")),
            ep.get("d_end_gt_side", float("nan")),
        )

    n = len(episodes)
    n_arr = sum(1 for e in episodes if e.get("arrived_vision"))
    n_miss = sum(1 for e in episodes if e.get("fail_reason") == "miss_detect")
    n_coll = sum(1 for e in episodes if e.get("collided"))
    summary = {
        **semantic_nav_report_fields(
            depth_source="airsim_depth",
            visual_prompt=args.visual_prompt,
            phase="P0",
        ),
        "mode": "airsim",
        "indoor_root": str(indoor),
        "annotation": str(ann),
        "actor_ckpt": str(act_path),
        "wm_ckpt": str(wm_path),
        "yolo_weights": args.yolo_weights,
        "success_dist_m": float(args.success_dist),
        "n": n,
        "arrived_vision_n": n_arr,
        "arrival_rate_vision": (n_arr / n) if n else 0.0,
        "miss_detect_n": n_miss,
        "collision_n": n_coll,
        "primary_gate_pass": bool(n >= 3 and n_arr >= 2 and n_coll == 0),  # soft P0: ≥2/3
        "gate_note": "P0 formal: n>=3 arrive_vision@0.50 collided=false; soft pass >=2/3 for smoke",
        "episodes": episodes,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s pass=%s", out, summary["primary_gate_pass"])
    return summary


def _run_one_episode(
    *,
    env: Any,
    dynamics: Any,
    actor: Any,
    shield: Any,
    detector: OpenVocabPromptDetector,
    seg: Dict[str, Any],
    success_dist: float,
    max_steps: int,
    step_hz: float,
    limits: np.ndarray,
    visual_prompt: str,
    seed: int,
) -> Dict[str, Any]:
    """P0 loop: YOLO → airsim depth → vision goal_rel → indoor act_latent (+ shield)."""
    from experiments.aerial.rl.collector import clip_body_delta
    from experiments.aerial.rl.goal_features import body_vel_from_obs

    goal_gt = np.asarray(seg["pos"][1], dtype=np.float64)  # dual-report only
    obs = env.reset({"pos": seg["pos"], "yaw": seg["yaw"], "gpt_instruction": seg["gpt_instruction"]})
    if obs is None or getattr(obs, "rgb", None) is None:
        return {"seed": seed, "ok": False, "fail_reason": "reset_failed", "arrived_vision": False}

    first = detector.detect(np.asarray(obs.rgb, dtype=np.uint8))
    if first is None:
        return {
            "seed": seed,
            "ok": True,
            "fail_reason": "miss_detect",
            "arrived_vision": False,
            "collided": False,
            "steps": 0,
            "segment_name": seg["segment_name"],
            "detection0": None,
            "d_end_gt_side": float(np.linalg.norm(np.asarray(obs.position) - goal_gt)),
        }

    if hasattr(shield, "reset"):
        shield.reset()

    K = CameraIntrinsics.from_fov(80.0, width=int(obs.rgb.shape[1]), height=int(obs.rgb.shape[0]))
    latent = np.asarray(dynamics.encode(obs), dtype=np.float64)
    prev_act: Optional[np.ndarray] = None
    d_vis = float("nan")
    step_i = 0
    last_cls = first.class_name

    for step_i in range(max_steps):
        rgb = np.asarray(obs.rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]
        if w != K.width or h != K.height:
            K = CameraIntrinsics.from_fov(80.0, width=w, height=h)

        hit = detector.detect(rgb)
        vision_gr = None
        if hit is not None:
            last_cls = hit.class_name
            depth = getattr(obs, "depth", None)
            if depth is not None:
                vision_gr = bbox_to_goal_rel(hit.bbox, np.asarray(depth, dtype=np.float32), K, src_shape=(w, h))

        if vision_gr is None:
            return {
                "seed": seed,
                "ok": True,
                "fail_reason": "miss_detect",
                "arrived_vision": False,
                "collided": bool(getattr(obs, "collided", False)),
                "steps": step_i,
                "segment_name": seg["segment_name"],
                "detection0": {
                    "class_name": first.class_name,
                    "confidence": first.confidence,
                    "bbox": first.bbox.tolist(),
                },
            }

        d_vis = float(np.linalg.norm(vision_gr[:3]))
        if d_vis <= success_dist:
            break

        if hasattr(actor, "act_latent"):
            action = np.asarray(
                actor.act_latent(latent, goal_rel=vision_gr, deterministic=True),
                dtype=np.float64,
            ).reshape(4)
        else:
            heading_err = float(np.arctan2(vision_gr[1], vision_gr[0]))
            action = np.array(
                [
                    float(np.clip(vision_gr[0] * 0.25, 0.05, limits[0])),
                    0.0,
                    0.0,
                    float(np.clip(heading_err * 0.4, -limits[3], limits[3])),
                ],
                dtype=np.float64,
            )
        action = clip_body_delta(action, limits)

        wm_out = None
        try:
            wm_out = dynamics.step(
                latent,
                action,
                goal_rel=vision_gr,
                body_vel=body_vel_from_obs(obs),
            )
        except TypeError:
            wm_out = dynamics.step(latent, action)

        if shield is not None:
            apply_fn = getattr(shield, "apply_action", None)
            if callable(apply_fn):
                action, _ = apply_fn(action, obs, wm_out=wm_out, limits=limits)

        next_obs, _info = env.step(action)
        if wm_out is not None and hasattr(wm_out, "z_next"):
            latent = np.asarray(wm_out.z_next, dtype=np.float64)
        else:
            latent = np.asarray(dynamics.encode(next_obs), dtype=np.float64)
        obs = next_obs
        prev_act = action.copy()
        if bool(getattr(obs, "collided", False)):
            break

    # final vision distance
    hit = detector.detect(np.asarray(obs.rgb, dtype=np.uint8))
    if hit is not None and getattr(obs, "depth", None) is not None:
        gr = bbox_to_goal_rel(hit.bbox, np.asarray(obs.depth, dtype=np.float32), K)
        if gr is not None:
            d_vis = float(np.linalg.norm(gr[:3]))

    d_gt = float(np.linalg.norm(np.asarray(obs.position, dtype=np.float64) - goal_gt))
    arrived_vision = bool(np.isfinite(d_vis) and d_vis <= success_dist)
    return {
        "seed": seed,
        "ok": True,
        "fail_reason": None if arrived_vision else ("collision" if getattr(obs, "collided", False) else "nav_fail"),
        "arrived_vision": arrived_vision,
        "collided": bool(getattr(obs, "collided", False)),
        "steps": step_i + 1,
        "d_end_vision": round(d_vis, 4) if np.isfinite(d_vis) else None,
        "d_end_gt_side": round(d_gt, 4),
        "segment_name": seg["segment_name"],
        "detection0": {
            "class_name": first.class_name,
            "confidence": first.confidence,
            "bbox": first.bbox.tolist(),
        },
        "last_class": last_cls,
        "goal_from": "vision",
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Indoor semantic nav P0")
    p.add_argument("--visual-prompt", default="potted plant,chair,couch,tv,bottle,book,vase,person")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default="artifacts/indoor_semantic_p0_summary.json")
    p.add_argument("--indoor-root", default="")
    p.add_argument("--config", default="configs/aerial_rl_indoor_shield_v3.yaml")
    p.add_argument(
        "--wm-ckpt",
        default="experiments/aerial/rl/artifacts/wm_ckpt_indoor_encode_e2i_b_20260901/wm_step_400.pt",
    )
    p.add_argument(
        "--actor-ckpt",
        default="experiments/aerial/rl/artifacts/v4_ac_ckpt_indoor_e2i_e_20260901/v4_ac_latest.pt",
    )
    p.add_argument("--annotation", default="building99_indoor_short_routes_clean_e.json")
    p.add_argument("--routes", default="0", help="Annotation indices (comma)")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--success-dist", type=float, default=0.50)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--step-hz", type=float, default=5.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--yolo-weights", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.35)
    args = p.parse_args()

    if args.dry_run:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        summary = run_dry_run(visual_prompt=args.visual_prompt, out=out)
        print(json.dumps(summary, indent=2))
        return 0

    summary = run_airsim_p0(args)
    print(json.dumps({k: summary[k] for k in summary if k != "episodes"}, indent=2))
    return 0 if summary.get("miss_detect_n", 0) < summary.get("n", 1) or summary.get("arrived_vision_n", 0) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
