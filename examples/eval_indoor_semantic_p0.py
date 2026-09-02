#!/usr/bin/env python3
"""Indoor semantic nav P0 eval — reuse vgoal detector/geometry; indoor env via AERIAL_INDOOR_ROOT.

P0 scope: target already in FOV · fixed visual_prompt · vision goal_rel · arrive @0.50.
Search (P1) and LLM instruction (P2) are out of scope here.

Dry-run (no AirSim)::

    python examples/eval_indoor_semantic_p0.py --dry-run --visual-prompt chair \\
      --out artifacts/indoor_semantic_p0_dryrun.json

125 full run needs Building99 + indoor ckpts (see README).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vgoal.detector import MockDetector, OpenVocabPromptDetector
from vgoal.geometry import CameraIntrinsics, bbox_to_goal_rel
from vgoal.report_meta import semantic_nav_report_fields


def _indoor_root() -> Path:
    env = os.environ.get("AERIAL_INDOOR_ROOT", "").strip()
    if env:
        return Path(env).expanduser()
    for cand in (
        Path.home() / "Projects" / "aerial-indoor-wam",
        Path("/home/yao/aerial-indoor-wam"),
        Path.home() / "aerial-indoor-wam",
    ):
        if cand.is_dir():
            return cand
    return Path.home() / "Projects" / "aerial-indoor-wam"


def run_dry_run(*, visual_prompt: str, out: Path) -> Dict[str, Any]:
    """Offline smoke: mock detection + depth → vision goal_rel + report meta."""
    h, w = 224, 224
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), 3.0, dtype=np.float32)
    # Put a fake box in the image center
    bbox = [w * 0.35, h * 0.35, w * 0.65, h * 0.65]
    inner = MockDetector(target_bbox=bbox, confidence=0.9, class_name=visual_prompt.split(",")[0].strip() or "target")
    det = OpenVocabPromptDetector(visual_prompt=visual_prompt, inner=inner)
    hit = det.detect(rgb)
    if hit is None:
        raise RuntimeError("dry-run mock detector returned None")
    K = CameraIntrinsics.from_fov(80.0, width=w, height=h)
    goal_rel = bbox_to_goal_rel(hit.bbox, depth, K)
    meta = semantic_nav_report_fields(
        depth_source="airsim_depth",
        visual_prompt=visual_prompt,
        phase="P0",
    )
    summary = {
        **meta,
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


def main() -> int:
    p = argparse.ArgumentParser(description="Indoor semantic nav P0 (reuse vgoal)")
    p.add_argument("--visual-prompt", default="chair", help="Fixed open-vocab / class prompt")
    p.add_argument("--dry-run", action="store_true", help="No AirSim; mock detect + report meta")
    p.add_argument(
        "--out",
        default="artifacts/indoor_semantic_p0_summary.json",
        help="Output JSON path",
    )
    p.add_argument(
        "--indoor-root",
        default="",
        help="Override AERIAL_INDOOR_ROOT (full AirSim path; unused in --dry-run)",
    )
    args = p.parse_args()
    out = Path(args.out)
    if args.indoor_root:
        os.environ["AERIAL_INDOOR_ROOT"] = args.indoor_root

    if args.dry_run:
        summary = run_dry_run(visual_prompt=args.visual_prompt, out=out)
        print(json.dumps(summary, indent=2))
        print(f"[ok] wrote {out}")
        return 0

    indoor = _indoor_root()
    if not indoor.is_dir():
        print(
            f"[error] indoor root not found: {indoor}. "
            "Set AERIAL_INDOOR_ROOT or pass --dry-run.",
            file=sys.stderr,
        )
        return 2
    print(
        f"[info] Full AirSim P0 not wired in this commit beyond path check "
        f"(indoor={indoor}). Use --dry-run for local gate of meta/detector; "
        "125 closed-loop hook follows in next commit once env import is verified.",
        file=sys.stderr,
    )
    # Still emit a miss_detect placeholder so CI/agents have a schema
    meta = semantic_nav_report_fields(
        depth_source="airsim_depth",
        visual_prompt=args.visual_prompt,
        phase="P0",
    )
    summary: Dict[str, Any] = {
        **meta,
        "mode": "airsim_pending",
        "indoor_root": str(indoor),
        "arrived": False,
        "fail_split": {"airsim_loop": 1},
        "note": "detector+meta ready; closed-loop driver next",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
