#!/usr/bin/env bash
# Indoor semantic P0 on 125 — Building99 + indoor ckpts + vgoal detector.
set -euo pipefail
VGOAL_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
INDOOR_ROOT="${AERIAL_INDOOR_ROOT:-/home/yao/aerial-indoor-wam}"
STAMP="${STAMP:-20260902_p0}"
OUT="${OUT:-$VGOAL_ROOT/artifacts/indoor_semantic_p0_summary_${STAMP}.json}"
PROMPT="${VISUAL_PROMPT:-refrigerator,potted plant,chair,couch,tv,bottle,book,vase,person,dining table}"
export INDOOR_CAPTURE_W="${INDOOR_CAPTURE_W:-640}"
export INDOOR_CAPTURE_H="${INDOOR_CAPTURE_H:-480}"
export AIRSIM_FANOUT_RGB=1
export WAM_ENCODE_SIZE="${WAM_ENCODE_SIZE:-224}"

cd "$INDOOR_ROOT"
# shellcheck disable=SC1091
source experiments/aerial/scripts/env_4090.sh
export AERIAL_INDOOR_ROOT="$INDOOR_ROOT"
export AIRSIM_CAMERA=0 AIRSIM_VEHICLE=drone_1

if ! pgrep -f 'Building_99/Binaries' >/dev/null || ! ss -ltn | grep -q 41451; then
  ON_SCREEN="${ON_SCREEN:-0}" bash experiments/aerial/scripts/recover_renderer_scene.sh building99 || true
  sleep 15
fi
ss -ltn | grep -q 41451

# Force single-cam CaptureSettings to indoor sensing res (fan-out derives WAM 224).
python3 - <<'PY'
import json
from pathlib import Path
p = Path("/home/yao/aerial_airsim_persistent/AirSim/settings_indoor.json")
if not p.is_file():
    raise SystemExit("missing settings_indoor.json")
d = json.loads(p.read_text())
changed = False
w, h = int(__import__("os").environ.get("INDOOR_CAPTURE_W", "640")), int(__import__("os").environ.get("INDOOR_CAPTURE_H", "480"))

def fix(caps):
    global changed
    for c in caps:
        if int(c.get("ImageType", -1)) == 0:
            if c.get("Width") != w or c.get("Height") != h:
                c["Width"] = w
                c["Height"] = h
                changed = True

fix(d["CameraDefaults"]["CaptureSettings"])
for cam in d["Vehicles"]["drone_1"]["Cameras"].values():
    fix(cam["CaptureSettings"])
if changed:
    p.write_text(json.dumps(d, indent=2) + "\n")
    print(f"[semantic_p0] settings Scene forced {w}x{h}")
PY

mkdir -p "$VGOAL_ROOT/artifacts" "$VGOAL_ROOT/logs"
LOG="$VGOAL_ROOT/logs/indoor_semantic_p0_${STAMP}.log"
echo "[semantic_p0] indoor=$INDOOR_ROOT vgoal=$VGOAL_ROOT prompt=$PROMPT $(date -Is)" | tee "$LOG"

cd "$VGOAL_ROOT"
$AERIAL_PY examples/eval_indoor_semantic_p0.py \
  --indoor-root "$INDOOR_ROOT" \
  --visual-prompt "$PROMPT" \
  --device cuda \
  --seeds "${SEEDS:-0,1,2}" \
  --routes "${ROUTES:-0}" \
  --out "$OUT" \
  2>&1 | tee -a "$LOG"

echo "[semantic_p0] done out=$OUT $(date -Is)" | tee -a "$LOG"
