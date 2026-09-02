#!/usr/bin/env bash
# Indoor fixed-object vgoal: find lobby pillar → stop 1m in front.
set -euo pipefail
VGOAL_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
INDOOR_ROOT="${AERIAL_INDOOR_ROOT:-/home/yao/aerial-indoor-wam}"
STAMP="${STAMP:-20260902_vgoal_pillar}"
ANN="${ANN:-$VGOAL_ROOT/artifacts/building99_indoor_pillar_standoff_east3.json}"
OUT="${OUT:-$VGOAL_ROOT/artifacts/indoor_vgoal_eval_${STAMP}.json}"
STANDOFF_M="${STANDOFF_M:-1.0}"
VISUAL_PROMPT="${VISUAL_PROMPT:-pillar}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-yolov8s-world.pt}"
YOLO_CONF="${YOLO_CONF:-0.25}"

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

mkdir -p "$VGOAL_ROOT/artifacts" "$VGOAL_ROOT/logs"
LOG="$VGOAL_ROOT/logs/indoor_vgoal_${STAMP}.log"
echo "[indoor_vgoal] target='$VISUAL_PROMPT' standoff=${STANDOFF_M}m weights=$YOLO_WEIGHTS $(date -Is)" | tee "$LOG"

cd "$VGOAL_ROOT"
# YOLO-World may download into cwd on first use
$AERIAL_PY examples/eval_indoor_semantic_p0.py \
  --indoor-root "$INDOOR_ROOT" \
  --annotation "$ANN" \
  --device cuda \
  --episodes "${EPISODES:-3}" \
  --standoff-m "${STANDOFF_M}" \
  --success-dist "${SUCCESS_DIST:-0.50}" \
  --yolo-weights "$YOLO_WEIGHTS" \
  --yolo-conf "$YOLO_CONF" \
  --yolo-imgsz "${YOLO_IMGSZ:-640}" \
  --visual-prompt "$VISUAL_PROMPT" \
  --out-report "$OUT" \
  2>&1 | tee -a "$LOG"

echo "[indoor_vgoal] done out=$OUT $(date -Is)" | tee -a "$LOG"
