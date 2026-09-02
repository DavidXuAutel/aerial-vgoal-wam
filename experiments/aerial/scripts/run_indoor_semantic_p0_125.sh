#!/usr/bin/env bash
# Indoor object-goal vgoal eval on 125 (fixed target class required).
set -euo pipefail
VGOAL_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
INDOOR_ROOT="${AERIAL_INDOOR_ROOT:-/home/yao/aerial-indoor-wam}"
STAMP="${STAMP:-20260902_vgoal_plant}"
# Spawn set facing a stable lobby plant (ann end unused; spawn-only).
ANN="${ANN:-$VGOAL_ROOT/artifacts/building99_indoor_plant_standoff_west3.json}"
OUT="${OUT:-$VGOAL_ROOT/artifacts/indoor_vgoal_eval_${STAMP}.json}"
STANDOFF_M="${STANDOFF_M:-1.0}"
# Fixed object — unfiltered / switching classes is not a valid trial.
TARGET_CLASSES="${TARGET_CLASSES:-potted plant}"

if [[ -z "${TARGET_CLASSES// }" ]]; then
  echo "[indoor_vgoal] TARGET_CLASSES must name a fixed object (e.g. 'potted plant')" >&2
  exit 2
fi

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
echo "[indoor_vgoal] indoor=$INDOOR_ROOT vgoal=$VGOAL_ROOT target='$TARGET_CLASSES' ann=$ANN $(date -Is)" | tee "$LOG"

cd "$VGOAL_ROOT"
$AERIAL_PY examples/eval_indoor_semantic_p0.py \
  --indoor-root "$INDOOR_ROOT" \
  --annotation "$ANN" \
  --device cuda \
  --episodes "${EPISODES:-3}" \
  --standoff-m "${STANDOFF_M}" \
  --success-dist "${SUCCESS_DIST:-0.50}" \
  --yolo-weights "${YOLO_WEIGHTS:-yolov8n.pt}" \
  --yolo-conf "${YOLO_CONF:-0.4}" \
  --yolo-imgsz "${YOLO_IMGSZ:-640}" \
  --target-classes "${TARGET_CLASSES}" \
  --out-report "$OUT" \
  2>&1 | tee -a "$LOG"

echo "[indoor_vgoal] done out=$OUT $(date -Is)" | tee -a "$LOG"
