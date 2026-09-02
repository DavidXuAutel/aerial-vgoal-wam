# Indoor Semantic Nav (Method B) — Design

> **Date**: 2026-09-02  
> **Home project**: `aerial-vgoal-wam` (separate from `aerial-indoor-wam`)  
> **Status**: approved direction (B + P0→P1→P2); awaiting plan then implement  
> **Depends on**: Aerial indoor Stick F-cap ✅ (gt_proxy probe); outdoor Phase-2 / F4 domain FT **not** required for this track

## 1. Goal

Natural-language instruction → open-vocabulary detect → **search + lock + fly-to** in Building99, using existing WAM π + shield **without** retraining the world model / actor for domain adaptation.

Success is **vision-sourced `goal_rel`**, not GT world coordinates (GT allowed only as dual-report side notes).

## 2. Why a separate project

| Track | Repo | Role |
|-------|------|------|
| Stick indoor arrive @0.50 | `aerial-indoor-wam` | **Closed** as mainline probe (2026-09-02) |
| Outdoor Phase-2 long horizon | `aerial-wam-v2` | Continues separately; unfinished Phase-2 must not block this |
| Visual / semantic object-goal | **`aerial-vgoal-wam`** | This design; Method B home |

Indoor repo stays the **runtime/env donor** (AirSim Building99, E-head ckpt, three-zone shield). This repo owns detector, tracker, search, instruction→prompt, and the bridge that emits `goal_rel`.

## 3. Architecture (thin adapter)

```text
instruction
  → (P2) optional LLM → visual_prompt
  → open-vocab detector (YOLO-World or equivalent)
  → D̂ center-box depth → camera back-project → goal_rel
  → FSM: SEARCH | TRACK | APPROACH | DONE/FAIL
  → existing π(a | z, goal_rel) + indoor three-zone shield
```

Hard rules:

- Do **not** change F-cap defaults or claim indoor product 12/12 close.
- Reports must set `method=semantic_nav`, `goal_from=vision`.
- Do **not** silently use GT goal as vision success.
- Do **not** retrain WM for F4 as a prerequisite.
- Renderer mutex with outdoor Phase-2 on `:41451` unchanged.

## 4. Phased delivery

### P0 — Detect + fly when visible

- Fixed `visual_prompt` (no LLM).
- Open-vocab (or class-filtered) detect on RGB.
- Back-project with depth head / sim depth stub (declare source).
- If target in view: TRACK/APPROACH with indoor E-head + shield @0.50.
- Gate (n≥3): arrive @0.50 · `collided=false` · `goal_from=vision`.

### P1 — Search when not visible

- Lobby-scale search (yaw sweep + short lawnmower / frontier-lite).
- Loss handling: EMA/Kalman coast ≤ configured timeout then re-SEARCH.
- Gate: same as P0 but start pose **without** target in FOV; find within timeout.

### P2 — Instruction front-end

- Wire `qwen_deploy` (or equivalent) instruction → `visual_prompt`.
- End-to-end: one Chinese/English sentence → search+fly.
- Gate: ≥3 scripted instructions; fail_split `miss_detect | search_timeout | nav_fail | llm_bad_prompt`.

## 5. Code map (intended)

| Piece | Location |
|-------|----------|
| Detector / tracker / geometry | `vgoal/` (existing) |
| Search planner | `vgoal/search_planner.py` |
| Bridge policy | `vgoal/bridge.py` → adapt indoor obs / shield hooks |
| Instruction service | `qwen_deploy/` (P2) |
| Indoor eval driver | new `experiments/` or scripts here; call indoor env via documented path / subprocess — **no mixed PYTHONPATH with phase2** |
| Ckpt / shield yaml | consume from `aerial-indoor-wam` artifacts by path config |

## 6. Non-goals (this track)

- Closing outdoor Phase-2 SR / Step M.
- E3 odom/`arrived_hat` G1 (indoor sign-C stands).
- Putting west into indoor primary gate.
- Blind H100 FT / F4 domain train as a blocker.
- Claiming product indoor semantic nav complete after P0 only.

## 7. First acceptance (P0)

Building99 lobby, spawn from existing clean east/south safe pose, target class present in FOV:

- Instruction or fixed prompt names the object.
- `arrived_hat` or scored arrive @0.50 with vision `goal_rel`.
- Dual-report GT distance optional.
- Artifact: `artifacts/indoor_semantic_p0_summary_<STAMP>.json`.

## 8. Open decisions (resolve in plan)

1. Detector weights: YOLO-World vs ultralytics open-vocab vs COCO YOLO + mapped labels for P0.
2. Depth for back-project: indoor D̂ head vs AirSim depth camera stub (must declare).
3. Whether first P0 runs **inside** indoor repo as a thin script that imports installed `vgoal`, or only from this repo with `AERIAL_INDOOR_ROOT`.
