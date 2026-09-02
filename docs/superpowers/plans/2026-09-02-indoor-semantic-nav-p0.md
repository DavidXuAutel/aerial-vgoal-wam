# Indoor Semantic Nav P0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse existing `vgoal/` to run Building99 P0: fixed `visual_prompt` → detect → vision `goal_rel` → fly-to @0.50 when target is already in FOV (no search yet).

**Architecture:** Thin adapters on top of `YOLOTargetDetector` / `MockDetector`, `geometry.bbox_to_goal_rel`, and `VisualGoalWAMPolicy`. Indoor AirSim + E-head + shield come from `AERIAL_INDOOR_ROOT` (`aerial-indoor-wam`). No WM retrain.

**Tech Stack:** Python 3.10+, numpy, ultralytics (YOLO / YOLO-World), pytest; AirSim via indoor repo on 125.

## Global Constraints

- Reuse `vgoal/` — do not fork a parallel detector/tracker stack.
- Reports: `method=semantic_nav`, `goal_from=vision`; never silent GT goal as success.
- P0 depth: declare `depth_source=airsim_depth` (sim depth map); D̂ head is later optional.
- P0 detector: prefer ultralytics YOLO-World when `model_path` contains `world`; else COCO YOLO filtered by classes parsed from `visual_prompt`.
- Driver lives in this repo; `sys.path` / env `AERIAL_INDOOR_ROOT` only — never mix phase2 PYTHONPATH.
- Do not change indoor F-cap defaults or claim product 12/12.

## Resolved open decisions (from design §8)

1. **Detector:** `OpenVocabPromptDetector` — YOLO-World if available; else `YOLOTargetDetector` + class tokens from prompt.
2. **Depth:** AirSim depth for P0 (`depth_source=airsim_depth`).
3. **Driver:** `examples/eval_indoor_semantic_p0.py` in this repo + `AERIAL_INDOOR_ROOT`.

## File map

| File | Role |
|------|------|
| `vgoal/prompt_classes.py` | Parse `visual_prompt` → class name tokens |
| `vgoal/detector.py` | Add `OpenVocabPromptDetector` |
| `vgoal/report_meta.py` | Standard report fields helper |
| `tests/test_prompt_classes.py` | Unit tests for prompt parsing |
| `tests/test_open_vocab_detector.py` | Unit tests (mock YOLO path) |
| `examples/eval_indoor_semantic_p0.py` | P0 AirSim eval driver |
| `docs/superpowers/specs/2026-09-02-indoor-semantic-nav-design.md` | Mark P0 plan link |

---

### Task 1: Prompt → class tokens

**Files:**
- Create: `vgoal/prompt_classes.py`
- Create: `tests/test_prompt_classes.py`
- Modify: `vgoal/__init__.py` (export `classes_from_visual_prompt`)

**Interfaces:**
- Produces: `classes_from_visual_prompt(prompt: str) -> list[str]`

- [ ] **Step 1: Write the failing test**

```python
from vgoal.prompt_classes import classes_from_visual_prompt

def test_splits_common_prompt():
    assert classes_from_visual_prompt("red chair, person") == ["red chair", "person"]

def test_strips_empty():
    assert classes_from_visual_prompt("  chair ,,  ") == ["chair"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prompt_classes.py -v`  
Expected: FAIL import or missing function

- [ ] **Step 3: Write minimal implementation**

```python
def classes_from_visual_prompt(prompt: str) -> list[str]:
    parts = [p.strip() for p in str(prompt).replace(";", ",").split(",")]
    return [p for p in parts if p]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_prompt_classes.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add vgoal/prompt_classes.py tests/test_prompt_classes.py vgoal/__init__.py
git commit -m "Add visual_prompt class token parser for semantic nav P0."
```

---

### Task 2: OpenVocabPromptDetector (reuse YOLO stack)

**Files:**
- Modify: `vgoal/detector.py`
- Create: `tests/test_open_vocab_detector.py`

**Interfaces:**
- Consumes: `classes_from_visual_prompt`, `YOLOTargetDetector`, `BaseDetector`, `DetectionResult`
- Produces: `OpenVocabPromptDetector(visual_prompt, model_path=..., conf_threshold=..., device=...)` with `.set_visual_prompt(prompt)` and `.detect(rgb)`

- [ ] **Step 1: Write the failing test**

```python
from vgoal.detector import OpenVocabPromptDetector, MockDetector

def test_open_vocab_uses_prompt_classes_with_inner_mock():
    # Construct with injected inner detector for unit test
    inner = MockDetector(target_bbox=[10, 10, 40, 40], class_name="chair")
    det = OpenVocabPromptDetector(visual_prompt="chair", inner=inner)
    import numpy as np
    out = det.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    assert out is not None
    assert out.class_name == "chair"
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_open_vocab_detector.py -v`  
Expected: FAIL — `OpenVocabPromptDetector` missing

- [ ] **Step 3: Implement**

Add class that:
- Stores `visual_prompt` and optional `inner` BaseDetector for tests.
- If `inner` is None: if `"world" in model_path.lower()` load ultralytics YOLO and call `.set_classes(classes)`; else wrap `YOLOTargetDetector(target_classes=classes)`.
- `detect` delegates to inner / YOLO-World and returns best box.

- [ ] **Step 4: pytest pass**

- [ ] **Step 5: Commit**

```bash
git commit -m "Add OpenVocabPromptDetector reusing YOLO / YOLO-World."
```

---

### Task 3: Report meta helper

**Files:**
- Create: `vgoal/report_meta.py`
- Create: `tests/test_report_meta.py`

**Interfaces:**
- Produces: `semantic_nav_report_fields(*, depth_source: str, visual_prompt: str, phase: str = "P0") -> dict`

Must include at least:
`method=semantic_nav`, `goal_from=vision`, `depth_source`, `visual_prompt`, `phase`

- [ ] Steps: failing test → implement → pytest → commit  
  Message: `Add semantic_nav report metadata helper.`

---

### Task 4: P0 eval driver (reuse bridge; indoor root)

**Files:**
- Create: `examples/eval_indoor_semantic_p0.py`
- Modify: `README.md` (P0 command blurb)

**Interfaces:**
- Env: `AERIAL_INDOOR_ROOT` default `~/Projects/aerial-indoor-wam` or `/home/yao/aerial-indoor-wam`
- CLI: `--visual-prompt`, `--dry-run` (MockDetector + fake depth, no AirSim), `--out artifacts/...`
- On full run: insert indoor root on `sys.path`, build `OpenVocabPromptDetector`, wire `VisualGoalWAMPolicy` like `examples/eval_visual_goal_airsim.py` but:
  - annotation from indoor clean east/south
  - stamp report with `semantic_nav_report_fields`
  - success_dist 0.50
  - require target in FOV at start for P0 (skip search)

- [ ] **Step 1:** `--dry-run` path writes JSON with `method=semantic_nav`, `goal_from=vision`
- [ ] **Step 2:** Document 125 command in README
- [ ] **Step 3:** Commit `Add indoor semantic P0 eval driver (dry-run + AirSim hook).`

---

### Task 5: Spec + STATUS pointer

**Files:**
- Modify: `docs/superpowers/specs/2026-09-02-indoor-semantic-nav-design.md` (status → P0 implementing; link plan)
- Create: `docs/handover/INDOOR_SEMANTIC_P0_STATUS.md` (checklist empty → fill after 125 run)

- [ ] Commit: `Point design at P0 plan; add P0 status stub.`

---

### Task 6: 125 smoke (human / agent on cursor-125)

Only after Tasks 1–5 green locally:

```bash
export AERIAL_INDOOR_ROOT=/home/yao/aerial-indoor-wam
cd /home/yao/aerial-vgoal-wam   # or sync this repo to 125
python examples/eval_indoor_semantic_p0.py \
  --visual-prompt "chair" \
  --seeds 0,1,2 \
  --out artifacts/indoor_semantic_p0_summary_20260902.json
```

Gate: n≥3 · arrive @0.50 · collided=false · `goal_from=vision`.  
If Building99 has no reliable COCO class in FOV, record `miss_detect` and stop — do not cheat with GT detector for PASS.

---

## P1 / P2 (out of this plan)

- P1: enable `search_planner` when no detection at t0  
- P2: `qwen_deploy` → `visual_prompt`  

Separate plans after P0 gate or honest miss_detect baseline.

## Spec coverage check

| Spec item | Task |
|-----------|------|
| Reuse vgoal | 2, 4 |
| Fixed visual_prompt P0 | 1, 2, 4 |
| Vision goal_rel / report fields | 3, 4 |
| AirSim depth declare | 3, 4 |
| AERIAL_INDOOR_ROOT driver | 4 |
| Gate n≥3 @0.50 | 6 |
| No F4 / no GT cheat | Global + Task 6 |
