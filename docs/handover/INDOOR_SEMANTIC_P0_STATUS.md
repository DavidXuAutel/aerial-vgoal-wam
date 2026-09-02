# Indoor Semantic P0 — STATUS

| 项 | 状态 |
|----|------|
| 接线 | ✅ 原 `VisualGoalWAMPolicy` + `YOLOTargetDetector`（conf=0.4 imgsz=640）→ indoor env/ckpt |
| Building99 e (n=1) | ✅ stamp `20260902_vgoal` · 1/1 · coll=0 |
| Building99 sg (n=3) | ⚠️ stamp `20260902_vgoal_sg` · **2/3 arrived** · coll=0 · `goal_from=vision` |
| Gate / archive | ❌ 未过（需 n≥3 均 arrive @0.50）— **不 archive** |

## sg 明细（`clean_sg`）

| route | id | arrived | d0→d_end | notes |
|-------|-----|---------|----------|-------|
| 0 | west_3m | ✅ | 3.00→0.40 | steps=17 |
| 1 | south_3m | ❌ | 2.47→1.36 | hit max_steps=250 · d_min=0.79 |
| 2 | east_from_1 | ✅ | 3.95→0.29 | steps=19 |

报告：`artifacts/indoor_vgoal_eval_20260902_vgoal_sg.json` · commit `0873a1f`

**禁止**：自改 YOLO 入参/扇出分支冒充原 vgoal；GT 检测器刷 PASS。
