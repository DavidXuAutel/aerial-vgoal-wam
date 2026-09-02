# Indoor Semantic P0 — STATUS

| 项 | 状态 |
|----|------|
| 接线 | ✅ 原 `VisualGoalWAMPolicy` + `YOLOTargetDetector`（conf=0.4 imgsz=640）→ indoor env/ckpt |
| Building99 闭环 (e, n=1) | ✅ stamp `20260902_vgoal` · 1/1 · coll=0 |
| Building99 闭环 (sg, n=3) | ⏳ stamp `20260902_vgoal_sg` · ann=`clean_sg` west/south/east |
| Gate | n≥3 · arrive @0.50 · coll=0 · `goal_from=vision` → 过则 archive |

**禁止**：自改 YOLO 入参/扇出分支冒充原 vgoal；GT 检测器刷 PASS。
