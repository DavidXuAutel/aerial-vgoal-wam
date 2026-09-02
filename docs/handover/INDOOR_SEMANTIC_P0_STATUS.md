# Indoor Semantic Nav P0 — STATUS

> **计划**：[`../superpowers/plans/2026-09-02-indoor-semantic-nav-p0.md`](../superpowers/plans/2026-09-02-indoor-semantic-nav-p0.md)  
> **设计**：[`../superpowers/specs/2026-09-02-indoor-semantic-nav-design.md`](../superpowers/specs/2026-09-02-indoor-semantic-nav-design.md)

| 项 | 状态 |
|----|------|
| 接线 | ✅ 原 `VisualGoalWAMPolicy` + `YOLOTargetDetector`（conf=0.4 imgsz=640）→ indoor env/ckpt |
| Building99 闭环 | ⏳ 125 跑 `run_indoor_semantic_p0_125.sh` |

**禁止**：自改 YOLO 入参/扇出分支冒充原 vgoal；GT 检测器刷 PASS。
