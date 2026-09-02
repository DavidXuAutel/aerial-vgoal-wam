# Indoor Semantic P0 — STATUS

| 项 | 状态 |
|----|------|
| 接线 | ✅ 原 `VisualGoalWAMPolicy` + `YOLOTargetDetector`（conf=0.4 imgsz=640）→ indoor env/ckpt |
| Building99 闭环 | ✅ `669e5fb` @125 · stamp `20260902_vgoal` · **1/1 arrived** · coll=0 · d0=3.00→d_end=0.37 @0.50 |
| 报告 | `artifacts/indoor_vgoal_eval_20260902_vgoal.json` |

注：当前 ann `building99_indoor_short_routes_clean_e.json` 仅 **1** 条 route；脚本 `--episodes 3` 被 `min(n, len(routes))` 截成 1。扩 n≥3 需换/扩 ann，不是改 vgoal 栈。

**禁止**：自改 YOLO 入参/扇出分支冒充原 vgoal；GT 检测器刷 PASS。
