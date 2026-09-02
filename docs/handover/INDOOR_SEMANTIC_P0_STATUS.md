# Indoor Semantic P0 — STATUS

> **2026-09-02 已归档 checkpoint**（非任务 PASS）：  
> [`INDOOR_SEMANTIC_P0_ARCHIVE_20260902.md`](INDOOR_SEMANTIC_P0_ARCHIVE_20260902.md)

| 项 | 状态 |
|----|------|
| 相机 fan-out | ✅ 单相机 640 → `rgb_vio` / `rgb`224 / `rgb_yolo`640 |
| 控制核 | WAM 未证伪；失败主因在 YOLO 选目标 |
| 感知 | ❌ 真柱弱报；`column` 易锁闸机 |
| Gate | **未过** — 大厅柱停前 1 m |

**下一步（重开时）**：优化 YOLO/选框/站位 → 再跑 → 再验 WAM 站停。
