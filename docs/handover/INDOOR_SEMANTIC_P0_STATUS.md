# Indoor Semantic P0 — STATUS

| 项 | 状态 |
|----|------|
| 相机 | **单相机 fan-out**：capture → `rgb_vio` / `rgb`(WAM 224) / `rgb_yolo`(原生 640×480) |
| 纠错 | 先前误把 YOLO 喂成 WAM 224；现 bridge 检测走 **`rgb_yolo`** |
| 任务 | 大厅 **pillar** → 停前 **1m** |
| pillar@rgb_yolo | ⏳ 重跑中 |

**禁止**：双相机叙事；YOLO 吃 224；换类乱跟。
