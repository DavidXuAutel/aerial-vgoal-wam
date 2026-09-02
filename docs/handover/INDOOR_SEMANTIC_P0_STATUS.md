# Indoor Semantic P0 — STATUS

| 项 | 状态 |
|----|------|
| 相机 | **单相机 fan-out**：capture → `rgb_vio` / `rgb`(WAM 224) / `rgb_yolo`(原生) |
| 纠错 | YOLO 必须吃 `rgb_yolo`；**CaptureSettings 也必须是 640×480**（曾误留 224 出图） |
| 任务 | 大厅 **pillar** → 停前 **1m** |
| pillar@rgb_yolo640 | ⏳ 重启 renderer 后重跑 |

**禁止**：双相机叙事；YOLO 吃 224；换类乱跟；用 outdoor `settings.json`(224) 跑 indoor。
