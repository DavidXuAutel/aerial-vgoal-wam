# Indoor Semantic P0 — ARCHIVED checkpoint (2026-09-02)

> **裁定**：归档为 **接线 + 根因 checkpoint** · **不是** semantic-nav 任务 PASS。  
> **活状态**：[`INDOOR_SEMANTIC_P0_STATUS.md`](INDOOR_SEMANTIC_P0_STATUS.md)  
> **重开**：优化 YOLO/选框/站位后再跑；再验 WAM 站停。

## 任务定义（未过门）

大厅 **固定柱**（vision）→ 停在柱前 **1 m**（`approach_standoff_m=1`，success ≤0.5 m）。  
ann 仅 spawn；`terminal_dock=False`。

## 已 archive（接线）

| 项 | 状态 |
|----|------|
| 单相机 fan-out | ✅ capture → `rgb_vio` / `rgb`(WAM 224) / `rgb_yolo`(原生) |
| CaptureSettings | ✅ indoor `640×480`（勿用 outdoor 224 settings 跑 Building99） |
| YOLO 输入 | ✅ `vgoal/bridge.py` 优先 `obs.rgb_yolo`，WAM 仍吃 `rgb` |
| Eval 强制 WH | ✅ `eval_indoor_semantic_p0.py` + `run_indoor_vgoal_pillar_125.sh` patch/export |
| WAM 控制核 | **未证伪**；当前失败主因不在 WAM |

## 权威跑数（125）

| Stamp | prompt / conf | yolo_wh | arrival | coll | 备注 |
|-------|---------------|---------|---------|------|------|
| `vgoal_pillar_yolo640` | pillar@0.15 | 报告 None* | **0/3** | 0/3 | detect_hits=0；词太弱 |
| `vgoal_column_yolo640` | column@0.05 | **640×480** | **0/3** | **3/3** | 有 hits；锁闸机非真柱 |

\* `yolo_wh=None` 因零检出未写分辨率；smoke 已证实 fan-out 为 480×640。

工件：

- `artifacts/indoor_vgoal_eval_20260902_vgoal_column_yolo640.json`
- `artifacts/indoor_vgoal_eval_20260902_vgoal_pillar_yolo640.json`
- 诊断帧：`artifacts/videos/indoor_pillar_yolo_diag_20260902/`（含 `*_column_boxes.jpg`：框在闸机）

## 根因（定论）

1. Open-vocab YOLO-World 对大厅圆柱弱；`pillar` 几乎不报。  
2. `column@低 conf` 常点到 **闸机细杆**；`prefer_nearest` 锁错 → 朝东穿闸机碰撞。  
3. 真柱在画面里（尺度大、低对比、植物遮挡）但未进稳定框。

## 明确不宣称

- vision standoff PASS / 产品 12/12  
- GT terminal dock 或 unfiltered COCO 乱跟为 vision 成功  
- WAM 已在「正确柱前 1 m」上验证

## 重开条件

改进检测/选框（拒过小框、更好 prompt/conf）或换正对真柱站位 → 固定类稳定框柱 → 再跑 standoff；有近障策略后再谈站停率。

## 停做（无人令）

- 把本 checkpoint 写成任务完成  
- 再开双相机叙事 / YOLO 吃 224  
- 用 outdoor `settings.json`(224) 跑 indoor semantic
