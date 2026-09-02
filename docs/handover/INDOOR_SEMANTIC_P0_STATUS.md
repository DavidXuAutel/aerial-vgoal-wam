# Indoor Semantic P0 — STATUS

| 项 | 状态 |
|----|------|
| 接线 | ✅ 原 `VisualGoalWAMPolicy` + `YOLOTargetDetector` → indoor env/ckpt |
| Building99 e (n=1) | ✅ 1/1 · coll=0 |
| Building99 sg (n=3) | ⚠️ **2/3** · coll=0 · stamp `20260902_vgoal_sg` |
| Gate / archive | ❌ 未过 — **不 archive** |

## South 失败根因（2026-09-02 diag）

复现 `west→south→east`：`artifacts/diag_sg_order_20260902.json`

1. **三条路 `policy_calls=0`**：indoor `RolloutCollector` 在 `d_goal≤35m` 且 `d_fwd≥1.5` 时走 **ann GT docking**，不调用 `VisualGoalWAMPolicy`。无 depth head 时 `d_fwd` 默认 5.0 → **几乎全程 GT dock**。报告 `goal_from=vision` **名不副实**。
2. **South 失败机制**：GT dock 在南向走廊 **冲过/横漂振荡**（y≈-5.5↔-3.1，x±1.4），`d_min≈0.99` 进不了 0.50；打满 250 步。West/east 短途 dock 能收在 0.50 内。
3. Spawn YOLO 能看到 `person`，但本评测 **未用** 视觉目标控机。

**禁止**：自改 YOLO 冒充原 vgoal；用 GT dock 刷 vision PASS。
