# Indoor Semantic P0 — STATUS

| 项 | 状态 |
|----|------|
| 语义 | ✅ 固定物体 → 前 1m standoff；ann 仅 spawn |
| 方法论 | ✅ 强制 `--target-classes`（禁止 unfiltered 换目标） |
| 固定 `potted plant` west×3 | ⚠️ **0/3** · 全 `last_det=potted plant` · 但 `min_obj≈15m`（跟到远株）· coll=3/3 |
| Gate / archive | ❌ |

报告：`artifacts/indoor_vgoal_eval_20260902_vgoal_plant.json`  
说明：类别已锁死，但 YOLO 仍可能选**同名远距离实例**；需实例锁定 / 近距优先，不是再换类乱跟。

**禁止**：无过滤换目标；ann 终点冒充 vision PASS。
