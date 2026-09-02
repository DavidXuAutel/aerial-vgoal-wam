# Indoor Semantic P0 — STATUS

| 项 | 状态 |
|----|------|
| 语义 | ✅ 固定物体 → 前 1m standoff 航点 → 飞到；ann 仅 spawn |
| 方法论纠错 | ❌ 先前 **unfiltered**（person/plant 乱跳）**无效**；现强制 `--target-classes` |
| 固定物体重跑 | ⏳ `potted plant` · west×3 spawn · stamp `20260902_vgoal_plant` |
| Gate / archive | ❌ 未过 |

探针：west 大厅 `potted plant` 稳定；`refrigerator` 在 sg 三向 FOV 里几乎看不到（不宜作本批固定目标）。

**禁止**：无过滤/换类冒充 object-goal；ann 终点冒充 vision PASS。
