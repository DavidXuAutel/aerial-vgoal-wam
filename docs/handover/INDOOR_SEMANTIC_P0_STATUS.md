# Indoor Semantic P0 — STATUS

| 项 | 状态 |
|----|------|
| 语义 | ✅ YOLO 目标 → **物体前 standoff 航点**（默认 1m）→ 飞到该点；ann **仅 spawn** |
| 真视觉+depth | ✅ `use_full_obs` 把 depth 交给反投影（`policy_view` 无 depth） |
| standoff2 @125 | ⚠️ **1/3** vision_arrived · coll=3/3 · stamp `20260902_vgoal_standoff2` |
| Gate / archive | ❌ 未过 |

## standoff2 明细

| route | vision_arr | min_standoff | min_obj | det | coll |
|-------|------------|--------------|---------|-----|------|
| west | ✅ | 0.36 | 0.91 | person | ✅ |
| south | ❌ | 1.28 | 2.28 | person | ✅ |
| east | ❌ | 2.40 | 3.19 | person | ✅ |

报告：`artifacts/indoor_vgoal_eval_20260902_vgoal_standoff2.json`  
commits：vgoal `81456ac` · indoor `f68481f`

**禁止**：用 ann 几何终点冒充 vision PASS；GT dock 刷分。
