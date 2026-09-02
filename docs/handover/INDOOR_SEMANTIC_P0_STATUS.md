# Indoor Semantic P0 — STATUS

| 项 | 状态 |
|----|------|
| 任务 | 找大厅 **pillar** → 停在前 **1m** |
| 检测 | YOLO-World `visual_prompt=pillar` · 近距优先 · ann 仅 spawn |
| pillar2 @125 | ⚠️ **0/3** arrived · 目标已锁 `pillar` · coll=3/3 · conf=0.01 |
| Gate / archive | ❌ |

## pillar2 明细

| route | det | hits | min_obj | min_standoff | coll |
|-------|-----|------|---------|--------------|------|
| east_a | pillar | 7 | 4.07 | 2.45 | Y |
| east_b | pillar | 1 | 21.55 | 20.54 | Y |
| east_c | pillar | 5 | 2.69 | 1.20 | Y |

报告：`artifacts/indoor_vgoal_eval_20260902_vgoal_pillar2.json`  
注：224 RGB 下 pillar 开词表置信度极低（~0.01）；east_c 最接近（obj≈2.7m / standoff≈1.2m）但撞了。

**禁止**：换类乱跟；ann 终点冒充 vision PASS。
