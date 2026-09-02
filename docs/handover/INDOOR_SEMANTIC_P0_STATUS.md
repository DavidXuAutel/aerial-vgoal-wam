# Indoor Semantic P0 — STATUS

| 项 | 状态 |
|----|------|
| 接线 | ✅ 原 vgoal + `terminal_dock=False`（真视觉） |
| GT-dock 伪测 (sg) | ⚠️ 2/3 — **无效**（`policy_calls=0`） |
| **真视觉** (sg) | ✅ 已跑 stamp `20260902_vgoal_vision` · **0/3** arrived · coll=0 · `mean_policy_calls=242` |
| Gate / archive | ❌ 未过 — **不 archive** |

## 真视觉结果（`clean_sg`）

| route | arrived | d0→d_end | policy_calls | detect | last_det |
|-------|---------|----------|--------------|--------|----------|
| west | ❌ | 3.00→4.34 | 242 | 173 | person |
| south | ❌ | 3.30→4.70 | 242 | 174 | potted plant |
| east | ❌ | 3.34→4.72 | 242 | 163 | person |

报告：`artifacts/indoor_vgoal_eval_20260902_vgoal_vision.json`  
commits：vgoal `07e6e9e` · indoor collector `96e3cc5`（`terminal_dock` 默认开，F-cap 不变）

解读：YOLO 锁到走廊旁 COCO 目标并飞离 ann 航点；ann 到达门与「跟视觉目标」不一致。P0 需固定 `visual_prompt`/目标与航点对齐，或改评分到视觉目标距。

**禁止**：GT dock 刷 vision PASS；乱改 YOLO 默认冒充原 vgoal。
