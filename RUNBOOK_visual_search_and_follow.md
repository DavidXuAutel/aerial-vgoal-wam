# Visual Object-Goal WAM: 自主区域搜寻与动态目标伴飞 Runbook

> **版本**：v1.0 (2026-08-28)  
> **前置依赖**：阶段 1 WAM 世界模型基线权重（`wm_step_3500.pt` + `v4_ac_latest.pt`）与基础几何反投影模块（`vgoal/`）全部验收通过。  
> **核心定位**：在现有 WAM 避障与空间导引底座上，实现**指定地理围栏内的自主全覆盖搜寻**、**目标识别锁定**、**3D 动态速度滤波**与**安全距离动态伴飞（Standoff Following）**。

---

## 一、 总体状态机与数据流

整个搜寻与伴飞流程由**五阶段闭环状态机（FSM）**驱动：

```text
               ┌──────────────────────────────┐
               │  Phase 1: AREA_SEARCHING     │ ◄──┐ (未见目标 / 搜索中)
               │  割草机航线/盘旋扫描 + YOLO   │    │
               └──────────────┬───────────────┘    │
                              │ 发现目标 (Confidence >= 0.5)
                              ▼                    │
               ┌──────────────────────────────┐    │
               │  Phase 2: TARGET_INTERCEPT   │    │
               │  计算目标 3D 坐标并逼近拦截   │    │
               └──────────────┬───────────────┘    │
                              │ 达到伴飞距离 (如 <= 8m)
                              ▼                    │
               ┌──────────────────────────────┐    │
               │  Phase 3: DYNAMIC_FOLLOW     │    │
               │  保持后上方偏置伴飞 + WAM避障 │    │
               └──────────────┬───────────────┘    │
                              │ 目标遮挡丢失 (< 3.0s)
                              ▼                    │
               ┌──────────────────────────────┐    │
               │  Phase 4: OCCLUDED_EXTRAP    │    │
               │  3D EKF 速度航位推算继续跟进  │    │
               └──────────────┬───────────────┘    │
                              │ 遮挡超时 (> 3.0s)  │
                              ▼                    │
               ┌──────────────────────────────┐    │
               │  Phase 5: LOCAL_REACQUIRE    │    │
               │  最后消失点半径 15m 盘旋搜寻  ├────┘ (重捕获失败回退全域)
               └──────────────────────────────┘
```

---

## 二、 分阶段实施计划（Roadmap）

### 阶段一：区域覆盖搜索航线生成器 (Area Search Planner)
* **目标**：输入多边形地理围栏 $[(x_1, y_1), ..., (x_n, y_n)]$，自动生成割草机扫描航线或螺旋盘旋航线，驱动无人机在无目标时自主扫描。
* **交付代码**：`vgoal/search_planner.py`、`tests/test_search_planner.py`。
* **核心指标**：
  * 区域覆盖完整率 $\ge 95\%$；
  * 转弯处平滑过渡，无急停震荡。

### 阶段二：动态目标 3D EKF 速度估计与伴飞偏移 (Dynamic Tracker & Standoff)
* **目标**：从单目反投影序列中滤波估计目标的 3D 世界速度 $\mathbf{v}_{\text{target}} \in \mathbb{R}^3$，并生成**后上方伴飞偏移量**。
* **交付代码**：`vgoal/dynamic_tracker.py`、`tests/test_dynamic_tracker.py`。
* **数学规范**：
  * 状态量：$\mathbf{x} = [p_x, p_y, p_z, v_x, v_y, v_z]^T$；
  * 虚拟伴飞目标点：
    $$\mathbf{P}_{\text{goal}}(t) = \mathbf{P}_{\text{target}}(t) - d_{\text{back}} \cdot \frac{\mathbf{v}_{\text{target}}}{\|\mathbf{v}_{\text{target}}\|} + h_{\text{above}} \cdot \mathbf{e}_z$$
  * （默认参数：后方 $d_{\text{back}} = 6.0\text{m}$，上方 $h_{\text{above}} = 3.0\text{m}$）。

### 阶段三：AirSim 移动目标动态闭环实测 (Moving Object AirSim Benchmark)
* **目标**：在 125 机器 AirSim 城镇/林区中放置匀速、变速及转弯移动车辆，进行真实动态搜寻与伴飞评测。
* **交付代码**：`examples/eval_dynamic_follow_airsim.py`、`artifacts/dynamic_follow_report.json`。
* **验收指标**：
  * **搜索发现率**：$\ge 90\%$（目标出现在搜索区 30 秒内被捕获）；
  * **动态伴飞保持率**：$\ge 80\%$（在移动过程中维持在目标 $4\text{m} \sim 10\text{m}$ 伴飞圈内）；
  * **严重碰撞率**：$\le 5.0\%$（在穿过树林或建筑物下方时，WAM 避障正常发挥作用，0 撞击）。

---

## 三、 执行指令与验证清单

### 1. 本地单元测试验证
```bash
cd ~/Projects/aerial-vgoal-wam
python3 -m unittest discover -s tests -p "test_*.py"
```

### 2. 同步至 125 机器
```bash
rsync -avz --exclude='.git' /Users/xudazhong/Projects/aerial-vgoal-wam/ cursor-125:~/aerial-vgoal-wam/
```

### 3. 在 125 上拉起动态伴飞闭环评测
```bash
ssh cursor-125 "bash -c 'cd ~/aerial-vgoal-wam && source ~/aerial-wam-v2/experiments/aerial/scripts/env_4090.sh && export PYTHONPATH=.:/home/yao/aerial-wam-v2 && nohup /home/yao/sim_verify/.venv/bin/python examples/eval_dynamic_follow_airsim.py --target-class car --episodes 8 > artifacts/eval_dynamic_follow.log 2>&1 & echo \$!'"
```

---

## 四、 责任分工与硬件约定

* **Mac 本机**：负责状态机设计、EKF 滤波算法、几何航线生成与单元测试。
* **cursor-125（4090）**：负责运行 AirSim 实时渲染、YOLO 目标检测推理与长程动态闭环评测。
* **主模型权重**：严格复用 `aerial-wam-v2` 阶段 1 结案权重，禁止在下游二次破坏性微调。
