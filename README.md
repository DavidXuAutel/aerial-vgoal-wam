# Aerial-VGoal-WAM (Visual Object-Goal Aerial World-Action Model)

> **项目定位**：基于视觉目标识别与世界模型（World Model）想象推演的无人机自主搜寻与绕障导航系统。  
> **核心优势**：**零人工标注、零大模型重训、轻量检测器旁路挂载、纯即插即用**。  

---

## 📖 项目简介

本仓库将开源轻量级目标检测模型（如 YOLOv8-Nano / MobileNet-SSD）与 Aerial-WAM 世界模型及学得策略无缝连接：
1. 单目摄像头实时识别目标物体（车辆、人员、停机坪等）；
2. 结合 WAM 密集深度预测头 $\hat{D}$ 完成 2D 像素到 3D 机体坐标的几何反投影；
3. 驱动 WAM 策略大脑在世界模型脑海推演中自主规划平滑、无碰撞轨迹扑向目标。

---

## 📂 核心文档与设计方案

* 完整设计方案详见：[`docs/plans/visual-object-goal-wam.md`](docs/plans/visual-object-goal-wam.md)

---

## 🛠️ 快速开始

```bash
# 1. 克隆项目与安装依赖
git clone https://github.com/your-org/aerial-vgoal-wam.git
cd aerial-vgoal-wam
pip install -r requirements.txt

# 2. 运行单帧反投影与目标跟踪测试
python -m vgoal.tests.test_target_tracker
```

---

## 架构示意

```text
RGB (5Hz) ──► [YOLO-Nano 检测] ──► 2D Bounding Box
                  │
                  ▼
RGB (5Hz) ──► [WAM 深度预测 D̂] ──► 2D-to-3D 反投影 ──► goal_rel [3D 相对目标]
                                                           │
                                                           ▼
                                            [WAM 策略大脑 π(a | z, goal_rel)]
                                                           │
                                                           ▼
                                            [自主绕障无碰撞飞抵目标]
```
