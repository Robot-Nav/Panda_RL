<div align="center">

# Franka Panda 机械臂PPO强化学习项目

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.3+-orange)](https://mujoco.org/)
[![SB3](https://img.shields.io/badge/Stable--Baselines3-2.6+-green)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.1+-red)](https://gymnasium.farama.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/fatu08/mujoco-learning?style=social)](https://github.com/fatu08/mujoco-learning)

> PPO + MuJoCo 强化学习训练框架 — 单臂目标到达 / 单臂避障 / 双臂避障
> 持续更新ing）


</div>

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 核心算法原理](#2-核心算法原理)
- [3. 任务详解](#3-任务详解)
  - [3.1 Reach Target（目标到达）](#31-reach-target目标到达)
  - [3.2 Obstacle Avoidance（单臂避障）](#32-obstacle-avoidance单臂避障)
  - [3.3 Dual Arm Obstacle（双臂避障）](#33-dual-arm-obstacle双臂避障)
- [4. 代码结构](#4-代码结构)
- [5. 关键实现详解](#5-关键实现详解)
- [6. 配置参数与调优指南](#6-配置参数与调优指南)
- [7. 运行步骤](#7-运行步骤)
- [8. 三文件对比分析](#8-三文件对比分析)
- [9. 常见问题与解决方案](#9-常见问题与解决方案)
- [贡献](#贡献)
- [引用](#引用)
- [License](#license)
- [致谢](#致谢)

---

## 1. 项目概述

本项目包含三个基于 **PPO (Proximal Policy Optimization)** 算法的 Franka Emika Panda 机械臂强化学习任务，使用 **MuJoCo** 物理引擎仿真和 **Stable-Baselines3** 训练框架。
相关仿真如下：

1）Reach Target

https://github.com/user-attachments/assets/8078ce89-4ace-4fbc-b761-b4de4069ecf0



2）Obstacle Avoidance

https://github.com/user-attachments/assets/87102296-5ea1-4094-8a23-10fccc956ecb



3）Dual Arm Obstacle

https://github.com/user-attachments/assets/422ab354-55e5-4989-8921-4b8322aff092


### 1.1 三个任务总览

| 属性 | Reach Target | Obstacle Avoidance | Dual Arm Obstacle |
|------|:---:|:---:|:---:|
| **主文件** | `rl_panda_reach_target_high_profile.py` | `rl_panda_obstacle_high_profile.py` | `rl_dual_arm_obstacle.py` |
| **任务** | 单臂末端到达随机目标点 | 单臂到达目标并避开障碍物 | 双臂分别到达目标并避开障碍物 |
| **机械臂数量** | 1 | 1 | 2 (左臂 + 右臂) |
| **障碍物** | 无 | 1个球体 | 1个球体 |
| **动作空间** | 7维 `[-1,1]` | 7维 `[-1,1]` | 14维 `[-1,1]` |
| **观测空间** | 20维 | 27维 | 50维 |
| **动作类型** | delta 增量控制 | absolute 绝对控制 | delta 增量控制 |
| **架构** | 模块化（env/reward/config/callbacks） | 单体文件 | 单体文件 |
| **模型场景** | `scene.xml` | `scene_pos_with_obstacles.xml` | `scene_dual_arm_with_obstacles.xml` |

### 1.2 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | 8 核 | 20+ 核 |
| GPU | 无（CPU可训练） | NVIDIA GPU (CUDA) |
| RAM | 16 GB | 32 GB |
| 操作系统 | Linux | Ubuntu 20.04+ |

### 1.3 核心依赖

```
Python 3.11+
MuJoCo 3.3+
Stable-Baselines3 2.6+
Gymnasium 1.1+
PyTorch 2.x
NumPy, SciPy
TensorBoard (可视化)
```

---

## 2. 核心算法原理

### 2.1 PPO 算法

PPO (Proximal Policy Optimization) 是一种基于策略梯度的强化学习算法，通过限制策略更新幅度来保证训练稳定性。

**核心思想**：在新旧策略之间加一个"信任域"约束，防止策略更新过大导致性能崩溃。

#### PPO-Clip 目标函数

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

其中概率比 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$，$\epsilon$ 为裁剪范围（通常 0.2），$\hat{A}_t$ 为优势函数估计。

#### GAE (Generalized Advantage Estimation)

$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

其中 TD 误差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，$\gamma$ 为折扣因子，$\lambda$ 控制偏差-方差权衡。

#### 价值函数损失

$$L^{VF}(\theta) = \mathbb{E}_t\left[(V_\theta(s_t) - V_t^{target})^2\right]$$

#### 熵奖励

$$L^{ENT}(\theta) = \mathbb{E}_t\left[\mathcal{H}(\pi_\theta(\cdot|s_t))\right]$$

总损失：$L = L^{CLIP} - c_1 L^{VF} + c_2 L^{ENT}$

### 2.2 训练架构

```
+------------------------------------------------------------------------------------+
|                              主进程 (GPU)                                           |
|  +-------------------------- PPO Agent ------------------------------------------+ |
|  |  MlpPolicy                                                                     | |
|  |  +----------------------+    +----------------------+                          | |
|  |  | Actor (pi)           |    | Critic (vf)          |                          | |
|  |  | 256->256->128        |    | 512->256->128        |                          | |
|  |  | -> action_dist       |    | -> value             |                          | |
|  |  +----------------------+    +----------------------+                          | |
|  |  activation: Tanh, ortho_init: True                                            | |
|  +--------------------------------------------------------------------------------+ |
|                        | rollout data (n_steps x n_envs)                           |
|  +--------------------------------------------------------------------------------+ |
|  |              SubprocVecEnv (24 并行进程, CPU)                                    | |
|  |  +--------+ +--------+ +--------+     +----------+                             | |
|  |  | Env 0  | | Env 1  | | Env 2  | ... | Env 23  |                             | |
|  |  | MuJoCo | | MuJoCo | | MuJoCo |     | MuJoCo  |                             | |
|  |  +--------+ +--------+ +--------+     +----------+                             | |
|  |        VecNormalize (obs + reward 归一化)                                       | |
|  +--------------------------------------------------------------------------------+ |
+------------------------------------------------------------------------------------+
```

**关键流程**：
1. 24 个 MuJoCo 环境在 CPU 上并行运行，每个独立仿真
2. 从所有环境收集 `n_steps` 步 transition
3. 数据从 CPU 传输到 GPU
4. PPO 进行 `n_epochs` 轮策略更新
5. 循环直到 `total_timesteps`

### 2.3 动作空间设计

#### Reach Target / Dual Arm (Delta Control)

```python
action in [-1, 1]^N
target_joint_pos = current_joint_pos + action * action_scale * joint_range_width
target_joint_pos = clip(target_joint_pos, joint_min, joint_max)
```

**优势**：动作在当前位置附近探索，轨迹天然平滑，适合精确到达任务。解决"机械臂运动过快"的关键设计。

#### Obstacle Avoidance (Absolute Control)

```python
action in [-1, 1]^N
scaled = joint_min + (action + 1) * 0.5 * (joint_max - joint_min)
```

**优势**：一步可到达关节空间任意位置，适合需要大范围运动的任务。

### 2.4 多子步仿真

每次环境步执行 `n_substeps` 次 MuJoCo 物理步进，确保 PD 位置控制器能平滑响应：

```python
for _ in range(self._n_substeps):  # 默认 5 次
    mujoco.mj_step(self.model, self.data)
```

---

## 3. 任务详解

### 3.1 Reach Target（目标到达）

**文件**：[`rl_panda_reach_target_high_profile.py`](rl_panda_reach_target_high_profile.py)

**MuJoCo 场景**：[`model/franka_emika_panda/scene.xml`](model/franka_emika_panda/scene.xml)

单臂 Panda 机械臂从初始位置出发，使末端执行器到达工作空间内的随机目标点。

#### 观测空间（20维）

$$O = [q_{1:7},\ \dot{q}_{1:7},\ p_{ee},\ p_{goal} - p_{ee}]$$

| 索引 | 内容 | 维度 | 物理含义 |
|------|------|------|----------|
| 0-6 | `joint_pos` | 7 | 关节角度 (rad) |
| 7-13 | `joint_vel` | 7 | 关节角速度 (rad/s) |
| 14-16 | `ee_pos` | 3 | 末端执行器三维位置 |
| 17-19 | `ee_to_goal` | 3 | ee -> goal 向量 |

#### 动作空间（7维）

a ∈ [-0.1, 0.1]^7   (delta control, action_scale = 0.1)

#### 奖励函数（7项组合）

由 `ReachTargetReward` 类实现：

$$R = R_{dist} + R_{lin} - P_{collision} - P_{smooth} - P_{orient} - P_{joint} - P_{deviation}$$

| 项 | 方向 | 公式 | 权重 |
|----|------|------|------|
| `distance_reward` | 正向 | 越近越大，到达阈值给成功奖励 `100.0` | `success_reward=100` |
| `linearity_reward` | 正向 | 奖励沿起点->目标直线的运动 `3.0/(1+error)` | `3.0` |
| `collision_penalty` | 负向 | 排除自碰撞，对外部碰撞计数惩罚 | `1.0` |
| `smooth_penalty` | 负向 | 惩罚相邻动作差异 `0.1 x ||delta_a||` | `0.1` |
| `orientation_penalty` | 负向 | 惩罚 ee z 轴偏离垂直向下 `0.3 x angle_err` | `0.3` |
| `joint_limit_penalty` | 负向 | 跨越关节限制时线性惩罚 | `0.5` |
| `deviation_penalty` | 负向 | 轨迹偏离历史最优直线路径 | `1.0` |

#### 终止条件

| 条件 | 信号 | 奖励 |
|------|------|------|
| `dist_to_goal < 0.005m` | `terminated=True` | `+50.0` |
| `step_count >= 500` | `truncated=True` | `-5.0` |

#### 高级特性

- **课程学习**：6级渐进目标距离 (0.2m -> 0.3m -> 0.4m -> 0.5m -> 0.6m -> 0.7m)，成功率 > 80% 自动升级
- **域随机化**：关节阻尼随机倍乘 [0.5, 2.0]，提升泛化性
- **EvalCallback**：训练中自动评估并保存最佳模型
- **CheckpointCallback**：定期保存检查点
- **线性学习率衰减**：训练过程中 LR 从 3e-4 线性衰减到 3e-6

---

### 3.2 Obstacle Avoidance（单臂避障）

**文件**：[`rl_panda_obstacle_high_profile.py`](rl_panda_obstacle_high_profile.py)

**MuJoCo 场景**：[`model/franka_emika_panda/scene_pos_with_obstacles.xml`](model/franka_emika_panda/scene_pos_with_obstacles.xml)

单臂 Panda 机械臂在工作空间内到达随机目标点，同时避开场景中的障碍球。

#### 观测空间（27维）

$$O = [q_{1:7},\ \dot{q}_{1:7},\ p_{ee},\ p_{goal} - p_{ee},\ p_{obs} - p_{ee},\ d_{obs},\ p_{obs}]$$

| 索引 | 内容 | 维度 | 物理含义 |
|------|------|------|----------|
| 0-6 | `joint_pos` | 7 | 关节角度 (rad) |
| 7-13 | `joint_vel` | 7 | 关节角速度 (rad/s) |
| 14-16 | `ee_pos` | 3 | 末端执行器三维位置 |
| 17-19 | `ee_to_goal` | 3 | ee -> goal 向量 |
| 20-22 | `ee_to_obstacle` | 3 | ee -> obstacle 向量 |
| 23 | `dist_to_obstacle` | 1 | ee 到障碍物表面的距离 |
| 24-26 | `obstacle_pos` | 3 | 障碍物三维位置 |

#### 动作空间（7维）

$$a \in [-1, 1]^7 \quad \text{(absolute control)}$$

$$q_{target}^i = q_{min}^i + \frac{a_i + 1}{2} \times (q_{max}^i - q_{min}^i)$$

#### 奖励函数

$$R = R_{dist} - P_{collision} - P_{smooth} - P_{joint} - P_{step}$$

| 奖励项 | 方向 | 公式 |
|----|------|------|
| `distance_reward` | 正向 | 分段函数：阈值内 `40.0->5.0`，阈值外 `1/(1+d)` |
| `collision_penalty` | 负向 | 检测到碰撞 `-10.0` 并立即终止 |
| `smooth_penalty` | 负向 | `0.001 x ||delta_a||` |
| `joint_penalty` | 负向 | 超出关节范围 `0.5 x error` |
| `step_penalty` | 负向 | `0.001 x step_count` 步数惩罚替代时间惩罚 |

#### 终止条件

| 条件 | 信号 | 奖励 |
|------|------|------|
| `dist_to_goal < 0.005m` | `terminated=True` | `+50.0` |
| 碰撞 | `terminated=True` | `-10.0` |
| `step_count >= 500` | `truncated=True` | `-5.0` |

#### 关键特性

- 单文件自包含实现（内联环境类 + 训练/测试函数）
- 绝对位置控制，适合需要大范围运动场景
- 碰撞检测排除自碰撞和地面接触
- 每步执行 1 次 `mj_step`（无子步）

---

### 3.3 Dual Arm Obstacle（双臂避障）

**文件**：[`rl_dual_arm_obstacle.py`](rl_dual_arm_obstacle.py)

**MuJoCo 场景**：[`model/franka_emika_panda/scene_dual_arm_with_obstacles.xml`](model/franka_emika_panda/scene_dual_arm_with_obstacles.xml)

左右两台 Panda 机械臂分别放置于 `(0, 0.4, 0)` 和 `(0, -0.4, 0)`，各自到达空间中的随机目标点，同时避开障碍物。**这是最具挑战性的任务**——14维动作空间和50维观测空间，需要协调双臂运动。

#### 观测空间（50维）

$$O = [q_L, q_R, \dot{q}_L, \dot{q}_R, p_L, p_R, \Delta p_L, \Delta p_R, \Delta o_L, \Delta o_R, d_{obs}^{min}, p_{obs}]$$

| 索引 | 内容 | 维度 | 物理含义 |
|------|------|------|----------|
| 0-6 | `left_joint_pos` | 7 | 左臂关节角度 |
| 7-13 | `right_joint_pos` | 7 | 右臂关节角度 |
| 14-20 | `left_joint_vel` | 7 | 左臂关节角速度 |
| 21-27 | `right_joint_vel` | 7 | 右臂关节角速度 |
| 28-30 | `left_ee_pos` | 3 | 左臂末端位置 |
| 31-33 | `right_ee_pos` | 3 | 右臂末端位置 |
| 34-36 | `left_ee_to_goal` | 3 | 左臂末端->目标 |
| 37-39 | `right_ee_to_goal` | 3 | 右臂末端->目标 |
| 40-42 | `left_ee_to_obs` | 3 | 左臂末端->障碍物 |
| 43-45 | `right_ee_to_obs` | 3 | 右臂末端->障碍物 |
| 46 | `min_dist_obs` | 1 | 双臂到障碍物的最小距离 |
| 47-49 | `obstacle_position` | 3 | 障碍物三维位置 |

#### 动作空间（14维）

$$a = [a_{left}^1, ..., a_{left}^7, a_{right}^1, ..., a_{right}^7] \in [-1, 1]^{14}$$

使用**增量位置控制**（delta control），`action_scale = 0.05`：

$$q_{target}^i = q_{current}^i + a_i \times 0.05 \times (q_{max}^i - q_{min}^i)$$

$$q_{target}^i = \text{clip}(q_{target}^i, q_{min}^i, q_{max}^i)$$

#### 奖励函数

$$R = \underbrace{-1.0 \cdot (d_{left} + d_{right})}_{距离塑形} + \underbrace{10.0 \cdot (\Delta d_{left} + \Delta d_{right})}_{进度奖励} - \underbrace{0.001 \cdot (||\dot{q}_L||^2 + ||\dot{q}_R||^2)}_{速度惩罚} - \underbrace{0.01 \cdot ||\delta a||}_{平滑惩罚} - \underbrace{20.0 \cdot \mathbb{1}_{collision}}_{碰撞惩罚} - \underbrace{P_{joint}}_{关节惩罚} - \underbrace{0.01}_{步数惩罚}$$

| 奖励项 | 方向 | 公式 | 权重/说明 |
|--------|------|------|----------|
| 距离塑形 | - | `-1.0 x (left_dist + right_dist)` | 提供连续的梯度信号 |
| 进度奖励 | + | `10.0 x (prev_dist - curr_dist)` | 鼓励每一步靠近目标 |
| 速度惩罚 | - | `0.001 x sum(vel^2)` | 抑制机械臂剧烈运动 |
| 平滑惩罚 | - | `0.01 x ||delta_a||` | 鼓励平滑的动作序列 |
| 碰撞惩罚 | - | `-20.0` 立即终止 | 避障核心约束 |
| 关节限制惩罚 | - | `0.5 x overshoot` | 防止超出关节范围 |

#### 到达奖励

| 条件 | 奖励 | 终止 |
|------|------|------|
| 双臂都到达 | `+200.0` | 是 |
| 仅左臂到达 | `+50.0` | 否（继续） |
| 仅右臂到达 | `+50.0` | 否（继续） |
| 碰撞 | `-20.0` | 是 |
| 超时 (>=1500步) | `-10.0` | 是 |

#### 关键设计

| 参数 | 值 | 设计原因 |
|------|------|----------|
| `max_episode_steps` | **1500** | 双臂任务需要更长探索时间（15秒仿真） |
| `n_substeps` | **5** | PD 控制器需要足够时间平滑响应 |
| `action_scale` | **0.05** | 每步最大移动5%关节范围，确保运动平滑 |
| `goal_arrival_threshold` | **0.02m** | 对双臂学习型智能体更合理（单臂用0.005m） |

#### 训练配置

| 参数 | 值 |
|------|------|
| 并行环境数 | 24 |
| 总训练步数 | 80,000,000 |
| n_steps | 4096 |
| batch_size | 256 |
| learning_rate | 1e-4 |
| ent_coef | 0.05 |
| 网络架构 (Actor) | [512, 256, 128] |
| 网络架构 (Critic) | [512, 512, 256] |

---

## 4. 代码结构

### 4.1 项目目录

```
mujoco-learning/
├── rl_panda_reach_target_high_profile.py   # Reach Target 主入口（模块化）
├── rl_panda_obstacle_high_profile.py       # Obstacle Avoidance 主入口（单体）
├── rl_dual_arm_obstacle.py                 # Dual Arm Obstacle 主入口（单体）
├── src/
│   ├── envs/
│   │   ├── panda_reach_env.py              # PandaReachEnv 环境类
│   │   └── reward.py                       # ReachTargetReward 模块化奖励
│   └── training/
│       ├── config.py                       # EnvConfig / PPOConfig / TrainConfig
│       └── callbacks.py                    # Checkpoint + Eval + LR 回调
├── model/
│   └── franka_emika_panda/
│       ├── scene.xml                       # 基础场景（Reach Target 使用）
│       ├── scene_pos_with_obstacles.xml    # 单臂避障场景（Obstacle 使用）
│       ├── scene_dual_arm_with_obstacles.xml # 双臂避障场景（Dual Arm 使用）
│       ├── panda.xml                       # 机器人本体（general actuator）
│       ├── panda_pos.xml                   # 机器人本体（position actuator）
│       ├── panda_dual_arm_pos.xml          # 双臂机器人本体
│       └── assets/                         # 网格和纹理文件
├── assets/model/
│   ├── rl_reach_target_checkpoint/         # Reach Target 训练好的模型
│   ├── rl_obstacle_avoidance_checkpoint/   # Obstacle Avoidance 训练好的模型
│   └── rl_dual_arm_checkpoint/             # Dual Arm 训练好的模型
└── requirements.txt
```

### 4.2 Reach Target 模块关系

```
rl_panda_reach_target_high_profile.py (主入口)
├── src/envs/panda_reach_env.py          [PandaReachEnv]
│   └── src/envs/reward.py               [ReachTargetReward]
├── src/training/config.py               [TrainConfig, EnvConfig, PPOConfig]
└── src/training/callbacks.py            [create_callbacks]
    ├── stable_baselines3 CheckpointCallback
    ├── stable_baselines3 EvalCallback
    └── LinearLRScheduleCallback (自定义)
```

### 4.3 MuJoCo 模型层级

```
model/franka_emika_panda/
├── panda.xml                          # 机器人本体定义（general actuator）
├── panda_pos.xml                      # 机器人本体（position actuator, kp/kv）
├── panda_dual_arm_pos.xml             # 双臂机器人本体
├── scene.xml                          # 基础场景 <- Reach Target 使用
├── scene_pos_with_obstacles.xml       # 单臂+障碍球 <- Obstacle 使用
└── scene_dual_arm_with_obstacles.xml  # 双臂+障碍球 <- Dual Arm 使用
```

---

## 5. 关键实现详解

### 5.1 MuJoCo 仿真控制

```python
# 位置控制：通过 PD 控制器驱动关节
# 在 panda_pos.xml 中定义：
# <position kp="800" kv="80" ctrlrange="-6.28 6.28" gear="1.0"/>

# 设置目标关节位置
self.data.ctrl[:7] = target_joint_positions

# 执行物理仿真
mujoco.mj_step(self.model, self.data)
```

### 5.2 PPO 模型配置

```python
model = PPO(
    policy="MlpPolicy",          # MLP 策略网络
    env=env,
    n_steps=2048,                # 每次 rollout 步数
    batch_size=256,              # mini-batch 大小
    n_epochs=10,                 # 每次 rollout 更新轮数
    gamma=0.99,                  # 折扣因子
    gae_lambda=0.95,             # GAE lambda 参数
    learning_rate=1e-4,          # 学习率
    ent_coef=0.01,               # 熵系数（鼓励探索）
    clip_range=0.2,              # PPO clip 范围
    max_grad_norm=0.5,           # 梯度裁剪
    normalize_advantage=True,    # 优势函数归一化
    device="cuda",               # GPU 训练
)
```

### 5.3 VecNormalize 归一化

```python
# 训练时：同时归一化观测和奖励
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

# 测试时必须加载训练时的统计信息
eval_env = VecNormalize.load(vec_normalize_path, eval_env)
eval_env.training = False      # 冻结统计更新
eval_env.norm_reward = False   # 不归一化奖励

# 测试时手动归一化观测
obs = eval_env.normalize_obs(obs)
```

### 5.4 碰撞检测

```python
def _check_collision(self) -> bool:
    for i in range(self.data.ncon):
        contact = self.data.contact[i]
        body1 = self.model.geom_bodyid[contact.geom1]
        body2 = self.model.geom_bodyid[contact.geom2]
        # 排除自碰撞（同一 body）和地面碰撞（body=0）
        if body1 != body2 and body1 != 0 and body2 != 0:
            return True
    return False
```

---

## 6. 配置参数与调优指南

### 6.1 环境配置

| 参数 | Reach Target | Obstacle | Dual Arm | 调优建议 |
|------|:---:|:---:|:---:|----------|
| `goal_threshold` | 0.005 | 0.005 | **0.02** | 越大越易成功但精度降低 |
| `max_episode_steps` | 500 | 500 | **1500** | 增大提高探索空间但减慢训练 |
| `n_substeps` | 5 | 1 | **5** | 增大提高仿真精度但增加计算量 |
| `action_type` | delta | absolute | **delta** | delta 更平滑，absolute 更快 |
| `action_scale` | 0.1 | N/A | **0.05** | 越小越平滑但到达越慢 |
| `curriculum` | True | 否 | 否 | 推荐开启，显著提升收敛速度 |

### 6.2 PPO 配置

| 参数 | Reach Target | Obstacle | Dual Arm | 调优建议 |
|------|:---:|:---:|:---:|----------|
| `n_envs` | 24 | 24 | 24 | 设为 CPU 核数 |
| `total_timesteps` | 100M | 60M | 80M | 任务越复杂需要越多 |
| `n_steps` | 2048 | 2048 | **4096** | 长回合任务需要更多步骤 |
| `learning_rate` | 3e-4 | 3e-4 | **1e-4** | 复杂任务降学习率更稳定 |
| `ent_coef` | 0.01 | 0.02 | **0.05** | 更大 -> 更多探索 |
| `batch_size` | 256 | 256 | 256 | `n_steps x n_envs / 8` 左右 |
| `net_arch_pi` | [256,256,128] | [256,256,128] | **[512,256,128]** | 任务越复杂网络越大 |
| `net_arch_vf` | [512,256,128] | [512,256,128] | **[512,512,256]** | Critic 应比 Actor 深 |

### 6.3 VecNormalize 配置

```python
# 训练时
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

# 评估时：加载训练统计，关闭更新
eval_env = VecNormalize.load(vec_normalize_path, eval_env)
eval_env.training = False      # 冻结统计更新
eval_env.norm_reward = False   # 不归一化奖励
```

### 6.4 训练加速技巧

| 方法 | 效果 |
|------|------|
| 关闭 viewer (`visualize=False`) | **10-100x** |
| `n_envs` 调整为 CPU 核数 | **2-3x** |
| 减少 `n_substeps` (如 5->3) | **1.5x**（降低物理精度） |
| 减小 `max_episode_steps` | **2x**（减少无效探索） |

---

## 7. 运行步骤

### 7.1 环境安装

```bash
# 克隆仓库
git clone https://github.com/fatu08/mujoco-learning.git
cd mujoco-learning

# 创建 conda 环境
conda create -n arm_mujoco_env python=3.11
conda activate arm_mujoco_env

# 安装核心依赖
pip install numpy mujoco gymnasium stable-baselines3 torch tensorboard

# 或从 requirements.txt 安装
pip install -r requirements.txt
```

### 7.2 Reach Target 任务

```bash
cd /home/fatu08/mujoco-learning

# 训练新模型
python3 rl_panda_reach_target_high_profile.py
# 修改 TRAIN_MODE = True

# 测试已有模型
# 修改 TRAIN_MODE = False，设置正确的 model_path

# 监控训练进度
tensorboard --logdir=./tensorboard/panda_reach_target_v3/
```

### 7.3 Obstacle Avoidance 任务

```bash
cd /home/fatu08/mujoco-learning

# 训练新模型
python3 rl_panda_obstacle_high_profile.py
# TRAIN_MODE = True

# 测试已有模型
# TRAIN_MODE = False，设置正确的 MODEL_PATH

# 监控训练进度
tensorboard --logdir=./tensorboard/panda_obstacle_avoidance/
```

### 7.4 Dual Arm Obstacle 任务

```bash
cd /home/fatu08/mujoco-learning

# 训练
python rl_dual_arm_obstacle.py
# 修改 TRAIN_MODE = True

# 测试（会弹出 MuJoCo Viewer 3D 可视化窗口）
# 修改 TRAIN_MODE = False
python rl_dual_arm_obstacle.py
```

### 7.5 预训练模型路径

| 任务 | 模型路径 |
|------|----------|
| Reach Target | `assets/model/rl_reach_target_checkpoint/panda_ppo_reach_target_v2` |
| Obstacle Avoidance | `assets/model/rl_obstacle_avoidance_checkpoint/panda_obstacle_avoidance_v6` |
| Dual Arm | `assets/model/rl_dual_arm_checkpoint/dual_arm_obstacle_v1` |

### 7.6 恢复训练

```python
# Reach Target
config.resume_from = "panda_ppo_reach_target_v3"

# Obstacle
train_ppo(
    n_envs=24,
    total_timesteps=30_000_000,
    model_save_path="...",
    visualize=False,
    resume_from="assets/model/.../panda_obstacle_avoidance_v5"
)

# Dual Arm
train_ppo(resume_from="assets/model/rl_dual_arm_checkpoint/dual_arm_obstacle_v1")
```

程序会自动检查 `{resume_from}_vecnormalize.pkl` 并加载归一化统计。

### 7.7 GPU 训练确认

```python
# 自动检测 CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"

# 手动指定
device = "cuda:0"  # 第一个 GPU
```

验证 GPU 正在使用：

```bash
watch -n 0.5 nvidia-smi
# 应看到 python 进程占用 GPU 显存
```

---

## 8. 三文件对比分析

### 8.1 架构对比

| 维度 | Reach Target | Obstacle Avoidance | Dual Arm |
|------|-------------|-------------------|----------|
| **文件数** | 5+（主脚本 + 4 模块） | 1（单体文件） | 1 |
| **环境类** | `src/envs/panda_reach_env.py` | 内联定义，197 行 | 内联定义 |
| **奖励类** | `src/envs/reward.py` (117行) | 内联 `_calc_reward()` | 内联 `_calc_reward()` |
| **配置** | `TrainConfig/EnvConfig/PPOConfig` dataclass | 函数参数 + 内联字典 | 函数参数 + 内联字典 |
| **Callbacks** | EvalCallback + CheckpointCallback + 自定义 LR | 无 | 无 |
| **模型缓存** | 类变量 `_cached_model` 共享 | 每个实例独立加载 | 每个实例独立加载 |

### 8.2 功能对比

| 功能 | Reach Target | Obstacle Avoidance | Dual Arm |
|------|:-----------:|:------------------:|:--------:|
| Delta 增量控制 | 是 | 否 | 是 |
| 课程学习 (6级) | 是 | 否 | 否 |
| 域随机化 (阻尼) | 是 | 否 | 否 |
| eval 独立进程 | 是 | 否 | 否 |
| 自动 checkpoint | 是 | 否 | 否 |
| 线性 LR 衰减 | 是 | 否 | 否 |
| 最佳模型自动保存 | 是 | 否 | 否 |
| VecNormalize | 是 | 是 | 是 |
| 碰撞检测排除自碰撞 | 是 | 是 | 是 |
| ee->目标向量 | 是 | 是 | 是 |
| ee->障碍物向量 | 不适用 | 是 | 是 |
| 障碍物距离 | 不适用 | 是 | 是 |
| Viewer 节流 | 是 (每5步) | 是 (每5步) | 是 (每5步) |
| terminate/truncate 正确使用 | 是 | 是 | 是 |

### 8.3 奖励函数对比

| 奖励项 | Reach Target | Obstacle Avoidance | Dual Arm |
|--------|:-----------:|:------------------:|:--------:|
| 距离塑形 | 是 `1/(1+d)` | 是 分段函数 | 是 `-1.0 x d` |
| 进度奖励 | 否 | 否 | 是 `10.0 x Delta_d` |
| 速度惩罚 | 否 | 否 | 是 `0.001 x sum(vel^2)` |
| 直线度奖励 | 是 `3.0/(1+e)` | 否 | 否 |
| 偏离惩罚 | 是 | 否 | 否 |
| 姿态惩罚 | 是 | 否 | 否 |
| 动作平滑 | 是 `0.1 x ||delta_a||` | 是 `0.005 x ||delta_a||` | 是 `0.01 x ||delta_a||` |
| 碰撞惩罚 | 是 `1.0/contact` | 是 `-10.0 x 2` | 是 `-20.0 x 2` |
| 关节限制 | 是 `0.5 x err` | 是 `0.5 x err` | 是 `0.5 x err` |
| 步数/时间惩罚 | 否 | 是 `0.001 x steps` | 是 `0.01` |

### 8.4 终止条件对比

| 条件 | Reach Target | Obstacle Avoidance | Dual Arm |
|------|:-----------:|:------------------:|:--------:|
| 到达阈值 | 0.005m | 0.005m | **0.02m** |
| 最大步数 | 500 | 500 | **1500** |
| 碰撞终止 | 继续（仅惩罚） | 是 | 是 |
| 到达奖励 | +50.0 | +50.0 | **+200.0** |
| 部分到达奖励 | N/A | N/A | **+50.0** (单臂) |
| 超时惩罚 | -5.0 | -5.0 | **-10.0** |

---

## 9. 常见问题与解决方案

### Q1: 训练时出现大量 Gymnasium 兼容层警告

```
UserWarning: You provided an OpenAI Gym environment.
```

**原因**: Stable-Baselines3 检测到旧版 Gym API

**解决**: 已修复。导入改为 `import gymnasium as gym`，并添加 `warnings.filterwarnings`。

---

### Q2: `net_arch` 格式警告

```
UserWarning: As shared layers in the mlp_extractor are removed since SB3 v1.8.0
```

**解决**: 将 `net_arch=[dict(...)]` 改为 `net_arch=dict(...)`。

---

### Q3: `ent_coef="auto"` 导致训练异常

**原因**: SB3 的 PPO **不支持** `"auto"` 熵系数（SAC 才支持），字符串会被当作 `0.0`。策略过早收敛到次优解。

**解决**: 使用数值，如 `ent_coef=0.01`。

---

### Q4: `_cached_model_path` AttributeError

**原因**: 类变量只定义了 `_cached_model`，未定义 `_cached_model_path`。

**解决**: 在类级别同时定义两个变量：
```python
class PandaReachEnv(gym.Env):
    _cached_model = None
    _cached_model_path = None
```

---

### Q5: `n_envs=64` 在 20 核 CPU 上性能下降

**原因**: 进程数远超 CPU 核心数，大量上下文切换。

**解决**: `n_envs = CPU 核心数` 或略高（如 `24`）。

---

### Q6: Viewer 测试时运行多轮后卡住

**原因**: `_render_scene()` 多次累积导致 MuJoCo viewer 内部状态不一致。

**解决**: 直接设 `ngeom=1`，不先置零；`sync()` 前先 `_render_scene()`，添加 step counter 节流。

---

### Q7: 碰撞惩罚误判自碰撞

**原因**: `self.data.ncon` 包含机器人自身关节之间的正常接触。

**解决**: 遍历 `contact` 数组，检查 `geom_bodyid`：
```python
if body1 != body2 and body1 != 0 and body2 != 0:  # 排除自碰撞和地面
```

---

### Q8: 测试时成功率远低于训练时的 EvalCallback

**原因**: 测试时没有加载 `VecNormalize` 统计，策略网络收到的是未归一化的原始观测（尺度完全不对）。

**解决**: 测试时加载 `_vecnormalize.pkl`，用 `eval_env.normalize_obs(obs)` 归一化观测。

---

### Q9: `mj_step` vs `mj_forward` 混淆

| 函数 | 作用 | 使用场景 |
|------|------|---------|
| `mj_step` | 完整物理步进 | `step()` 中执行仿真 |
| `mj_forward` | 只更新运动学 / 碰撞检测 | `reset()` 中初始化、修改物体位置后同步 |

---

### Q10: `terminated` vs `truncated` 的区别

| 信号 | 含义 | 对 Value Function 的影响 |
|------|------|------------------------|
| `terminated=True` | episode 自然结束（到达目标/碰撞） | `V(s_terminal) = 0` |
| `truncated=True` | episode 被截断（超时/超步数） | Bootstrap 估计 `V(s_truncated)` |

**正确使用**: 到达目标 -> `terminated=True`；步数超限 -> `truncated=True`。混淆会导致 Critic 学习错误的价值估计。

---

### Q11: NumPy / SciPy 版本警告

```
A NumPy version >=1.22.4 and <2.3.0 is required for this version of SciPy
```

**解决**: 降级 NumPy（不影响运行）：
```bash
pip install "numpy>=1.22.4,<2.3.0"
```

---

### Q12: 如何加速训练？

| 方法 | 效果 |
|------|------|
| 关闭 viewer (`visualize=False`) | **10-100x** |
| `n_envs` 调整为 CPU 核数 | **2-3x** |
| 减少 `n_substeps` (如 5->3) | **1.5x**（降低物理精度） |
| 减小 `max_episode_steps` | **2x**（减少无效探索） |
| 使用 GPU (`device="cuda"`) | GPU 利用率低，加速有限（MLP 模型小） |

---

## 贡献

欢迎贡献代码、提交 Issue 或参与讨论！

### 贡献方式

1. **提交 Bug**：创建 [GitHub Issue](https://github.com/fatu08/mujoco-learning/issues) 描述问题
2. **功能建议**：通过 Issue 提交想法
3. **Pull Request**：
   - Fork 本仓库
   - 创建功能分支 (`git checkout -b feature/amazing-feature`)
   - 提交更改 (`git commit -m 'Add amazing feature'`)
   - 推送到分支 (`git push origin feature/amazing-feature`)
   - 创建 Pull Request

---


### 参考项目

- [MuJoCo](https://mujoco.org/) -- 物理引擎
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) -- PPO 实现
- [Gymnasium](https://gymnasium.farama.org/) -- 强化学习接口
- [Franka Emika Panda 模型](https://github.com/google-deepmind/mujoco_menagerie) -- MuJoCo Menagerie

---

## License

本项目基于 **Apache License 2.0** 开源。

```
Copyright 2026 fatu08

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

Franka Emika Panda 机器人模型遵循 [Apache 2.0](model/franka_emika_panda/LICENSE) 协议。

---

## 致谢

- Google DeepMind 的 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 提供 Franka Panda 模型
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) 团队提供优秀的 RL 算法实现
- 开源项目：[LitchiCheng/mujoco-learning](https://github.com/LitchiCheng/mujoco-learning)
---

<div align="center">

**如果这个项目对你有帮助，请给一个 Star 支持！**

[![GitHub stars](https://img.shields.io/github/stars/fatu08/mujoco-learning?style=social)](https://github.com/fatu08/mujoco-learning)
[![GitHub forks](https://img.shields.io/github/forks/fatu08/mujoco-learning?style=social)](https://github.com/fatu08/mujoco-learning)


</div>
