import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import torch.nn as nn
import warnings
import torch
import mujoco.viewer
import time
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.vec_env.patch_gym")

import os

def write_flag_file(flag_filename="rl_dual_arm_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    try:
        with open(flag_path, "w") as f:
            f.write("This is a flag file")
        return True
    except Exception as e:
        return False

def check_flag_file(flag_filename="rl_dual_arm_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    return os.path.exists(flag_path)

def delete_flag_file(flag_filename="rl_dual_arm_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    if not os.path.exists(flag_path):
        return True
    try:
        os.remove(flag_path)
        return True
    except Exception as e:
        return False

class DualArmObstacleEnv(gym.Env):
    def __init__(self, visualize: bool = False):
        super(DualArmObstacleEnv, self).__init__()
        if not check_flag_file():
            write_flag_file()
            self.visualize = visualize
        else:
            self.visualize = False
        self.handle = None

        self.model = mujoco.MjModel.from_xml_path('./model/franka_emika_panda/scene_dual_arm_with_obstacles.xml')
        self.data = mujoco.MjData(self.model)

        if self.visualize:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3.0
            self.handle.cam.azimuth = 0.0
            self.handle.cam.elevation = -30.0
            self.handle.cam.lookat = np.array([0.2, 0.0, 0.4])

        self.np_random = np.random.default_rng(None)

        self.left_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'left_ee_center_body')
        self.right_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_ee_center_body')
        self.left_home = np.array(self.model.key_qpos[0][:9], dtype=np.float32)
        self.right_home = np.array(self.model.key_qpos[0][9:18], dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)

        self.obs_size = 7 + 7 + 7 + 7 + 3 + 3 + 3 + 3 + 3 + 3 + 1 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32)

        self.left_goal = np.array([0.4, 0.3, 0.4], dtype=np.float32)
        self.right_goal = np.array([0.4, -0.3, 0.4], dtype=np.float32)
        self.goal_arrival_threshold = 0.02
        self.goal_visu_size = 0.02
        self.left_goal_rgba = [0.1, 0.3, 0.3, 0.8]
        self.right_goal_rgba = [0.3, 0.1, 0.3, 0.8]

        self.obstacle_id = None
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name == "obstacle_0":
                self.obstacle_id = i
        self.obstacle_position = np.array(self.model.geom_pos[self.obstacle_id], dtype=np.float32)
        self.obstacle_size = self.model.geom_size[self.obstacle_id][0]

        self.left_joint_ids = []
        self.right_joint_ids = []
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and name.startswith("left_joint"):
                self.left_joint_ids.append(i)
            elif name and name.startswith("right_joint"):
                self.right_joint_ids.append(i)
        self.left_joint_ids.sort()
        self.right_joint_ids.sort()

        self.left_qpos_adr = self.model.jnt_qposadr[self.left_joint_ids]
        self.right_qpos_adr = self.model.jnt_qposadr[self.right_joint_ids]

        self.left_last_action = np.zeros(7, dtype=np.float32)
        self.right_last_action = np.zeros(7, dtype=np.float32)
        self._visu_step_counter = 0
        self._visu_sync_interval = 5
        self._step_count = 0
        self._max_episode_steps = 1500
        self._n_substeps = 5
        self._action_scale = 0.05
        self._left_prev_dist = 0.0
        self._right_prev_dist = 0.0

    def _render_scene(self) -> None:
        if not self.visualize or self.handle is None:
            return
        self.handle.user_scn.ngeom = 2
        mujoco.mjv_initGeom(
            self.handle.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.goal_visu_size, 0.0, 0.0],
            pos=self.left_goal,
            mat=np.eye(3).flatten(),
            rgba=np.array(self.left_goal_rgba, dtype=np.float32)
        )
        mujoco.mjv_initGeom(
            self.handle.user_scn.geoms[1],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.goal_visu_size, 0.0, 0.0],
            pos=self.right_goal,
            mat=np.eye(3).flatten(),
            rgba=np.array(self.right_goal_rgba, dtype=np.float32)
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:9] = self.left_home
        self.data.qpos[9:18] = self.right_home
        mujoco.mj_forward(self.model, self.data)

        self.left_goal = np.array([0.4, self.np_random.uniform(0.1, 0.5), 0.4], dtype=np.float32)
        self.right_goal = np.array([0.4, self.np_random.uniform(-0.5, -0.1), 0.4], dtype=np.float32)
        self.obstacle_position = np.array([0.3, self.np_random.uniform(-0.15, 0.15), 0.5], dtype=np.float32)
        self.model.geom_pos[self.obstacle_id] = self.obstacle_position
        mujoco.mj_forward(self.model, self.data)

        left_ee_pos = self.data.body(self.left_ee_id).xpos.copy()
        right_ee_pos = self.data.body(self.right_ee_id).xpos.copy()
        self._left_prev_dist = np.linalg.norm(left_ee_pos - self.left_goal)
        self._right_prev_dist = np.linalg.norm(right_ee_pos - self.right_goal)

        if self.visualize:
            self._render_scene()

        obs = self._get_observation()
        self._visu_step_counter = 0
        self._step_count = 0
        self.left_last_action = np.zeros(7, dtype=np.float32)
        self.right_last_action = np.zeros(7, dtype=np.float32)
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        left_jpos = self.data.qpos[self.left_qpos_adr].copy().astype(np.float32)
        right_jpos = self.data.qpos[self.right_qpos_adr].copy().astype(np.float32)
        left_jvel = np.zeros(7, dtype=np.float32)
        right_jvel = np.zeros(7, dtype=np.float32)
        for i, adr in enumerate(self.left_qpos_adr):
            left_jvel[i] = self.data.qvel[adr]
        for i, adr in enumerate(self.right_qpos_adr):
            right_jvel[i] = self.data.qvel[adr]

        left_ee = self.data.body(self.left_ee_id).xpos.copy().astype(np.float32)
        right_ee = self.data.body(self.right_ee_id).xpos.copy().astype(np.float32)
        left_to_goal = (self.left_goal - left_ee).astype(np.float32)
        right_to_goal = (self.right_goal - right_ee).astype(np.float32)
        left_to_obs = (self.obstacle_position - left_ee).astype(np.float32)
        right_to_obs = (self.obstacle_position - right_ee).astype(np.float32)
        dist_to_obs = np.array([np.linalg.norm(left_to_obs), np.linalg.norm(right_to_obs)], dtype=np.float32)
        min_dist_obs = np.array([np.min(dist_to_obs)], dtype=np.float32)

        return np.concatenate([
            left_jpos, right_jpos,
            left_jvel, right_jvel,
            left_ee, right_ee,
            left_to_goal, right_to_goal,
            left_to_obs, right_to_obs,
            min_dist_obs,
            self.obstacle_position,
        ])

    def _check_collision(self) -> bool:
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if body1 != body2 and body1 != 0 and body2 != 0:
                return True
        return False

    def _calc_reward(self, left_action: np.ndarray, right_action: np.ndarray) -> tuple[float, float, float, bool]:
        left_ee_pos = self.data.body(self.left_ee_id).xpos.copy()
        right_ee_pos = self.data.body(self.right_ee_id).xpos.copy()
        left_dist = np.linalg.norm(left_ee_pos - self.left_goal)
        right_dist = np.linalg.norm(right_ee_pos - self.right_goal)

        dist_reward = -1.0 * (left_dist + right_dist)

        left_progress = self._left_prev_dist - left_dist
        right_progress = self._right_prev_dist - right_dist
        progress_reward = 10.0 * (left_progress + right_progress)

        left_jvel = np.array([self.data.qvel[adr] for adr in self.left_qpos_adr])
        right_jvel = np.array([self.data.qvel[adr] for adr in self.right_qpos_adr])
        vel_penalty = 0.001 * (np.sum(left_jvel**2) + np.sum(right_jvel**2))

        smooth_penalty = 0.01 * (np.linalg.norm(left_action - self.left_last_action) +
                                  np.linalg.norm(right_action - self.right_last_action))

        left_jpos = self.data.qpos[self.left_qpos_adr]
        right_jpos = self.data.qpos[self.right_qpos_adr]
        joint_penalty = 0.0
        for jpos, jids in [(left_jpos, self.left_joint_ids), (right_jpos, self.right_joint_ids)]:
            for i, jid in enumerate(jids):
                min_a, max_a = self.model.jnt_range[jid]
                if jpos[i] < min_a:
                    joint_penalty += 0.5 * (min_a - jpos[i])
                elif jpos[i] > max_a:
                    joint_penalty += 0.5 * (jpos[i] - max_a)

        step_penalty = 0.01

        total_reward = (dist_reward
                        + progress_reward
                        - vel_penalty
                        - smooth_penalty
                        - joint_penalty
                        - step_penalty)

        self._left_prev_dist = left_dist
        self._right_prev_dist = right_dist
        self.left_last_action = left_action.copy()
        self.right_last_action = right_action.copy()

        return total_reward, left_dist, right_dist

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float32, bool, bool, dict]:
        left_action = action[:7]
        right_action = action[7:14]

        left_ranges = self.model.jnt_range[self.left_joint_ids]
        right_ranges = self.model.jnt_range[self.right_joint_ids]

        left_current = self.data.qpos[self.left_qpos_adr].copy()
        right_current = self.data.qpos[self.right_qpos_adr].copy()

        left_scaled = np.zeros(7, dtype=np.float32)
        right_scaled = np.zeros(7, dtype=np.float32)
        for i in range(7):
            left_range_width = left_ranges[i][1] - left_ranges[i][0]
            right_range_width = right_ranges[i][1] - right_ranges[i][0]
            left_scaled[i] = left_current[i] + left_action[i] * self._action_scale * left_range_width
            right_scaled[i] = right_current[i] + right_action[i] * self._action_scale * right_range_width
            left_scaled[i] = np.clip(left_scaled[i], left_ranges[i][0], left_ranges[i][1])
            right_scaled[i] = np.clip(right_scaled[i], right_ranges[i][0], right_ranges[i][1])

        self.data.ctrl[:7] = left_scaled
        self.data.ctrl[8:15] = right_scaled
        self.data.qpos[7:9] = [0.04, 0.04]
        self.data.qpos[16:18] = [0.04, 0.04]

        collision = False
        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)
            if not collision and self._check_collision():
                collision = True

        self._step_count += 1
        reward, left_dist, right_dist = self._calc_reward(left_action, right_action)
        terminated = False

        if collision:
            reward -= 20.0
            terminated = True

        left_arrived = left_dist < self.goal_arrival_threshold
        right_arrived = right_dist < self.goal_arrival_threshold

        if left_arrived and right_arrived:
            reward += 200.0
            terminated = True
            if self.visualize:
                print(f"[双臂成功] 左距离: {left_dist:.3f}, 右距离: {right_dist:.3f}")
        elif left_arrived:
            reward += 50.0
            if self.visualize:
                print(f"[左臂到达] 左距离: {left_dist:.3f}, 右距离: {right_dist:.3f}")
        elif right_arrived:
            reward += 50.0
            if self.visualize:
                print(f"[右臂到达] 左距离: {left_dist:.3f}, 右距离: {right_dist:.3f}")

        truncated = False
        if not terminated:
            if self._step_count >= self._max_episode_steps:
                reward -= 10.0
                truncated = True

        if self.visualize and self.handle is not None:
            self._visu_step_counter += 1
            if self._visu_step_counter % self._visu_sync_interval == 0:
                self._render_scene()
                self.handle.sync()
                time.sleep(0.01)

        obs = self._get_observation()
        info = {
            'is_success': not collision and left_arrived and right_arrived,
            'left_arrived': left_arrived,
            'right_arrived': right_arrived,
            'left_distance': left_dist,
            'right_distance': right_dist,
            'collision': collision,
            'step_count': self._step_count,
        }

        return obs, reward.astype(np.float32), terminated, truncated, info

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self) -> None:
        if self.visualize and self.handle is not None:
            self.handle.close()
            self.handle = None

def train_ppo(
    n_envs: int = 24,
    total_timesteps: int = 80_000_000,
    model_save_path: str = "dual_arm_ppo",
    visualize: bool = False,
    resume_from: Optional[str] = None
) -> None:

    ENV_KWARGS = {'visualize': visualize}

    env = make_vec_env(
        env_id=lambda: DualArmObstacleEnv(**ENV_KWARGS),
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"}
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    if resume_from is not None:
        model = PPO.load(resume_from, env=env)
        vec_normalize_path = resume_from + "_vecnormalize.pkl"
        if os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = True
            env.norm_reward = True
    else:
        POLICY_KWARGS = dict(
            activation_fn=nn.Tanh,
            net_arch=dict(
                pi=[512, 256, 128],
                vf=[512, 512, 256]
            ),
            ortho_init=True,
        )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=POLICY_KWARGS,
            verbose=1,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.05,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            normalize_advantage=True,
            learning_rate=1e-4,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log="./tensorboard/dual_arm_obstacle_avoidance/"
        )

    print(f"并行环境数: {n_envs}, 本次训练新增步数: {total_timesteps}")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )

    model.save(model_save_path)
    env.save(f"{model_save_path}_vecnormalize.pkl")
    env.close()
    print(f"模型已保存至: {model_save_path}")


def test_ppo(
    model_path: str = "dual_arm_ppo",
    total_episodes: int = 5,
) -> None:
    env = DualArmObstacleEnv(visualize=True)

    vec_normalize_path = model_path + "_vecnormalize.pkl"
    if os.path.exists(vec_normalize_path):
        eval_env = make_vec_env(
            env_id=lambda: DualArmObstacleEnv(visualize=False),
            n_envs=1,
        )
        eval_env = VecNormalize.load(vec_normalize_path, eval_env)
        eval_env.norm_reward = False
        eval_env.training = False
        model = PPO.load(model_path, env=eval_env)
    else:
        model = PPO.load(model_path, env=env)

    success_count = 0
    left_success_count = 0
    right_success_count = 0
    total_left_dist = 0.0
    total_right_dist = 0.0
    total_steps = 0
    print(f"测试轮数: {total_episodes}")

    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            if os.path.exists(vec_normalize_path):
                obs = eval_env.normalize_obs(obs)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        if info['is_success']:
            success_count += 1
        if info.get('left_arrived', False):
            left_success_count += 1
        if info.get('right_arrived', False):
            right_success_count += 1
        total_left_dist += info.get('left_distance', 0.0)
        total_right_dist += info.get('right_distance', 0.0)
        total_steps += info.get('step_count', 0)

        result = '双臂成功' if info['is_success'] else ('碰撞' if info.get('collision') else '未到达')
        print(f"轮次 {ep+1:2d} | 奖励: {episode_reward:6.2f} | 左距离: {info.get('left_distance',0):.4f} | 右距离: {info.get('right_distance',0):.4f} | 步数: {info.get('step_count',0):3d} | {result}")

    success_rate = (success_count / total_episodes) * 100
    left_rate = (left_success_count / total_episodes) * 100
    right_rate = (right_success_count / total_episodes) * 100
    avg_left = total_left_dist / total_episodes
    avg_right = total_right_dist / total_episodes
    avg_steps = total_steps / total_episodes
    print(f"双臂成功率: {success_rate:.1f}% | 左臂: {left_rate:.1f}% | 右臂: {right_rate:.1f}%")
    print(f"平均左距离: {avg_left:.4f} | 平均右距离: {avg_right:.4f} | 平均步数: {avg_steps:.1f}")

    env.close()


if __name__ == "__main__":
    TRAIN_MODE = True
    VISUALIZE_TRAIN = True
    if TRAIN_MODE:
        import os
        os.system("rm -rf /home/dar/mujoco-bin/mujoco-learning/tensorboard/dual_arm*")
    delete_flag_file()
    MODEL_PATH = "assets/model/rl_dual_arm_checkpoint/dual_arm_obstacle_v1"
    RESUME_MODEL_PATH = ""
    if TRAIN_MODE:
        train_ppo(
            n_envs=24,
            total_timesteps=80_000_000,
            model_save_path=MODEL_PATH,
            visualize=VISUALIZE_TRAIN,
            resume_from=None
        )
    else:
        test_ppo(
            model_path=MODEL_PATH,
            total_episodes=100,
        )
    os.system("date")