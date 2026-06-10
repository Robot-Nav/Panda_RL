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

def write_flag_file(flag_filename="rl_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    try:
        with open(flag_path, "w") as f:
            f.write("This is a flag file")
        return True
    except Exception as e:
        return False

def check_flag_file(flag_filename="rl_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    return os.path.exists(flag_path)

def delete_flag_file(flag_filename="rl_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    if not os.path.exists(flag_path):
        return True
    try:
        os.remove(flag_path)
        return True
    except Exception as e:
        return False

class PandaObstacleEnv(gym.Env):
    def __init__(self, visualize: bool = False):
        super(PandaObstacleEnv, self).__init__()
        if not check_flag_file():
            write_flag_file()
            self.visualize = visualize
        else:
            self.visualize = False
        self.handle = None

        self.model = mujoco.MjModel.from_xml_path('./model/franka_emika_panda/scene_pos_with_obstacles.xml')
        self.data = mujoco.MjData(self.model)

        if self.visualize:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3.0
            self.handle.cam.azimuth = 0.0
            self.handle.cam.elevation = -30.0
            self.handle.cam.lookat = np.array([0.2, 0.0, 0.4])

        self.np_random = np.random.default_rng(None)

        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ee_center_body')
        self.home_joint_pos = np.array(self.model.key_qpos[0][:7], dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.obs_size = 7 + 7 + 3 + 3 + 3 + 1 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32)

        self.goal_position = np.array([0.4, -0.3, 0.4], dtype=np.float32)
        self.goal_arrival_threshold = 0.005
        self.goal_visu_size = 0.02
        self.goal_visu_rgba = [0.1, 0.3, 0.3, 0.8]

        self.obstacle_body_id = None
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name == "obstacle_0":
                self.obstacle_id_1 = i
                self.obstacle_body_id = self.model.geom_bodyid[i]
        self.obstacle_position = np.array(self.model.geom_pos[self.obstacle_id_1], dtype=np.float32)
        self.obstacle_size = self.model.geom_size[self.obstacle_id_1][0]

        self.last_action = self.home_joint_pos
        self._visu_step_counter = 0
        self._visu_sync_interval = 5
        self._step_count = 0
        self._max_episode_steps = 500
        self._prev_dist_to_goal = 0.0

    def _render_scene(self) -> None:
        if not self.visualize or self.handle is None:
            return
        self.handle.user_scn.ngeom = 1
        mujoco.mjv_initGeom(
            self.handle.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.goal_visu_size, 0.0, 0.0],
            pos=self.goal_position,
            mat=np.eye(3).flatten(),
            rgba=np.array(self.goal_visu_rgba, dtype=np.float32)
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.home_joint_pos
        self.data.qpos[7:] = [0.04, 0.04]
        mujoco.mj_forward(self.model, self.data)

        self.goal_position = np.array([self.goal_position[0], self.np_random.uniform(-0.3, 0.3), self.goal_position[2]], dtype=np.float32)
        self.obstacle_position = np.array([self.obstacle_position[0], self.np_random.uniform(-0.3, 0.3), self.obstacle_position[2]], dtype=np.float32)
        self.model.geom_pos[self.obstacle_id_1] = self.obstacle_position
        mujoco.mj_forward(self.model, self.data)

        ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        self._prev_dist_to_goal = np.linalg.norm(ee_pos - self.goal_position)

        if self.visualize:
            self._render_scene()

        obs = self._get_observation()
        self.start_t = time.time()
        self._visu_step_counter = 0
        self._step_count = 0
        self.last_action = self.home_joint_pos
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        joint_pos = self.data.qpos[:7].copy().astype(np.float32)
        joint_vel = self.data.qvel[:7].copy().astype(np.float32)
        ee_pos = self.data.body(self.end_effector_id).xpos.copy().astype(np.float32)
        ee_to_goal = (self.goal_position - ee_pos).astype(np.float32)
        ee_to_obstacle = (self.obstacle_position - ee_pos).astype(np.float32)
        dist_to_obstacle = np.array([np.linalg.norm(ee_to_obstacle)], dtype=np.float32)
        return np.concatenate([
            joint_pos,
            joint_vel,
            ee_pos,
            ee_to_goal,
            ee_to_obstacle,
            dist_to_obstacle,
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

    def _calc_reward(self, joint_angles: np.ndarray, action: np.ndarray) -> tuple[float, float, bool]:
        now_ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        dist_to_goal = np.linalg.norm(now_ee_pos - self.goal_position)

        progress = self._prev_dist_to_goal - dist_to_goal
        progress_reward = 50.0 * progress

        smooth_penalty = 0.005 * np.linalg.norm(action - self.last_action)

        collision = self._check_collision()
        collision_penalty = 10.0 if collision else 0.0

        joint_penalty = 0.0
        for i in range(7):
            min_angle, max_angle = self.model.jnt_range[:7][i]
            if joint_angles[i] < min_angle:
                joint_penalty += 0.5 * (min_angle - joint_angles[i])
            elif joint_angles[i] > max_angle:
                joint_penalty += 0.5 * (joint_angles[i] - max_angle)

        step_penalty = 0.01

        total_reward = (progress_reward
                        - collision_penalty
                        - smooth_penalty
                        - joint_penalty
                        - step_penalty)

        self._prev_dist_to_goal = dist_to_goal
        self.last_action = action.copy()

        return total_reward, dist_to_goal, collision

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float32, bool, bool, dict]:
        joint_ranges = self.model.jnt_range[:7]
        scaled_action = np.zeros(7, dtype=np.float32)
        for i in range(7):
            scaled_action[i] = joint_ranges[i][0] + (action[i] + 1) * 0.5 * (joint_ranges[i][1] - joint_ranges[i][0])
        self.data.ctrl[:7] = scaled_action
        self.data.qpos[7:] = [0.04, 0.04]
        mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        reward, dist_to_goal, collision = self._calc_reward(self.data.qpos[:7], action)
        terminated = False

        if collision:
            reward -= 10.0
            terminated = True

        if dist_to_goal < self.goal_arrival_threshold:
            reward += 50.0
            terminated = True
            if self.visualize:
                print(f"[成功] 距离目标: {dist_to_goal:.3f}, 奖励: {reward:.3f}")

        truncated = False
        if not terminated:
            if self._step_count >= self._max_episode_steps:
                reward -= 5.0
                truncated = True

        if self.visualize and self.handle is not None:
            self._visu_step_counter += 1
            if self._visu_step_counter % self._visu_sync_interval == 0:
                self._render_scene()
                self.handle.sync()
                time.sleep(0.01)

        obs = self._get_observation()
        info = {
            'is_success': not collision and terminated and (dist_to_goal < self.goal_arrival_threshold),
            'distance_to_goal': dist_to_goal,
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
    model_save_path: str = "panda_ppo_reach_target",
    visualize: bool = False,
    resume_from: Optional[str] = None
) -> None:

    ENV_KWARGS = {'visualize': visualize}

    env = make_vec_env(
        env_id=lambda: PandaObstacleEnv(** ENV_KWARGS),
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
                pi=[256, 256, 128],
                vf=[512, 256, 128]
            ),
            ortho_init=True,
        )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=POLICY_KWARGS,
            verbose=1,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.02,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            normalize_advantage=True,
            learning_rate=3e-4,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log="./tensorboard/panda_obstacle_avoidance/"
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
    model_path: str = "panda_obstacle_avoidance",
    total_episodes: int = 5,
) -> None:
    env = PandaObstacleEnv(visualize=True)

    vec_normalize_path = model_path + "_vecnormalize.pkl"
    if os.path.exists(vec_normalize_path):
        eval_env = make_vec_env(
            env_id=lambda: PandaObstacleEnv(visualize=False),
            n_envs=1,
        )
        eval_env = VecNormalize.load(vec_normalize_path, eval_env)
        eval_env.norm_reward = False
        eval_env.training = False
        model = PPO.load(model_path, env=eval_env)
    else:
        model = PPO.load(model_path, env=env)

    success_count = 0
    total_dist = 0.0
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
        total_dist += info.get('distance_to_goal', 0.0)
        total_steps += info.get('step_count', 0)
        print(f"轮次 {ep+1:2d} | 总奖励: {episode_reward:6.2f} | 距离: {info.get('distance_to_goal', 0.0):.4f} | 步数: {info.get('step_count', 0):3d} | 结果: {'成功' if info['is_success'] else '碰撞/失败'}")

    success_rate = (success_count / total_episodes) * 100
    avg_dist = total_dist / total_episodes
    avg_steps = total_steps / total_episodes
    print(f"总成功率: {success_rate:.1f}% | 平均距离: {avg_dist:.4f} | 平均步数: {avg_steps:.1f}")

    env.close()


if __name__ == "__main__":
    TRAIN_MODE = False
    if TRAIN_MODE:
        import os
        os.system("rm -rf /home/dar/mujoco-bin/mujoco-learning/tensorboard*")
    delete_flag_file()
    MODEL_PATH = "assets/model/rl_obstacle_avoidance_checkpoint/panda_obstacle_avoidance_v6"
    RESUME_MODEL_PATH = ""
    if TRAIN_MODE:
        train_ppo(
            n_envs=24,
            total_timesteps=60_000_000,
            model_save_path=MODEL_PATH,
            visualize=False,
            resume_from=None
        )
    else:
        test_ppo(
            model_path=MODEL_PATH,
            total_episodes=100,
        )
    os.system("date")