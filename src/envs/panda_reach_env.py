import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import logging

from src.envs.reward import ReachTargetReward

logger = logging.getLogger(__name__)


class PandaReachEnv(gym.Env):
    _cached_model = None
    _cached_model_path = None

    def __init__(
        self,
        model_path: str = "./model/franka_emika_panda/scene.xml",
        visualize: bool = False,
        goal_threshold: float = 0.005,
        max_episode_steps: int = 500,
        n_substeps: int = 5,
        action_type: str = "delta",
        delta_action_scale: float = 0.1,
        curriculum: bool = False,
        domain_rand: bool = False,
        initial_joint_noise: float = 0.0,
        reward_kwargs: dict = None,
    ):
        super(PandaReachEnv, self).__init__()

        self.model_path = model_path
        self.visualize = visualize
        self.goal_threshold = goal_threshold
        self.max_episode_steps = max_episode_steps
        self.n_substeps = n_substeps
        self.action_type = action_type
        self.delta_action_scale = delta_action_scale
        self.curriculum = curriculum
        self.domain_rand = domain_rand
        self.initial_joint_noise = initial_joint_noise

        reward_kwargs = reward_kwargs or {}
        reward_kwargs.setdefault("goal_threshold", goal_threshold)
        self.reward_fn = ReachTargetReward(**reward_kwargs)

        if PandaReachEnv._cached_model is None or PandaReachEnv._cached_model_path != model_path:
            try:
                PandaReachEnv._cached_model = mujoco.MjModel.from_xml_path(model_path)
                PandaReachEnv._cached_model_path = model_path
            except Exception as e:
                logger.error(f"模型加载失败: {model_path}, 错误: {e}")
                raise

        self.model = PandaReachEnv._cached_model
        self.data = mujoco.MjData(self.model)
        self.handle = None

        if self.visualize:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3.0
            self.handle.cam.azimuth = 0.0
            self.handle.cam.elevation = -30.0
            self.handle.cam.lookat = np.array([0.2, 0.0, 0.4])

        self.end_effector_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_center_body"
        )
        self.home_joint_pos = np.array(
            [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4],
            dtype=np.float32,
        )

        self.workspace = {"x": [-0.5, 0.8], "y": [-0.5, 0.5], "z": [0.05, 0.7]}
        self.curriculum_levels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        self.current_level = 0
        self.success_buffer = []

        n_joints = 7
        if self.action_type == "delta":
            self.action_space = spaces.Box(
                low=-self.delta_action_scale, high=self.delta_action_scale, shape=(n_joints,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32
            )

        self.obs_size = n_joints + n_joints + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )

        self.goal = np.zeros(3, dtype=np.float32)
        self.np_random = np.random.default_rng(None)
        self.prev_action = np.zeros(n_joints, dtype=np.float32)
        self.step_count = 0
        self.initial_ee_pos = np.zeros(3, dtype=np.float32)
        self.start_ee_pos = np.zeros(3, dtype=np.float32)

        self._visu_step_counter = 0
        self._visu_sync_interval = 5

        if self.domain_rand:
            self._original_damping = self.model.dof_damping[:n_joints].copy()

    def _get_valid_goal(self) -> np.ndarray:
        max_dist = self.curriculum_levels[self.current_level] if self.curriculum else 0.7
        min_dist = 0.1 if self.curriculum else 0.15
        for _ in range(1000):
            goal = self.np_random.uniform(
                low=[self.workspace["x"][0], self.workspace["y"][0], self.workspace["z"][0]],
                high=[self.workspace["x"][1], self.workspace["y"][1], self.workspace["z"][1]],
            )
            dist = np.linalg.norm(goal - self.initial_ee_pos)
            if min_dist < dist < max_dist and goal[0] > 0.1 and goal[2] > 0.05:
                return goal.astype(np.float32)
        return self.initial_ee_pos + np.array([0.3, 0.0, 0.1], dtype=np.float32)

    def _update_curriculum(self, success: bool):
        if not self.curriculum:
            return
        self.success_buffer.append(success)
        if len(self.success_buffer) > 50:
            self.success_buffer.pop(0)
        if len(self.success_buffer) >= 50:
            recent_rate = sum(self.success_buffer) / len(self.success_buffer)
            if recent_rate > 0.8 and self.current_level < len(self.curriculum_levels) - 1:
                self.current_level += 1
                self.success_buffer.clear()
                logger.info(f"课程学习升级到 Level {self.current_level}, 目标距离上限: {self.curriculum_levels[self.current_level]}")

    def _apply_domain_randomization(self):
        if not self.domain_rand:
            return
        n_joints = 7
        self.model.dof_damping[:n_joints] = self._original_damping * self.np_random.uniform(
            0.5, 2.0, size=n_joints
        )

    def _quat_to_z_axis(self, quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        return np.array(
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x ** 2 + y ** 2)],
            dtype=np.float32,
        )

    def _render_scene(self):
        if not self.visualize or self.handle is None:
            return
        self.handle.user_scn.ngeom = 1
        goal_rgba = np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32)
        mujoco.mjv_initGeom(
            self.handle.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.03, 0.0, 0.0],
            pos=self.goal,
            mat=np.eye(3).flatten(),
            rgba=goal_rgba,
        )

    def _get_observation(self) -> np.ndarray:
        joint_pos = self.data.qpos[:7].copy().astype(np.float32)
        joint_vel = self.data.qvel[:7].copy().astype(np.float32)
        ee_pos = self.data.body(self.end_effector_id).xpos.copy().astype(np.float32)
        ee_to_goal = (self.goal - ee_pos).astype(np.float32)
        return np.concatenate([joint_pos, joint_vel, ee_pos, ee_to_goal])

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        if self.initial_joint_noise > 0:
            noise = self.np_random.normal(0, self.initial_joint_noise, size=7).astype(np.float32)
            self.data.qpos[:7] = self.home_joint_pos + noise
        else:
            self.data.qpos[:7] = self.home_joint_pos

        mujoco.mj_forward(self.model, self.data)
        self.initial_ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        self.start_ee_pos = self.initial_ee_pos.copy()

        self._apply_domain_randomization()

        self.goal = self._get_valid_goal()
        self.reward_fn.reset()
        self.prev_action = np.zeros(7, dtype=np.float32)
        self.step_count = 0
        self._visu_step_counter = 0

        if self.visualize:
            self._render_scene()

        obs = self._get_observation()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple:
        if self.action_type == "delta":
            current_pos = self.data.qpos[:7].copy()
            target_pos = current_pos + action
            joint_ranges = self.model.jnt_range[:7]
            target_pos = np.clip(target_pos, joint_ranges[:, 0], joint_ranges[:, 1])
            self.data.ctrl[:7] = target_pos
        else:
            joint_ranges = self.model.jnt_range[:7]
            scaled_action = np.zeros(7, dtype=np.float32)
            for i in range(7):
                scaled_action[i] = (
                    joint_ranges[i][0] + (action[i] + 1) * 0.5 * (joint_ranges[i][1] - joint_ranges[i][0])
                )
            self.data.ctrl[:7] = scaled_action

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        ee_quat = self.data.body(self.end_effector_id).xquat.copy()
        ee_z_axis = self._quat_to_z_axis(ee_quat)
        joint_angles = self.data.qpos[:7].copy()

        reward, dist_to_goal = self.reward_fn.compute(
            ee_pos=ee_pos,
            ee_z_axis=ee_z_axis,
            joint_angles=joint_angles,
            action=action,
            prev_action=self.prev_action,
            start_ee_pos=self.start_ee_pos,
            goal=self.goal,
            mj_model=self.model,
            mj_data=self.data,
        )

        self.prev_action = action.copy()
        self.step_count += 1

        terminated = False
        if dist_to_goal < self.goal_threshold:
            terminated = True
            reward += 50.0

        truncated = False
        if self.step_count >= self.max_episode_steps:
            truncated = True
            reward -= 5.0

        self._update_curriculum(terminated and dist_to_goal < self.goal_threshold)

        if self.visualize and self.handle is not None:
            self._visu_step_counter += 1
            if self._visu_step_counter % self._visu_sync_interval == 0:
                self._render_scene()
                self.handle.sync()

        obs = self._get_observation()
        info = {
            "is_success": terminated and (dist_to_goal < self.goal_threshold),
            "distance_to_goal": dist_to_goal,
            "step_count": self.step_count,
        }

        return obs, np.float32(reward), terminated, truncated, info

    def close(self):
        if self.visualize and self.handle is not None:
            self.handle.close()
            self.handle = None
        logger.info("环境已关闭，资源释放完成")
