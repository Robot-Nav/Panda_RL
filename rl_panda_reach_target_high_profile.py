import numpy as np
import mujoco
import torch
import torch.nn as nn
import warnings
import logging
import os
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from src.envs.panda_reach_env import PandaReachEnv
from src.training.config import TrainConfig, EnvConfig, PPOConfig
from src.training.callbacks import create_callbacks

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.vec_env.patch_gym")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def make_env(env_config: EnvConfig):
    def _init():
        return PandaReachEnv(
            model_path=env_config.model_path,
            visualize=env_config.visualize,
            goal_threshold=env_config.goal_threshold,
            max_episode_steps=env_config.max_episode_steps,
            n_substeps=env_config.n_substeps,
            action_type=env_config.action_type,
            delta_action_scale=env_config.delta_action_scale,
            curriculum=env_config.curriculum,
            domain_rand=env_config.domain_rand,
            initial_joint_noise=env_config.initial_joint_noise,
            reward_kwargs=env_config.reward_kwargs,
        )
    return _init


def train_ppo(config: TrainConfig) -> None:
    env_config = config.env
    ppo_config = config.ppo

    env_fns = [make_env(env_config) for _ in range(ppo_config.n_envs)]
    env = SubprocVecEnv(env_fns, start_method="fork")
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    eval_env_config = EnvConfig(**{**vars(env_config), "visualize": False})
    eval_env_fns = [make_env(eval_env_config) for _ in range(config.n_eval_envs)]
    eval_env = SubprocVecEnv(eval_env_fns, start_method="fork")
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_reward=10.0)

    activation_fn = nn.Tanh if ppo_config.activation_fn == "tanh" else nn.ReLU
    policy_kwargs = dict(
        activation_fn=activation_fn,
        net_arch=dict(pi=ppo_config.net_arch_pi, vf=ppo_config.net_arch_vf),
        ortho_init=ppo_config.ortho_init,
    )

    device = ppo_config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.resume_from is not None:
        logger.info(f"从检查点恢复训练: {config.resume_from}")
        model = PPO.load(config.resume_from, env=env, device=device)
        vec_normalize_path = config.resume_from + "_vecnormalize.pkl"
        if os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = True
            env.norm_reward = True
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=ppo_config.n_steps,
            batch_size=ppo_config.batch_size,
            n_epochs=ppo_config.n_epochs,
            gamma=ppo_config.gamma,
            gae_lambda=ppo_config.gae_lambda,
            learning_rate=ppo_config.learning_rate,
            clip_range=ppo_config.clip_range,
            ent_coef=ppo_config.ent_coef,
            vf_coef=ppo_config.vf_coef,
            max_grad_norm=ppo_config.max_grad_norm,
            normalize_advantage=ppo_config.normalize_advantage,
            device=device,
            tensorboard_log=ppo_config.tensorboard_log,
        )

    logger.info(
        f"训练配置: 并行环境数={ppo_config.n_envs}, "
        f"总步数={ppo_config.total_timesteps}, "
        f"设备={device}, "
        f"动作类型={env_config.action_type}, "
        f"课程学习={env_config.curriculum}, "
        f"域随机化={env_config.domain_rand}"
    )

    callbacks = create_callbacks(config, eval_env)

    model.learn(
        total_timesteps=ppo_config.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(config.model_save_path)
    env.save(f"{config.model_save_path}_vecnormalize.pkl")
    env.close()
    eval_env.close()
    logger.info(f"模型已保存至: {config.model_save_path}")


def test_ppo(
    model_path: str = "panda_ppo_reach_target_v3",
    total_episodes: int = 15,
    deterministic: bool = True,
) -> None:
    env = PandaReachEnv(visualize=True)

    vec_normalize_path = model_path + "_vecnormalize.pkl"
    if os.path.exists(vec_normalize_path):
        eval_env = make_vec_env(
            env_id=lambda: PandaReachEnv(visualize=False),
            n_envs=1,
        )
        eval_env = VecNormalize.load(vec_normalize_path, eval_env)
        eval_env.norm_reward = False
        eval_env.training = False
        model = PPO.load(model_path, env=eval_env)
    else:
        try:
            model = PPO.load(model_path, env=env)
        except Exception as e:
            logger.error(f"模型加载失败: {model_path}, 错误: {e}")
            env.close()
            return

    success_count = 0
    total_dist = 0.0
    total_steps = 0
    logger.info(f"测试轮数: {total_episodes}")

    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            if os.path.exists(vec_normalize_path):
                obs = eval_env.normalize_obs(obs)
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        is_success = info.get("is_success", False)
        dist = info.get("distance_to_goal", float("inf"))
        steps = info.get("step_count", 0)

        if is_success:
            success_count += 1
        total_dist += dist
        total_steps += steps

        logger.info(
            f"轮次 {ep + 1:2d} | 总奖励: {episode_reward:6.2f} | "
            f"距离: {dist:.4f} | 步数: {steps:3d} | "
            f"结果: {'成功' if is_success else '失败'}"
        )

    success_rate = (success_count / total_episodes) * 100
    avg_dist = total_dist / total_episodes
    avg_steps = total_steps / total_episodes
    logger.info(f"总成功率: {success_rate:.1f}% | 平均距离: {avg_dist:.4f} | 平均步数: {avg_steps:.1f}")

    env.close()


if __name__ == "__main__":
    config = TrainConfig(
        env=EnvConfig(
            model_path="./model/franka_emika_panda/scene.xml",
            visualize=False,
            curriculum=True,
            domain_rand=False,
            initial_joint_noise=0.05,
            action_type="delta",
            delta_action_scale=0.1,
            n_substeps=5,
            max_episode_steps=500,
        ),
        ppo=PPOConfig(
            n_envs=24,
            total_timesteps=100_000_000,
            learning_rate=3e-4,
            lr_schedule="linear",
            batch_size=256,
            ent_coef=0.01,
            net_arch_pi=[256, 256, 128],
            net_arch_vf=[512, 256, 128],
            activation_fn="tanh",
            ortho_init=True,
            tensorboard_log="./tensorboard/panda_reach_target_v3/",
        ),
        model_save_path="panda_ppo_reach_target_v3",
        resume_from=None,
        checkpoint_freq=500_000,
        eval_freq=100_000,
    )

    TRAIN_MODE = False

    if TRAIN_MODE:
        train_ppo(config)
    else:
        test_ppo(
            model_path="assets/model/rl_reach_target_checkpoint/panda_ppo_reach_target_v2",
            total_episodes=15,
        )
