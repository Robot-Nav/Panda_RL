import os
import logging
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList

logger = logging.getLogger(__name__)


class CurriculumLoggingCallback(BaseCallback):
    def __init__(self, env_factory, verbose=0):
        super().__init__(verbose)
        self.env_factory = env_factory
        self._episode_rewards = []
        self._episode_lengths = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_lengths.append(info["episode"]["l"])
        return True


class LinearLRScheduleCallback(BaseCallback):
    def __init__(self, initial_lr: float, final_lr: float, total_steps: int, verbose=0):
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_steps = total_steps

    def _on_step(self):
        progress = min(self.num_timesteps / self.total_steps, 1.0)
        current_lr = self.initial_lr + (self.final_lr - self.initial_lr) * progress
        self.model.lr_schedule = lambda _: current_lr
        return True


def create_callbacks(config, eval_env=None):
    callbacks = []

    n_envs = config.ppo.n_envs
    checkpoint_freq = max(config.checkpoint_freq // n_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="./checkpoints/",
        name_prefix="panda_ppo_reach",
    )
    callbacks.append(checkpoint_callback)

    if eval_env is not None:
        eval_freq = max(config.eval_freq // n_envs, 1)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./best_model/",
            eval_freq=eval_freq,
            n_eval_episodes=config.eval_n_episodes,
            deterministic=True,
            verbose=1,
        )
        callbacks.append(eval_callback)

    if config.ppo.lr_schedule == "linear":
        lr_callback = LinearLRScheduleCallback(
            initial_lr=config.ppo.learning_rate,
            final_lr=config.ppo.learning_rate * 0.01,
            total_steps=config.ppo.total_timesteps,
        )
        callbacks.append(lr_callback)

    return CallbackList(callbacks)
