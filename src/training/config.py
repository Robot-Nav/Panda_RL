from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    model_path: str = "./model/franka_emika_panda/scene.xml"
    goal_threshold: float = 0.005
    max_episode_steps: int = 500
    n_substeps: int = 5
    action_type: str = "delta"
    delta_action_scale: float = 0.1
    curriculum: bool = True
    domain_rand: bool = False
    initial_joint_noise: float = 0.05
    visualize: bool = False
    reward_kwargs: dict = field(default_factory=dict)


@dataclass
class PPOConfig:
    n_envs: int = 24
    total_timesteps: int = 100_000_000
    learning_rate: float = 3e-4
    lr_schedule: str = "linear"
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    net_arch_pi: list = field(default_factory=lambda: [256, 256, 128])
    net_arch_vf: list = field(default_factory=lambda: [512, 256, 128])
    activation_fn: str = "tanh"
    ortho_init: bool = True
    tensorboard_log: str = "./tensorboard/panda_reach_target/"
    device: str = "auto"


@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    model_save_path: str = "panda_ppo_reach_target_v3"
    resume_from: Optional[str] = None
    checkpoint_freq: int = 500_000
    eval_freq: int = 100_000
    eval_n_episodes: int = 20
    n_eval_envs: int = 4
