"""RL configuration for Unitree G1 AMP locomanipulation task."""

import os
from dataclasses import dataclass, field
from typing import List

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
)
from src.tasks.locomanipulation.config.g1.rl_cfg import SymmetryPpoAlgorithmCfg

_MOTION_DATA_DIR = os.path.normpath(
  os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "assets", "motions", "g1", "amp",
  )
)


@dataclass
class RslRlAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Extended runner config with AMP-specific parameters."""
  amp_reward_coef: float = 0.05
  amp_motion_files: str = ""
  amp_num_preload_transitions: int = 200000
  amp_task_reward_lerp: float = 0.9
  amp_discr_hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 256])
  min_normalized_std: List[float] = field(default_factory=lambda: [0.05] * 15)
  amp_body_names: tuple = ()
  amp_anchor_name: str = ""


def unitree_g1_locomanipulation_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  """Create AMP RL runner configuration for Unitree G1 locomanipulation task."""
  return RslRlAmpRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=SymmetryPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      class_name="AMPPPO",
    ),
    experiment_name="g1_locomanipulation_amp",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=10001,
    # AMP parameters
    amp_reward_coef=0.05,
    amp_motion_files=os.path.join(_MOTION_DATA_DIR, "WalkandRun"),
    amp_num_preload_transitions=200000,
    amp_task_reward_lerp=0.9,
    amp_discr_hidden_dims=[1024, 512, 256],
    min_normalized_std=[0.05] * 15,
    amp_body_names=(
      "pelvis",
      "left_hip_roll_link",
      "left_knee_link",
      "left_ankle_roll_link",
      "right_hip_roll_link",
      "right_knee_link",
      "right_ankle_roll_link",
    ),
    amp_anchor_name="pelvis",
  )
