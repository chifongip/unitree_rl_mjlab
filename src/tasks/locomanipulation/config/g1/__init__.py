from mjlab.tasks.registry import register_mjlab_task
from src.tasks.locomanipulation.rl import LocomanipulationOnPolicyRunner

from .env_cfgs import (
  unitree_g1_locomanipulation_flat_env_cfg,
  unitree_g1_locomanipulation_rough_env_cfg,
)
from .rl_cfg import unitree_g1_locomanipulation_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-Locomanipulation-Rough",
  env_cfg=unitree_g1_locomanipulation_rough_env_cfg(),
  play_env_cfg=unitree_g1_locomanipulation_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_locomanipulation_ppo_runner_cfg(),
  runner_cls=LocomanipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Locomanipulation-Flat",
  env_cfg=unitree_g1_locomanipulation_flat_env_cfg(),
  play_env_cfg=unitree_g1_locomanipulation_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_locomanipulation_ppo_runner_cfg(),
  runner_cls=LocomanipulationOnPolicyRunner,
)
