from mjlab.tasks.registry import register_mjlab_task
from src.tasks.locomanipulation.rl import X2_LocomanipulationOnPolicyRunner

from .env_cfgs import (
  agibot_x2_locomanipulation_flat_env_cfg,
  agibot_x2_locomanipulation_rough_env_cfg,
)
from .rl_cfg import agibot_x2_locomanipulation_ppo_runner_cfg

register_mjlab_task(
  task_id="Agibot-X2-Locomanipulation-Rough",
  env_cfg=agibot_x2_locomanipulation_rough_env_cfg(),
  play_env_cfg=agibot_x2_locomanipulation_rough_env_cfg(play=True),
  rl_cfg=agibot_x2_locomanipulation_ppo_runner_cfg(),
  runner_cls=X2_LocomanipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Agibot-X2-Locomanipulation-Flat",
  env_cfg=agibot_x2_locomanipulation_flat_env_cfg(),
  play_env_cfg=agibot_x2_locomanipulation_flat_env_cfg(play=True),
  rl_cfg=agibot_x2_locomanipulation_ppo_runner_cfg(),
  runner_cls=X2_LocomanipulationOnPolicyRunner,
)
