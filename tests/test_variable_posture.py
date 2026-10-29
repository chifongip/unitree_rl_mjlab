"""Standing and moving height references for locomanipulation rewards."""

from types import SimpleNamespace

import pytest
import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from scripts.postures_x2_knees_out import HEIGHT_POSTURES as STANDING_POSTURES
from scripts.postures_x2_recorded import HEIGHT_POSTURES as WALKING_POSTURES
from src.tasks.locomanipulation.config.x2.env_cfgs import (
  HEIGHT_POSTURE_JOINT_NAMES,
  agibot_x2_locomanipulation_carry_finetune_flat_env_cfg,
  agibot_x2_locomanipulation_flat_env_cfg,
  agibot_x2_locomanipulation_rough_env_cfg,
)
from src.tasks.locomanipulation.mdp.rewards import stand_still, variable_posture


def _reward_env(commands, heights, names=("knee", "hip")):
  count = len(commands)
  # A noncontiguous selection catches accidental use of full robot joint order.
  asset_cfg = SceneEntityCfg("robot", joint_names=names)
  asset_cfg.joint_ids = [2, 0]
  asset = SimpleNamespace(
    data=SimpleNamespace(
      default_joint_pos=torch.zeros(count, 3), joint_pos=torch.zeros(count, 3),
    ),
    find_joints=lambda patterns: (asset_cfg.joint_ids, list(names)),
  )
  command_map = {"twist": torch.tensor(commands), "base_height": torch.tensor(heights)[:, None]}
  env = SimpleNamespace(
    device="cpu", scene={"robot": asset},
    command_manager=SimpleNamespace(get_command=command_map.get),
  )
  return env, asset_cfg


@pytest.mark.parametrize("height,standing,moving", [
  (0.31, [0.2, -0.3], [0.1, -0.4]),
  (0.59, [0.4, -0.6], [0.3, -0.8]),
])
def test_reference_switch_uses_command_speed_and_nearest_height(height, standing, moving):
  commands = [
    [0., 0., 0.], [0.0999, 0., 0.], [0.1, 0., 0.],
    [0.1001, 0., 0.], [1.5, 0., 0.], [0., -0.1, 0.],
    [0., 0., -0.1], [0.06, 0., 0.05],
  ]
  env, asset_cfg = _reward_env(commands, [height] * len(commands))
  params = dict(
    asset_cfg=asset_cfg, command_name="twist", walking_threshold=0.1,
    std_standing={"knee": 0.05, "hip": 0.1},
    std_walking={"knee": 0.5, "hip": 0.2},
    std_running={"knee": 0.6, "hip": 0.3},
    base_height_command_name="base_height",
    # Deliberately reverse dictionary insertion order relative to joint order.
    standing_height_postures={0.3: {"hip": -0.3, "knee": 0.2}, 0.6: {"hip": -0.6, "knee": 0.4}},
    walking_height_postures={0.32: {"hip": -0.4, "knee": 0.1}, 0.58: {"hip": -0.8, "knee": 0.3}},
  )
  reward = variable_posture(RewardTermCfg(func=variable_posture, weight=1.0, params=params), env)
  positions = env.scene["robot"].data.joint_pos
  targets = torch.tensor([standing] * 2 + [moving] * 6)
  positions[:, asset_cfg.joint_ids] = targets
  torch.testing.assert_close(reward(env, **params), torch.ones(len(commands)))
  # Swapping references must lower the reward in every regime.
  positions[:, asset_cfg.joint_ids] = torch.tensor([moving] * 2 + [standing] * 6)
  assert torch.all(reward(env, **params) < 1.)


@pytest.mark.parametrize("use_height_table", [False, True])
def test_omitted_walking_table_preserves_shared_reference(use_height_table):
  env, asset_cfg = _reward_env([[0., 0., 0.], [0.2, 0., 0.], [2., 0., 0.]], [0.4] * 3)
  params = dict(
    asset_cfg=asset_cfg, command_name="twist", walking_threshold=0.1,
    std_standing={".*": 0.1}, std_walking={".*": 0.2}, std_running={".*": 0.3},
    base_height_command_name="base_height",
    standing_height_postures={0.4: {"hip": -0.3, "knee": 0.2}} if use_height_table else None,
  )
  reward = variable_posture(RewardTermCfg(func=variable_posture, weight=1.0, params=params), env)
  if use_height_table:
    env.scene["robot"].data.joint_pos[:, asset_cfg.joint_ids] = torch.tensor([0.2, -0.3])
  torch.testing.assert_close(reward(env, **params), torch.ones(3))


@pytest.mark.parametrize("factory", [
  agibot_x2_locomanipulation_rough_env_cfg,
  agibot_x2_locomanipulation_flat_env_cfg,
  agibot_x2_locomanipulation_carry_finetune_flat_env_cfg,
])
@pytest.mark.parametrize("play", [False, True])
def test_x2_config_selects_standing_and_moving_tables(factory, play):
  cfg = factory(play=play)
  pose = cfg.rewards["pose"].params
  still = cfg.rewards["stand_still"].params
  assert pose["walking_threshold"] == still["command_threshold"] == 0.1
  def expected(table):
    return {
      h: {name: row[name] for name in HEIGHT_POSTURE_JOINT_NAMES}
      for h, row in table.items()
    }
  assert pose["standing_height_postures"] == still["standing_height_postures"] == expected(STANDING_POSTURES)
  assert pose["walking_height_postures"] == expected(WALKING_POSTURES)


def test_stand_still_uses_knees_out_and_gates_moving_commands():
  names = ("left_hip_roll_joint", "right_hip_roll_joint")
  env, asset_cfg = _reward_env([[0., 0., 0.], [0.1, 0., 0.]], [0.4, 0.4], names)
  params = dict(
    asset_cfg=asset_cfg, command_name="twist", command_threshold=0.1,
    base_height_command_name="base_height",
    standing_height_postures={h: {name: row[name] for name in names} for h, row in STANDING_POSTURES.items()},
  )
  reward = stand_still(RewardTermCfg(func=stand_still, weight=-1.0, params=params), env)
  positions = env.scene["robot"].data.joint_pos
  positions[:, asset_cfg.joint_ids] = torch.tensor([STANDING_POSTURES[0.4][name] for name in names])
  torch.testing.assert_close(reward(env, **params), torch.zeros(2))
  positions.zero_()
  penalty = reward(env, **params)
  assert penalty[0] > 0
  assert penalty[1] == 0
