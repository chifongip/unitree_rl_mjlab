"""AMP locomanipulation task configuration.

Extends the base locomanipulation config with AMP-specific observation group
and motion reset events.  Robot-specific configurations call the factory and
customize body names, motion paths, etc.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.locomanipulation import mdp
from src.tasks.locomanipulation.mdp import amp_observations
from src.tasks.locomanipulation.locomanipulation_env_cfg import (
  make_locomanipulation_env_cfg,
)


def make_locomanipulation_amp_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create AMP locomanipulation task configuration.

  Extends the base locomanipulation config with:
  - AMP observation group (body-level kinematics in anchor frame)
  - Motion loader startup event
  - Motion-based reset event
  """
  cfg = make_locomanipulation_env_cfg()

  ##
  # AMP Observations
  ##

  amp_terms = {
    "body_pos_b": ObservationTermCfg(
      func=amp_observations.robot_body_pos_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "body_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "body_ori_b": ObservationTermCfg(
      func=amp_observations.robot_body_ori_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),
        "body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    "body_lin_vel_b": ObservationTermCfg(
      func=amp_observations.robot_body_lin_vel_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),
        "body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    "body_ang_vel_b": ObservationTermCfg(
      func=amp_observations.robot_body_ang_vel_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),
        "body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
  }

  cfg.observations["amp"] = ObservationGroupCfg(
    terms=amp_terms,
    concatenate_terms=True,
    enable_corruption=False,
    history_length=1,
  )

  ##
  # AMP Events
  ##

  cfg.events["init_motion_loader"] = EventTermCfg(
    func=mdp.init_motion_loader,
    mode="startup",
    params={
      "motion_dir": "",  # Set per-robot.
    },
  )
  cfg.events["reset_from_motion"] = EventTermCfg(
    func=mdp.reset_from_motion_data,
    mode="reset",
    params={
      "motion_dir": "",  # Set per-robot (must match init_motion_loader).
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )

  return cfg
