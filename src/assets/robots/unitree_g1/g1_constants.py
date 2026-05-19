"""Unitree G1 constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

G1_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
)
assert G1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, G1_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Motor specs (from Unitree).
ROTOR_INERTIAS_5020 = (
  0.139e-4,
  0.017e-4,
  0.169e-4,
)
GEARS_5020 = (
  1,
  1 + (46 / 18),
  1 + (56 / 16),
)
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_5020, GEARS_5020
)

ROTOR_INERTIAS_7520_14 = (
  0.489e-4,
  0.098e-4,
  0.533e-4,
)
GEARS_7520_14 = (
  1,
  4.5,
  1 + (48 / 22),
)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ROTOR_INERTIAS_7520_22 = (
  0.489e-4,
  0.109e-4,
  0.738e-4,
)
GEARS_7520_22 = (
  1,
  4.5,
  5,
)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

ROTOR_INERTIAS_4010 = (
  0.068e-4,
  0.0,
  0.0,
)
GEARS_4010 = (
  1,
  5,
  5,
)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_4010, GEARS_4010
)

ACTUATOR_5020 = ElectricActuator(
  reflected_inertia=ARMATURE_5020,
  velocity_limit=37.0,
  effort_limit=25.0,
)
ACTUATOR_7520_14 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14,
  velocity_limit=32.0,
  effort_limit=88.0,
)
ACTUATOR_7520_22 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_22,
  velocity_limit=20.0,
  effort_limit=139.0,
)
ACTUATOR_4010 = ElectricActuator(
  reflected_inertia=ARMATURE_4010,
  velocity_limit=22.0,
  effort_limit=5.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

# Motor config per joint pattern: (effort_limit, armature).
# 4-bar linkage joints (waist pitch/roll, ankles) use doubled values.
_G1_MOTOR_CFGS: dict[str, tuple[float, float]] = {
  ".*_hip_pitch_joint": (ACTUATOR_7520_14.effort_limit, ACTUATOR_7520_14.reflected_inertia),
  ".*_hip_roll_joint": (ACTUATOR_7520_22.effort_limit, ACTUATOR_7520_22.reflected_inertia),
  ".*_hip_yaw_joint": (ACTUATOR_7520_14.effort_limit, ACTUATOR_7520_14.reflected_inertia),
  ".*_knee_joint": (ACTUATOR_7520_22.effort_limit, ACTUATOR_7520_22.reflected_inertia),
  ".*_ankle_pitch_joint": (ACTUATOR_5020.effort_limit * 2, ACTUATOR_5020.reflected_inertia * 2),
  ".*_ankle_roll_joint": (ACTUATOR_5020.effort_limit * 2, ACTUATOR_5020.reflected_inertia * 2),
  "waist_yaw_joint": (ACTUATOR_7520_14.effort_limit, ACTUATOR_7520_14.reflected_inertia),
  "waist_roll_joint": (ACTUATOR_5020.effort_limit * 2, ACTUATOR_5020.reflected_inertia * 2),
  "waist_pitch_joint": (ACTUATOR_5020.effort_limit * 2, ACTUATOR_5020.reflected_inertia * 2),
  ".*_shoulder_pitch_joint": (ACTUATOR_5020.effort_limit, ACTUATOR_5020.reflected_inertia),
  ".*_shoulder_roll_joint": (ACTUATOR_5020.effort_limit, ACTUATOR_5020.reflected_inertia),
  ".*_shoulder_yaw_joint": (ACTUATOR_5020.effort_limit, ACTUATOR_5020.reflected_inertia),
  ".*_elbow_joint": (ACTUATOR_5020.effort_limit, ACTUATOR_5020.reflected_inertia),
  ".*_wrist_roll_joint": (ACTUATOR_5020.effort_limit, ACTUATOR_5020.reflected_inertia),
  ".*_wrist_pitch_joint": (ACTUATOR_4010.effort_limit, ACTUATOR_4010.reflected_inertia),
  ".*_wrist_yaw_joint": (ACTUATOR_4010.effort_limit, ACTUATOR_4010.reflected_inertia),
}

# Named gain presets: joint_pattern -> (stiffness, damping).
# Edit these directly to tune gains. Add new presets as needed.
G1_GAIN_PRESETS: dict[str, dict[str, tuple[float, float]]] = {
  "default": {
    ".*_hip_pitch_joint": (STIFFNESS_7520_14, DAMPING_7520_14),
    ".*_hip_roll_joint": (STIFFNESS_7520_22, DAMPING_7520_22),
    ".*_hip_yaw_joint": (STIFFNESS_7520_14, DAMPING_7520_14),
    ".*_knee_joint": (STIFFNESS_7520_22, DAMPING_7520_22),
    ".*_ankle_pitch_joint": (STIFFNESS_5020 * 2, DAMPING_5020 * 2),
    ".*_ankle_roll_joint": (STIFFNESS_5020 * 2, DAMPING_5020 * 2),
    "waist_yaw_joint": (STIFFNESS_7520_14, DAMPING_7520_14),
    "waist_roll_joint": (STIFFNESS_5020 * 2, DAMPING_5020 * 2),
    "waist_pitch_joint": (STIFFNESS_5020 * 2, DAMPING_5020 * 2),
    ".*_shoulder_pitch_joint": (STIFFNESS_5020, DAMPING_5020),
    ".*_shoulder_roll_joint": (STIFFNESS_5020, DAMPING_5020),
    ".*_shoulder_yaw_joint": (STIFFNESS_5020, DAMPING_5020),
    ".*_elbow_joint": (STIFFNESS_5020, DAMPING_5020),
    ".*_wrist_roll_joint": (STIFFNESS_5020, DAMPING_5020),
    ".*_wrist_pitch_joint": (STIFFNESS_4010, DAMPING_4010),
    ".*_wrist_yaw_joint": (STIFFNESS_4010, DAMPING_4010),
  },
  # Add more presets here, e.g.:
  # "stiff": { ... },
  # "locomanipulation": { ... },
  "unitree": {
    ".*_hip_pitch_joint": (100, 2.5),
    ".*_hip_roll_joint": (100, 2.5),
    ".*_hip_yaw_joint": (100, 2.5),
    ".*_knee_joint": (200, 5.0),
    ".*_ankle_pitch_joint": (20, 0.2),
    ".*_ankle_roll_joint": (20, 0.1),
    "waist_yaw_joint": (200, 5.0),
    "waist_roll_joint": (1200, 5.0),
    "waist_pitch_joint": (1200, 5.0),
    ".*_shoulder_pitch_joint": (90, 2.0),
    ".*_shoulder_roll_joint": (60, 1.0),
    ".*_shoulder_yaw_joint": (20, 0.4),
    ".*_elbow_joint": (60, 1.0),
    ".*_wrist_roll_joint": (4, 0.2),
    ".*_wrist_pitch_joint": (4, 0.2),
    ".*_wrist_yaw_joint": (4, 0.2),
  },
  "unitree_stiff": {
    ".*_hip_pitch_joint": (100, 2.5),
    ".*_hip_roll_joint": (100, 2.5),
    ".*_hip_yaw_joint": (100, 2.5),
    ".*_knee_joint": (200, 5.0),
    ".*_ankle_pitch_joint": (20, 0.2),
    ".*_ankle_roll_joint": (20, 0.1),
    "waist_yaw_joint": (200, 5.0),
    "waist_roll_joint": (1200, 5.0),
    "waist_pitch_joint": (1200, 5.0),
    ".*_shoulder_pitch_joint": (60, 1.5),
    ".*_shoulder_roll_joint": (60, 1.5),
    ".*_shoulder_yaw_joint": (60, 1.5),
    ".*_elbow_joint": (60, 1.5),
    ".*_wrist_roll_joint": (60, 1.5),
    ".*_wrist_pitch_joint": (60, 1.5),
    ".*_wrist_yaw_joint": (60, 1.5),
  },
}


def _make_g1_actuators_and_scale(
  gains: dict[str, tuple[float, float]],
) -> tuple[tuple[BuiltinPositionActuatorCfg, ...], dict[str, float]]:
  """Build actuator configs and action scale from per-joint gains."""
  actuators: list[BuiltinPositionActuatorCfg] = []
  scale: dict[str, float] = {}
  for pattern, (stiffness, damping) in gains.items():
    effort_limit, armature = _G1_MOTOR_CFGS[pattern]
    actuators.append(BuiltinPositionActuatorCfg(
      target_names_expr=(pattern,),
      stiffness=stiffness,
      damping=damping,
      effort_limit=effort_limit,
      armature=armature,
    ))
    scale[pattern] = 0.25 * effort_limit / stiffness
  return tuple(actuators), scale


_DEFAULT_ACTUATORS, G1_ACTION_SCALE = _make_g1_actuators_and_scale(
  G1_GAIN_PRESETS["default"]
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.8),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.78),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=_DEFAULT_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_robot_cfg(
  preset: str = "default",
) -> tuple[EntityCfg, dict[str, float]]:
  """Get a G1 robot configuration with the named gain preset.

  Returns:
    (entity_cfg, action_scale) — both fresh instances.
  """
  gains = G1_GAIN_PRESETS[preset]
  actuators, action_scale = _make_g1_actuators_and_scale(gains)
  articulation = EntityArticulationInfoCfg(
    actuators=actuators,
    soft_joint_pos_limit_factor=0.9,
  )
  entity_cfg = EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=articulation,
  )
  return entity_cfg, action_scale


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot_cfg, _ = get_g1_robot_cfg()
  robot = Entity(robot_cfg)

  viewer.launch(robot.spec.compile())
