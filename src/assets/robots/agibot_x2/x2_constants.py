"""Agibot X2 constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

X2_XML: Path = (
  SRC_PATH / "assets" / "robots" / "agibot_x2" / "xmls" / "x2_ultra_no_head.xml"
)
assert X2_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(X2_XML))


##
# Actuator config.
##
# Effort limits from XML actuatorfrcrange. Armature=0.03 from XML default class.
# Gains derived from G1 ratios: stiffness ≈ effort / 1.7, damping ≈ stiffness * 0.117.

X2_ACTUATOR_CFGS: dict[str, tuple[float, float, float, float]] = {
  # pattern: (effort_limit, stiffness, damping, armature)
  ".*_hip_pitch_joint": (118.0, 69.4, 8.1, 0.03),
  ".*_hip_roll_joint": (118.0, 69.4, 8.1, 0.03),
  ".*_hip_yaw_joint": (118.0, 69.4, 8.1, 0.03),
  ".*_knee_joint": (118.0, 69.4, 8.1, 0.03),
  ".*_ankle_pitch_joint": (36.0, 21.2, 2.5, 0.03),
  ".*_ankle_roll_joint": (24.0, 14.1, 1.7, 0.03),
  "waist_yaw_joint": (118.0, 69.4, 8.1, 0.03),
  "waist_pitch_joint": (48.0, 28.2, 3.3, 0.03),
  "waist_roll_joint": (48.0, 28.2, 3.3, 0.03),
  ".*_shoulder_pitch_joint": (36.0, 21.2, 2.5, 0.03),
  ".*_shoulder_roll_joint": (36.0, 21.2, 2.5, 0.03),
  ".*_shoulder_yaw_joint": (24.0, 14.1, 1.7, 0.03),
  ".*_elbow_joint": (24.0, 14.1, 1.7, 0.03),
  ".*_wrist_yaw_joint": (24.0, 14.1, 1.7, 0.03),
  ".*_wrist_pitch_joint": (2.2, 1.3, 0.15, 0.03),
  ".*_wrist_roll_joint": (2.2, 1.3, 0.15, 0.03),
}


def _make_x2_actuators_and_scale() -> (
  tuple[tuple[BuiltinPositionActuatorCfg, ...], dict[str, float]]
):
  """Build actuator configs and action scale from per-joint configs."""
  actuators: list[BuiltinPositionActuatorCfg] = []
  scale: dict[str, float] = {}
  for pattern, (effort, stiffness, damping, armature) in X2_ACTUATOR_CFGS.items():
    actuators.append(BuiltinPositionActuatorCfg(
      target_names_expr=(pattern,),
      stiffness=stiffness,
      damping=damping,
      effort_limit=effort,
      armature=armature,
    ))
    scale[pattern] = 0.25 * effort / stiffness
  return tuple(actuators), scale


_DEFAULT_ACTUATORS, X2_ACTION_SCALE = _make_x2_actuators_and_scale()

##
# Keyframe config.
##
# Standing pose adapted from G1. Pelvis at 0.68m (X2's nominal height).
# Bent knees and angled ankles for stable standing.

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.68),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    ".*_elbow_joint": -0.87,
    "left_shoulder_roll_joint": 0.1,
    "right_shoulder_roll_joint": -0.1,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot\d+_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot\d+_collision$": 1},
  friction={r"^(left|right)_foot\d+_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot\d+_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot\d+_collision$": 1},
  friction={r"^(left|right)_foot\d+_collision$": (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot\d+_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

X2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=_DEFAULT_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)


def get_agibot_x2_robot_cfg() -> tuple[EntityCfg, dict[str, float]]:
  """Get a fresh Agibot X2 robot configuration instance.

  Returns:
    (entity_cfg, action_scale) — both fresh instances.
  """
  actuators, action_scale = _make_x2_actuators_and_scale()
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

  robot_cfg, _ = get_agibot_x2_robot_cfg()
  robot = Entity(robot_cfg)

  viewer.launch(robot.spec.compile())
