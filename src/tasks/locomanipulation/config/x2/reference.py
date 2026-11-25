"""Original X2 carry grasp, also used by RoboJuDo-Plus."""

from src import SRC_PATH
from src.assets.robots.agibot_x2.x2_constants import X2_XML

from ..carry_reference import CarryPoseReference
from ..observer_reference import ObserverReference


X2_CARRY_POSE = {
  "left_shoulder_pitch_joint": -0.081,
  "left_shoulder_roll_joint": 0.026,
  "left_shoulder_yaw_joint": -0.049,
  "left_elbow_joint": -2.188,
  "left_wrist_yaw_joint": 0.238,
  "left_wrist_pitch_joint": 0.326,
  "left_wrist_roll_joint": 0.215,
  "right_shoulder_pitch_joint": -0.081,
  "right_shoulder_roll_joint": -0.026,
  "right_shoulder_yaw_joint": 0.049,
  "right_elbow_joint": -2.188,
  "right_wrist_yaw_joint": -0.238,
  "right_wrist_pitch_joint": 0.326,
  "right_wrist_roll_joint": -0.215,
}


# Frozen RoboJuDo-Plus simulation fit, independent of plant randomization.
X2_ARM_JOINT_NAMES = tuple(X2_CARRY_POSE)
X2_OBSERVER_REFERENCE = ObserverReference(
  joint_names=X2_ARM_JOINT_NAMES,
  inertia=(
    0.15532404980256578, 0.10078795654951321, 0.07674159569739615, 0.07182380882978089, 0.03, 0.03, 0.03,
    0.1553018849773343, 0.1012386139709888, 0.07677501375357826, 0.07193011707176135, 0.03, 0.03, 0.03,
  ),
  friction_limit=0.3,
  friction_viscosity=(
    106.80829994233635, 228.63155219753165, 29.852550817628547, 44.32684035243929,
    28.85973128906408, 31.97077929396376, 32.10787293174743, 106.63727466536166,
    228.8259861441458, 29.86332093192248, 44.346794114158094, 28.860217878180926,
    31.968164529938882, 32.102352113935815,
  ),
  bandwidth_rad_s=8.0,
)


X2_CARRY_REFERENCE = CarryPoseReference(
  xml=X2_XML,
  joint_count=29,
  arm_names=X2_ARM_JOINT_NAMES,
  hands=("left_wrist_roll_link", "right_wrist_roll_link"),
  pose=X2_CARRY_POSE,
  dataset=SRC_PATH / "assets/data/x2/bones_seed/carry_poses_x2.pkl",
  motion_dof_indices=tuple(range(15, 29)),
  supports_ground_parallel=True,
)
