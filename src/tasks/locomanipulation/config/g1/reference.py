"""RoboJuDo-Plus d5cb94f G1 simulation observer and carry references."""

from src import SRC_PATH
from src.assets.robots.unitree_g1.g1_constants import G1_XML
from src.assets.robots.unitree_g1.g1_23dof_constants import G1_23DOF_XML

from ..carry_reference import CarryPoseReference
from ..observer_reference import ObserverReference


CARRY_29 = {
  "left_shoulder_pitch_joint": -0.11419255959405335,
  "left_shoulder_roll_joint": 0.27251687674442737,
  "left_shoulder_yaw_joint": 0.10410689202940233,
  "left_elbow_joint": -0.19539157984520328,
  "left_wrist_roll_joint": -0.29943273717757873,
  "left_wrist_pitch_joint": -0.14033675393208384,
  "left_wrist_yaw_joint": -0.05876427125423291,
  "right_shoulder_pitch_joint": -0.10820148059755065,
  "right_shoulder_roll_joint": -0.2735168993191418,
  "right_shoulder_yaw_joint": -0.10306724654180412,
  "right_elbow_joint": -0.1963821038834506,
  "right_wrist_roll_joint": 0.29971813739005054,
  "right_wrist_pitch_joint": -0.13789937278443318,
  "right_wrist_yaw_joint": 0.05800083238605073,
}

ARM_29 = tuple(CARRY_29)

G1_OBSERVER_REFERENCE_29 = ObserverReference(
  joint_names=ARM_29,
  inertia=(
    0.12860527766360236, 0.08197342060766341, 0.042021806329721426, 0.03591442046213406,
    0.01037, 0.01, 0.01, 0.12842812374598464,
    0.08193883941324785, 0.04205172550977063, 0.03592922927115913, 0.01037,
    0.01, 0.01,
  ),
  friction_viscosity=(
    54.397412283847785, 45.566913270335945, 24.466240880403888, 22.920117692001146,
    9.050174705830033, 11.102207997914238, 10.08360479789641, 54.3756049953329,
    45.56993876282425, 24.469906029843813, 22.90731944787912, 9.0501308295424,
    11.102028936320897, 10.083644323233575,
  ),
  friction_limit=0.1,
  bandwidth_rad_s=8.0,
)


CARRY_23 = {
  "left_shoulder_pitch_joint": -0.10538606454009122,
  "left_shoulder_roll_joint": 0.3110875633300908,
  "left_shoulder_yaw_joint": 0.10263418707300465,
  "left_elbow_joint": -0.35178299386885914,
  "left_wrist_roll_joint": -0.33279432595174296,
  "right_shoulder_pitch_joint": -0.09901032254672848,
  "right_shoulder_roll_joint": -0.31164459060494387,
  "right_shoulder_yaw_joint": -0.10147326142369337,
  "right_elbow_joint": -0.35069419705870486,
  "right_wrist_roll_joint": 0.33277267784391723,
}

ARM_23 = tuple(CARRY_23)

G1_OBSERVER_REFERENCE_23 = ObserverReference(
  joint_names=ARM_23,
  inertia=(
    0.09186950253917056, 0.061019161763885936, 0.026431705651179672, 0.02342746849523782,
    0.0102, 0.0918514835622441, 0.06106578483046442, 0.026449785699018978,
    0.023426484888785452, 0.0102,
  ),
  friction_viscosity=(
    52.902388048386, 45.217739673251415, 21.16724895257324, 20.570311618666157,
    8.898014352329428, 52.86294285903987, 45.21531807390772, 21.169453138257435,
    20.563470711838786, 8.898013126563626,
  ),
  friction_limit=0.1,
  bandwidth_rad_s=8.0,
)


# World up expressed in each hand frame at the reference grasp.
HAND_UP_29 = (
  (0.44999313214179937, 0.0024089814401616625, 0.8930287665207854),
  (0.4432659840747714, -0.0026594064692518184, 0.8963861862609525),
)

HAND_UP_23 = (
  (0.4499931321417994, 0.002408981440161673, 0.8930287665207849),
  (0.44326598407477164, -0.002659406469251617, 0.8963861862609528),
)


G1_CARRY_REFERENCE_29 = CarryPoseReference(
  xml=G1_XML,
  joint_count=29,
  arm_names=ARM_29,
  hands=("left_wrist_yaw_link", "right_wrist_yaw_link"),
  pose=CARRY_29,
  dataset=SRC_PATH / "assets/data/g1/bones_seed/carry_poses_g1.pkl",
  motion_dof_indices=tuple(range(15, 29)),
  hand_up_local=HAND_UP_29,
)


G1_CARRY_REFERENCE_23 = CarryPoseReference(
  xml=G1_23DOF_XML,
  joint_count=23,
  arm_names=ARM_23,
  hands=("left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand"),
  pose=CARRY_23,
  dataset=SRC_PATH / "assets/data/g1/bones_seed/carry_poses_g1_23dof.pkl",
  motion_dof_indices=tuple(range(13, 23)),
  hand_up_local=HAND_UP_23,
)
