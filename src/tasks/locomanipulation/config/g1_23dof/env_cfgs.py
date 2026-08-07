"""Unitree G1-23DOF locomanipulation environment configurations."""

import math
import re
from pathlib import Path

from src.assets.robots import get_g1_23dof_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from src import SRC_PATH
from src.tasks.locomanipulation import mdp
from src.tasks.locomanipulation.mdp import UniformVelocityCommandCfg
from src.tasks.locomanipulation.mdp.events import TriangleWaveForceEvent
from src.tasks.locomanipulation.mdp.upper_body_action import UpperBodyMotionActionCfg
from src.tasks.locomanipulation.locomanipulation_env_cfg import make_locomanipulation_env_cfg


# 29-DOF motion data column indices for the 10 upper-body DOFs in 23-DOF.
# Maps: left_shoulder_pitch(15), left_shoulder_roll(16),
# left_shoulder_yaw(17), left_elbow(18), left_wrist_roll(19),
# right_shoulder_pitch(22), right_shoulder_roll(23), right_shoulder_yaw(24),
# right_elbow(25), right_wrist_roll(26).
MOTION_DOF_INDICES_23DOF = (15, 16, 17, 18, 19, 22, 23, 24, 25, 26)

LOWER_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
)

LOWER_BODY_JOINT_PATTERNS = (
    r".*_hip_pitch_joint",
    r".*_hip_roll_joint",
    r".*_hip_yaw_joint",
    r".*_knee_joint",
    r".*_ankle_pitch_joint",
    r".*_ankle_roll_joint",
)

LOWER_BODY_JOINT_CFG = SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINT_PATTERNS)


def unitree_g1_23dof_locomanipulation_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1-23DOF rough terrain locomanipulation configuration."""
  cfg = make_locomanipulation_env_cfg()

  ## Scene & Sensors ##

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  robot_cfg, action_scale = get_g1_23dof_robot_cfg(preset="unitree_stiff")
  cfg.scene.entities = {"robot": robot_cfg}
  lower_body_action_scale = {
    pat: val
    for pat, val in action_scale.items()
    if any(re.fullmatch(pat, jn) for jn in LOWER_BODY_JOINT_NAMES)
  }

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  ## Actions ##

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = lower_body_action_scale
  joint_pos_action.actuator_names = (
    r".*_hip_pitch_joint",
    r".*_hip_roll_joint",
    r".*_hip_yaw_joint",
    r".*_knee_joint",
    r".*_ankle_pitch_joint",
    r".*_ankle_roll_joint",
    r"waist_yaw_joint",
  )

  # Upper-body motion playback from BONES-SEED.
  # Standing envs → manipulation gestures (bones_seed/bones_seed.pkl).
  # Walking envs → coordinated locomotion arm swing (bones_seed/bones_seed_locomotion.pkl).
  # motion_dof_indices remaps 29-DOF motion data to 23-DOF joint layout.
  motion_file = str(SRC_PATH / "assets" / "data" / "g1" / "bones_seed" / "bones_seed_g1_23dof_split.pkl")
  loco_motion_file = str(SRC_PATH / "assets" / "data" / "g1" / "bones_seed" / "bones_seed_locomotion_g1_23dof_split.pkl")
  cfg.actions["upper_body_motion"] = UpperBodyMotionActionCfg(
    entity_name="robot",
    motion_file=motion_file,
    locomotion_motion_file=loco_motion_file,
    motion_dof_indices=MOTION_DOF_INDICES_23DOF,
    default_pose_ratio=1.0,
    command_threshold=0.1,
    waist_yaw_only=True,
    exclude_waist=True,
    pose_only=False,
  )

  ## Commands ##

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  ## Observations ##

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names
  cfg.observations["critic"].terms["wrist_force"].params[
    "asset_cfg"
  ].body_names = ("left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand")

  ## Hyperparameters ##

  # Command tuning.
  cfg.commands["base_height"].nominal_height = 0.76
  cfg.commands["base_height"].max_deviation_down = 0.26
  cfg.commands["base_height"].max_deviation_up = 0.02
  cfg.commands["base_height"].nominal_height_ratio = 0.05
  cfg.commands["waist_yaw"].nominal_yaw_ratio = 0.05

  # Observation tuning.
  cfg.observations["actor"].terms["phase"].params["period"] = 0.6

  # Reward weights.
  cfg.rewards["leg_joint_vel_penalty"].weight = -0.05
  cfg.rewards["base_drift_penalty"].weight = -2.0
  cfg.rewards["foot_swing_height"].weight = 0.0

  # Reward tuning params.
  cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)
  cfg.rewards["track_angular_velocity"].params["ang_vel_xy_weight"] = 0.05
  cfg.rewards["stand_still"].params["command_threshold"] = 0.1
  cfg.rewards["foot_gait"].params["period"] = 0.6
  cfg.rewards["track_base_height"].params["walking_weight"] = 0.25
  cfg.rewards["body_orientation_l2"].params["standing_weight"] = 2.0
  cfg.rewards["pose"].params["nominal_height"] = 0.76

  ## Events ##

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # External force on hands for carrying-heavy-object training.
  # Standing envs: triangle wave oscillation. Walking envs: resistance projection.
  cfg.events["hand_force"] = EventTermCfg(
    func=TriangleWaveForceEvent,
    mode="step",
    params={
      "force_range_max": {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (-40.0, 0.0),
      },
      "force_scale": 0.0,
      "torque_range": (0.0, 0.0),
      "period_s": (6.0, 10.0),
      "no_force_ratio": 0.3,
      "body_point_offset_range": {
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.05, 0.05),
      },
      "asset_cfg": SceneEntityCfg(
        "robot",
        body_names=("left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand"),
      ),
      "max_force_estimation": True,
      "constraint_joint_names": (
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_roll_joint",
      ),
      "command_name": "twist",
      "command_threshold": 0.1,
      "no_projection_ratio": 0.2,
    },
  )

  ## Rewards ##

  cfg.rewards["pose"].params["asset_cfg"] = LOWER_BODY_JOINT_CFG
  cfg.rewards["pose"].params["std_standing"] = {
    r".*hip_pitch.*": 0.05,
    r".*hip_roll.*": 0.05,
    r".*hip_yaw.*": 0.05,
    r".*knee.*": 0.05,
    r".*ankle_pitch.*": 0.05,
    r".*ankle_roll.*": 0.05,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.35,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
  }

  # G1 23-DOF has no waist_roll/pitch joints.
  cfg.rewards.pop("waist_regulation")

  # Restrict stand_still, joint_acc_l2, joint_pos_limits and leg_joint_vel_penalty to
  # lower-body joints.
  cfg.rewards["stand_still"].params["asset_cfg"] = LOWER_BODY_JOINT_CFG
  cfg.rewards["stand_still"].params["base_height_command_name"] = "base_height"
  cfg.rewards["joint_acc_l2"].params["asset_cfg"] = LOWER_BODY_JOINT_CFG
  cfg.rewards["joint_pos_limits"].params["asset_cfg"] = LOWER_BODY_JOINT_CFG
  cfg.rewards["leg_joint_vel_penalty"].params["asset_cfg"] = LOWER_BODY_JOINT_CFG

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_orientation_l2"].params["nominal_height"] = 0.76
  cfg.rewards["body_orientation_l2"].params["height_command_name"] = "base_height"
  cfg.rewards["body_orientation_l2"].params["height_relax_threshold"] = 0.16
  cfg.rewards["body_orientation_l2"].params["height_relax_min_scale"] = 0.2

  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_swing_height"].params["asset_cfg"].site_names = site_names
  cfg.rewards["feet_distance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  ## Height Postures ##

  import importlib.util as _ilu
  _postures_path = str(Path(__file__).resolve().parents[5] / "scripts" / "postures.py")
  _spec = _ilu.spec_from_file_location("postures", _postures_path)
  _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
  height_postures = _mod.HEIGHT_POSTURES
  cfg.rewards["pose"].params["height_postures"] = height_postures
  cfg.rewards["stand_still"].params["height_postures"] = height_postures

  ## Curriculum ##

  cfg.curriculum["force_curriculum"] = CurriculumTermCfg(
    func=mdp.force_scale_staged,
    params={
      "event_name": "hand_force",
      "stages": [
        {"step": 0, "scale": 0.0},
        {"step": 1000 * 24, "scale": 0.2},
        {"step": 2000 * 24, "scale": 0.4},
        {"step": 3000 * 24, "scale": 0.6},
        {"step": 4000 * 24, "scale": 0.8},
        {"step": 5000 * 24, "scale": 1.0},
      ],
    },
  )
  cfg.curriculum["default_pose_ratio"] = CurriculumTermCfg(
    func=mdp.default_pose_ratio_staged,
    params={
      "action_name": "upper_body_motion",
      "stages": [
        {"step": 0, "ratio": 1.0},
        {"step": 1000 * 24, "ratio": 0.8},
        {"step": 2000 * 24, "ratio": 0.6},
        {"step": 3000 * 24, "ratio": 0.4},
        {"step": 4000 * 24, "ratio": 0.2},
        {"step": 5000 * 24, "ratio": 0.05},
      ],
    },
  )
  cfg.curriculum["height_scale"] = CurriculumTermCfg(
    func=mdp.height_scale_staged,
    params={
      "command_name": "base_height",
      "stages": [
        {"step": 0, "scale": 0.0},
        {"step": 1000 * 24, "scale": 0.2},
        {"step": 2000 * 24, "scale": 0.4},
        {"step": 3000 * 24, "scale": 0.6},
        {"step": 4000 * 24, "scale": 0.8},
        {"step": 5000 * 24, "scale": 1.0},
      ],
    },
  )
  cfg.curriculum["waist_yaw_scale"] = CurriculumTermCfg(
    func=mdp.waist_yaw_scale_staged,
    params={
      "command_name": "waist_yaw",
      "stages": [
        {"step": 0, "scale": 0.0},
        {"step": 1000 * 24, "scale": 0.2},
        {"step": 2000 * 24, "scale": 0.4},
        {"step": 3000 * 24, "scale": 0.6},
        {"step": 4000 * 24, "scale": 0.8},
        {"step": 5000 * 24, "scale": 1.0},
      ],
    },
  )

  ## Play Mode ##

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

    cfg.events["hand_force"].params["no_force_ratio"] = 0.0
    cfg.events["hand_force"].params["force_scale"] = 1.0
    cfg.events["hand_force"].params["force_range_max"] = {
      "x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-40.0, 0.0),
    }

    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    # Pin upper body to zero pose (arms at sides).
    cfg.actions["upper_body_motion"].fixed_upper_body_pose = {
      "left_shoulder_pitch_joint": 0.0,
      "left_shoulder_roll_joint": 0.0,
      "left_shoulder_yaw_joint": 0.0,
      "left_elbow_joint": 0.0,
      "left_wrist_roll_joint": 0.0,
      "right_shoulder_pitch_joint": 0.0,
      "right_shoulder_roll_joint": 0.0,
      "right_shoulder_yaw_joint": 0.0,
      "right_elbow_joint": 0.0,
      "right_wrist_roll_joint": 0.0,
    }

    # Per-hand constant force (per-body format, body_frame rotates with robot):
    # cfg.events["hand_force"].params["constant_force"] = {
    #     "left_wrist_roll_rubber_hand": {"x": 5.0, "y": -5.0, "z": -20.0},
    #     "right_wrist_roll_rubber_hand": {"x": 5.0, "y": 5.0, "z": -20.0},
    # }
    # cfg.events["hand_force"].params["body_frame"] = True

    # Uniform force (same on both hands):
    cfg.events["hand_force"].params["constant_force"] = {"x": 0.0, "y": 0.0, "z": 0.0}

    cfg.commands["twist"].fixed_command = (0.0, 0.0, 0.0)
    cfg.commands["base_height"].fixed_height = 0.76
    cfg.commands["waist_yaw"].fixed_waist_yaw = 0.0

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_23dof_locomanipulation_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1-23DOF flat terrain locomanipulation configuration."""
  cfg = unitree_g1_23dof_locomanipulation_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
