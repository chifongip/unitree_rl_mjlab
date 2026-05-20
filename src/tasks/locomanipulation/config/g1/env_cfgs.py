"""Unitree G1 locomanipulation environment configurations."""

import re

from src.assets.robots import get_g1_robot_cfg
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
from src.tasks.locomanipulation.mdp.events import HandForceEvent
from src.tasks.locomanipulation.mdp.upper_body_action import UpperBodyMotionActionCfg
from src.tasks.locomanipulation.locomanipulation_env_cfg import make_locomanipulation_env_cfg


def unitree_g1_locomanipulation_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain locomanipulation configuration."""
  cfg = make_locomanipulation_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

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
  )

  robot_cfg, action_scale = get_g1_robot_cfg(preset="unitree_stiff")
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
  )

  # Upper-body motion playback from ACCAD dataset.
  # In play mode, default_pose_ratio=1.0 so all envs hold HOME_KEYFRAME.
  motion_file = str(SRC_PATH / "assets" / "data" / "g1" / "accad_all.pkl")
  cfg.actions["upper_body_motion"] = UpperBodyMotionActionCfg(
    entity_name="robot",
    motion_file=motion_file,
    default_pose_ratio=1.0,
    waist_yaw_only=True,
    pose_only=True,
  )

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names
  cfg.observations["critic"].terms["wrist_force"].params[
    "asset_cfg"
  ].body_names = ("left_wrist_yaw_link", "right_wrist_yaw_link")

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # External force on hands for carrying-heavy-object training.
  # In play mode, no external force is applied.
  # Jacobian-based max force estimation ensures forces are physically plausible.
  cfg.events["hand_force"] = EventTermCfg(
    func=HandForceEvent,
    mode="step",
    params={
      "force_range_max": {
        "x": (-40.0, 40.0),
        "y": (-40.0, 40.0),
        "z": (-50.0, 5.0),
      },
      "force_scale": 0.0,
      "torque_range": (0.0, 0.0),
      "duration_s": (8.0, 12.0),
      "cooldown_s": (2.0, 4.0),
      "no_force_ratio": 0.05,
      "zero_force_prob": {
        "x": 0.25,
        "y": 0.25,
        "z": 0.25,
      },
      "body_point_offset_range": {
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.05, 0.05),
      },
      "asset_cfg": SceneEntityCfg(
        "robot",
        body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
      ),
      "max_force_estimation": True,
      "constraint_joint_names": (
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_roll_joint",
        ".*_wrist_pitch_joint",
        ".*_wrist_yaw_joint",
        # "waist_yaw_joint",                                                                                                                                               
        # "waist_roll_joint",                                                                                                                                              
        # "waist_pitch_joint",
      ),
    },
  )
  cfg.curriculum["force_curriculum"] = CurriculumTermCfg(
    func=mdp.force_scale_staged,
    params={
      "event_name": "hand_force",
      "stages": [
        {"step": 0, "scale": 0.0},
        {"step": 2000 * 24, "scale": 0.2},
        {"step": 4500 * 24, "scale": 0.4},
        {"step": 7500 * 24, "scale": 0.6},
        {"step": 11000 * 24, "scale": 0.8},
        {"step": 15000 * 24, "scale": 1.0},
      ],
    },
  )
  cfg.curriculum["default_pose_ratio"] = CurriculumTermCfg(
    func=mdp.default_pose_ratio_staged,
    params={
      "action_name": "upper_body_motion",
      "stages": [
        {"step": 0, "ratio": 1.0},
        {"step": 2000 * 24, "ratio": 0.8},
        {"step": 4500 * 24, "ratio": 0.6},
        {"step": 7500 * 24, "ratio": 0.4},
        {"step": 11000 * 24, "ratio": 0.2},
        {"step": 15000 * 24, "ratio": 0.05},
      ],
    },
  )
  cfg.curriculum["height_scale"] = CurriculumTermCfg(
    func=mdp.height_scale_staged,
    params={
      "command_name": "base_height",
      "stages": [
        {"step": 0, "scale": 0.0},
        {"step": 2000 * 24, "scale": 0.2},
        {"step": 4500 * 24, "scale": 0.4},
        {"step": 7500 * 24, "scale": 0.6},
        {"step": 11000 * 24, "scale": 0.8},
        {"step": 15000 * 24, "scale": 1.0},
      ],
    },
  )
  cfg.commands["base_height"].nominal_height_ratio = 0.05

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  # Restrict pose reward to lower-body joints only — upper body is controlled by
  # motion playback, not the policy, so penalizing its deviation is misleading.
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=(
      r".*_hip_pitch_joint",
      r".*_hip_roll_joint",
      r".*_hip_yaw_joint",
      r".*_knee_joint",
      r".*_ankle_pitch_joint",
      r".*_ankle_roll_joint",
    )
  )
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
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
  }
  cfg.rewards["pose"].params["height_postures"] = {
    0.5: {
      "left_hip_pitch_joint": -1.055, "left_hip_roll_joint": 0.0001, "left_hip_yaw_joint": -0.0,
      "left_knee_joint": 1.949, "left_ankle_pitch_joint": -0.8727, "left_ankle_roll_joint": -0.0001,
      "right_hip_pitch_joint": -1.055, "right_hip_roll_joint": -0.0001, "right_hip_yaw_joint": 0.0,
      "right_knee_joint": 1.949, "right_ankle_pitch_joint": -0.8727, "right_ankle_roll_joint": 0.0,
    },
    0.55: {
      "left_hip_pitch_joint": -0.8771, "left_hip_roll_joint": 0.0001, "left_hip_yaw_joint": -0.0,
      "left_knee_joint": 1.7667, "left_ankle_pitch_joint": -0.8727, "left_ankle_roll_joint": -0.0001,
      "right_hip_pitch_joint": -0.8771, "right_hip_roll_joint": -0.0, "right_hip_yaw_joint": 0.0,
      "right_knee_joint": 1.7667, "right_ankle_pitch_joint": -0.8727, "right_ankle_roll_joint": 0.0,
    },
    0.6: {
      "left_hip_pitch_joint": -0.6721, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": -0.0,
      "left_knee_joint": 1.5523, "left_ankle_pitch_joint": -0.8727, "left_ankle_roll_joint": -0.0,
      "right_hip_pitch_joint": -0.6721, "right_hip_roll_joint": -0.0, "right_hip_yaw_joint": 0.0,
      "right_knee_joint": 1.5523, "right_ankle_pitch_joint": -0.8727, "right_ankle_roll_joint": 0.0,
    },
    0.65: {
      "left_hip_pitch_joint": -0.509, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": -0.0,
      "left_knee_joint": 1.3006, "left_ankle_pitch_joint": -0.7916, "left_ankle_roll_joint": 0.0,
      "right_hip_pitch_joint": -0.509, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": -0.0,
      "right_knee_joint": 1.3006, "right_ankle_pitch_joint": -0.7916, "right_ankle_roll_joint": 0.0,
    },
    0.7: {
      "left_hip_pitch_joint": -0.3858, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": -0.0,
      "left_knee_joint": 1.0131, "left_ankle_pitch_joint": -0.6273, "left_ankle_roll_joint": 0.0,
      "right_hip_pitch_joint": -0.3858, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": -0.0,
      "right_knee_joint": 1.0131, "right_ankle_pitch_joint": -0.6273, "right_ankle_roll_joint": 0.0,
    },
    0.75: {
      "left_hip_pitch_joint": -0.2081, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": -0.0,
      "left_knee_joint": 0.6159, "left_ankle_pitch_joint": -0.4078, "left_ankle_roll_joint": 0.0,
      "right_hip_pitch_joint": -0.2081, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": -0.0,
      "right_knee_joint": 0.6159, "right_ankle_pitch_joint": -0.4078, "right_ankle_roll_joint": 0.0,
    },
    0.785: {
      "left_hip_pitch_joint": 0.0142, "left_hip_roll_joint": -0.0003, "left_hip_yaw_joint": 0.0001,
      "left_knee_joint": 0.0448, "left_ankle_pitch_joint": -0.0372, "left_ankle_roll_joint": 0.0008,
      "right_hip_pitch_joint": 0.0142, "right_hip_roll_joint": 0.0003, "right_hip_yaw_joint": -0.0001,
      "right_knee_joint": 0.0448, "right_ankle_pitch_joint": -0.0372, "right_ankle_roll_joint": -0.0004,
    },
  }

  # Restrict stand_still, joint_acc_l2 and joint_pos_limits to lower-body joints — upper body is
  # driven by motion playback, not the policy.
  cfg.rewards["stand_still"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=(
      r".*_hip_pitch_joint",
      r".*_hip_roll_joint",
      r".*_hip_yaw_joint",
      r".*_knee_joint",
      r".*_ankle_pitch_joint",
      r".*_ankle_roll_joint",
    )
  )
  cfg.rewards["joint_acc_l2"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=(
      r".*_hip_pitch_joint",
      r".*_hip_roll_joint",
      r".*_hip_yaw_joint",
      r".*_knee_joint",
      r".*_ankle_pitch_joint",
      r".*_ankle_roll_joint",
    )
  )
  cfg.rewards["joint_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=(
      r".*_hip_pitch_joint",
      r".*_hip_roll_joint",
      r".*_hip_yaw_joint",
      r".*_knee_joint",
      r".*_ankle_pitch_joint",
      r".*_ankle_roll_joint",
    )
  )

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

    # Keep hand_force for constant-force testing; disable random impulse lifecycle.
    cfg.events["hand_force"].params["no_force_ratio"] = 1.0
    cfg.events["hand_force"].params["force_range_max"] = {
      "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)
    }
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    cfg.actions["upper_body_motion"].fixed_upper_body_pose = {
      "waist_yaw_joint": 0.0,
      "waist_roll_joint": 0.0,
      "waist_pitch_joint": 0.0,
      "left_shoulder_pitch_joint": 0.0,
      "left_shoulder_roll_joint": 0.0,
      "left_shoulder_yaw_joint": 0.0,
      "left_elbow_joint": 0.0,
      "left_wrist_roll_joint": 0.0,
      "left_wrist_pitch_joint": 0.0,
      "left_wrist_yaw_joint": 0.0,
      "right_shoulder_pitch_joint": 0.0,
      "right_shoulder_roll_joint": 0.0,
      "right_shoulder_yaw_joint": 0.0,
      "right_elbow_joint": 0.0,
      "right_wrist_roll_joint": 0.0,
      "right_wrist_pitch_joint": 0.0,
      "right_wrist_yaw_joint": 0.0,
    }
    cfg.events["hand_force"].params["constant_force"] = {"x": 0.0, "y": 0.0, "z": -30.0}
    cfg.commands["twist"].fixed_command = (1.0, 0.0, 0.0)
    cfg.commands["base_height"].fixed_height = 0.785

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_locomanipulation_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain locomanipulation configuration."""
  cfg = unitree_g1_locomanipulation_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
