"""Physical payload for X2 playback with a rigid two-hand grasp."""

from functools import partial
import math
from pathlib import Path

import joblib
import mujoco
import numpy as np
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import quat_apply, quat_mul


def _box_spec(mass: float, size: tuple[float, float, float]) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="box")
  body.add_freejoint(name="box_joint")
  body.add_geom(
    name="box_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
    size=np.asarray(size) / 2, mass=mass, rgba=(0.65, 0.38, 0.16, 1.0),
    contype=1, conaffinity=1,
  )
  return spec


def _reset_box(env, env_ids, offset: tuple[float, float, float],
               joint_positions: tuple[float, ...] | None = None) -> None:
  """Reset the payload with the robot's randomized root translation/yaw."""
  if env_ids is None:
    env_ids = slice(None)
  robot = env.scene["robot"]
  box = env.scene["carry_box"]
  if joint_positions is not None:
    positions = torch.tensor(joint_positions, device=env.device, dtype=torch.float32)
    positions = positions.expand(len(box.data.default_root_state[env_ids]), -1)
    robot.write_joint_state_to_sim(positions, torch.zeros_like(positions), env_ids=env_ids)
  env.sim.forward()
  pose = robot.data.root_link_pose_w[env_ids]
  state = box.data.default_root_state[env_ids].clone()
  local_offset = torch.tensor(
    offset, device=env.device, dtype=pose.dtype,
  ).expand(len(state), -1)
  state[:, :3] = pose[:, :3] + quat_apply(pose[:, 3:7], local_offset)
  state[:, 3:7] = pose[:, 3:7]
  state[:, 7:] = 0.0
  box.write_root_state_to_sim(state, env_ids=env_ids)


def add_carry_box(cfg, pose: dict[str, float], mass: float,
                  depth: float = 0.3, height: float = 0.25) -> None:
  """Add a massive box, positioned between wrists and welded to both."""
  if not all(math.isfinite(v) and v > 0 for v in (mass, depth, height)):
    raise ValueError("Box mass, depth, and height must be finite and positive")
  robot_cfg = cfg.scene.entities["robot"]
  # Resolve regex defaults before inserting exact carry-pose joint values.
  entity = Entity(robot_cfg)
  model = entity.spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  for name, value in pose.items():
    data.qpos[model.joint(name).qposadr[0]] = value
  mujoco.mj_forward(model, data)
  joint_positions = tuple(
    float(data.qpos[model.joint(name).qposadr[0]])
    for name in entity.joint_names
  )
  cfg.actions["upper_body_motion"].fixed_upper_body_pose = dict(pose)

  hands = ("left_wrist_roll_link", "right_wrist_roll_link")
  positions = [data.xpos[model.body(name).id].copy() for name in hands]
  # Hand capsules extend from local z=-0.08 to -0.14 beyond the roll link.
  palms = [
    position + data.xmat[model.body(name).id].reshape(3, 3) @ np.array([0, 0, -0.11])
    for name, position in zip(hands, positions)
  ]
  center = (palms[0] + palms[1]) / 2
  width = abs(palms[0][1] - palms[1][1]) - 2 * 0.044
  # Keep the rear face beyond collision and visual torso bounds, including
  # when a larger payload depth is requested.
  torso_names = {"pelvis", "waist_yaw_link", "waist_pitch_link", "torso_link"}
  for geom in range(model.ngeom):
    if model.body(int(model.geom_bodyid[geom])).name not in torso_names:
      continue
    rotation = data.geom_xmat[geom].reshape(3, 3)
    bound_center = data.geom_xpos[geom] + rotation @ model.geom_aabb[geom, :3]
    bound_half = np.abs(rotation) @ model.geom_aabb[geom, 3:]
    if abs(bound_center[2] - center[2]) <= bound_half[2] + height / 2:
      center[0] = max(center[0], bound_center[0] + bound_half[0] + 0.03 + depth / 2)
  size = (depth, max(width, 0.05), height)
  cfg.scene.entities["carry_box"] = EntityCfg(
    spec_fn=partial(_box_spec, mass, size),
    init_state=EntityCfg.InitialStateCfg(pos=tuple(center)),
  )

  transforms = []
  for name, position in zip(hands, positions):
    rotation = data.xmat[model.body(name).id].reshape(3, 3)
    relative_position = rotation.T @ (center - position)
    relative_quat = np.zeros(4)
    mujoco.mju_mat2Quat(relative_quat, rotation.T.reshape(-1))
    transforms.append([0, 0, 0, *relative_position, *relative_quat, 1])

  previous_edit = cfg.scene.spec_fn

  def attach(spec: mujoco.MjSpec) -> None:
    if previous_edit is not None:
      previous_edit(spec)
    # Set the scene's starting pose without changing the robot's nominal joint
    # positions: trained policies observe joint angles relative to those defaults.
    qpos = np.array(spec.keys[0].qpos)
    qpos[:model.nq] = data.qpos
    spec.keys[0].qpos = qpos
    for name, transform in zip(hands, transforms):
      spec.add_equality(
        name=f"carry_box_{name}", type=mujoco.mjtEq.mjEQ_WELD,
        objtype=mujoco.mjtObj.mjOBJ_BODY, name1=f"robot/{name}",
        name2="carry_box/box", data=transform, solref=(0.01, 1),
        solimp=(0.9999, 0.9999, 0.001, 0.5, 2),
      )
    # Exclude only the grasping wrists/hands; preserve body and terrain contact.
    for body in entity.spec.bodies:
      if body.name and "wrist" in body.name:
        spec.add_exclude(bodyname1=f"robot/{body.name}", bodyname2="carry_box/box")

  cfg.scene.spec_fn = attach
  cfg.sim.mujoco.iterations = max(cfg.sim.mujoco.iterations, 50)
  cfg.events["hand_force"].params["constant_force"] = {"x": 0.0, "y": 0.0, "z": 0.0}
  cfg.curriculum.pop("force_curriculum", None)
  # Keep the grasp geometry exact at reset, even if training used joint jitter.
  cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
  root_pos = np.asarray(robot_cfg.init_state.pos)
  cfg.events["reset_carry_box"] = EventTermCfg(
    func=_reset_box, mode="reset",
    params={"offset": tuple(center - root_pos), "joint_positions": joint_positions},
  )


def _sampled_box_geometry(model, data, depth: float, height: float):
  """Fit a level, heading-aligned box between the current palm centers."""
  hands = [model.body(f"{side}_wrist_roll_link").id for side in ("left", "right")]
  wrist_rotations = data.xmat[hands].reshape(2, 3, 3)
  palms = data.xpos[hands] + wrist_rotations @ np.array([0, 0, -0.11])
  lateral = palms[0] - palms[1]
  lateral[2] = 0
  norm = np.linalg.norm(lateral)
  if norm < 1e-8:
    raise ValueError("Carry pose has no horizontal palm separation")
  lateral /= norm
  forward = np.cross(lateral, [0, 0, 1])
  rotation = np.column_stack((forward, lateral, [0, 0, 1]))
  width = abs((palms[0] - palms[1]) @ rotation[:, 1]) - 0.088
  if width < 0.05 or abs(palms[0, 2] - palms[1, 2]) > 0.002:
    raise ValueError("Carry pose cannot center both palms on the box side faces")
  size = np.array([depth, width, height])
  half_x = np.abs(rotation[0]) @ (size / 2)
  center = palms.mean(axis=0)
  for geom in range(model.ngeom):
    if model.body(int(model.geom_bodyid[geom])).name not in {
      "pelvis", "waist_yaw_link", "waist_pitch_link", "torso_link",
    }:
      continue
    geom_rotation = data.geom_xmat[geom].reshape(3, 3)
    bound_center = data.geom_xpos[geom] + geom_rotation @ model.geom_aabb[geom, :3]
    bound_half = np.abs(geom_rotation) @ model.geom_aabb[geom, 3:]
    if abs(bound_center[2] - center[2]) <= bound_half[2] + height / 2:
      if center[0] - half_x < bound_center[0] + bound_half[0] + 0.03:
        raise ValueError("Carry pose cannot center this box between the palms with torso clearance")
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, rotation.reshape(-1))
  transforms = []
  for sign, hand, hand_rotation in zip((1, -1), hands, wrist_rotations):
    relative_quat = np.zeros(4)
    mujoco.mju_mat2Quat(relative_quat, (hand_rotation.T @ rotation).reshape(-1))
    box_anchor = np.array([0, sign * width / 2, 0])
    grasp_point = center + rotation @ box_anchor
    hand_anchor = hand_rotation.T @ (grasp_point - data.xpos[hand])
    transforms.append([*box_anchor, *hand_anchor, *relative_quat, 1])
  return center, quat, size, np.array(transforms)


@requires_model_fields(
  "geom_size", "geom_aabb", "geom_rbound", "body_inertia", "eq_data",
  recompute=RecomputeLevel.set_const_0,
)
class SampledCarryBox(ManagerTermBase):
  """Fit the payload after the action manager has sampled the reset's pose."""

  def __init__(self, cfg, env):
    super().__init__(env)
    self.depth_range = cfg.params["depth_range"]
    self.height_range = cfg.params["height_range"]
    self.mass = cfg.params["mass"]
    self.model = Entity(env.cfg.scene.entities["robot"]).spec.compile()
    self.data = mujoco.MjData(self.model)
    self.joint_names = env.scene["robot"].joint_names
    compiled = env.sim.mj_model
    self.geom_id = compiled.geom("carry_box/box_geom").id
    self.body_id = compiled.body("carry_box/box").id
    self.weld_ids = [compiled.equality(f"carry_box_{side}_wrist_roll_link").id
                     for side in ("left", "right")]

  def __call__(self, env, env_ids, **kwargs):
    # Reset events run before arm sampling. EventManager.reset calls our reset
    # after ActionManager.reset, when the selected pose and targets are final.
    pass

  def reset(self, env_ids=None):
    env = self._env
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
      return
    robot = env.scene["robot"]
    box = env.scene["carry_box"]
    env.sim.forward()
    root_pose = robot.data.root_link_pose_w[env_ids]
    root_velocity = robot.data.root_link_vel_w[env_ids]
    joint_values = robot.data.joint_pos[env_ids].cpu().numpy()
    draws = torch.rand(len(env_ids), 2, device=env.device).cpu().numpy()
    centers, quats, sizes, welds = [], [], [], []
    for joints, draw in zip(joint_values, draws):
      mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
      for name, value in zip(self.joint_names, joints):
        self.data.qpos[self.model.joint(name).qposadr[0]] = value
      mujoco.mj_forward(self.model, self.data)
      depth = self.depth_range[0] + draw[0] * (self.depth_range[1] - self.depth_range[0])
      height = self.height_range[0] + draw[1] * (self.height_range[1] - self.height_range[0])
      center, quat, size, transforms = _sampled_box_geometry(self.model, self.data, depth, height)
      centers.append(center - np.asarray(robot.cfg.init_state.pos))
      quats.append(quat)
      sizes.append(size)
      welds.append(transforms)
    def tensor(values):
      return torch.as_tensor(np.asarray(values), device=env.device, dtype=root_pose.dtype)
    size = tensor(sizes)
    half = size / 2
    fields = env.sim.model
    fields.geom_size[env_ids, self.geom_id] = half
    fields.geom_aabb[env_ids, self.geom_id, 0] = 0
    fields.geom_aabb[env_ids, self.geom_id, 1] = half
    fields.geom_rbound[env_ids, self.geom_id] = torch.linalg.norm(half, dim=-1)
    inertia = self.mass / 12 * (size.square().sum(dim=-1, keepdim=True) - size.square())
    fields.body_inertia[env_ids, self.body_id] = inertia
    fields.eq_data[env_ids[:, None], self.weld_ids] = tensor(welds)
    state = box.data.default_root_state[env_ids].clone()
    offset = quat_apply(root_pose[:, 3:7], tensor(centers))
    state[:, :3] = root_pose[:, :3] + offset
    state[:, 3:7] = quat_mul(root_pose[:, 3:7], tensor(quats))
    state[:, 7:10] = root_velocity[:, :3] + torch.cross(root_velocity[:, 3:], offset, dim=-1)
    state[:, 10:] = root_velocity[:, 3:]
    box.write_root_state_to_sim(state, env_ids=env_ids)
    env.sim.recompute_constants(RecomputeLevel.set_const_0)
    env.sim.forward()
    action = env.action_manager.get_term("upper_body_motion")
    for index, dimensions in zip(action._clip_idx[env_ids].tolist(), sizes):
      print(f"[INFO]: Carry test pose {index}: box XYZ={np.round(dimensions, 3)} m, mass={self.mass:g} kg")


def add_sampled_carry_box(cfg, pose_file: str, mass: float,
                         depth_range=(0.2, 0.4), height_range=(0.15, 0.3)) -> None:
  """Enable dataset pose sampling with variable payload dimensions on resets."""
  for bounds in (depth_range, height_range):
    if not all(math.isfinite(value) and value > 0 for value in bounds) or bounds[0] > bounds[1]:
      raise ValueError("Box dimension ranges must be finite, positive, and ordered")
  clips = joblib.load(Path(pose_file))
  if not clips:
    raise ValueError("Carry pose dataset is empty")
  entity = Entity(cfg.scene.entities["robot"])
  joint_names = entity.joint_names
  for clip in clips.values():
    dof = np.asarray(clip["dof"])
    if dof.ndim != 2 or dof.shape[0] < 1 or dof.shape[1] != len(joint_names):
      raise ValueError("Carry pose clips must have shape (frames, 29)")
    if not np.isfinite(dof).all() or not np.all(dof == dof[0]):
      raise ValueError("Carry box sampling requires finite static pose clips")
  model = entity.spec.compile()
  data = mujoco.MjData(model)
  eligible = []
  # Requiring clearance at the maximum dimensions guarantees every requested
  # size can be centered at the hands, rather than hiding an offset in the weld.
  for index, clip in enumerate(clips.values()):
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for column, name in enumerate(joint_names):
      if any(part in name for part in ("shoulder", "elbow", "wrist")):
        data.qpos[model.joint(name).qposadr[0]] = clip["dof"][0, column]
    mujoco.mj_forward(model, data)
    try:
      _sampled_box_geometry(model, data, depth_range[1], height_range[1])
    except ValueError:
      continue
    eligible.append(index)
  if not eligible:
    raise ValueError("No carry poses fit the requested box size ranges with centered grasps and torso clearance")
  print(f"[INFO]: {len(eligible)}/{len(clips)} carry poses fit the requested box dimensions")
  first = list(clips.values())[eligible[0]]["dof"][0]
  pose = {name: float(first[index]) for index, name in enumerate(joint_names) if "shoulder" in name or "elbow" in name or "wrist" in name}
  add_carry_box(cfg, pose, mass, sum(depth_range) / 2, sum(height_range) / 2)
  action = cfg.actions["upper_body_motion"]
  action.motion_file = str(pose_file)
  action.locomotion_motion_file = None
  action.fixed_upper_body_pose = None
  action.pose_only = True
  action.default_pose_ratio = 0.0
  action.sample_clip_indices = tuple(eligible)
  cfg.curriculum.pop("default_pose_ratio", None)
  cfg.events["reset_carry_box"] = EventTermCfg(
    func=SampledCarryBox, mode="reset",
    params={"mass": mass, "depth_range": depth_range, "height_range": height_range},
  )
