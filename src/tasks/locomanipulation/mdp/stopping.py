"""Standing/walking command profiles and bounded stopping rewards for X2."""

from dataclasses import dataclass
from copy import copy
import inspect

import mujoco
import numpy as np
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.lab_api.math import quat_apply, wrap_to_pi
from .velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg


@dataclass(kw_only=True)
class StopVelocityCommandCfg(UniformVelocityCommandCfg):
  sample_profiles: bool = True
  standing_ratio: float = .3
  transition_ratio: float = .3
  start_walking_ratio: float = .2
  transition_s: float = 1.2
  gamma: float = .99
  pose_std: float = .2
  placement_weight: float = 1.
  swing_weight: float = .5
  slip_weight: float = .25
  yaw_slip_weight: float = .25
  shaping_weight: float = .1

  def build(self, env):
    return StopVelocityCommand(self, env)


class StopVelocityCommand(UniformVelocityCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    ratios = (cfg.standing_ratio, cfg.transition_ratio, cfg.start_walking_ratio)
    if not (all(0 <= ratio <= 1 for ratio in ratios) and sum(ratios) <= 1):
      raise ValueError("Invalid standing/transition episode fractions")
    if cfg.transition_s <= 0:
      raise ValueError("transition_s must be positive")
    self.profile = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self.switch_at = torch.zeros(self.num_envs, device=self.device)
    self.walk_command = torch.zeros((self.num_envs, 3), device=self.device)
    self.stop_state = None

  def reset(self, env_ids):
    draw = torch.rand(len(env_ids), device=self.device)
    stand_end = self.cfg.standing_ratio
    stop_end = stand_end + self.cfg.transition_ratio
    start_end = stop_end + self.cfg.start_walking_ratio
    # Profiles: 0 walk, 1 stand, 2 walk-to-stand, 3 stand-to-walk.
    self.profile[env_ids] = torch.where(draw < stand_end, 1,
      torch.where(draw < stop_end, 2, torch.where(draw < start_end, 3, 0)))
    self.switch_at[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(2., 4.)
    if self.stop_state is not None:
      self.stop_state.reset(env_ids)
    return super().reset(env_ids)

  def _resample_command(self, env_ids):
    super()._resample_command(env_ids)
    if self.cfg.sample_profiles:
      self.walk_command[env_ids] = self.vel_command_b[env_ids]
      self.is_heading_env[env_ids] = False
      self.is_standing_env[env_ids] = self.profile[env_ids] == 1
      # One command profile per episode; never resample away the stop request.
      self.time_left[env_ids] = self._env.max_episode_length_s + 1

  def _update_command(self, env_ids=None):
    super()._update_command(env_ids)
    if self.cfg.sample_profiles:
      ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
      switched = self._env.episode_length_buf[ids] * self._env.step_dt >= self.switch_at[ids]
      starting = self.profile[ids] == 3
      # Restore the sampled command after the initial standing interval.
      self.vel_command_b[ids[starting]] = self.walk_command[ids[starting]]
      standing = (self.profile[ids] == 1) | ((self.profile[ids] == 2) & switched) | (starting & ~switched)
      self.vel_command_b[ids[standing]] = 0.
    if self.stop_state is not None:
      # Commands are resampled AFTER rewards. Reconcile next-observation state
      # without advancing the timers that belong to the executed action.
      self.stop_state.reconcile()

  def state(self):
    if self.stop_state is None:
      self.stop_state = StopState(self._env, self)
    return self.stop_state


class StopState:
  """Reward phases: 0 walking, 1 stopping grace period, 2 standing."""

  def __init__(self, env, command):
    self.env, self.command, self.robot = env, command, env.scene["robot"]
    device, n = env.device, env.num_envs
    # Reset starts in standing; only a moving-command edge starts a transition.
    self.mode = torch.full((n,), 2, device=device, dtype=torch.long)
    self.age = torch.zeros(n, device=device)
    self.starting = torch.zeros(n, device=device, dtype=torch.bool)
    self.was_moving = torch.zeros(n, device=device, dtype=torch.bool)
    self.foot_stage = torch.full((n, 2), 2, device=device, dtype=torch.long)
    self.contact_ticks = torch.zeros((n, 2), device=device, dtype=torch.long)
    self.air_ticks = torch.zeros_like(self.contact_ticks)
    self.touchdown = torch.zeros((n, 2), device=device, dtype=torch.bool)
    self.landing_paid = torch.zeros((n, 2), device=device, dtype=torch.bool)
    self.landing_guidance = torch.zeros_like(self.landing_paid)
    self.guidance_active = torch.zeros(n, device=device, dtype=torch.bool)
    self.finished = torch.zeros(n, device=device, dtype=torch.bool)
    self.deadline = torch.zeros_like(self.finished)
    self.previous_phi = torch.zeros(n, device=device)
    self.shaping = torch.zeros(n, device=device)
    self.last_step = -1
    self.body_ids, _ = self.robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"])
    table = env.cfg.rewards["pose"].params["standing_height_postures"]
    self.names = list(next(iter(table.values())))
    self.joint_ids, names = self.robot.find_joints(self.names, preserve_order=True)
    assert list(names) == self.names
    self.keys = torch.tensor(sorted(table), device=device)
    self.targets = torch.tensor([[table[h][k] for k in self.names] for h in sorted(table)], device=device)
    self.leg_columns = [[i for i, name in enumerate(self.names) if name.startswith(side + "_")]
                        for side in ("left", "right")]
    self.hip_yaw_ids = [self.names.index(f"{side}_hip_yaw_joint") for side in ("left", "right")]
    # Precompute the same capsule-centerline centroid used by the generator.
    from src.assets.robots.agibot_x2.x2_constants import get_spec
    model = get_spec().compile()
    data = mujoco.MjData(model)
    offsets = []
    for side in ("left", "right"):
      geoms = [model.geom(f"{side}_foot{i}_collision").id for i in range(1, 8)]
      points = model.geom_pos[geoms]
      offsets.append(points[np.abs(points[:, 2] - np.median(points[:, 2])) <= .005].mean(axis=0))
    self.offsets = torch.tensor(np.array(offsets), dtype=torch.float32, device=device)
    references = []
    walking_table = env.cfg.rewards["pose"].params["walking_height_postures"]
    for reference in (table, walking_table):
      geometry = []
      headings = []
      for h in sorted(table):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = 0
        data.qpos[3:7] = (1, 0, 0, 0)
        for name, value in reference[min(reference, key=lambda k: abs(k-h))].items():
          data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        mujoco.mj_forward(model, data)
        feet = []
        for side, offset in zip(("left", "right"), offsets):
          b = model.body(f"{side}_ankle_roll_link").id
          feet.append(data.xpos[b] + data.xmat[b].reshape(3, 3) @ offset)
        # Relative feet are independent of root translation. Undo root pitch
        # using the known, flat target sole rotation.
        b = model.body("left_ankle_roll_link").id
        r = data.xmat[b].reshape(3, 3)
        pitch = np.arctan2(-r[0, 2], r[2, 2])
        c, sn = np.cos(pitch), np.sin(pitch)
        base = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])
        row = []
        for side in ("left", "right"):
          foot_rotation = base @ data.xmat[model.body(f"{side}_ankle_roll_link").id].reshape(3, 3)
          row.append(np.arctan2(foot_rotation[1, 0], foot_rotation[0, 0]))
        headings.append(row)
        centered = np.array(feet) @ base.T
        geometry.append(centered[:, :2] - centered[:, :2].mean(axis=0))
      references.append((torch.tensor(np.array(geometry), dtype=torch.float32, device=device),
                         torch.tensor(headings, dtype=torch.float32, device=device)))
    (self.goal_geometry, self.goal_headings), (self.walk_geometry, self.walk_headings) = references

  def reset(self, ids):
    self.mode[ids] = 2
    self.foot_stage[ids] = 2
    for value in (self.age, self.previous_phi, self.shaping, self.starting, self.was_moving,
                  self.contact_ticks, self.air_ticks, self.touchdown, self.finished, self.deadline, self.guidance_active, self.landing_paid, self.landing_guidance):
      value[ids] = 0

  def reconcile(self):
    cmd = self.command.command
    moving = torch.linalg.norm(cmd[:, :2], dim=1) + cmd[:, 2].abs() >= .1
    changed = moving != self.was_moving
    self.starting[changed] = moving[changed]
    self.mode[changed] = torch.where(moving[changed], 0, 1)
    self.age[changed] = 0
    contact = self.env.scene["feet_ground_contact"].data.found > 0
    # Carry confirmed sensor history across command edges. A single raw
    # dropout must not initialize a swing that never actually happened.
    confirmed_air = ~contact & (self.air_ticks >= 2)
    self.foot_stage[changed] = torch.where(confirmed_air[changed], 1, 0)
    self.landing_paid[changed] = False
    self.landing_guidance[changed] = False
    self.finished[changed] = False
    self.deadline[changed] = False
    self.touchdown[changed] = False
    self.guidance_active[changed] = False
    self.was_moving = moving.clone()
    # Preserve zero potential on entry and close any previous stopping phase.

  @property
  def active(self):
    return self.starting | (self.mode == 1)

  def pose_mask(self):
    mask = torch.ones_like(self.q_error)
    for foot, columns in enumerate(self.leg_columns):
      pending = self.active & (self.foot_stage[:, foot] == 0)
      mask[:, columns] = (~pending).float()[:, None]
    return mask

  def feet_critic(self):
    self.reconcile()
    return torch.cat((self.starting.float()[:, None], self.foot_stage.float() / 2), 1)

  def speed(self):
    return torch.linalg.norm(self.robot.data.root_link_lin_vel_b[:, :2], dim=1)

  def geometry(self, destination=False):
    r = self.robot.data
    quats = r.body_link_quat_w[:, self.body_ids]
    local = self.offsets[None].expand(self.env.num_envs, -1, -1)
    centers = r.body_link_pos_w[:, self.body_ids] + quat_apply(quats, local)
    xy = centers[:, :, :2] - centers[:, :, :2].mean(dim=1, keepdim=True)
    # A body-relative stance is observable from proprioception. Capturing a
    # historical heading would introduce an unseen goal after turning stops.
    heading = r.heading_w
    c, s = heading.cos()[:, None], heading.sin()[:, None]
    xy = torch.stack((c * xy[..., 0] + s * xy[..., 1], -s * xy[..., 0] + c * xy[..., 1]), -1)
    x_axis = torch.zeros_like(local)
    x_axis[..., 0] = 1
    axes = quat_apply(quats, x_axis)
    yaw = wrap_to_pi(torch.atan2(axes[..., 1], axes[..., 0]) - heading[:, None])
    heights = self.env.command_manager.get_command("base_height")[:, 0]
    index = (heights[:, None] - self.keys[None]).abs().argmin(1)
    target_xy, target_yaw = self.goal_geometry[index], self.goal_headings[index]
    if destination:
      moving = self.was_moving
      target_xy = torch.where(moving[:, None, None], self.walk_geometry[index], target_xy)
      target_yaw = torch.where(moving[:, None], self.walk_headings[index], target_yaw)
    error_xy = xy - target_xy
    error_yaw = wrap_to_pi(yaw - target_yaw)
    return torch.cat((error_xy, error_yaw[..., None]), -1), index

  def advance(self):
    self.reconcile()
    if self.last_step == self.env.common_step_counter:
      return
    self.last_step = self.env.common_step_counter
    dt = self.env.step_dt
    errors, index = self.geometry()
    self.errors = errors
    q_error = self.robot.data.joint_pos[:, self.joint_ids] - self.targets[index]
    self.q_error = q_error
    active = self.active.clone()
    contact = self.env.scene["feet_ground_contact"].data.found > 0
    self.contact_ticks = torch.where(contact, self.contact_ticks + 1, 0)
    self.air_ticks = torch.where(~contact, self.air_ticks + 1, 0)
    lifted = active[:, None] & (self.foot_stage == 0) & (self.air_ticks >= 2)
    self.foot_stage[lifted] = 1
    self.touchdown = active[:, None] & (self.foot_stage == 1) & (self.contact_ticks >= 2)
    self.landing_guidance = self.touchdown & ~self.landing_paid
    self.landing_paid |= self.landing_guidance
    self.foot_stage[self.touchdown] = 2
    self.age[active] += dt
    self.deadline = active & (self.age >= self.command.cfg.transition_s - 1e-6)
    placed = (self.foot_stage == 2).all(1)
    # Startup can continue stepping. Stopping must not declare completion
    # while a previously landed foot is swinging again. The deadline remains bounded.
    supported = (self.contact_ticks >= 2).all(1)
    complete = placed & (self.starting | supported)
    self.finished = active & (complete | self.deadline)
    self.guidance_active = active  # Pay landing guidance on the completion sample.
    self.mode[self.finished & (self.mode == 1)] = 2
    self.starting[self.finished] = False
    phi = -self.speed()
    # Close the potential at every transition exit, including command changes
    # and true termination. The boundary increment is paid outside reward gating.
    next_phi = torch.where((self.mode == 1) & ~self.env.reset_terminated, phi, torch.zeros_like(phi))
    self.shaping = (self.command.cfg.gamma * next_phi - self.previous_phi) / dt
    self.previous_phi = next_phi
    self.env.extras["log"]["Transition/standing_fraction"] = (self.mode == 2).float().mean()
    # Walking deliberately uses a different posture reference.
    standing_command = (self.mode != 0).float()
    count = standing_command.sum().clamp_min(1)
    placement_error = errors[..., :2].norm(dim=-1).mean(1)
    hip_yaw_error = q_error[:, self.hip_yaw_ids].abs().mean(1)
    self.env.extras["log"]["Transition/placement_error"] = (placement_error * standing_command).sum() / count
    self.env.extras["log"]["Transition/hip_yaw_error"] = (hip_yaw_error * standing_command).sum() / count

  def critic(self):
    self.reconcile()  # no time advancement after command resampling
    errors, _ = self.geometry()
    return torch.cat((torch.nn.functional.one_hot(self.mode, 3).float(),
                      (self.age / self.command.cfg.transition_s).clamp(max=1)[:, None],
                      errors.flatten(1)), 1)


def transition_feet_observation(env):
  return env.command_manager.get_term("twist").state().feet_critic()


def stop_observation(env):
  return env.command_manager.get_term("twist").state().critic()


class StopReward:
  """Route an existing reward without changing its parameter resolution."""
  def __init__(self, cfg, env):
    self.kind = cfg.params["transition_kind"]
    original = cfg.params["original_func"]
    inner = copy(cfg)
    inner.params = {k: v for k, v in cfg.params.items() if k not in ("original_func", "transition_kind")}
    self.func = original(inner, env) if inspect.isclass(original) else original

  def reset(self, env_ids=None):
    if hasattr(self.func, "reset"):
      self.func.reset(env_ids)

  def __call__(self, env, original_func=None, transition_kind=None, **params):
    state = env.command_manager.get_term("twist").state()
    state.advance()
    stopping = state.mode == 1
    if self.kind == "pose":
      mask = state.pose_mask()
      columns = [state.joint_ids.index(joint) for joint in params["asset_cfg"].joint_ids]
      value = self.func(env, joint_mask=mask[:, columns], **params)
      soft = torch.exp(-(state.q_error.square() * mask).mean(1) / state.command.cfg.pose_std**2)
      return torch.where(stopping, soft, value)
    value = self.func(env, **params)
    return torch.where(stopping, torch.zeros_like(value), value)


def stopping_reward(env):
  state = env.command_manager.get_term("twist").state()
  state.advance()
  e, _ = state.geometry(destination=True)
  eligible = state.guidance_active[:, None] & state.landing_guidance
  heading = torch.exp(-e[..., 2].square() / .3**2)
  placement = torch.exp(-e[..., :2].square().sum(-1) / .05**2)
  cfg = state.command.cfg
  value = (cfg.swing_weight * heading * eligible).sum(1)
  value += cfg.placement_weight * (placement * eligible).sum(1) * ~state.was_moving
  # RewardManager scales rates by dt: each foot earns this bonus once,
  # independent of swing duration or the control rate.
  return value / env.step_dt + cfg.shaping_weight * state.shaping


def contact_slip_cost(contact, sole_velocity, angular_velocity):
  """Separate grounded translation and world-vertical rotation costs."""
  linear = (sole_velocity[..., :2].square().sum(-1) * contact).sum(-1)
  yaw = (angular_velocity[..., 2].square() * contact).sum(-1)
  return linear, yaw


def contact_slip_reward(env):
  """Penalize planted-foot sliding and yaw in every command phase."""
  state = env.command_manager.get_term("twist").state()
  r = state.robot.data
  angular = r.body_link_ang_vel_w[:, state.body_ids]
  offset = quat_apply(r.body_link_quat_w[:, state.body_ids],
                      state.offsets[None].expand(env.num_envs, -1, -1))
  velocity = r.body_link_lin_vel_w[:, state.body_ids] + torch.linalg.cross(angular, offset)
  contact = env.scene["feet_ground_contact"].data.found > 0
  linear, yaw = contact_slip_cost(contact, velocity, angular)
  cfg = state.command.cfg
  env.extras["log"]["Transition/contact_linear_slip_cost"] = linear.mean()
  env.extras["log"]["Transition/contact_yaw_slip_cost"] = yaw.mean()
  return -cfg.slip_weight * linear - cfg.yaw_slip_weight * yaw


def configure_stop_training(cfg, play: bool = False) -> None:
  """Add command profiles and stopping rewards to the flat training task."""
  from mjlab.managers.observation_manager import ObservationTermCfg
  command = cfg.commands["twist"]
  cfg.commands["twist"] = StopVelocityCommandCfg(**vars(command), sample_profiles=not play)
  cfg.observations["critic"].terms["stop_transition"] = ObservationTermCfg(func=stop_observation)
  cfg.observations["critic"].terms["transition_feet"] = ObservationTermCfg(func=transition_feet_observation)
  for name in ("pose", "stand_still", "leg_joint_vel_penalty", "base_drift_penalty",
               "track_linear_velocity", "track_angular_velocity", "foot_gait"):
    term = cfg.rewards[name]
    term.params = dict(term.params, original_func=term.func,
                       transition_kind="pose" if name == "pose" else "gate")
    term.func = StopReward
  cfg.rewards["stopping"] = RewardTermCfg(func=stopping_reward, weight=1.)
  cfg.rewards["foot_slip"] = RewardTermCfg(func=contact_slip_reward, weight=1.)
  if not play:
    # Keep the existing curricula and dynamics; hold auxiliary commands within
    # each episode so a stopping trial does not also change height or waist yaw.
    duration = cfg.episode_length_s + 1
    for name in ("base_height", "waist_yaw"):
      cfg.commands[name].resampling_time_range = (duration, duration)
