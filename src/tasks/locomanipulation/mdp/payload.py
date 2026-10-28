"""Reduced carrying loads, using the existing external-wrench event interface."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.utils.lab_api.math import matrix_from_quat

from .events import _resolve_constant_forces

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_BOX_HALF_HEIGHT_M = 0.06
_CONTENTS_RADIUS_M = 0.025


def default_payload_params() -> dict:
  """Return serializable settings for the tilting box and sliding contents."""
  return {
    "mass_range_kg": (2.0, 6.0),
    "contents_fraction": 0.75,
    "travel_xy_m": (0.10, 0.08),
    "com_height_m": -0.05,
    "static_friction_coefficient": 0.15,
    "kinetic_friction_coefficient": 0.10,
    "wall_restitution": 0.10,
    "damping_per_s": 1.0,
    "initial_offset_m": 0.01,
    "acceleration_filter_s": 0.10,
    "min_hand_separation_m": 0.10,
  }


def validate_payload_params(params: dict) -> dict:
  """Validate a complete model description, including restored YAML settings."""
  if set(params) != set(default_payload_params()):
    raise ValueError("Payload settings must contain exactly the required fields")
  result = dict(params)
  for name in ("mass_range_kg", "travel_xy_m"):
    values = tuple(params[name])
    if len(values) != 2 or not all(math.isfinite(v) and v > 0 for v in values):
      raise ValueError(f"Payload {name} must contain two finite positive values")
    result[name] = values
  if result["mass_range_kg"][0] > result["mass_range_kg"][1]:
    raise ValueError("Payload mass_range_kg must be ordered")
  for name in set(result) - {"mass_range_kg", "travel_xy_m"}:
    if not math.isfinite(result[name]):
      raise ValueError(f"Payload {name} must be finite")
  if not 0 <= result["contents_fraction"] <= 1:
    raise ValueError("Payload contents_fraction must lie in [0, 1]")
  for name in (
    "damping_per_s",
    "initial_offset_m",
    "acceleration_filter_s",
    "static_friction_coefficient",
    "kinetic_friction_coefficient",
  ):
    if result[name] < 0:
      raise ValueError(f"Payload {name} must be nonnegative")
  if result["initial_offset_m"] > min(result["travel_xy_m"]):
    raise ValueError("Initial contents offset exceeds payload travel")
  if result["min_hand_separation_m"] <= 0:
    raise ValueError("Minimum hand separation must be positive")
  if result["kinetic_friction_coefficient"] > result["static_friction_coefficient"]:
    raise ValueError("Kinetic friction cannot exceed static friction")
  if not 0 <= result["wall_restitution"] <= 1:
    raise ValueError("Wall restitution must lie in [0, 1]")
  return result


def distribute_hand_wrench(
  hands: torch.Tensor,
  force: torch.Tensor,
  moment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Reproduce a midpoint wrench using balanced hand forces and residual torque."""
  span = hands[:, 0] - hands[:, 1]
  delta = torch.cross(moment, span, dim=-1) / span.square().sum(-1, keepdim=True).clamp_min(1e-8)
  forces = torch.stack((0.5 * force + delta, 0.5 * force - delta), dim=1)
  residual = moment - torch.cross(span, delta, dim=-1)
  return forces, (0.5 * residual).unsqueeze(1).expand_as(forces)


def box_rotation(
  hands: torch.Tensor,
  hand_quats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Infer a right-handed box frame from the hand span and average wrist +X up.

  The X2 carry datasets define wrist +X as the top of a side grasp. Projecting
  the averaged up direction perpendicular to the span reconciles the two grips.
  """
  span = hands[:, 0] - hands[:, 1]
  separation = torch.linalg.vector_norm(span, dim=-1)
  axis_y = span / separation.clamp_min(1e-8).unsqueeze(-1)
  hand_up = matrix_from_quat(hand_quats)[..., :, 0].mean(dim=1)
  up = hand_up - (hand_up * axis_y).sum(-1, keepdim=True) * axis_y
  up_length = torch.linalg.vector_norm(up, dim=-1)
  axis_z = up / up_length.clamp_min(1e-8).unsqueeze(-1)
  axis_x = torch.cross(axis_y, axis_z, dim=-1)
  return torch.stack((axis_x, axis_y, axis_z), dim=-1), separation, up_length > 1e-4


def advance_frictional_contents(
  position: torch.Tensor,
  velocity: torch.Tensor,
  drive: torch.Tensor,
  normal_acceleration: torch.Tensor,
  travel: torch.Tensor,
  static_friction: float,
  kinetic_friction: float,
  damping: float,
  restitution: float,
  dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Integrate slip in the rotating box plane with load-dependent dry friction.

  Static friction holds a resting mass up to mu_s*N; moving friction uses mu_k.
  Velocity shrinkage cannot reverse motion. Wall impacts retain the configured
  fraction of outward speed, and their impulse enters the returned acceleration.
  """
  normal = normal_acceleration.clamp_min(0).unsqueeze(-1)
  sticking = (torch.linalg.vector_norm(velocity, dim=-1) <= 1e-6) & (
    torch.linalg.vector_norm(drive, dim=-1) <= static_friction * normal[:, 0]
  )
  free_velocity = velocity + dt * drive
  speed = torch.linalg.vector_norm(free_velocity, dim=-1, keepdim=True)
  slip = (
    free_velocity
    * ((speed - dt * kinetic_friction * normal).clamp_min(0) / speed.clamp_min(1e-12))
    / (1 + dt * damping)
  )
  slip = torch.where(sticking.unsqueeze(-1), 0.0, slip)
  proposed = position + dt * slip
  next_position = proposed.clamp(-travel, travel)
  impact = ((proposed >= travel) & (slip > 0)) | ((proposed <= -travel) & (slip < 0))
  next_velocity = torch.where(impact, -restitution * slip, slip)
  return next_position, next_velocity, (next_velocity - velocity) / dt


def rotational_acceleration(
  omega: torch.Tensor, alpha: torch.Tensor, offset: torch.Tensor
) -> torch.Tensor:
  """Euler and centripetal acceleration of a box-fixed offset in world frame."""
  return torch.cross(alpha, offset, dim=-1) + torch.cross(
    omega,
    torch.cross(omega, offset, dim=-1),
    dim=-1,
  )


def _carry_no_force_ratio(default_ratio: float, no_force_ratio: float) -> float:
  """Allocate the unloaded episode budget remaining after sampled default poses."""
  if not 0 <= default_ratio <= no_force_ratio <= 1:
    raise ValueError("Payload requires 0 <= default_pose_ratio <= no_force_ratio <= 1")
  if default_ratio == 1:
    return 0.0  # All sampled episodes already use the unloaded default pose.
  return (no_force_ratio - default_ratio) / (1 - default_ratio)


class SlidingBoxForceEvent:
  """A hand-carried box with one sliding mass, evaluated once per policy step.

  Only the existing applied-wrench and no-force-mask interfaces are exposed to
  observations/rewards. Hand tilt drives gravity and rotating-frame inertia;
  floor loading sets static and kinetic friction.
  Vertical flight, independent box motion and grip/contact solving are omitted.
  no_force_ratio includes sampled default-pose episodes, which are always unloaded.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._env = env
    asset_cfg = cfg.params["asset_cfg"]
    self._asset = env.scene[asset_cfg.name]
    self._body_ids = asset_cfg.body_ids
    self._body_names = asset_cfg.body_names
    if len(self._body_ids) != 2 or self._body_ids[0] == self._body_ids[1]:
      raise ValueError("Sliding payload requires two distinct ordered hand bodies")
    self._params = validate_payload_params(cfg.params["payload"])
    self._event_params = cfg.params
    self._num_envs = env.num_envs
    self._device = env.device
    self._dt = env.step_dt
    self._travel = torch.tensor(self._params["travel_xy_m"], device=env.device)
    self._gravity = torch.tensor(
      env.sim.mj_model.opt.gravity.copy(), device=env.device, dtype=torch.float32
    )
    self._position = torch.zeros(env.num_envs, 2, device=env.device)
    self._velocity = torch.zeros_like(self._position)
    self._previous_velocity = torch.zeros(env.num_envs, 3, device=env.device)
    self._acceleration = torch.zeros_like(self._previous_velocity)
    self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._rotation = torch.eye(3, device=env.device).repeat(env.num_envs, 1, 1)
    self._half_size = torch.tensor(
      [
        self._params["travel_xy_m"][0] + _CONTENTS_RADIUS_M,
        self._params["travel_xy_m"][1] + _CONTENTS_RADIUS_M,
        _BOX_HALF_HEIGHT_M,
      ],
      device=env.device,
    ).repeat(env.num_envs, 1)
    self._omega = torch.zeros_like(self._previous_velocity)
    self._angular_acceleration = torch.zeros_like(self._previous_velocity)
    tau = self._params["acceleration_filter_s"]
    self._filter_alpha = -math.expm1(-self._dt / tau) if tau else 1.0
    self._mass = torch.zeros(env.num_envs, device=env.device)
    self._episode_draw = torch.zeros_like(self._mass)
    self._no_force_mask = torch.zeros_like(self._initialized)
    self.reset()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset selected payloads without reading potentially stale robot kinematics."""
    if env_ids is None:
      env_ids = slice(None)
    default_ratio, default_pose = self._get_default_pose_sampling(
      self._event_params.get("upper_body_action_name"),
    )
    carry_ratio = 0.0
    if self._event_params.get("constant_force") is None:
      carry_ratio = _carry_no_force_ratio(
        default_ratio, self._event_params.get("no_force_ratio", 0.0),
      )
    count = self._mass[env_ids].numel()
    lo, hi = self._params["mass_range_kg"]
    self._mass[env_ids] = lo + (hi - lo) * torch.rand(count, device=self._device)
    self._episode_draw[env_ids] = torch.rand(count, device=self._device)
    self._no_force_mask[env_ids] = default_pose[env_ids] | (self._episode_draw[env_ids] < carry_ratio)
    self._position[env_ids] = (2 * torch.rand(count, 2, device=self._device) - 1) * self._params[
      "initial_offset_m"
    ]
    self._velocity[env_ids] = 0
    self._previous_velocity[env_ids] = 0
    self._acceleration[env_ids] = 0
    self._initialized[env_ids] = False
    self._rotation[env_ids] = torch.eye(3, device=self._device)
    self._omega[env_ids] = 0
    self._angular_acceleration[env_ids] = 0
    zeros = torch.zeros(count, 2, 3, device=self._device)
    self._asset.write_external_wrench_to_sim(zeros, zeros, env_ids=env_ids, body_ids=self._body_ids)

  def _get_default_pose_sampling(self, action_name: str | None) -> tuple[float, torch.Tensor]:
    # The event manager is constructed before the action manager. Its initial
    # reset only clears state; episode resets can read the newly selected pose.
    action_manager = getattr(self._env, "action_manager", None)
    if action_name is None or action_manager is None:
      return 0.0, torch.zeros_like(self._no_force_mask)
    action = action_manager.get_term(action_name)
    return action.default_pose_sampling_ratio, action.default_pose_mask

  def __call__(
    self,
    env,
    env_ids,
    asset_cfg,
    payload: dict,
    force_scale: float = 0.0,
    no_force_ratio: float = 0.0,
    constant_force: dict | None = None,
    body_frame: bool = False,
    upper_body_action_name: str | None = None,
  ) -> None:
    del env, env_ids, asset_cfg, payload
    if not math.isfinite(force_scale) or not 0 <= no_force_ratio <= 1:
      raise ValueError("Payload requires finite force_scale and no_force_ratio in [0, 1]")
    default_ratio, default_pose = self._get_default_pose_sampling(upper_body_action_name)
    if constant_force is not None:
      forces = _resolve_constant_forces(
        constant_force,
        self._num_envs,
        2,
        self._body_names,
        self._device,
        body_frame=body_frame,
        asset=self._asset,
      )
      forces[default_pose] = 0.0
      self._asset.write_external_wrench_to_sim(
        forces, torch.zeros_like(forces), body_ids=self._body_ids
      )
      self._no_force_mask[:] = ~forces.ne(0).any(dim=(1, 2))
      self._initialized[:] = False
      self._velocity.zero_()
      return

    hands = self._asset.data.body_com_pos_w[:, self._body_ids]
    hand_quats = self._asset.data.body_link_quat_w[:, self._body_ids]
    rotation, separation, valid = box_rotation(hands, hand_quats)
    valid &= separation >= self._params["min_hand_separation_m"]
    # The reduced floor-contact model requires an upright, nondegenerate grasp.
    valid &= rotation[:, 2, 2] > 0.1
    carry_ratio = _carry_no_force_ratio(default_ratio, no_force_ratio)
    self._no_force_mask[:] = default_pose | (self._episode_draw < carry_ratio) | ~valid
    active = ~self._no_force_mask & (force_scale > 0)
    continuing = (self._initialized & active).unsqueeze(-1)
    hand_velocity = self._asset.data.body_com_lin_vel_w[:, self._body_ids].mean(dim=1)
    raw_acceleration = (hand_velocity - self._previous_velocity) / self._dt
    acceleration = self._acceleration + self._filter_alpha * (raw_acceleration - self._acceleration)
    acceleration = torch.where(continuing, acceleration, 0.0)
    # sum_i (R_prev e_i) x (R e_i) / (2 dt) = sin(theta) axis / dt.
    # This avoids quaternion sign discontinuities for small physics-step turns.
    raw_omega = torch.cross(self._rotation, rotation, dim=1).sum(dim=2) / (2 * self._dt)
    omega = self._omega + self._filter_alpha * (raw_omega - self._omega)
    omega = torch.where(continuing, omega, 0.0)
    raw_alpha = (omega - self._omega) / self._dt
    alpha = self._angular_acceleration + self._filter_alpha * (
      raw_alpha - self._angular_acceleration
    )
    alpha = torch.where(continuing, alpha, 0.0)

    shell_local = torch.zeros_like(acceleration)
    shell_local[:, 2] = self._params["com_height_m"]
    contents_local = shell_local.clone()
    contents_local[:, :2] = self._position
    # Contents rest on the bottom of the proxy box, not at the shell COM.
    contents_local[:, 2] -= _BOX_HALF_HEIGHT_M - _CONTENTS_RADIUS_M
    shell_offset = torch.einsum("bij,bj->bi", rotation, shell_local)
    contents_offset = torch.einsum("bij,bj->bi", rotation, contents_local)
    relative_velocity = torch.einsum("bij,bj->bi", rotation[:, :, :2], self._velocity)
    shell_acceleration = acceleration + rotational_acceleration(omega, alpha, shell_offset)
    contents_acceleration = acceleration + rotational_acceleration(omega, alpha, contents_offset)
    contents_acceleration += 2 * torch.cross(omega, relative_velocity, dim=-1)
    effective_gravity = self._gravity - contents_acceleration
    drive = torch.einsum("bij,bi->bj", rotation, effective_gravity)
    position, velocity, relative_acceleration = advance_frictional_contents(
      self._position,
      self._velocity,
      drive[:, :2],
      -drive[:, 2],
      self._travel,
      self._params["static_friction_coefficient"],
      self._params["kinetic_friction_coefficient"],
      self._params["damping_per_s"],
      self._params["wall_restitution"],
      self._dt,
    )
    relative_acceleration_w = torch.einsum("bij,bj->bi", rotation[:, :, :2], relative_acceleration)
    mass = (self._mass * min(max(force_scale, 0.0), 1.0) * active).unsqueeze(-1)
    contents_mass = mass * self._params["contents_fraction"]
    shell_mass = mass - contents_mass
    shell_force = shell_mass * (self._gravity - shell_acceleration)
    contents_force = contents_mass * (
      self._gravity - contents_acceleration - relative_acceleration_w
    )
    # Uniform shell cuboid inertia about its COM. Contents are an aggregate point
    # mass, whose offset inertia is already in its translational reaction moment.
    self._half_size[:, 1] = (0.5 * separation).clamp_min(
      self._params["travel_xy_m"][1] + _CONTENTS_RADIUS_M,
    )
    size = 2 * self._half_size
    inertia = shell_mass / 12 * (size.square().sum(-1, keepdim=True) - size.square())
    omega_b = torch.einsum("bij,bi->bj", rotation, omega)
    alpha_b = torch.einsum("bij,bi->bj", rotation, alpha)
    shell_spin = inertia * alpha_b + torch.cross(omega_b, inertia * omega_b, dim=-1)
    moment = torch.cross(shell_offset, shell_force, dim=-1) + torch.cross(
      contents_offset, contents_force, dim=-1
    )
    moment -= torch.einsum("bij,bj->bi", rotation, shell_spin)
    forces, torques = distribute_hand_wrench(hands, shell_force + contents_force, moment)
    self._asset.write_external_wrench_to_sim(forces, torques, body_ids=self._body_ids)

    self._position[:] = torch.where(active.unsqueeze(-1), position, self._position)
    self._velocity[:] = torch.where(active.unsqueeze(-1), velocity, 0.0)
    self._previous_velocity[:] = hand_velocity
    self._acceleration[:] = acceleration
    self._rotation[:] = rotation
    self._omega[:] = omega
    self._angular_acceleration[:] = alpha
    self._initialized[:] = active

  def debug_vis(self, visualizer) -> None:
    """Draw the reduced box, sliding contents and combined COM as viewer overlays."""
    for index in visualizer.get_env_indices(self._num_envs):
      if self._initialized[index]:
        hands = self._asset.data.body_com_pos_w[index, self._body_ids].cpu().numpy()
        rotation = self._rotation[index].cpu().numpy()
        center = hands.mean(axis=0)
        center += self._params["com_height_m"] * rotation[:, 2]
        offset = rotation[:, :2] @ self._position[index].cpu().numpy()
        offset -= (_BOX_HALF_HEIGHT_M - _CONTENTS_RADIUS_M) * rotation[:, 2]
        visualizer.add_box(
          center=center,
          size=self._half_size[index].cpu().numpy(),
          mat=rotation,
          color=(0.65, 0.40, 0.20, 0.18),
          label="Reduced box",
        )
        visualizer.add_sphere(
          center=center + offset,
          radius=_CONTENTS_RADIUS_M,
          color=(1.0, 0.45, 0.05, 1.0),
          label="Sliding contents",
        )
        com = center + self._params["contents_fraction"] * offset
        visualizer.add_sphere(
          center=com,
          radius=0.008,
          color=(0.05, 0.85, 1.0, 1.0),
          label="Combined payload COM",
        )
        # Extend a COM indicator above the envelope so the box cannot hide it.
        visualizer.add_cylinder(
          start=com,
          end=com + np.array([0.0, 0.0, 0.11]),
          radius=0.002,
          color=(0.05, 0.85, 1.0, 1.0),
        )
      for body in self._body_ids:
        force = self._asset.data.body_external_wrench[index, body, :3]
        if force.square().sum().item() < 1.0:
          continue
        start = self._asset.data.body_com_pos_w[index, body].cpu().numpy()
        visualizer.add_arrow(
          start=start,
          end=start + 0.02 * force.cpu().numpy(),
          color=(0.9, 0.2, 0.8, 0.9),
          width=0.005,
        )


def restore_payload_params(cfg, overrides: dict, *, training: bool = False) -> None:
  """Restore payload settings without changing task activation or play overrides."""
  saved = overrides.get("events", {}).get("hand_force", {}).get("params", {})
  if "payload" not in saved:
    return
  event = cfg.events.get("hand_force")
  if event is None or event.func is not SlidingBoxForceEvent:
    raise ValueError("Saved payload requires a carry task with SlidingBoxForceEvent")
  payload = dict(saved["payload"])
  payload.pop("model_version", None)
  payload = validate_payload_params(payload)
  ratio = event.params.get("no_force_ratio", 0.0)
  if training and "no_force_ratio" in saved:
    ratio = saved["no_force_ratio"]
    if not 0 <= ratio <= 1:
      raise ValueError("Saved no_force_ratio must lie in [0, 1]")
  if training and event.params.get("constant_force") is None:
    action_name = event.params.get("upper_body_action_name")
    action = cfg.actions[action_name] if action_name is not None else None
    default_ratio = (
      action.default_pose_ratio
      if action is not None and action.fixed_upper_body_pose is None else 0.0
    )
    _carry_no_force_ratio(default_ratio, ratio)
  event.params["payload"] = payload
  if training:
    event.params["no_force_ratio"] = ratio


def enable_payload_playback(cfg) -> None:
  """Enable the carry load and dataset poses instead of the arms-down play default."""
  event = cfg.events.get("hand_force")
  if event is None or event.func is not SlidingBoxForceEvent:
    raise ValueError("Sliding payload playback requires the carry task")
  event.params.update(constant_force=None, force_scale=1.0, no_force_ratio=0.0)
  action = cfg.actions["upper_body_motion"]
  action.fixed_upper_body_pose = None
  action.default_pose_ratio = 0.0
