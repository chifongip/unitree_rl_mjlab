"""Synchronous Torch port of RoboJuDo's bounded joint disturbance observers.

Only targets, joint feedback, elapsed time and accepted command history enter
the controller. Nominal model coefficients are independent of simulated plant
parameters. All state has shape (num_envs, num_joints).
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(kw_only=True)
class JointObserverCfg:
  joint_names: tuple[str, ...]
  inertia: float | tuple[float, ...]
  correction_limits_nm: float | tuple[float, ...]
  algorithm: Literal["linearized", "nonlinear"] = "nonlinear"
  mode: Literal["OFF", "SHADOW", "ACTIVE"] = "ACTIVE"
  friction_limit: float | tuple[float, ...] = 0.0
  friction_viscosity: float | tuple[float, ...] = 0.0
  bandwidth_rad_s: float = 8.0
  slew_nm_s: float = 2.0
  release_slew_nm_s: float | None = None
  position_noise_std: float = 0.0
  velocity_noise_std: float = 0.0


class BatchedJointObserver:
  """One fixed model and independent observer histories for parallel worlds.

  ``step`` includes command headroom limiting and acknowledgment. It assumes
  one fresh synchronous sample per control interval and no command transport
  delay. OFF/SHADOW issue zero effort; SHADOW predicts that issued zero.
  """

  _FLOAT_STATE_FIELDS = (
    "effort", "estimate", "previous_position", "previous_velocity",
    "previous_target", "prediction", "innovation", "proposed", "headroom_limited",
  )

  def __init__(
    self,
    cfg: JointObserverCfg,
    stiffness: tuple[float, ...],
    damping: tuple[float, ...],
    actuator_limits: tuple[float, ...],
    *,
    num_envs: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
  ) -> None:
    self.cfg = deepcopy(cfg)
    cfg = self.cfg
    if not cfg.joint_names or len(set(cfg.joint_names)) != len(cfg.joint_names):
      raise ValueError("Observer joint names must be nonempty and unique")
    if num_envs <= 0 or not dtype.is_floating_point:
      raise ValueError("Observer requires positive num_envs and floating dtype")
    if cfg.algorithm not in ("linearized", "nonlinear"):
      raise ValueError("Unknown observer algorithm")
    if cfg.mode not in ("OFF", "SHADOW", "ACTIVE"):
      raise ValueError("Unknown observer mode")
    release = cfg.slew_nm_s if cfg.release_slew_nm_s is None else cfg.release_slew_nm_s
    for value in (cfg.bandwidth_rad_s, cfg.slew_nm_s, release):
      if not math.isfinite(value) or value <= 0:
        raise ValueError("Observer bandwidth and slew must be positive and finite")
    if cfg.algorithm == "linearized" and release != cfg.slew_nm_s:
      raise ValueError("Linearized observer requires symmetric slew")
    for value in (cfg.position_noise_std, cfg.velocity_noise_std):
      if not math.isfinite(value) or value < 0:
        raise ValueError("Observer measurement noise must be nonnegative and finite")
    self.release_slew = release
    self.shape = (num_envs, len(cfg.joint_names))
    # Compare feedback against the allocated device, resolving aliases such as
    # "cuda" to "cuda:0" and "cpu:0" to "cpu".
    self.device = torch.empty(0, device=device).device
    self.dtype = dtype

    def vector(
      value: float | tuple[float, ...], *, positive: bool = False,
    ) -> torch.Tensor:
      result = torch.as_tensor(value, device=self.device, dtype=dtype)
      if result.ndim > 1 or (result.ndim == 1 and result.shape != (self.shape[1],)):
        raise ValueError("Observer parameters must be scalar or match joint_names")
      result = result.broadcast_to((1, self.shape[1])).clone()
      if not torch.isfinite(result).all() or (result <= 0 if positive else result < 0).any():
        raise ValueError("Invalid observer model parameters")
      return result

    self.kp = vector(stiffness)
    self.kd = vector(damping)
    self.torque_limits = vector(actuator_limits, positive=True)
    self.inertia = vector(cfg.inertia, positive=True)
    self.friction_limit = vector(cfg.friction_limit)
    self.friction_viscosity = vector(cfg.friction_viscosity)
    self.limits = vector(cfg.correction_limits_nm, positive=True)
    if (self.limits > self.torque_limits).any():
      raise ValueError("Correction limits cannot exceed actuator limits")
    for name in self._FLOAT_STATE_FIELDS:
      setattr(self, name, torch.zeros(self.shape, device=self.device, dtype=dtype))
    self.initialized = torch.zeros((num_envs, 1), device=self.device, dtype=torch.bool)
    self.antiwindup = torch.zeros(self.shape, device=self.device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Clear only selected environments, including their issued correction."""
    env_ids = slice(None) if env_ids is None else env_ids
    for name in (*self._FLOAT_STATE_FIELDS, "initialized", "antiwindup"):
      getattr(self, name)[env_ids] = 0

  def predict_velocity(
    self,
    position: torch.Tensor,
    velocity: torch.Tensor,
    target: torch.Tensor,
    effort: torch.Tensor,
    disturbance: torch.Tensor,
    dt: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Four implicit local-model substeps and torque-to-velocity sensitivity."""
    q, v = position.clone(), velocity.clone()
    dq, dv = torch.zeros_like(q), torch.zeros_like(v)
    h = dt / 4
    mass_h = self.inertia / h
    for _ in range(4):
      raw = self.kp * (target - q) - self.kd * v + effort
      saturated = raw.abs() > self.torque_limits
      alpha = torch.where(saturated, mass_h, mass_h + self.kd + self.kp * h)
      force = torch.where(
        saturated,
        mass_h * v + raw.clamp(-self.torque_limits, self.torque_limits),
        mass_h * v + self.kp * (target - q) + effort,
      ) + disturbance
      derivative = mass_h * dv + torch.where(saturated, 0.0, -self.kp * dq) + 1
      trial = force / (alpha + self.friction_viscosity)
      sliding = (self.friction_viscosity * trial).abs() > self.friction_limit
      denominator = torch.where(sliding, alpha, alpha + self.friction_viscosity)
      v = torch.where(
        sliding, (force - force.sign() * self.friction_limit) / alpha, trial
      )
      dv = derivative / denominator
      q = q + h * v
      dq = dq + h * dv
    return v, dv

  def _predict(
    self, disturbance: torch.Tensor, dt: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    return self.predict_velocity(
      self.previous_position, self.previous_velocity, self.previous_target,
      self.effort, disturbance, dt,
    )

  def _infer_disturbance(self, velocity: torch.Tensor, dt: float) -> torch.Tensor:
    """Match the deployment nonlinear observer's twenty bounded bisections."""
    lower = -self.limits.expand(self.shape).clone()
    upper = self.limits.expand(self.shape).clone()
    best = self.estimate.clone()
    predicted, _ = self._predict(best, dt)
    residual = (predicted - velocity).abs()
    for candidate in (lower, upper):
      predicted, _ = self._predict(candidate, dt)
      error = (predicted - velocity).abs()
      best = torch.where(error < residual, candidate, best)
      residual = torch.minimum(residual, error)
    for _ in range(20):
      midpoint = (lower + upper) / 2
      predicted, _ = self._predict(midpoint, dt)
      error = (predicted - velocity).abs()
      best = torch.where(error < residual, midpoint, best)
      residual = torch.minimum(residual, error)
      below = predicted < velocity
      lower = torch.where(below, midpoint, lower)
      upper = torch.where(below, upper, midpoint)
    return best

  @staticmethod
  def _allow_unwinding(
    previous: torch.Tensor, estimate: torch.Tensor, blocked_direction: torch.Tensor,
  ) -> torch.Tensor:
    old, new = -previous, -estimate
    outward = new * blocked_direction > (old * blocked_direction).clamp_min(0)
    retained = torch.where(old * blocked_direction >= 0, old, 0.0)
    return -torch.where(outward, retained, new)

  @torch.no_grad()
  def step(
    self,
    target: torch.Tensor,
    position: torch.Tensor,
    velocity: torch.Tensor,
    dt: float,
  ) -> torch.Tensor:
    """Estimate, limit and acknowledge correction for the upcoming interval.

    Invalid feedback resets only its world and issues zero there. No simulator
    forces, masses, accelerations or dynamics matrices are accepted as inputs.
    """
    if not math.isfinite(dt) or dt <= 0:
      self.reset()
      raise ValueError("Observer dt must be positive and finite")
    values = (target, position, velocity)
    if any(x.shape != self.shape or x.device != self.device or x.dtype != self.dtype for x in values):
      raise ValueError("Observer feedback must match state shape, device and dtype")
    if self.cfg.mode == "OFF":
      self.reset()
      return self.effort.clone()

    valid = torch.stack([torch.isfinite(x).all(dim=1, keepdim=True) for x in values]).all(dim=0)
    target, position, velocity = [torch.where(valid, x, 0.0) for x in values]
    fresh = self.initialized & valid
    previous = self.estimate.clone()
    predicted, gain = self._predict(previous, dt)
    self.prediction = torch.where(fresh, predicted, velocity)
    self.innovation = torch.where(fresh, velocity - predicted, 0.0)
    alpha = -math.expm1(-self.cfg.bandwidth_rad_s * dt)
    if self.cfg.algorithm == "linearized":
      estimate = previous + alpha * self.innovation / gain
    else:
      instantaneous = self._infer_disturbance(velocity, dt)
      estimate = previous + alpha * (instantaneous - previous)
    estimate = torch.where(fresh, estimate.clamp(-self.limits, self.limits), previous)
    desired = -estimate
    if self.cfg.algorithm == "nonlinear":
      releasing = (self.effort != 0) & (
        (desired * self.effort <= 0) | (desired.abs() < self.effort.abs())
      )
      destination = torch.where(desired * self.effort < 0, 0.0, desired)
      rate = torch.where(releasing, torch.full_like(desired, self.release_slew), self.cfg.slew_nm_s)
    else:
      destination, rate = desired, self.cfg.slew_nm_s
    increment = (destination - self.effort).clamp(-rate * dt, rate * dt)
    increment = torch.where(fresh, increment, 0.0)
    candidate = (self.effort + increment).clamp(-self.limits, self.limits)
    pd = self.kp * (target - position) - self.kd * velocity
    total = pd + candidate
    blocked = ((total > self.torque_limits) & (increment > 0)) | (
      (total < -self.torque_limits) & (increment < 0)
    )
    proposed = torch.where(blocked, self.effort, candidate)
    if self.cfg.algorithm == "linearized":
      estimate = torch.where(blocked, previous, estimate)
    else:
      estimate = self._allow_unwinding(previous, estimate, torch.where(blocked, total.sign(), 0.0))

    # Match JointObserver's command headroom, including saturation-reducing effort.
    lower = (-self.torque_limits - pd).clamp_max(0)
    upper = (self.torque_limits - pd).clamp_min(0)
    limited = proposed.clamp(lower, upper)
    issued = limited if self.cfg.mode == "ACTIVE" else torch.zeros_like(limited)
    downstream_limited = issued != proposed
    if self.cfg.mode != "SHADOW":
      if self.cfg.algorithm == "linearized":
        estimate = torch.where(downstream_limited, previous, estimate)
      else:
        estimate = self._allow_unwinding(previous, estimate, (proposed - issued).sign())

    self.proposed = torch.where(valid, proposed, 0.0)
    self.headroom_limited = torch.where(valid, limited, 0.0)
    self.effort = torch.where(valid, issued, 0.0)
    self.estimate = torch.where(valid, estimate, 0.0)
    self.antiwindup = valid & (blocked | downstream_limited)
    self.previous_position = position.clone()
    self.previous_velocity = velocity.clone()
    self.previous_target = target.clone()
    self.initialized = valid.clone()
    return self.effort.clone()
